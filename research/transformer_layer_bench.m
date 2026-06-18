// transformer_layer_bench.m — Full Transformer Layer GPU+ANE Split Benchmark
// SwiGLU FFN split, attention, multi-layer pipeline, decode mode
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface \
//     -framework Metal -framework MetalPerformanceShaders -ldl \
//     -o transformer_layer_bench transformer_layer_bench.m
//
// Run:
//   ./transformer_layer_bench

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <dispatch/dispatch.h>

// ============================================================================
// 1. INFRASTRUCTURE
// ============================================================================

static mach_timebase_info_data_t g_tb;
static double ticks_to_us(uint64_t t) {
    return (double)t * g_tb.numer / g_tb.denom / 1e3;
}

static IOSurfaceRef create_surface(size_t bytes) {
    size_t ps = getpagesize();
    size_t as = ((bytes + ps - 1) / ps) * ps;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(as), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(as),
        (id)kIOSurfaceAllocSize: @(as), (id)kIOSurfacePixelFormat: @0
    });
}

static void fill_random(void *ptr, size_t n_floats) {
    float *p = (float *)ptr;
    for (size_t i = 0; i < n_floats; i++)
        p[i] = ((float)(arc4random() % 2000) / 100000.0f - 0.01f);
}

// ============================================================================
// 2. ANE INFRASTRUCTURE
// ============================================================================

typedef struct { id model; id request; NSString *tmpDir; } ANEKernel;
static Class g_ANEDesc, g_ANEInMem, g_ANEReq, g_ANEIO;

static bool init_ane_classes(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_ANEDesc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
    g_ANEReq   = NSClassFromString(@"_ANERequest");
    g_ANEIO    = NSClassFromString(@"_ANEIOSurfaceObject");
    return g_ANEDesc && g_ANEInMem && g_ANEReq && g_ANEIO;
}

static uint8_t *build_weight_blob(int out_ch, int in_ch, size_t *out_len) {
    int ws = out_ch * in_ch * 2;
    int total = 128 + ws;
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    *(uint32_t*)(buf + 72) = ws;
    *(uint32_t*)(buf + 80) = 128;
    _Float16 *fp16 = (_Float16 *)(buf + 128);
    for (int i = 0; i < out_ch * in_ch; i++)
        fp16[i] = (_Float16)((float)(arc4random() % 1000) / 50000.0f - 0.01f);
    *out_len = total;
    return buf;
}

static NSString *gen_mil_conv(int in_ch, int out_ch, int spatial) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]"
        "\n{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
        "    } -> (y);\n}\n",
        in_ch, spatial, in_ch, spatial,
        out_ch, in_ch, out_ch, in_ch,
        out_ch, spatial, out_ch, spatial];
}

static ANEKernel *compile_ane(int in_ch, int out_ch, int spatial,
                               IOSurfaceRef ioIn, IOSurfaceRef ioOut) {
    @autoreleasepool {
        NSError *e = nil;
        NSData *mil = [[gen_mil_conv(in_ch, out_ch, spatial) dataUsingEncoding:NSUTF8StringEncoding] copy];
        size_t wbLen;
        uint8_t *wbBuf = build_weight_blob(out_ch, in_ch, &wbLen);
        NSData *wb = [NSData dataWithBytesNoCopy:wbBuf length:wbLen freeWhenDone:YES];

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            mil, @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wb}}, nil);
        if (!desc) return NULL;

        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) return NULL;

        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wb writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            [fm removeItemAtPath:td error:nil]; return NULL;
        }
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            [fm removeItemAtPath:td error:nil]; return NULL;
        }

        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioIn);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioOut);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

        ANEKernel *k = calloc(1, sizeof(ANEKernel));
        k->model = mdl; k->request = req; k->tmpDir = td;
        return k;
    }
}

static bool ane_eval(ANEKernel *k) {
    @autoreleasepool {
        NSError *e = nil;
        return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            k->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, k->request, &e);
    }
}

static void ane_free(ANEKernel *k) {
    @autoreleasepool {
        if (!k) return;
        NSError *e = nil;
        if (k->model)
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(k->model, @selector(unloadWithQoS:error:), 21, &e);
        if (k->tmpDir) [[NSFileManager defaultManager] removeItemAtPath:k->tmpDir error:nil];
        free(k);
    }
}

// ============================================================================
// 3. METAL SHADERS (embedded MSL, compiled at runtime)
// ============================================================================

static NSString *g_shaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

// SiLU×Mul: gate and up are in a single buffer [2*hidden, seq]
// First half = gate (W1 output), second half = up (W3 output)
// Output = silu(gate) * up, size [hidden, seq]
kernel void silu_mul(const device float *gate_up [[buffer(0)]],
                     device float *out           [[buffer(1)]],
                     constant int &half_size     [[buffer(2)]],
                     uint idx [[thread_position_in_grid]]) {
    float g = gate_up[idx];
    float u = gate_up[idx + half_size];
    out[idx] = (g / (1.0f + exp(-g))) * u;
}

// RMSNorm: channel-first [dim, seq] layout
// Each thread handles one position (column)
kernel void rmsnorm(const device float *x   [[buffer(0)]],
                    const device float *w   [[buffer(1)]],
                    device float *out       [[buffer(2)]],
                    constant int &dim       [[buffer(3)]],
                    constant int &seq       [[buffer(4)]],
                    uint pos [[thread_position_in_grid]]) {
    float ss = 0.0f;
    for (int c = 0; c < dim; c++) {
        float v = x[c * seq + pos];
        ss += v * v;
    }
    float rms = rsqrt(ss / float(dim) + 1e-5f);
    for (int c = 0; c < dim; c++) {
        out[c * seq + pos] = x[c * seq + pos] * w[c] * rms;
    }
}

// Attention scores: S[h][q][k] = sum_d Q[h*hd+d][q] * K[h*hd+d][k] * scale
// Channel-first layout: Q[c][s] at c*seq+s
kernel void attn_scores(const device float *Q   [[buffer(0)]],
                        const device float *K   [[buffer(1)]],
                        device float *S         [[buffer(2)]],
                        constant int &seq       [[buffer(3)]],
                        constant int &head_dim  [[buffer(4)]],
                        constant float &scale   [[buffer(5)]],
                        uint3 gid [[thread_position_in_grid]]) {
    int k_pos = gid.x;  // key position
    int q_pos = gid.y;  // query position
    int h = gid.z;      // head
    if (k_pos >= seq || q_pos >= seq) return;

    float sum = 0.0f;
    int base = h * head_dim;
    for (int d = 0; d < head_dim; d++) {
        sum += Q[(base + d) * seq + q_pos] * K[(base + d) * seq + k_pos];
    }
    S[h * seq * seq + q_pos * seq + k_pos] = sum * scale;
}

// Causal softmax: per-row softmax with causal mask
// Each thread handles one row: scores[h][q][0..seq-1]
kernel void causal_softmax(device float *S      [[buffer(0)]],
                            constant int &seq    [[buffer(1)]],
                            uint gid [[thread_position_in_grid]]) {
    int row = gid;
    int q_pos = row % seq;
    device float *row_ptr = S + row * seq;

    // Causal mask + find max
    float mx = -INFINITY;
    for (int k = 0; k <= q_pos; k++) mx = max(mx, row_ptr[k]);

    // Exp + sum
    float sm = 0.0f;
    for (int k = 0; k <= q_pos; k++) {
        float e = exp(row_ptr[k] - mx);
        row_ptr[k] = e;
        sm += e;
    }

    // Normalize
    float inv = 1.0f / sm;
    for (int k = 0; k <= q_pos; k++) row_ptr[k] *= inv;
    for (int k = q_pos + 1; k < seq; k++) row_ptr[k] = 0.0f;
}

// Attention context: C[h*hd+d][q] = sum_k S[h][q][k] * V[h*hd+d][k]
kernel void attn_context(const device float *S   [[buffer(0)]],
                          const device float *V   [[buffer(1)]],
                          device float *C         [[buffer(2)]],
                          constant int &seq       [[buffer(3)]],
                          constant int &head_dim  [[buffer(4)]],
                          uint3 gid [[thread_position_in_grid]]) {
    int d = gid.x;      // head_dim index
    int q = gid.y;       // query position
    int h = gid.z;       // head
    if (d >= head_dim || q >= seq) return;

    float sum = 0.0f;
    int s_base = h * seq * seq + q * seq;
    int v_idx_base = (h * head_dim + d) * seq;
    for (int k = 0; k < seq; k++) {
        sum += S[s_base + k] * V[v_idx_base + k];
    }
    C[(h * head_dim + d) * seq + q] = sum;
}

// Element-wise add (for residuals and partial reduction)
kernel void elem_add(const device float *a [[buffer(0)]],
                     const device float *b [[buffer(1)]],
                     device float *out     [[buffer(2)]],
                     uint idx [[thread_position_in_grid]]) {
    out[idx] = a[idx] + b[idx];
}

// Decode-mode attention: seq_len=1 query against KV cache
// scores[h] = sum_d Q[h*hd+d][0] * K_cache[h*hd+d][k] * scale for k in [0, cache_len)
kernel void decode_attn_scores(const device float *Q       [[buffer(0)]],
                                const device float *K_cache [[buffer(1)]],
                                device float *scores        [[buffer(2)]],
                                constant int &cache_len     [[buffer(3)]],
                                constant int &max_seq       [[buffer(4)]],
                                constant int &head_dim      [[buffer(5)]],
                                constant float &scale       [[buffer(6)]],
                                uint2 gid [[thread_position_in_grid]]) {
    int k_pos = gid.x;  // cache position
    int h = gid.y;       // head
    if (k_pos >= cache_len) return;

    float sum = 0.0f;
    int base = h * head_dim;
    for (int d = 0; d < head_dim; d++) {
        sum += Q[base + d] * K_cache[(base + d) * max_seq + k_pos];
    }
    scores[h * cache_len + k_pos] = sum * scale;
}

// Decode softmax: full row (no causal mask needed, all positions visible)
kernel void decode_softmax(device float *scores  [[buffer(0)]],
                            constant int &len     [[buffer(1)]],
                            uint h [[thread_position_in_grid]]) {
    device float *row = scores + h * len;
    float mx = -INFINITY;
    for (int i = 0; i < len; i++) mx = max(mx, row[i]);
    float sm = 0.0f;
    for (int i = 0; i < len; i++) { float e = exp(row[i] - mx); row[i] = e; sm += e; }
    float inv = 1.0f / sm;
    for (int i = 0; i < len; i++) row[i] *= inv;
}

// Decode context: C[h*hd+d] = sum_k scores[h][k] * V_cache[h*hd+d][k]
kernel void decode_attn_context(const device float *scores  [[buffer(0)]],
                                 const device float *V_cache [[buffer(1)]],
                                 device float *C             [[buffer(2)]],
                                 constant int &cache_len     [[buffer(3)]],
                                 constant int &max_seq       [[buffer(4)]],
                                 constant int &head_dim      [[buffer(5)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    int d = gid.x;  // head_dim index
    int h = gid.y;  // head
    if (d >= head_dim) return;

    float sum = 0.0f;
    int v_base = (h * head_dim + d) * max_seq;
    int s_base = h * cache_len;
    for (int k = 0; k < cache_len; k++) {
        sum += scores[s_base + k] * V_cache[v_base + k];
    }
    C[h * head_dim + d] = sum;
}
)";

// ============================================================================
// 4. SHADER PIPELINE SETUP
// ============================================================================

typedef struct {
    id<MTLComputePipelineState> silu_mul;
    id<MTLComputePipelineState> rmsnorm;
    id<MTLComputePipelineState> attn_scores;
    id<MTLComputePipelineState> causal_softmax;
    id<MTLComputePipelineState> attn_context;
    id<MTLComputePipelineState> elem_add;
    id<MTLComputePipelineState> decode_attn_scores;
    id<MTLComputePipelineState> decode_softmax;
    id<MTLComputePipelineState> decode_attn_context;
} ShaderPipelines;

static ShaderPipelines setup_shaders(id<MTLDevice> dev) {
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:g_shaderSource options:nil error:&err];
    if (!lib) {
        fprintf(stderr, "Metal shader compile error: %s\n", [[err description] UTF8String]);
        exit(1);
    }

    ShaderPipelines p;
    NSArray *names = @[@"silu_mul", @"rmsnorm", @"attn_scores", @"causal_softmax",
                       @"attn_context", @"elem_add", @"decode_attn_scores",
                       @"decode_softmax", @"decode_attn_context"];
    id<MTLComputePipelineState> __strong *ptrs[] = {
        &p.silu_mul, &p.rmsnorm, &p.attn_scores, &p.causal_softmax,
        &p.attn_context, &p.elem_add, &p.decode_attn_scores,
        &p.decode_softmax, &p.decode_attn_context
    };

    for (int i = 0; i < (int)names.count; i++) {
        id<MTLFunction> fn = [lib newFunctionWithName:names[i]];
        if (!fn) { fprintf(stderr, "Missing shader: %s\n", [names[i] UTF8String]); exit(1); }
        *ptrs[i] = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!*ptrs[i]) { fprintf(stderr, "Pipeline error: %s\n", [[err description] UTF8String]); exit(1); }
    }
    return p;
}

// ============================================================================
// 5. MODEL CONFIG + LAYER STATE
// ============================================================================

typedef struct {
    int dim, hidden, n_heads, head_dim, n_layers, seq;
    float ane_ratio;
    int ane_hidden, gpu_hidden;
} TConfig;

static TConfig make_config(int dim, int hidden, int n_heads, int n_layers, int seq, float ane_ratio) {
    TConfig c;
    c.dim = dim; c.hidden = hidden; c.n_heads = n_heads;
    c.head_dim = dim / n_heads; c.n_layers = n_layers; c.seq = seq;
    c.ane_ratio = ane_ratio;
    c.ane_hidden = ((int)(hidden * ane_ratio) / 16) * 16;
    if (c.ane_hidden < 16) c.ane_hidden = 16;
    c.gpu_hidden = hidden - c.ane_hidden;
    if (c.gpu_hidden <= 0) { c.gpu_hidden = 16; c.ane_hidden = hidden - 16; }
    return c;
}

// Layer weights (all Metal buffers, randomly initialized)
typedef struct {
    id<MTLBuffer> attn_norm_w;  // [dim]
    id<MTLBuffer> ffn_norm_w;   // [dim]
    id<MTLBuffer> Wqkv;         // [3*dim, dim] fused QKV
    id<MTLBuffer> Wo;           // [dim, dim]
    id<MTLBuffer> W1W3_gpu;     // [2*gpu_hidden, dim]
    id<MTLBuffer> W2_gpu;       // [dim, gpu_hidden]
    MPSMatrixMultiplication *mps_qkv, *mps_wo, *mps_w1w3, *mps_w2;
} LayerWeights;

// Pre-allocated state for layer execution
typedef struct {
    // Main tensor — IOSurface for ANE sharing
    IOSurfaceRef ioX;
    id<MTLBuffer> bufX;         // view of ioX

    // Attention intermediates (GPU-only)
    id<MTLBuffer> bufXNorm;     // [dim, seq]
    id<MTLBuffer> bufQKV;       // [3*dim, seq]
    id<MTLBuffer> bufScores;    // [n_heads, seq, seq]
    id<MTLBuffer> bufContext;   // [dim, seq]
    id<MTLBuffer> bufAttnOut;   // [dim, seq]

    // FFN intermediates
    id<MTLBuffer> bufW1W3Gpu;   // [2*gpu_hidden, seq]
    id<MTLBuffer> bufActGpu;    // [gpu_hidden, seq]
    id<MTLBuffer> bufW2GpuPart; // [dim, seq]

    // ANE IOSurfaces for FFN
    IOSurfaceRef ioAneW1W3;     // ANE W1W3 output [2*ane_hidden, seq]
    IOSurfaceRef ioActAne;      // silu_mul ANE result [ane_hidden, seq]
    IOSurfaceRef ioAneW2Out;    // ANE W2 output [dim, seq]
    id<MTLBuffer> bufAneW1W3;   // view of ioAneW1W3
    id<MTLBuffer> bufActAneView;// view of ioActAne
    id<MTLBuffer> bufAneW2View; // view of ioAneW2Out

    id<MTLBuffer> bufFFNOut;    // [dim, seq]

    // ANE kernels (reused across layers)
    ANEKernel *aneW1W3;
    ANEKernel *aneW2;
} LayerState;

static id<MTLBuffer> make_buf(id<MTLDevice> dev, size_t bytes) {
    return [dev newBufferWithLength:((bytes + 15) & ~15) options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> buf_from_surface(id<MTLDevice> dev, IOSurfaceRef s) {
    return [dev newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(s)
                                 length:IOSurfaceGetAllocSize(s)
                                options:MTLResourceStorageModeShared
                            deallocator:nil];
}

static LayerWeights create_weights(id<MTLDevice> dev, id<MTLCommandQueue> q, TConfig *c) {
    LayerWeights w;
    int seq = c->seq, dim = c->dim;

    w.attn_norm_w = make_buf(dev, dim * 4);
    fill_random(w.attn_norm_w.contents, dim);
    // Set norm weights to ~1.0 for stability
    float *nw = (float *)w.attn_norm_w.contents;
    for (int i = 0; i < dim; i++) nw[i] = 1.0f + nw[i];

    w.ffn_norm_w = make_buf(dev, dim * 4);
    fill_random(w.ffn_norm_w.contents, dim);
    nw = (float *)w.ffn_norm_w.contents;
    for (int i = 0; i < dim; i++) nw[i] = 1.0f + nw[i];

    // QKV fused weight [3*dim, dim]
    w.Wqkv = make_buf(dev, 3 * dim * dim * 4);
    fill_random(w.Wqkv.contents, 3 * dim * dim);

    // Output projection [dim, dim]
    w.Wo = make_buf(dev, dim * dim * 4);
    fill_random(w.Wo.contents, dim * dim);

    // FFN GPU shards
    w.W1W3_gpu = make_buf(dev, 2 * c->gpu_hidden * dim * 4);
    fill_random(w.W1W3_gpu.contents, 2 * c->gpu_hidden * dim);
    w.W2_gpu = make_buf(dev, dim * c->gpu_hidden * 4);
    fill_random(w.W2_gpu.contents, dim * c->gpu_hidden);

    // MPS kernels
    // QKV: [3*dim, dim] @ [dim, seq] → [3*dim, seq]
    MPSMatrixDescriptor *dQKVw = [MPSMatrixDescriptor matrixDescriptorWithRows:3*dim columns:dim
        rowBytes:dim*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dX = [MPSMatrixDescriptor matrixDescriptorWithRows:dim columns:seq
        rowBytes:seq*4 dataType:MPSDataTypeFloat32];
    (void)dQKVw; (void)dX;
    w.mps_qkv = [[MPSMatrixMultiplication alloc] initWithDevice:dev
        transposeLeft:NO transposeRight:NO
        resultRows:3*dim resultColumns:seq interiorColumns:dim alpha:1.0 beta:0.0];

    // Wo: [dim, dim] @ [dim, seq] → [dim, seq]
    w.mps_wo = [[MPSMatrixMultiplication alloc] initWithDevice:dev
        transposeLeft:NO transposeRight:NO
        resultRows:dim resultColumns:seq interiorColumns:dim alpha:1.0 beta:0.0];

    // W1W3 GPU: [2*gpu_hidden, dim] @ [dim, seq] → [2*gpu_hidden, seq]
    w.mps_w1w3 = [[MPSMatrixMultiplication alloc] initWithDevice:dev
        transposeLeft:NO transposeRight:NO
        resultRows:2*c->gpu_hidden resultColumns:seq interiorColumns:dim alpha:1.0 beta:0.0];

    // W2 GPU: [dim, gpu_hidden] @ [gpu_hidden, seq] → [dim, seq]
    w.mps_w2 = [[MPSMatrixMultiplication alloc] initWithDevice:dev
        transposeLeft:NO transposeRight:NO
        resultRows:dim resultColumns:seq interiorColumns:c->gpu_hidden alpha:1.0 beta:0.0];

    return w;
}

static LayerState create_state(id<MTLDevice> dev, TConfig *c) {
    LayerState s;
    int dim = c->dim, seq = c->seq, nh = c->n_heads;

    // Main tensor
    s.ioX = create_surface(dim * seq * 4);
    s.bufX = buf_from_surface(dev, s.ioX);

    // Attention
    s.bufXNorm = make_buf(dev, dim * seq * 4);
    s.bufQKV = make_buf(dev, 3 * dim * seq * 4);
    s.bufScores = make_buf(dev, nh * seq * seq * 4);
    s.bufContext = make_buf(dev, dim * seq * 4);
    s.bufAttnOut = make_buf(dev, dim * seq * 4);

    // FFN GPU
    s.bufW1W3Gpu = make_buf(dev, 2 * c->gpu_hidden * seq * 4);
    s.bufActGpu = make_buf(dev, c->gpu_hidden * seq * 4);
    s.bufW2GpuPart = make_buf(dev, dim * seq * 4);

    // FFN ANE IOSurfaces
    s.ioAneW1W3 = create_surface(2 * c->ane_hidden * seq * 4);
    s.ioActAne = create_surface(c->ane_hidden * seq * 4);
    s.ioAneW2Out = create_surface(dim * seq * 4);
    s.bufAneW1W3 = buf_from_surface(dev, s.ioAneW1W3);
    s.bufActAneView = buf_from_surface(dev, s.ioActAne);
    s.bufAneW2View = buf_from_surface(dev, s.ioAneW2Out);

    s.bufFFNOut = make_buf(dev, dim * seq * 4);

    // ANE kernels (fused W1W3 and W2)
    s.aneW1W3 = compile_ane(dim, 2 * c->ane_hidden, seq, s.ioX, s.ioAneW1W3);
    if (!s.aneW1W3) {
        fprintf(stderr, "WARN: Fused W1W3 ANE compile failed (out_ch=%d). Trying separate.\n", 2*c->ane_hidden);
        // Fallback: compile W1 and W3 separately, store W1 in aneW1W3 (we'll handle in eval)
        s.aneW1W3 = compile_ane(dim, c->ane_hidden, seq, s.ioX, s.ioAneW1W3);
    }
    s.aneW2 = compile_ane(c->ane_hidden, dim, seq, s.ioActAne, s.ioAneW2Out);
    if (!s.aneW2)
        fprintf(stderr, "WARN: W2 ANE compile failed (in_ch=%d, out_ch=%d)\n", c->ane_hidden, dim);

    return s;
}

static void free_state(LayerState *s) {
    if (s->aneW1W3) ane_free(s->aneW1W3);
    if (s->aneW2) ane_free(s->aneW2);
    if (s->ioX) CFRelease(s->ioX);
    if (s->ioAneW1W3) CFRelease(s->ioAneW1W3);
    if (s->ioActAne) CFRelease(s->ioActAne);
    if (s->ioAneW2Out) CFRelease(s->ioAneW2Out);
}

// ============================================================================
// 6. ATTENTION BLOCK (GPU-only)
// ============================================================================

static void run_attention(id<MTLCommandQueue> q, ShaderPipelines *sh,
                           LayerWeights *w, LayerState *s, TConfig *c) {
    int dim = c->dim, seq = c->seq, nh = c->n_heads, hd = c->head_dim;

    // MPS matrix descriptors for this call
    MPSMatrixDescriptor *dNorm = [MPSMatrixDescriptor matrixDescriptorWithRows:dim columns:seq
        rowBytes:seq*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dQKV = [MPSMatrixDescriptor matrixDescriptorWithRows:3*dim columns:seq
        rowBytes:seq*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dWqkv = [MPSMatrixDescriptor matrixDescriptorWithRows:3*dim columns:dim
        rowBytes:dim*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dCtx = [MPSMatrixDescriptor matrixDescriptorWithRows:dim columns:seq
        rowBytes:seq*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dWo = [MPSMatrixDescriptor matrixDescriptorWithRows:dim columns:dim
        rowBytes:dim*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dOut = [MPSMatrixDescriptor matrixDescriptorWithRows:dim columns:seq
        rowBytes:seq*4 dataType:MPSDataTypeFloat32];

    MPSMatrix *mNorm = [[MPSMatrix alloc] initWithBuffer:s->bufXNorm descriptor:dNorm];
    MPSMatrix *mQKV = [[MPSMatrix alloc] initWithBuffer:s->bufQKV descriptor:dQKV];
    MPSMatrix *mWqkv = [[MPSMatrix alloc] initWithBuffer:w->Wqkv descriptor:dWqkv];
    MPSMatrix *mCtx = [[MPSMatrix alloc] initWithBuffer:s->bufContext descriptor:dCtx];
    MPSMatrix *mWo = [[MPSMatrix alloc] initWithBuffer:w->Wo descriptor:dWo];
    MPSMatrix *mOut = [[MPSMatrix alloc] initWithBuffer:s->bufAttnOut descriptor:dOut];

    id<MTLCommandBuffer> cb = [q commandBuffer];

    // 1. QKV projection: [3*dim, dim] @ [dim, seq] → [3*dim, seq]
    [w->mps_qkv encodeToCommandBuffer:cb leftMatrix:mWqkv rightMatrix:mNorm resultMatrix:mQKV];

    // 2. Attention scores: custom shader [n_heads, seq, seq]
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sh->attn_scores];
        [enc setBuffer:s->bufQKV offset:0 atIndex:0];                    // Q portion
        [enc setBuffer:s->bufQKV offset:dim*seq*4 atIndex:1];            // K portion
        [enc setBuffer:s->bufScores offset:0 atIndex:2];
        int seq_val = seq, hd_val = hd;
        float scale = 1.0f / sqrtf((float)hd);
        [enc setBytes:&seq_val length:4 atIndex:3];
        [enc setBytes:&hd_val length:4 atIndex:4];
        [enc setBytes:&scale length:4 atIndex:5];
        [enc dispatchThreads:MTLSizeMake(seq, seq, nh) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        [enc endEncoding];
    }

    // 3. Causal softmax
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sh->causal_softmax];
        [enc setBuffer:s->bufScores offset:0 atIndex:0];
        int seq_val = seq;
        [enc setBytes:&seq_val length:4 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(nh * seq, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
    }

    // 4. Attention context
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sh->attn_context];
        [enc setBuffer:s->bufScores offset:0 atIndex:0];
        [enc setBuffer:s->bufQKV offset:2*dim*seq*4 atIndex:1];         // V portion
        [enc setBuffer:s->bufContext offset:0 atIndex:2];
        int seq_val = seq, hd_val = hd;
        [enc setBytes:&seq_val length:4 atIndex:3];
        [enc setBytes:&hd_val length:4 atIndex:4];
        [enc dispatchThreads:MTLSizeMake(hd, seq, nh) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        [enc endEncoding];
    }

    // 5. Output projection: [dim, dim] @ [dim, seq] → [dim, seq]
    [w->mps_wo encodeToCommandBuffer:cb leftMatrix:mWo rightMatrix:mCtx resultMatrix:mOut];

    [cb commit];
    [cb waitUntilCompleted];
}

// ============================================================================
// 7. SWIGLU SPLIT FFN
// ============================================================================

static void run_swiglu_split(id<MTLDevice> dev, id<MTLCommandQueue> q,
                              ShaderPipelines *sh, LayerWeights *w, LayerState *s, TConfig *c) {
    int dim = c->dim, seq = c->seq;
    int gpu_h = c->gpu_hidden, ane_h = c->ane_hidden;

    // MPS matrices for GPU shards
    MPSMatrixDescriptor *dNorm = [MPSMatrixDescriptor matrixDescriptorWithRows:dim columns:seq
        rowBytes:seq*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dGpuW1W3 = [MPSMatrixDescriptor matrixDescriptorWithRows:2*gpu_h columns:seq
        rowBytes:seq*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dW1W3w = [MPSMatrixDescriptor matrixDescriptorWithRows:2*gpu_h columns:dim
        rowBytes:dim*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dActGpu = [MPSMatrixDescriptor matrixDescriptorWithRows:gpu_h columns:seq
        rowBytes:seq*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dW2w = [MPSMatrixDescriptor matrixDescriptorWithRows:dim columns:gpu_h
        rowBytes:gpu_h*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dW2out = [MPSMatrixDescriptor matrixDescriptorWithRows:dim columns:seq
        rowBytes:seq*4 dataType:MPSDataTypeFloat32];

    MPSMatrix *mNorm = [[MPSMatrix alloc] initWithBuffer:s->bufXNorm descriptor:dNorm];
    MPSMatrix *mGpuW1W3 = [[MPSMatrix alloc] initWithBuffer:s->bufW1W3Gpu descriptor:dGpuW1W3];
    MPSMatrix *mW1W3w = [[MPSMatrix alloc] initWithBuffer:w->W1W3_gpu descriptor:dW1W3w];
    MPSMatrix *mActGpu = [[MPSMatrix alloc] initWithBuffer:s->bufActGpu descriptor:dActGpu];
    MPSMatrix *mW2w = [[MPSMatrix alloc] initWithBuffer:w->W2_gpu descriptor:dW2w];
    MPSMatrix *mW2out = [[MPSMatrix alloc] initWithBuffer:s->bufW2GpuPart descriptor:dW2out];

    // ── Phase 1: Concurrent W1W3 ─────────────────────────────────────
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    dispatch_queue_t ane_q = dispatch_queue_create("ane.ffn", DISPATCH_QUEUE_SERIAL);

    // Launch ANE W1W3 (reads from ioX via compile-time binding)
    if (s->aneW1W3) {
        dispatch_async(ane_q, ^{
            ane_eval(s->aneW1W3);
            dispatch_semaphore_signal(sem);
        });
    }

    // GPU W1W3: [2*gpu_hidden, dim] @ [dim, seq] → [2*gpu_hidden, seq]
    {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        [w->mps_w1w3 encodeToCommandBuffer:cb leftMatrix:mW1W3w rightMatrix:mNorm resultMatrix:mGpuW1W3];
        [cb commit];
        [cb waitUntilCompleted];
    }

    // Wait for ANE W1W3
    if (s->aneW1W3) dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

    // ── Phase 2: SiLU × Mul on both shards ────────────────────────────
    {
        id<MTLCommandBuffer> cb = [q commandBuffer];

        // SiLU×Mul on GPU shard
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:sh->silu_mul];
            [enc setBuffer:s->bufW1W3Gpu offset:0 atIndex:0];
            [enc setBuffer:s->bufActGpu offset:0 atIndex:1];
            int half = gpu_h * seq;
            [enc setBytes:&half length:4 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(half, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // SiLU×Mul on ANE shard (GPU reads ANE IOSurface, writes to act IOSurface)
        if (s->aneW1W3) {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:sh->silu_mul];
            [enc setBuffer:s->bufAneW1W3 offset:0 atIndex:0];
            [enc setBuffer:s->bufActAneView offset:0 atIndex:1];
            int half = ane_h * seq;
            [enc setBytes:&half length:4 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(half, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        [cb commit];
        [cb waitUntilCompleted];
    }

    // ── Phase 3: Concurrent W2 ────────────────────────────────────────
    // ANE W2: reads from ioActAne [ane_hidden, seq] → writes ioAneW2Out [dim, seq]
    if (s->aneW2) {
        dispatch_async(ane_q, ^{
            ane_eval(s->aneW2);
            dispatch_semaphore_signal(sem);
        });
    }

    // GPU W2: [dim, gpu_hidden] @ [gpu_hidden, seq] → [dim, seq]
    {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        [w->mps_w2 encodeToCommandBuffer:cb leftMatrix:mW2w rightMatrix:mActGpu resultMatrix:mW2out];
        [cb commit];
        [cb waitUntilCompleted];
    }

    // Wait for ANE W2
    if (s->aneW2) dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

    // ── Phase 4: Add partials → FFN output ────────────────────────────
    {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sh->elem_add];
        [enc setBuffer:s->bufW2GpuPart offset:0 atIndex:0];
        [enc setBuffer:s->bufAneW2View offset:0 atIndex:1];
        [enc setBuffer:s->bufFFNOut offset:0 atIndex:2];
        int total = dim * seq;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

// GPU-only SwiGLU state (pre-allocated)
typedef struct {
    id<MTLBuffer> W1W3_full, W2_full, buf_w1w3_out, buf_act;
    MPSMatrix *mIn, *mW1W3w, *mW1W3o, *mAct, *mW2w, *mOut;
    MPSMatrixMultiplication *mps1, *mps2;
} GpuOnlyFFNState;

static GpuOnlyFFNState create_gpu_only_ffn(id<MTLDevice> dev, TConfig *c,
                                            id<MTLBuffer> bufIn, id<MTLBuffer> bufOut) {
    GpuOnlyFFNState g;
    int dim = c->dim, seq = c->seq, hidden = c->hidden;

    g.W1W3_full = make_buf(dev, 2 * hidden * dim * 4);
    fill_random(g.W1W3_full.contents, 2 * hidden * dim);
    g.W2_full = make_buf(dev, dim * hidden * 4);
    fill_random(g.W2_full.contents, dim * hidden);
    g.buf_w1w3_out = make_buf(dev, 2 * hidden * seq * 4);
    g.buf_act = make_buf(dev, hidden * seq * 4);

    MPSMatrixDescriptor *dIn = [MPSMatrixDescriptor matrixDescriptorWithRows:dim columns:seq
        rowBytes:seq*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dW1W3w = [MPSMatrixDescriptor matrixDescriptorWithRows:2*hidden columns:dim
        rowBytes:dim*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dW1W3o = [MPSMatrixDescriptor matrixDescriptorWithRows:2*hidden columns:seq
        rowBytes:seq*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dAct = [MPSMatrixDescriptor matrixDescriptorWithRows:hidden columns:seq
        rowBytes:seq*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dW2w = [MPSMatrixDescriptor matrixDescriptorWithRows:dim columns:hidden
        rowBytes:hidden*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dOut = [MPSMatrixDescriptor matrixDescriptorWithRows:dim columns:seq
        rowBytes:seq*4 dataType:MPSDataTypeFloat32];

    g.mIn = [[MPSMatrix alloc] initWithBuffer:bufIn descriptor:dIn];
    g.mW1W3w = [[MPSMatrix alloc] initWithBuffer:g.W1W3_full descriptor:dW1W3w];
    g.mW1W3o = [[MPSMatrix alloc] initWithBuffer:g.buf_w1w3_out descriptor:dW1W3o];
    g.mAct = [[MPSMatrix alloc] initWithBuffer:g.buf_act descriptor:dAct];
    g.mW2w = [[MPSMatrix alloc] initWithBuffer:g.W2_full descriptor:dW2w];
    g.mOut = [[MPSMatrix alloc] initWithBuffer:bufOut descriptor:dOut];

    g.mps1 = [[MPSMatrixMultiplication alloc] initWithDevice:dev
        transposeLeft:NO transposeRight:NO
        resultRows:2*hidden resultColumns:seq interiorColumns:dim alpha:1.0 beta:0.0];
    g.mps2 = [[MPSMatrixMultiplication alloc] initWithDevice:dev
        transposeLeft:NO transposeRight:NO
        resultRows:dim resultColumns:seq interiorColumns:hidden alpha:1.0 beta:0.0];

    return g;
}

static void run_swiglu_gpu_only(id<MTLCommandQueue> q, ShaderPipelines *sh,
                                 GpuOnlyFFNState *g, int hidden, int seq) {
    id<MTLCommandBuffer> cb = [q commandBuffer];

    // W1W3
    [g->mps1 encodeToCommandBuffer:cb leftMatrix:g->mW1W3w rightMatrix:g->mIn resultMatrix:g->mW1W3o];

    // silu_mul
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:sh->silu_mul];
    [enc setBuffer:g->buf_w1w3_out offset:0 atIndex:0];
    [enc setBuffer:g->buf_act offset:0 atIndex:1];
    int half = hidden * seq;
    [enc setBytes:&half length:4 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(half, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];

    // W2
    [g->mps2 encodeToCommandBuffer:cb leftMatrix:g->mW2w rightMatrix:g->mAct resultMatrix:g->mOut];

    [cb commit];
    [cb waitUntilCompleted];
}

// ============================================================================
// 8. FULL TRANSFORMER LAYER
// ============================================================================

static void run_layer(id<MTLDevice> dev, id<MTLCommandQueue> q,
                       ShaderPipelines *sh, LayerWeights *w, LayerState *s, TConfig *c) {
    int dim = c->dim, seq = c->seq;

    // ── Attention block ──────────────────────────────────
    // RMSNorm
    {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sh->rmsnorm];
        [enc setBuffer:s->bufX offset:0 atIndex:0];
        [enc setBuffer:w->attn_norm_w offset:0 atIndex:1];
        [enc setBuffer:s->bufXNorm offset:0 atIndex:2];
        int d = dim, sq = seq;
        [enc setBytes:&d length:4 atIndex:3];
        [enc setBytes:&sq length:4 atIndex:4];
        [enc dispatchThreads:MTLSizeMake(seq, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        [cb commit]; [cb waitUntilCompleted];
    }

    // Attention
    run_attention(q, sh, w, s, c);

    // Residual: x = x + attn_out
    {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sh->elem_add];
        [enc setBuffer:s->bufX offset:0 atIndex:0];
        [enc setBuffer:s->bufAttnOut offset:0 atIndex:1];
        [enc setBuffer:s->bufX offset:0 atIndex:2]; // in-place
        int total = dim * seq;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit]; [cb waitUntilCompleted];
    }

    // ── FFN block (SwiGLU split) ─────────────────────────
    // RMSNorm
    {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sh->rmsnorm];
        [enc setBuffer:s->bufX offset:0 atIndex:0];
        [enc setBuffer:w->ffn_norm_w offset:0 atIndex:1];
        [enc setBuffer:s->bufXNorm offset:0 atIndex:2];
        int d = dim, sq = seq;
        [enc setBytes:&d length:4 atIndex:3];
        [enc setBytes:&sq length:4 atIndex:4];
        [enc dispatchThreads:MTLSizeMake(seq, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        [cb commit]; [cb waitUntilCompleted];
    }

    // Copy normalized x to IOSurface for ANE (bufXNorm → ioX)
    memcpy(IOSurfaceGetBaseAddress(s->ioX), s->bufXNorm.contents, dim * seq * 4);

    // SwiGLU split FFN
    run_swiglu_split(dev, q, sh, w, s, c);

    // Residual: x = x + ffn_out
    {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sh->elem_add];
        [enc setBuffer:s->bufX offset:0 atIndex:0];
        [enc setBuffer:s->bufFFNOut offset:0 atIndex:1];
        [enc setBuffer:s->bufX offset:0 atIndex:2];
        int total = dim * seq;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit]; [cb waitUntilCompleted];
    }
}

// GPU-only layer (for baseline)
static void run_layer_gpu_only(id<MTLDevice> dev, id<MTLCommandQueue> q,
                                ShaderPipelines *sh, LayerWeights *w, LayerState *s, TConfig *c,
                                GpuOnlyFFNState *gf) {
    int dim = c->dim, seq = c->seq;

    // Attention block (same as split — attention stays GPU-only)
    {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sh->rmsnorm];
        [enc setBuffer:s->bufX offset:0 atIndex:0];
        [enc setBuffer:w->attn_norm_w offset:0 atIndex:1];
        [enc setBuffer:s->bufXNorm offset:0 atIndex:2];
        int d = dim, sq = seq;
        [enc setBytes:&d length:4 atIndex:3];
        [enc setBytes:&sq length:4 atIndex:4];
        [enc dispatchThreads:MTLSizeMake(seq, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        [cb commit]; [cb waitUntilCompleted];
    }
    run_attention(q, sh, w, s, c);
    {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sh->elem_add];
        [enc setBuffer:s->bufX offset:0 atIndex:0];
        [enc setBuffer:s->bufAttnOut offset:0 atIndex:1];
        [enc setBuffer:s->bufX offset:0 atIndex:2];
        int total = dim * seq;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit]; [cb waitUntilCompleted];
    }

    // FFN block GPU-only
    {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sh->rmsnorm];
        [enc setBuffer:s->bufX offset:0 atIndex:0];
        [enc setBuffer:w->ffn_norm_w offset:0 atIndex:1];
        [enc setBuffer:s->bufXNorm offset:0 atIndex:2];
        int d = dim, sq = seq;
        [enc setBytes:&d length:4 atIndex:3];
        [enc setBytes:&sq length:4 atIndex:4];
        [enc dispatchThreads:MTLSizeMake(seq, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        [cb commit]; [cb waitUntilCompleted];
    }
    run_swiglu_gpu_only(q, sh, gf, c->hidden, c->seq);
    {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sh->elem_add];
        [enc setBuffer:s->bufX offset:0 atIndex:0];
        [enc setBuffer:s->bufFFNOut offset:0 atIndex:1];
        [enc setBuffer:s->bufX offset:0 atIndex:2];
        int total = dim * seq;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit]; [cb waitUntilCompleted];
    }
}

// ============================================================================
// 9. BENCHMARKS
// ============================================================================

static void bench_swiglu(id<MTLDevice> dev, id<MTLCommandQueue> q, ShaderPipelines *sh) {
    printf("\n================================================================\n");
    printf("  Benchmark 1: SwiGLU Split vs GPU-only\n");
    printf("================================================================\n\n");

    typedef struct { const char *name; int dim; int hidden; int seq; } SC;
    SC configs[] = {
        {"Stories110M/256", 768, 2048, 256},
        {"Qwen3-0.6B/256", 1024, 2816, 256},
        {"Llama-7B/128",   4096, 11008, 128},
    };
    int nc = 3;
    float ratios[] = {0.7, 0.8, 0.9};
    int nr = 3;

    printf("  %-18s %6s %8s %8s %10s %10s %8s\n",
           "Config", "Ratio", "ANE ch", "GPU ch", "GPU-only", "Split μs", "Speedup");
    printf("  %-18s %6s %8s %8s %10s %10s %8s\n",
           "------------------", "------", "--------", "--------", "----------", "----------", "--------");

    for (int ci = 0; ci < nc; ci++) {
        SC sc = configs[ci];

        // GPU-only baseline (pre-allocate once)
        TConfig c0 = make_config(sc.dim, sc.hidden, sc.dim/128 > 0 ? sc.dim/128 : 1, 1, sc.seq, 0.5);
        printf("  [DBG] Config %s: dim=%d hidden=%d seq=%d n_heads=%d\n", sc.name, c0.dim, c0.hidden, c0.seq, c0.n_heads);
        printf("  [DBG] Creating GPU-only buffers...\n");
        id<MTLBuffer> bufIn = make_buf(dev, sc.dim * sc.seq * 4);
        fill_random(bufIn.contents, sc.dim * sc.seq);
        id<MTLBuffer> bufOut = make_buf(dev, sc.dim * sc.seq * 4);
        printf("  [DBG] Creating GPU-only FFN state...\n");
        GpuOnlyFFNState gf0 = create_gpu_only_ffn(dev, &c0, bufIn, bufOut);
        printf("  [DBG] GPU-only FFN state created. Running warmup...\n");

        // Warmup
        for (int i = 0; i < 3; i++) {
            printf("  [DBG] Warmup iter %d...\n", i);
            run_swiglu_gpu_only(q, sh, &gf0, c0.hidden, c0.seq);
        }
        printf("  [DBG] Warmup done. Timing...\n");
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < 20; i++) run_swiglu_gpu_only(q, sh, &gf0, c0.hidden, c0.seq);
        double gpu_us = ticks_to_us(mach_absolute_time() - t0) / 20;
        printf("  [DBG] GPU-only: %.0f μs. Now compiling split configs...\n", gpu_us);

        for (int ri = 0; ri < nr; ri++) {
            TConfig c = make_config(sc.dim, sc.hidden, sc.dim/128, 1, sc.seq, ratios[ri]);
            LayerWeights w = create_weights(dev, q, &c);
            LayerState s = create_state(dev, &c);

            if (!s.aneW1W3 || !s.aneW2) {
                printf("  %-18s %6.1f %8s\n", sc.name, ratios[ri], "FAIL");
                free_state(&s);
                continue;
            }

            // Fill input IOSurface
            fill_random(IOSurfaceGetBaseAddress(s.ioX), c.dim * c.seq);
            memcpy(s.bufXNorm.contents, IOSurfaceGetBaseAddress(s.ioX), c.dim * c.seq * 4);

            // Warmup
            for (int i = 0; i < 3; i++) run_swiglu_split(dev, q, sh, &w, &s, &c);

            t0 = mach_absolute_time();
            for (int i = 0; i < 20; i++) run_swiglu_split(dev, q, sh, &w, &s, &c);
            double split_us = ticks_to_us(mach_absolute_time() - t0) / 20;

            double speedup = gpu_us / split_us;
            printf("  %-18s %6.1f %8d %8d %10.0f %10.0f %7.2fx %s\n",
                   sc.name, ratios[ri], c.ane_hidden, c.gpu_hidden,
                   gpu_us, split_us, speedup, speedup > 1.0 ? "✓" : "✗");

            free_state(&s);
        }
    }
}

static void bench_layer(id<MTLDevice> dev, id<MTLCommandQueue> q, ShaderPipelines *sh) {
    printf("\n================================================================\n");
    printf("  Benchmark 2: Full Transformer Layer (Split vs GPU-only)\n");
    printf("================================================================\n\n");

    typedef struct { const char *name; int dim; int hidden; int seq; } SC;
    SC configs[] = {
        {"Llama-7B/128", 4096, 11008, 128},
    };
    int nc = 1;
    float ratios[] = {0.8};
    int nr = 1;

    for (int ci = 0; ci < nc; ci++) {
        SC sc = configs[ci];
        int n_heads = sc.dim / 128;

        // GPU-only layer
        TConfig c0 = make_config(sc.dim, sc.hidden, n_heads, 1, sc.seq, 0.5);
        LayerWeights w0 = create_weights(dev, q, &c0);
        LayerState s0 = create_state(dev, &c0);
        fill_random(s0.bufX.contents, c0.dim * c0.seq);
        GpuOnlyFFNState gf0 = create_gpu_only_ffn(dev, &c0, s0.bufXNorm, s0.bufFFNOut);

        for (int i = 0; i < 3; i++) run_layer_gpu_only(dev, q, sh, &w0, &s0, &c0, &gf0);
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < 10; i++) run_layer_gpu_only(dev, q, sh, &w0, &s0, &c0, &gf0);
        double gpu_layer_us = ticks_to_us(mach_absolute_time() - t0) / 10;

        printf("  %s:\n", sc.name);
        printf("    GPU-only layer: %.0f μs\n", gpu_layer_us);

        free_state(&s0);

        // Split layer
        for (int ri = 0; ri < nr; ri++) {
            TConfig c = make_config(sc.dim, sc.hidden, n_heads, 1, sc.seq, ratios[ri]);
            LayerWeights w = create_weights(dev, q, &c);
            LayerState s = create_state(dev, &c);

            if (!s.aneW1W3 || !s.aneW2) {
                printf("    Split ratio=%.1f: FAIL\n", ratios[ri]);
                free_state(&s);
                continue;
            }

            fill_random(s.bufX.contents, c.dim * c.seq);
            fill_random(IOSurfaceGetBaseAddress(s.ioX), c.dim * c.seq);

            for (int i = 0; i < 3; i++) run_layer(dev, q, sh, &w, &s, &c);
            t0 = mach_absolute_time();
            for (int i = 0; i < 10; i++) run_layer(dev, q, sh, &w, &s, &c);
            double split_layer_us = ticks_to_us(mach_absolute_time() - t0) / 10;

            double speedup = gpu_layer_us / split_layer_us;
            printf("    Split ratio=%.1f: %.0f μs (%.2fx %s)\n",
                   ratios[ri], split_layer_us, speedup, speedup > 1.0 ? "✓ WIN" : "✗ LOSE");

            // Breakdown: estimate attention vs FFN
            // Run attention-only timing
            fill_random(s.bufXNorm.contents, c.dim * c.seq);
            t0 = mach_absolute_time();
            for (int i = 0; i < 10; i++) run_attention(q, sh, &w, &s, &c);
            double attn_us = ticks_to_us(mach_absolute_time() - t0) / 10;

            fill_random(IOSurfaceGetBaseAddress(s.ioX), c.dim * c.seq);
            memcpy(s.bufXNorm.contents, IOSurfaceGetBaseAddress(s.ioX), c.dim * c.seq * 4);
            t0 = mach_absolute_time();
            for (int i = 0; i < 10; i++) run_swiglu_split(dev, q, sh, &w, &s, &c);
            double ffn_us = ticks_to_us(mach_absolute_time() - t0) / 10;

            printf("    Breakdown: attention=%.0f μs  FFN(split)=%.0f μs  overhead=%.0f μs\n",
                   attn_us, ffn_us, split_layer_us - attn_us - ffn_us);

            free_state(&s);
        }
    }
}

static void bench_multilayer(id<MTLDevice> dev, id<MTLCommandQueue> q, ShaderPipelines *sh) {
    printf("\n================================================================\n");
    printf("  Benchmark 3: 32-Layer Pipeline (IOSurface reuse)\n");
    printf("================================================================\n\n");

    int dim = 4096, hidden = 11008, n_heads = 32, seq = 128, n_layers = 32;
    float ratio = 0.8;

    TConfig c = make_config(dim, hidden, n_heads, n_layers, seq, ratio);
    LayerWeights w = create_weights(dev, q, &c);
    LayerState s = create_state(dev, &c);

    if (!s.aneW1W3 || !s.aneW2) {
        printf("  ANE compile failed. Skipping.\n");
        free_state(&s);
        return;
    }

    fill_random(s.bufX.contents, dim * seq);
    fill_random(IOSurfaceGetBaseAddress(s.ioX), dim * seq);

    printf("  Config: Llama-7B (%d layers, seq=%d, ratio=%.1f)\n", n_layers, seq, ratio);
    printf("  ANE kernels: 2 (W1W3 + W2), reused across all layers\n\n");

    // Warmup
    for (int i = 0; i < 2; i++) {
        for (int l = 0; l < n_layers; l++) run_layer(dev, q, sh, &w, &s, &c);
    }

    // Benchmark
    uint64_t t0 = mach_absolute_time();
    int n_runs = 3;
    for (int r = 0; r < n_runs; r++) {
        fill_random(s.bufX.contents, dim * seq);
        fill_random(IOSurfaceGetBaseAddress(s.ioX), dim * seq);
        for (int l = 0; l < n_layers; l++) {
            run_layer(dev, q, sh, &w, &s, &c);
        }
    }
    double total_us = ticks_to_us(mach_absolute_time() - t0) / n_runs;
    double per_layer_us = total_us / n_layers;

    // Single-layer reference
    fill_random(s.bufX.contents, dim * seq);
    fill_random(IOSurfaceGetBaseAddress(s.ioX), dim * seq);
    uint64_t t1 = mach_absolute_time();
    for (int i = 0; i < 5; i++) run_layer(dev, q, sh, &w, &s, &c);
    double single_us = ticks_to_us(mach_absolute_time() - t1) / 5;

    printf("  Single layer: %.0f μs\n", single_us);
    printf("  32-layer total: %.0f μs (%.1f ms)\n", total_us, total_us / 1000);
    printf("  Per-layer in pipeline: %.0f μs\n", per_layer_us);
    printf("  Pipeline efficiency: %.1f%% (vs %.0f μs × 32 = %.0f μs)\n",
           (single_us * n_layers) / total_us * 100, single_us, single_us * n_layers);

    // Prefill tokens/sec
    double tokens_per_sec = seq / (total_us / 1e6);
    printf("\n  Prefill throughput: %.0f tokens/sec (%.1f ms for %d tokens)\n",
           tokens_per_sec, total_us / 1000, seq);

    free_state(&s);
}

static void bench_decode(id<MTLDevice> dev, id<MTLCommandQueue> q, ShaderPipelines *sh) {
    printf("\n================================================================\n");
    printf("  Benchmark 4: Decode Mode (seq=1 → 64, KV-cache)\n");
    printf("================================================================\n\n");

    int dim = 4096, hidden = 11008, n_heads = 32;
    int decode_seqs[] = {1, 4, 8, 16, 32, 64, 128};
    int n_seqs = 7;
    int cache_len = 512;

    printf("  %-8s %10s %10s %8s %8s\n",
           "seq_len", "GPU-only", "Split μs", "Speedup", "Verdict");
    printf("  %-8s %10s %10s %8s %8s\n",
           "--------", "----------", "----------", "--------", "--------");

    for (int si = 0; si < n_seqs; si++) {
        int seq = decode_seqs[si];

        // GPU-only SwiGLU at this seq_len (pre-allocate)
        TConfig c0 = make_config(dim, hidden, n_heads, 1, seq, 0.5);
        id<MTLBuffer> bufIn = make_buf(dev, dim * seq * 4);
        fill_random(bufIn.contents, dim * seq);
        id<MTLBuffer> bufOut = make_buf(dev, dim * seq * 4);
        GpuOnlyFFNState gf0 = create_gpu_only_ffn(dev, &c0, bufIn, bufOut);

        for (int i = 0; i < 5; i++) run_swiglu_gpu_only(q, sh, &gf0, c0.hidden, c0.seq);
        uint64_t t0 = mach_absolute_time();
        int iters = seq <= 4 ? 50 : 20;
        for (int i = 0; i < iters; i++) run_swiglu_gpu_only(q, sh, &gf0, c0.hidden, c0.seq);
        double gpu_us = ticks_to_us(mach_absolute_time() - t0) / iters;

        // Split SwiGLU
        float ratio = 0.8;
        TConfig c = make_config(dim, hidden, n_heads, 1, seq, ratio);
        LayerWeights w = create_weights(dev, q, &c);
        LayerState s = create_state(dev, &c);

        if (!s.aneW1W3 || !s.aneW2) {
            printf("  %-8d %10.0f %10s\n", seq, gpu_us, "FAIL");
            free_state(&s);
            continue;
        }

        fill_random(IOSurfaceGetBaseAddress(s.ioX), c.dim * c.seq);
        memcpy(s.bufXNorm.contents, IOSurfaceGetBaseAddress(s.ioX), c.dim * c.seq * 4);

        for (int i = 0; i < 5; i++) run_swiglu_split(dev, q, sh, &w, &s, &c);
        t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++) run_swiglu_split(dev, q, sh, &w, &s, &c);
        double split_us = ticks_to_us(mach_absolute_time() - t0) / iters;

        double speedup = gpu_us / split_us;
        printf("  %-8d %10.0f %10.0f %7.2fx %8s\n",
               seq, gpu_us, split_us, speedup, speedup > 1.0 ? "✓ WIN" : "✗ LOSE");

        free_state(&s);
    }

    printf("\n  Note: At seq=1, ANE dispatch floor (~22 μs) relative to compute\n");
    printf("  determines whether splitting is profitable.\n");
}

// ============================================================================
// 10. MAIN
// ============================================================================

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        setbuf(stdout, NULL);
        setbuf(stderr, NULL);

        printf("================================================================\n");
        printf("  Transformer Layer GPU+ANE Split Benchmark\n");
        printf("================================================================\n\n");

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "FATAL: No Metal device\n"); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        printf("  GPU: %s\n", [[dev name] UTF8String]);

        if (!init_ane_classes()) {
            fprintf(stderr, "FATAL: Failed to load ANE classes\n");
            return 1;
        }
        printf("  ANE: initialized ✓\n");

        ShaderPipelines sh = setup_shaders(dev);
        printf("  Metal shaders: compiled ✓\n\n");

        // Run benchmarks
        bench_swiglu(dev, q, &sh);
        bench_layer(dev, q, &sh);
        bench_multilayer(dev, q, &sh);
        bench_decode(dev, q, &sh);

        printf("\n================================================================\n");
        printf("  All benchmarks complete.\n");
        printf("================================================================\n\n");

        return 0;
    }
}
