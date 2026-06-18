// qwen_split_infer.m — Qwen2.5-0.5B-Instruct GPU+ANE Split Inference
//
// Usage: ./qwen_split_infer <weights.bin> <token_ids_csv> <max_tokens> <eos_id>
// Output: token IDs streamed as generated, timing on stderr
//
// Features:
//   - fp16 pipeline (MPS + Metal compute)
//   - Fused online-softmax GQA decode attention
//   - ANE split for FFN (80% ANE, 20% GPU, concurrent)
//   - Batched prefill (all prompt tokens at once)
//   - Token streaming (print each token as generated)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <dispatch/dispatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// 1. CONFIG + TIMING
// ============================================================================

static mach_timebase_info_data_t g_tb;
static double ticks_us(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1000.0; }

#define MAX_LAYERS 24
#define MAX_SEQ 2048
#define ANE_RATIO 0.0f  // ANE disabled: conv I/O format needs debugging. GPU-only: 33-41 tok/s

typedef struct {
    int n_layers, dim, hidden, n_heads, n_kv_heads, vocab, head_dim, kv_dim;
    int heads_per_group;
    int ane_hidden, gpu_hidden;
    float rope_theta;
} QConfig;

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

static IOSurfaceRef create_surface(size_t bytes) {
    size_t ps = getpagesize();
    size_t as = ((bytes + ps - 1) / ps) * ps;
    if (as < ps) as = ps;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(as), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(as),
        (id)kIOSurfaceAllocSize: @(as), (id)kIOSurfacePixelFormat: @0
    });
}

static id<MTLBuffer> buf_from_surface(id<MTLDevice> dev, IOSurfaceRef s) {
    return [dev newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(s)
                                 length:IOSurfaceGetAllocSize(s)
                                options:MTLResourceStorageModeShared
                            deallocator:nil];
}

// Build ANE weight blob with real weights (fp16)
// 128-byte header (ANE blob format), weight data at byte 128.
// The MIL BLOBFILE offset=64 points to the blob's secondary header at byte 64,
// which contains data_offset=128, telling the ANE to read weights from byte 128.
static uint8_t *build_weight_blob_real(const uint16_t *weights, int out_ch, int in_ch, size_t *out_len) {
    int ws = out_ch * in_ch * 2;
    int total = 128 + ws;
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    *(uint32_t*)(buf + 72) = ws;
    *(uint32_t*)(buf + 80) = 128;
    memcpy(buf + 128, weights, ws);
    *out_len = total;
    return buf;
}

static NSString *gen_mil_conv(int in_ch, int out_ch, int spatial) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
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

static ANEKernel *compile_ane_real(int in_ch, int out_ch, int spatial,
                                    const uint16_t *weights,
                                    IOSurfaceRef ioIn, IOSurfaceRef ioOut) {
    @autoreleasepool {
        NSError *e = nil;
        NSData *mil = [[gen_mil_conv(in_ch, out_ch, spatial) dataUsingEncoding:NSUTF8StringEncoding] copy];
        size_t wbLen;
        uint8_t *wbBuf = build_weight_blob_real(weights, out_ch, in_ch, &wbLen);
        NSData *wb = [NSData dataWithBytesNoCopy:wbBuf length:wbLen freeWhenDone:YES];

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            mil, @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wb}}, nil);
        if (!desc) { fprintf(stderr, "    ANE: descriptor failed (in=%d out=%d)\n", in_ch, out_ch); return NULL; }

        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) { fprintf(stderr, "    ANE: model init failed (in=%d out=%d)\n", in_ch, out_ch); return NULL; }

        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wb writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            fprintf(stderr, "    ANE: compile failed (in=%d out=%d): %s\n",
                    in_ch, out_ch, e ? [[e description] UTF8String] : "unknown");
            [fm removeItemAtPath:td error:nil]; return NULL;
        }
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            fprintf(stderr, "    ANE: load failed (in=%d out=%d): %s\n",
                    in_ch, out_ch, e ? [[e description] UTF8String] : "unknown");
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
// 3. METAL SHADER SOURCE (FP16)
// ============================================================================

static NSString *g_shaderSrc = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void embed_lookup(const device half *table [[buffer(0)]],
                          device half *out        [[buffer(1)]],
                          constant int &token_id  [[buffer(2)]],
                          constant int &dim       [[buffer(3)]],
                          uint idx [[thread_position_in_grid]]) {
    if (idx < (uint)dim)
        out[idx] = table[(uint)token_id * (uint)dim + idx];
}

// Batched embedding: embed multiple tokens → [dim, seq_len]
kernel void embed_batch(const device half *table [[buffer(0)]],
                         const device int *ids   [[buffer(1)]],
                         device half *out         [[buffer(2)]],
                         constant int &dim        [[buffer(3)]],
                         constant int &seq_len    [[buffer(4)]],
                         uint2 gid [[thread_position_in_grid]]) {
    // gid.x = dim index, gid.y = token index
    if (gid.x < (uint)dim && gid.y < (uint)seq_len)
        out[gid.x * (uint)seq_len + gid.y] = table[(uint)ids[gid.y] * (uint)dim + gid.x];
}

kernel void rmsnorm_h(const device half *x    [[buffer(0)]],
                       const device half *w    [[buffer(1)]],
                       device half *out        [[buffer(2)]],
                       constant int &dim       [[buffer(3)]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint tgs [[threads_per_threadgroup]]) {
    threadgroup float shared[256];
    float my_ss = 0.0f;
    for (uint i = tid; i < (uint)dim; i += tgs) {
        float v = (float)x[i];
        my_ss += v * v;
    }
    shared[tid] = my_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = rsqrt(shared[0] / (float)dim + 1e-6f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < (uint)dim; i += tgs) {
        out[i] = (half)((float)x[i] * (float)w[i] * rms);
    }
}

kernel void add_bias_h(device half *x          [[buffer(0)]],
                        const device half *bias [[buffer(1)]],
                        uint idx [[thread_position_in_grid]]) {
    x[idx] += bias[idx];
}

kernel void rope_h(device half *qk            [[buffer(0)]],
                    const device float *cos_t  [[buffer(1)]],
                    const device float *sin_t  [[buffer(2)]],
                    constant int &n_heads      [[buffer(3)]],
                    constant int &head_dim     [[buffer(4)]],
                    uint idx [[thread_position_in_grid]]) {
    uint half_hd = (uint)head_dim / 2;
    uint total = (uint)n_heads * half_hd;
    if (idx >= total) return;
    uint h = idx / half_hd;
    uint i = idx % half_hd;
    uint base = h * (uint)head_dim + i;
    float c = cos_t[i];
    float s = sin_t[i];
    float x0 = (float)qk[base];
    float x1 = (float)qk[base + half_hd];
    qk[base]           = (half)(x0 * c - x1 * s);
    qk[base + half_hd] = (half)(x0 * s + x1 * c);
}

kernel void kv_cache_write(const device half *kv   [[buffer(0)]],
                            device half *cache      [[buffer(1)]],
                            constant int &pos       [[buffer(2)]],
                            constant int &max_seq   [[buffer(3)]],
                            uint idx [[thread_position_in_grid]]) {
    cache[idx * (uint)max_seq + (uint)pos] = kv[idx];
}

kernel void fused_decode_gqa(const device half *Q        [[buffer(0)]],
                              const device half *K_cache  [[buffer(1)]],
                              const device half *V_cache  [[buffer(2)]],
                              device half *out            [[buffer(3)]],
                              constant int &cache_len     [[buffer(4)]],
                              constant int &max_seq       [[buffer(5)]],
                              constant int &head_dim      [[buffer(6)]],
                              constant int &n_kv_heads    [[buffer(7)]],
                              constant int &heads_per_grp [[buffer(8)]],
                              constant float &scale       [[buffer(9)]],
                              uint2 gid [[thread_position_in_grid]]) {
    int d = gid.x;
    int qh = gid.y;
    if (d >= head_dim) return;
    int kvh = qh / heads_per_grp;
    float max_s = -INFINITY;
    float sum_exp = 0.0f;
    float weighted_v = 0.0f;
    for (int pos = 0; pos < cache_len; pos++) {
        float score = 0.0f;
        for (int dd = 0; dd < head_dim; dd++) {
            score += (float)Q[qh * head_dim + dd] *
                     (float)K_cache[(kvh * head_dim + dd) * max_seq + pos];
        }
        score *= scale;
        float new_max = max(max_s, score);
        float rescale = exp(max_s - new_max);
        float new_exp = exp(score - new_max);
        weighted_v = weighted_v * rescale +
                     new_exp * (float)V_cache[(kvh * head_dim + d) * max_seq + pos];
        sum_exp = sum_exp * rescale + new_exp;
        max_s = new_max;
    }
    out[qh * head_dim + d] = (half)(weighted_v / sum_exp);
}

// SiLU(gate) * up for split ANE results (fp32 IOSurface data)
kernel void silu_mul_f32(const device float *gate [[buffer(0)]],
                          const device float *up   [[buffer(1)]],
                          device float *act        [[buffer(2)]],
                          uint idx [[thread_position_in_grid]]) {
    float g = gate[idx];
    act[idx] = (g / (1.0f + exp(-g))) * up[idx];
}

kernel void silu_mul_h(const device half *gate [[buffer(0)]],
                        const device half *up   [[buffer(1)]],
                        device half *act        [[buffer(2)]],
                        uint idx [[thread_position_in_grid]]) {
    float g = (float)gate[idx];
    act[idx] = (half)((g / (1.0f + exp(-g))) * (float)up[idx]);
}

kernel void elem_add_h(const device half *a [[buffer(0)]],
                        const device half *b [[buffer(1)]],
                        device half *out     [[buffer(2)]],
                        uint idx [[thread_position_in_grid]]) {
    out[idx] = a[idx] + b[idx];
}

kernel void inplace_add_h(device half *x       [[buffer(0)]],
                           const device half *y [[buffer(1)]],
                           uint idx [[thread_position_in_grid]]) {
    x[idx] += y[idx];
}

// Add fp32 ANE result to fp16 buffer (cast + add)
kernel void add_f32_to_f16(device half *dst         [[buffer(0)]],
                            const device float *src  [[buffer(1)]],
                            uint idx [[thread_position_in_grid]]) {
    dst[idx] += (half)src[idx];
}

// Copy fp16 to fp32 (for ANE input)
kernel void fp16_to_fp32(const device half *src [[buffer(0)]],
                          device float *dst      [[buffer(1)]],
                          uint idx [[thread_position_in_grid]]) {
    dst[idx] = (float)src[idx];
}

// Copy fp32 to fp16 (from ANE output)
kernel void fp32_to_fp16(const device float *src [[buffer(0)]],
                          device half *dst        [[buffer(1)]],
                          uint idx [[thread_position_in_grid]]) {
    dst[idx] = (half)src[idx];
}

// Batched prefill attention: causal masked, GQA
// One thread per (query_pos, d) for a given head
// Processes ALL key positions for this query position
kernel void prefill_gqa_attn(const device half *Q       [[buffer(0)]],  // [dim, seq]
                              const device half *K       [[buffer(1)]],  // [kv_dim, seq]
                              const device half *V       [[buffer(2)]],  // [kv_dim, seq]
                              device half *out           [[buffer(3)]],  // [dim, seq]
                              constant int &seq_len      [[buffer(4)]],
                              constant int &head_dim     [[buffer(5)]],
                              constant int &n_kv_heads   [[buffer(6)]],
                              constant int &heads_per_grp[[buffer(7)]],
                              constant float &scale      [[buffer(8)]],
                              uint3 gid [[thread_position_in_grid]]) {
    int d = gid.x;       // [0, head_dim)
    int q_pos = gid.y;   // [0, seq_len)
    int qh = gid.z;      // [0, n_heads)
    if (d >= head_dim || q_pos >= seq_len) return;
    int kvh = qh / heads_per_grp;
    float max_s = -INFINITY;
    float sum_exp = 0.0f;
    float weighted_v = 0.0f;
    // Causal: only attend to positions 0..q_pos
    for (int k_pos = 0; k_pos <= q_pos; k_pos++) {
        float score = 0.0f;
        for (int dd = 0; dd < head_dim; dd++) {
            score += (float)Q[(qh * head_dim + dd) * seq_len + q_pos] *
                     (float)K[(kvh * head_dim + dd) * seq_len + k_pos];
        }
        score *= scale;
        float new_max = max(max_s, score);
        float rescale = exp(max_s - new_max);
        float new_exp = exp(score - new_max);
        weighted_v = weighted_v * rescale +
                     new_exp * (float)V[(kvh * head_dim + d) * seq_len + k_pos];
        sum_exp = sum_exp * rescale + new_exp;
        max_s = new_max;
    }
    out[(qh * head_dim + d) * seq_len + q_pos] = (half)(weighted_v / sum_exp);
}

// Batched RoPE for prefill: Q/K shape [n_heads * head_dim, seq_len]
kernel void rope_batch_h(device half *qk           [[buffer(0)]],
                          const device float *cos_t [[buffer(1)]],
                          const device float *sin_t [[buffer(2)]],
                          constant int &n_heads     [[buffer(3)]],
                          constant int &head_dim    [[buffer(4)]],
                          constant int &seq_len     [[buffer(5)]],
                          constant int &pos_offset  [[buffer(6)]],
                          uint2 gid [[thread_position_in_grid]]) {
    // gid.x = head*half_hd + pair, gid.y = seq position
    uint half_hd = (uint)head_dim / 2;
    uint total_pairs = (uint)n_heads * half_hd;
    if (gid.x >= total_pairs || gid.y >= (uint)seq_len) return;
    uint h = gid.x / half_hd;
    uint i = gid.x % half_hd;
    int pos = (int)gid.y + pos_offset;
    uint base_row = h * (uint)head_dim + i;
    uint idx0 = base_row * (uint)seq_len + gid.y;
    uint idx1 = (base_row + half_hd) * (uint)seq_len + gid.y;
    // cos/sin for this position
    uint ct_idx = (uint)pos * half_hd + i;
    float c = cos_t[ct_idx];
    float s = sin_t[ct_idx];
    float x0 = (float)qk[idx0];
    float x1 = (float)qk[idx1];
    qk[idx0] = (half)(x0 * c - x1 * s);
    qk[idx1] = (half)(x0 * s + x1 * c);
}

// Batched KV cache write: copy [kv_dim, seq] to cache at positions [0..seq-1] + offset
kernel void kv_cache_write_batch(const device half *kv  [[buffer(0)]],
                                  device half *cache     [[buffer(1)]],
                                  constant int &max_seq  [[buffer(2)]],
                                  constant int &seq_len  [[buffer(3)]],
                                  constant int &offset   [[buffer(4)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    // gid.x = channel, gid.y = seq position
    uint ch = gid.x;
    uint s = gid.y;
    if (s >= (uint)seq_len) return;
    cache[ch * (uint)max_seq + s + (uint)offset] = kv[ch * (uint)seq_len + s];
}

// Batched RMSNorm: process [dim, seq_len] column by column
kernel void rmsnorm_batch_h(const device half *x    [[buffer(0)]],
                             const device half *w    [[buffer(1)]],
                             device half *out        [[buffer(2)]],
                             constant int &dim       [[buffer(3)]],
                             constant int &seq_len   [[buffer(4)]],
                             uint2 gid [[thread_position_in_grid]]) {
    // gid.x = dim_idx, gid.y = seq position
    uint col = gid.y;
    if (col >= (uint)seq_len) return;
    // Each column needs its own RMS - use thread 0 per column for reduction
    // Simple: each thread computes full column RMS (dim=896, acceptable)
    if (gid.x > 0) return;
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) {
        float v = (float)x[i * (uint)seq_len + col];
        ss += v * v;
    }
    float rms = rsqrt(ss / (float)dim + 1e-6f);
    for (int i = 0; i < dim; i++) {
        uint idx = i * (uint)seq_len + col;
        out[idx] = (half)((float)x[idx] * (float)w[i] * rms);
    }
}

// Batched bias add for prefill: x[i * seq + j] += bias[i]
kernel void add_bias_batch_h(device half *x            [[buffer(0)]],
                              const device half *bias   [[buffer(1)]],
                              constant int &rows        [[buffer(2)]],
                              constant int &seq_len     [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= (uint)rows || gid.y >= (uint)seq_len) return;
    x[gid.x * (uint)seq_len + gid.y] += bias[gid.x];
}

// Batched silu_mul for prefill
kernel void silu_mul_batch_h(const device half *gate [[buffer(0)]],
                              const device half *up   [[buffer(1)]],
                              device half *act        [[buffer(2)]],
                              uint idx [[thread_position_in_grid]]) {
    float g = (float)gate[idx];
    act[idx] = (half)((g / (1.0f + exp(-g))) * (float)up[idx]);
}

// Batched residual add
kernel void inplace_add_batch_h(device half *x       [[buffer(0)]],
                                 const device half *y [[buffer(1)]],
                                 uint idx [[thread_position_in_grid]]) {
    x[idx] += y[idx];
}
)";

// ============================================================================
// 4. SHADER + WEIGHT STRUCTS
// ============================================================================

typedef struct {
    id<MTLComputePipelineState> embed_lookup, embed_batch;
    id<MTLComputePipelineState> rmsnorm, rmsnorm_batch;
    id<MTLComputePipelineState> add_bias, add_bias_batch;
    id<MTLComputePipelineState> rope, rope_batch;
    id<MTLComputePipelineState> kv_write, kv_write_batch;
    id<MTLComputePipelineState> fused_attn, prefill_attn;
    id<MTLComputePipelineState> silu_mul, silu_mul_f32, silu_mul_batch;
    id<MTLComputePipelineState> elem_add, inplace_add, inplace_add_batch;
    id<MTLComputePipelineState> add_f32_to_f16, fp16_to_fp32, fp32_to_fp16;
} Shaders;

typedef struct {
    id<MTLBuffer> attn_norm, q_w, q_b, k_w, k_b, v_w, v_b, o_w;
    id<MTLBuffer> ffn_norm, gate_w, up_w, down_w;
} LayerW;

typedef struct {
    MPSMatrix *q_w, *k_w, *v_w, *o_w, *gate_w, *up_w, *down_w;
} LayerMat;

// Per-layer ANE kernels
typedef struct {
    ANEKernel *w1w3;   // fused gate+up ANE kernel
    ANEKernel *w2;     // down ANE kernel
    IOSurfaceRef ioIn;     // input to ANE [dim, 1] fp32
    IOSurfaceRef ioW1W3Out;// ANE W1W3 output [2*ane_hidden, 1] fp32
    IOSurfaceRef ioActIn;  // silu_mul result [ane_hidden, 1] fp32
    IOSurfaceRef ioW2Out;  // ANE W2 output [dim, 1] fp32
    id<MTLBuffer> bufIn, bufW1W3Out, bufActIn, bufW2Out;
    // GPU shard weights
    id<MTLBuffer> gpu_gate_w;  // [gpu_hidden, dim] fp16
    id<MTLBuffer> gpu_up_w;    // [gpu_hidden, dim] fp16
    id<MTLBuffer> gpu_down_w;  // [dim, gpu_hidden] fp16
    MPSMatrix *mat_gpu_gate, *mat_gpu_up, *mat_gpu_down;
} LayerANE;

typedef struct {
    QConfig cfg;
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    Shaders sh;
    bool ane_available;

    // Weights
    id<MTLBuffer> embed, final_norm;
    LayerW lw[MAX_LAYERS];
    LayerMat lm[MAX_LAYERS];
    LayerANE la[MAX_LAYERS];

    // Decode activation buffers (fp16, seq=1)
    id<MTLBuffer> x, xnorm, q, k, v, attn_out;
    id<MTLBuffer> gate, up, act, down, logits;  // gpu_h sized (for ANE split)
    id<MTLBuffer> gate_full, up_full, act_full;  // hidden sized (for GPU-only fallback)

    // KV cache per layer
    id<MTLBuffer> k_cache[MAX_LAYERS];
    id<MTLBuffer> v_cache[MAX_LAYERS];

    // RoPE tables (fp32)
    id<MTLBuffer> rope_cos, rope_sin;

    // MPS for decode (seq=1)
    MPSMatrixMultiplication *mps_q, *mps_kv, *mps_o;
    MPSMatrixMultiplication *mps_lmhead;
    // GPU shard MPS for ANE split (gpu_hidden rows)
    MPSMatrixMultiplication *mps_gpu_gate, *mps_gpu_up, *mps_gpu_down;
    // Full-size MPS for GPU-only fallback (hidden rows)
    MPSMatrixMultiplication *mps_full_gate, *mps_full_up, *mps_full_down;

    MPSMatrix *mat_xnorm, *mat_q, *mat_k, *mat_v;
    MPSMatrix *mat_attn_out, *mat_gate, *mat_up, *mat_act, *mat_down;
    MPSMatrix *mat_gate_full, *mat_up_full, *mat_act_full;
    MPSMatrix *mat_logits, *mat_embed;
} Model;

// ============================================================================
// 5. HELPERS
// ============================================================================

static id<MTLBuffer> read_buf(id<MTLDevice> dev, FILE *f, size_t n) {
    size_t bytes = n * 2;
    id<MTLBuffer> buf = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (fread(buf.contents, 1, bytes, f) != bytes) { fprintf(stderr, "Read err\n"); exit(1); }
    return buf;
}

static id<MTLBuffer> alloc_buf(id<MTLDevice> dev, size_t n) {
    return [dev newBufferWithLength:MAX(n * 2, 16) options:MTLResourceStorageModeShared];
}

static MPSMatrix *mat16(id<MTLBuffer> buf, int rows, int cols) {
    MPSMatrixDescriptor *d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols
        rowBytes:cols * 2 dataType:MPSDataTypeFloat16];
    return [[MPSMatrix alloc] initWithBuffer:buf descriptor:d];
}

// ============================================================================
// 6. SHADER SETUP
// ============================================================================

static Shaders setup_shaders(id<MTLDevice> dev) {
    NSError *err = nil;
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    opts.fastMathEnabled = YES;
    id<MTLLibrary> lib = [dev newLibraryWithSource:g_shaderSrc options:opts error:&err];
    if (!lib) { fprintf(stderr, "Shader err: %s\n", [[err description] UTF8String]); exit(1); }

    Shaders s;
    NSArray *names = @[@"embed_lookup", @"embed_batch", @"rmsnorm_h", @"rmsnorm_batch_h",
                       @"add_bias_h", @"add_bias_batch_h", @"rope_h", @"rope_batch_h",
                       @"kv_cache_write", @"kv_cache_write_batch",
                       @"fused_decode_gqa", @"prefill_gqa_attn",
                       @"silu_mul_h", @"silu_mul_f32", @"silu_mul_batch_h",
                       @"elem_add_h", @"inplace_add_h", @"inplace_add_batch_h",
                       @"add_f32_to_f16", @"fp16_to_fp32", @"fp32_to_fp16"];
    id<MTLComputePipelineState> __strong *ptrs[] = {
        &s.embed_lookup, &s.embed_batch, &s.rmsnorm, &s.rmsnorm_batch,
        &s.add_bias, &s.add_bias_batch, &s.rope, &s.rope_batch,
        &s.kv_write, &s.kv_write_batch,
        &s.fused_attn, &s.prefill_attn,
        &s.silu_mul, &s.silu_mul_f32, &s.silu_mul_batch,
        &s.elem_add, &s.inplace_add, &s.inplace_add_batch,
        &s.add_f32_to_f16, &s.fp16_to_fp32, &s.fp32_to_fp16
    };
    for (int i = 0; i < (int)names.count; i++) {
        id<MTLFunction> fn = [lib newFunctionWithName:names[i]];
        if (!fn) { fprintf(stderr, "Missing: %s\n", [names[i] UTF8String]); exit(1); }
        *ptrs[i] = [dev newComputePipelineStateWithFunction:fn error:&err];
    }
    return s;
}

// ============================================================================
// 7. MODEL + ENGINE SETUP
// ============================================================================

static void load_model(Model *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    char magic[4]; uint32_t hdr[8];
    fread(magic, 1, 4, f); fread(hdr, 4, 8, f); fseek(f, 64, SEEK_SET);
    if (memcmp(magic, "QWEN", 4)) { fprintf(stderr, "Bad magic\n"); exit(1); }

    QConfig *c = &m->cfg;
    c->n_layers = hdr[0]; c->dim = hdr[1]; c->hidden = hdr[2];
    c->n_heads = hdr[3]; c->n_kv_heads = hdr[4]; c->vocab = hdr[5];
    c->head_dim = hdr[6]; c->kv_dim = hdr[7];
    c->heads_per_group = c->n_heads / c->n_kv_heads;
    c->rope_theta = 1000000.0f;
    c->ane_hidden = ((int)(c->hidden * ANE_RATIO) / 16) * 16;
    c->gpu_hidden = c->hidden - c->ane_hidden;
    if (c->gpu_hidden <= 0) { c->gpu_hidden = 16; c->ane_hidden = c->hidden - 16; }

    fprintf(stderr, "Model: dim=%d hidden=%d heads=%d kv=%d layers=%d\n",
            c->dim, c->hidden, c->n_heads, c->n_kv_heads, c->n_layers);
    fprintf(stderr, "Split: ane_hidden=%d gpu_hidden=%d (%.0f%% ANE)\n",
            c->ane_hidden, c->gpu_hidden, 100.0f * c->ane_hidden / c->hidden);

    id<MTLDevice> dev = m->dev;
    m->embed = read_buf(dev, f, (size_t)c->vocab * c->dim);
    for (int l = 0; l < c->n_layers; l++) {
        LayerW *w = &m->lw[l];
        w->attn_norm = read_buf(dev, f, c->dim);
        w->q_w = read_buf(dev, f, c->dim * c->dim);
        w->q_b = read_buf(dev, f, c->dim);
        w->k_w = read_buf(dev, f, c->kv_dim * c->dim);
        w->k_b = read_buf(dev, f, c->kv_dim);
        w->v_w = read_buf(dev, f, c->kv_dim * c->dim);
        w->v_b = read_buf(dev, f, c->kv_dim);
        w->o_w = read_buf(dev, f, c->dim * c->dim);
        w->ffn_norm = read_buf(dev, f, c->dim);
        w->gate_w = read_buf(dev, f, c->hidden * c->dim);
        w->up_w = read_buf(dev, f, c->hidden * c->dim);
        w->down_w = read_buf(dev, f, c->dim * c->hidden);
    }
    m->final_norm = read_buf(dev, f, c->dim);
    fclose(f);
}

static void setup_ane_split(Model *m) {
    QConfig *c = &m->cfg;
    int dim = c->dim, ane_h = c->ane_hidden, gpu_h = c->gpu_hidden;

    if (!init_ane_classes()) {
        fprintf(stderr, "ANE: framework not available, GPU-only mode\n");
        m->ane_available = false;
        return;
    }

    fprintf(stderr, "Compiling %d ANE kernels...\n", c->n_layers * 2);
    int ok = 0, fail = 0;

    for (int l = 0; l < c->n_layers; l++) {
        LayerANE *la = &m->la[l];
        LayerW *lw = &m->lw[l];
        const uint16_t *gate_data = (const uint16_t *)lw->gate_w.contents;
        const uint16_t *up_data = (const uint16_t *)lw->up_w.contents;
        const uint16_t *down_data = (const uint16_t *)lw->down_w.contents;

        // Create IOSurfaces (ANE uses fp32)
        la->ioIn = create_surface(dim * 4);
        la->ioW1W3Out = create_surface(2 * ane_h * 4);
        la->ioActIn = create_surface(ane_h * 4);
        la->ioW2Out = create_surface(dim * 4);
        la->bufIn = buf_from_surface(m->dev, la->ioIn);
        la->bufW1W3Out = buf_from_surface(m->dev, la->ioW1W3Out);
        la->bufActIn = buf_from_surface(m->dev, la->ioActIn);
        la->bufW2Out = buf_from_surface(m->dev, la->ioW2Out);

        // Build fused W1W3 weights for ANE: [2*ane_h, dim] fp16
        // = concat(gate_w[0:ane_h, :], up_w[0:ane_h, :])
        size_t fused_elems = 2 * ane_h * dim;
        uint16_t *fused_w = (uint16_t *)malloc(fused_elems * 2);
        // gate rows 0..ane_h-1
        memcpy(fused_w, gate_data, ane_h * dim * 2);
        // up rows 0..ane_h-1
        memcpy(fused_w + ane_h * dim, up_data, ane_h * dim * 2);

        la->w1w3 = compile_ane_real(dim, 2 * ane_h, 1, fused_w, la->ioIn, la->ioW1W3Out);
        free(fused_w);

        // Build W2 weights for ANE: [dim, ane_h] fp16
        // = down_w[:, 0:ane_h] — need to extract columns
        uint16_t *w2_ane = (uint16_t *)malloc(dim * ane_h * 2);
        for (int r = 0; r < dim; r++) {
            memcpy(w2_ane + r * ane_h, down_data + r * c->hidden, ane_h * 2);
        }
        la->w2 = compile_ane_real(ane_h, dim, 1, w2_ane, la->ioActIn, la->ioW2Out);
        free(w2_ane);

        if (la->w1w3 && la->w2) ok += 2; else fail += 2;

        // GPU shard weights: gate_w[ane_h:, :], up_w[ane_h:, :], down_w[:, ane_h:]
        la->gpu_gate_w = [m->dev newBufferWithBytes:gate_data + ane_h * dim
                                             length:gpu_h * dim * 2
                                            options:MTLResourceStorageModeShared];
        la->gpu_up_w = [m->dev newBufferWithBytes:up_data + ane_h * dim
                                           length:gpu_h * dim * 2
                                          options:MTLResourceStorageModeShared];
        // down_w[:, ane_h:hidden] — extract columns
        uint16_t *down_gpu = (uint16_t *)malloc(dim * gpu_h * 2);
        for (int r = 0; r < dim; r++) {
            memcpy(down_gpu + r * gpu_h, down_data + r * c->hidden + ane_h, gpu_h * 2);
        }
        la->gpu_down_w = [m->dev newBufferWithBytes:down_gpu length:dim * gpu_h * 2
                                            options:MTLResourceStorageModeShared];
        free(down_gpu);

        la->mat_gpu_gate = mat16(la->gpu_gate_w, gpu_h, dim);
        la->mat_gpu_up = mat16(la->gpu_up_w, gpu_h, dim);
        la->mat_gpu_down = mat16(la->gpu_down_w, dim, gpu_h);

        if (l % 6 == 5 || l == c->n_layers - 1)
            fprintf(stderr, "  ANE layers 0-%d: compiled\n", l);
    }

    m->ane_available = (ok > 0);
    fprintf(stderr, "ANE: %d kernels ok, %d failed\n", ok, fail);
}

static void setup_engine(Model *m) {
    id<MTLDevice> dev = m->dev;
    QConfig *c = &m->cfg;
    int dim = c->dim, hidden = c->hidden, kv_dim = c->kv_dim, vocab = c->vocab;
    int gpu_h = c->gpu_hidden;

    // Activation buffers
    m->x = alloc_buf(dev, dim); m->xnorm = alloc_buf(dev, dim);
    m->q = alloc_buf(dev, dim); m->k = alloc_buf(dev, kv_dim);
    m->v = alloc_buf(dev, kv_dim); m->attn_out = alloc_buf(dev, dim);
    m->gate = alloc_buf(dev, gpu_h); m->up = alloc_buf(dev, gpu_h);
    m->act = alloc_buf(dev, gpu_h); m->down = alloc_buf(dev, dim);
    m->gate_full = alloc_buf(dev, hidden); m->up_full = alloc_buf(dev, hidden);
    m->act_full = alloc_buf(dev, hidden);
    m->logits = alloc_buf(dev, vocab);

    for (int l = 0; l < c->n_layers; l++) {
        m->k_cache[l] = alloc_buf(dev, kv_dim * MAX_SEQ);
        m->v_cache[l] = alloc_buf(dev, kv_dim * MAX_SEQ);
    }

    int half_hd = c->head_dim / 2;
    m->rope_cos = [dev newBufferWithLength:MAX_SEQ * half_hd * 4 options:MTLResourceStorageModeShared];
    m->rope_sin = [dev newBufferWithLength:MAX_SEQ * half_hd * 4 options:MTLResourceStorageModeShared];
    float *cos_d = (float *)m->rope_cos.contents, *sin_d = (float *)m->rope_sin.contents;
    for (int pos = 0; pos < MAX_SEQ; pos++) {
        for (int i = 0; i < half_hd; i++) {
            float freq = 1.0f / powf(c->rope_theta, 2.0f * i / c->head_dim);
            float angle = pos * freq;
            cos_d[pos * half_hd + i] = cosf(angle);
            sin_d[pos * half_hd + i] = sinf(angle);
        }
    }

    // MPS for decode
    m->mps_q = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:NO
        resultRows:dim resultColumns:1 interiorColumns:dim alpha:1.0 beta:0.0];
    m->mps_kv = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:NO
        resultRows:kv_dim resultColumns:1 interiorColumns:dim alpha:1.0 beta:0.0];
    m->mps_o = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:NO
        resultRows:dim resultColumns:1 interiorColumns:dim alpha:1.0 beta:0.0];
    // GPU shard MPS (for ANE split decode)
    m->mps_gpu_gate = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:NO
        resultRows:gpu_h resultColumns:1 interiorColumns:dim alpha:1.0 beta:0.0];
    m->mps_gpu_up = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:NO
        resultRows:gpu_h resultColumns:1 interiorColumns:dim alpha:1.0 beta:0.0];
    m->mps_gpu_down = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:NO
        resultRows:dim resultColumns:1 interiorColumns:gpu_h alpha:1.0 beta:0.0];
    // Full-size MPS (for GPU-only fallback)
    m->mps_full_gate = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:NO
        resultRows:hidden resultColumns:1 interiorColumns:dim alpha:1.0 beta:0.0];
    m->mps_full_up = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:NO
        resultRows:hidden resultColumns:1 interiorColumns:dim alpha:1.0 beta:0.0];
    m->mps_full_down = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:NO
        resultRows:dim resultColumns:1 interiorColumns:hidden alpha:1.0 beta:0.0];
    m->mps_lmhead = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:NO
        resultRows:vocab resultColumns:1 interiorColumns:dim alpha:1.0 beta:0.0];

    // Activation MPS matrices
    m->mat_xnorm = mat16(m->xnorm, dim, 1);
    m->mat_q = mat16(m->q, dim, 1); m->mat_k = mat16(m->k, kv_dim, 1);
    m->mat_v = mat16(m->v, kv_dim, 1); m->mat_attn_out = mat16(m->attn_out, dim, 1);
    m->mat_gate = mat16(m->gate, gpu_h, 1); m->mat_up = mat16(m->up, gpu_h, 1);
    m->mat_act = mat16(m->act, gpu_h, 1); m->mat_down = mat16(m->down, dim, 1);
    m->mat_gate_full = mat16(m->gate_full, hidden, 1); m->mat_up_full = mat16(m->up_full, hidden, 1);
    m->mat_act_full = mat16(m->act_full, hidden, 1);
    m->mat_logits = mat16(m->logits, vocab, 1);
    m->mat_embed = mat16(m->embed, vocab, dim);

    for (int l = 0; l < c->n_layers; l++) {
        LayerW *w = &m->lw[l]; LayerMat *lm = &m->lm[l];
        lm->q_w = mat16(w->q_w, dim, dim);
        lm->k_w = mat16(w->k_w, kv_dim, dim);
        lm->v_w = mat16(w->v_w, kv_dim, dim);
        lm->o_w = mat16(w->o_w, dim, dim);
        lm->gate_w = mat16(w->gate_w, hidden, dim);
        lm->up_w = mat16(w->up_w, hidden, dim);
        lm->down_w = mat16(w->down_w, dim, hidden);
    }
}

// ============================================================================
// 8. DECODE STEP (one token, all layers, with ANE split)
// ============================================================================

static void decode_step(Model *m, int token_id, int pos) {
    QConfig *c = &m->cfg;
    Shaders *sh = &m->sh;
    int dim = c->dim, kv_dim = c->kv_dim, n_heads = c->n_heads;
    int head_dim = c->head_dim, half_hd = head_dim / 2;
    int gpu_h = c->gpu_hidden;

    id<MTLCommandBuffer> cb = [m->queue commandBuffer];

    // ── Embedding ──
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sh->embed_lookup];
        [enc setBuffer:m->embed offset:0 atIndex:0];
        [enc setBuffer:m->x offset:0 atIndex:1];
        [enc setBytes:&token_id length:4 atIndex:2];
        [enc setBytes:&dim length:4 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(dim,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(dim,256),1,1)];
        [enc endEncoding];
    }

    for (int l = 0; l < c->n_layers; l++) {
        LayerW *w = &m->lw[l]; LayerMat *lm = &m->lm[l]; LayerANE *la = &m->la[l];

        // ── Attn RMSNorm ──
        { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:sh->rmsnorm];
          [enc setBuffer:m->x offset:0 atIndex:0]; [enc setBuffer:w->attn_norm offset:0 atIndex:1];
          [enc setBuffer:m->xnorm offset:0 atIndex:2]; [enc setBytes:&dim length:4 atIndex:3];
          [enc dispatchThreads:MTLSizeMake(256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
          [enc endEncoding]; }

        // ── Q, K, V projections + bias ──
        [m->mps_q encodeToCommandBuffer:cb leftMatrix:lm->q_w rightMatrix:m->mat_xnorm resultMatrix:m->mat_q];
        { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:sh->add_bias]; [enc setBuffer:m->q offset:0 atIndex:0];
          [enc setBuffer:w->q_b offset:0 atIndex:1];
          [enc dispatchThreads:MTLSizeMake(dim,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(dim,256),1,1)];
          [enc endEncoding]; }

        [m->mps_kv encodeToCommandBuffer:cb leftMatrix:lm->k_w rightMatrix:m->mat_xnorm resultMatrix:m->mat_k];
        { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:sh->add_bias]; [enc setBuffer:m->k offset:0 atIndex:0];
          [enc setBuffer:w->k_b offset:0 atIndex:1];
          [enc dispatchThreads:MTLSizeMake(kv_dim,1,1) threadsPerThreadgroup:MTLSizeMake(kv_dim,1,1)];
          [enc endEncoding]; }

        [m->mps_kv encodeToCommandBuffer:cb leftMatrix:lm->v_w rightMatrix:m->mat_xnorm resultMatrix:m->mat_v];
        { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:sh->add_bias]; [enc setBuffer:m->v offset:0 atIndex:0];
          [enc setBuffer:w->v_b offset:0 atIndex:1];
          [enc dispatchThreads:MTLSizeMake(kv_dim,1,1) threadsPerThreadgroup:MTLSizeMake(kv_dim,1,1)];
          [enc endEncoding]; }

        // ── RoPE Q, K ──
        { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:sh->rope]; [enc setBuffer:m->q offset:0 atIndex:0];
          [enc setBuffer:m->rope_cos offset:pos*half_hd*4 atIndex:1];
          [enc setBuffer:m->rope_sin offset:pos*half_hd*4 atIndex:2];
          [enc setBytes:&n_heads length:4 atIndex:3]; [enc setBytes:&head_dim length:4 atIndex:4];
          int t = n_heads * half_hd;
          [enc dispatchThreads:MTLSizeMake(t,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(t,256),1,1)];
          [enc endEncoding]; }
        { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:sh->rope]; [enc setBuffer:m->k offset:0 atIndex:0];
          [enc setBuffer:m->rope_cos offset:pos*half_hd*4 atIndex:1];
          [enc setBuffer:m->rope_sin offset:pos*half_hd*4 atIndex:2];
          int nkv = c->n_kv_heads;
          [enc setBytes:&nkv length:4 atIndex:3]; [enc setBytes:&head_dim length:4 atIndex:4];
          int t = c->n_kv_heads * half_hd;
          [enc dispatchThreads:MTLSizeMake(t,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(t,256),1,1)];
          [enc endEncoding]; }

        // ── KV cache write ──
        { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:sh->kv_write]; [enc setBuffer:m->k offset:0 atIndex:0];
          [enc setBuffer:m->k_cache[l] offset:0 atIndex:1];
          int ms = MAX_SEQ; [enc setBytes:&pos length:4 atIndex:2]; [enc setBytes:&ms length:4 atIndex:3];
          [enc dispatchThreads:MTLSizeMake(kv_dim,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(kv_dim,256),1,1)];
          [enc endEncoding]; }
        { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:sh->kv_write]; [enc setBuffer:m->v offset:0 atIndex:0];
          [enc setBuffer:m->v_cache[l] offset:0 atIndex:1];
          int ms = MAX_SEQ; [enc setBytes:&pos length:4 atIndex:2]; [enc setBytes:&ms length:4 atIndex:3];
          [enc dispatchThreads:MTLSizeMake(kv_dim,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(kv_dim,256),1,1)];
          [enc endEncoding]; }

        // ── Fused GQA attention ──
        { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:sh->fused_attn];
          [enc setBuffer:m->q offset:0 atIndex:0]; [enc setBuffer:m->k_cache[l] offset:0 atIndex:1];
          [enc setBuffer:m->v_cache[l] offset:0 atIndex:2]; [enc setBuffer:m->attn_out offset:0 atIndex:3];
          int cl = pos+1, ms = MAX_SEQ, nkv = c->n_kv_heads, hpg = c->heads_per_group;
          float sc = 1.0f / sqrtf((float)head_dim);
          [enc setBytes:&cl length:4 atIndex:4]; [enc setBytes:&ms length:4 atIndex:5];
          [enc setBytes:&head_dim length:4 atIndex:6]; [enc setBytes:&nkv length:4 atIndex:7];
          [enc setBytes:&hpg length:4 atIndex:8]; [enc setBytes:&sc length:4 atIndex:9];
          [enc dispatchThreads:MTLSizeMake(head_dim, n_heads, 1) threadsPerThreadgroup:MTLSizeMake(head_dim,1,1)];
          [enc endEncoding]; }

        // ── O projection + residual ──
        [m->mps_o encodeToCommandBuffer:cb leftMatrix:lm->o_w rightMatrix:m->mat_attn_out resultMatrix:m->mat_down];
        { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:sh->inplace_add]; [enc setBuffer:m->x offset:0 atIndex:0];
          [enc setBuffer:m->down offset:0 atIndex:1];
          [enc dispatchThreads:MTLSizeMake(dim,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(dim,256),1,1)];
          [enc endEncoding]; }

        // ── FFN RMSNorm ──
        { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:sh->rmsnorm]; [enc setBuffer:m->x offset:0 atIndex:0];
          [enc setBuffer:w->ffn_norm offset:0 atIndex:1]; [enc setBuffer:m->xnorm offset:0 atIndex:2];
          [enc setBytes:&dim length:4 atIndex:3];
          [enc dispatchThreads:MTLSizeMake(256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
          [enc endEncoding]; }

        // ── FFN with ANE split ──
        if (m->ane_available && la->w1w3 && la->w2) {
            // Commit current GPU work, wait for xnorm to be ready
            [cb commit]; [cb waitUntilCompleted];

            // Copy xnorm (fp16) → ANE input IOSurface (fp32)
            {
                id<MTLCommandBuffer> cb2 = [m->queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb2 computeCommandEncoder];
                [enc setComputePipelineState:sh->fp16_to_fp32];
                [enc setBuffer:m->xnorm offset:0 atIndex:0]; [enc setBuffer:la->bufIn offset:0 atIndex:1];
                [enc dispatchThreads:MTLSizeMake(dim,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(dim,256),1,1)];
                [enc endEncoding]; [cb2 commit]; [cb2 waitUntilCompleted];
            }

            // ── Phase 1: Concurrent W1W3 ──
            dispatch_semaphore_t sem = dispatch_semaphore_create(0);
            dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^{
                ane_eval(la->w1w3);
                dispatch_semaphore_signal(sem);
            });

            // GPU shard: gate, up [gpu_h, dim] @ xnorm → [gpu_h]
            {
                id<MTLCommandBuffer> cb2 = [m->queue commandBuffer];
                [m->mps_gpu_gate encodeToCommandBuffer:cb2 leftMatrix:la->mat_gpu_gate
                    rightMatrix:m->mat_xnorm resultMatrix:m->mat_gate];
                [m->mps_gpu_up encodeToCommandBuffer:cb2 leftMatrix:la->mat_gpu_up
                    rightMatrix:m->mat_xnorm resultMatrix:m->mat_up];
                [cb2 commit]; [cb2 waitUntilCompleted];
            }

            dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

            // ── Phase 2: SiLU×Mul on both shards ──
            {
                id<MTLCommandBuffer> cb2 = [m->queue commandBuffer];
                // GPU shard: silu_mul fp16
                { id<MTLComputeCommandEncoder> enc = [cb2 computeCommandEncoder];
                  [enc setComputePipelineState:sh->silu_mul]; [enc setBuffer:m->gate offset:0 atIndex:0];
                  [enc setBuffer:m->up offset:0 atIndex:1]; [enc setBuffer:m->act offset:0 atIndex:2];
                  [enc dispatchThreads:MTLSizeMake(gpu_h,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(gpu_h,256),1,1)];
                  [enc endEncoding]; }
                // ANE shard: silu_mul fp32 (data in IOSurface)
                { id<MTLComputeCommandEncoder> enc = [cb2 computeCommandEncoder];
                  int ane_h = c->ane_hidden;
                  [enc setComputePipelineState:sh->silu_mul_f32];
                  [enc setBuffer:la->bufW1W3Out offset:0 atIndex:0];  // gate part
                  [enc setBuffer:la->bufW1W3Out offset:ane_h*4 atIndex:1]; // up part
                  [enc setBuffer:la->bufActIn offset:0 atIndex:2];
                  [enc dispatchThreads:MTLSizeMake(ane_h,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(ane_h,256),1,1)];
                  [enc endEncoding]; }
                [cb2 commit]; [cb2 waitUntilCompleted];
            }

            // ── Phase 3: Concurrent W2 ──
            dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^{
                ane_eval(la->w2);
                dispatch_semaphore_signal(sem);
            });
            {
                id<MTLCommandBuffer> cb2 = [m->queue commandBuffer];
                [m->mps_gpu_down encodeToCommandBuffer:cb2 leftMatrix:la->mat_gpu_down
                    rightMatrix:m->mat_act resultMatrix:m->mat_down];
                [cb2 commit]; [cb2 waitUntilCompleted];
            }
            dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

            // ── Phase 4: Add partials + residual ──
            cb = [m->queue commandBuffer];
            { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
              [enc setComputePipelineState:sh->add_f32_to_f16];
              [enc setBuffer:m->down offset:0 atIndex:0]; [enc setBuffer:la->bufW2Out offset:0 atIndex:1];
              [enc dispatchThreads:MTLSizeMake(dim,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(dim,256),1,1)];
              [enc endEncoding]; }
            { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
              [enc setComputePipelineState:sh->inplace_add]; [enc setBuffer:m->x offset:0 atIndex:0];
              [enc setBuffer:m->down offset:0 atIndex:1];
              [enc dispatchThreads:MTLSizeMake(dim,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(dim,256),1,1)];
              [enc endEncoding]; }
        } else {
            // GPU-only FFN fallback (uses full hidden-dim buffers)
            int hidden = c->hidden;
            [m->mps_full_gate encodeToCommandBuffer:cb leftMatrix:lm->gate_w rightMatrix:m->mat_xnorm resultMatrix:m->mat_gate_full];
            [m->mps_full_up encodeToCommandBuffer:cb leftMatrix:lm->up_w rightMatrix:m->mat_xnorm resultMatrix:m->mat_up_full];
            { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
              [enc setComputePipelineState:sh->silu_mul]; [enc setBuffer:m->gate_full offset:0 atIndex:0];
              [enc setBuffer:m->up_full offset:0 atIndex:1]; [enc setBuffer:m->act_full offset:0 atIndex:2];
              [enc dispatchThreads:MTLSizeMake(hidden,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(hidden,256),1,1)];
              [enc endEncoding]; }
            [m->mps_full_down encodeToCommandBuffer:cb leftMatrix:lm->down_w rightMatrix:m->mat_act_full resultMatrix:m->mat_down];
            { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
              [enc setComputePipelineState:sh->inplace_add]; [enc setBuffer:m->x offset:0 atIndex:0];
              [enc setBuffer:m->down offset:0 atIndex:1];
              [enc dispatchThreads:MTLSizeMake(dim,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(dim,256),1,1)];
              [enc endEncoding]; }
        }
    }

    // ── Final RMSNorm + LM Head ──
    { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:sh->rmsnorm]; [enc setBuffer:m->x offset:0 atIndex:0];
      [enc setBuffer:m->final_norm offset:0 atIndex:1]; [enc setBuffer:m->xnorm offset:0 atIndex:2];
      [enc setBytes:&dim length:4 atIndex:3];
      [enc dispatchThreads:MTLSizeMake(256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
      [enc endEncoding]; }
    [m->mps_lmhead encodeToCommandBuffer:cb leftMatrix:m->mat_embed
                         rightMatrix:m->mat_xnorm resultMatrix:m->mat_logits];

    [cb commit]; [cb waitUntilCompleted];
}

// ============================================================================
// 9. BATCHED PREFILL
// ============================================================================

static void prefill_batch(Model *m, const int *tokens, int seq_len) {
    QConfig *c = &m->cfg;
    Shaders *sh = &m->sh;
    id<MTLDevice> dev = m->dev;
    int dim = c->dim, kv_dim = c->kv_dim, hidden = c->hidden;
    int n_heads = c->n_heads, head_dim = c->head_dim, half_hd = head_dim / 2;

    // Allocate batched buffers
    id<MTLBuffer> bx      = alloc_buf(dev, dim * seq_len);
    id<MTLBuffer> bxnorm  = alloc_buf(dev, dim * seq_len);
    id<MTLBuffer> bq      = alloc_buf(dev, dim * seq_len);
    id<MTLBuffer> bk      = alloc_buf(dev, kv_dim * seq_len);
    id<MTLBuffer> bv      = alloc_buf(dev, kv_dim * seq_len);
    id<MTLBuffer> battn   = alloc_buf(dev, dim * seq_len);
    id<MTLBuffer> bgate   = alloc_buf(dev, hidden * seq_len);
    id<MTLBuffer> bup     = alloc_buf(dev, hidden * seq_len);
    id<MTLBuffer> bact    = alloc_buf(dev, hidden * seq_len);
    id<MTLBuffer> bdown   = alloc_buf(dev, dim * seq_len);
    id<MTLBuffer> btokens = [dev newBufferWithBytes:tokens length:seq_len * 4
                                           options:MTLResourceStorageModeShared];

    // MPS for batched operations
    MPSMatrixMultiplication *bm_q = [[MPSMatrixMultiplication alloc] initWithDevice:dev
        transposeLeft:NO transposeRight:NO resultRows:dim resultColumns:seq_len interiorColumns:dim alpha:1 beta:0];
    MPSMatrixMultiplication *bm_kv = [[MPSMatrixMultiplication alloc] initWithDevice:dev
        transposeLeft:NO transposeRight:NO resultRows:kv_dim resultColumns:seq_len interiorColumns:dim alpha:1 beta:0];
    MPSMatrixMultiplication *bm_o = [[MPSMatrixMultiplication alloc] initWithDevice:dev
        transposeLeft:NO transposeRight:NO resultRows:dim resultColumns:seq_len interiorColumns:dim alpha:1 beta:0];
    MPSMatrixMultiplication *bm_gate = [[MPSMatrixMultiplication alloc] initWithDevice:dev
        transposeLeft:NO transposeRight:NO resultRows:hidden resultColumns:seq_len interiorColumns:dim alpha:1 beta:0];
    MPSMatrixMultiplication *bm_up = [[MPSMatrixMultiplication alloc] initWithDevice:dev
        transposeLeft:NO transposeRight:NO resultRows:hidden resultColumns:seq_len interiorColumns:dim alpha:1 beta:0];
    MPSMatrixMultiplication *bm_down = [[MPSMatrixMultiplication alloc] initWithDevice:dev
        transposeLeft:NO transposeRight:NO resultRows:dim resultColumns:seq_len interiorColumns:hidden alpha:1 beta:0];

    MPSMatrix *mx = mat16(bx, dim, seq_len), *mxn = mat16(bxnorm, dim, seq_len);
    MPSMatrix *mq = mat16(bq, dim, seq_len), *mk = mat16(bk, kv_dim, seq_len);
    MPSMatrix *mv = mat16(bv, kv_dim, seq_len), *ma = mat16(battn, dim, seq_len);
    MPSMatrix *mg = mat16(bgate, hidden, seq_len), *mu = mat16(bup, hidden, seq_len);
    MPSMatrix *mac = mat16(bact, hidden, seq_len), *md = mat16(bdown, dim, seq_len);

    id<MTLCommandBuffer> cb = [m->queue commandBuffer];

    // ── Embed all tokens ──
    { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:sh->embed_batch]; [enc setBuffer:m->embed offset:0 atIndex:0];
      [enc setBuffer:btokens offset:0 atIndex:1]; [enc setBuffer:bx offset:0 atIndex:2];
      [enc setBytes:&dim length:4 atIndex:3]; [enc setBytes:&seq_len length:4 atIndex:4];
      [enc dispatchThreads:MTLSizeMake(dim, seq_len, 1) threadsPerThreadgroup:MTLSizeMake(MIN(dim,256),1,1)];
      [enc endEncoding]; }
    [cb commit]; [cb waitUntilCompleted];

    // ── Layer loop (GPU-only for prefill) ──
    for (int l = 0; l < c->n_layers; l++) {
        LayerW *w = &m->lw[l]; LayerMat *lm = &m->lm[l];
        cb = [m->queue commandBuffer];

        // Batched RMSNorm
        { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:sh->rmsnorm_batch]; [enc setBuffer:bx offset:0 atIndex:0];
          [enc setBuffer:w->attn_norm offset:0 atIndex:1]; [enc setBuffer:bxnorm offset:0 atIndex:2];
          [enc setBytes:&dim length:4 atIndex:3]; [enc setBytes:&seq_len length:4 atIndex:4];
          [enc dispatchThreads:MTLSizeMake(1, seq_len, 1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
          [enc endEncoding]; }

        // Q, K, V projections
        [bm_q encodeToCommandBuffer:cb leftMatrix:lm->q_w rightMatrix:mxn resultMatrix:mq];
        [bm_kv encodeToCommandBuffer:cb leftMatrix:lm->k_w rightMatrix:mxn resultMatrix:mk];
        [bm_kv encodeToCommandBuffer:cb leftMatrix:lm->v_w rightMatrix:mxn resultMatrix:mv];

        // Batched bias
        { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:sh->add_bias_batch]; [enc setBuffer:bq offset:0 atIndex:0];
          [enc setBuffer:w->q_b offset:0 atIndex:1];
          [enc setBytes:&dim length:4 atIndex:2]; [enc setBytes:&seq_len length:4 atIndex:3];
          [enc dispatchThreads:MTLSizeMake(dim, seq_len, 1) threadsPerThreadgroup:MTLSizeMake(MIN(dim,16), MIN(seq_len,16), 1)];
          [enc endEncoding]; }
        { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:sh->add_bias_batch]; [enc setBuffer:bk offset:0 atIndex:0];
          [enc setBuffer:w->k_b offset:0 atIndex:1];
          [enc setBytes:&kv_dim length:4 atIndex:2]; [enc setBytes:&seq_len length:4 atIndex:3];
          [enc dispatchThreads:MTLSizeMake(kv_dim, seq_len, 1) threadsPerThreadgroup:MTLSizeMake(kv_dim, 1, 1)];
          [enc endEncoding]; }
        { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:sh->add_bias_batch]; [enc setBuffer:bv offset:0 atIndex:0];
          [enc setBuffer:w->v_b offset:0 atIndex:1];
          [enc setBytes:&kv_dim length:4 atIndex:2]; [enc setBytes:&seq_len length:4 atIndex:3];
          [enc dispatchThreads:MTLSizeMake(kv_dim, seq_len, 1) threadsPerThreadgroup:MTLSizeMake(kv_dim, 1, 1)];
          [enc endEncoding]; }

        // Batched RoPE
        int pos_offset = 0;
        { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:sh->rope_batch]; [enc setBuffer:bq offset:0 atIndex:0];
          [enc setBuffer:m->rope_cos offset:0 atIndex:1]; [enc setBuffer:m->rope_sin offset:0 atIndex:2];
          [enc setBytes:&n_heads length:4 atIndex:3]; [enc setBytes:&head_dim length:4 atIndex:4];
          [enc setBytes:&seq_len length:4 atIndex:5]; [enc setBytes:&pos_offset length:4 atIndex:6];
          int t = n_heads * half_hd;
          [enc dispatchThreads:MTLSizeMake(t, seq_len, 1) threadsPerThreadgroup:MTLSizeMake(MIN(t,256),1,1)];
          [enc endEncoding]; }
        { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          int nkv = c->n_kv_heads;
          [enc setComputePipelineState:sh->rope_batch]; [enc setBuffer:bk offset:0 atIndex:0];
          [enc setBuffer:m->rope_cos offset:0 atIndex:1]; [enc setBuffer:m->rope_sin offset:0 atIndex:2];
          [enc setBytes:&nkv length:4 atIndex:3]; [enc setBytes:&head_dim length:4 atIndex:4];
          [enc setBytes:&seq_len length:4 atIndex:5]; [enc setBytes:&pos_offset length:4 atIndex:6];
          int t = c->n_kv_heads * half_hd;
          [enc dispatchThreads:MTLSizeMake(t, seq_len, 1) threadsPerThreadgroup:MTLSizeMake(MIN(t,256),1,1)];
          [enc endEncoding]; }

        // Write K, V to cache
        { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          int ms = MAX_SEQ, off = 0;
          [enc setComputePipelineState:sh->kv_write_batch]; [enc setBuffer:bk offset:0 atIndex:0];
          [enc setBuffer:m->k_cache[l] offset:0 atIndex:1]; [enc setBytes:&ms length:4 atIndex:2];
          [enc setBytes:&seq_len length:4 atIndex:3]; [enc setBytes:&off length:4 atIndex:4];
          [enc dispatchThreads:MTLSizeMake(kv_dim, seq_len, 1) threadsPerThreadgroup:MTLSizeMake(MIN(kv_dim,256),1,1)];
          [enc endEncoding]; }
        { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          int ms = MAX_SEQ, off = 0;
          [enc setComputePipelineState:sh->kv_write_batch]; [enc setBuffer:bv offset:0 atIndex:0];
          [enc setBuffer:m->v_cache[l] offset:0 atIndex:1]; [enc setBytes:&ms length:4 atIndex:2];
          [enc setBytes:&seq_len length:4 atIndex:3]; [enc setBytes:&off length:4 atIndex:4];
          [enc dispatchThreads:MTLSizeMake(kv_dim, seq_len, 1) threadsPerThreadgroup:MTLSizeMake(MIN(kv_dim,256),1,1)];
          [enc endEncoding]; }

        // Prefill attention (causal GQA)
        { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          int nkv = c->n_kv_heads, hpg = c->heads_per_group;
          float sc = 1.0f / sqrtf((float)head_dim);
          [enc setComputePipelineState:sh->prefill_attn]; [enc setBuffer:bq offset:0 atIndex:0];
          [enc setBuffer:bk offset:0 atIndex:1]; [enc setBuffer:bv offset:0 atIndex:2];
          [enc setBuffer:battn offset:0 atIndex:3];
          [enc setBytes:&seq_len length:4 atIndex:4]; [enc setBytes:&head_dim length:4 atIndex:5];
          [enc setBytes:&nkv length:4 atIndex:6]; [enc setBytes:&hpg length:4 atIndex:7];
          [enc setBytes:&sc length:4 atIndex:8];
          [enc dispatchThreads:MTLSizeMake(head_dim, seq_len, n_heads)
               threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
          [enc endEncoding]; }

        // O projection + residual
        [bm_o encodeToCommandBuffer:cb leftMatrix:lm->o_w rightMatrix:ma resultMatrix:md];
        { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          int total = dim * seq_len;
          [enc setComputePipelineState:sh->inplace_add_batch]; [enc setBuffer:bx offset:0 atIndex:0];
          [enc setBuffer:bdown offset:0 atIndex:1];
          [enc dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(total,256),1,1)];
          [enc endEncoding]; }

        // FFN RMSNorm
        { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:sh->rmsnorm_batch]; [enc setBuffer:bx offset:0 atIndex:0];
          [enc setBuffer:w->ffn_norm offset:0 atIndex:1]; [enc setBuffer:bxnorm offset:0 atIndex:2];
          [enc setBytes:&dim length:4 atIndex:3]; [enc setBytes:&seq_len length:4 atIndex:4];
          [enc dispatchThreads:MTLSizeMake(1, seq_len, 1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
          [enc endEncoding]; }

        // FFN (GPU-only for prefill — ANE split not worth it for batched)
        [bm_gate encodeToCommandBuffer:cb leftMatrix:lm->gate_w rightMatrix:mxn resultMatrix:mg];
        [bm_up encodeToCommandBuffer:cb leftMatrix:lm->up_w rightMatrix:mxn resultMatrix:mu];
        { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          int total = hidden * seq_len;
          [enc setComputePipelineState:sh->silu_mul_batch]; [enc setBuffer:bgate offset:0 atIndex:0];
          [enc setBuffer:bup offset:0 atIndex:1]; [enc setBuffer:bact offset:0 atIndex:2];
          [enc dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(total,256),1,1)];
          [enc endEncoding]; }
        [bm_down encodeToCommandBuffer:cb leftMatrix:lm->down_w rightMatrix:mac resultMatrix:md];
        { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          int total = dim * seq_len;
          [enc setComputePipelineState:sh->inplace_add_batch]; [enc setBuffer:bx offset:0 atIndex:0];
          [enc setBuffer:bdown offset:0 atIndex:1];
          [enc dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(total,256),1,1)];
          [enc endEncoding]; }

        [cb commit]; [cb waitUntilCompleted];
    }

    // Copy last token's hidden state to m->x for decode
    {
        id<MTLCommandBuffer> cb2 = [m->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb2 computeCommandEncoder];
        // Copy column (seq_len-1) from bx [dim, seq_len] to m->x [dim]
        [enc setComputePipelineState:sh->embed_lookup]; // reuse: copies from table[id*dim] to out
        // Actually, we need a custom copy. Let's just do it on CPU.
        [enc endEncoding]; [cb2 commit]; [cb2 waitUntilCompleted];
    }
    // CPU copy: last column of bx to m->x
    const uint16_t *src = (const uint16_t *)bx.contents;
    uint16_t *dst = (uint16_t *)m->x.contents;
    for (int i = 0; i < dim; i++) {
        dst[i] = src[i * seq_len + (seq_len - 1)];
    }
}

// ============================================================================
// 10. ARGMAX
// ============================================================================

static int argmax_fp16(const uint16_t *data, int n) {
    int best = 0; float best_val = -1e30f;
    for (int i = 0; i < n; i++) {
        uint16_t h = data[i];
        uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
        float val;
        if (exp == 0) val = (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * (1.0f / 16384.0f);
        else if (exp == 31) val = sign ? -1e30f : 1e30f;
        else { uint32_t fp32 = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13); memcpy(&val, &fp32, 4); }
        if (val > best_val) { best_val = val; best = i; }
    }
    return best;
}

// ============================================================================
// 11. MAIN — with streaming + benchmarking
// ============================================================================

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        setbuf(stdout, NULL); setbuf(stderr, NULL);

        if (argc < 5) {
            fprintf(stderr, "Usage: %s <weights.bin> <tokens_csv> <max_tokens> <eos_id>\n", argv[0]);
            return 1;
        }

        const char *weights_path = argv[1], *token_csv = argv[2];
        int max_tokens = atoi(argv[3]), eos_id = atoi(argv[4]);

        int input_ids[4096]; int n_input = 0;
        { char *csv = strdup(token_csv); char *tok = strtok(csv, ",");
          while (tok && n_input < 4096) { input_ids[n_input++] = atoi(tok); tok = strtok(NULL, ","); }
          free(csv); }
        fprintf(stderr, "Input: %d tokens, max_gen=%d, eos=%d\n", n_input, max_tokens, eos_id);

        Model m; memset((void*)&m, 0, sizeof(m));
        m.dev = MTLCreateSystemDefaultDevice();
        m.queue = [m.dev newCommandQueue];
        fprintf(stderr, "GPU: %s\n", [[m.dev name] UTF8String]);

        m.sh = setup_shaders(m.dev);
        fprintf(stderr, "Shaders: %d compiled ✓\n", 21);

        uint64_t t0 = mach_absolute_time();
        load_model(&m, weights_path);
        fprintf(stderr, "Weights: %.0f ms\n", ticks_us(mach_absolute_time()-t0)/1000);

        t0 = mach_absolute_time();
        setup_engine(&m);
        fprintf(stderr, "Engine: %.0f ms\n", ticks_us(mach_absolute_time()-t0)/1000);

        t0 = mach_absolute_time();
        setup_ane_split(&m);
        fprintf(stderr, "ANE setup: %.0f ms\n", ticks_us(mach_absolute_time()-t0)/1000);

        // ── Batched Prefill ──
        fprintf(stderr, "Prefilling %d tokens (batched)...\n", n_input);
        t0 = mach_absolute_time();
        prefill_batch(&m, input_ids, n_input);
        double prefill_ms = ticks_us(mach_absolute_time()-t0)/1000;
        fprintf(stderr, "Prefill: %.1f ms total (%.1f ms/tok)\n", prefill_ms, prefill_ms/n_input);

        // Run final norm + lm_head on prefilled state to get first logits
        {
            int dim = m.cfg.dim;
            id<MTLCommandBuffer> cb = [m.queue commandBuffer];
            { id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
              [enc setComputePipelineState:m.sh.rmsnorm]; [enc setBuffer:m.x offset:0 atIndex:0];
              [enc setBuffer:m.final_norm offset:0 atIndex:1]; [enc setBuffer:m.xnorm offset:0 atIndex:2];
              [enc setBytes:&dim length:4 atIndex:3];
              [enc dispatchThreads:MTLSizeMake(256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
              [enc endEncoding]; }
            [m.mps_lmhead encodeToCommandBuffer:cb leftMatrix:m.mat_embed
                               rightMatrix:m.mat_xnorm resultMatrix:m.mat_logits];
            [cb commit]; [cb waitUntilCompleted];
        }

        // ── Streaming Decode ──
        fprintf(stderr, "Generating (streaming)...\n");
        int output_ids[4096]; int n_output = 0;
        int pos = n_input;

        t0 = mach_absolute_time();
        for (int i = 0; i < max_tokens; i++) {
            int next_token = argmax_fp16((const uint16_t *)m.logits.contents, m.cfg.vocab);
            output_ids[n_output++] = next_token;

            // Stream: print token ID immediately
            if (i > 0) printf(",");
            printf("%d", next_token);
            fflush(stdout);

            if (next_token == eos_id || pos >= MAX_SEQ - 1) break;
            decode_step(&m, next_token, pos);
            pos++;
        }
        printf("\n");
        double decode_ms = ticks_us(mach_absolute_time()-t0)/1000;
        double tps = n_output / (decode_ms / 1000);

        fprintf(stderr, "Decode: %d tokens in %.1f ms (%.1f tok/s)\n", n_output, decode_ms, tps);
        return 0;
    }
}
