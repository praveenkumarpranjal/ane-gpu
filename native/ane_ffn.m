// ane_ffn.m — libanegpu.dylib : C API to run fused SwiGLU FFN (and plain matmul)
// sub-blocks on the Apple Neural Engine via private AppleNeuralEngine.framework.
// Designed to be called from Python (ctypes) and overlapped with MLX GPU work.
//
// All matmuls are expressed as 1x1 convs over a channel-first [1, C, 1, seq] fp16
// tensor. seq MUST be a multiple of 16 (ANE tiling constraint) or the conv returns
// zeros. I/O is fp16, through IOSurface-backed buffers (CPU-accessible, page-aligned).
//
// Build: see Makefile (xcrun clang -dynamiclib ...).

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static Class g_ANEDesc, g_ANEInMem, g_ANEReq, g_ANEIO;
static int g_ready = 0;

typedef struct {
    id model, request;
    NSString *tmpDir;
    IOSurfaceRef ioIn, ioOut;
    void *inPtr, *outPtr;
    size_t inBytes, outBytes;
} ANEHandle;

int ane_init(void) {
    if (g_ready) return 0;
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_ANEDesc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
    g_ANEReq   = NSClassFromString(@"_ANERequest");
    g_ANEIO    = NSClassFromString(@"_ANEIOSurfaceObject");
    g_ready = (g_ANEDesc && g_ANEInMem && g_ANEReq && g_ANEIO) ? 1 : 0;
    return g_ready ? 0 : -1;
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

// weight blob: 128-byte header, fp16 data at byte 128, MIL BLOBFILE offset=64
static uint8_t *blob(const uint16_t *w, int out_ch, int in_ch, size_t *len) {
    int ws = out_ch * in_ch * 2, total = 128 + ws;
    uint8_t *b = calloc(total, 1);
    b[0]=0x01; b[4]=0x02; b[64]=0xEF; b[65]=0xBE; b[66]=0xAD; b[67]=0xDE; b[68]=0x01;
    *(uint32_t*)(b+72)=ws; *(uint32_t*)(b+80)=128;
    memcpy(b+128, w, ws); *len=total; return b;
}

// plain matmul: y[1,out,1,S] = conv(W[out,in,1,1], x[1,in,1,S]), all fp16
static NSString *mil_matmul(int in_ch, int out_ch, int S) {
    return [NSString stringWithFormat:
      @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
      "{\n  func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
      "    string vpt = const()[name=string(\"vpt\"), val=string(\"valid\")];\n"
      "    tensor<int32,[2]> st = const()[name=string(\"st\"), val=tensor<int32,[2]>([1,1])];\n"
      "    tensor<int32,[4]> pd = const()[name=string(\"pd\"), val=tensor<int32,[4]>([0,0,0,0])];\n"
      "    tensor<int32,[2]> dl = const()[name=string(\"dl\"), val=tensor<int32,[2]>([1,1])];\n"
      "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
      "    tensor<fp16,[%d,%d,1,1]> W = const()[name=string(\"W\"), val=tensor<fp16,[%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];\n"
      "    tensor<fp16,[1,%d,1,%d]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=vpt, strides=st, weight=W, x=x)[name=string(\"y\")];\n"
      "  } -> (y);\n}\n",
      in_ch,S, out_ch,in_ch,out_ch,in_ch, out_ch,S];
}

// activation+combine block: hh = act(gate) * up.
//   gelu=1 -> tanh-approx GELU (GeGLU, e.g. Gemma);  gelu=0 -> SiLU (SwiGLU, e.g. Qwen/Llama)
static NSString *act_block(int h, int S, int gelu) {
    if (gelu)
        return [NSString stringWithFormat:
          @"    tensor<fp16,[1,%d,1,%d]> gact = gelu(mode=string(\"TANH_APPROXIMATION\"), x=gate)[name=string(\"gact\")];\n"
          "    tensor<fp16,[1,%d,1,%d]> hh = mul(x=gact, y=up)[name=string(\"hh\")];\n",
          h,S, h,S];
    return [NSString stringWithFormat:
      @"    tensor<fp16,[1,%d,1,%d]> sg = sigmoid(x=gate)[name=string(\"sg\")];\n"
      "    tensor<fp16,[1,%d,1,%d]> silu = mul(x=gate, y=sg)[name=string(\"silu\")];\n"
      "    tensor<fp16,[1,%d,1,%d]> hh = mul(x=silu, y=up)[name=string(\"hh\")];\n",
      h,S, h,S, h,S];
}

// fused gated FFN: y = (act(x@W1) * (x@W3)) @ W2 ; act = SiLU (gelu=0) or GELU (gelu=1)
static NSString *mil_ffn(int dim, int h, int S, int gelu) {
    return [NSString stringWithFormat:
      @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
      "{\n  func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
      "    string vpt = const()[name=string(\"vpt\"), val=string(\"valid\")];\n"
      "    tensor<int32,[2]> st = const()[name=string(\"st\"), val=tensor<int32,[2]>([1,1])];\n"
      "    tensor<int32,[4]> pd = const()[name=string(\"pd\"), val=tensor<int32,[4]>([0,0,0,0])];\n"
      "    tensor<int32,[2]> dl = const()[name=string(\"dl\"), val=tensor<int32,[2]>([1,1])];\n"
      "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
      "    tensor<fp16,[%d,%d,1,1]> W1 = const()[name=string(\"W1\"), val=tensor<fp16,[%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64)))];\n"
      "    tensor<fp16,[%d,%d,1,1]> W3 = const()[name=string(\"W3\"), val=tensor<fp16,[%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3.bin\"), offset=uint64(64)))];\n"
      "    tensor<fp16,[%d,%d,1,1]> W2 = const()[name=string(\"W2\"), val=tensor<fp16,[%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64)))];\n"
      "    tensor<fp16,[1,%d,1,%d]> gate = conv(dilations=dl, groups=gr, pad=pd, pad_type=vpt, strides=st, weight=W1, x=x)[name=string(\"gate\")];\n"
      "    tensor<fp16,[1,%d,1,%d]> up = conv(dilations=dl, groups=gr, pad=pd, pad_type=vpt, strides=st, weight=W3, x=x)[name=string(\"up\")];\n"
      "%@"
      "    tensor<fp16,[1,%d,1,%d]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=vpt, strides=st, weight=W2, x=hh)[name=string(\"y\")];\n"
      "  } -> (y);\n}\n",
      dim,S, h,dim,h,dim, h,dim,h,dim, dim,h,dim,h,
      h,S, h,S, act_block(h,S,gelu), dim,S];
}

// --- INT8 weight path: int8 weights dequantized in-engine (constexpr_affine_dequantize).
// Streams ~half the weight bytes per eval (the FFN bottleneck) at fp16 I/O. ---

// quantize fp16 weights -> int8 PER OUTPUT CHANNEL (each row = one out_ch gets its own
// symmetric scale = max|row|/127). Writes int8 to `out`, fp16 scales [out_ch] to `scales`.
// Per-channel (vs per-tensor) cuts the error ~6x and prevents fp16 overflow->NaN on deep
// models (e.g. Qwen-1.5B, 28 layers) where accumulated per-tensor error tips past 65504.
static void quantize_q8(const uint16_t *w_bits, int out_ch, int in_ch, int8_t *out, uint16_t *scales) {
    const _Float16 *w = (const _Float16 *)w_bits;
    _Float16 *sc = (_Float16 *)scales;
    for (int c = 0; c < out_ch; c++) {
        const _Float16 *row = w + (size_t)c * in_ch;
        int8_t *orow = out + (size_t)c * in_ch;
        float maxabs = 0.0f;
        for (int i = 0; i < in_ch; i++) { float a = fabsf((float)row[i]); if (a > maxabs) maxabs = a; }
        float scale = maxabs / 127.0f; if (scale == 0.0f) scale = 1.0f;
        for (int i = 0; i < in_ch; i++) {
            int q = (int)lroundf((float)row[i] / scale);
            orow[i] = q < -128 ? -128 : (q > 127 ? 127 : q);
        }
        sc[c] = (_Float16)scale;
    }
}

// int8 weight blob: same proven header, 1 byte/elem
static uint8_t *blob_q8(const int8_t *q, int out_ch, int in_ch, size_t *len) {
    int ws = out_ch * in_ch, total = 128 + ws;
    uint8_t *b = calloc(total, 1);
    b[0]=0x01; b[4]=0x02; b[64]=0xEF; b[65]=0xBE; b[66]=0xAD; b[67]=0xDE; b[68]=0x01;
    *(uint32_t*)(b+72)=ws; *(uint32_t*)(b+80)=128;
    memcpy(b+128, q, ws); *len=total; return b;
}

// fused gated FFN with int8 weights, PER-CHANNEL scale vectors (BLOBFILE s1/s3/s2.bin); act per gelu flag
static NSString *mil_ffn_q8(int dim, int h, int S, int gelu) {
    return [NSString stringWithFormat:
      @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
      "{\n  func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
      "    string vpt = const()[name=string(\"vpt\"), val=string(\"valid\")];\n"
      "    tensor<int32,[2]> st = const()[name=string(\"st\"), val=tensor<int32,[2]>([1,1])];\n"
      "    tensor<int32,[4]> pd = const()[name=string(\"pd\"), val=tensor<int32,[4]>([0,0,0,0])];\n"
      "    tensor<int32,[2]> dl = const()[name=string(\"dl\"), val=tensor<int32,[2]>([1,1])];\n"
      "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
      "    tensor<fp16,[%d,%d,1,1]> W1 = constexpr_affine_dequantize()[axis=int32(0), name=string(\"W1\"), quantized_data=tensor<int8,[%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64))), scale=tensor<fp16,[%d]>(BLOBFILE(path=string(\"@model_path/weights/s1.bin\"), offset=uint64(64))), zero_point=int8(0)];\n"
      "    tensor<fp16,[%d,%d,1,1]> W3 = constexpr_affine_dequantize()[axis=int32(0), name=string(\"W3\"), quantized_data=tensor<int8,[%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3.bin\"), offset=uint64(64))), scale=tensor<fp16,[%d]>(BLOBFILE(path=string(\"@model_path/weights/s3.bin\"), offset=uint64(64))), zero_point=int8(0)];\n"
      "    tensor<fp16,[%d,%d,1,1]> W2 = constexpr_affine_dequantize()[axis=int32(0), name=string(\"W2\"), quantized_data=tensor<int8,[%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64))), scale=tensor<fp16,[%d]>(BLOBFILE(path=string(\"@model_path/weights/s2.bin\"), offset=uint64(64))), zero_point=int8(0)];\n"
      "    tensor<fp16,[1,%d,1,%d]> gate = conv(dilations=dl, groups=gr, pad=pd, pad_type=vpt, strides=st, weight=W1, x=x)[name=string(\"gate\")];\n"
      "    tensor<fp16,[1,%d,1,%d]> up = conv(dilations=dl, groups=gr, pad=pd, pad_type=vpt, strides=st, weight=W3, x=x)[name=string(\"up\")];\n"
      "%@"
      "    tensor<fp16,[1,%d,1,%d]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=vpt, strides=st, weight=W2, x=hh)[name=string(\"y\")];\n"
      "  } -> (y);\n}\n",
      dim,S, h,dim,h,dim,h, h,dim,h,dim,h, dim,h,dim,h,dim,
      h,S, h,S, act_block(h,S,gelu), dim,S];
}

// compile a MIL program with named weight blobs; alloc IOSurfaces sized in/out (fp16 channels x S)
static ANEHandle *compile_common(NSString *mil, NSDictionary *weightsDict, NSArray *files,
                              int in_ch, int out_ch, int S) {
    @autoreleasepool {
        NSError *e = nil;
        NSData *milData = [[mil dataUsingEncoding:NSUTF8StringEncoding] copy];
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:), milData, weightsDict, nil);
        if (!desc) return NULL;
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) return NULL;
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        for (NSArray *pair in files)  // pair = (relpath, NSData)
            [(NSData*)pair[1] writeToFile:[td stringByAppendingPathComponent:pair[0]] atomically:YES];
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            fprintf(stderr, "ane_ffn: compile failed: %s\n", e?[[e description] UTF8String]:"?");
            [fm removeItemAtPath:td error:nil]; return NULL;
        }
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            [fm removeItemAtPath:td error:nil]; return NULL;
        }
        ANEHandle *h = calloc(1, sizeof(ANEHandle));
        h->inBytes  = (size_t)in_ch * S * 2;
        h->outBytes = (size_t)out_ch * S * 2;
        h->ioIn  = create_surface(h->inBytes);
        h->ioOut = create_surface(h->outBytes);
        h->inPtr  = IOSurfaceGetBaseAddress(h->ioIn);
        h->outPtr = IOSurfaceGetBaseAddress(h->ioOut);
        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), h->ioIn);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), h->ioOut);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
        h->model = mdl;       // ARC retains into the __strong struct fields
        h->request = req;
        h->tmpDir = td;
        return h;
    }
}

static void *ffn_compile_impl(int dim, int hidden, int seq,
                      const uint16_t *W1, const uint16_t *W3, const uint16_t *W2, int gelu) {
    if (!g_ready && ane_init() != 0) return NULL;
    @autoreleasepool {
        size_t l1,l3,l2;
        uint8_t *b1=blob(W1,hidden,dim,&l1), *b3=blob(W3,hidden,dim,&l3), *b2=blob(W2,dim,hidden,&l2);
        NSData *d1=[NSData dataWithBytesNoCopy:b1 length:l1 freeWhenDone:YES];
        NSData *d3=[NSData dataWithBytesNoCopy:b3 length:l3 freeWhenDone:YES];
        NSData *d2=[NSData dataWithBytesNoCopy:b2 length:l2 freeWhenDone:YES];
        NSDictionary *wd = @{@"@model_path/weights/w1.bin":@{@"offset":@0,@"data":d1},
                             @"@model_path/weights/w3.bin":@{@"offset":@0,@"data":d3},
                             @"@model_path/weights/w2.bin":@{@"offset":@0,@"data":d2}};
        NSArray *files = @[@[@"weights/w1.bin",d1], @[@"weights/w3.bin",d3], @[@"weights/w2.bin",d2]];
        return compile_common(mil_ffn(dim,hidden,seq,gelu), wd, files, dim, dim, seq);
    }
}

void *ane_ffn_compile(int dim, int hidden, int seq,
                      const uint16_t *W1, const uint16_t *W3, const uint16_t *W2) {
    return ffn_compile_impl(dim,hidden,seq,W1,W3,W2,0);   // SwiGLU (SiLU)
}
void *ane_ffn_compile_gelu(int dim, int hidden, int seq,
                      const uint16_t *W1, const uint16_t *W3, const uint16_t *W2) {
    return ffn_compile_impl(dim,hidden,seq,W1,W3,W2,1);   // GeGLU (tanh-approx GELU)
}

// fused FFN with INT8 weights. Takes fp16 weights, quantizes internally. fp16 I/O.
static void *ffn_compile_int8_impl(int dim, int hidden, int seq,
                           const uint16_t *W1, const uint16_t *W3, const uint16_t *W2, int gelu) {
    if (!g_ready && ane_init() != 0) return NULL;
    @autoreleasepool {
        int n1 = hidden * dim, n2 = dim * hidden;
        int8_t *q1 = malloc(n1), *q3 = malloc(n1), *q2 = malloc(n2);
        uint16_t *sc1 = malloc(hidden * 2), *sc3 = malloc(hidden * 2), *sc2 = malloc(dim * 2);
        quantize_q8(W1, hidden, dim, q1, sc1);    // per output channel
        quantize_q8(W3, hidden, dim, q3, sc3);
        quantize_q8(W2, dim, hidden, q2, sc2);
        size_t l1, l3, l2, ls1, ls3, ls2;
        uint8_t *b1 = blob_q8(q1, hidden, dim, &l1), *b3 = blob_q8(q3, hidden, dim, &l3), *b2 = blob_q8(q2, dim, hidden, &l2);
        uint8_t *bs1 = blob(sc1, hidden, 1, &ls1), *bs3 = blob(sc3, hidden, 1, &ls3), *bs2 = blob(sc2, dim, 1, &ls2);
        free(q1); free(q3); free(q2); free(sc1); free(sc3); free(sc2);
        NSData *d1=[NSData dataWithBytesNoCopy:b1 length:l1 freeWhenDone:YES];
        NSData *d3=[NSData dataWithBytesNoCopy:b3 length:l3 freeWhenDone:YES];
        NSData *d2=[NSData dataWithBytesNoCopy:b2 length:l2 freeWhenDone:YES];
        NSData *ds1=[NSData dataWithBytesNoCopy:bs1 length:ls1 freeWhenDone:YES];
        NSData *ds3=[NSData dataWithBytesNoCopy:bs3 length:ls3 freeWhenDone:YES];
        NSData *ds2=[NSData dataWithBytesNoCopy:bs2 length:ls2 freeWhenDone:YES];
        NSDictionary *wd = @{@"@model_path/weights/w1.bin":@{@"offset":@0,@"data":d1},
                             @"@model_path/weights/w3.bin":@{@"offset":@0,@"data":d3},
                             @"@model_path/weights/w2.bin":@{@"offset":@0,@"data":d2},
                             @"@model_path/weights/s1.bin":@{@"offset":@0,@"data":ds1},
                             @"@model_path/weights/s3.bin":@{@"offset":@0,@"data":ds3},
                             @"@model_path/weights/s2.bin":@{@"offset":@0,@"data":ds2}};
        NSArray *files = @[@[@"weights/w1.bin",d1], @[@"weights/w3.bin",d3], @[@"weights/w2.bin",d2],
                           @[@"weights/s1.bin",ds1], @[@"weights/s3.bin",ds3], @[@"weights/s2.bin",ds2]];
        return compile_common(mil_ffn_q8(dim,hidden,seq,gelu), wd, files, dim, dim, seq);
    }
}

void *ane_ffn_compile_int8(int dim, int hidden, int seq,
                           const uint16_t *W1, const uint16_t *W3, const uint16_t *W2) {
    return ffn_compile_int8_impl(dim,hidden,seq,W1,W3,W2,0);   // SwiGLU
}
void *ane_ffn_compile_int8_gelu(int dim, int hidden, int seq,
                           const uint16_t *W1, const uint16_t *W3, const uint16_t *W2) {
    return ffn_compile_int8_impl(dim,hidden,seq,W1,W3,W2,1);   // GeGLU
}

void *ane_matmul_compile(int in_ch, int out_ch, int seq, const uint16_t *W) {
    if (!g_ready && ane_init() != 0) return NULL;
    @autoreleasepool {
        size_t l; uint8_t *b=blob(W,out_ch,in_ch,&l);
        NSData *d=[NSData dataWithBytesNoCopy:b length:l freeWhenDone:YES];
        NSDictionary *wd = @{@"@model_path/weights/w.bin":@{@"offset":@0,@"data":d}};
        NSArray *files = @[@[@"weights/w.bin",d]];
        return compile_common(mil_matmul(in_ch,out_ch,seq), wd, files, in_ch, out_ch, seq);
    }
}

void  ane_set_input(void *hp, const void *src, size_t bytes) {
    ANEHandle *h=(ANEHandle*)hp; if(!h) return;
    memcpy(h->inPtr, src, bytes < h->inBytes ? bytes : h->inBytes);
}
int   ane_run(void *hp) {
    ANEHandle *h=(ANEHandle*)hp; if(!h) return -1;
    NSString *lastErr = nil;
    for (int attempt = 0; attempt < 4; attempt++) {
        @autoreleasepool {
            NSError *e=nil;
            BOOL ok=((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                h->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, h->request, &e);
            if (ok) return 0;
            lastErr = e ? [e description] : @"(nil error)";
        }
        usleep(200);  // transient ANE busy/contention — back off and retry
    }
    fprintf(stderr, "ane_run: eval failed after retries: %s\n", lastErr ? [lastErr UTF8String] : "?");
    return -1;
}
void  ane_get_output(void *hp, void *dst, size_t bytes) {
    ANEHandle *h=(ANEHandle*)hp; if(!h) return;
    memcpy(dst, h->outPtr, bytes < h->outBytes ? bytes : h->outBytes);
}
void *ane_input_ptr(void *hp){ ANEHandle*h=(ANEHandle*)hp; return h?h->inPtr:NULL; }
void *ane_output_ptr(void *hp){ ANEHandle*h=(ANEHandle*)hp; return h?h->outPtr:NULL; }
size_t ane_input_bytes(void *hp){ ANEHandle*h=(ANEHandle*)hp; return h?h->inBytes:0; }
size_t ane_output_bytes(void *hp){ ANEHandle*h=(ANEHandle*)hp; return h?h->outBytes:0; }

void ane_free(void *hp) {
    ANEHandle *h=(ANEHandle*)hp; if(!h) return;
    @autoreleasepool {
        NSError *e=nil;
        if (h->model) ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(h->model, @selector(unloadWithQoS:error:), 21, &e);
        if (h->tmpDir) [[NSFileManager defaultManager] removeItemAtPath:h->tmpDir error:nil];
        if (h->ioIn)  CFRelease(h->ioIn);
        if (h->ioOut) CFRelease(h->ioOut);
        h->model = nil;       // let ARC release
        h->request = nil;
        h->tmpDir = nil;
        free(h);
    }
}
