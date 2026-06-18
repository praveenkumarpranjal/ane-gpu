// ane_ffn_probe.m — de-risk gate: does the ANE 1x1-conv-as-matmul path
// (verbatim from qwen_split_infer.m) compile, evaluate (correct selector),
// and produce NUMERICALLY CORRECT output for the real Qwen2.5-0.5B FFN shapes,
// at decode (spatial=1) and prefill (spatial=128) — and how fast?
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//       -o ane_ffn_probe ane_ffn_probe.m

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

static mach_timebase_info_data_t g_tb;
static double ticks_us(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1000.0; }

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

// verbatim from qwen_split_infer.m
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
        if (!desc) { fprintf(stderr, "  descriptor failed (in=%d out=%d)\n", in_ch, out_ch); return NULL; }

        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) { fprintf(stderr, "  model init failed (in=%d out=%d)\n", in_ch, out_ch); return NULL; }

        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wb writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            fprintf(stderr, "  compile failed (in=%d out=%d): %s\n",
                    in_ch, out_ch, e ? [[e description] UTF8String] : "unknown");
            [fm removeItemAtPath:td error:nil]; return NULL;
        }
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            fprintf(stderr, "  load failed (in=%d out=%d): %s\n",
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

// ---- test one shape ----
static void test_shape(const char *name, int in_ch, int out_ch, int spatial) {
    size_t in_bytes  = (size_t)in_ch  * spatial * 4; // fp32
    size_t out_bytes = (size_t)out_ch * spatial * 4; // fp32
    IOSurfaceRef ioIn  = create_surface(in_bytes);
    IOSurfaceRef ioOut = create_surface(out_bytes);

    // random fp16 weights [out_ch, in_ch], small magnitude
    _Float16 *W = malloc((size_t)out_ch * in_ch * sizeof(_Float16));
    for (size_t i = 0; i < (size_t)out_ch * in_ch; i++)
        W[i] = (_Float16)(((float)rand()/RAND_MAX - 0.5f) * 0.1f);

    // random fp32 input [in_ch, spatial] channel-major (x[c*spatial + s])
    float *x = malloc(in_bytes);
    for (size_t i = 0; i < (size_t)in_ch * spatial; i++)
        x[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;

    IOSurfaceLock(ioIn, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(ioIn), x, in_bytes);
    IOSurfaceUnlock(ioIn, 0, NULL);

    uint64_t c0 = mach_absolute_time();
    ANEKernel *k = compile_ane_real(in_ch, out_ch, spatial, (const uint16_t*)W, ioIn, ioOut);
    uint64_t c1 = mach_absolute_time();
    if (!k) { printf("%-22s FAILED to compile\n", name); free(W); free(x); return; }

    // warmup + timed
    for (int i = 0; i < 10; i++) ane_eval(k);
    int iters = 50;
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) ane_eval(k);
    uint64_t t1 = mach_absolute_time();
    double us_per = ticks_us(t1 - t0) / iters;

    // read output [out_ch, spatial], channel-major
    float *y = malloc(out_bytes);
    IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
    memcpy(y, IOSurfaceGetBaseAddress(ioOut), out_bytes);
    IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);

    // CPU reference (fp32 accumulation of fp16 operands; input cast to fp16 like the MIL does)
    double max_abs = 0, max_rel = 0, ref_mag = 0;
    int check_s = spatial < 4 ? spatial : 4; // check a few spatial positions
    for (int s = 0; s < check_s; s++) {
        for (int o = 0; o < out_ch; o++) {
            float acc = 0;
            for (int i = 0; i < in_ch; i++) {
                float xv = (float)(_Float16)x[(size_t)i*spatial + s];
                float wv = (float)W[(size_t)o*in_ch + i];
                acc += xv * wv;
            }
            float got = y[(size_t)o*spatial + s];
            float ae = fabsf(got - acc);
            float re = ae / (fabsf(acc) + 1e-6f);
            if (ae > max_abs) max_abs = ae;
            if (re > max_rel) max_rel = re;
            if (fabsf(acc) > ref_mag) ref_mag = fabsf(acc);
        }
    }
    double gflop = 2.0 * in_ch * out_ch * spatial / 1e9;
    double tflops = gflop / (us_per / 1e6) / 1e3;
    printf("%-22s in=%-5d out=%-5d sp=%-4d | compile %6.1f ms | %7.1f us/eval | %5.2f TFLOPS | max_abs=%.4f max_rel=%.3f (refmag=%.2f) %s\n",
           name, in_ch, out_ch, spatial,
           ticks_us(c1-c0)/1000.0, us_per, tflops,
           max_abs, max_rel, ref_mag,
           (max_rel < 0.05 || max_abs < 0.02) ? "OK" : "*** MISMATCH");
    free(W); free(x); free(y);
}

int main(void) {
    mach_timebase_info(&g_tb);
    if (!init_ane_classes()) { fprintf(stderr, "ANE classes failed\n"); return 1; }
    printf("ANE classes loaded. Qwen2.5-0.5B FFN shapes: dim=896, hidden=4864.\n");
    printf("Real FFN matmuls per layer: W1/W3 [896->4864], W2 [4864->896].\n\n");

    printf("== DECODE shapes (spatial=1, single token) ==\n");
    test_shape("W1W3 half (50%)", 896, 2432, 1);   // gate+up shard at 50% split
    test_shape("W1W3 full",       896, 4864, 1);
    test_shape("W2 half (50%)",   2432, 896, 1);
    test_shape("W2 full",         4864, 896, 1);

    printf("\n== PREFILL shapes (spatial=128, batched) ==\n");
    test_shape("W1W3 half sp128",  896, 2432, 128);
    test_shape("W1W3 full sp128",  896, 4864, 128);
    test_shape("W2 half sp128",    2432, 896, 128);
    test_shape("W2 full sp128",    4864, 896, 128);

    printf("\n== PREFILL shapes (spatial=256) ==\n");
    test_shape("W1W3 full sp256",  896, 4864, 256);
    test_shape("W2 full sp256",    4864, 896, 256);

    printf("\n== SPATIAL THRESHOLD SWEEP (W1W3 896->4864): where does correctness begin? ==\n");
    int sweep[] = {1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 0};
    for (int i = 0; sweep[i]; i++) {
        char nm[32]; snprintf(nm, sizeof nm, "sp=%d", sweep[i]);
        test_shape(nm, 896, 4864, sweep[i]);
    }
    return 0;
}
