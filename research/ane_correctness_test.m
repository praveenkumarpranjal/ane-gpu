// ane_correctness_test.m - Verify ANE conv produces correct matmul output
// Compile: xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl -o ane_correctness_test ane_correctness_test.m
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#include <stdio.h>
#include <math.h>

// ─── ANE runtime classes ───
static Class g_ANEDesc, g_ANEInMem, g_ANEReq, g_ANEIO;

static bool load_ane_runtime(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_ANEDesc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
    g_ANEReq   = NSClassFromString(@"_ANERequest");
    g_ANEIO    = NSClassFromString(@"_ANEIOSurfaceObject");
    return g_ANEDesc && g_ANEInMem && g_ANEReq && g_ANEIO;
}

// ─── IOSurface helpers ── (flat buffer, same as qwen_split_infer.m)
static IOSurfaceRef make_io_surface(int n_floats) {
    size_t bytes = n_floats * sizeof(float);
    size_t ps = getpagesize();
    size_t as = ((bytes + ps - 1) / ps) * ps;
    if (as < ps) as = ps;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(as), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(as),
        (id)kIOSurfaceAllocSize: @(as), (id)kIOSurfacePixelFormat: @0
    });
}

// ─── MIL generation ───
static NSString *gen_mil_conv(int in_ch, int out_ch, int spatial, uint64_t weight_offset) {
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
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%llu)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
        "    } -> (y);\n}\n",
        in_ch, spatial, in_ch, spatial,
        out_ch, in_ch, out_ch, in_ch, (unsigned long long)weight_offset,
        out_ch, spatial, out_ch, spatial];
}

// ─── Blob builders with different offsets ───
typedef struct {
    uint8_t *data;
    size_t len;
    uint64_t mil_offset;
    const char *name;
} BlobVariant;

static BlobVariant make_blob_128hdr(const uint16_t *weights, int out_ch, int in_ch) {
    // Original format: 128-byte header, data at 128, MIL offset=64
    int ws = out_ch * in_ch * 2;
    int total = 128 + ws;
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    *(uint32_t*)(buf + 72) = ws;
    *(uint32_t*)(buf + 80) = 128;
    memcpy(buf + 128, weights, ws);
    return (BlobVariant){buf, total, 64, "128hdr_off64"};
}

static BlobVariant make_blob_128hdr_off128(const uint16_t *weights, int out_ch, int in_ch) {
    // 128-byte header, data at 128, MIL offset=128
    int ws = out_ch * in_ch * 2;
    int total = 128 + ws;
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    *(uint32_t*)(buf + 72) = ws;
    *(uint32_t*)(buf + 80) = 128;
    memcpy(buf + 128, weights, ws);
    return (BlobVariant){buf, total, 128, "128hdr_off128"};
}

static BlobVariant make_blob_data_at_64(const uint16_t *weights, int out_ch, int in_ch) {
    // Weights start at byte 64, header in first 64 bytes only
    int ws = out_ch * in_ch * 2;
    int total = 64 + ws;
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    memcpy(buf + 64, weights, ws);
    return (BlobVariant){buf, total, 64, "64hdr_off64"};
}

static BlobVariant make_blob_no_header(const uint16_t *weights, int out_ch, int in_ch) {
    // No header at all, weights at byte 0, MIL offset=0
    int ws = out_ch * in_ch * 2;
    uint8_t *buf = (uint8_t *)calloc(ws, 1);
    memcpy(buf, weights, ws);
    return (BlobVariant){buf, ws, 0, "nohdr_off0"};
}

static BlobVariant make_blob_128hdr_data_at_both(const uint16_t *weights, int out_ch, int in_ch) {
    // 128-byte header, weights at BOTH byte 64 AND byte 128
    // Overwrites secondary header with weight data at 64
    int ws = out_ch * in_ch * 2;
    int total = 128 + ws;
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    // secondary header (may be overwritten by weights if ws > 64)
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    *(uint32_t*)(buf + 72) = ws;
    *(uint32_t*)(buf + 80) = 128;
    memcpy(buf + 128, weights, ws); // data at 128
    // Also copy first min(ws, 64) bytes of weights at byte 64 (over secondary header)
    // But we want the FULL weights at 64, so extend the blob
    int total2 = 64 + ws;
    if (total2 > total) {
        buf = realloc(buf, total2);
        memset(buf + total, 0, total2 - total);
    }
    // Don't overwrite the header - just keep data at 128
    // Instead, create a new blob that's 128+ws but with weights copied at 64 too
    // This would overwrite bytes 64-127 of the header
    // Only safe if ws >= 64
    if (ws >= 64) {
        memcpy(buf + 64, weights, ws); // overwrite secondary header with weight data
    }
    return (BlobVariant){buf, total, 64, "128hdr_data@64+128"};
}

// ─── Try compile + run ANE kernel ───
typedef struct {
    id model;
    id request;
    NSString *tmpDir;
} ANEKernel;

static bool ane_eval(ANEKernel *k) {
    @autoreleasepool {
        NSError *e = nil;
        return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            k->model, @selector(evaluateWithQoS:savedOutputs:options:error:),
            21, k->request, @{}, &e);
    }
}

static ANEKernel *try_compile_run(int in_ch, int out_ch, int spatial,
                                    BlobVariant blob,
                                    IOSurfaceRef ioIn, IOSurfaceRef ioOut,
                                    const char **error_stage) {
    @autoreleasepool {
        *error_stage = "none";
        NSError *e = nil;
        NSString *milText = gen_mil_conv(in_ch, out_ch, spatial, blob.mil_offset);
        NSData *mil = [[milText dataUsingEncoding:NSUTF8StringEncoding] copy];
        NSData *wb = [NSData dataWithBytes:blob.data length:blob.len];
        
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            mil, @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wb}}, nil);
        if (!desc) { *error_stage = "descriptor"; return NULL; }

        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) { *error_stage = "model_init"; return NULL; }

        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wb writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            *error_stage = "compile";
            [fm removeItemAtPath:td error:nil]; return NULL;
        }
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            *error_stage = "load";
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

// ─── Main test ───
int main(int argc, const char *argv[]) {
    @autoreleasepool {
        printf("═══════════════════════════════════════════════════════════════\n");
        printf("  ANE Correctness Test — Verify conv produces correct matmul\n");
        printf("═══════════════════════════════════════════════════════════════\n\n");
        
        if (!load_ane_runtime()) {
            printf("FATAL: Cannot load ANE runtime\n");
            return 1;
        }
        printf("✓ ANE runtime loaded\n\n");
        
        // Test dimensions (must be multiple of 16 for ANE)
        int in_ch = 32;
        int out_ch = 32;
        int spatial = 1;
        
        // Create known weights: W[i][j] = (i * in_ch + j + 1) * 0.001
        int n_weights = out_ch * in_ch;
        uint16_t *weights_fp16 = calloc(n_weights, sizeof(uint16_t));
        float *weights_fp32 = calloc(n_weights, sizeof(float));
        for (int i = 0; i < n_weights; i++) {
            float val = (float)(i + 1) * 0.001f;
            weights_fp32[i] = val;
            // Convert fp32 → fp16 manually
            uint32_t fbits; memcpy(&fbits, &val, 4);
            uint32_t sign = (fbits >> 16) & 0x8000;
            int exp = ((fbits >> 23) & 0xFF) - 127 + 15;
            uint32_t mant = (fbits >> 13) & 0x03FF;
            if (exp <= 0) { exp = 0; mant = 0; }
            else if (exp >= 31) { exp = 31; mant = 0; }
            weights_fp16[i] = sign | (exp << 10) | mant;
        }
        
        // Known input: x = [1, 2, 3, 4, 0, 0, ...]
        float input_fp32[32];
        for (int i = 0; i < in_ch; i++) input_fp32[i] = (i < 4) ? (float)(i+1) : 0.0f;
        
        // Expected output: y[i] = sum_j(W[i][j] * x[j])
        float expected[32];
        for (int i = 0; i < out_ch; i++) {
            expected[i] = 0;
            for (int j = 0; j < in_ch; j++)
                expected[i] += weights_fp32[i * in_ch + j] * input_fp32[j];
        }
        
        printf("Config: in_ch=%d, out_ch=%d, spatial=%d\n", in_ch, out_ch, spatial);
        printf("Input:    [%.1f, %.1f, %.1f, %.1f, 0, ...]\n", 
               input_fp32[0], input_fp32[1], input_fp32[2], input_fp32[3]);
        printf("Expected: [%.4f, %.4f, %.4f, %.4f, ...]\n\n", 
               expected[0], expected[1], expected[2], expected[3]);
        
        // Create IOSurfaces (flat, matching qwen_split_infer.m)
        IOSurfaceRef ioIn  = make_io_surface(in_ch);
        IOSurfaceRef ioOut = make_io_surface(out_ch);
        if (!ioIn || !ioOut) { printf("FATAL: IOSurface creation failed\n"); return 1; }
        printf("IOSurface alloc: in=%zu bytes, out=%zu bytes\n\n",
               IOSurfaceGetAllocSize(ioIn), IOSurfaceGetAllocSize(ioOut));
        
        // Test each blob variant
        BlobVariant variants[] = {
            make_blob_128hdr(weights_fp16, out_ch, in_ch),
            make_blob_128hdr_off128(weights_fp16, out_ch, in_ch),
            make_blob_data_at_64(weights_fp16, out_ch, in_ch),
            make_blob_no_header(weights_fp16, out_ch, in_ch),
            make_blob_128hdr_data_at_both(weights_fp16, out_ch, in_ch),
        };
        int n_variants = sizeof(variants) / sizeof(variants[0]);
        
        for (int v = 0; v < n_variants; v++) {
            BlobVariant blob = variants[v];
            printf("─── Variant %d: %s (offset=%llu, blob=%zu bytes) ───\n",
                   v, blob.name, (unsigned long long)blob.mil_offset, blob.len);
            
            // Write input to IOSurface (flat, contiguous fp32)
            IOSurfaceLock(ioIn, 0, NULL);
            float *in_ptr = (float *)IOSurfaceGetBaseAddress(ioIn);
            memset(in_ptr, 0, IOSurfaceGetAllocSize(ioIn));
            memcpy(in_ptr, input_fp32, in_ch * sizeof(float));
            IOSurfaceUnlock(ioIn, 0, NULL);
            
            // Zero output
            IOSurfaceLock(ioOut, 0, NULL);
            memset(IOSurfaceGetBaseAddress(ioOut), 0, IOSurfaceGetAllocSize(ioOut));
            IOSurfaceUnlock(ioOut, 0, NULL);
            
            // Try compile + run
            const char *err_stage;
            ANEKernel *k = try_compile_run(in_ch, out_ch, spatial, blob, ioIn, ioOut, &err_stage);
            
            if (!k) {
                printf("  FAILED at stage: %s\n\n", err_stage);
                free(blob.data);
                continue;
            }
            printf("  ✓ Compiled + loaded\n");
            
            bool ok = ane_eval(k);
            if (!ok) {
                printf("  FAILED: ane_eval returned false\n\n");
                free(blob.data);
                continue;
            }
            printf("  ✓ Evaluated\n");
            
            // Read output (flat, contiguous fp32)
            IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
            float *out_ptr = (float *)IOSurfaceGetBaseAddress(ioOut);
            float output[32];
            memcpy(output, out_ptr, out_ch * sizeof(float));
            IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);
            
            // Check correctness
            float max_err = 0; int worst_idx = 0; bool all_zero = true;
            for (int i = 0; i < out_ch; i++) {
                if (output[i] != 0) all_zero = false;
                float err = fabsf(output[i] - expected[i]);
                if (err > max_err) { max_err = err; worst_idx = i; }
            }
            
            printf("  Output[0..3]: [%.6f, %.6f, %.6f, %.6f]\n",
                   output[0], output[1], output[2], output[3]);
            printf("  Expect[0..3]: [%.6f, %.6f, %.6f, %.6f]\n",
                   expected[0], expected[1], expected[2], expected[3]);
            
            if (all_zero) {
                printf("  ❌ ALL ZEROS — ANE produced no output\n");
            } else if (max_err < 0.05f) {
                printf("  ✅ CORRECT! max_err=%.6f at idx=%d\n", max_err, worst_idx);
            } else {
                printf("  ❌ WRONG! max_err=%.6f at idx=%d (got=%.6f, exp=%.6f)\n",
                       max_err, worst_idx, output[worst_idx], expected[worst_idx]);
                printf("  Output[4..7]: [%.6f, %.6f, %.6f, %.6f]\n",
                       output[4], output[5], output[6], output[7]);
            }
            printf("\n");
            free(blob.data);
        }
        
        free(weights_fp16);
        free(weights_fp32);
        printf("Done.\n");
        return 0;
    }
}

