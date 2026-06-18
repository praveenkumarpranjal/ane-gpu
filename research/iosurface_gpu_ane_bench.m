// iosurface_gpu_ane_bench.m — Direct IOSurface GPU↔ANE Handoff Benchmark
// Eliminates CPU-mediated sync: Metal MPS + ANE share the same IOSurface memory.
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface \
//     -framework Metal -framework MetalPerformanceShaders -ldl \
//     -o iosurface_gpu_ane_bench iosurface_gpu_ane_bench.m
//
// Run:
//   ./iosurface_gpu_ane_bench

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
// Timing
// ============================================================================

static mach_timebase_info_data_t g_tb;
static double ticks_to_us(uint64_t t) {
    return (double)t * g_tb.numer / g_tb.denom / 1e3;
}

// ============================================================================
// IOSurface helpers
// ============================================================================

static IOSurfaceRef create_surface(size_t bytes) {
    // Round up to page size for Metal compatibility
    size_t page_size = getpagesize();
    size_t alloc_size = ((bytes + page_size - 1) / page_size) * page_size;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(alloc_size),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(alloc_size),
        (id)kIOSurfaceAllocSize: @(alloc_size),
        (id)kIOSurfacePixelFormat: @0
    });
}

// ============================================================================
// ANE weight blob builder (same format as ane_bridge.m)
// ============================================================================

static uint8_t *build_weight_blob(int out_ch, int in_ch, size_t *out_len) {
    int wsize = out_ch * in_ch * 2; // fp16
    int total = 128 + wsize;
    uint8_t *buf = (uint8_t *)calloc(total, 1);

    // Global header
    buf[0] = 0x01; buf[4] = 0x02;
    // Chunk header at offset 64
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    *(uint32_t*)(buf + 72) = wsize;
    *(uint32_t*)(buf + 80) = 128;

    // Random fp16 weights
    _Float16 *fp16 = (_Float16 *)(buf + 128);
    for (int i = 0; i < out_ch * in_ch; i++) {
        fp16[i] = (_Float16)((float)(arc4random() % 1000) / 50000.0f - 0.01f);
    }

    *out_len = total;
    return buf;
}

// ============================================================================
// MIL generator
// ============================================================================

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
        "    } -> (y);\n"
        "}\n",
        in_ch, spatial, in_ch, spatial,
        out_ch, in_ch, out_ch, in_ch,
        out_ch, spatial, out_ch, spatial];
}

// ============================================================================
// ANE kernel handle
// ============================================================================

typedef struct {
    id model;
    id request;
    NSString *tmpDir;
    IOSurfaceRef ioInput;
    IOSurfaceRef ioOutput;
    int in_ch, out_ch, spatial;
} ANEKernel;

static Class g_ANEDesc, g_ANEInMem, g_ANEReq, g_ANEIO;

static bool init_ane_classes(void) {
    void *h = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    if (!h) return false;
    g_ANEDesc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
    g_ANEReq   = NSClassFromString(@"_ANERequest");
    g_ANEIO    = NSClassFromString(@"_ANEIOSurfaceObject");
    return g_ANEDesc && g_ANEInMem && g_ANEReq && g_ANEIO;
}

// Compile ANE kernel with EXTERNAL IOSurfaces (the key difference from ane_bridge)
static ANEKernel *compile_ane_kernel(int in_ch, int out_ch, int spatial,
                                      IOSurfaceRef extInput, IOSurfaceRef extOutput) {
    @autoreleasepool {
        NSError *e = nil;
        NSString *milStr = gen_mil_conv(in_ch, out_ch, spatial);
        NSData *milData = [milStr dataUsingEncoding:NSUTF8StringEncoding];

        // Build weight blob
        size_t wbLen;
        uint8_t *wbBuf = build_weight_blob(out_ch, in_ch, &wbLen);
        NSData *wbData = [NSData dataWithBytesNoCopy:wbBuf length:wbLen freeWhenDone:YES];

        // Create descriptor
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wbData}}, nil);
        if (!desc) { fprintf(stderr, "ANE: modelWithMILText failed\n"); return NULL; }

        // Create model
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) { fprintf(stderr, "ANE: inMemoryModel failed\n"); return NULL; }

        // Write temp files
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wbData writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        // Compile
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            fprintf(stderr, "ANE: compile failed: %s\n", e ? [[e description] UTF8String] : "?");
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }

        // Load
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            fprintf(stderr, "ANE: load failed: %s\n", e ? [[e description] UTF8String] : "?");
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }

        // Build request using EXTERNAL IOSurfaces
        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_ANEIO, @selector(objectWithIOSurface:), extInput);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_ANEIO, @selector(objectWithIOSurface:), extOutput);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

        ANEKernel *k = calloc(1, sizeof(ANEKernel));
        k->model = mdl;
        k->request = req;
        k->tmpDir = td;
        k->ioInput = extInput;
        k->ioOutput = extOutput;
        k->in_ch = in_ch;
        k->out_ch = out_ch;
        k->spatial = spatial;
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
        if (k->model) {
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                k->model, @selector(unloadWithQoS:error:), 21, &e);
        }
        if (k->tmpDir) {
            [[NSFileManager defaultManager] removeItemAtPath:k->tmpDir error:nil];
        }
        k->model = nil;
        k->request = nil;
        k->tmpDir = nil;
        free(k);
    }
}

// ============================================================================
// Metal + MPS GPU matmul
// ============================================================================

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    MPSMatrixMultiplication *mpsKernel;
    id<MTLBuffer> bufW;      // GPU weight shard
    id<MTLBuffer> bufX;      // shared input (from IOSurface)
    id<MTLBuffer> bufY;      // GPU output (from IOSurface)
    MPSMatrix *matW, *matX, *matY;
    int gpu_out_ch, in_ch, spatial;
} GPUKernel;

static GPUKernel *create_gpu_kernel(id<MTLDevice> device, id<MTLCommandQueue> queue,
                                     int in_ch, int gpu_out_ch, int spatial,
                                     IOSurfaceRef ioX, IOSurfaceRef ioY_gpu) {
    GPUKernel *g = calloc(1, sizeof(GPUKernel));
    g->device = device;
    g->queue = queue;
    g->gpu_out_ch = gpu_out_ch;
    g->in_ch = in_ch;
    g->spatial = spatial;

    // Weight buffer (owned by GPU, not shared via IOSurface)
    size_t w_bytes = gpu_out_ch * in_ch * sizeof(float);
    g->bufW = [device newBufferWithLength:w_bytes options:MTLResourceStorageModeShared];
    float *wptr = (float *)g->bufW.contents;
    for (int i = 0; i < gpu_out_ch * in_ch; i++)
        wptr[i] = ((float)(arc4random() % 1000) / 50000.0f - 0.01f);

    // Input buffer from IOSurface — zero-copy!
    size_t x_bytes = in_ch * spatial * sizeof(float);
    void *x_base = IOSurfaceGetBaseAddress(ioX);
    g->bufX = [device newBufferWithBytesNoCopy:x_base
                                        length:IOSurfaceGetAllocSize(ioX)
                                       options:MTLResourceStorageModeShared
                                   deallocator:nil];

    // Output buffer from IOSurface — zero-copy!
    size_t y_bytes = gpu_out_ch * spatial * sizeof(float);
    void *y_base = IOSurfaceGetBaseAddress(ioY_gpu);
    g->bufY = [device newBufferWithBytesNoCopy:y_base
                                        length:IOSurfaceGetAllocSize(ioY_gpu)
                                       options:MTLResourceStorageModeShared
                                   deallocator:nil];

    // MPS matrix descriptors
    // W: [gpu_out_ch, in_ch] row-major
    MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
        matrixDescriptorWithRows:gpu_out_ch columns:in_ch
                        rowBytes:in_ch * sizeof(float)
                        dataType:MPSDataTypeFloat32];
    g->matW = [[MPSMatrix alloc] initWithBuffer:g->bufW descriptor:descW];

    // X: [in_ch, spatial] row-major (channel-first layout = rows of spatial)
    MPSMatrixDescriptor *descX = [MPSMatrixDescriptor
        matrixDescriptorWithRows:in_ch columns:spatial
                        rowBytes:spatial * sizeof(float)
                        dataType:MPSDataTypeFloat32];
    g->matX = [[MPSMatrix alloc] initWithBuffer:g->bufX descriptor:descX];

    // Y: [gpu_out_ch, spatial]
    MPSMatrixDescriptor *descY = [MPSMatrixDescriptor
        matrixDescriptorWithRows:gpu_out_ch columns:spatial
                        rowBytes:spatial * sizeof(float)
                        dataType:MPSDataTypeFloat32];
    g->matY = [[MPSMatrix alloc] initWithBuffer:g->bufY descriptor:descY];

    // MPS kernel: Y = W @ X
    g->mpsKernel = [[MPSMatrixMultiplication alloc]
        initWithDevice:device transposeLeft:NO transposeRight:NO
            resultRows:gpu_out_ch resultColumns:spatial interiorColumns:in_ch
                 alpha:1.0 beta:0.0];

    return g;
}

static void gpu_eval_sync(GPUKernel *g) {
    id<MTLCommandBuffer> cb = [g->queue commandBuffer];
    [g->mpsKernel encodeToCommandBuffer:cb leftMatrix:g->matW rightMatrix:g->matX resultMatrix:g->matY];
    [cb commit];
    [cb waitUntilCompleted];
}

static void gpu_free(GPUKernel *g) {
    if (g) free(g);
}

// ============================================================================
// Benchmark configs
// ============================================================================

typedef struct {
    const char *name;
    int dim, hidden, seq;
} Config;

static Config CONFIGS[] = {
    {"Stories110M",  768,  2048, 256},
    {"Qwen3-0.6B",  1024, 2816, 256},
    {"Llama-7B/128", 4096, 11008, 128},
    {"Llama-7B/256", 4096, 11008, 256},
};
static int N_CONFIGS = 4;

static float SPLIT_RATIOS[] = {0.5, 0.7, 0.8, 0.9};
static int N_RATIOS = 4;

// ============================================================================
// Benchmarks
// ============================================================================

static void fill_surface_random(IOSurfaceRef s, size_t bytes) {
    float *p = (float *)IOSurfaceGetBaseAddress(s);
    for (size_t i = 0; i < bytes / sizeof(float); i++)
        p[i] = ((float)(arc4random() % 1000) / 50000.0f - 0.01f);
}

static void bench_gpu_only(id<MTLDevice> dev, id<MTLCommandQueue> q,
                            int in_ch, int out_ch, int spatial,
                            int warmup, int iters,
                            double *out_mean_us) {
    size_t x_bytes = in_ch * spatial * sizeof(float);
    size_t y_bytes = out_ch * spatial * sizeof(float);
    IOSurfaceRef ioX = create_surface(x_bytes);
    IOSurfaceRef ioY = create_surface(y_bytes);
    fill_surface_random(ioX, x_bytes);

    GPUKernel *g = create_gpu_kernel(dev, q, in_ch, out_ch, spatial, ioX, ioY);

    for (int i = 0; i < warmup; i++) gpu_eval_sync(g);

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) gpu_eval_sync(g);
    double total_us = ticks_to_us(mach_absolute_time() - t0);
    *out_mean_us = total_us / iters;

    gpu_free(g);
    CFRelease(ioX);
    CFRelease(ioY);
}

static void bench_ane_only(int in_ch, int out_ch, int spatial,
                            int warmup, int iters,
                            double *out_mean_us) {
    size_t x_bytes = in_ch * spatial * sizeof(float);
    size_t y_bytes = out_ch * spatial * sizeof(float);
    IOSurfaceRef ioX = create_surface(x_bytes);
    IOSurfaceRef ioY = create_surface(y_bytes);
    fill_surface_random(ioX, x_bytes);

    ANEKernel *k = compile_ane_kernel(in_ch, out_ch, spatial, ioX, ioY);
    if (!k) { *out_mean_us = -1; CFRelease(ioX); CFRelease(ioY); return; }

    for (int i = 0; i < warmup; i++) ane_eval(k);

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) ane_eval(k);
    double total_us = ticks_to_us(mach_absolute_time() - t0);
    *out_mean_us = total_us / iters;

    ane_free(k);
    CFRelease(ioX);
    CFRelease(ioY);
}

static void bench_split(id<MTLDevice> dev, id<MTLCommandQueue> q,
                         int in_ch, int out_ch, int spatial, float ratio,
                         int warmup, int iters,
                         double *out_mean_us, int *out_ane_ch, int *out_gpu_ch) {
    int ane_out_ch = (int)(out_ch * ratio);
    ane_out_ch = (ane_out_ch / 16) * 16;
    if (ane_out_ch < 16) ane_out_ch = 16;
    int gpu_out_ch = out_ch - ane_out_ch;
    if (gpu_out_ch <= 0) { *out_mean_us = -1; return; }

    *out_ane_ch = ane_out_ch;
    *out_gpu_ch = gpu_out_ch;

    // Shared input IOSurface — SAME memory for GPU and ANE
    size_t x_bytes = in_ch * spatial * sizeof(float);
    IOSurfaceRef ioX = create_surface(x_bytes);
    fill_surface_random(ioX, x_bytes);

    // Separate output IOSurfaces
    size_t y_gpu_bytes = gpu_out_ch * spatial * sizeof(float);
    size_t y_ane_bytes = ane_out_ch * spatial * sizeof(float);
    IOSurfaceRef ioY_gpu = create_surface(y_gpu_bytes);
    IOSurfaceRef ioY_ane = create_surface(y_ane_bytes);

    // Create GPU kernel — reads from shared ioX, writes to ioY_gpu
    GPUKernel *gk = create_gpu_kernel(dev, q, in_ch, gpu_out_ch, spatial, ioX, ioY_gpu);

    // Create ANE kernel — reads from SAME ioX, writes to ioY_ane
    ANEKernel *ak = compile_ane_kernel(in_ch, ane_out_ch, spatial, ioX, ioY_ane);
    if (!ak) {
        *out_mean_us = -1;
        gpu_free(gk);
        CFRelease(ioX); CFRelease(ioY_gpu); CFRelease(ioY_ane);
        return;
    }

    // Warmup
    for (int i = 0; i < warmup; i++) {
        gpu_eval_sync(gk);
        ane_eval(ak);
    }

    // Benchmark: concurrent GPU + ANE on shared input
    dispatch_semaphore_t sem_ane = dispatch_semaphore_create(0);
    dispatch_queue_t ane_q = dispatch_queue_create("ane.eval", DISPATCH_QUEUE_SERIAL);

    double total_us = 0;
    for (int i = 0; i < iters; i++) {
        uint64_t t0 = mach_absolute_time();

        // Launch ANE eval on background queue
        dispatch_async(ane_q, ^{
            ane_eval(ak);
            dispatch_semaphore_signal(sem_ane);
        });

        // GPU eval on main thread
        id<MTLCommandBuffer> cb = [q commandBuffer];
        [gk->mpsKernel encodeToCommandBuffer:cb leftMatrix:gk->matW
                                  rightMatrix:gk->matX resultMatrix:gk->matY];
        [cb commit];
        [cb waitUntilCompleted];

        // Wait for ANE
        dispatch_semaphore_wait(sem_ane, DISPATCH_TIME_FOREVER);

        uint64_t t1 = mach_absolute_time();
        total_us += ticks_to_us(t1 - t0);
    }

    *out_mean_us = total_us / iters;

    ane_free(ak);
    gpu_free(gk);
    CFRelease(ioX);
    CFRelease(ioY_gpu);
    CFRelease(ioY_ane);
}

// ============================================================================
// Boundary cost measurement
// ============================================================================

static void bench_boundary_gpu_to_ane(id<MTLDevice> dev, id<MTLCommandQueue> q,
                                       int in_ch, int spatial, int iters,
                                       double *out_mean_us) {
    // GPU writes to IOSurface, then ANE reads from same IOSurface
    // Measures: GPU commit → GPU complete → ANE eval → ANE complete
    size_t bytes = in_ch * spatial * sizeof(float);
    IOSurfaceRef ioS = create_surface(bytes);
    IOSurfaceRef ioOut = create_surface(bytes);
    fill_surface_random(ioS, bytes);

    // Trivial ANE kernel (identity-ish) — just to measure handoff
    ANEKernel *ak = compile_ane_kernel(in_ch, in_ch, spatial, ioS, ioOut);
    if (!ak) { *out_mean_us = -1; CFRelease(ioS); CFRelease(ioOut); return; }

    // GPU: trivial write via Metal blit (fill buffer)
    // ANE: eval reading same surface

    // Warmup
    for (int i = 0; i < 10; i++) ane_eval(ak);

    // Measure: just ANE eval on IOSurface (no GPU write — baseline)
    double ane_only = 0;
    {
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++) ane_eval(ak);
        ane_only = ticks_to_us(mach_absolute_time() - t0) / iters;
    }

    // Measure: GPU commit+wait THEN ANE eval (sequential boundary)
    double gpu_then_ane = 0;
    {
        id<MTLBuffer> buf = [dev newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(ioS)
                                                  length:IOSurfaceGetAllocSize(ioS)
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++) {
            // GPU "touches" the surface
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            [blit fillBuffer:buf range:NSMakeRange(0, 4) value:0x42]; // minimal GPU work
            [blit endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            // Now ANE reads same surface
            ane_eval(ak);
        }
        gpu_then_ane = ticks_to_us(mach_absolute_time() - t0) / iters;
    }

    *out_mean_us = gpu_then_ane - ane_only; // boundary overhead
    printf("    ANE-only: %.1f μs  GPU→ANE sequential: %.1f μs  boundary overhead: %.1f μs\n",
           ane_only, gpu_then_ane, gpu_then_ane - ane_only);

    ane_free(ak);
    CFRelease(ioS);
    CFRelease(ioOut);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);

        printf("================================================================\n");
        printf("  Direct IOSurface GPU↔ANE Handoff Benchmark\n");
        printf("================================================================\n\n");

        // Init Metal
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "FATAL: No Metal device\n"); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        printf("  GPU: %s\n", [[dev name] UTF8String]);

        // Init ANE
        if (!init_ane_classes()) {
            fprintf(stderr, "FATAL: Failed to load ANE private classes\n");
            return 1;
        }
        printf("  ANE: initialized ✓\n\n");

        // ── Benchmark 1: Boundary Cost ──────────────────────────────
        printf("================================================================\n");
        printf("  Benchmark 1: GPU→ANE Boundary Cost (IOSurface)\n");
        printf("================================================================\n");
        for (int i = 0; i < N_CONFIGS; i++) {
            Config c = CONFIGS[i];
            printf("  %s (dim=%d, seq=%d):\n", c.name, c.dim, c.seq);
            double boundary_us;
            bench_boundary_gpu_to_ane(dev, q, c.dim, c.seq, 50, &boundary_us);
        }

        // ── Benchmark 2: GPU-only vs ANE-only baselines ─────────────
        printf("\n================================================================\n");
        printf("  Benchmark 2: GPU-only vs ANE-only (direct IOSurface)\n");
        printf("================================================================\n\n");
        printf("  %-16s %10s %10s %8s %10s %10s\n",
               "Config", "GPU μs", "ANE μs", "GFLOP", "GPU TFLOPS", "ANE TFLOPS");
        printf("  %-16s %10s %10s %8s %10s %10s\n",
               "----------------", "----------", "----------", "--------", "----------", "----------");

        double gpu_baselines[N_CONFIGS], ane_baselines[N_CONFIGS];

        for (int i = 0; i < N_CONFIGS; i++) {
            Config c = CONFIGS[i];
            double gflops = 2.0 * c.dim * c.hidden * c.seq / 1e9;

            bench_gpu_only(dev, q, c.dim, c.hidden, c.seq, 10, 50, &gpu_baselines[i]);
            bench_ane_only(c.dim, c.hidden, c.seq, 10, 50, &ane_baselines[i]);

            double gpu_tf = gflops / (gpu_baselines[i] / 1e6) / 1e3;
            double ane_tf = ane_baselines[i] > 0 ? gflops / (ane_baselines[i] / 1e6) / 1e3 : 0;

            if (ane_baselines[i] > 0)
                printf("  %-16s %10.0f %10.0f %8.2f %10.2f %10.2f\n",
                       c.name, gpu_baselines[i], ane_baselines[i], gflops, gpu_tf, ane_tf);
            else
                printf("  %-16s %10.0f %10s %8.2f %10.2f %10s\n",
                       c.name, gpu_baselines[i], "FAIL", gflops, gpu_tf, "FAIL");
        }

        // ── Benchmark 3: Split FFN with direct IOSurface ────────────
        printf("\n================================================================\n");
        printf("  Benchmark 3: Tensor-Split FFN (direct IOSurface, no CPU copy)\n");
        printf("================================================================\n");

        for (int i = 0; i < N_CONFIGS; i++) {
            Config c = CONFIGS[i];
            double gpu_base = gpu_baselines[i];

            printf("\n  ── %s (dim=%d, hidden=%d, seq=%d) ──\n",
                   c.name, c.dim, c.hidden, c.seq);
            printf("  GPU-only baseline: %.0f μs\n", gpu_base);
            printf("  %7s %7s %7s %10s %8s %8s\n",
                   "Ratio", "ANE ch", "GPU ch", "Split μs", "Speedup", "Verdict");
            printf("  %7s %7s %7s %10s %8s %8s\n",
                   "-------", "-------", "-------", "----------", "--------", "--------");

            double best_speedup = 0;
            float best_ratio = 0;

            for (int r = 0; r < N_RATIOS; r++) {
                double split_us;
                int ane_ch, gpu_ch;
                bench_split(dev, q, c.dim, c.hidden, c.seq, SPLIT_RATIOS[r],
                           10, 50, &split_us, &ane_ch, &gpu_ch);

                if (split_us < 0) {
                    printf("  %7.1f %7s\n", SPLIT_RATIOS[r], "FAIL");
                    continue;
                }

                double speedup = gpu_base / split_us;
                const char *verdict = speedup > 1.0 ? "✓ WIN" : "✗ LOSE";

                if (speedup > best_speedup) {
                    best_speedup = speedup;
                    best_ratio = SPLIT_RATIOS[r];
                }

                printf("  %7.1f %7d %7d %10.0f %7.2fx %8s\n",
                       SPLIT_RATIOS[r], ane_ch, gpu_ch, split_us, speedup, verdict);
            }

            if (best_ratio > 0)
                printf("\n  Best: ratio=%.1f → %.2fx %s\n",
                       best_ratio, best_speedup,
                       best_speedup > 1.0 ? "(wins!)" : "(still loses)");
        }

        // ── Summary ─────────────────────────────────────────────────
        printf("\n================================================================\n");
        printf("  Summary: Direct IOSurface vs Python CPU-Mediated\n");
        printf("================================================================\n");
        printf("  (Compare these numbers against hetero_ffn_bench.py results)\n");
        printf("  Key advantage: no numpy copy, no Python threading overhead,\n");
        printf("  GPU and ANE share SAME IOSurface memory.\n\n");

        return 0;
    }
}
