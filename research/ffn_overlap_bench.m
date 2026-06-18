// ffn_overlap_bench.m — Does GPU || ANE concurrent execution give ADDITIVE throughput?
//
// Measures the real Qwen2.5-0.5B FFN matmuls at prefill shape (seq=256), splitting
// output channels between GPU (MPS fp16) and ANE (1x1 conv fp16), running them
// TRULY CONCURRENTLY (commit GPU cmd buffer -> run blocking ANE eval while GPU runs
// async -> join). Sweeps split ratio. Compares vs GPU-only-full and ANE-only-full.
//
// Zero-copy fp16 handoff: X lives once in an IOSurface, wrapped as an MTLBuffer for
// MPS AND as an _ANEIOSurfaceObject for ANE. No fp16<->fp32 conversion.
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface \
//     -framework Metal -framework MetalPerformanceShaders -ldl -o ffn_overlap_bench ffn_overlap_bench.m

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
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
static id<MTLBuffer> buf_from_surface(id<MTLDevice> dev, IOSurfaceRef s) {
    return [dev newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(s)
                                 length:IOSurfaceGetAllocSize(s)
                                options:MTLResourceStorageModeShared deallocator:nil];
}

static uint8_t *build_weight_blob_real(const uint16_t *weights, int out_ch, int in_ch, size_t *out_len) {
    int ws = out_ch * in_ch * 2;
    int total = 128 + ws;
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    buf[0]=0x01; buf[4]=0x02; buf[64]=0xEF; buf[65]=0xBE; buf[66]=0xAD; buf[67]=0xDE; buf[68]=0x01;
    *(uint32_t*)(buf+72)=ws; *(uint32_t*)(buf+80)=128;
    memcpy(buf+128, weights, ws);
    *out_len = total; return buf;
}

// fp16-in / fp16-out conv (zero-copy fp16 handoff; no fp32 cast)
static NSString *gen_mil_conv_fp16(int in_ch, int out_ch, int spatial) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x16) {\n"
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n"
        "    } -> (y16);\n}\n",
        in_ch, spatial, out_ch, in_ch, out_ch, in_ch, out_ch, spatial];
}

static ANEKernel *compile_ane(int in_ch, int out_ch, int spatial, const uint16_t *weights,
                              IOSurfaceRef ioIn, IOSurfaceRef ioOut) {
    @autoreleasepool {
        NSError *e = nil;
        NSData *mil = [[gen_mil_conv_fp16(in_ch, out_ch, spatial) dataUsingEncoding:NSUTF8StringEncoding] copy];
        size_t wbLen; uint8_t *wbBuf = build_weight_blob_real(weights, out_ch, in_ch, &wbLen);
        NSData *wb = [NSData dataWithBytesNoCopy:wbBuf length:wbLen freeWhenDone:YES];
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            mil, @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wb}}, nil);
        if (!desc) return NULL;
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) return NULL;
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wb writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            fprintf(stderr, "  ANE compile failed (in=%d out=%d sp=%d): %s\n", in_ch, out_ch, spatial, e?[[e description] UTF8String]:"?");
            return NULL;
        }
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) return NULL;
        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioIn);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioOut);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
        ANEKernel *k = calloc(1, sizeof(ANEKernel)); k->model=mdl; k->request=req; k->tmpDir=td; return k;
    }
}
static bool ane_eval(ANEKernel *k) {
    @autoreleasepool {
        NSError *e = nil;
        return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            k->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, k->request, &e);
    }
}

static int round16(int v){ int r=(v/16)*16; return r<16?16:r; }

// ---- globals for device ----
static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_q;

// MPS matmul: Y[M,N] = W[M,K] * X[K,N], fp16
static MPSMatrix* mk_mat(id<MTLBuffer> b, int rows, int cols){
    int rb = ((cols*2+15)/16)*16;
    MPSMatrixDescriptor *d=[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols rowBytes:rb dataType:MPSDataTypeFloat16];
    return [[MPSMatrix alloc] initWithBuffer:b descriptor:d];
}

typedef struct {
    int in_ch, out_ch, S;
    id<MTLBuffer> bufX;        // [in, S] fp16 (shared IOSurface-backed)
    IOSurfaceRef ioX;
    // GPU full
    id<MTLBuffer> bufW_full, bufY_full; MPSMatrix *mW_full,*mX,*mY_full; MPSMatrixMultiplication *mm_full;
} FFNMat;

int main(void) {
    mach_timebase_info(&g_tb);
    if (!init_ane_classes()) { fprintf(stderr,"ANE init failed\n"); return 1; }
    g_dev = MTLCreateSystemDefaultDevice();
    g_q = [g_dev newCommandQueue];

    const int WARM=8, IT=40;
    printf("Qwen2.5-0.5B FFN matmuls @ prefill seq=256. GPU(MPS fp16) || ANE(conv fp16), true overlap.\n");
    printf("%-16s | GPU-only | ANE-only | best split (ratio) -> wall | speedup vs GPU | aggregate TFLOPS\n", "matmul");
    printf("------------------------------------------------------------------------------------------------\n");

    struct { const char*name; int in_ch, out_ch; } shapes[] = {
        {"gate/up 896->4864", 896, 4864},
        {"down 4864->896",    4864, 896},
    };
    int S = 256;

    for (int si=0; si<2; si++) {
        int in_ch=shapes[si].in_ch, out_ch=shapes[si].out_ch;
        double gflop_full = 2.0*in_ch*out_ch*S/1e9;

        // shared X in IOSurface (fp16), wrapped for both MPS and ANE
        IOSurfaceRef ioX = create_surface((size_t)in_ch*S*2);
        id<MTLBuffer> bufX = buf_from_surface(g_dev, ioX);
        _Float16 *X = (_Float16*)IOSurfaceGetBaseAddress(ioX);
        for (size_t i=0;i<(size_t)in_ch*S;i++) X[i]=(_Float16)(((float)rand()/RAND_MAX-0.5f)*2.0f);
        MPSMatrix *mX = mk_mat(bufX, in_ch, S);

        // weights fp16 [out_ch, in_ch]
        _Float16 *W = malloc((size_t)out_ch*in_ch*2);
        for (size_t i=0;i<(size_t)out_ch*in_ch;i++) W[i]=(_Float16)(((float)rand()/RAND_MAX-0.5f)*0.1f);

        // ---- GPU-only full ----
        id<MTLBuffer> bufW=[g_dev newBufferWithBytes:W length:(size_t)out_ch*in_ch*2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufY=[g_dev newBufferWithLength:(size_t)out_ch*S*2 options:MTLResourceStorageModeShared];
        MPSMatrix *mW=mk_mat(bufW,out_ch,in_ch), *mY=mk_mat(bufY,out_ch,S);
        MPSMatrixMultiplication *mm=[[MPSMatrixMultiplication alloc] initWithDevice:g_dev transposeLeft:NO transposeRight:NO resultRows:out_ch resultColumns:S interiorColumns:in_ch alpha:1 beta:0];
        for(int i=0;i<WARM;i++){ id<MTLCommandBuffer> cb=[g_q commandBuffer]; [mm encodeToCommandBuffer:cb leftMatrix:mW rightMatrix:mX resultMatrix:mY]; [cb commit]; [cb waitUntilCompleted]; }
        uint64_t t0=mach_absolute_time();
        for(int i=0;i<IT;i++){ id<MTLCommandBuffer> cb=[g_q commandBuffer]; [mm encodeToCommandBuffer:cb leftMatrix:mW rightMatrix:mX resultMatrix:mY]; [cb commit]; [cb waitUntilCompleted]; }
        double gpu_full_us=ticks_us(mach_absolute_time()-t0)/IT;

        // ---- ANE-only full ----
        IOSurfaceRef ioYfull=create_surface((size_t)out_ch*S*2);
        ANEKernel *ane_full=compile_ane(in_ch,out_ch,S,(const uint16_t*)W,ioX,ioYfull);
        double ane_full_us=0;
        if(ane_full){ for(int i=0;i<WARM;i++)ane_eval(ane_full); uint64_t a0=mach_absolute_time(); for(int i=0;i<IT;i++)ane_eval(ane_full); ane_full_us=ticks_us(mach_absolute_time()-a0)/IT; }

        // ---- split sweep ----
        double best_us=gpu_full_us; double best_r=0;
        double ratios[]={0.2,0.3,0.4,0.5,0.6,0.7};
        for(int ri=0; ri<6; ri++){
            double r=ratios[ri];
            int ane_out=round16((int)(r*out_ch)); if(ane_out>=out_ch) ane_out=out_ch-16;
            int gpu_out=out_ch-ane_out;
            // GPU shard uses first gpu_out rows of W
            id<MTLBuffer> bWg=[g_dev newBufferWithBytes:W length:(size_t)gpu_out*in_ch*2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bYg=[g_dev newBufferWithLength:(size_t)gpu_out*S*2 options:MTLResourceStorageModeShared];
            MPSMatrix *mWg=mk_mat(bWg,gpu_out,in_ch), *mYg=mk_mat(bYg,gpu_out,S);
            MPSMatrixMultiplication *mmg=[[MPSMatrixMultiplication alloc] initWithDevice:g_dev transposeLeft:NO transposeRight:NO resultRows:gpu_out resultColumns:S interiorColumns:in_ch alpha:1 beta:0];
            // ANE shard uses last ane_out rows of W
            IOSurfaceRef ioYa=create_surface((size_t)ane_out*S*2);
            ANEKernel *ane=compile_ane(in_ch,ane_out,S,(const uint16_t*)(W+(size_t)gpu_out*in_ch),ioX,ioYa);
            if(!ane){ continue; }
            // warmup
            for(int i=0;i<WARM;i++){ id<MTLCommandBuffer> cb=[g_q commandBuffer]; [mmg encodeToCommandBuffer:cb leftMatrix:mWg rightMatrix:mX resultMatrix:mYg]; [cb commit]; ane_eval(ane); [cb waitUntilCompleted]; }
            uint64_t s0=mach_absolute_time();
            for(int i=0;i<IT;i++){
                id<MTLCommandBuffer> cb=[g_q commandBuffer];
                [mmg encodeToCommandBuffer:cb leftMatrix:mWg rightMatrix:mX resultMatrix:mYg];
                [cb commit];          // GPU starts async
                ane_eval(ane);        // ANE runs concurrently (blocks CPU ~T_ane)
                [cb waitUntilCompleted]; // join
            }
            double split_us=ticks_us(mach_absolute_time()-s0)/IT;
            if(split_us<best_us){ best_us=split_us; best_r=(double)ane_out/out_ch; }
            printf("   r=%.2f (ane_out=%d gpu_out=%d): %.1f us\n", r, ane_out, gpu_out, split_us);
        }
        double speedup=gpu_full_us/best_us;
        double agg_tflops=gflop_full/(best_us/1e6)/1e3;
        printf("%-16s | %7.1f  | %7.1f  | r=%.2f -> %.1f us | %.2fx | %.2f TFLOPS\n\n",
               shapes[si].name, gpu_full_us, ane_full_us, best_r, best_us, speedup, agg_tflops);
        free(W);
    }
    return 0;
}
