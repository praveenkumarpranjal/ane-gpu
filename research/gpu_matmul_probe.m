// gpu_matmul_probe.m — what is the HONEST GPU (MPS fp16) matmul throughput on this M4?
// Separates per-commit CPU<->GPU latency from real kernel time, so we don't compare
// ANE against a crippled GPU baseline.
//
// Build: xcrun clang -O2 -fobjc-arc -framework Foundation -framework Metal \
//          -framework MetalPerformanceShaders -o gpu_matmul_probe gpu_matmul_probe.m

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>

static mach_timebase_info_data_t g_tb;
static double ticks_us(uint64_t t){ return (double)t*g_tb.numer/g_tb.denom/1000.0; }

static MPSMatrix* mk(id<MTLBuffer> b,int r,int c){
    int rb=((c*2+15)/16)*16;
    return [[MPSMatrix alloc] initWithBuffer:b descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:r columns:c rowBytes:rb dataType:MPSDataTypeFloat16]];
}

static void probe(id<MTLDevice> dev,id<MTLCommandQueue> q,const char*name,int M,int K,int N){
    id<MTLBuffer> bW=[dev newBufferWithLength:(size_t)M*K*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bX=[dev newBufferWithLength:(size_t)K*N*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bY=[dev newBufferWithLength:(size_t)M*N*2 options:MTLResourceStorageModeShared];
    _Float16 *W=(_Float16*)bW.contents,*X=(_Float16*)bX.contents;
    for(size_t i=0;i<(size_t)M*K;i++)W[i]=(_Float16)(((float)rand()/RAND_MAX-0.5f)*0.1f);
    for(size_t i=0;i<(size_t)K*N;i++)X[i]=(_Float16)(((float)rand()/RAND_MAX-0.5f)*2.0f);
    MPSMatrix *mW=mk(bW,M,K),*mX=mk(bX,K,N),*mY=mk(bY,M,N);
    MPSMatrixMultiplication *mm=[[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:NO resultRows:M resultColumns:N interiorColumns:K alpha:1 beta:0];
    double gflop=2.0*M*K*N/1e9;
    int WARM=10,IT=50;

    // A: per-iter commit+wait
    for(int i=0;i<WARM;i++){id<MTLCommandBuffer> cb=[q commandBuffer];[mm encodeToCommandBuffer:cb leftMatrix:mW rightMatrix:mX resultMatrix:mY];[cb commit];[cb waitUntilCompleted];}
    uint64_t t0=mach_absolute_time();
    for(int i=0;i<IT;i++){id<MTLCommandBuffer> cb=[q commandBuffer];[mm encodeToCommandBuffer:cb leftMatrix:mW rightMatrix:mX resultMatrix:mY];[cb commit];[cb waitUntilCompleted];}
    double a=ticks_us(mach_absolute_time()-t0)/IT;

    // B: IT matmuls in ONE command buffer, commit once (amortizes commit; measures kernel throughput)
    { id<MTLCommandBuffer> cb=[q commandBuffer]; for(int i=0;i<WARM;i++)[mm encodeToCommandBuffer:cb leftMatrix:mW rightMatrix:mX resultMatrix:mY]; [cb commit];[cb waitUntilCompleted]; }
    id<MTLCommandBuffer> cb=[q commandBuffer];
    for(int i=0;i<IT;i++)[mm encodeToCommandBuffer:cb leftMatrix:mW rightMatrix:mX resultMatrix:mY];
    uint64_t t1=mach_absolute_time(); [cb commit];[cb waitUntilCompleted];
    double b=ticks_us(mach_absolute_time()-t1)/IT;

    // C: pipelined — commit IT separate cmd buffers without waiting, wait only the last
    NSMutableArray *cbs=[NSMutableArray array];
    uint64_t t2=mach_absolute_time();
    id<MTLCommandBuffer> last=nil;
    for(int i=0;i<IT;i++){id<MTLCommandBuffer> c=[q commandBuffer];[mm encodeToCommandBuffer:c leftMatrix:mW rightMatrix:mX resultMatrix:mY];[c commit];last=c;}
    [last waitUntilCompleted];
    double c=ticks_us(mach_absolute_time()-t2)/IT;

    printf("%-18s M=%-5d K=%-5d N=%-4d (%.2f GFLOP) | A per-commit %.1fus(%.2fTF) | B batched %.1fus(%.2fTF) | C pipelined %.1fus(%.2fTF)\n",
        name,M,K,N,gflop, a,gflop/(a/1e6)/1e3, b,gflop/(b/1e6)/1e3, c,gflop/(c/1e6)/1e3);
}

int main(void){
    mach_timebase_info(&g_tb);
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q=[dev newCommandQueue];
    printf("GPU: %s\n", [[dev name] UTF8String]);
    probe(dev,q,"gate/up s256",4864,896,256);
    probe(dev,q,"down s256",   896,4864,256);
    probe(dev,q,"gate/up s512",4864,896,512);
    probe(dev,q,"gate/up s128",4864,896,128);
    probe(dev,q,"gate/up s1(decode)",4864,896,1);
    return 0;
}
