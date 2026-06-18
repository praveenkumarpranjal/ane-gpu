// fused_ffn_probe.m — Can the ANE run a WHOLE SwiGLU FFN sub-block as ONE MIL program?
//   y = (silu(x@W1) * (x@W3)) @ W2   over a hidden slice (ane_h channels)
// Verifies compile + numerical correctness vs CPU + latency at prefill seq=256.
// This is "Design 1": each device computes a complete sub-FFN over its hidden slice,
// so only ONE handoff in (x) and ONE out (partial y) per layer — no per-op ping-pong.
//
// Build: xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//          -o fused_ffn_probe fused_ffn_probe.m

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static mach_timebase_info_data_t g_tb;
static double ticks_us(uint64_t t){ return (double)t*g_tb.numer/g_tb.denom/1000.0; }

typedef struct { id model; id request; NSString *tmpDir; } ANEKernel;
static Class g_ANEDesc, g_ANEInMem, g_ANEReq, g_ANEIO;
static bool init_ane(void){
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_ANEDesc=NSClassFromString(@"_ANEInMemoryModelDescriptor"); g_ANEInMem=NSClassFromString(@"_ANEInMemoryModel");
    g_ANEReq=NSClassFromString(@"_ANERequest"); g_ANEIO=NSClassFromString(@"_ANEIOSurfaceObject");
    return g_ANEDesc&&g_ANEInMem&&g_ANEReq&&g_ANEIO;
}
static IOSurfaceRef create_surface(size_t bytes){
    size_t ps=getpagesize(), as=((bytes+ps-1)/ps)*ps; if(as<ps)as=ps;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{(id)kIOSurfaceWidth:@(as),(id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(as),(id)kIOSurfaceAllocSize:@(as),(id)kIOSurfacePixelFormat:@0});
}
static uint8_t *blob(const uint16_t *w,int out_ch,int in_ch,size_t *len){
    int ws=out_ch*in_ch*2,total=128+ws; uint8_t *b=calloc(total,1);
    b[0]=0x01;b[4]=0x02;b[64]=0xEF;b[65]=0xBE;b[66]=0xAD;b[67]=0xDE;b[68]=0x01;
    *(uint32_t*)(b+72)=ws;*(uint32_t*)(b+80)=128; memcpy(b+128,w,ws); *len=total; return b;
}

// Fused SwiGLU FFN MIL: x16[1,dim,1,S] -> gate=conv(W1), up=conv(W3),
//   silu = gate*sigmoid(gate), h = silu*up, y = conv(h, W2) -> [1,dim,1,S]
static NSString *gen_mil_ffn(int dim,int h,int S){
    return [NSString stringWithFormat:
      @"program(1.3)\n"
      "[buildInfo = dict<string, string>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
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
      "    tensor<fp16,[1,%d,1,%d]> sg = sigmoid(x=gate)[name=string(\"sg\")];\n"
      "    tensor<fp16,[1,%d,1,%d]> silu = mul(x=gate, y=sg)[name=string(\"silu\")];\n"
      "    tensor<fp16,[1,%d,1,%d]> hh = mul(x=silu, y=up)[name=string(\"hh\")];\n"
      "    tensor<fp16,[1,%d,1,%d]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=vpt, strides=st, weight=W2, x=hh)[name=string(\"y\")];\n"
      "  } -> (y);\n}\n",
      dim,S, h,dim,h,dim, h,dim,h,dim, dim,h,dim,h,
      h,S, h,S, h,S, h,S, h,S, dim,S];
}

static ANEKernel *compile_ffn(int dim,int h,int S,const uint16_t*W1,const uint16_t*W3,const uint16_t*W2,
                              IOSurfaceRef ioIn,IOSurfaceRef ioOut){
  @autoreleasepool{
    NSError *e=nil;
    NSData *mil=[[gen_mil_ffn(dim,h,S) dataUsingEncoding:NSUTF8StringEncoding] copy];
    size_t l1,l3,l2; uint8_t *b1=blob(W1,h,dim,&l1),*b3=blob(W3,h,dim,&l3),*b2=blob(W2,dim,h,&l2);
    NSData *d1=[NSData dataWithBytesNoCopy:b1 length:l1 freeWhenDone:YES];
    NSData *d3=[NSData dataWithBytesNoCopy:b3 length:l3 freeWhenDone:YES];
    NSData *d2=[NSData dataWithBytesNoCopy:b2 length:l2 freeWhenDone:YES];
    id desc=((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_ANEDesc,@selector(modelWithMILText:weights:optionsPlist:),
        mil, @{@"@model_path/weights/w1.bin":@{@"offset":@0,@"data":d1},
               @"@model_path/weights/w3.bin":@{@"offset":@0,@"data":d3},
               @"@model_path/weights/w2.bin":@{@"offset":@0,@"data":d2}}, nil);
    if(!desc){fprintf(stderr,"desc failed\n");return NULL;}
    id mdl=((id(*)(Class,SEL,id))objc_msgSend)(g_ANEInMem,@selector(inMemoryModelWithDescriptor:),desc);
    if(!mdl){fprintf(stderr,"model failed\n");return NULL;}
    id hx=((id(*)(id,SEL))objc_msgSend)(mdl,@selector(hexStringIdentifier));
    NSString *td=[NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm=[NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [d1 writeToFile:[td stringByAppendingPathComponent:@"weights/w1.bin"] atomically:YES];
    [d3 writeToFile:[td stringByAppendingPathComponent:@"weights/w3.bin"] atomically:YES];
    [d2 writeToFile:[td stringByAppendingPathComponent:@"weights/w2.bin"] atomically:YES];
    if(!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(compileWithQoS:options:error:),21,@{},&e)){
        fprintf(stderr,"compile failed: %s\n", e?[[e description] UTF8String]:"?"); return NULL; }
    if(!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(loadWithQoS:options:error:),21,@{},&e)){
        fprintf(stderr,"load failed\n"); return NULL; }
    id wI=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO,@selector(objectWithIOSurface:),ioIn);
    id wO=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO,@selector(objectWithIOSurface:),ioOut);
    id req=((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_ANEReq,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI],@[@0],@[wO],@[@0],nil,nil,@0);
    ANEKernel *k=calloc(1,sizeof(ANEKernel)); k->model=mdl;k->request=req;k->tmpDir=td; return k;
  }
}
static bool ffn_eval(ANEKernel*k){ @autoreleasepool{ NSError*e=nil;
    return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(k->model,@selector(evaluateWithQoS:options:request:error:),21,@{},k->request,&e);} }

static void test(int dim,int h,int S){
    IOSurfaceRef ioIn=create_surface((size_t)dim*S*2), ioOut=create_surface((size_t)dim*S*2);
    _Float16 *W1=malloc((size_t)h*dim*2),*W3=malloc((size_t)h*dim*2),*W2=malloc((size_t)dim*h*2);
    for(size_t i=0;i<(size_t)h*dim;i++){W1[i]=(_Float16)(((float)rand()/RAND_MAX-0.5f)*0.1f);W3[i]=(_Float16)(((float)rand()/RAND_MAX-0.5f)*0.1f);}
    for(size_t i=0;i<(size_t)dim*h;i++)W2[i]=(_Float16)(((float)rand()/RAND_MAX-0.5f)*0.05f);
    _Float16 *x=(_Float16*)IOSurfaceGetBaseAddress(ioIn);
    for(size_t i=0;i<(size_t)dim*S;i++)x[i]=(_Float16)(((float)rand()/RAND_MAX-0.5f)*2.0f);

    uint64_t c0=mach_absolute_time();
    ANEKernel *k=compile_ffn(dim,h,S,(uint16_t*)W1,(uint16_t*)W3,(uint16_t*)W2,ioIn,ioOut);
    double cms=ticks_us(mach_absolute_time()-c0)/1000.0;
    if(!k){printf("dim=%d h=%d S=%d: COMPILE FAILED\n",dim,h,S); return;}
    for(int i=0;i<10;i++)ffn_eval(k);
    uint64_t t0=mach_absolute_time(); int IT=40; for(int i=0;i<IT;i++)ffn_eval(k);
    double us=ticks_us(mach_absolute_time()-t0)/IT;

    _Float16 *y=(_Float16*)IOSurfaceGetBaseAddress(ioOut);
    // CPU ref for a few (d,s)
    double maxabs=0,refmag=0; int CS=S<4?S:4;
    for(int s=0;s<CS;s++) for(int d=0;d<dim;d+= (dim/64>0?dim/64:1)){
        // gate,up over all h, then y[d]
        double yd=0;
        for(int o=0;o<h;o++){
            double g=0,u=0;
            for(int i=0;i<dim;i++){ float xv=(float)x[(size_t)i*S+s]; g+=xv*(float)W1[(size_t)o*dim+i]; u+=xv*(float)W3[(size_t)o*dim+i]; }
            double silu=g*(1.0/(1.0+exp(-g)));
            double hh=silu*u;
            yd+=hh*(float)W2[(size_t)d*h+o];
        }
        float got=(float)y[(size_t)d*S+s];
        double ae=fabs(got-yd); if(ae>maxabs)maxabs=ae; if(fabs(yd)>refmag)refmag=fabs(yd);
    }
    double gflop=2.0*((double)dim*h + (double)dim*h + (double)h*dim)*S/1e9; // gate+up+down
    printf("dim=%-4d h=%-4d S=%-4d | compile %.0fms | %.1f us/eval | %.2f TFLOPS | max_abs=%.3f (refmag=%.2f) %s\n",
        dim,h,S,cms,us,gflop/(us/1e6)/1e3,maxabs,refmag,(maxabs<0.05*refmag+0.02)?"OK":"*** MISMATCH");
    free(W1);free(W3);free(W2);
}

int main(void){
    mach_timebase_info(&g_tb);
    if(!init_ane()){fprintf(stderr,"ANE init failed\n");return 1;}
    printf("Fused SwiGLU FFN on ANE (one MIL program). Qwen2.5-0.5B: dim=896, hidden=4864.\n\n");
    test(896,4864,256);   // full FFN, prefill
    test(896,2912,256);   // 60%% hidden slice (ANE's share in a split)
    test(896,1952,256);   // 40%% hidden slice
    test(896,4864,128);   // full FFN, shorter prefill
    return 0;
}
