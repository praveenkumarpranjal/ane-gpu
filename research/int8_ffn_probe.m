// int8_ffn_probe.m — DE-RISK: does an int8-WEIGHT fused FFN on the ANE eval FASTER than
// fp16 (i.e. does the ANE stream fewer weight bytes per eval = a real bandwidth win),
// or does constexpr_affine_dequantize just expand to fp16 at load (no win)?
//
// Isolates the weight-bandwidth effect: int8 weights, fp16 activations (NO quant round-trip).
// Compares the SAME quantized weights run as int8 (constexpr_affine_dequantize) vs fp16.
//
// Build (from research/):
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl -o int8_ffn_probe int8_ffn_probe.m

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
typedef struct { id model, request; NSString *tmpDir; } K;
static Class gD,gM,gR,gI;
static bool ane_init(void){
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    gD=NSClassFromString(@"_ANEInMemoryModelDescriptor"); gM=NSClassFromString(@"_ANEInMemoryModel");
    gR=NSClassFromString(@"_ANERequest"); gI=NSClassFromString(@"_ANEIOSurfaceObject");
    return gD&&gM&&gR&&gI;
}
static IOSurfaceRef surf(size_t bytes){
    size_t ps=getpagesize(), as=((bytes+ps-1)/ps)*ps; if(as<ps)as=ps;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{(id)kIOSurfaceWidth:@(as),(id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(as),(id)kIOSurfaceAllocSize:@(as),(id)kIOSurfacePixelFormat:@0});
}
// PROVEN blob format (from fused_ffn_probe build_weight_blob_real): byte count @72,
// data_offset(128) @80, raw data @128. dtype comes from the MIL, not the blob.
static NSData *blob_fp16(const _Float16 *w, int out_ch, int in_ch){
    size_t ws=(size_t)out_ch*in_ch*2, total=128+ws; uint8_t *b=calloc(total,1);
    b[0]=0x01;b[4]=0x02; b[64]=0xEF;b[65]=0xBE;b[66]=0xAD;b[67]=0xDE; b[68]=0x01;
    *(uint32_t*)(b+72)=(uint32_t)ws; *(uint32_t*)(b+80)=128;
    memcpy(b+128,w,ws); return [NSData dataWithBytesNoCopy:b length:total freeWhenDone:YES];
}
static NSData *blob_int8(const int8_t *q, int out_ch, int in_ch){
    size_t ws=(size_t)out_ch*in_ch, total=128+ws; uint8_t *b=calloc(total,1);
    b[0]=0x01;b[4]=0x02; b[64]=0xEF;b[65]=0xBE;b[66]=0xAD;b[67]=0xDE; b[68]=0x01;
    *(uint32_t*)(b+72)=(uint32_t)ws; *(uint32_t*)(b+80)=128;
    memcpy(b+128,q,ws); return [NSData dataWithBytesNoCopy:b length:total freeWhenDone:YES];
}

// fused SwiGLU FFN, fp16 weights
static NSString *mil_fp16(int dim,int h,int S){
    return [NSString stringWithFormat:
      @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n  func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
      "    string vpt = const()[name=string(\"vpt\"), val=string(\"valid\")];\n    tensor<int32,[2]> st = const()[name=string(\"st\"), val=tensor<int32,[2]>([1,1])];\n"
      "    tensor<int32,[4]> pd = const()[name=string(\"pd\"), val=tensor<int32,[4]>([0,0,0,0])];\n    tensor<int32,[2]> dl = const()[name=string(\"dl\"), val=tensor<int32,[2]>([1,1])];\n    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
      "    tensor<fp16,[%d,%d,1,1]> W1 = const()[name=string(\"W1\"), val=tensor<fp16,[%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64)))];\n"
      "    tensor<fp16,[%d,%d,1,1]> W3 = const()[name=string(\"W3\"), val=tensor<fp16,[%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3.bin\"), offset=uint64(64)))];\n"
      "    tensor<fp16,[%d,%d,1,1]> W2 = const()[name=string(\"W2\"), val=tensor<fp16,[%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64)))];\n"
      "    tensor<fp16,[1,%d,1,%d]> gate = conv(dilations=dl, groups=gr, pad=pd, pad_type=vpt, strides=st, weight=W1, x=x)[name=string(\"gate\")];\n"
      "    tensor<fp16,[1,%d,1,%d]> up = conv(dilations=dl, groups=gr, pad=pd, pad_type=vpt, strides=st, weight=W3, x=x)[name=string(\"up\")];\n"
      "    tensor<fp16,[1,%d,1,%d]> sg = sigmoid(x=gate)[name=string(\"sg\")];\n    tensor<fp16,[1,%d,1,%d]> silu = mul(x=gate, y=sg)[name=string(\"silu\")];\n    tensor<fp16,[1,%d,1,%d]> hh = mul(x=silu, y=up)[name=string(\"hh\")];\n"
      "    tensor<fp16,[1,%d,1,%d]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=vpt, strides=st, weight=W2, x=hh)[name=string(\"y\")];\n  } -> (y);\n}\n",
      dim,S, h,dim,h,dim, h,dim,h,dim, dim,h,dim,h, h,S, h,S, h,S, h,S, h,S, dim,S];
}
// fused SwiGLU FFN, int8 weights via constexpr_affine_dequantize (sc1,sc3,sc2 = scales)
static NSString *mil_int8(int dim,int h,int S,double sc1,double sc3,double sc2){
    return [NSString stringWithFormat:
      @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n  func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
      "    string vpt = const()[name=string(\"vpt\"), val=string(\"valid\")];\n    tensor<int32,[2]> st = const()[name=string(\"st\"), val=tensor<int32,[2]>([1,1])];\n"
      "    tensor<int32,[4]> pd = const()[name=string(\"pd\"), val=tensor<int32,[4]>([0,0,0,0])];\n    tensor<int32,[2]> dl = const()[name=string(\"dl\"), val=tensor<int32,[2]>([1,1])];\n    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
      "    tensor<fp16,[%d,%d,1,1]> W1 = constexpr_affine_dequantize()[axis=int32(0), name=string(\"W1\"), quantized_data=tensor<int8,[%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64))), scale=fp16(%.9g), zero_point=int8(0)];\n"
      "    tensor<fp16,[%d,%d,1,1]> W3 = constexpr_affine_dequantize()[axis=int32(0), name=string(\"W3\"), quantized_data=tensor<int8,[%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3.bin\"), offset=uint64(64))), scale=fp16(%.9g), zero_point=int8(0)];\n"
      "    tensor<fp16,[%d,%d,1,1]> W2 = constexpr_affine_dequantize()[axis=int32(0), name=string(\"W2\"), quantized_data=tensor<int8,[%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64))), scale=fp16(%.9g), zero_point=int8(0)];\n"
      "    tensor<fp16,[1,%d,1,%d]> gate = conv(dilations=dl, groups=gr, pad=pd, pad_type=vpt, strides=st, weight=W1, x=x)[name=string(\"gate\")];\n"
      "    tensor<fp16,[1,%d,1,%d]> up = conv(dilations=dl, groups=gr, pad=pd, pad_type=vpt, strides=st, weight=W3, x=x)[name=string(\"up\")];\n"
      "    tensor<fp16,[1,%d,1,%d]> sg = sigmoid(x=gate)[name=string(\"sg\")];\n    tensor<fp16,[1,%d,1,%d]> silu = mul(x=gate, y=sg)[name=string(\"silu\")];\n    tensor<fp16,[1,%d,1,%d]> hh = mul(x=silu, y=up)[name=string(\"hh\")];\n"
      "    tensor<fp16,[1,%d,1,%d]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=vpt, strides=st, weight=W2, x=hh)[name=string(\"y\")];\n  } -> (y);\n}\n",
      dim,S, h,dim,h,dim,sc1, h,dim,h,dim,sc3, dim,h,dim,h,sc2, h,S, h,S, h,S, h,S, h,S, dim,S];
}

static K *compile(NSString *mil, NSDictionary *wd, NSArray *files, int dim, int S, IOSurfaceRef io_in, IOSurfaceRef io_out){
  @autoreleasepool{
    NSError *e=nil;
    NSData *md=[[mil dataUsingEncoding:NSUTF8StringEncoding] copy];
    id desc=((id(*)(Class,SEL,id,id,id))objc_msgSend)(gD,@selector(modelWithMILText:weights:optionsPlist:),md,wd,nil);
    if(!desc){fprintf(stderr,"desc fail\n");return NULL;}
    id mdl=((id(*)(Class,SEL,id))objc_msgSend)(gM,@selector(inMemoryModelWithDescriptor:),desc);
    if(!mdl){fprintf(stderr,"model fail\n");return NULL;}
    id hx=((id(*)(id,SEL))objc_msgSend)(mdl,@selector(hexStringIdentifier));
    NSString *td=[NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm=[NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    for(NSArray *p in files)[(NSData*)p[1] writeToFile:[td stringByAppendingPathComponent:p[0]] atomically:YES];
    if(!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(compileWithQoS:options:error:),21,@{},&e)){fprintf(stderr,"compile fail: %s\n",e?[[e description]UTF8String]:"?");return NULL;}
    if(!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(loadWithQoS:options:error:),21,@{},&e)){fprintf(stderr,"load fail\n");return NULL;}
    id wI=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(gI,@selector(objectWithIOSurface:),io_in);
    id wO=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(gI,@selector(objectWithIOSurface:),io_out);
    id req=((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(gR,@selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),@[wI],@[@0],@[wO],@[@0],nil,nil,@0);
    K *k=calloc(1,sizeof(K)); k->model=mdl;k->request=req;k->tmpDir=td; return k;
  }
}
static bool ev(K*k){ @autoreleasepool{ NSError*e=nil; return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(k->model,@selector(evaluateWithQoS:options:request:error:),21,@{},k->request,&e);} }
static double timed(K*k){ for(int i=0;i<10;i++)ev(k); uint64_t t0=mach_absolute_time(); int N=50; for(int i=0;i<N;i++)ev(k); return ticks_us(mach_absolute_time()-t0)/N; }

int main(void){
    mach_timebase_info(&g_tb);
    if(!ane_init()){fprintf(stderr,"ANE init fail\n");return 1;}
    int dim=896, h=4864, S=256;   // Qwen2.5-0.5B FFN @ prefill
    printf("INT8-weight vs FP16-weight fused FFN on ANE (Qwen2.5-0.5B: dim=%d h=%d S=%d)\n", dim,h,S);
    printf("weights: W1/W3 [%d,%d], W2 [%d,%d]  fp16=%.1fMB  int8=%.1fMB (half)\n\n",
        h,dim,dim,h, 3.0*h*dim*2/1048576.0, 3.0*h*dim*1/1048576.0);

    srand(1);
    int n1=h*dim, n2=dim*h;
    _Float16 *W1=malloc(n1*2),*W3=malloc(n1*2),*W2=malloc(n2*2);
    int8_t *Q1=malloc(n1),*Q3=malloc(n1),*Q2=malloc(n2);
    double s1=0,s3=0,s2=0;
    for(int i=0;i<n1;i++){ float v=((float)rand()/RAND_MAX-0.5f)*0.1f; W1[i]=(_Float16)v; if(fabs(v)>s1)s1=fabs(v); }
    for(int i=0;i<n1;i++){ float v=((float)rand()/RAND_MAX-0.5f)*0.1f; W3[i]=(_Float16)v; if(fabs(v)>s3)s3=fabs(v); }
    for(int i=0;i<n2;i++){ float v=((float)rand()/RAND_MAX-0.5f)*0.05f; W2[i]=(_Float16)v; if(fabs(v)>s2)s2=fabs(v); }
    s1/=127; s3/=127; s2/=127;
    for(int i=0;i<n1;i++){ int q=(int)lround((float)W1[i]/s1); Q1[i]=q<-128?-128:q>127?127:q; }
    for(int i=0;i<n1;i++){ int q=(int)lround((float)W3[i]/s3); Q3[i]=q<-128?-128:q>127?127:q; }
    for(int i=0;i<n2;i++){ int q=(int)lround((float)W2[i]/s2); Q2[i]=q<-128?-128:q>127?127:q; }

    IOSurfaceRef in1=surf((size_t)dim*S*2), out1=surf((size_t)dim*S*2);
    IOSurfaceRef in2=surf((size_t)dim*S*2), out2=surf((size_t)dim*S*2);
    _Float16 *x=(_Float16*)IOSurfaceGetBaseAddress(in1);
    for(int i=0;i<dim*S;i++)x[i]=(_Float16)(((float)rand()/RAND_MAX-0.5f)*2.0f);
    memcpy(IOSurfaceGetBaseAddress(in2), x, (size_t)dim*S*2);

    // fp16
    NSData *f1=blob_fp16(W1,h,dim),*f3=blob_fp16(W3,h,dim),*f2=blob_fp16(W2,dim,h);
    NSDictionary *wdf=@{@"@model_path/weights/w1.bin":@{@"offset":@0,@"data":f1},@"@model_path/weights/w3.bin":@{@"offset":@0,@"data":f3},@"@model_path/weights/w2.bin":@{@"offset":@0,@"data":f2}};
    NSArray *ff=@[@[@"weights/w1.bin",f1],@[@"weights/w3.bin",f3],@[@"weights/w2.bin",f2]];
    K *kf=compile(mil_fp16(dim,h,S), wdf, ff, dim, S, in1, out1);
    // int8
    NSData *q1=blob_int8(Q1,h,dim),*q3=blob_int8(Q3,h,dim),*q2=blob_int8(Q2,dim,h);
    NSDictionary *wdq=@{@"@model_path/weights/w1.bin":@{@"offset":@0,@"data":q1},@"@model_path/weights/w3.bin":@{@"offset":@0,@"data":q3},@"@model_path/weights/w2.bin":@{@"offset":@0,@"data":q2}};
    NSArray *fq=@[@[@"weights/w1.bin",q1],@[@"weights/w3.bin",q3],@[@"weights/w2.bin",q2]];
    K *kq=compile(mil_int8(dim,h,S,s1,s3,s2), wdq, fq, dim, S, in2, out2);

    if(!kf){printf("FP16 compile FAILED\n");}
    if(!kq){printf("INT8 compile FAILED\n");}
    double tf = kf? timed(kf):0, tq = kq? timed(kq):0;
    double gflop = 2.0*((double)dim*h*3)*S/1e9;
    printf("FP16 weights : %s\n", kf? "" : "(failed)");
    if(kf) printf("  %.1f us/eval   %.2f TFLOPS\n", tf, gflop/(tf/1e6)/1e3);
    printf("INT8 weights : %s\n", kq? "" : "(failed)");
    if(kq) printf("  %.1f us/eval   %.2f TFLOPS\n", tq, gflop/(tq/1e6)/1e3);
    if(kf&&kq){
        // correctness: compare int8 output to fp16 output
        _Float16 *yf=(_Float16*)IOSurfaceGetBaseAddress(out1), *yq=(_Float16*)IOSurfaceGetBaseAddress(out2);
        double maxabs=0,refmag=0;
        for(int i=0;i<dim*S;i++){ double d=fabs((float)yf[i]-(float)yq[i]); if(d>maxabs)maxabs=d; if(fabs((float)yf[i])>refmag)refmag=fabs((float)yf[i]); }
        printf("\n>>> INT8 vs FP16 eval speedup: %.2fx   (int8 correctness vs fp16: max_abs=%.3f refmag=%.2f)\n", tf/tq, maxabs, refmag);
        printf(">>> VERDICT: %s\n", (tf/tq > 1.15) ? "INT8 streams fewer bytes -> REAL bandwidth win, pursue it" :
                                    (tf/tq > 0.9)  ? "INT8 ~same as fp16 -> weights expanded at load, NO per-eval win" :
                                                     "INT8 SLOWER -> dequant overhead dominates");
    }
    return 0;
}
