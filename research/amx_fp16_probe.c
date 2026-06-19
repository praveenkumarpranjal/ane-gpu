// Fair AMX de-risk: fp16 GEMM via Accelerate/BNNS (uses the AMX fp16 path), the real
// inference dtype — vs the earlier numpy-fp32 (1.1 TFLOPS) which upcast and undersold it.
// out[N,M] = in[N,K] @ W[K,M], W stored [M,K] for the FC layer (y = W @ x per row).
// Build: xcrun clang -O2 -Wno-deprecated-declarations -framework Accelerate -o /tmp/amxp research/amx_fp16_probe.c
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>

static double now_s(void) {
  static mach_timebase_info_data_t tb;
  if (tb.denom == 0) mach_timebase_info(&tb);
  return mach_absolute_time() * (double)tb.numer / tb.denom / 1e9;
}

static double bench(int K, int M, int N, int iters, BNNSDataType dt) {
  size_t es = (dt == BNNSDataTypeFloat16) ? 2 : 4;
  void *W  = malloc(es*(size_t)M*K);
  void *in = malloc(es*(size_t)N*K);
  void *out= malloc(es*(size_t)N*M);
  if (dt == BNNSDataTypeFloat16) {
    __fp16 *w=W,*i=in; for (size_t x=0;x<(size_t)M*K;x++) w[x]=(__fp16)((rand()/(double)RAND_MAX-0.5)*0.1);
    for (size_t x=0;x<(size_t)N*K;x++) i[x]=(__fp16)((rand()/(double)RAND_MAX-0.5));
  } else {
    float *w=W,*i=in; for (size_t x=0;x<(size_t)M*K;x++) w[x]=(float)((rand()/(double)RAND_MAX-0.5)*0.1);
    for (size_t x=0;x<(size_t)N*K;x++) i[x]=(float)((rand()/(double)RAND_MAX-0.5));
  }
  BNNSNDArrayDescriptor id; memset(&id,0,sizeof(id));
  id.layout=BNNSDataLayoutVector; id.size[0]=(size_t)K; id.data_type=dt;
  BNNSNDArrayDescriptor od; memset(&od,0,sizeof(od));
  od.layout=BNNSDataLayoutVector; od.size[0]=(size_t)M; od.data_type=dt;
  BNNSNDArrayDescriptor wd; memset(&wd,0,sizeof(wd));
  wd.layout=BNNSDataLayoutRowMajorMatrix; wd.size[0]=(size_t)K; wd.size[1]=(size_t)M; wd.data=W; wd.data_type=dt;
  BNNSLayerParametersFullyConnected p; memset(&p,0,sizeof(p));
  p.i_desc=id; p.o_desc=od; p.w_desc=wd;
  p.activation = (BNNSActivation){ .function=BNNSActivationFunctionIdentity };
  BNNSFilter f = BNNSFilterCreateLayerFullyConnected(&p, NULL);
  if (!f) { printf("  FC create FAILED (K=%d M=%d)\n", K, M); free(W);free(in);free(out); return -1; }
  for (int w=0;w<5;w++) BNNSFilterApplyBatch(f, (size_t)N, in, (size_t)K, out, (size_t)M);
  double t0=now_s();
  for (int it=0; it<iters; it++) BNNSFilterApplyBatch(f, (size_t)N, in, (size_t)K, out, (size_t)M);
  double dt_s=(now_s()-t0)/iters;
  BNNSFilterDestroy(f); free(W);free(in);free(out);
  double gflop=2.0*M*K*N/1e9;
  printf("  K=%4d M=%4d N=%5d: %8.0f us  %5.2f TFLOPS\n", K, M, N, dt_s*1e6, gflop/dt_s/1e3);
  return gflop/dt_s/1e3;
}

int main(void) {
  int D=896,H=4864;
  printf("Accelerate/BNNS fp16 GEMM (AMX fp16 path):\n");
  for (int N=552; N<=4096; N*=2) { bench(D,H,N,30,BNNSDataTypeFloat16); }   // gate/up shape
  bench(H,D,4096,30,BNNSDataTypeFloat16);                                    // down shape
  printf("bf16 (AMX native low-precision):\n");
  bench(D,H,552,30,BNNSDataTypeBFloat16); bench(D,H,2208,30,BNNSDataTypeBFloat16); bench(H,D,4096,30,BNNSDataTypeBFloat16);
  printf("fp32 (cross-check vs numpy 1.1 TF):\n");
  bench(D,H,552,30,BNNSDataTypeFloat32); bench(D,H,4096,30,BNNSDataTypeFloat32); bench(H,D,4096,30,BNNSDataTypeFloat32);
  // full fp16 FFN cost for a 13.5%% token slice (gate+up+down)
  int Nc=552; double t0=now_s();
  // approximate via 3 GEMM benches summed is misleading; instead time the dominant two
  printf("(per-GEMM above; FFN slice = gate+up+down at its N)\n");
  return 0;
}
