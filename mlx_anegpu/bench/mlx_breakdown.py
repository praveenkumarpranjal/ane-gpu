"""Honest breakdown: where does Qwen prefill time go, and is ANE actually faster than
MLX-GPU at the FFN in a FAIR pipelined comparison? Settles whether the split CAN win."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mlx.core as mx
from ane_gpu import _bridge as ane

DIM, HIDDEN = 896, 4864
def t_pipelined(make, iters=50, warm=10):
    # build `iters` independent outputs, async_eval all, ONE eval at end (no per-op sync)
    for _ in range(warm):
        o = make(); mx.eval(o)
    t0 = time.perf_counter()
    outs = [make() for _ in range(iters)]
    mx.eval(outs)
    return (time.perf_counter() - t0) / iters

for SEQ in (256, 512):
    print(f"\n===== seq={SEQ} =====")
    x = mx.random.normal((SEQ, DIM)).astype(mx.float16)
    W1 = mx.random.normal((DIM, HIDDEN)).astype(mx.float16)*0.05   # pre-T: x@W1 -> [seq,hidden]
    W3 = mx.random.normal((DIM, HIDDEN)).astype(mx.float16)*0.05
    W2 = mx.random.normal((HIDDEN, DIM)).astype(mx.float16)*0.03

    # MLX single matmul (gate/up shape) throughput
    def mm(): return x @ W1
    tmm = t_pipelined(mm)
    gf = 2*SEQ*DIM*HIDDEN/1e9
    print(f"MLX matmul gate/up   : {tmm*1e6:8.1f} us  ({gf/tmm/1e3:.2f} TFLOPS)")

    # MLX full FFN (gate+up+silu+down) throughput, pipelined
    def ffn():
        g = x @ W1; u = x @ W3
        return ((g*mx.sigmoid(g))*u) @ W2
    tffn = t_pipelined(ffn)
    gff = 2*(SEQ*DIM*HIDDEN*3)/1e9
    print(f"MLX FFN (3 matmuls)  : {tffn*1e6:8.1f} us  ({gff/tffn/1e3:.2f} TFLOPS)")

    # ANE fused FFN at same shape
    W1a = np.ascontiguousarray(np.array(W1.T))  # [hidden,dim]
    W3a = np.ascontiguousarray(np.array(W3.T))
    W2a = np.ascontiguousarray(np.array(W2.T))  # [dim,hidden]
    k = ane.compile_ffn(DIM, HIDDEN, SEQ, W1a, W3a, W2a)
    xt = np.ascontiguousarray(np.array(x.T))
    for _ in range(10):
        k.inbuf[:] = xt; k.run()
    t0=time.perf_counter()
    for _ in range(50):
        k.inbuf[:] = xt; k.run()
    tane=(time.perf_counter()-t0)/50
    print(f"ANE fused FFN        : {tane*1e6:8.1f} us  ({gff/tane/1e3:.2f} TFLOPS)")
    print(f"  -> ANE vs MLX FFN  : {tffn/tane:.2f}x  (>1 means ANE faster)")

    # overhead of the integration glue (transpose + copies), per call
    t0=time.perf_counter()
    for _ in range(50):
        xnp = np.array(x.astype(mx.float16))         # MLX->numpy
        k.inbuf[:, :SEQ] = xnp.T                      # transpose into IOSurface
        yb = np.ascontiguousarray(k.outbuf.T)         # transpose out
        ym = mx.array(yb); mx.eval(ym)                # numpy->MLX
    tover=(time.perf_counter()-t0)/50
    print(f"glue overhead/call   : {tover*1e6:8.1f} us  (transpose+copy+materialize)")
    k.free()
