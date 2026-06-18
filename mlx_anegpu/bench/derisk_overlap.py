"""De-risk: from MLX, run one SwiGLU FFN three ways and compare correctness + speed:
  (1) pure MLX (GPU only)
  (2) ANE only (fused FFN)
  (3) concurrent split: MLX GPU sub-FFN  ||  ANE sub-FFN  (overlapped via mx.async_eval)

This is the proof that ANE+GPU parallelism actually beats GPU-only from inside MLX,
including the real transpose/handoff/join overhead.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mlx.core as mx
from ane_gpu import _bridge as ane

DIM, HIDDEN, SEQ = 896, 4864, 256          # Qwen2.5-0.5B FFN, prefill chunk
assert SEQ % 16 == 0

rng = np.random.default_rng(0)
def rnd(*shape, s=1.0):
    return (rng.standard_normal(shape).astype(np.float32) * s).astype(np.float16)

W1 = rnd(HIDDEN, DIM, s=0.05)   # gate
W3 = rnd(HIDDEN, DIM, s=0.05)   # up
W2 = rnd(DIM, HIDDEN, s=0.03)   # down
X  = rnd(SEQ, DIM, s=1.0)       # activations [seq, dim]

mW1, mW3, mW2 = mx.array(W1), mx.array(W3), mx.array(W2)
mX = mx.array(X)

def mlx_ffn(x, w1, w3, w2):
    g = x @ w1.T
    u = x @ w3.T
    h = (g * mx.sigmoid(g)) * u
    return h @ w2.T

# ---- reference ----
y_ref = mlx_ffn(mX, mW1, mW3, mW2); mx.eval(y_ref)
y_ref_np = np.array(y_ref).astype(np.float32)
ref_mag = np.abs(y_ref_np).mean()

def err(y):
    return np.abs(np.array(y).astype(np.float32) - y_ref_np).max()

def bench(fn, iters=40, warm=8):
    for _ in range(warm): fn()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    return (time.perf_counter() - t0) / iters * 1e6  # us

# ---- (1) pure MLX ----
def f_mlx():
    y = mlx_ffn(mX, mW1, mW3, mW2); mx.eval(y); return y
t_mlx = bench(f_mlx)
print(f"(1) pure MLX (GPU only)   : {t_mlx:8.1f} us   err={err(f_mlx()):.4f} (ref_mag={ref_mag:.3f})")

# ---- (2) ANE only (full fused FFN) ----
k_full = ane.compile_ffn(DIM, HIDDEN, SEQ, W1, W3, W2)
Xt = np.ascontiguousarray(X.T)  # [dim, seq]
def f_ane():
    k_full.inbuf[:] = Xt
    k_full.run()
    return k_full.outbuf  # [dim, seq]
t_ane = bench(f_ane)
y_ane_full = mx.array(np.ascontiguousarray(f_ane().T))
print(f"(2) ANE only (fused FFN)  : {t_ane:8.1f} us   err={err(y_ane_full):.4f}")

# ---- (3) concurrent split sweep ----
print("(3) concurrent split  MLX(GPU) || ANE :")
best = (t_mlx, 0.0, 0.0)
for frac in (0.4, 0.5, 0.6, 0.7, 0.8):
    ane_h = max(16, int(round(frac * HIDDEN / 16)) * 16)
    ane_h = min(ane_h, HIDDEN - 16)
    gpu_h = HIDDEN - ane_h
    # ANE gets the first ane_h hidden channels; GPU the rest
    k = ane.compile_ffn(DIM, ane_h, SEQ, W1[:ane_h], W3[:ane_h], W2[:, :ane_h])
    mW1g, mW3g = mx.array(W1[ane_h:]), mx.array(W3[ane_h:])
    mW2g = mx.array(np.ascontiguousarray(W2[:, ane_h:]))

    def split():
        k.inbuf[:] = Xt                       # hand x to ANE (zero-copy into IOSurface)
        g = mX @ mW1g.T; u = mX @ mW3g.T       # GPU sub-FFN
        h = (g * mx.sigmoid(g)) * u
        y_gpu = h @ mW2g.T
        mx.async_eval(y_gpu)                    # GPU starts, returns immediately
        k.run()                                # ANE runs concurrently (blocks CPU)
        y_ane = mx.array(np.ascontiguousarray(k.outbuf.T))
        mx.eval(y_gpu)                          # join
        return y_gpu + y_ane

    e = err(split())
    t = bench(split)
    flag = "OK" if e < 0.05 * ref_mag + 0.05 else "*** MISMATCH"
    print(f"    ane={frac:.0%} (ane_h={ane_h:4d} gpu_h={gpu_h:4d}): {t:8.1f} us  err={e:.4f} {flag}")
    if t < best[0] and "OK" in flag:
        best = (t, frac, e)
    k.free()

print()
print(f"BEST split: {best[0]:.1f} us @ ane={best[1]:.0%}")
print(f"  speedup vs pure MLX : {t_mlx/best[0]:.2f}x")
print(f"  speedup vs ANE-only : {t_ane/best[0]:.2f}x")
k_full.free()
