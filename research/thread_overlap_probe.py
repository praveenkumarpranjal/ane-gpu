"""DE-RISK for true parallelism: can the GPU (MLX) and the ANE (ctypes) run CONCURRENTLY
on two threads? This is the primitive the whole request-pipeline depends on.

Runs GPU attention-projection matmuls (batch B) on the main thread while the ANE computes
a fused FFN (batch A, independent) on a worker thread. If wall ≈ max(gpu, ane) -> they
overlap (pipeline viable). If wall ≈ gpu + ane -> the runtime serialized them.
"""
import sys, os, time, threading
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx
from mlx_lm import load
from anegpu import _native as ane

B, S = 16, 256
N = B * S
model, tok = load("Qwen/Qwen2.5-0.5B-Instruct"); model.set_dtype(mx.float16); mx.eval(model.parameters())
L0 = model.model.layers[0]
dim = model.args.hidden_size; H = model.args.intermediate_size
aneh = (int(0.7 * H) // 16) * 16

# GPU side: real attention projection weights (batch B input)
Wq = mx.contiguous(L0.self_attn.q_proj.weight.T)
Wk = mx.contiguous(L0.self_attn.k_proj.weight.T)
Wv = mx.contiguous(L0.self_attn.v_proj.weight.T)
Wo = mx.contiguous(L0.self_attn.o_proj.weight.T)
xB = mx.random.normal((N, dim)).astype(mx.float16); mx.eval(xB)

# ANE side: fused FFN over the ANE's hidden slice (batch A input)
def npw(w): return np.ascontiguousarray(np.array(w.astype(mx.float16)))
W1 = npw(L0.mlp.gate_proj.weight)[:aneh]; W3 = npw(L0.mlp.up_proj.weight)[:aneh]
W2 = np.ascontiguousarray(npw(L0.mlp.down_proj.weight)[:, :aneh])
ffn = ane.compile_ffn(dim, aneh, N, W1, W3, W2)
xA = np.ascontiguousarray((np.random.randn(dim, N) * 0.5).astype(np.float16))

def gpu_attn():
    q = xB @ Wq; k = xB @ Wk; v = xB @ Wv; o = q @ Wo
    mx.eval([q, k, v, o])

def ane_ffn():
    ffn.inbuf[:] = xA
    ffn.run()

# warm
for _ in range(5): gpu_attn(); ane_ffn()

R = 30
def bench(fn):
    t0 = time.perf_counter()
    for _ in range(R): fn()
    return (time.perf_counter() - t0) / R * 1e3

t_gpu = bench(gpu_attn)
t_ane = bench(ane_ffn)

# overlapped: ANE on a worker thread, GPU on main, per iteration
t0 = time.perf_counter()
for _ in range(R):
    th = threading.Thread(target=ane_ffn); th.start()
    gpu_attn()
    th.join()
t_overlap = (time.perf_counter() - t0) / R * 1e3

serial = t_gpu + t_ane
print(f"GPU attention (proj matmuls): {t_gpu:6.2f} ms")
print(f"ANE fused FFN               : {t_ane:6.2f} ms")
print(f"SERIAL (one then the other) : {serial:6.2f} ms")
print(f"OVERLAPPED (2 threads)      : {t_overlap:6.2f} ms")
print(f"\n>>> overlap efficiency: {serial/t_overlap:.2f}x   (ideal = {serial/max(t_gpu,t_ane):.2f}x at full overlap)")
print(f">>> VERDICT: {'TRUE OVERLAP — both engines run at once, pipeline is viable' if t_overlap < 0.85*serial else 'SERIALIZED — threads did not overlap'}")
ffn.free()
