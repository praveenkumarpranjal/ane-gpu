"""Showcase: accelerate a real MLX Qwen2.5-0.5B with ANE+GPU and measure throughput.

Usage: python demo_qwen.py [batch] [ane_frac]
  batch=1  -> single-stream (ANE gated off by default; no regression)
  batch=16 -> batched/serving throughput (ANE+GPU run concurrently -> speedup)
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx
from mlx_lm import load
import ane_gpu

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
B = int(sys.argv[1]) if len(sys.argv) > 1 else 16
FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
S = 256

print(f"Loading {MODEL} ...")
model, tok = load(MODEL); model.set_dtype(mx.float16); mx.eval(model.parameters())
ids = tok.encode("The history of computing spans many disciplines. " * 80)[:S]
x = mx.array([ids] * B)                      # [B, S]
toks = B * S
print(f"workload: batch={B} x seq={S} = {toks} tokens/forward")

def bench(fn, it=8, w=3):
    for _ in range(w): fn()
    t0 = time.perf_counter()
    for _ in range(it): fn()
    return (time.perf_counter() - t0) / it

t_base = bench(lambda: mx.eval(model(x)))
logits_base = np.array(model(x)[:, -1].astype(mx.float32))

# engage ANE for this workload (min_seq low so the demo exercises it even at B=1)
model = ane_gpu.accelerate(model, ane_frac=FRAC, min_seq=64)
mx.eval(model(x))                            # warm/compile ANE kernels
t_acc = bench(lambda: mx.eval(model(x)))
logits_acc = np.array(model(x)[:, -1].astype(mx.float32))

am_b, am_a = logits_base[0].argmax(), logits_acc[0].argmax()
rel = np.abs(logits_acc - logits_base).max() / (np.abs(logits_base).max() + 1e-6)
print(f"\nGPU-only : {t_base*1e3:7.1f} ms  ({toks/t_base:7.0f} tok/s)")
print(f"ANE+GPU  : {t_acc*1e3:7.1f} ms  ({toks/t_acc:7.0f} tok/s)")
print(f"SPEEDUP  : {t_base/t_acc:.2f}x")
print(f"correct  : argmax {'MATCH' if am_b==am_a else 'DIFFER'}, logit rel|Δ|={rel:.4f}")
