"""Benchmark + verify anegpu.PipelinedRunner (2-stage GPU||ANE pipeline) on a real model.

Usage: python pipeline.py [batch]   (batch must be even, >=2)
Compares: GPU-only forward vs accelerate() FFN-only vs PipelinedRunner.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx
from mlx_lm import load
import anegpu

B = int(sys.argv[1]) if len(sys.argv) > 1 else 32
S = 256
MODEL = os.environ.get("ANEGPU_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
model, tok = load(MODEL); model.set_dtype(mx.float16); mx.eval(model.parameters())
ids = tok.encode("The history of computing spans many disciplines. " * 80)[:S]
x = mx.array([ids] * B)
toks = B * S
print(f"{MODEL}  batch={B}x{S} = {toks} tokens\n")

runner = anegpu.PipelinedRunner(model)

# ---- correctness: pipelined logits == GPU-only logits ----
ref = np.array(model(x)[:, -1].astype(mx.float32))
mx.eval(runner(x))                              # warm/compile
got = np.array(runner(x)[:, -1].astype(mx.float32))
match = sum(int(ref[i].argmax() == got[i].argmax()) for i in range(B))
relmax = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-6)
print(f"correctness: {match}/{B} rows argmax MATCH, max rel|Δ|={relmax:.4f}\n")

# thermally-fair: interleave GPU-only and the pipeline on ONE model (a 2nd model + its
# ANE kernels would cause memory pressure and skew the result)
for _ in range(2): mx.eval(model(x)); mx.eval(runner(x))      # warm
tg = tp = 0.0; R = 8
for _ in range(R):
    t0 = time.perf_counter(); mx.eval(model(x));  tg += time.perf_counter() - t0
    t0 = time.perf_counter(); mx.eval(runner(x)); tp += time.perf_counter() - t0
tg /= R; tp /= R
print(f"{'GPU-only':28}: {tg*1e3:7.1f} ms   {toks/tg:7.0f} tok/s")
print(f"{'PipelinedRunner (overlap)':28}: {tp*1e3:7.1f} ms   {toks/tp:7.0f} tok/s   ({tg/tp:.2f}x)")
runner.free()
