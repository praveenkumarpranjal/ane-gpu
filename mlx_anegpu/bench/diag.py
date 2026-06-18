"""Single-shot: inner-model (no lm_head) baseline vs accelerated at one ane_frac.
Usage: python diag.py <ane_frac>   (e.g. 1.0)  — run once per process to avoid ANE compile pileup."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx
from mlx_lm import load
import ane_gpu

FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
model, tok = load("Qwen/Qwen2.5-0.5B-Instruct")
model.set_dtype(mx.float16); mx.eval(model.parameters())
ids = tok.encode(("The history of computing spans many disciplines. " * 60))[:512]
x = mx.array([ids]); N = x.shape[1]
inner = model.model

def bench(fn, it=12, w=3):
    for _ in range(w): fn()
    t0=time.perf_counter()
    for _ in range(it): fn()
    return (time.perf_counter()-t0)/it

t_base = bench(lambda: mx.eval(inner(x)))
ane_gpu.accelerate(model, ane_frac=FRAC, min_seq=32, verbose=False)
t_acc = bench(lambda: mx.eval(inner(x)))
print(f"N={N}  inner baseline {t_base*1e3:6.1f} ms | accel(frac={FRAC}) {t_acc*1e3:6.1f} ms | speedup {t_base/t_acc:.2f}x")
