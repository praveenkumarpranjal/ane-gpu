"""Thermally-fair A/B: same accelerated model + same workload, toggling the ANE on/off
and INTERLEAVING the two measurements so both see the same thermal drift. Reports median."""
import sys, os, time, statistics
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx
from mlx_lm import load
import ane_gpu
from ane_gpu import split_mlp

B = int(sys.argv[1]) if len(sys.argv) > 1 else 16
FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
S = 256
model, tok = load("Qwen/Qwen2.5-0.5B-Instruct"); model.set_dtype(mx.float16); mx.eval(model.parameters())
ids = tok.encode("The history of computing spans many disciplines. " * 80)[:S]
x = mx.array([ids] * B)
ane_gpu.accelerate(model, ane_frac=FRAC, min_seq=64, verbose=False)
mx.eval(model(x))  # warm/compile

def one():
    t0 = time.perf_counter(); mx.eval(model(x)); return time.perf_counter() - t0

# warm both paths
for _ in range(3):
    split_mlp.ENABLED = False; one(); split_mlp.ENABLED = True; one()

gpu, ane = [], []
for _ in range(10):                         # interleave: GPU, ANE, GPU, ANE, ...
    split_mlp.ENABLED = False; gpu.append(one())
    split_mlp.ENABLED = True;  ane.append(one())
gb, ab = statistics.median(gpu), statistics.median(ane)
toks = B * S
print(f"B={B} S={S} ({toks} tok) frac={FRAC}: "
      f"GPU {gb*1e3:.0f}ms ({toks/gb:.0f} tok/s) | ANE+GPU {ab*1e3:.0f}ms ({toks/ab:.0f} tok/s) | "
      f"median speedup {gb/ab:.2f}x")
