# ane-gpu — using the Apple Neural Engine **and** GPU together for LLM inference

Research + a working library for running LLM inference across the **Apple Neural Engine
(ANE) and the GPU concurrently** on Apple Silicon, treating the two engines as one chip.

The headline deliverable is [`mlx_anegpu`](mlx_anegpu/README.md): a drop-in MLX accelerator
you can apply to any mlx-lm model with one call.

```python
from mlx_lm import load
import ane_gpu

model, tok = load("Qwen/Qwen2.5-0.5B-Instruct")
model = ane_gpu.accelerate(model)     # FFNs now run on ANE + GPU together
```

## TL;DR — honest results (Qwen2.5-0.5B, M4)

- The ANE computes a fused SwiGLU FFN **~3.5× faster than MLX-GPU in isolation** (1501 µs
  vs 5328 µs @ seq 512), and the split is **numerically correct** (logits match the
  original: argmax match, top-5 5/5).
- **End-to-end it is a real but modest win, and only in the batched / throughput regime.**
  Thermally-fair A/B: **B=16 → 1.08×, B=32 → 1.17×** (grows with batch). Single-stream
  decode is bandwidth-bound and the per-layer lazy↔eager hand-off dominates, so
  `accelerate()` **gates the ANE off below `min_seq=1024`** — no regression for chat.
- It is **not** a universal 2–5× speedup. The ceiling is set by Amdahl (attention stays on
  the GPU) plus the cost of bridging lazy MLX to the eager ANE. See `mlx_anegpu/bench/` for
  the full, reproducible analysis.

## What's here

```
mlx_anegpu/        the product: ane_gpu.accelerate() + ANE bridge dylib + benchmarks
  ane_gpu/         the importable package (accelerate, SplitMLP, ctypes bridge)
  src/ane_ffn.m    the ANE bridge -> libane_ffn.dylib (fused FFN + matmul)
  bench/           ab_interleaved (the fair benchmark), demo_qwen, mlx_breakdown
  README.md        product docs + tuning

research/          self-contained ANE microbenchmarks (Objective-C) that established the facts:
  ane_ffn_probe.m       ANE conv correctness + the seq%32 tiling rule
  gpu_matmul_probe.m    honest MLX/MPS matmul throughput
  fused_ffn_probe.m     the whole SwiGLU FFN as one ANE program
  ffn_overlap_bench.m   GPU || ANE concurrent overlap
  iosurface_*, transformer_layer_*, ane_correctness_*  (earlier exploration + results)

qwen_split_infer.m    a standalone Objective-C Qwen2.5-0.5B engine (Metal + ANE split path)
export_qwen.py        export HF Qwen weights -> qwen_weights.bin (gitignored, ~1 GB)
ane_dim_sweep.py      ANE matmul TFLOPS sweep (uses bridge/)
hetero_ffn_bench.py   heterogeneous FFN split sweep (uses bridge/)
bridge/               C-callable ANE bridge (compile/eval + fp16/int8 weight blobs)
```

## Key technical facts discovered

- **ANE 1×1-conv-as-matmul requires `seq % 32 == 0`** for the fused FFN (a multiple of 16
  is *not* enough — `16×odd` silently returns zeros / a "Program Inference error"). This was
  the reason the ANE path had been disabled in the original engine.
- The ANE FFN is **weight-bandwidth-bound** (~230 µs floor to stream W1W3); it only pays off
  when weights amortize over many tokens — i.e. prefill / batch, never single-token decode.
- On an **M4 base**, MLX/MPS only reaches ~1.9–2.7 TFLOPS on FFN-shaped matmuls while the ANE
  reaches ~7–9 TFLOPS on the fused FFN — they are genuinely complementary, but the
  hand-off + serial transformer dependency limit the end-to-end gain.
- Measure **interleaved**, never baseline-then-accel: the second pass runs hotter and
  inflates the ratio (we saw bogus 1.24–1.51× that way).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd mlx_anegpu/src && make && cd ../..      # build libane_ffn.dylib
cd bridge && make && cd ..                 # build libane_bridge.dylib (for the python sweeps)

# reproduce the fair benchmark:
python mlx_anegpu/bench/ab_interleaved.py 16 0.6
```

## Caveats

Apple Silicon only (developed on M4, macOS 26). Uses **private** `AppleNeuralEngine`
framework APIs — research/enthusiast use, not App-Store-safe, may break across macOS
updates. The design rationale and methodology are documented in the per-module READMEs
and `mlx_anegpu/bench/`.

Built on the private-ANE-API approach from [maderix/ANE](https://github.com/maderix/ANE).
