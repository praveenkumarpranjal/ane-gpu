# anegpu — run MLX model FFNs on the Apple Neural Engine **and** GPU at the same time

`anegpu` is a drop-in accelerator for [MLX](https://github.com/ml-explore/mlx) /
[mlx-lm](https://github.com/ml-explore/mlx-lm) models on Apple Silicon. One call —
`anegpu.accelerate(model)` — rewrites each transformer block's SwiGLU FFN so that it
runs **concurrently across the Apple Neural Engine and the GPU**, treating the two
engines as one. The ANE computes a fused sub-FFN over part of the hidden dimension while
the GPU (MLX) computes the rest; the partial results are summed.

```python
from mlx_lm import load
import anegpu

model, tok = load("Qwen/Qwen2.5-0.5B-Instruct")
model = anegpu.accelerate(model)           # FFNs now run on ANE + GPU together
# ... use mlx_lm.generate / model(...) exactly as before
```

It is numerically transparent: on Qwen2.5-0.5B the accelerated model reproduces the
original logits (argmax match, top-5 5/5, max relative logit error ~0.009).

## When it helps (read this — the honest part)

The Neural Engine adds **compute**, not memory bandwidth, and a transformer's forward
pass is a serial chain. So ANE+GPU only run *usefully in parallel* when there is enough
independent compute per layer to hide the hand-off. Concretely:

| Workload | What happens | Result (Qwen2.5-0.5B, M4) |
|---|---|---|
| **Batched serving / large prefill** (≳4k tokens in flight) | ANE and GPU both stay busy on the FFN | **~1.1–1.2× throughput, grows with batch** |
| **Single-stream decode** (1 token at a time) | bandwidth-bound; ANE can't help | falls back to GPU (no change) |
| **Short single prompt** (<1k tokens) | hand-off sync costs more than it saves | falls back to GPU automatically |

`accelerate()` therefore **only engages the ANE when the token count per forward pass is
large enough to win** (default `min_seq=1024`); below that it runs the original GPU path,
so there is no regression for chat-style single-stream use.

Measured throughput — **thermally-fair, interleaved** A/B (`ane_frac=0.6`, the honest way
to measure; naive baseline-then-accel inflates the ratio because the second run is hotter):

```
 256 tok (B=1)  : <1x          -> gated to GPU by default (no regression)
4096 tok (B=16) : 1.3x - 1.47x  (cool -> sustained load)
8192 tok (B=32) : 1.3x - 1.50x  (~3.2k tok/s prefill; ratio rises as the GPU throttles)
```

A **real** throughput gain in the batched/serving regime, not a universal 2–5× speedup.
The biggest hand-off cost was the CPU transpose of the activation (3.8 ms batched vs 22 µs
on GPU) — done with `mx.transpose` it lifted the win from ~1.1× to ~1.3×. The remaining
ceiling is Amdahl (attention stays on GPU) plus the unavoidable per-layer sync. See
`../benchmarks/` for the full analysis.

## Requirements / caveats

- **Apple Silicon only** (tested on M4, macOS 26). Uses **private** `AppleNeuralEngine`
  framework APIs (`_ANEInMemoryModel`, `_ANEIOSurfaceObject`) — research/enthusiast use,
  not App-Store-safe, and may break across macOS updates.
- Non-quantized fp16 SwiGLU MLPs (gate/up/down `nn.Linear`). Quantized layers are skipped.
- Best for **throughput** (serving multiple requests, batched/long prefill), not
  single-stream latency.

## Build

```bash
# from the repo root:
cd native && make        # builds anegpu/libanegpu.dylib
pip install mlx mlx-lm
```

## How it works

1. `accelerate()` walks `model.model.layers` and swaps each `.mlp` for a `SplitMLP`.
2. `SplitMLP` splits the hidden dimension: the ANE gets a fused MIL program for
   `down( silu(gate(x)) * up(x) )` over `ane_frac` of the channels; the GPU (MLX) computes
   the rest.
3. Per call: the GPU sub-FFN is dispatched with `mx.async_eval` (non-blocking) and the ANE
   sub-FFN runs concurrently via the native `libanegpu` bridge; the two partials are summed.
4. The ANE matmuls are 1×1 convs over a `[1, C, 1, seq]` fp16 tensor; **seq is padded to a
   multiple of 32** (an ANE tiling requirement — multiples of 16 are not sufficient).

## Tuning

- `ane_frac` (default 0.7): fraction of the FFN hidden dim given to the ANE. Lower values
  give the GPU more of the (efficient, batched) FFN; ~0.5–0.7 is the sweet spot on M4.
- `min_seq` (default 1024): minimum tokens-per-forward before the ANE engages.
- `anegpu.set_enabled(False)` toggles the ANE off at runtime (used by the fair benchmark).

## Files

```
anegpu/             this package: accelerate.py, _native.py, libanegpu.dylib
native/ane_ffn.m    the ANE bridge C source -> anegpu/libanegpu.dylib
benchmarks/         benchmark.py (fair A/B), demo.py, breakdown.py
```
