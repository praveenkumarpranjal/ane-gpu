# anegpu — ANE + GPU LLM inference for MLX

Run MLX / mlx-lm model FFNs on the **Apple Neural Engine and GPU together** on Apple Silicon.
See the [top-level README](../README.md) for full results, findings, and the honest scorecard
of where this helps (batched throughput, long-prompt prefill) and where it can't (chat decode).

## Three entry points

```python
from mlx_lm import load
import mlx.core as mx
import anegpu

model, tok = load("Qwen/Qwen2.5-1.5B-Instruct"); model.set_dtype(mx.float16)
```

**1. `PipelinedRunner` — batched, true ANE‖GPU parallelism** (batch ≥ 2, ~1.85–2.5×)
```python
runner = anegpu.PipelinedRunner(model, int8=True)
logits = runner(tokens)          # tokens [B, S], B even >= 2; logits == model(tokens)
runner.free()
```
Splits the batch into two micro-batches offset by one stage: the ANE computes one batch's FFN
on a worker thread while the GPU computes the other's attention. MLX is touched only on the
main thread. `int8=True` for speed + to fit larger models.

**2. `SingleStreamRunner` — single long prompt** (batch = 1, ~2.3× prefill)
```python
runner = anegpu.SingleStreamRunner(model)
for tid in runner.generate(prompt_ids, max_tokens=256, eos_id=tok.eos_token_id):
    ...
# or:  logits = runner.prefill(mx.array([ids]), cache)
```
Runs each layer's attention on the GPU and its FFN on the ANE. Prefill only (decode stays MLX;
it's bandwidth-bound). Note: ANE kernels compile per length-bucket (~10 s, one-time) — best for
workloads with many long prompts. See the main README caveat.

**3. `accelerate(model)` — legacy drop-in** (`SplitMLP`; the runners supersede it)

## What's automatic

- **Architecture detection** — SiLU (Qwen/Llama/Mistral) vs GELU (Gemma) is auto-detected by
  matching the ANE kernel to the model's own `mlp()`; unsupported architectures transparently
  fall back to plain MLX (never crashes, never wrong).
- **int8 robustness** — per-output-channel scales + a per-layer fp16 fallback for the few
  layers whose int8 conv would overflow fp16.

## Layout

```
pipeline.py        PipelinedRunner          single_stream.py   SingleStreamRunner
accelerate.py      SplitMLP / accelerate()  _native.py         ctypes -> libanegpu.dylib
```

The kernel source is `../native/ane_ffn.m` (fused FFN: fp16/int8 × SiLU/GELU), built to
`libanegpu.dylib`. Apple Silicon only; uses private `AppleNeuralEngine` APIs.
