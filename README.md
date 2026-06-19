# ane-gpu — Apple Neural Engine + GPU together for LLM inference

A working library + research record for running MLX LLM inference across the **Apple Neural
Engine (ANE) and the GPU concurrently** on Apple Silicon, treating the two engines as one
fabric. Built and measured on an **M4 base (10-core, 16 GB, macOS 26/27)**.

The honest one-liner: **this accelerates the compute- and throughput-bound regimes (batched
inference, long-prompt prefill) by 1.5–2.5×. It does *not* speed up single-stream chat
decode — that's memory-bandwidth-bound and 4-bit MLX is already the floor.** This README is
explicit about both, because knowing *which regime you're in* is the whole game.

---

## Where it wins (and where it doesn't)

| Workload | Tool | Result on M4 |
|---|---|---|
| **Batched prefill / serving** (many sequences at once) | `PipelinedRunner` | **1.85–2.5×** over GPU-only |
| **Single long prompt → short output** (RAG, classify, extract, summarize) | `SingleStreamRunner` | **1.9–2.3×** prefill vs 4-bit GPU |
| **Interactive chat decode** (one token at a time) | — | no gain — already optimal at 4-bit |

The dividing line is **arithmetic intensity**. When each weight is reused across many tokens
(batch, or a long prefill), the work is compute-bound and the ANE's faster FFN wins. When each
weight is streamed for a *single* token (decode), the work is bandwidth-bound — and no compute
trick helps; the only lever is fewer bytes (quantization, already maxed at 4-bit).

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd native && make && cd ..        # builds anegpu/libanegpu.dylib
```

```python
from mlx_lm import load
import mlx.core as mx
import anegpu

model, tok = load("Qwen/Qwen2.5-1.5B-Instruct"); model.set_dtype(mx.float16)

# (1) Batched throughput — true ANE‖GPU parallelism across micro-batches
runner = anegpu.PipelinedRunner(model, int8=True)
logits = runner(tokens)            # tokens [B, S], B even >= 2  ->  ~1.85-2.5x

# (2) Single long prompt — ANE-offloaded FFN prefill
runner = anegpu.SingleStreamRunner(model)
for tid in runner.generate(prompt_ids, max_tokens=256):  # ~2.3x prefill on long prompts
    ...
```

Interactive chat with metrics (`prefill` / `decode` tok/s, latency, peak memory):

```bash
python3 chat.py                       # default: pre-quantized 4-bit (~1 GB, ~100 tok/s decode)
python3 chat.py --ane-prefill         # ANE prefill for prompts >= 256 tok (see caveat below)
python3 chat.py --max-tokens 2048 Qwen/Qwen2.5-1.5B-Instruct --4bit
```

---

## Measured results (M4 base)

**Batched pipeline (`PipelinedRunner`, Qwen2.5-0.5B, B=32, S=256):**

| Path | Throughput | vs GPU-only |
|---|---|---|
| GPU-only | 2509 tok/s | 1.00× |
| fp16 pipeline | 3782 tok/s | 1.51× |
| **int8 pipeline** | **4638 tok/s** | **1.85×** |

Qwen2.5-**1.5B** int8 pipeline: **2.46×** (1482 tok/s @ B=16). Numerically correct
(argmax matches the reference model).

**Single-stream prefill (`SingleStreamRunner`, Qwen2.5-1.5B), vs 4-bit GPU:**

| Prompt length | ANE-FFN | 4-bit GPU | speedup |
|---|---|---|---|
| 128 | 705 tok/s | 768 tok/s | 0.92× (too short) |
| 512 | 1423 tok/s | 750 tok/s | **1.90×** |
| 1024 | 1358 tok/s | 717 tok/s | **1.89×** |

First 12 generated tokens are **identical** to `mlx_lm` (greedy). Decode runs on plain MLX.

**Chat decode** (the thing you feel) is plain 4-bit MLX: **~100 tok/s, beating ollama's ~72–80**
on the same machine. The ANE is not involved — it can't beat 4-bit on a bandwidth-bound op.

> All numbers are **interleaved** A/B (alternating runs), never baseline-then-accel — the
> second pass runs hotter and inflates the ratio (we saw bogus 1.24–1.51× that way).

---

## How it works

The transformer layer is `h += attn(norm(h)); h += FFN(norm(h))`. The FFN is the bulk of both
the **compute** (prefill) and the **weight bandwidth** (decode). The ANE runs a **fused SwiGLU /
GeGLU FFN** ~3–4× faster than the GPU. Two ways to exploit that:

**`PipelinedRunner` (batch ≥ 2): true cross-engine parallelism.** Split the batch into two
micro-batches offset by one stage. While the ANE computes micro-batch A's FFN (on a ctypes
worker thread — releases the GIL), the GPU computes micro-batch B's attention (MLX, main
thread). Neither engine idles. MLX is touched **only** on the main thread.

**`SingleStreamRunner` (batch = 1): faster-engine offload.** No second stream to overlap, but
the ANE FFN is faster than the GPU FFN *even at B=1*, so each layer runs attention on the GPU
and the FFN on the ANE. Serial, but the faster engine still wins for prefill.

**The ANE FFN kernel** (`native/ane_ffn.m` → `libanegpu.dylib`): each matmul is a 1×1 conv over
a channel-first `[1, C, 1, seq]` fp16 tensor, fused as `gate → act → ×up → down`, via the
private `_ANEInMemoryModel` API. Supports:
- **fp16** and **int8** weights (int8 = `constexpr_affine_dequantize`, **per-output-channel**
  scales, streams ~half the bytes → ~1.5× and the memory to fit bigger models).
- **SiLU** (SwiGLU: Qwen/Llama/Mistral) and **GELU** (GeGLU: Gemma) — auto-detected.

**Robustness built in:**
- *Architecture auto-detect:* on first call the runner verifies the ANE kernel reproduces the
  model's own `mlp()` on a probe; SiLU vs GELU is picked automatically, and any unsupported
  architecture **transparently falls back to plain MLX** — never crashes, never silently wrong.
- *int8 overflow hybrid:* the ANE int8 conv accumulates in fp16; a few large-activation layers
  (e.g. Qwen-1.5B layer 1) overflow → a one-time calibration keeps int8 only where it's finite
  and accurate, else fp16 for that layer (typically 2–3 of 28).

---

## Key findings (incl. the negative results)

The research that established the above — many are *negative*, which is the point:

- **ANE 1×1-conv-as-matmul silently returns zeros unless `seq % 32 == 0`** (a multiple of 16 is
  *not* enough). This single bug is why the ANE path is usually written off as broken.
- **The ANE FFN is weight-bandwidth-bound** (~230 µs floor to stream the weights); it only pays
  off when weights amortize over many tokens — prefill / batch, never single-token decode.
- **The GPU-side hand-off matters more than the compute.** The original killer was a CPU
  transpose of the activation (3.8 ms for `[8192,896]`) vs **22 µs on the GPU** — use
  `mx.transpose`, never numpy. This fix is what turned single-stream from 0.45× into ~2×.
- **Stage balancing (split the FFN across ANE+GPU) is ≈ neutral** — the GPU FFN is too slow to
  use its idle slack profitably.
- **The CPU's AMX is too weak to be a 3rd FFN worker.** Measured across *every* dtype: fp16/bf16
  ~1.3–1.5 TF (slower than fp32!), fp32 ~1.5–1.7 TF, int8 ~1.8 TF — all ~7–8× slower than the
  int8 ANE. The real 3-way overlap made a layer **1.4× slower** (core/dispatch contention).
- **Attention is not the floor (yet).** With int8 the FFN (8.5 ms/layer) is still the
  bottleneck vs attention (6.8 ms) — so attention optimization wouldn't raise throughput.
- **Decode can't use the ANE.** It's bandwidth-bound; the ANE FFN streams int8 (2× the bytes of
  4-bit) and would need int4 (risky/unsupported), so 4-bit GPU decode wins.
- **Bigger fp16 models swap on 16 GB** (1.5B fp16 @ B=32 → 0.57×); int8 halves the weights and
  fixes it — int8's second purpose beyond speed.

See `research/` for the self-contained probes behind each.

---

## API

**`PipelinedRunner(model, ane_frac=1.0, int8=False)`** — batched ANE‖GPU pipeline.
`runner(tokens)` with `tokens` shape `[B, S]`, B even ≥ 2; returns logits identical to
`model(tokens)`. `int8=True` for ~1.85× and to fit larger models. `free()` to release kernels.

**`SingleStreamRunner(model)`** — B=1 ANE-FFN prefill. `runner.prefill(tokens, cache)` returns
last-position logits and populates a KV cache; `runner.generate(prompt_ids, max_tokens, eos_id)`
yields token ids (ANE prefill + MLX decode). Works on fp16 or quantized models.

**`accelerate(model, min_seq=1024)`** — drop-in `SplitMLP` swap (legacy; the runners supersede it).

---

## Caveats

- **Apple Silicon only.** Uses **private** `AppleNeuralEngine` framework APIs — research/
  enthusiast use, not App-Store-safe, may break across macOS updates.
- **`SingleStreamRunner` compile cost:** ANE kernels compile **per length-bucket (~10 s for the
  layer stack, one-time)**; lengths bucket to 128 so nearby prompts reuse them. So it wins for
  workloads with *many* long prompts (doc batches, evals, RAG, agents), **not** a one-off chat
  turn (which pays the ~10 s once and rarely reuses it). This is why `--ane-prefill` is opt-in.
- **Memory ceiling is 16 GB.** Raising `iogpu.wired_limit_mb` (Apple's recommended sysctl)
  lets big models/batches avoid swap; bandwidth itself (~120 GB/s) is a hardware ceiling, not
  a software lock.

---

## Repo layout

```
anegpu/                importable package
  pipeline.py            PipelinedRunner — batched ANE‖GPU parallelism
  single_stream.py       SingleStreamRunner — B=1 ANE-FFN prefill
  accelerate.py          SplitMLP + accelerate() (legacy drop-in)
  _native.py             ctypes bridge -> libanegpu.dylib
native/ane_ffn.m       ANE kernel C source (fused FFN: fp16/int8 × SiLU/GELU) -> libanegpu.dylib
chat.py                interactive chat: 4-bit decode + KV-cache reuse + metrics + --ane-prefill
research/              self-contained probes that established every fact above
  single_stream_prefill.py   B=1 ANE-FFN prefill de-risk (2.3x)
  cpu_amx_probe.py, amx_fp16_probe.c   why the CPU/AMX can't be a 3rd worker
  pipeline_demo.py, stage_times.py, thread_overlap_probe.py, int8_ffn_probe.m, ...
benchmarks/            benchmark.py (fair A/B), demo.py, breakdown.py
```

Built on the private-ANE-API approach from [maderix/ANE](https://github.com/maderix/ANE).
