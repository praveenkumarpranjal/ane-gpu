#!/usr/bin/env python3
"""Interactive multi-turn chat with an mlx-lm model.

    python chat.py                              # default: pre-quantized 4-bit (~1 GB, ~84 tok/s)
    python chat.py Qwen/Qwen2.5-1.5B-Instruct --4bit   # fp16 repo + in-process 4-bit quant
    python chat.py --8bit Qwen/Qwen2.5-0.5B-Instruct

DECODE SPEED: chat is memory-bandwidth-bound -- every token streams all the weights -- so
quantization is the lever. fp16 ~31 tok/s vs 4-bit ~84-100 tok/s on M4 (matches/beats ollama).
The default is a PRE-quantized repo (~1 GB peak, like ollama). Quantizing an fp16 repo in
process with --4bit is also fast at decode but uses ~3 GB (holds fp16 weights too).
First run downloads the model (set HF_HUB_OFFLINE=0 if it isn't cached yet).

NOTE on the ANE+GPU pipeline: it accelerates BATCHED PREFILL (batch>=2), NOT single-stream
decode -- so it doesn't apply to chat. See batch_demo() for where it helps.
"""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache

argv = sys.argv[1:]
MAX_TOKENS = 512
if "--max-tokens" in argv:                       # e.g. --max-tokens 2048
    i = argv.index("--max-tokens"); MAX_TOKENS = int(argv[i + 1]); del argv[i:i + 2]
BITS = 4 if "--4bit" in argv else (8 if "--8bit" in argv else None)
_args = [a for a in argv if not a.startswith("--")]
MODEL = _args[0] if _args else "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


def chat():
    print(f"loading {MODEL}{f' (+ in-process {BITS}-bit quant)' if BITS else ''} ...")
    model, tok = load(MODEL)
    if BITS:
        nn.quantize(model, group_size=64, bits=BITS)
        mx.eval(model.parameters())
    print("ready. KV cache reused across turns -> only NEW tokens are prefilled each turn.")
    print("('exit' to quit, 'reset' to clear history)\n")
    history = []
    cache = make_prompt_cache(model)
    cached_ids = []        # the token ids currently held in the KV cache
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            break
        if user.lower() == "reset":
            history = []; cache = make_prompt_cache(model); cached_ids = []
            print("(history cleared)\n"); continue
        history.append({"role": "user", "content": user})
        full = list(tok.apply_chat_template(history, add_generation_prompt=True))
        if full[:len(cached_ids)] != cached_ids:        # tokenization diverged -> rebuild cache
            cache = make_prompt_cache(model); cached_ids = []
        new = full[len(cached_ids):]                     # feed ONLY the uncached suffix
        print("bot> ", end="", flush=True)
        reply = ""
        last = None
        gen_ids = []
        for r in stream_generate(model, tok, prompt=new, max_tokens=MAX_TOKENS, prompt_cache=cache):
            print(r.text, end="", flush=True)
            reply += r.text
            gen_ids.append(r.token)
            last = r
        print()
        cached_ids = full + gen_ids                      # cache now holds prompt + response
        if last is not None:
            pt, ptps = last.prompt_tokens, last.prompt_tps
            gt, gtps = last.generation_tokens, last.generation_tps
            pre_ms = (pt / ptps * 1000) if ptps else 0.0
            dec_s = (gt / gtps) if gtps else 0.0
            print(f"  \033[90m[prefill {pt} tok @ {ptps:6.1f} tok/s ({pre_ms:5.0f} ms)"
                  f"  |  decode {gt} tok @ {gtps:6.1f} tok/s ({dec_s:5.2f} s)"
                  f"  |  total {pre_ms/1000 + dec_s:5.2f} s  |  peak {last.peak_memory:.2f} GB]\033[0m\n")
        history.append({"role": "assistant", "content": reply})


def batch_demo():
    """Where the ANE+GPU pipeline DOES help: answering many prompts at once (batched prefill)."""
    import anegpu
    model, tok = load(MODEL); model.set_dtype(mx.float16); mx.eval(model.parameters())
    runner = anegpu.PipelinedRunner(model, int8=True)   # accelerates the batched prefill
    prompts = ["Explain gravity simply.", "Name three primes.", "What is MLX?", "Define entropy."]
    enc = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True) for p in prompts]
    S = max(len(e) for e in enc)
    batch = mx.array([[tok.eos_token_id] * (S - len(e)) + list(e) for e in enc])  # left-pad
    logits = runner(batch)        # one pipelined ANE+GPU prefill over all prompts
    mx.eval(logits)
    print(f"batched prefill of {len(prompts)} prompts -> logits {logits.shape} (ANE+GPU pipeline)")
    runner.free()


if __name__ == "__main__":
    if "--batch-demo" in sys.argv:
        batch_demo()
    else:
        chat()
