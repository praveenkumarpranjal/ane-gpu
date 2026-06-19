#!/usr/bin/env python3
"""Interactive multi-turn chat with an mlx-lm model.

    python chat.py                              # default Qwen2.5-1.5B-Instruct
    python chat.py Qwen/Qwen2.5-0.5B-Instruct   # any mlx-lm chat model

NOTE on acceleration: this is single-stream autoregressive decode (batch=1, one token at a
time), which CANNOT use the ANE+GPU pipeline (it needs batch>=2 to overlap micro-batches).
So chat runs on plain MLX (GPU). The anegpu PipelinedRunner accelerates BATCHED PREFILL /
throughput serving (many prompts at once) -- see batch_demo() below for where it helps.
"""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx
from mlx_lm import load, stream_generate

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"


def chat():
    print(f"loading {MODEL} ...")
    model, tok = load(MODEL)
    print("ready. type your message ('exit' to quit, 'reset' to clear history).\n")
    history = []
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
            history = []; print("(history cleared)\n"); continue
        history.append({"role": "user", "content": user})
        prompt = tok.apply_chat_template(history, add_generation_prompt=True)
        print("bot> ", end="", flush=True)
        reply = ""
        last = None
        for r in stream_generate(model, tok, prompt=prompt, max_tokens=512):
            print(r.text, end="", flush=True)
            reply += r.text
            last = r
        print()
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
