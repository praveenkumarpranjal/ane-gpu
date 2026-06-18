#!/usr/bin/env python3
"""Export Qwen2.5-0.5B-Instruct weights and run GPU+ANE split inference."""

import struct, json, os, sys, subprocess, time
import numpy as np

HF_MODEL_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/"
    "snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "qwen_weights.bin")
ENGINE_SRC = os.path.join(SCRIPT_DIR, "qwen_split_infer.m")
ENGINE_BIN = os.path.join(SCRIPT_DIR, "qwen_split_infer")


def bf16_to_fp16(raw: bytes, count: int) -> bytes:
    """Convert BF16 raw bytes to FP16 via FP32 intermediate."""
    bf16 = np.frombuffer(raw, dtype=np.uint16, count=count)
    fp32 = (bf16.astype(np.uint32) << 16).view(np.float32)
    fp16 = fp32.astype(np.float16)
    return fp16.tobytes()


def export_weights():
    """Parse safetensors and write packed fp16 binary blob."""
    st_path = os.path.join(HF_MODEL_DIR, "model.safetensors")
    with open(os.path.join(HF_MODEL_DIR, "config.json")) as f:
        cfg = json.load(f)

    dim      = cfg["hidden_size"]          # 896
    hidden   = cfg["intermediate_size"]    # 4864
    n_heads  = cfg["num_attention_heads"]  # 14
    n_kv     = cfg["num_key_value_heads"]  # 2
    n_layers = cfg["num_hidden_layers"]    # 24
    vocab    = cfg["vocab_size"]           # 151936
    head_dim = dim // n_heads              # 64
    kv_dim   = n_kv * head_dim             # 128

    # Parse safetensors header
    with open(st_path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    data_base = 8 + hlen

    def tensor_info(name):
        info = header[name]
        start, end = info["data_offsets"]
        count = 1
        for s in info["shape"]:
            count *= s
        return data_base + start, end - start, count, info["shape"]

    print(f"Exporting Qwen2.5-0.5B-Instruct → {WEIGHTS_PATH}")
    print(f"  dim={dim} hidden={hidden} heads={n_heads} kv_heads={n_kv} layers={n_layers}")

    with open(st_path, "rb") as sf, open(WEIGHTS_PATH, "wb") as out:
        # 64-byte header
        hdr = struct.pack("<4sIIIIIIII28s",
            b"QWEN", n_layers, dim, hidden, n_heads, n_kv,
            vocab, head_dim, kv_dim, b"\x00" * 28)
        out.write(hdr)

        def write_tensor(name, expected_shape=None):
            off, sz, count, shape = tensor_info(name)
            if expected_shape:
                assert list(shape) == list(expected_shape), \
                    f"{name}: expected {expected_shape}, got {shape}"
            sf.seek(off)
            raw = sf.read(sz)
            out.write(bf16_to_fp16(raw, count))

        # Embedding table [vocab, dim]
        write_tensor("model.embed_tokens.weight", [vocab, dim])
        print(f"  embed_tokens: {vocab * dim * 2 / 1024 / 1024:.1f} MB")

        # Per-layer weights (strict order must match ObjC loader)
        for l in range(n_layers):
            p = f"model.layers.{l}"
            write_tensor(f"{p}.input_layernorm.weight", [dim])
            write_tensor(f"{p}.self_attn.q_proj.weight", [dim, dim])
            write_tensor(f"{p}.self_attn.q_proj.bias", [dim])
            write_tensor(f"{p}.self_attn.k_proj.weight", [kv_dim, dim])
            write_tensor(f"{p}.self_attn.k_proj.bias", [kv_dim])
            write_tensor(f"{p}.self_attn.v_proj.weight", [kv_dim, dim])
            write_tensor(f"{p}.self_attn.v_proj.bias", [kv_dim])
            write_tensor(f"{p}.self_attn.o_proj.weight", [dim, dim])
            write_tensor(f"{p}.post_attention_layernorm.weight", [dim])
            write_tensor(f"{p}.mlp.gate_proj.weight", [hidden, dim])
            write_tensor(f"{p}.mlp.up_proj.weight", [hidden, dim])
            write_tensor(f"{p}.mlp.down_proj.weight", [dim, hidden])
            if l % 6 == 5 or l == n_layers - 1:
                print(f"  layers 0–{l}: done")

        # Final norm
        write_tensor("model.norm.weight", [dim])

        total = out.tell()
        print(f"  Total: {total / 1024 / 1024:.1f} MB")


def build_engine():
    """Compile ObjC inference engine if source is newer than binary."""
    if os.path.exists(ENGINE_BIN) and \
       os.path.getmtime(ENGINE_BIN) > os.path.getmtime(ENGINE_SRC):
        print(f"Engine up to date: {ENGINE_BIN}")
        return
    print("Compiling inference engine...")
    subprocess.check_call([
        "xcrun", "clang", "-O2", "-fobjc-arc",
        "-framework", "Foundation", "-framework", "IOSurface",
        "-framework", "Metal", "-framework", "MetalPerformanceShaders",
        "-ldl", "-o", ENGINE_BIN, ENGINE_SRC
    ])
    print("  Done.")


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What is the meaning of life?"
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    # Step 1: Export weights if needed
    if not os.path.exists(WEIGHTS_PATH):
        t0 = time.time()
        export_weights()
        print(f"  Export time: {time.time() - t0:.1f}s\n")
    else:
        sz = os.path.getsize(WEIGHTS_PATH) / 1024 / 1024
        print(f"Weights cached: {WEIGHTS_PATH} ({sz:.0f} MB)")

    # Step 2: Build engine
    build_engine()

    # Step 3: Tokenize
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_DIR, trust_remote_code=True)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(text)
    eos_id = 151645  # <|im_end|>
    print(f"\nPrompt: {prompt}")
    print(f"Tokens ({len(input_ids)}): {input_ids[:8]}{'...' if len(input_ids) > 8 else ''}")

    # Step 4: Run inference with streaming
    token_csv = ",".join(str(t) for t in input_ids)
    t0 = time.time()

    proc = subprocess.Popen(
        [ENGINE_BIN, WEIGHTS_PATH, token_csv, str(max_tokens), str(eos_id)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Read stdout (streamed token IDs) and stderr (timing) concurrently
    import threading
    stderr_lines = []
    def read_stderr():
        for line in proc.stderr:
            stderr_lines.append(line.rstrip())
    t = threading.Thread(target=read_stderr, daemon=True)
    t.start()

    # Stream tokens as they arrive
    print(f"\n{'=' * 60}")
    print("Streaming output:")
    print(f"{'=' * 60}")

    raw_stdout = proc.stdout.read()
    proc.wait()
    t.join(timeout=5)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"ERROR (exit {proc.returncode}):")
        for line in stderr_lines[-20:]:
            print(f"  {line}")
        print(raw_stdout[-500:] if raw_stdout else "(no stdout)")
        return

    # Parse output token IDs from stdout (single line, comma-separated)
    output_ids = [int(x) for x in raw_stdout.strip().split(",") if x.strip()]

    # Decode and display
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    n_gen = len(output_ids)
    tps = n_gen / elapsed if elapsed > 0 else 0

    print(generated_text)
    print(f"\n{'=' * 60}")
    print(f"Generated {n_gen} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)")
    print(f"{'=' * 60}")

    # Print timing info from engine
    for line in stderr_lines:
        print(f"  {line}")


if __name__ == "__main__":
    main()
