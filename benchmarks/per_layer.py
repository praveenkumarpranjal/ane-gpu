"""Per-layer ANE vs GPU characterization of the real Qwen2.5-0.5B (fp16, native format).

For every transformer layer, extract the real weights and time each matmul on BOTH engines:
  ANE  = anegpu._native conv-matmul / fused FFN (per-eval)
  GPU  = MLX fp16 matmul (pipelined, the honest in-forward cost)

Tested at seq=256 (prefill). The ANE cannot run seq=1 (decode), so decode is GPU-only.
Usage: python per_layer.py [seq]
"""
import sys, os, time, statistics
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx
from mlx_lm import load
from anegpu import _native as ane

SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 256
assert SEQ % 32 == 0, "ANE fused FFN needs seq % 32 == 0"

print(f"Loading Qwen2.5-0.5B-Instruct (fp16, seq={SEQ}) ...")
model, tok = load("Qwen/Qwen2.5-0.5B-Instruct"); model.set_dtype(mx.float16); mx.eval(model.parameters())
layers = model.model.layers
dim = model.args.hidden_size
hidden = model.args.intermediate_size
print(f"{len(layers)} layers, dim={dim}, hidden={hidden}\n")

# shared random inputs (timing is value-independent); ANE wants channel-major [in_ch, seq]
def rand_mx(r, c): return mx.random.normal((r, c)).astype(mx.float16)
X = rand_mx(SEQ, dim); mx.eval(X)                       # [seq, dim] for GPU
Xa = {dim: np.ascontiguousarray(np.array(mx.transpose(X)))}  # [dim, seq] for ANE

WARM, IT = 3, 15

def gpu_us(make):
    for _ in range(WARM): mx.eval(make())
    t0 = time.perf_counter()
    outs = [make() for _ in range(IT)]
    mx.eval(outs)
    return (time.perf_counter() - t0) / IT * 1e6

def ane_matmul_us(W_np, in_ch, out_ch):
    k = ane.compile_matmul(in_ch, out_ch, SEQ, W_np)
    if k is None: return None
    if in_ch not in Xa: Xa[in_ch] = np.ascontiguousarray((np.random.randn(in_ch, SEQ) * 0.5).astype(np.float16))
    k.inbuf[:] = Xa[in_ch]
    for _ in range(WARM):
        if not k.run(): k.free(); return None
    t0 = time.perf_counter()
    for _ in range(IT): k.run()
    us = (time.perf_counter() - t0) / IT * 1e6
    k.free(); return us

def ane_ffn_us(W1, W3, W2):
    k = ane.compile_ffn(dim, hidden, SEQ, W1, W3, W2)
    if k is None: return None
    k.inbuf[:] = Xa[dim]
    for _ in range(WARM):
        if not k.run(): k.free(); return None
    t0 = time.perf_counter()
    for _ in range(IT): k.run()
    us = (time.perf_counter() - t0) / IT * 1e6
    k.free(); return us

def npw(w): return np.ascontiguousarray(np.array(w.astype(mx.float16)))   # [out, in]

OPS = [("q", "896->896"), ("k", "896->128"), ("v", "896->128"),
       ("o", "896->896"), ("gate", "896->4864"), ("up", "896->4864"),
       ("down", "4864->896"), ("FFN", "fused")]
rows = []
print(f"{'L':>2} | " + " | ".join(f"{n:>4}A/G" for n, _ in OPS))
for li, blk in enumerate(layers):
    a = blk.self_attn; m = blk.mlp
    W = {"q": a.q_proj.weight, "k": a.k_proj.weight, "v": a.v_proj.weight, "o": a.o_proj.weight,
         "gate": m.gate_proj.weight, "up": m.up_proj.weight, "down": m.down_proj.weight}
    res = {}
    # individual matmuls: ANE + GPU
    for nm in ("q", "k", "v", "o", "gate", "up", "down"):
        Wm = W[nm]; out_ch, in_ch = Wm.shape
        res[nm + "_ane"] = ane_matmul_us(npw(Wm), in_ch, out_ch)
        Wt = mx.contiguous(Wm.T) if hasattr(mx, "contiguous") else (Wm.T + 0)
        Xin = X if in_ch == dim else rand_mx(SEQ, in_ch)
        res[nm + "_gpu"] = gpu_us(lambda Xin=Xin, Wt=Wt: Xin @ Wt)
    # fused FFN: ANE vs GPU (3 matmuls + silu)
    W1, W3, W2 = npw(W["gate"]), npw(W["up"]), npw(W["down"])
    res["FFN_ane"] = ane_ffn_us(W1, W3, W2)
    g1, g3 = mx.contiguous(W["gate"].T) if hasattr(mx,"contiguous") else W["gate"].T+0, \
             mx.contiguous(W["up"].T) if hasattr(mx,"contiguous") else W["up"].T+0
    g2 = mx.contiguous(W["down"].T) if hasattr(mx,"contiguous") else W["down"].T+0
    res["FFN_gpu"] = gpu_us(lambda: ((lambda g: (g*mx.sigmoid(g))*(X@g3))(X@g1)) @ g2)
    rows.append(res)
    def cell(nm):
        a_, g_ = res.get(nm+"_ane"), res.get(nm+"_gpu")
        return f"{(a_ or 0):4.0f}/{(g_ or 0):<4.0f}"
    print(f"{li:>2} | " + " | ".join(cell(n) for n, _ in OPS))

# ---- per-op summary across layers ----
print("\n=== per-op summary (mean over 24 layers, us) ===")
print(f"{'op':>5} {'shape':>11} | {'ANE us':>8} {'GPU us':>8} | {'ANE/GPU':>8}")
for nm, shape in OPS:
    a_ = [r[nm+"_ane"] for r in rows if r.get(nm+"_ane")]
    g_ = [r[nm+"_gpu"] for r in rows if r.get(nm+"_gpu")]
    if not a_ or not g_:
        print(f"{nm:>5} {shape:>11} | {'n/a':>8}"); continue
    am, gm = statistics.mean(a_), statistics.mean(g_)
    winner = "ANE" if am < gm else "GPU"
    print(f"{nm:>5} {shape:>11} | {am:8.0f} {gm:8.0f} | {gm/am:7.2f}x  -> {winner} faster")
