"""What is the GPU attention stage made of? Projections (movable to the ANE's slack) vs
SDPA (scores/softmax/AV — must stay on GPU). This sets the stage-balancing ceiling.

Times one real layer at the micro-batch scale (B x S), pipelined."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention

B, S = 16, 256
model, tok = load("Qwen/Qwen2.5-0.5B-Instruct"); model.set_dtype(mx.float16); mx.eval(model.parameters())
a = model.model.layers[0].self_attn
dim = model.args.hidden_size
nh, nkv = a.n_heads, a.n_kv_heads
hd = dim // nh
x = mx.random.normal((B, S, dim)).astype(mx.float16); mx.eval(x)
mask = create_attention_mask(x, None)

Wq = mx.contiguous(a.q_proj.weight.T); bq = a.q_proj.bias
Wk = mx.contiguous(a.k_proj.weight.T); bk = a.k_proj.bias
Wv = mx.contiguous(a.v_proj.weight.T); bv = a.v_proj.bias
Wo = mx.contiguous(a.o_proj.weight.T)

def t_pipe(make, it=40, w=10):
    for _ in range(w): mx.eval(make())
    t0 = time.perf_counter()
    outs = [make() for _ in range(it)]
    mx.eval(outs)
    return (time.perf_counter() - t0) / it * 1e3

def proj_q(): return x @ Wq + bq
def proj_k(): return x @ Wk + bk
def proj_v(): return x @ Wv + bv
def proj_o():
    o = (x @ Wq + bq)            # dummy attn_out-shaped input for o_proj timing
    return o @ Wo
def full_attn(): return a(x, mask, None)

def sdpa_only():
    q = (x @ Wq + bq).reshape(B, S, nh, hd).transpose(0, 2, 1, 3)
    k = (x @ Wk + bk).reshape(B, S, nkv, hd).transpose(0, 2, 1, 3)
    v = (x @ Wv + bv).reshape(B, S, nkv, hd).transpose(0, 2, 1, 3)
    q = a.rope(q); k = a.rope(k)
    return scaled_dot_product_attention(q, k, v, cache=None, scale=a.scale, mask=mask)

tq, tk, tv, to = t_pipe(proj_q), t_pipe(proj_k), t_pipe(proj_v), t_pipe(proj_o)
tfull = t_pipe(full_attn)
tsdpa_path = t_pipe(sdpa_only)   # includes q/k/v proj + rope + scores+softmax+AV
proj_total = tq + tk + tv + to
sdpa_est = tsdpa_path - (tq + tk + tv)   # subtract the projections in the path -> ~rope+SDPA

print(f"per layer @ B={B} S={S} (GPU, ms):")
print(f"  q proj 896->896 : {tq:6.3f}")
print(f"  k proj 896->128 : {tk:6.3f}")
print(f"  v proj 896->128 : {tv:6.3f}")
print(f"  o proj 896->896 : {to:6.3f}")
print(f"  -> projections   : {proj_total:6.3f}")
print(f"  SDPA(+rope) est  : {sdpa_est:6.3f}   (rope+scores+softmax+AV — must stay GPU)")
print(f"  full attention   : {tfull:6.3f}")
print(f"\n  movable to ANE (q+o): {tq+to:6.3f} ms ({(tq+to)/tfull*100:.0f}% of attention)")
print(f"  movable (q+k+v+o)   : {proj_total:6.3f} ms ({proj_total/tfull*100:.0f}% of attention)")
print(f"  stuck on GPU (SDPA) : {sdpa_est:6.3f} ms ({sdpa_est/tfull*100:.0f}% of attention)")
