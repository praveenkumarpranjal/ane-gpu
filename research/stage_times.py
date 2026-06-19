"""Pipeline stage balance: GPU attention vs ANE FFN vs GPU FFN, per layer at micro-batch
scale. Tells us which stage is the bottleneck and the optimal FFN split for balancing."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from anegpu import _native as ane

B, S = 16, 256          # one micro-batch
N = B * S
model, tok = load("Qwen/Qwen2.5-0.5B-Instruct"); model.set_dtype(mx.float16); mx.eval(model.parameters())
layer = model.model.layers[0]
dim = model.args.hidden_size; H = model.args.intermediate_size
x = mx.random.normal((B, S, dim)).astype(mx.float16); mx.eval(x)
mask = create_attention_mask(x, None)

def t_pipe(make, it=30, w=8):
    for _ in range(w): mx.eval(make())
    t0 = time.perf_counter(); outs = [make() for _ in range(it)]; mx.eval(outs)
    return (time.perf_counter() - t0) / it * 1e3

# GPU attention stage (full)
t_attn = t_pipe(lambda: layer.self_attn(layer.input_layernorm(x), mask, None))

# GPU FFN stage
W1 = mx.contiguous(layer.mlp.gate_proj.weight.T); W3 = mx.contiguous(layer.mlp.up_proj.weight.T)
W2 = mx.contiguous(layer.mlp.down_proj.weight.T)
xn = layer.post_attention_layernorm(x); mx.eval(xn)
def gpu_ffn():
    g = xn @ W1; u = xn @ W3; return ((g * mx.sigmoid(g)) * u) @ W2
t_gffn = t_pipe(gpu_ffn)

# ANE FFN stage
def npw(w): return np.ascontiguousarray(np.array(w.astype(mx.float16)))
k = ane.compile_ffn(dim, H, N, npw(layer.mlp.gate_proj.weight), npw(layer.mlp.up_proj.weight), npw(layer.mlp.down_proj.weight))
xt = np.ascontiguousarray(np.array(mx.transpose(xn.reshape(-1, dim))))
for _ in range(8): k.inbuf[:] = xt; k.run()
t0 = time.perf_counter()
for _ in range(30): k.inbuf[:] = xt; k.run()
t_affn = (time.perf_counter() - t0) / 30 * 1e3
k.free()

print(f"per layer @ B={B} S={S} (ms):")
print(f"  GPU attention : {t_attn:6.2f}")
print(f"  ANE FFN       : {t_affn:6.2f}")
print(f"  GPU FFN       : {t_gffn:6.2f}   (ANE FFN is {t_gffn/t_affn:.2f}x faster)")
print()
print(f"  current pipeline (FFN 100% ANE): GPU stage = attn = {t_attn:.2f} ms")
print(f"                                   ANE stage = FFN  = {t_affn:.2f} ms")
bottleneck = "ANE FFN" if t_affn > t_attn else "GPU attention"
print(f"  -> bottleneck: {bottleneck}  (slack on the other engine = {abs(t_affn-t_attn):.2f} ms/layer)")
# optimal FFN split so GPU stage (attn + (1-f)*gpu_ffn) == ANE stage (f*ane_ffn)
# attn + (1-f)*t_gffn = f*t_affn  ->  f = (attn + t_gffn) / (t_affn + t_gffn)
f = (t_attn + t_gffn) / (t_affn + t_gffn)
f = max(0.0, min(1.0, f))
bal = f * t_affn
print(f"\n  OPTIMAL: ANE does {f*100:.0f}% of FFN, GPU does {(1-f)*100:.0f}% (in its slack)")
print(f"  -> balanced stage = {bal:.2f} ms/layer vs current {max(t_attn,t_affn):.2f} -> {max(t_attn,t_affn)/bal:.2f}x over current pipeline")
