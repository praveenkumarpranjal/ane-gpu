"""TRUE PARALLELISM demo: a 2-batch staggered pipeline so neither engine sits idle.

While the ANE computes batch A's FFN (worker thread, pure ctypes), the GPU computes
batch B's attention (main thread, MLX) — offset by one stage. Compares:
  - GPU-only         : normal MLX forward (baseline)
  - serial ANE-FFN   : attn(GPU) then FFN(ANE) per layer, no overlap (today's approach)
  - PIPELINED        : the 2-batch staggered overlap

MLX is touched ONLY on the main thread; the worker does only kernel.run().
"""
import sys, os, time, threading
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from anegpu import _native as ane

B, S = 16, 256
model, tok = load("Qwen/Qwen2.5-0.5B-Instruct"); model.set_dtype(mx.float16); mx.eval(model.parameters())
mdl = model.model
layers = mdl.layers
L = len(layers)
dim = model.args.hidden_size; H = model.args.intermediate_size
print(f"{L} layers, dim={dim}, hidden={H}, batch={B}x{S} per stream, 2 streams = {2*B*S} tokens\n")

# two INDEPENDENT input streams
ids = tok.encode("The history of computing spans many disciplines. " * 80)[:S]
tok0 = mx.array([ids] * B); tok1 = mx.array([ids[::-1]] * B)

def npw(w): return np.ascontiguousarray(np.array(w.astype(mx.float16)))
print("compiling 24 ANE FFN kernels (full hidden) ...")
kernels = [ane.compile_ffn(dim, H, B * S,
                           npw(l.mlp.gate_proj.weight), npw(l.mlp.up_proj.weight), npw(l.mlp.down_proj.weight))
           for l in layers]
assert all(kernels), "ANE FFN compile failed"

def attn_stage(h, layer, mask):
    """GPU: attention + residual + post-norm -> (residual-carry h, ffn_in). MLX only."""
    r = layer.self_attn(layer.input_layernorm(h), mask, None)
    h = h + r
    return h, layer.post_attention_layernorm(h)

def ffn_prep(ffn_in, kernel):              # main thread (MLX -> IOSurface)
    flat = ffn_in.reshape(-1, dim)
    xt = mx.transpose(flat.astype(mx.float16)); mx.eval(xt)
    kernel.inbuf[:] = np.array(xt)

def ffn_read(kernel, shape):               # main thread (IOSurface -> MLX)
    return mx.transpose(mx.array(kernel.outbuf)).reshape(shape)

def finalize(h):
    return mdl.embed_tokens.as_linear(mdl.norm(h))   # tied embeddings

# ---------- baselines ----------
def gpu_only(t0_, t1_):
    return mx.eval([model(t0_), model(t1_)])

def serial_ane(t0_, t1_):
    mask = create_attention_mask(mdl.embed_tokens(t0_), None)
    outs = []
    for toks in (t0_, t1_):
        h = mdl.embed_tokens(toks)
        for li, layer in enumerate(layers):
            h, ffn_in = attn_stage(h, layer, mask); mx.eval(h, ffn_in)
            ffn_prep(ffn_in, kernels[li]); kernels[li].run()
            h = h + ffn_read(kernels[li], h.shape)
        outs.append(finalize(h))
    return mx.eval(outs)

# ---------- the pipeline ----------
def pipelined(t0_, t1_):
    mask = create_attention_mask(mdl.embed_tokens(t0_), None)
    h = [mdl.embed_tokens(t0_), mdl.embed_tokens(t1_)]
    ffn_in = [None, None]
    seq = [(b, l) for l in range(L) for b in (0, 1)]   # (0,0),(1,0),(0,1),(1,1),...
    # prime: attn(0,0)
    b0, l0 = seq[0]
    h[b0], ffn_in[b0] = attn_stage(h[b0], layers[l0], mask); mx.eval(h[b0], ffn_in[b0])
    for i in range(1, len(seq)):
        b, l = seq[i]
        pb, pl = seq[i - 1]                            # FFN to run now = previous attn
        ffn_prep(ffn_in[pb], kernels[pl])              # main: stage ANE input
        th = threading.Thread(target=kernels[pl].run)  # worker: ANE run (ctypes only)
        th.start()
        h[b], ffn_in[b] = attn_stage(h[b], layers[l], mask)  # main: GPU attention overlaps ANE
        mx.eval(h[b], ffn_in[b])
        th.join()
        h[pb] = h[pb] + ffn_read(kernels[pl], h[pb].shape)   # apply FFN result
    # last FFN
    lb, ll = seq[-1]
    ffn_prep(ffn_in[lb], kernels[ll]); kernels[ll].run()
    h[lb] = h[lb] + ffn_read(kernels[ll], h[lb].shape)
    return mx.eval([finalize(h[0]), finalize(h[1])])

# ---------- correctness ----------
ref0 = np.array(model(tok0)[:, -1].astype(mx.float32)); ref1 = np.array(model(tok1)[:, -1].astype(mx.float32))
import importlib
# run pipelined once and compare last-token argmax
pipelined(tok0, tok1)
# recompute pipelined logits for correctness (full)
def pipe_logits():
    mask = create_attention_mask(mdl.embed_tokens(tok0), None)
    h = [mdl.embed_tokens(tok0), mdl.embed_tokens(tok1)]
    ffn_in=[None,None]; seq=[(b,l) for l in range(L) for b in (0,1)]
    b0,l0=seq[0]; h[b0],ffn_in[b0]=attn_stage(h[b0],layers[l0],mask); mx.eval(h[b0],ffn_in[b0])
    for i in range(1,len(seq)):
        b,l=seq[i]; pb,pl=seq[i-1]
        ffn_prep(ffn_in[pb],kernels[pl]); th=threading.Thread(target=kernels[pl].run); th.start()
        h[b],ffn_in[b]=attn_stage(h[b],layers[l],mask); mx.eval(h[b],ffn_in[b]); th.join()
        h[pb]=h[pb]+ffn_read(kernels[pl],h[pb].shape)
    lb,ll=seq[-1]; ffn_prep(ffn_in[lb],kernels[ll]); kernels[ll].run(); h[lb]=h[lb]+ffn_read(kernels[ll],h[lb].shape)
    return np.array(finalize(h[0])[:,-1].astype(mx.float32)), np.array(finalize(h[1])[:,-1].astype(mx.float32))
p0,p1 = pipe_logits()
ok0 = ref0[0].argmax()==p0[0].argmax(); ok1 = ref1[0].argmax()==p1[0].argmax()
print(f"correctness: stream0 argmax {'MATCH' if ok0 else 'DIFFER'}, stream1 {'MATCH' if ok1 else 'DIFFER'}\n")

# ---------- timing ----------
def bench(fn, it=8, w=2):
    for _ in range(w): fn()
    t0=time.perf_counter()
    for _ in range(it): fn()
    return (time.perf_counter()-t0)/it
toks = 2 * B * S
for name, fn in [("GPU-only (baseline)", gpu_only), ("serial ANE-FFN (no overlap)", serial_ane), ("PIPELINED (overlap)", pipelined)]:
    t = bench(lambda fn=fn: fn(tok0, tok1))
    print(f"{name:30}: {t*1e3:7.1f} ms   {toks/t:7.0f} tok/s")
for k in kernels: k.free()
