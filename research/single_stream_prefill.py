"""Phase 0 de-risk: single-stream (B=1) prefill with the FFN offloaded to the ANE.

The transformer is serial across layers, so we can't pipeline two streams at B=1 -- but the
ANE FFN is ~3.9x faster than the GPU FFN at S=512, so just running each layer's FFN on the
ANE (GPU does attention) makes each layer faster. Hand-off is GPU-side mx.transpose + one
IOSurface copy (the old numpy-transpose tax is gone). Per-layer fp16 fallback for int8
overflow layers (same as the batched runner).

Measures: prefill tok/s GPU-only vs ANE-FFN, and argmax correctness, at several prompt lens.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask as mkmask
from anegpu import _native as ane

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
m, tok = load(MODEL); m.set_dtype(mx.float16); mx.eval(m.parameters())
mdl = m.model
D, H, L = m.args.hidden_size, m.args.intermediate_size, len(mdl.layers)
tied = getattr(m.args, "tie_word_embeddings", False)


def npw(w):
    return np.ascontiguousarray(np.array(w.astype(mx.float16)))


def finalize(h):
    h = mdl.norm(h)
    if tied or getattr(m, "lm_head", None) is None:
        return mdl.embed_tokens.as_linear(h)
    return m.lm_head(h)


class SS:
    """Single-stream ANE-FFN forward, kernels cached per padded length."""
    def __init__(self):
        self.kern = {}       # (li, Npad) -> kernel
        self.ok = {}         # (li, Npad) -> int8 finite?

    def _kernel(self, li, Npad):
        k = self.kern.get((li, Npad))
        if k is None:
            l = mdl.layers[li]
            k = ane.compile_ffn_int8(D, H, Npad, npw(l.mlp.gate_proj.weight),
                                     npw(l.mlp.up_proj.weight), npw(l.mlp.down_proj.weight))
            self.kern[(li, Npad)] = k
        return k

    def _ffn(self, li, ffn_in, S, Npad):
        if not self.ok.get((li, Npad), True):
            return mdl.layers[li].mlp(ffn_in)            # fp16 fallback (overflow layer)
        k = self._kernel(li, Npad)
        xt = mx.transpose(ffn_in.reshape(-1, D))         # GPU transpose [dim, S]
        mx.eval(xt)
        k.inbuf[:, :S] = np.array(xt)
        if Npad > S:
            k.inbuf[:, S:] = 0
        k.run()
        return mx.transpose(mx.array(k.outbuf[:, :S])).reshape(ffn_in.shape)

    def calibrate(self, x):
        S = x.shape[1]; Npad = ((S + 31) // 32) * 32
        h = mdl.embed_tokens(x); mask = mkmask(h, None)
        for li, l in enumerate(mdl.layers):
            h = h + l.self_attn(l.input_layernorm(h), mask, None)
            fin = l.post_attention_layernorm(h)
            k = self._kernel(li, Npad)
            xt = mx.transpose(fin.reshape(-1, D)); mx.eval(xt)
            k.inbuf[:, :S] = np.array(xt); k.inbuf[:, S:] = 0 if Npad > S else k.inbuf[:, S:]
            k.run()
            got = mx.transpose(mx.array(k.outbuf[:, :S])).reshape(fin.shape)
            ref = l.mlp(fin); mx.eval(got, ref)
            self.ok[(li, Npad)] = bool(mx.all(mx.isfinite(got))) and \
                float(mx.mean(mx.abs(got - ref)) / (mx.mean(mx.abs(ref)) + 1e-6)) < 0.1
            h = h + ref
        bad = sum(1 for li in range(L) if not self.ok[(li, Npad)])
        return bad

    def __call__(self, x):
        S = x.shape[1]; Npad = ((S + 31) // 32) * 32
        h = mdl.embed_tokens(x); mask = mkmask(h, None)
        for li, l in enumerate(mdl.layers):
            h = h + l.self_attn(l.input_layernorm(h), mask, None)
            h = h + self._ffn(li, l.post_attention_layernorm(h), S, Npad)
        return finalize(h)


ss = SS()
print(f"{MODEL}  (D={D} H={H} L={L})")
for S in (128, 512, 1024):
    x = mx.array([list(range(100, 100 + S))])
    bad = ss.calibrate(x)
    ref = mx.argmax(m(x)[:, -1, :], axis=-1)
    out = mx.argmax(ss(x)[:, -1, :], axis=-1)
    match = bool(mx.all(out == ref))
    for _ in range(2): mx.eval(m(x)); mx.eval(ss(x))
    def t(fn, it=5):
        t0 = time.perf_counter()
        for _ in range(it): mx.eval(fn(x))
        return (time.perf_counter() - t0) / it
    tg, ta = t(m), t(ss)
    print(f"  S={S:5d}: GPU {S/tg:6.0f} tok/s | ANE-FFN {S/ta:6.0f} tok/s | {tg/ta:.2f}x | "
          f"argmax {'OK' if match else 'MISMATCH'} | {bad}/{L} layers fp16")
# Fair baseline: 4-bit GPU prefill (what chat actually runs) vs the ANE-FFN numbers above
import mlx.nn as nn
nn.quantize(m, group_size=64, bits=4); mx.eval(m.parameters())
print("4-bit GPU baseline (same model, for comparison vs ANE-FFN above):")
for S in (128, 512, 1024):
    x = mx.array([list(range(100, 100 + S))])
    for _ in range(2): mx.eval(m(x))
    t0 = time.perf_counter()
    for _ in range(5): mx.eval(m(x))
    print(f"  S={S:5d}: 4-bit GPU {S/((time.perf_counter()-t0)/5):6.0f} tok/s")
for k in ss.kern.values():
    k.free()
