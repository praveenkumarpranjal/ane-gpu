"""PipelinedRunner — batched inference with TRUE GPU+ANE parallelism.

Splits a batch into two micro-batches and runs them through the model offset by one stage:
while the ANE computes micro-batch A's FFN (worker thread, ctypes only), the GPU computes
micro-batch B's attention (main thread, MLX). Neither engine sits idle.

    import anegpu
    runner = anegpu.PipelinedRunner(model)     # mlx_lm model
    logits = runner(tokens)                     # [B, S, vocab], == model(tokens)

Requires an even batch >= 2 (independent sequences to overlap); falls back to model() for
B<2 or odd B, or when disabled. MLX is touched ONLY on the main thread (the worker does
nothing but kernel.run()), so it stays thread-safe.
"""
import threading
import numpy as np
import mlx.core as mx
from . import _native as ane
from .accelerate import _cpad, ENABLED

try:
    from mlx_lm.models.base import create_attention_mask as _make_mask
except Exception:
    _make_mask = None


class PipelinedRunner:
    def __init__(self, model):
        self.model = model
        self.mdl = model.model
        self.layers = self.mdl.layers
        self.dim = model.args.hidden_size
        self.hidden = model.args.intermediate_size
        self.tied = getattr(model.args, "tie_word_embeddings", False)
        # per-layer FFN weights as fp16 numpy (for the ANE); whole FFN goes to the ANE
        # (in the pipeline the GPU is busy with attention).
        self._W = [(self._np(l.mlp.gate_proj.weight), self._np(l.mlp.up_proj.weight),
                    self._np(l.mlp.down_proj.weight)) for l in self.layers]
        self._kernels = {}   # (layer_idx, Npad) -> ANE kernel

    @staticmethod
    def _np(w):
        return np.ascontiguousarray(np.array(w.astype(mx.float16)))

    def _kernel(self, li, Npad):
        k = self._kernels.get((li, Npad))
        if k is None:
            g, u, d = self._W[li]
            k = ane.compile_ffn(self.dim, self.hidden, Npad, g, u, d)
            self._kernels[(li, Npad)] = k
        return k

    # --- GPU stage: attention + residual + post-norm -> (residual-carry h, ffn_in) ---
    def _attn(self, h, layer, mask):
        r = layer.self_attn(layer.input_layernorm(h), mask, None)
        h = h + r
        return h, layer.post_attention_layernorm(h)

    # --- ANE stage I/O (MLX, main thread only) ---
    def _ffn_prep(self, ffn_in, kernel, N, Npad):
        xt = mx.transpose(ffn_in.reshape(-1, self.dim).astype(mx.float16))   # [dim, N] on GPU
        mx.eval(xt)
        kernel.inbuf[:, :N] = np.array(xt)
        if Npad > N:
            kernel.inbuf[:, N:] = 0

    def _ffn_read(self, kernel, N, shape):
        return mx.transpose(mx.array(kernel.outbuf[:, :N])).reshape(shape)

    def _finalize(self, h):
        h = self.mdl.norm(h)
        return self.mdl.embed_tokens.as_linear(h) if self.tied else self.model.lm_head(h)

    def __call__(self, tokens):
        B, S = tokens.shape
        if not ENABLED or _make_mask is None or B < 2 or (B % 2) != 0:
            return self.model(tokens)                      # graceful fallback

        half = B // 2
        t = [tokens[:half], tokens[half:]]
        N = half * S
        Npad = _cpad(N)
        L = len(self.layers)
        kern = [self._kernel(li, Npad) for li in range(L)]
        if not all(kern):
            return self.model(tokens)

        mask = _make_mask(self.mdl.embed_tokens(t[0]), None)
        h = [self.mdl.embed_tokens(t[0]), self.mdl.embed_tokens(t[1])]
        ffn_in = [None, None]
        sched = [(b, l) for l in range(L) for b in (0, 1)]   # (0,0),(1,0),(0,1),(1,1),...

        # prime the pipeline with the first attention
        b0, l0 = sched[0]
        h[b0], ffn_in[b0] = self._attn(h[b0], self.layers[l0], mask)
        mx.eval(h[b0], ffn_in[b0])

        for i in range(1, len(sched)):
            b, l = sched[i]
            pb, pl = sched[i - 1]                            # FFN to run now = previous attn
            self._ffn_prep(ffn_in[pb], kern[pl], N, Npad)    # stage ANE input (main)
            th = threading.Thread(target=kern[pl].run)       # ANE runs (worker, ctypes only)...
            th.start()
            h[b], ffn_in[b] = self._attn(h[b], self.layers[l], mask)   # ...while GPU does attention
            mx.eval(h[b], ffn_in[b])
            th.join()
            h[pb] = h[pb] + self._ffn_read(kern[pl], N, h[pb].shape)   # apply FFN result

        # drain the last FFN
        lb, ll = sched[-1]
        self._ffn_prep(ffn_in[lb], kern[ll], N, Npad)
        kern[ll].run()
        h[lb] = h[lb] + self._ffn_read(kern[ll], N, h[lb].shape)

        return mx.concatenate([self._finalize(h[0]), self._finalize(h[1])], axis=0)

    def free(self):
        for k in self._kernels.values():
            k.free()
        self._kernels = {}
