"""PipelinedRunner — batched inference with TRUE GPU+ANE parallelism + stage-balancing.

Splits a batch into two micro-batches and runs them offset by one stage: while the ANE
computes micro-batch A's FFN (worker thread, ctypes only), the GPU computes micro-batch B's
attention (main thread, MLX). Neither engine sits idle.

Measured stage balance (Qwen2.5-0.5B, B=16): GPU attention ~6.8 ms/layer, ANE FFN ~14 ms/
layer -> the ANE FFN is the bottleneck. `ane_frac` can hand the GPU's slack some of the FFN
(it computes the shard for the other micro-batch alongside this one's attention), but on
M4-base this measured ~NEUTRAL (the GPU FFN is 3x slower, so its shard's cost + the combine
cancel the theoretical ~1.14x). So default ane_frac=1.0 (whole FFN on the ANE). The knob is
kept for M4 Pro/Max (faster GPU FFN) and for INT8 (which shrinks the ANE bottleneck — the
real lever, ~1.55x).

    import anegpu
    runner = anegpu.PipelinedRunner(model)
    logits = runner(tokens)               # [B, S, vocab], == model(tokens), B even >= 2

MLX is touched ONLY on the main thread (the worker does nothing but kernel.run()).
"""
import threading
import numpy as np
import mlx.core as mx
from . import _native as ane
from .accelerate import _cpad, _r16, ENABLED

try:
    from mlx_lm.models.base import create_attention_mask as _make_mask
except Exception:
    _make_mask = None

try:
    from mlx_lm.models.gemma3 import clip_residual as _clip_residual_fn
except Exception:
    _clip_residual_fn = None


class PipelinedRunner:
    def __init__(self, model, ane_frac=1.0, int8=False):
        self.model = model
        self.int8 = int8
        self.mdl = model.model
        self.layers = self.mdl.layers
        self.dim = model.args.hidden_size
        self.hidden = model.args.intermediate_size
        self.tied = getattr(model.args, "tie_word_embeddings", False)
        # --- architecture adapter (Qwen/Llama SwiGLU  vs  Gemma GeGLU sandwich-norm) ---
        l0 = self.layers[0]
        self._gemma = (hasattr(l0, "pre_feedforward_layernorm")
                       and hasattr(l0, "post_feedforward_layernorm"))
        self._gelu = False          # FFN activation, auto-detected in _check_compatible
        self._sw_pattern = getattr(self.mdl, "sliding_window_pattern", 1)
        self._window = getattr(self.mdl, "window_size", None)
        ah = _r16(ane_frac * self.hidden)
        ah = min(ah, self.hidden - 16) if ane_frac < 1.0 else self.hidden
        self.ane_h, self.gpu_h = ah, self.hidden - ah

        self._Wa = []     # ANE shard weights (numpy fp16) per layer
        self._Wg = []     # GPU shard weights (MLX fp16) per layer
        for l in self.layers:
            W1, W3, W2 = self._np(l.mlp.gate_proj.weight), self._np(l.mlp.up_proj.weight), self._np(l.mlp.down_proj.weight)
            self._Wa.append((np.ascontiguousarray(W1[:ah]), np.ascontiguousarray(W3[:ah]),
                             np.ascontiguousarray(W2[:, :ah])))
            if self.gpu_h > 0:
                self._Wg.append((mx.array(np.ascontiguousarray(W1[ah:].T)),    # [dim, gpu_h]
                                 mx.array(np.ascontiguousarray(W3[ah:].T)),
                                 mx.array(np.ascontiguousarray(W2[:, ah:].T))))  # [gpu_h, dim]
            else:
                self._Wg.append(None)
        self._kernels = {}
        self._compatible = None     # lazily verified on first call (SwiGLU-or-fallback)

    @staticmethod
    def _np(w):
        return np.ascontiguousarray(np.array(w.astype(mx.float16)))

    def _kernel(self, li, Npad):
        k = self._kernels.get((li, Npad))
        if k is None:
            g, u, d = self._Wa[li]
            compile_fn = ane.compile_ffn_int8 if self.int8 else ane.compile_ffn
            k = compile_fn(self.dim, self.ane_h, Npad, g, u, d, gelu=self._gelu)
            self._kernels[(li, Npad)] = k
        return k

    def _embed(self, toks):
        h = self.mdl.embed_tokens(toks)
        if self._gemma:                                   # Gemma scales embeddings by sqrt(dim)
            h = h * mx.array(self.dim ** 0.5, mx.bfloat16).astype(h.dtype)
        return h

    def _clip_residual(self, x, y):
        if _clip_residual_fn is not None:
            return _clip_residual_fn(x, y)
        if x.dtype != mx.float16:
            return x + y
        b = 65504.0      # fp16 max — Gemma clips the residual to avoid overflow
        return mx.clip(x.astype(mx.float32) + y.astype(mx.float32), -b, b).astype(mx.float16)

    def _masks(self, h0):
        """Returns (global_mask, sliding_mask). Gemma alternates them per layer; others
        use a single causal mask (sliding=None)."""
        gm = _make_mask(h0, None)
        sw = None
        if self._gemma and self._sw_pattern > 1 and self._window:
            try:
                sw = _make_mask(h0, None, window_size=self._window)
            except Exception:
                sw = None        # short seq (< window) -> sliding == causal anyway
        return gm, sw

    def _mask_for(self, l, gm, sw):
        if sw is None:
            return gm
        is_global = (l % self._sw_pattern == self._sw_pattern - 1)
        return gm if is_global else sw

    def _attn(self, h, layer, mask):
        r = layer.self_attn(layer.input_layernorm(h), mask, None)
        if self._gemma:                                   # post-attn norm on the attn OUTPUT
            h = self._clip_residual(h, layer.post_attention_layernorm(r))
            return h, layer.pre_feedforward_layernorm(h)
        h = h + r
        return h, layer.post_attention_layernorm(h)

    def _ffn_combine(self, h_after, ffn_out, layer):
        if self._gemma:
            return self._clip_residual(h_after, layer.post_feedforward_layernorm(ffn_out))
        return h_after + ffn_out

    def _gpu_ffn_shard(self, ffn_in_flat, li):     # GPU does (1-ane_frac) of the FFN
        W1g, W3g, W2g = self._Wg[li]
        g = ffn_in_flat @ W1g
        u = ffn_in_flat @ W3g
        return ((g * mx.sigmoid(g)) * u) @ W2g     # [N, dim] partial

    def _ffn_prep(self, ffn_in, kernel, N, Npad):
        xt = mx.transpose(ffn_in.reshape(-1, self.dim).astype(mx.float16))
        mx.eval(xt)
        kernel.inbuf[:, :N] = np.array(xt)
        if Npad > N:
            kernel.inbuf[:, N:] = 0

    def _ffn_read(self, kernel, N, shape):
        return mx.transpose(mx.array(kernel.outbuf[:, :N])).reshape(shape)

    def _finalize(self, h):
        h = self.mdl.norm(h)
        if self.tied or getattr(self.model, "lm_head", None) is None:
            return self.mdl.embed_tokens.as_linear(h)
        return self.model.lm_head(h)

    def _check_compatible(self):
        """Auto-detect this model's FFN activation by checking which fused kernel (SiLU
        vs GELU) reproduces the model's own mlp() on a probe. Sets self._gelu. Returns
        False (-> fall back to plain MLX) if neither matches — unknown/unsupported arch,
        so we never crash and never emit wrong output."""
        try:
            if self.gpu_h > 0:        # split mode is Qwen/SwiGLU-only; skip detection
                self._gelu = False
                return True
            l0 = self.layers[0]
            probe = (mx.random.normal((32, self.dim)) * 0.5).astype(mx.float16)
            ref = l0.mlp(probe); mx.eval(ref)
            xin = np.array(mx.transpose(probe.astype(mx.float16)))
            g, u, d = self._Wa[0]
            for gelu in (False, True):                       # fp16 probe = clean activation signal
                k = ane.compile_ffn(self.dim, self.ane_h, 32, g, u, d, gelu=gelu)
                if not k:
                    continue
                k.inbuf[:, :32] = xin
                ok = k.run()
                got = mx.transpose(mx.array(k.outbuf[:, :32])); mx.eval(got)
                rel = float(mx.mean(mx.abs(got - ref)) / (mx.mean(mx.abs(ref)) + 1e-6))
                k.free()
                if ok and rel < 0.05:
                    self._gelu = gelu
                    return True
            return False
        except Exception:
            return False

    def __call__(self, tokens):
        B, S = tokens.shape
        if not ENABLED or _make_mask is None or B < 2 or (B % 2) != 0:
            return self.model(tokens)
        if self._compatible is None:
            self._compatible = self._check_compatible()
        if not self._compatible:
            return self.model(tokens)        # non-SwiGLU arch -> plain MLX, transparently

        half = B // 2
        t = [tokens[:half], tokens[half:]]
        N = half * S
        Npad = _cpad(N)
        L = len(self.layers)
        kern = [self._kernel(li, Npad) for li in range(L)]
        if not all(kern):
            return self.model(tokens)
        split = self.gpu_h > 0

        h = [self._embed(t[0]), self._embed(t[1])]
        gm, sw = self._masks(h[0])
        ffn_in = [None, None]
        sched = [(b, l) for l in range(L) for b in (0, 1)]

        b0, l0 = sched[0]
        h[b0], ffn_in[b0] = self._attn(h[b0], self.layers[l0], self._mask_for(l0, gm, sw))
        mx.eval(h[b0], ffn_in[b0])

        for i in range(1, len(sched)):
            b, l = sched[i]
            pb, pl = sched[i - 1]                              # FFN to run now = previous attn
            self._ffn_prep(ffn_in[pb], kern[pl], N, Npad)      # stage ANE input (ANE shard)
            th = threading.Thread(target=kern[pl].run)         # ANE runs the ANE shard ...
            th.start()
            h[b], ffn_in[b] = self._attn(h[b], self.layers[l], self._mask_for(l, gm, sw))  # GPU attn
            y_gpu = self._gpu_ffn_shard(ffn_in[pb].reshape(-1, self.dim), pl) if split else None
            mx.eval([h[b], ffn_in[b]] + ([y_gpu] if split else []))     # ... + the GPU FFN shard
            th.join()
            y = self._ffn_read(kern[pl], N, h[pb].shape)
            if split:
                y = y + y_gpu.reshape(h[pb].shape)
            h[pb] = self._ffn_combine(h[pb], y, self.layers[pl])

        # drain last FFN
        lb, ll = sched[-1]
        self._ffn_prep(ffn_in[lb], kern[ll], N, Npad)
        y_gpu = self._gpu_ffn_shard(ffn_in[lb].reshape(-1, self.dim), ll) if split else None
        kern[ll].run()
        y = self._ffn_read(kern[ll], N, h[lb].shape)
        if split:
            y = y + y_gpu.reshape(h[lb].shape)
        h[lb] = self._ffn_combine(h[lb], y, self.layers[ll])

        return mx.concatenate([self._finalize(h[0]), self._finalize(h[1])], axis=0)

    def free(self):
        for k in self._kernels.values():
            k.free()
        self._kernels = {}
