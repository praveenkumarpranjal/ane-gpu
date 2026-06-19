"""SingleStreamRunner — accelerate a SINGLE prompt (batch=1), no batching needed.

The 2-stream PipelinedRunner needs batch>=2 to overlap micro-batches. This runner instead
exploits that the ANE FFN is ~4x faster than the GPU FFN even at B=1: each layer runs its
attention on the GPU and its FFN on the ANE. The transformer is serial across layers, so the
two engines don't overlap -- but using the faster engine for the FFN still makes prefill
~1.9x faster than 4-bit GPU for prompts >=512 tokens (measured, Qwen2.5-1.5B).

PREFILL only. Decode (1 token/step) is memory-bandwidth-bound and the ANE FFN streams int8
(2x the bytes of 4-bit) so it can't beat 4-bit GPU there -- decode stays plain MLX.

COST MODEL (important): ANE kernels compile per length-bucket (~10s for the layer stack the
first time a bucket is seen; lengths are bucketed to 128 so nearby prompts reuse kernels).
After that, every long prompt in that bucket prefills ~2.3x faster. So this WINS for workloads
that process MANY long prompts of similar length (doc batches, evals, agents, RAG) where the
one-time compile amortizes -- NOT for a single one-off chat turn (which pays the ~10s once).

    runner = SingleStreamRunner(model)
    for tok in runner.generate(prompt_ids, max_tokens=256): ...
"""
import numpy as np
import mlx.core as mx
from . import _native as ane

try:
    from mlx_lm.models.base import create_attention_mask as _mkmask
    from mlx_lm.models.cache import make_prompt_cache
except Exception:
    _mkmask = make_prompt_cache = None

MIN_PREFILL = 256        # below this, ANE hand-off overhead isn't worth it -> plain GPU


def _w_fp16(lin):
    """fp16 [out,in] weight from a Linear or QuantizedLinear."""
    if hasattr(lin, "scales"):       # QuantizedLinear -> dequantize
        w = mx.dequantize(lin.weight, lin.scales, lin.biases,
                          group_size=lin.group_size, bits=lin.bits)
    else:
        w = lin.weight
    return np.ascontiguousarray(np.array(w.astype(mx.float16)))


class SingleStreamRunner:
    def __init__(self, model):
        self.model = model
        self.mdl = model.model
        self.layers = self.mdl.layers
        self.dim = model.args.hidden_size
        self.hidden = model.args.intermediate_size
        self.tied = getattr(model.args, "tie_word_embeddings", False)
        self._kern = {}
        self._ok = {}
        self.enabled = (_mkmask is not None) and ane is not None

    @staticmethod
    def _npad(S):
        return ((S + 127) // 128) * 128       # bucket length -> kernels reuse across turns

    def _kernel(self, li, Npad):
        k = self._kern.get((li, Npad))
        if k is None:
            mlp = self.layers[li].mlp         # extract fp16 weights transiently (no persistent copy)
            k = ane.compile_ffn_int8(self.dim, self.hidden, Npad,
                                     _w_fp16(mlp.gate_proj), _w_fp16(mlp.up_proj), _w_fp16(mlp.down_proj))
            self._kern[(li, Npad)] = k
        return k

    def _ane_ffn(self, li, ffn_in, S, Npad):
        if not self._ok.get((li, Npad), True):
            return self.layers[li].mlp(ffn_in)
        k = self._kernel(li, Npad)
        xt = mx.transpose(ffn_in.reshape(-1, self.dim).astype(mx.float16)); mx.eval(xt)
        k.inbuf[:, :S] = np.array(xt)
        if Npad > S:
            k.inbuf[:, S:] = 0
        k.run()
        return mx.transpose(mx.array(k.outbuf[:, :S])).reshape(ffn_in.shape)

    def _finalize(self, h):
        h = self.mdl.norm(h)
        if self.tied or getattr(self.model, "lm_head", None) is None:
            return self.mdl.embed_tokens.as_linear(h)
        return self.model.lm_head(h)

    def _calibrate(self, x, Npad):
        """One fp16-reference forward: keep int8 per layer only if finite + accurate."""
        S = x.shape[1]
        h = self.mdl.embed_tokens(x); mask = _mkmask(h, None)
        for li, l in enumerate(self.layers):
            h = h + l.self_attn(l.input_layernorm(h), mask, None)
            fin = l.post_attention_layernorm(h)
            k = self._kernel(li, Npad)
            xt = mx.transpose(fin.reshape(-1, self.dim).astype(mx.float16)); mx.eval(xt)
            k.inbuf[:, :S] = np.array(xt)
            if Npad > S:
                k.inbuf[:, S:] = 0
            k.run()
            got = mx.transpose(mx.array(k.outbuf[:, :S])).reshape(fin.shape)
            ref = l.mlp(fin); mx.eval(got, ref)
            self._ok[(li, Npad)] = bool(mx.all(mx.isfinite(got))) and \
                float(mx.mean(mx.abs(got - ref)) / (mx.mean(mx.abs(ref)) + 1e-6)) < 0.1
            h = h + ref

    def prefill(self, tokens, cache):
        """ANE-FFN prefill over `tokens` (a 1-row mx.array), populating `cache`. Returns
        logits for the last position. Falls back to plain GPU for short prompts."""
        S = tokens.shape[1]
        if not self.enabled or S < MIN_PREFILL:
            return self.model(tokens, cache=cache)[:, -1:, :]
        Npad = self._npad(S)
        if (0, Npad) not in self._ok:
            self._calibrate(tokens, Npad)
        h = self.mdl.embed_tokens(tokens)
        mask = _mkmask(h, cache[0])
        for li, l in enumerate(self.layers):
            h = h + l.self_attn(l.input_layernorm(h), mask, cache[li])     # GPU attn (fills cache)
            h = h + self._ane_ffn(li, l.post_attention_layernorm(h), S, Npad)  # ANE FFN
        return self._finalize(h[:, -1:, :])

    def generate(self, prompt_ids, max_tokens=256, eos_id=None):
        """Yield generated token ids. Prefill on the ANE, decode on plain MLX (4-bit if the
        model is quantized) with the KV cache."""
        cache = make_prompt_cache(self.model)
        x = mx.array([list(prompt_ids)])
        logits = self.prefill(x, cache)
        y = int(mx.argmax(logits[0, -1]))
        for _ in range(max_tokens):
            if eos_id is not None and y == eos_id:
                break
            yield y
            logits = self.model(mx.array([[y]]), cache=cache)     # MLX decode step
            y = int(mx.argmax(logits[0, -1]))

    def free(self):
        for k in self._kern.values():
            k.free()
        self._kern = {}
