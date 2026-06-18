"""SplitMLP — a drop-in replacement for an MLX SwiGLU MLP that runs the FFN across
the Apple Neural Engine and the GPU *concurrently*.

  y = down( silu(gate(x)) * up(x) )

The hidden dimension is split: the ANE computes a fused sub-FFN over the first
`ane_h` channels while the GPU (MLX) computes the rest; the two partial outputs are
summed. The ANE call overlaps the GPU sub-FFN via mx.async_eval.

Only engages for compute-bound shapes (total tokens >= min_seq, i.e. prefill / batched).
For single-token decode it transparently falls back to the original GPU MLP.
"""
import os
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from . import _native as ane

try:  # for SplitAttention: the dynamic part (scores/softmax/AV) stays on the GPU
    from mlx_lm.models.base import scaled_dot_product_attention as _sdpa
except Exception:
    _sdpa = None

_PROFILE = os.environ.get("ANEGPU_PROFILE", "") == "1"
ENABLED = True   # runtime toggle: when False, SplitMLP runs the pure-GPU path (for fair A/B)
def set_enabled(v):
    """Globally enable/disable the ANE path (disabled -> SplitMLP runs pure GPU)."""
    global ENABLED
    ENABLED = bool(v)
PROF = {"gpu_dispatch": 0.0, "materialize": 0.0, "ane_run": 0.0, "out_copy": 0.0, "join": 0.0, "n": 0, "fallback": 0}
def prof_reset():
    for k in PROF: PROF[k] = 0.0
def prof_report():
    n = max(PROF["n"], 1)
    parts = [f"{k}={PROF[k]/n*1e6:.0f}us" for k in PROF if k not in ("n", "fallback")]
    return " ".join(parts) + f"  (n={PROF['n']} fallbacks={PROF['fallback']})"


def _r16(v):   # floor to multiple of 16 (for channel splits)
    return max(16, (int(v) // 16) * 16)

# The fused SwiGLU FFN on the ANE only computes correctly when seq is a multiple of 32
# (a multiple of 16 is NOT enough — 16xodd silently returns a Program Inference error).
SEQ_PAD = 32
def _cpad(v):  # ceil to multiple of SEQ_PAD (must fit all tokens)
    return max(SEQ_PAD, ((int(v) + SEQ_PAD - 1) // SEQ_PAD) * SEQ_PAD)


class SplitMLP(nn.Module):
    def __init__(self, gate_w, up_w, down_w, ane_frac=0.7, min_seq=1024):
        super().__init__()
        # weights as fp16 numpy: gate/up [hidden, dim], down [dim, hidden]
        W1 = np.ascontiguousarray(np.array(gate_w.astype(mx.float16)))
        W3 = np.ascontiguousarray(np.array(up_w.astype(mx.float16)))
        W2 = np.ascontiguousarray(np.array(down_w.astype(mx.float16)))
        self.hidden, self.dim = W1.shape
        self.min_seq = min_seq

        ane_h = _r16(ane_frac * self.hidden)
        ane_h = min(ane_h, self.hidden - 16) if ane_frac < 1.0 else self.hidden
        self.ane_h = ane_h
        self.gpu_h = self.hidden - ane_h

        # ANE shard weights (compiled lazily per padded-seq bucket)
        self._W1a = np.ascontiguousarray(W1[:ane_h])
        self._W3a = np.ascontiguousarray(W3[:ane_h])
        self._W2a = np.ascontiguousarray(W2[:, :ane_h])
        self._kernels = {}

        # full FFN weights (GPU). The split GPU shard is a *slice* of these (columns
        # [ane_h:]), and they double as the decode/small-seq fallback — so we don't store
        # a separate shard copy (keeps memory ~1.5x the FFN instead of ~2x, matters for
        # bigger models).
        self.W1f = mx.array(np.ascontiguousarray(W1.T))   # [dim, hidden]
        self.W3f = mx.array(np.ascontiguousarray(W3.T))
        self.W2f = mx.array(np.ascontiguousarray(W2.T))   # [hidden, dim]

    def _kernel(self, seq):
        k = self._kernels.get(seq)
        if k is None:
            k = ane.compile_ffn(self.dim, self.ane_h, seq, self._W1a, self._W3a, self._W2a)
            self._kernels[seq] = k
        return k

    def _gpu_full(self, x):
        g = x @ self.W1f
        u = x @ self.W3f
        return ((g * mx.sigmoid(g)) * u) @ self.W2f

    def __call__(self, x):
        shape = x.shape
        D = shape[-1]
        flat = x.reshape(-1, D)
        N = flat.shape[0]
        if not ENABLED or N < self.min_seq:
            return self._gpu_full(x)

        Npad = _cpad(N)
        k = self._kernel(Npad)
        P = _PROFILE; t = time.perf_counter if P else None

        # GPU sub-FFN (its hidden shard) — dispatched async so it overlaps the ANE
        if P: t0 = t()
        y_gpu = None
        if self.gpu_h > 0:
            h = self.ane_h                                # GPU computes the hidden slice [h:]
            g = flat @ self.W1f[:, h:]
            u = flat @ self.W3f[:, h:]
            y_gpu = ((g * mx.sigmoid(g)) * u) @ self.W2f[h:, :]
            mx.async_eval(y_gpu)
        if P: PROF["gpu_dispatch"] += t() - t0

        # ANE sub-FFN — hand x over as fp16 [dim, Npad] (transpose on the GPU, ~22us,
        # not on the CPU which costs ~3.8ms for a batched activation)
        if P: t0 = t()
        xt = mx.transpose(flat.astype(mx.float16))        # [dim, N] on GPU
        mx.eval(xt)                                       # forces x; the GPU did the transpose
        k.inbuf[:, :N] = np.array(xt)                     # one memcpy into the input IOSurface
        if Npad > N:
            k.inbuf[:, N:] = 0
        if P: PROF["materialize"] += t() - t0; t0 = t()
        ok = k.run()                                      # ANE runs concurrently with GPU
        if P: PROF["ane_run"] += t() - t0; t0 = t()
        if not ok:
            PROF["fallback"] += 1
            return self._gpu_full(x)                      # graceful GPU fallback
        y_ane = mx.transpose(mx.array(k.outbuf[:, :N]))   # [N, dim] (transpose on GPU)
        if P: PROF["out_copy"] += t() - t0; t0 = t()

        if y_gpu is not None:
            mx.eval(y_gpu)                                # join
            y = y_gpu + y_ane
        else:
            y = y_ane
        if P: PROF["join"] += t() - t0; PROF["n"] += 1
        return y.reshape(shape)


class SplitLinear(nn.Module):
    """Drop-in for nn.Linear (y = x @ W.T + bias) with output channels split ANE||GPU.
    Used for the big attention projections (q_proj, o_proj). Falls back to pure GPU
    below min_tokens or on ANE failure. seq is padded to a multiple of 32."""
    def __init__(self, weight, bias=None, ane_frac=0.6, min_tokens=1024):
        super().__init__()
        W = np.ascontiguousarray(np.array(weight.astype(mx.float16)))   # [out, in]
        self.out_, self.in_ = W.shape
        self.min_tokens = min_tokens
        ane_out = _r16(ane_frac * self.out_)
        ane_out = min(ane_out, self.out_ - 16) if ane_frac < 1.0 else self.out_
        self.ane_out, self.gpu_out = ane_out, self.out_ - ane_out
        self._Wa = np.ascontiguousarray(W[:ane_out])                    # ANE shard [ane_out, in]
        self._kernels = {}
        if self.gpu_out > 0:
            self.Wg = mx.array(np.ascontiguousarray(W[ane_out:].T))     # [in, gpu_out]
        self.Wf = mx.array(np.ascontiguousarray(W.T))                   # [in, out] full-GPU fallback
        self.bias = mx.array(np.array(bias.astype(mx.float16))) if bias is not None else None

    def _kernel(self, seq):
        k = self._kernels.get(seq)
        if k is None:
            k = ane.compile_matmul(self.in_, self.ane_out, seq, self._Wa)
            self._kernels[seq] = k
        return k

    def _gpu_full(self, flat, shape):
        y = flat @ self.Wf
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], self.out_)

    def __call__(self, x):
        shape = x.shape
        flat = x.reshape(-1, self.in_)
        N = flat.shape[0]
        if not ENABLED or N < self.min_tokens:
            return self._gpu_full(flat, shape)
        Npad = _cpad(N)
        k = self._kernel(Npad)
        y_gpu = (flat @ self.Wg) if self.gpu_out > 0 else None
        if y_gpu is not None:
            mx.async_eval(y_gpu)
        xt = mx.transpose(flat.astype(mx.float16))         # [in, N] on GPU (not CPU)
        mx.eval(xt)
        k.inbuf[:, :N] = np.array(xt)                      # one memcpy into the input IOSurface
        if Npad > N:
            k.inbuf[:, N:] = 0
        if not k.run():                                    # ANE failure -> graceful GPU fallback
            PROF["fallback"] += 1
            return self._gpu_full(flat, shape)
        y_ane = mx.transpose(mx.array(k.outbuf[:, :N]))    # [N, ane_out] (transpose on GPU)
        if y_gpu is not None:
            mx.eval(y_gpu)
            y = mx.concatenate([y_ane, y_gpu], axis=-1)    # ANE owns channels [0:ane_out]
        else:
            y = y_ane
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], self.out_)


class SplitAttention(nn.Module):
    """Heterogeneous attention: q_proj runs on the ANE *concurrently* with k_proj/v_proj
    on the GPU (the three projections are independent — same input), then the dynamic
    part (RoPE, scores, softmax, AV) and o_proj stay on the GPU. k/v stay on GPU because
    they are only 128 out-channels (the per-op map shows the GPU wins there). Falls back
    to the original attention below min_tokens or when ANE is disabled."""
    def __init__(self, orig, min_tokens=1024):
        super().__init__()
        self.n_heads = orig.n_heads
        self.n_kv_heads = orig.n_kv_heads
        self.scale = orig.scale
        self.rope = orig.rope
        self.k_proj = orig.k_proj
        self.v_proj = orig.v_proj
        self.o_proj = orig.o_proj
        self.q_proj = orig.q_proj                                  # GPU fallback
        self.q_ane = SplitLinear(orig.q_proj.weight, getattr(orig.q_proj, "bias", None),
                                 ane_frac=1.0, min_tokens=min_tokens)   # whole q on ANE
        self.min_tokens = min_tokens

    def __call__(self, x, mask=None, cache=None):
        B, L, D = x.shape
        if ENABLED and _sdpa is not None and B * L >= self.min_tokens:
            keys = self.k_proj(x); values = self.v_proj(x)
            mx.async_eval(keys, values)            # GPU computes k,v ...
            queries = self.q_ane(x)                # ... while the ANE computes q
        else:
            queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        output = _sdpa(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


def accelerate(model, ane_frac=0.7, min_seq=1024, attention=False, verbose=True):
    """Walk an mlx_lm model and run its FFNs (and optionally attention) on ANE+GPU.

    - Each TransformerBlock's SwiGLU MLP -> SplitMLP (ANE+GPU split, the main win).
    - attention="parallel": q_proj on the ANE runs CONCURRENTLY with k/v on the GPU
      (heterogeneous per-op placement from the per-layer benchmark), o/k/v stay GPU.
    - attention="split" (or True): split q_proj+o_proj output channels ANE||GPU.

    NOTE (measured): BOTH attention modes slightly HURT vs FFN-only — "split" 1.23->1.10x,
    "parallel" 3382->3011 tok/s @B32. The attention projections are too small to overcome
    the per-ANE-call sync, even with op-level parallelism. Default attention=False (FFN-only,
    the production-optimal schedule); the modes are kept for research / bigger models.

    Only non-quantized fp16 Linear layers are touched. Decode falls back to GPU.
    """
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        raise ValueError("could not find model.model.layers — unsupported architecture")
    mode = ("parallel" if attention == "parallel" else
            "split" if attention in (True, "split") else None)
    n_mlp = n_attn = 0
    for blk in layers:
        mlp = getattr(blk, "mlp", None)
        if mlp is not None and all(hasattr(mlp, p) for p in ("gate_proj", "up_proj", "down_proj")):
            gp, up, dn = mlp.gate_proj, mlp.up_proj, mlp.down_proj
            if isinstance(gp, nn.Linear) and isinstance(up, nn.Linear) and isinstance(dn, nn.Linear):
                blk.mlp = SplitMLP(gp.weight, up.weight, dn.weight, ane_frac=ane_frac, min_seq=min_seq)
                n_mlp += 1
        attn = getattr(blk, "self_attn", None)
        if mode and attn is not None and isinstance(getattr(attn, "q_proj", None), nn.Linear):
            if mode == "parallel":
                blk.self_attn = SplitAttention(attn, min_tokens=min_seq)
                n_attn += 1
            elif isinstance(getattr(attn, "o_proj", None), nn.Linear):
                attn.q_proj = SplitLinear(attn.q_proj.weight, getattr(attn.q_proj, "bias", None),
                                          ane_frac=ane_frac, min_tokens=min_seq)
                attn.o_proj = SplitLinear(attn.o_proj.weight, getattr(attn.o_proj, "bias", None),
                                          ane_frac=ane_frac, min_tokens=min_seq)
                n_attn += 1
    if verbose:
        extra = f" + {n_attn} attention({mode})" if mode else ""
        print(f"[anegpu] accelerated {n_mlp} FFN{extra} layers (ane_frac={ane_frac}, min_seq={min_seq})")
    return model
