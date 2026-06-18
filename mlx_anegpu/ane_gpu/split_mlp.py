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
from . import _bridge as ane

_PROFILE = os.environ.get("ANE_GPU_PROFILE", "") == "1"
ENABLED = True   # runtime toggle: when False, SplitMLP runs the pure-GPU path (for fair A/B)
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

        # GPU shard weights (pre-transposed for x @ W): keep as MLX fp16 arrays
        if self.gpu_h > 0:
            self.W1g = mx.array(np.ascontiguousarray(W1[ane_h:].T))   # [dim, gpu_h]
            self.W3g = mx.array(np.ascontiguousarray(W3[ane_h:].T))   # [dim, gpu_h]
            self.W2g = mx.array(np.ascontiguousarray(W2[:, ane_h:].T))  # [gpu_h, dim]

        # full-FFN GPU fallback weights (for decode / small seq)
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
            g = flat @ self.W1g
            u = flat @ self.W3g
            y_gpu = ((g * mx.sigmoid(g)) * u) @ self.W2g
            mx.async_eval(y_gpu)
        if P: PROF["gpu_dispatch"] += t() - t0

        # ANE sub-FFN — needs x materialized as fp16 [dim, Npad] (channel-major)
        if P: t0 = t()
        xnp = np.array(flat.astype(mx.float16))          # [N, dim]  (forces eval of x)
        k.inbuf[:, :N] = xnp.T                            # write straight into IOSurface
        if Npad > N:
            k.inbuf[:, N:] = 0
        if P: PROF["materialize"] += t() - t0; t0 = t()
        ok = k.run()                                      # ANE runs concurrently with GPU
        if P: PROF["ane_run"] += t() - t0; t0 = t()
        if not ok:
            PROF["fallback"] += 1
            return self._gpu_full(x)                      # graceful GPU fallback
        y_ane = mx.array(np.ascontiguousarray(k.outbuf[:, :N].T))  # [N, dim]
        if P: PROF["out_copy"] += t() - t0; t0 = t()

        if y_gpu is not None:
            mx.eval(y_gpu)                                # join
            y = y_gpu + y_ane
        else:
            y = y_ane
        if P: PROF["join"] += t() - t0; PROF["n"] += 1
        return y.reshape(shape)


def accelerate(model, ane_frac=0.7, min_seq=1024, verbose=True):
    """Walk an mlx_lm model and replace each TransformerBlock's SwiGLU MLP with a
    SplitMLP that runs the FFN on ANE+GPU concurrently. Returns the same model.

    Only non-quantized gate/up/down Linear MLPs are accelerated; anything else is
    left untouched. Prefill (and batched) FFNs run on both engines; decode falls back
    to GPU automatically.
    """
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        raise ValueError("could not find model.model.layers — unsupported architecture")
    n = 0
    for blk in layers:
        mlp = getattr(blk, "mlp", None)
        if mlp is None or not all(hasattr(mlp, p) for p in ("gate_proj", "up_proj", "down_proj")):
            continue
        gp, up, dn = mlp.gate_proj, mlp.up_proj, mlp.down_proj
        if not (isinstance(gp, nn.Linear) and isinstance(up, nn.Linear) and isinstance(dn, nn.Linear)):
            continue  # quantized or fused — skip
        blk.mlp = SplitMLP(gp.weight, up.weight, dn.weight, ane_frac=ane_frac, min_seq=min_seq)
        n += 1
    if verbose:
        print(f"[ane_gpu] accelerated {n} FFN layers (ane_frac={ane_frac}, min_seq={min_seq})")
    return model
