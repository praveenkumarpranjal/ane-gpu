"""ctypes bridge to libanegpu.dylib — run fused SwiGLU FFN / matmul sub-blocks on the ANE.

All weights are fp16 numpy arrays. Activations are fp16, channel-first [C, seq] (i.e.
x[c, s] contiguous in s). seq must be a multiple of 16 (ANE tiling constraint).
"""
import ctypes
import os
import numpy as np

_DLL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libanegpu.dylib")
_lib = ctypes.CDLL(_DLL)

_c = ctypes.c_void_p
_lib.ane_init.restype = ctypes.c_int
_lib.ane_ffn_compile.restype = _c
_lib.ane_ffn_compile.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, _c, _c, _c]
_lib.ane_ffn_compile_int8.restype = _c
_lib.ane_ffn_compile_int8.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, _c, _c, _c]
_lib.ane_matmul_compile.restype = _c
_lib.ane_matmul_compile.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, _c]
_lib.ane_set_input.argtypes = [_c, _c, ctypes.c_size_t]
_lib.ane_run.restype = ctypes.c_int
_lib.ane_run.argtypes = [_c]
_lib.ane_get_output.argtypes = [_c, _c, ctypes.c_size_t]
_lib.ane_input_ptr.restype = _c
_lib.ane_input_ptr.argtypes = [_c]
_lib.ane_output_ptr.restype = _c
_lib.ane_output_ptr.argtypes = [_c]
_lib.ane_input_bytes.restype = ctypes.c_size_t
_lib.ane_input_bytes.argtypes = [_c]
_lib.ane_output_bytes.restype = ctypes.c_size_t
_lib.ane_output_bytes.argtypes = [_c]
_lib.ane_free.argtypes = [_c]

if _lib.ane_init() != 0:
    raise RuntimeError("ANE init failed (private AppleNeuralEngine.framework unavailable?)")


def _ptr(arr: np.ndarray):
    return ctypes.c_void_p(arr.ctypes.data)


class ANEKernel:
    """A compiled ANE kernel with internal input/output IOSurfaces."""

    def __init__(self, handle, in_ch, out_ch, seq):
        if not handle:
            raise RuntimeError("ANE kernel compile returned NULL")
        self._h = handle
        self.in_ch, self.out_ch, self.seq = in_ch, out_ch, seq
        # numpy views directly over the IOSurface memory (zero-copy in/out)
        ip = _lib.ane_input_ptr(self._h)
        op = _lib.ane_output_ptr(self._h)
        ibytes = _lib.ane_input_bytes(self._h)
        obytes = _lib.ane_output_bytes(self._h)
        self.inbuf = np.ctypeslib.as_array(
            (ctypes.c_uint16 * (ibytes // 2)).from_address(ip)
        ).view(np.float16).reshape(in_ch, seq)
        self.outbuf = np.ctypeslib.as_array(
            (ctypes.c_uint16 * (obytes // 2)).from_address(op)
        ).view(np.float16).reshape(out_ch, seq)

    def set_input(self, x_chan_first: np.ndarray):
        """x_chan_first: fp16 [in_ch, seq] (channel-major). Copies into the input surface."""
        self.inbuf[:] = x_chan_first  # write straight into IOSurface

    def run(self) -> bool:
        """Returns True on success, False on ANE inference failure (caller may fall back)."""
        return _lib.ane_run(self._h) == 0

    def output(self) -> np.ndarray:
        """Returns a view of the output surface, fp16 [out_ch, seq]."""
        return self.outbuf

    def free(self):
        if self._h:
            _lib.ane_free(self._h)
            self._h = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


def compile_ffn(dim, hidden, seq, W1, W3, W2) -> ANEKernel:
    """Fused SwiGLU FFN over `hidden` channels.
    W1 (gate) [hidden,dim], W3 (up) [hidden,dim], W2 (down) [dim,hidden], all fp16 contiguous."""
    W1 = np.ascontiguousarray(W1, dtype=np.float16)
    W3 = np.ascontiguousarray(W3, dtype=np.float16)
    W2 = np.ascontiguousarray(W2, dtype=np.float16)
    h = _lib.ane_ffn_compile(dim, hidden, seq, _ptr(W1), _ptr(W3), _ptr(W2))
    return ANEKernel(h, dim, dim, seq)


def compile_ffn_int8(dim, hidden, seq, W1, W3, W2) -> ANEKernel:
    """Fused SwiGLU FFN with INT8 weights (quantized internally, dequantized in-engine).
    fp16 I/O; streams ~half the weight bytes per eval. ~1.5x faster than fp16, ~1% error."""
    W1 = np.ascontiguousarray(W1, dtype=np.float16)
    W3 = np.ascontiguousarray(W3, dtype=np.float16)
    W2 = np.ascontiguousarray(W2, dtype=np.float16)
    h = _lib.ane_ffn_compile_int8(dim, hidden, seq, _ptr(W1), _ptr(W3), _ptr(W2))
    return ANEKernel(h, dim, dim, seq)


def compile_matmul(in_ch, out_ch, seq, W) -> ANEKernel:
    """Plain matmul y = W @ x. W [out_ch, in_ch] fp16 contiguous."""
    W = np.ascontiguousarray(W, dtype=np.float16)
    h = _lib.ane_matmul_compile(in_ch, out_ch, seq, _ptr(W))
    return ANEKernel(h, in_ch, out_ch, seq)
