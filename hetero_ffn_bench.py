#!/usr/bin/env python3
"""
hetero_ffn_bench.py — Strategy A Feasibility Benchmark
=======================================================
Tests whether tensor-splitting an FFN linear layer across GPU + ANE
beats GPU-only on Apple Silicon.

Uses:
  - MLX for GPU matmul (optimized Apple Silicon backend)
  - ane_bridge.dylib (ctypes) for ANE conv-as-linear

Run:
  cd ANE
  source .venv/bin/activate
  python hetero_ffn_bench.py
"""

import ctypes
import ctypes.util
import json
import os
import platform
import subprocess
import struct
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

# ============================================================================
# Section 1: ANE Bridge ctypes bindings
# ============================================================================

# Opaque handle
class ANEKernelHandle(ctypes.Structure):
    pass

ANEKernelHandlePtr = ctypes.POINTER(ANEKernelHandle)

def load_ane_bridge():
    """Load libane_bridge.dylib and bind all functions."""
    bridge_path = Path(__file__).parent / "bridge" / "libane_bridge.dylib"
    if not bridge_path.exists():
        print(f"ERROR: Bridge dylib not found at {bridge_path}")
        print("Run: cd bridge && make")
        sys.exit(1)

    lib = ctypes.CDLL(str(bridge_path))

    # int ane_bridge_init(void)
    lib.ane_bridge_init.argtypes = []
    lib.ane_bridge_init.restype = ctypes.c_int

    # ANEKernelHandle *ane_bridge_compile(...)
    lib.ane_bridge_compile.argtypes = [
        ctypes.c_char_p,                    # mil_text
        ctypes.c_size_t,                    # mil_len
        ctypes.POINTER(ctypes.c_uint8),     # weight_data
        ctypes.c_size_t,                    # weight_len
        ctypes.c_int,                       # n_inputs
        ctypes.POINTER(ctypes.c_size_t),    # input_sizes
        ctypes.c_int,                       # n_outputs
        ctypes.POINTER(ctypes.c_size_t),    # output_sizes
    ]
    lib.ane_bridge_compile.restype = ANEKernelHandlePtr

    # bool ane_bridge_eval(ANEKernelHandle *kernel)
    lib.ane_bridge_eval.argtypes = [ANEKernelHandlePtr]
    lib.ane_bridge_eval.restype = ctypes.c_bool

    # void ane_bridge_write_input(ANEKernelHandle *kernel, int idx, const void *data, size_t bytes)
    lib.ane_bridge_write_input.argtypes = [
        ANEKernelHandlePtr, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t
    ]
    lib.ane_bridge_write_input.restype = None

    # void ane_bridge_read_output(ANEKernelHandle *kernel, int idx, void *data, size_t bytes)
    lib.ane_bridge_read_output.argtypes = [
        ANEKernelHandlePtr, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t
    ]
    lib.ane_bridge_read_output.restype = None

    # void ane_bridge_free(ANEKernelHandle *kernel)
    lib.ane_bridge_free.argtypes = [ANEKernelHandlePtr]
    lib.ane_bridge_free.restype = None

    # uint8_t *ane_bridge_build_weight_blob(const float *src, int rows, int cols, size_t *out_len)
    lib.ane_bridge_build_weight_blob.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t)
    ]
    lib.ane_bridge_build_weight_blob.restype = ctypes.POINTER(ctypes.c_uint8)

    # uint8_t *ane_bridge_build_weight_blob_transposed(...)
    lib.ane_bridge_build_weight_blob_transposed.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t)
    ]
    lib.ane_bridge_build_weight_blob_transposed.restype = ctypes.POINTER(ctypes.c_uint8)

    # int ane_bridge_get_compile_count(void)
    lib.ane_bridge_get_compile_count.argtypes = []
    lib.ane_bridge_get_compile_count.restype = ctypes.c_int

    # void ane_bridge_reset_compile_count(void)
    lib.ane_bridge_reset_compile_count.argtypes = []
    lib.ane_bridge_reset_compile_count.restype = None

    # uint32_t ane_bridge_get_input_surface_id(ANEKernelHandle *kernel, int idx)
    lib.ane_bridge_get_input_surface_id.argtypes = [ANEKernelHandlePtr, ctypes.c_int]
    lib.ane_bridge_get_input_surface_id.restype = ctypes.c_uint32

    # uint32_t ane_bridge_get_output_surface_id(ANEKernelHandle *kernel, int idx)
    lib.ane_bridge_get_output_surface_id.argtypes = [ANEKernelHandlePtr, ctypes.c_int]
    lib.ane_bridge_get_output_surface_id.restype = ctypes.c_uint32

    return lib


# ============================================================================
# Section 2: MIL Generator (Python port of ane_mil_gen.h patterns)
# ============================================================================

MIL_HEADER = (
    'program(1.3)\n'
    '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, '
    '{"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, '
    '{"coremltools-version", "9.0"}})]\n'
    '{\n'
)

def gen_mil_conv_linear(in_ch, out_ch, spatial):
    """
    Generate MIL for a single linear layer implemented as 1x1 conv.
    
    Computes Y = conv(W, X) where:
      X: [1, in_ch, 1, spatial] fp32 input
      W: [out_ch, in_ch, 1, 1] fp16 baked weights (from BLOBFILE)
      Y: [1, out_ch, 1, spatial] fp32 output
    
    This is the ANE pattern for a linear: conv with 1x1 kernel = matmul.
    Weight blob: [64B global header][64B chunk header][fp16 data]
    BLOBFILE offset points to the chunk header (64), NOT the data (128).
    The ANE runtime reads the chunk header (DEADBEEF magic) to locate the data.
    """
    mil = MIL_HEADER
    mil += f'    func main<ios18>(tensor<fp32, [1, {in_ch}, 1, {spatial}]> x) {{\n'

    # Conv constants
    mil += '        string c_pad_type = const()[name = string("c_pad_type"), val = string("valid")];\n'
    mil += '        tensor<int32, [2]> c_strides = const()[name = string("c_strides"), val = tensor<int32, [2]>([1, 1])];\n'
    mil += '        tensor<int32, [4]> c_pad = const()[name = string("c_pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n'
    mil += '        tensor<int32, [2]> c_dilations = const()[name = string("c_dilations"), val = tensor<int32, [2]>([1, 1])];\n'
    mil += '        int32 c_groups = const()[name = string("c_groups"), val = int32(1)];\n'

    # Cast input to fp16
    mil += '        string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];\n'
    mil += f'        tensor<fp16, [1, {in_ch}, 1, {spatial}]> x16 = cast(dtype = to_fp16, x = x)[name = string("cast_in")];\n'

    # Weight from blobfile — offset 64 points to chunk header (64B global + 64B chunk + data)
    mil += (
        f'        tensor<fp16, [{out_ch}, {in_ch}, 1, 1]> W = const()'
        f'[name = string("W"), val = tensor<fp16, [{out_ch}, {in_ch}, 1, 1]>'
        f'(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];\n'
    )

    # Conv
    mil += (
        f'        tensor<fp16, [1, {out_ch}, 1, {spatial}]> y16 = conv('
        f'dilations = c_dilations, groups = c_groups, pad = c_pad, '
        f'pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)'
        f'[name = string("conv")];\n'
    )

    # Cast output to fp32
    mil += '        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];\n'
    mil += f'        tensor<fp32, [1, {out_ch}, 1, {spatial}]> y = cast(dtype = to_fp32, x = y16)[name = string("cast_out")];\n'

    mil += '    } -> (y);\n'
    mil += '}\n'
    return mil


# ============================================================================
# Section 3: ANE Kernel Wrapper
# ============================================================================

class ANEKernel:
    """Wraps ANE bridge compile/eval/free into a Python-friendly interface."""

    def __init__(self, lib, in_ch, out_ch, spatial, weights_f32):
        """
        Compile a conv-linear kernel on ANE.
        
        weights_f32: numpy float32 array of shape [out_ch, in_ch]
        """
        self.lib = lib
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.spatial = spatial

        # Generate MIL
        mil_str = gen_mil_conv_linear(in_ch, out_ch, spatial)
        mil_bytes = mil_str.encode('utf-8')

        # Build weight blob using the bridge helper
        # The bridge builds: 128-byte header + fp16 data
        w_flat = np.ascontiguousarray(weights_f32.flatten(), dtype=np.float32)
        blob_len = ctypes.c_size_t(0)
        blob_ptr = lib.ane_bridge_build_weight_blob(
            w_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(out_ch),
            ctypes.c_int(in_ch),
            ctypes.byref(blob_len),
        )
        if not blob_ptr:
            raise RuntimeError("ane_bridge_build_weight_blob returned NULL")

        # Input/output sizes in bytes (fp32)
        self.input_bytes = in_ch * spatial * 4
        self.output_bytes = out_ch * spatial * 4

        input_sizes = (ctypes.c_size_t * 1)(self.input_bytes)
        output_sizes = (ctypes.c_size_t * 1)(self.output_bytes)

        # Compile
        self.handle = lib.ane_bridge_compile(
            ctypes.c_char_p(mil_bytes),
            ctypes.c_size_t(len(mil_bytes)),
            blob_ptr,
            blob_len,
            ctypes.c_int(1),
            input_sizes,
            ctypes.c_int(1),
            output_sizes,
        )
        if not self.handle:
            raise RuntimeError(
                f"ANE compile failed for conv [{out_ch}, {in_ch}, 1, 1] @ spatial={spatial}. "
                f"Compile count: {lib.ane_bridge_get_compile_count()}"
            )

        # Free the blob (bridge made its own copy)
        ctypes.cdll.LoadLibrary(ctypes.util.find_library("c")).free(blob_ptr)

    def eval(self, input_f32):
        """Run ANE evaluation. input_f32: numpy float32 array, flat or shaped."""
        inp = np.ascontiguousarray(input_f32.flatten(), dtype=np.float32)
        self.lib.ane_bridge_write_input(
            self.handle, 0,
            inp.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(self.input_bytes),
        )
        ok = self.lib.ane_bridge_eval(self.handle)
        if not ok:
            raise RuntimeError("ANE eval failed")
        out = np.empty(self.out_ch * self.spatial, dtype=np.float32)
        self.lib.ane_bridge_read_output(
            self.handle, 0,
            out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(self.output_bytes),
        )
        return out.reshape(1, self.out_ch, 1, self.spatial)

    def eval_write_only(self, input_f32):
        """Write input + eval only (no readback). For timing dispatch + compute."""
        inp = np.ascontiguousarray(input_f32.flatten(), dtype=np.float32)
        self.lib.ane_bridge_write_input(
            self.handle, 0,
            inp.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(self.input_bytes),
        )
        ok = self.lib.ane_bridge_eval(self.handle)
        return ok

    def eval_no_io(self):
        """Eval only, no I/O. For measuring pure dispatch + compute time."""
        return self.lib.ane_bridge_eval(self.handle)

    def free(self):
        if self.handle:
            self.lib.ane_bridge_free(self.handle)
            self.handle = None

    def __del__(self):
        self.free()


# ============================================================================
# Section 4: Benchmark Suite
# ============================================================================

def get_chip_info():
    """Get Apple Silicon chip info."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        brand = result.stdout.strip()
    except Exception:
        brand = "Unknown"

    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        mem_gb = int(result.stdout.strip()) / (1024**3)
    except Exception:
        mem_gb = 0

    try:
        result = subprocess.run(
            ["sw_vers", "-productVersion"],
            capture_output=True, text=True, timeout=5,
        )
        macos = result.stdout.strip()
    except Exception:
        macos = "Unknown"

    return {"chip": brand, "memory_gb": round(mem_gb, 1), "macos": macos}


def bench_gpu_matmul(dim_in, dim_out, seq_len, warmup=10, iters=50):
    """
    Benchmark GPU-only matmul via MLX.
    Computes Y = X @ W^T  where X:[seq_len, dim_in], W:[dim_out, dim_in]
    Returns mean time in microseconds.
    """
    X = mx.random.normal(shape=(seq_len, dim_in))
    W = mx.random.normal(shape=(dim_out, dim_in))
    mx.eval(X, W)

    # Warmup
    for _ in range(warmup):
        Y = X @ W.T
        mx.eval(Y)

    # Benchmark
    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        Y = X @ W.T
        mx.eval(Y)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)  # ns → μs

    return {
        "mean_us": np.mean(times),
        "std_us": np.std(times),
        "min_us": np.min(times),
        "p50_us": np.median(times),
        "p95_us": np.percentile(times, 95),
        "p99_us": np.percentile(times, 99),
    }


def bench_ane_conv_linear(lib, dim_in, dim_out, seq_len, warmup=10, iters=50):
    """
    Benchmark ANE-only conv-as-linear.
    Computes Y = conv(W, X) where X:[1, dim_in, 1, seq_len], W:[dim_out, dim_in, 1, 1]
    Returns mean time in microseconds, or None if compile fails.
    """
    W_np = np.random.randn(dim_out, dim_in).astype(np.float32) * 0.02
    X_np = np.random.randn(1, dim_in, 1, seq_len).astype(np.float32)

    try:
        kernel = ANEKernel(lib, dim_in, dim_out, seq_len, W_np)
    except RuntimeError as e:
        print(f"    ANE compile failed: {e}")
        return None

    # Warmup
    for _ in range(warmup):
        kernel.eval_write_only(X_np)

    # Benchmark (write + eval, no readback — measures dispatch + compute)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        kernel.eval_write_only(X_np)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)

    kernel.free()
    return {
        "mean_us": np.mean(times),
        "std_us": np.std(times),
        "min_us": np.min(times),
        "p50_us": np.median(times),
        "p95_us": np.percentile(times, 95),
        "p99_us": np.percentile(times, 99),
    }


def bench_ane_dispatch_floor(lib, warmup=20, iters=100):
    """
    Measure the ANE dispatch floor — eval a tiny kernel with no meaningful compute.
    Uses a small 16→16 conv at spatial=1 (minimal work).
    """
    W_np = np.random.randn(16, 16).astype(np.float32) * 0.02
    X_np = np.random.randn(1, 16, 1, 1).astype(np.float32)

    try:
        kernel = ANEKernel(lib, 16, 16, 1, W_np)
    except RuntimeError as e:
        print(f"    ANE dispatch floor compile failed: {e}")
        return None

    # Warmup
    for _ in range(warmup):
        kernel.eval_no_io()

    # Benchmark (eval only, no I/O)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        kernel.eval_no_io()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)

    kernel.free()
    return {
        "mean_us": np.mean(times),
        "std_us": np.std(times),
        "min_us": np.min(times),
        "p50_us": np.median(times),
    }


def bench_boundary_cost(dim_in, seq_len, iters=100):
    """
    Measure CPU-mediated boundary cost: time to move data between
    MLX GPU arrays and numpy (which is what ANE bridge reads from).
    
    This simulates the GPU→ANE handoff in the prototype:
      1. mx.eval(gpu_result) — force GPU completion
      2. np.array(gpu_result) — copy GPU→CPU numpy
      3. numpy→ctypes — ready for ane_bridge_write_input
    """
    X_mx = mx.random.normal(shape=(seq_len, dim_in))
    mx.eval(X_mx)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        # This is the boundary: MLX array → numpy
        X_np = np.array(X_mx)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)

    return {
        "mean_us": np.mean(times),
        "std_us": np.std(times),
        "tensor_bytes": dim_in * seq_len * 4,
        "tensor_mb": dim_in * seq_len * 4 / (1024 * 1024),
    }


def bench_split_ffn(lib, dim_in, dim_out, seq_len, split_ratio, warmup=10, iters=50):
    """
    Benchmark tensor-split FFN: GPU does (1-r) of output channels, ANE does r.
    
    GPU shard: X @ W_gpu^T → [seq_len, gpu_out_ch]
    ANE shard: conv(W_ane, X) → [1, ane_out_ch, 1, seq_len]
    CPU: concat results
    
    GPU and ANE run concurrently via threading.
    Returns wall-clock time for the combined operation.
    """
    ane_out_ch = max(16, int(dim_out * split_ratio))
    # Round to nearest multiple of 16 for ANE alignment
    ane_out_ch = (ane_out_ch // 16) * 16
    if ane_out_ch < 16:
        ane_out_ch = 16
    gpu_out_ch = dim_out - ane_out_ch

    if gpu_out_ch <= 0 or ane_out_ch <= 0:
        return None

    # Prepare weights
    W_full = np.random.randn(dim_out, dim_in).astype(np.float32) * 0.02
    W_gpu_np = W_full[:gpu_out_ch, :]
    W_ane_np = W_full[gpu_out_ch:, :]

    # MLX arrays
    X_mx = mx.array(np.random.randn(seq_len, dim_in).astype(np.float32))
    W_gpu_mx = mx.array(W_gpu_np)
    mx.eval(X_mx, W_gpu_mx)

    # ANE kernel
    try:
        ane_kernel = ANEKernel(lib, dim_in, ane_out_ch, seq_len, W_ane_np)
    except RuntimeError as e:
        print(f"    ANE split compile failed (ane_out_ch={ane_out_ch}): {e}")
        return None

    # Prepare ANE input (X in [1, C, 1, S] layout — channel-first)
    # MLX/numpy X is [seq_len, dim_in], ANE wants [1, dim_in, 1, seq_len]
    X_np_ane = np.random.randn(1, dim_in, 1, seq_len).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        Y_gpu = X_mx @ W_gpu_mx.T
        mx.eval(Y_gpu)
        ane_kernel.eval_write_only(X_np_ane)

    # Benchmark: concurrent GPU + ANE
    times = []
    ane_result_holder = [None]  # For thread to write into

    for _ in range(iters):
        def ane_work():
            ane_kernel.eval_write_only(X_np_ane)

        t0 = time.perf_counter_ns()

        # Launch ANE in background thread
        ane_thread = threading.Thread(target=ane_work)
        ane_thread.start()

        # GPU work on main thread
        Y_gpu = X_mx @ W_gpu_mx.T
        mx.eval(Y_gpu)

        # Wait for ANE
        ane_thread.join()

        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)

    ane_kernel.free()

    return {
        "mean_us": np.mean(times),
        "std_us": np.std(times),
        "min_us": np.min(times),
        "p50_us": np.median(times),
        "p95_us": np.percentile(times, 95),
        "split_ratio": split_ratio,
        "ane_out_ch": ane_out_ch,
        "gpu_out_ch": gpu_out_ch,
    }


def verify_correctness(lib, dim_in, dim_out, seq_len):
    """
    Verify that GPU-only and GPU+ANE split produce matching results.
    Returns max absolute error.
    """
    W_full = np.random.randn(dim_out, dim_in).astype(np.float32) * 0.02
    X_np = np.random.randn(seq_len, dim_in).astype(np.float32)

    # GPU-only reference (via numpy for exact comparison)
    Y_ref = X_np @ W_full.T  # [seq_len, dim_out]

    # ANE shard
    ane_out_ch = dim_out // 2
    gpu_out_ch = dim_out - ane_out_ch
    W_ane = W_full[gpu_out_ch:, :]

    try:
        kernel = ANEKernel(lib, dim_in, ane_out_ch, seq_len, W_ane)
    except RuntimeError:
        return None, "ANE compile failed"

    # ANE input layout: [1, dim_in, 1, seq_len] — X transposed to channel-first
    X_ane = X_np.T.reshape(1, dim_in, 1, seq_len).copy()
    Y_ane = kernel.eval(X_ane)  # [1, ane_out_ch, 1, seq_len]
    kernel.free()

    # ANE output is [1, out_ch, 1, spatial] — convert back to [seq_len, out_ch]
    Y_ane_flat = Y_ane.reshape(ane_out_ch, seq_len).T  # [seq_len, ane_out_ch]

    # GPU shard (via numpy)
    W_gpu = W_full[:gpu_out_ch, :]
    Y_gpu = X_np @ W_gpu.T  # [seq_len, gpu_out_ch]

    # Combined
    Y_combined = np.concatenate([Y_gpu, Y_ane_flat], axis=1)  # [seq_len, dim_out]

    max_err = np.max(np.abs(Y_ref - Y_combined))
    mean_err = np.mean(np.abs(Y_ref - Y_combined))

    return {"max_abs_error": float(max_err), "mean_abs_error": float(mean_err)}, None


# ============================================================================
# Section 5: Main Benchmark Runner + Reporting
# ============================================================================

# Model dimension configs to test
CONFIGS = [
    {"name": "Stories110M", "dim": 768, "hidden": 2048, "seq": 256},
    {"name": "Qwen3-0.6B", "dim": 1024, "hidden": 2816, "seq": 256},
    {"name": "Llama-7B (s128)", "dim": 4096, "hidden": 11008, "seq": 128},
    {"name": "Llama-7B (s256)", "dim": 4096, "hidden": 11008, "seq": 256},
]

SPLIT_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def print_header(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def main():
    print("=" * 72)
    print("  Strategy A Feasibility Benchmark — Tensor-Split FFN (GPU+ANE)")
    print("=" * 72)

    # System info
    chip_info = get_chip_info()
    print(f"\n  Chip:    {chip_info['chip']}")
    print(f"  Memory:  {chip_info['memory_gb']} GB")
    print(f"  macOS:   {chip_info['macos']}")
    print(f"  Python:  {sys.version.split()[0]}")
    print(f"  MLX:     {mx.__version__}")

    # Initialize ANE bridge
    lib = load_ane_bridge()
    ret = lib.ane_bridge_init()
    if ret != 0:
        print("\nFATAL: ane_bridge_init() failed. Are you on Apple Silicon with macOS 15+?")
        sys.exit(1)
    print(f"\n  ANE bridge initialized ✓")
    lib.ane_bridge_reset_compile_count()

    all_results = {
        "system": chip_info,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "benchmarks": {},
    }

    # ── Benchmark 1: ANE Dispatch Floor ──────────────────────────────────
    print_header("Benchmark 1: ANE Dispatch Floor")
    floor = bench_ane_dispatch_floor(lib)
    if floor:
        print(f"  Dispatch floor (16→16 conv, spatial=1, eval-only):")
        print(f"    Mean: {floor['mean_us']:.1f} μs")
        print(f"    Std:  {floor['std_us']:.1f} μs")
        print(f"    Min:  {floor['min_us']:.1f} μs")
        print(f"    p50:  {floor['p50_us']:.1f} μs")
        all_results["benchmarks"]["dispatch_floor"] = floor
    else:
        print("  FAILED — could not measure dispatch floor")

    # ── Benchmark 2: Boundary Cost ───────────────────────────────────────
    print_header("Benchmark 2: CPU-Mediated Boundary Cost (MLX→numpy)")
    boundary_results = {}
    for cfg in CONFIGS:
        cost = bench_boundary_cost(cfg["dim"], cfg["seq"])
        print(f"  {cfg['name']:20s}  {cost['tensor_mb']:.1f} MB → {cost['mean_us']:.0f} μs (±{cost['std_us']:.0f})")
        boundary_results[cfg["name"]] = cost
    all_results["benchmarks"]["boundary_cost"] = boundary_results

    # ── Benchmark 3: Correctness Verification ────────────────────────────
    print_header("Benchmark 3: Correctness Verification")
    for cfg in CONFIGS[:2]:  # Just verify small configs
        errs, fail_reason = verify_correctness(lib, cfg["dim"], cfg["hidden"], cfg["seq"])
        if errs:
            print(f"  {cfg['name']:20s}  max_err={errs['max_abs_error']:.6f}  mean_err={errs['mean_abs_error']:.6f}  "
                  f"{'✓ PASS' if errs['max_abs_error'] < 1.0 else '✗ FAIL'}")
        else:
            print(f"  {cfg['name']:20s}  {fail_reason}")
    print(f"  (fp16 tolerance note: errors up to ~0.1 are expected from fp16 rounding)")

    # ── Benchmark 4: GPU-only vs ANE-only Baselines ──────────────────────
    print_header("Benchmark 4: GPU-only vs ANE-only Baselines")
    print(f"\n  {'Config':20s}  {'GPU μs':>10s}  {'ANE μs':>10s}  {'GFLOP':>8s}  {'GPU TFLOPS':>10s}  {'ANE TFLOPS':>10s}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}")

    baseline_results = {}
    for cfg in CONFIGS:
        dim_in = cfg["dim"]
        dim_out = cfg["hidden"]
        seq = cfg["seq"]
        gflops = 2.0 * dim_in * dim_out * seq / 1e9

        # GPU baseline
        gpu = bench_gpu_matmul(dim_in, dim_out, seq)

        # ANE baseline
        ane = bench_ane_conv_linear(lib, dim_in, dim_out, seq)

        gpu_tflops = gflops / (gpu["mean_us"] / 1e6) / 1e3 if gpu["mean_us"] > 0 else 0
        ane_tflops = gflops / (ane["mean_us"] / 1e6) / 1e3 if ane and ane["mean_us"] > 0 else 0

        ane_str = f"{ane['mean_us']:.0f}" if ane else "FAIL"
        ane_tf_str = f"{ane_tflops:.2f}" if ane else "FAIL"

        print(f"  {cfg['name']:20s}  {gpu['mean_us']:>10.0f}  {ane_str:>10s}  {gflops:>8.2f}  {gpu_tflops:>10.2f}  {ane_tf_str:>10s}")

        baseline_results[cfg["name"]] = {
            "config": cfg,
            "gflops": gflops,
            "gpu": gpu,
            "ane": ane,
            "gpu_tflops": gpu_tflops,
            "ane_tflops": ane_tflops,
        }
    all_results["benchmarks"]["baselines"] = baseline_results

    # ── Benchmark 5: Tensor-Split Sweep ──────────────────────────────────
    print_header("Benchmark 5: Tensor-Split FFN Sweep")

    split_results = {}
    for cfg in CONFIGS:
        dim_in = cfg["dim"]
        dim_out = cfg["hidden"]
        seq = cfg["seq"]
        name = cfg["name"]

        gpu_baseline = baseline_results[name]["gpu"]["mean_us"]

        print(f"\n  ── {name} (dim={dim_in}, hidden={dim_out}, seq={seq}) ──")
        print(f"  GPU-only baseline: {gpu_baseline:.0f} μs")
        print(f"  {'Ratio':>7s}  {'ANE ch':>7s}  {'GPU ch':>7s}  {'Split μs':>10s}  {'Speedup':>8s}  {'Verdict':>8s}")
        print(f"  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*8}  {'-'*8}")

        cfg_split_results = []
        best_speedup = 0
        best_ratio = None

        for r in SPLIT_RATIOS:
            result = bench_split_ffn(lib, dim_in, dim_out, seq, r)
            if result is None:
                print(f"  {r:>7.1f}  {'FAIL':>7s}")
                continue

            speedup = gpu_baseline / result["mean_us"]
            verdict = "✓ WIN" if speedup > 1.0 else "✗ LOSE"

            if speedup > best_speedup:
                best_speedup = speedup
                best_ratio = r

            print(f"  {r:>7.1f}  {result['ane_out_ch']:>7d}  {result['gpu_out_ch']:>7d}  "
                  f"{result['mean_us']:>10.0f}  {speedup:>7.2f}x  {verdict:>8s}")

            result["speedup"] = speedup
            cfg_split_results.append(result)

        if best_ratio is not None:
            print(f"\n  Best: ratio={best_ratio:.1f} → {best_speedup:.2f}x {'(wins!)' if best_speedup > 1.0 else '(still loses)'}")

        split_results[name] = {
            "gpu_baseline_us": gpu_baseline,
            "splits": cfg_split_results,
            "best_speedup": best_speedup,
            "best_ratio": best_ratio,
        }

    all_results["benchmarks"]["split_sweep"] = split_results

    # ── Summary ──────────────────────────────────────────────────────────
    print_header("Summary")

    any_win = False
    for name, data in split_results.items():
        status = "✓ WIN" if data["best_speedup"] > 1.0 else "✗ NO WIN"
        if data["best_speedup"] > 1.0:
            any_win = True
        print(f"  {name:20s}  best={data['best_speedup']:.2f}x @ ratio={data['best_ratio']}  {status}")

    print()
    if any_win:
        print("  ★ Strategy A shows promise! At least one config wins with tensor splitting.")
        print("  → Next: measure with direct IOSurface handoff (eliminates CPU-mediated copy)")
    else:
        print("  ✗ No config wins with CPU-mediated tensor splitting.")
        print("  → This could mean:")
        print("    1. CPU-mediated sync adds too much overhead (try direct IOSurface)")
        print("    2. Dimensions are too small for ANE dispatch floor")
        print("    3. Strategy A may not be viable — consider Strategy B (GPU prefill + ANE decode)")

    compile_count = lib.ane_bridge_get_compile_count()
    print(f"\n  ANE compiles used: {compile_count}/119")

    # Save raw results
    results_path = Path(__file__).parent / "hetero_ffn_results.json"
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert_numpy)
    print(f"\n  Raw results saved to: {results_path}")
    print()


if __name__ == "__main__":
    main()
