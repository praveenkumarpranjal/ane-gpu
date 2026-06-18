#!/usr/bin/env python3
"""
ane_dim_sweep.py — ANE Channel Dimension Anomaly Profiler
==========================================================
Sweeps ANE out_ch and in_ch dimensions to find:
  1. Which dimensions compile successfully
  2. Which are fast vs slow (throughput anomalies)
  3. Alignment patterns (multiples of 16/32/64/128/256/512)
  4. Dimensions to avoid in tensor-split FFN

Uses os.execv() restart every ~90 compiles to stay under the ~119 limit.

Run:
  cd ANE
  source .venv/bin/activate
  python ane_dim_sweep.py
"""

import ctypes
import ctypes.util
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ============================================================================
# ANE Bridge bindings (same as hetero_ffn_bench.py)
# ============================================================================

class ANEKernelHandle(ctypes.Structure):
    pass

ANEKernelHandlePtr = ctypes.POINTER(ANEKernelHandle)

MIL_HEADER = (
    'program(1.3)\n'
    '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, '
    '{"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, '
    '{"coremltools-version", "9.0"}})]\n'
    '{\n'
)


def load_bridge():
    bridge_path = Path(__file__).resolve().parent.parent / "bridge" / "libane_bridge.dylib"
    if not bridge_path.exists():
        print(f"ERROR: {bridge_path} not found. Run: cd bridge && make")
        sys.exit(1)
    lib = ctypes.CDLL(str(bridge_path))

    lib.ane_bridge_init.argtypes = []
    lib.ane_bridge_init.restype = ctypes.c_int

    lib.ane_bridge_compile.argtypes = [
        ctypes.c_char_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.ane_bridge_compile.restype = ANEKernelHandlePtr

    lib.ane_bridge_eval.argtypes = [ANEKernelHandlePtr]
    lib.ane_bridge_eval.restype = ctypes.c_bool

    lib.ane_bridge_write_input.argtypes = [
        ANEKernelHandlePtr, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t
    ]
    lib.ane_bridge_write_input.restype = None

    lib.ane_bridge_free.argtypes = [ANEKernelHandlePtr]
    lib.ane_bridge_free.restype = None

    lib.ane_bridge_build_weight_blob.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t)
    ]
    lib.ane_bridge_build_weight_blob.restype = ctypes.POINTER(ctypes.c_uint8)

    lib.ane_bridge_get_compile_count.argtypes = []
    lib.ane_bridge_get_compile_count.restype = ctypes.c_int

    lib.ane_bridge_reset_compile_count.argtypes = []
    lib.ane_bridge_reset_compile_count.restype = None

    return lib


def gen_mil(in_ch, out_ch, spatial):
    mil = MIL_HEADER
    mil += f'    func main<ios18>(tensor<fp32, [1, {in_ch}, 1, {spatial}]> x) {{\n'
    mil += '        string c_pad_type = const()[name = string("c_pad_type"), val = string("valid")];\n'
    mil += '        tensor<int32, [2]> c_strides = const()[name = string("c_strides"), val = tensor<int32, [2]>([1, 1])];\n'
    mil += '        tensor<int32, [4]> c_pad = const()[name = string("c_pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n'
    mil += '        tensor<int32, [2]> c_dilations = const()[name = string("c_dilations"), val = tensor<int32, [2]>([1, 1])];\n'
    mil += '        int32 c_groups = const()[name = string("c_groups"), val = int32(1)];\n'
    mil += '        string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];\n'
    mil += f'        tensor<fp16, [1, {in_ch}, 1, {spatial}]> x16 = cast(dtype = to_fp16, x = x)[name = string("cast_in")];\n'
    mil += (
        f'        tensor<fp16, [{out_ch}, {in_ch}, 1, 1]> W = const()'
        f'[name = string("W"), val = tensor<fp16, [{out_ch}, {in_ch}, 1, 1]>'
        f'(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];\n'
    )
    mil += (
        f'        tensor<fp16, [1, {out_ch}, 1, {spatial}]> y16 = conv('
        f'dilations = c_dilations, groups = c_groups, pad = c_pad, '
        f'pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)'
        f'[name = string("conv")];\n'
    )
    mil += '        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];\n'
    mil += f'        tensor<fp32, [1, {out_ch}, 1, {spatial}]> y = cast(dtype = to_fp32, x = y16)[name = string("cast_out")];\n'
    mil += '    } -> (y);\n}\n'
    return mil


def bench_one(lib, in_ch, out_ch, spatial, warmup=5, iters=20):
    """Compile and benchmark one config. Returns dict or None on failure."""
    W_np = np.random.randn(out_ch, in_ch).astype(np.float32) * 0.02
    mil_str = gen_mil(in_ch, out_ch, spatial)
    mil_bytes = mil_str.encode('utf-8')

    blob_len = ctypes.c_size_t(0)
    w_flat = np.ascontiguousarray(W_np.flatten(), dtype=np.float32)
    blob_ptr = lib.ane_bridge_build_weight_blob(
        w_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(out_ch), ctypes.c_int(in_ch),
        ctypes.byref(blob_len),
    )
    if not blob_ptr:
        return None

    input_bytes = in_ch * spatial * 4
    output_bytes = out_ch * spatial * 4
    input_sizes = (ctypes.c_size_t * 1)(input_bytes)
    output_sizes = (ctypes.c_size_t * 1)(output_bytes)

    handle = lib.ane_bridge_compile(
        ctypes.c_char_p(mil_bytes), ctypes.c_size_t(len(mil_bytes)),
        blob_ptr, blob_len,
        ctypes.c_int(1), input_sizes,
        ctypes.c_int(1), output_sizes,
    )

    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    libc.free(blob_ptr)

    if not handle:
        return {"compile_ok": False}

    # Prepare input
    X_np = np.random.randn(1, in_ch, 1, spatial).astype(np.float32)
    inp = np.ascontiguousarray(X_np.flatten(), dtype=np.float32)

    # Warmup
    for _ in range(warmup):
        lib.ane_bridge_write_input(handle, 0, inp.ctypes.data_as(ctypes.c_void_p),
                                   ctypes.c_size_t(input_bytes))
        lib.ane_bridge_eval(handle)

    # Benchmark
    times = []
    for _ in range(iters):
        lib.ane_bridge_write_input(handle, 0, inp.ctypes.data_as(ctypes.c_void_p),
                                   ctypes.c_size_t(input_bytes))
        t0 = time.perf_counter_ns()
        lib.ane_bridge_eval(handle)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)

    lib.ane_bridge_free(handle)

    gflops = 2.0 * in_ch * out_ch * spatial / 1e9
    mean_us = np.mean(times)
    tflops = gflops / (mean_us / 1e6) / 1e3 if mean_us > 0 else 0

    return {
        "compile_ok": True,
        "mean_us": float(mean_us),
        "std_us": float(np.std(times)),
        "min_us": float(np.min(times)),
        "tflops": float(tflops),
        "gflops": float(gflops),
    }


# ============================================================================
# Sweep definitions
# ============================================================================

SWEEPS = [
    {
        "name": "out_ch_sweep_768",
        "desc": "out_ch sweep (in_ch=768, spatial=256) — Stories110M scale",
        "in_ch": 768,
        "spatial": 256,
        "variable": "out_ch",
        "values": list(range(16, 2576, 16)),  # 16 to 2560 step 16
    },
    {
        "name": "out_ch_sweep_4096",
        "desc": "out_ch sweep (in_ch=4096, spatial=128) — Llama-7B scale",
        "in_ch": 4096,
        "spatial": 128,
        "variable": "out_ch",
        "values": list(range(64, 11264, 64)),  # 64 to 11200 step 64
    },
    {
        "name": "in_ch_sweep_512",
        "desc": "in_ch sweep (out_ch=512, spatial=256) — lane width probe",
        "out_ch": 512,
        "spatial": 256,
        "variable": "in_ch",
        "values": list(range(16, 4352, 16)),  # 16 to 4336 step 16
    },
]

RESULTS_FILE = Path(__file__).parent / "ane_dim_sweep_results.csv"
COMPILES_PER_BATCH = 85  # restart before 119 limit


def run_sweep_batch(sweep_name, start_idx):
    """Run a batch of benchmarks for one sweep, starting at start_idx."""
    lib = load_bridge()
    ret = lib.ane_bridge_init()
    if ret != 0:
        print("FATAL: ane_bridge_init() failed")
        sys.exit(1)
    lib.ane_bridge_reset_compile_count()

    # Find sweep config
    sweep = None
    for s in SWEEPS:
        if s["name"] == sweep_name:
            sweep = s
            break
    if not sweep:
        print(f"Unknown sweep: {sweep_name}")
        sys.exit(1)

    values = sweep["values"]
    in_ch = sweep.get("in_ch", None)
    out_ch = sweep.get("out_ch", None)
    spatial = sweep["spatial"]
    variable = sweep["variable"]

    # Open CSV for append
    file_exists = RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "sweep", "in_ch", "out_ch", "spatial", "compile_ok",
                "mean_us", "std_us", "min_us", "tflops", "gflops",
            ])

        compiles = 0
        idx = start_idx
        while idx < len(values) and compiles < COMPILES_PER_BATCH:
            val = values[idx]
            if variable == "out_ch":
                cur_in, cur_out = in_ch, val
            else:
                cur_in, cur_out = val, out_ch

            sys.stdout.write(f"\r  [{sweep_name}] {idx+1}/{len(values)}: "
                           f"in_ch={cur_in} out_ch={cur_out} sp={spatial} ... ")
            sys.stdout.flush()

            result = bench_one(lib, cur_in, cur_out, spatial)

            if result is None:
                writer.writerow([sweep_name, cur_in, cur_out, spatial,
                               False, "", "", "", "", ""])
                sys.stdout.write("BLOB_FAIL\n")
            elif not result["compile_ok"]:
                writer.writerow([sweep_name, cur_in, cur_out, spatial,
                               False, "", "", "", "", ""])
                sys.stdout.write("COMPILE_FAIL\n")
                compiles += 1
            else:
                writer.writerow([sweep_name, cur_in, cur_out, spatial,
                               True, f"{result['mean_us']:.1f}",
                               f"{result['std_us']:.1f}",
                               f"{result['min_us']:.1f}",
                               f"{result['tflops']:.3f}",
                               f"{result['gflops']:.3f}"])
                sys.stdout.write(
                    f"{result['mean_us']:.0f} μs  "
                    f"{result['tflops']:.2f} TFLOPS\n"
                )
                compiles += 1

            idx += 1

    return idx  # next index to process


def exec_restart(sweep_name, start_idx):
    """Restart process to reset ANE compile count."""
    print(f"\n  → Restarting process (compile limit) at index {start_idx}...")
    python = sys.executable
    os.execv(python, [python, __file__, "--continue", sweep_name, str(start_idx)])


def run_full_sweep(sweep):
    """Run a complete sweep with exec() restarts as needed."""
    values = sweep["values"]
    total = len(values)
    print(f"\n{'='*60}")
    print(f"  {sweep['desc']}")
    print(f"  {total} configs to test")
    print(f"{'='*60}")

    idx = 0
    while idx < total:
        idx = run_sweep_batch(sweep["name"], idx)
        if idx < total:
            exec_restart(sweep["name"], idx)
            return  # exec_restart replaces the process


def analyze_results():
    """Analyze sweep results and print findings."""
    if not RESULTS_FILE.exists():
        print("No results file found.")
        return

    print(f"\n{'='*72}")
    print(f"  ANE Dimension Sweep — Analysis")
    print(f"{'='*72}")

    rows = []
    with open(RESULTS_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Group by sweep
    sweeps = {}
    for row in rows:
        name = row["sweep"]
        if name not in sweeps:
            sweeps[name] = []
        sweeps[name].append(row)

    for sweep_name, data in sweeps.items():
        total = len(data)
        ok = [r for r in data if r["compile_ok"] == "True"]
        fail = [r for r in data if r["compile_ok"] == "False"]

        print(f"\n  ── {sweep_name} ──")
        print(f"  Total: {total}  Compiled: {len(ok)}  Failed: {len(fail)}")

        if not ok:
            print("  No successful compiles.")
            continue

        # Find throughput stats
        tflops_vals = [float(r["tflops"]) for r in ok]
        mean_tflops = np.mean(tflops_vals)
        max_tflops = np.max(tflops_vals)
        min_tflops = np.min(tflops_vals)

        print(f"  TFLOPS: mean={mean_tflops:.2f}  max={max_tflops:.2f}  min={min_tflops:.2f}")

        # Find anomalies (>2x slower than median)
        median_tflops = np.median(tflops_vals)
        threshold = median_tflops * 0.5  # less than half median throughput

        anomalies = []
        for r in ok:
            tf = float(r["tflops"])
            if tf < threshold:
                variable = "out_ch" if "out_ch_sweep" in sweep_name else "in_ch"
                dim_val = r["out_ch"] if variable == "out_ch" else r["in_ch"]
                anomalies.append((int(dim_val), tf))

        if anomalies:
            print(f"\n  ⚠ Anomalous dimensions ({len(anomalies)} found, <{threshold:.1f} TFLOPS):")
            for dim_val, tf in sorted(anomalies):
                print(f"    {dim_val:>6d}  →  {tf:.2f} TFLOPS  ({tf/median_tflops*100:.0f}% of median)")
        else:
            print(f"\n  ✓ No anomalies found (all >{threshold:.1f} TFLOPS)")

        # Find failed dimensions
        if fail:
            fail_dims = []
            for r in fail:
                variable = "out_ch" if "out_ch_sweep" in sweep_name else "in_ch"
                dim_val = r["out_ch"] if variable == "out_ch" else r["in_ch"]
                fail_dims.append(int(dim_val))
            print(f"\n  ✗ Failed dimensions ({len(fail_dims)}):")
            # Group consecutive failures
            if len(fail_dims) <= 20:
                print(f"    {fail_dims}")
            else:
                print(f"    First 10: {fail_dims[:10]}")
                print(f"    Last 10:  {fail_dims[-10:]}")

        # Find best dimensions
        best = sorted(ok, key=lambda r: float(r["tflops"]), reverse=True)[:5]
        print(f"\n  ★ Top 5 dimensions:")
        for r in best:
            variable = "out_ch" if "out_ch_sweep" in sweep_name else "in_ch"
            dim_val = r["out_ch"] if variable == "out_ch" else r["in_ch"]
            print(f"    {int(dim_val):>6d}  →  {float(r['tflops']):.2f} TFLOPS  ({float(r['mean_us']):.0f} μs)")

        # Alignment analysis
        print(f"\n  Alignment analysis:")
        for align in [16, 32, 64, 128, 256, 512]:
            aligned = [r for r in ok if int(r["out_ch"] if "out_ch_sweep" in sweep_name else r["in_ch"]) % align == 0]
            unaligned = [r for r in ok if int(r["out_ch"] if "out_ch_sweep" in sweep_name else r["in_ch"]) % align != 0]
            if aligned and unaligned:
                avg_aligned = np.mean([float(r["tflops"]) for r in aligned])
                avg_unaligned = np.mean([float(r["tflops"]) for r in unaligned])
                ratio = avg_aligned / avg_unaligned if avg_unaligned > 0 else 0
                marker = " ★" if ratio > 1.1 else ""
                print(f"    mod {align:>4d}: aligned={avg_aligned:.2f} TFLOPS  "
                      f"unaligned={avg_unaligned:.2f} TFLOPS  "
                      f"ratio={ratio:.2f}x{marker}")

    # Save analysis as JSON
    analysis_path = Path(__file__).parent / "ane_dim_sweep_analysis.json"
    analysis = {}
    for sweep_name, data in sweeps.items():
        ok = [r for r in data if r["compile_ok"] == "True"]
        fail = [r for r in data if r["compile_ok"] == "False"]
        analysis[sweep_name] = {
            "total": len(data),
            "compiled": len(ok),
            "failed": len(fail),
            "fail_dims": [int(r.get("out_ch", r.get("in_ch", 0))) for r in fail],
            "best_dim": int(max(ok, key=lambda r: float(r["tflops"])).get("out_ch", 0) or
                          max(ok, key=lambda r: float(r["tflops"])).get("in_ch", 0)) if ok else 0,
            "best_tflops": float(max(ok, key=lambda r: float(r["tflops"]))["tflops"]) if ok else 0,
        }
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Analysis saved to: {analysis_path}")


def main():
    # Handle --continue for exec() restart
    if len(sys.argv) >= 4 and sys.argv[1] == "--continue":
        sweep_name = sys.argv[2]
        start_idx = int(sys.argv[3])
        sweep = None
        for s in SWEEPS:
            if s["name"] == sweep_name:
                sweep = s
                break
        if sweep:
            values = sweep["values"]
            idx = start_idx
            while idx < len(values):
                idx = run_sweep_batch(sweep_name, idx)
                if idx < len(values):
                    exec_restart(sweep_name, idx)
                    return

            # This sweep is done — find and run next sweep
            found_current = False
            for s in SWEEPS:
                if s["name"] == sweep_name:
                    found_current = True
                    continue
                if found_current:
                    run_full_sweep(s)
                    return

            # All sweeps done
            analyze_results()
            return

    # Handle --analyze
    if len(sys.argv) >= 2 and sys.argv[1] == "--analyze":
        analyze_results()
        return

    # Fresh start
    print("=" * 60)
    print("  ANE Dimension Anomaly Sweep")
    print("=" * 60)

    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                         capture_output=True, text=True).stdout.strip()
    print(f"  Chip: {chip}")
    print(f"  Sweeps: {len(SWEEPS)}")
    total_configs = sum(len(s["values"]) for s in SWEEPS)
    print(f"  Total configs: {total_configs}")
    print(f"  Estimated restarts: {total_configs // COMPILES_PER_BATCH}")

    # Delete old results
    if RESULTS_FILE.exists():
        RESULTS_FILE.unlink()

    # Run first sweep (will chain to others via exec_restart or sequential calls)
    run_full_sweep(SWEEPS[0])


if __name__ == "__main__":
    main()
