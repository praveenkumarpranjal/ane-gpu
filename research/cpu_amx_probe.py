"""De-risk lever #3: recruit the CPU's AMX units as a third FFN worker (token-split).

While the ANE+GPU run, the P-cores are idle. Route a token slice of the FFN through
Accelerate (numpy fp32 GEMM uses AMX and releases the GIL) on a worker thread.

Finding: the threading overlaps the ANE fine, BUT numpy/Accelerate fp32 only reaches
~1.1 TFLOPS -> ~11x slower than the int8 ANE (12.8 TFLOPS). So the CPU can only carry
~8% of tokens before it becomes the bottleneck -> ~1.09x on the FFN stage, and the GPU
attention floor (6.8 ms/layer) caps it anyway. A real fp16 BNNS path (~2 TFLOPS) would
reach ~1.12x. Marginal and floor-limited; not worth the build on M4-base.
"""
import sys, os, time, threading
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from anegpu import _native as ane

D, H, N = 896, 4864, 4096

print("CPU(numpy fp32 / Accelerate AMX) FFN throughput for a token slice:")
for frac in (0.08, 0.135, 0.20):
    Nc = int(N * frac)
    W1 = (np.random.randn(D, H) * 0.05).astype(np.float32)
    W3 = (np.random.randn(D, H) * 0.05).astype(np.float32)
    W2 = (np.random.randn(H, D) * 0.03).astype(np.float32)
    xc = (np.random.randn(Nc, D) * 0.5).astype(np.float32)
    def cpu_ffn():
        g = xc @ W1; u = xc @ W3
        return ((g * (1 / (1 + np.exp(-g)))) * u) @ W2
    for _ in range(3): cpu_ffn()
    t0 = time.perf_counter()
    for _ in range(20): cpu_ffn()
    us = (time.perf_counter() - t0) / 20 * 1e6
    gflop = 2 * (D * H * 3) * Nc / 1e9
    print(f"  {frac:5.0%} tokens (Nc={Nc:4d}): {us:7.0f} us  {gflop/(us/1e6)/1e3:.2f} TFLOPS")

# overlap check: ANE int8 FFN (rest of tokens) || CPU FFN (slice) on a thread
frac = 0.08
Nc = int(N * frac); Na = N - Nc
def npw(*s): return (np.random.randn(*s) * 0.05).astype(np.float16)
ka = ane.compile_ffn_int8(D, H, ((Na + 31)//32)*32, npw(H, D), npw(H, D), npw(D, H))
ka.inbuf[:, :Na] = npw(D, Na)
W1 = (np.random.randn(D, H)*0.05).astype(np.float32); W3 = (np.random.randn(D, H)*0.05).astype(np.float32); W2 = (np.random.randn(H, D)*0.03).astype(np.float32)
xc = (np.random.randn(Nc, D)*0.5).astype(np.float32)
def cpu_ffn(): g = xc@W1; u = xc@W3; return ((g*(1/(1+np.exp(-g))))*u)@W2
for _ in range(5): cpu_ffn(); ka.run()
def t(fn, it=20):
    t0 = time.perf_counter()
    for _ in range(it): fn()
    return (time.perf_counter()-t0)/it*1e3
ta, tc = t(ka.run), t(cpu_ffn)
t0 = time.perf_counter()
for _ in range(20):
    th = threading.Thread(target=cpu_ffn); th.start(); ka.run(); th.join()
tov = (time.perf_counter()-t0)/20*1e3
print(f"\nANE int8 FFN ({1-frac:.0%}): {ta:.2f} ms | CPU FFN ({frac:.0%}): {tc:.2f} ms")
print(f"SERIAL {ta+tc:.2f} ms | OVERLAPPED {tov:.2f} ms -> {(ta+tc)/tov:.2f}x  ({'overlap OK' if tov<0.85*(ta+tc) else 'serialized'})")
ka.free()
