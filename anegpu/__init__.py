"""anegpu — run MLX model FFNs across the Apple Neural Engine + GPU concurrently.

    from mlx_lm import load
    import anegpu
    model, tok = load("Qwen/Qwen2.5-0.5B-Instruct")
    model = anegpu.accelerate(model)   # FFNs now run on ANE+GPU together

Accelerates compute-bound phases (prefill, long context, batched). Single-token
decode falls back to GPU automatically. Apple Silicon only; uses private ANE APIs.
"""
from .accelerate import accelerate, SplitMLP, SplitLinear, SplitAttention, set_enabled
from .pipeline import PipelinedRunner
from . import _native

__all__ = ["accelerate", "SplitMLP", "SplitLinear", "SplitAttention",
           "PipelinedRunner", "set_enabled", "_native"]
__version__ = "0.1.0"
