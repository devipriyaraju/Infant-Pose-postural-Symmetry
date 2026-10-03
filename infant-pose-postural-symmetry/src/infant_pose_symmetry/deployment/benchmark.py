from __future__ import annotations

import time

import torch


def benchmark_callable(fn, x, warmup: int = 20, iterations: int = 100) -> float:
    """Return mean wall-clock seconds per call for a CUDA inference function."""
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iterations
