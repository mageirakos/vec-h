#!/usr/bin/env python3
"""
Flush CPU L2/L3 caches between benchmark repetitions.

Port of MaxBench flush_caches() from benchmarks/utils.hpp.
Allocates a 256 MB buffer (larger than any typical L3 cache: 32-64 MB)
and reads it with cache-line stride to evict existing cache lines.

Usage:
    # Standalone (between benchmark reps):
    python3 flush_caches.py

    # Importable:
    from flush_caches import flush_cpu_caches
    flush_cpu_caches()
"""

import ctypes
import sys

# 256 MB — covers L3 caches up to 256 MB (GH200 Grace has ~114 MB L3)
FLUSH_BYTES = 256 * 1024 * 1024
CACHE_LINE_SIZE = 64


def flush_cpu_caches(flush_bytes: int = FLUSH_BYTES) -> None:
    """Evict L2/L3 cache contents by reading a large buffer with cache-line stride."""
    buf = bytearray(flush_bytes)
    # Write to force physical page allocation
    ctypes.memset((ctypes.c_char * flush_bytes).from_buffer(buf), 1, flush_bytes)
    # Read with cache-line stride to pollute all cache sets
    acc = 0
    mv = memoryview(buf)
    for i in range(0, flush_bytes, CACHE_LINE_SIZE):
        acc += mv[i]
    # Prevent dead-code elimination
    if acc == -1:
        print(acc, file=sys.stderr)


if __name__ == "__main__":
    flush_cpu_caches()
    print("CPU L2/L3 caches flushed (256 MB sweep)")
