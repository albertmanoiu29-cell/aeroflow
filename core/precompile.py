"""Compile every numba and CUDA kernel ahead of time (run by the installer / setup script).

numba caches compiled kernels next to their source files and CuPy caches CUDA binaries per user, so
running this once at install time means the first launch of the app (and of every Lab) starts instantly
instead of compiling for several seconds.

Usage: ``python -m core.precompile``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    t0 = time.perf_counter()
    from core import accel

    dt = accel.warmup()
    print(f"    CPU kernels compiled ({accel.backend()}, {dt:.1f} s)")
    try:
        from core import compute

        info = compute.gpu_info()
        if info.available:
            from core import lbm2d_gpu, nozzle_gpu, supersonic_gpu

            supersonic_gpu._module()
            lbm2d_gpu._module()
            nozzle_gpu.load_module(True)
            nozzle_gpu.load_module(False)
            print("    GPU kernels compiled")
        else:
            print(f"    No GPU kernels to compile ({info.reason or 'no CUDA GPU'})")
    except Exception as exc:  # noqa: BLE001 - optional; compiled on first use instead
        print(f"    GPU kernels skipped: {exc}")
    print(f"    Kernel pre-compilation done in {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
