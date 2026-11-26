"""Compute-device selection: run the live solvers on the GPU (CUDA via CuPy) or on the CPU (numba).

The Supersonic Lab (compressible Euler) and the Live Flow Lab (lattice Boltzmann) have CUDA kernels in
:mod:`core.supersonic_gpu` and :mod:`core.lbm2d_gpu`. This module decides whether they are used:

* :class:`Device` — the user's preference: ``AUTO`` (GPU when one works, else CPU), ``GPU`` or ``CPU``.
* :func:`gpu_info` — probes CUDA once (lazily, on first use) and caches the answer, including *why* the
  GPU is unavailable when it is (no NVIDIA driver, CuPy missing, kernel compiler missing ...).
* :func:`use_gpu` — the resolved decision every solver asks before stepping.

CuPy needs NVIDIA's run-time compiler (NVRTC) and the CUDA headers to build kernels. They come either
from a CUDA Toolkit install (``CUDA_PATH``) or from NVIDIA's pip wheels (``nvidia-cuda-nvrtc-cu12`` and
``nvidia-cuda-runtime-cu12``); :func:`_prepare_cuda_paths` points CuPy at the pip wheels when no
toolkit is installed, so no system-wide CUDA install is needed.

The environment variable ``AEROFLOW_DEVICE`` (``auto`` / ``gpu`` / ``cpu``) overrides the stored
preference (useful for testing and troubleshooting).
"""

from __future__ import annotations

import glob
import logging
import os
import site
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

log = logging.getLogger("aeroflow.core.compute")


class Device(str, Enum):
    """Where the live solvers run."""

    AUTO = "auto"
    GPU = "gpu"
    CPU = "cpu"

    @property
    def label(self) -> str:
        return {"auto": "Auto (GPU if available)", "gpu": "GPU (CUDA)", "cpu": "CPU (numba)"}[self.value]


@dataclass(frozen=True)
class GPUInfo:
    """Result of the one-time CUDA probe."""

    available: bool
    name: str = ""
    memory_gb: float = 0.0
    compute_capability: str = ""
    reason: str = ""            # why the GPU is unavailable (empty when available)

    def summary(self) -> str:
        if self.available:
            return f"{self.name} ({self.memory_gb:.0f} GB, CUDA sm_{self.compute_capability})"
        return f"GPU unavailable: {self.reason}"


_lock = threading.RLock()
_info: Optional[GPUInfo] = None
_cupy: Any = None
_listeners: list[Callable[[Device], None]] = []


def _env_device() -> Optional[Device]:
    v = os.environ.get("AEROFLOW_DEVICE", "").strip().lower()
    try:
        return Device(v) if v else None
    except ValueError:
        log.warning("Ignoring AEROFLOW_DEVICE=%r (use auto, gpu or cpu).", v)
        return None


_preference: Device = _env_device() or Device.AUTO


# ---------------------------------------------------------------------------
# CUDA probing
# ---------------------------------------------------------------------------
def _site_dirs() -> list[str]:
    dirs: list[str] = []
    try:
        dirs.extend(site.getsitepackages())
    except Exception:  # noqa: BLE001 - virtualenvs without getsitepackages
        pass
    try:
        dirs.append(site.getusersitepackages())
    except Exception:  # noqa: BLE001
        pass
    dirs.extend(p for p in sys.path if p.endswith("site-packages"))
    seen: list[str] = []
    for d in dirs:
        if d and d not in seen and os.path.isdir(d):
            seen.append(d)
    return seen


def _prepare_cuda_paths() -> None:
    """Make NVIDIA's pip-installed CUDA components (NVRTC, runtime headers) visible to CuPy.

    No-op when a CUDA Toolkit is installed (``CUDA_PATH`` with an ``include`` folder)."""
    toolkit = os.environ.get("CUDA_PATH", "")
    have_toolkit = bool(toolkit) and os.path.isdir(os.path.join(toolkit, "include"))
    for sp in _site_dirs():
        nv = os.path.join(sp, "nvidia")
        if not os.path.isdir(nv):
            continue
        for sub in ("bin", "lib"):
            for d in glob.glob(os.path.join(nv, "*", sub)):
                if not os.path.isdir(d):
                    continue
                if sys.platform == "win32":
                    try:
                        os.add_dll_directory(d)
                    except OSError:
                        pass
                # NVRTC loads its builtins library with a plain LoadLibrary / dlopen, so PATH (Windows)
                # must contain the folder too.
                path = os.environ.get("PATH", "")
                if d not in path.split(os.pathsep):
                    os.environ["PATH"] = d + os.pathsep + path
        runtime = os.path.join(nv, "cuda_runtime")
        if not have_toolkit and os.path.isdir(os.path.join(runtime, "include")):
            os.environ["CUDA_PATH"] = runtime
            have_toolkit = True


def _probe() -> GPUInfo:
    if os.environ.get("AEROFLOW_NO_GPU", "").strip() not in ("", "0", "false", "False"):
        return GPUInfo(False, reason="disabled by AEROFLOW_NO_GPU")
    _prepare_cuda_paths()
    try:
        import warnings

        # CuPy 14 cannot derive a single "CUDA root" from NVIDIA's CUDA 12 pip wheels (each component has
        # its own folder) and warns, although it finds NVRTC through cuda-pathfinder and works normally.
        warnings.filterwarnings("ignore", message="CUDA path could not be detected", category=UserWarning)
        import cupy as cp
    except Exception as exc:  # noqa: BLE001 - ImportError or a broken CUDA install
        return GPUInfo(False, reason=f"CuPy is not installed ({type(exc).__name__}). "
                                     "Run setup_windows.bat, or: pip install cupy-cuda12x "
                                     "nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12")
    try:
        n = cp.cuda.runtime.getDeviceCount()
        if n < 1:
            return GPUInfo(False, reason="no CUDA-capable GPU found")
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode(errors="replace") if isinstance(props["name"], bytes) else str(props["name"])
        mem = float(props["totalGlobalMem"]) / 2 ** 30
        cc = f"{props['major']}{props['minor']}"
        # Compile and run a tiny kernel: catches a missing NVRTC / headers / too-old driver now rather
        # than in the middle of a simulation.
        k = cp.RawKernel(r'''extern "C" __global__ void aeroflow_probe(double* x) {
            x[threadIdx.x] = 2.0 * x[threadIdx.x] + 1.0; }''', "aeroflow_probe")
        a = cp.ones(32, dtype=cp.float64)
        k((1,), (32,), (a,))
        if float(a.sum().get()) != 96.0:
            return GPUInfo(False, name=name, reason="the GPU test kernel returned a wrong result")
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        return GPUInfo(False, reason=f"CUDA initialisation failed: {msg[:200]}")
    global _cupy
    _cupy = cp
    return GPUInfo(True, name=name, memory_gb=mem, compute_capability=cc)


def gpu_info() -> GPUInfo:
    """Probe CUDA once (thread-safe) and return the cached result."""
    global _info
    with _lock:
        if _info is None:
            _info = _probe()
            if _info.available:
                log.info("GPU compute ready: %s", _info.summary())
            else:
                log.info("GPU compute not used — %s", _info.reason)
        return _info


def probed() -> bool:
    """True once the GPU probe has finished (so :func:`gpu_info` returns without blocking)."""
    return _info is not None


def cupy() -> Any:
    """The CuPy module (only valid when :func:`gpu_info` reports the GPU as available)."""
    if not gpu_info().available:
        raise RuntimeError(gpu_info().reason)
    return _cupy


def gpu_failed(exc: BaseException) -> None:
    """Mark the GPU unusable for the rest of the session after a kernel failure (solvers then fall back
    to the CPU); the preference is kept so the next start tries again."""
    global _info
    with _lock:
        name = _info.name if _info is not None else ""
        _info = GPUInfo(False, name=name, reason=f"a GPU kernel failed: {str(exc)[:200]}")
    log.warning("GPU compute disabled for this session (%s); using the CPU.", exc)
    _notify()


# ---------------------------------------------------------------------------
# Preference
# ---------------------------------------------------------------------------
def preference() -> Device:
    return _preference


def set_preference(device: Device | str) -> None:
    """Choose the compute device (listeners are notified; solvers switch at their next step)."""
    global _preference
    dev = Device(device)
    env = _env_device()
    if env is not None and env is not dev:
        log.info("AEROFLOW_DEVICE=%s overrides the selected compute device.", env.value)
        dev = env
    if dev is _preference:
        return
    _preference = dev
    log.info("Compute device: %s", describe())
    _notify()


def use_gpu(wait: bool = True) -> bool:
    """Whether the live solvers should run on the GPU right now.

    With ``wait=False`` (GUI frame loops) this never blocks: until the background probe has finished it
    answers False (the CPU runs meanwhile) and starts the probe if nobody has yet."""
    if _preference is Device.CPU:
        return False
    if not wait and _info is None:
        warmup_async()
        return False
    return gpu_info().available


def describe() -> str:
    """One line for status bars: what the live solvers run on."""
    if use_gpu():
        return f"GPU · {gpu_info().name}"
    from core import accel

    cpu = f"CPU · {accel.backend()}"
    if _preference is Device.GPU:
        return f"{cpu} — {gpu_info().reason}"
    return cpu


def add_listener(fn: Callable[[Device], None]) -> None:
    """Call ``fn(preference)`` whenever the preference (or GPU availability) changes."""
    if fn not in _listeners:
        _listeners.append(fn)


def remove_listener(fn: Callable[[Device], None]) -> None:
    if fn in _listeners:
        _listeners.remove(fn)


def _notify() -> None:
    for fn in list(_listeners):
        try:
            fn(_preference)
        except Exception:  # noqa: BLE001 - a broken listener must not break the switch
            log.debug("compute listener failed", exc_info=True)


_executor: Optional[ThreadPoolExecutor] = None


def cpu_executor() -> ThreadPoolExecutor:
    """One shared worker thread for CPU solver batches run in the background (the numba kernels release
    the GIL, so the GUI keeps drawing while they run). One thread is enough: the kernels are already
    parallel, and they must not run concurrently anyway (see ``core.accel._KERNEL_LOCK``)."""
    global _executor
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aeroflow-lab",
                                           initializer=_init_cpu_worker)
        return _executor


def _init_cpu_worker() -> None:
    """Leave two hardware threads to the GUI: numba's thread count is per calling thread, so this
    affects only the background Lab batches, not the 3-D solver."""
    try:
        import numba

        n = int(numba.config.NUMBA_NUM_THREADS)
        if n > 4:
            numba.set_num_threads(n - 2)
    except Exception:  # noqa: BLE001 - numba missing: the NumPy fallback ignores threads
        pass


_probe_started = False


def warmup_async() -> None:
    """Probe the GPU on a background thread (importing CuPy and creating the CUDA context take ~1 s) and
    pre-compile the Lab kernels, so the first GPU frame does not stall. Only the first call does work."""
    global _probe_started
    with _lock:
        if _probe_started or _info is not None:
            return
        _probe_started = True

    def run() -> None:
        info = gpu_info()
        _notify()                          # status bars / selectors can show the result right away
        if info.available and _preference is not Device.CPU:
            try:
                from core import lbm2d_gpu, nozzle_gpu, supersonic_gpu

                supersonic_gpu._module()
                lbm2d_gpu._module()
                nozzle_gpu.NozzleGPU._load_module()
            except Exception as exc:  # noqa: BLE001 - compiled again (and reported) on first use
                log.debug("GPU kernel pre-compilation skipped: %s", exc)

    threading.Thread(target=run, name="aeroflow-gpu-probe", daemon=True).start()
