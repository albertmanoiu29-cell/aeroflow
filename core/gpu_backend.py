"""GPU back ends of the Lab solvers: CUDA (CuPy, NVIDIA) and OpenCL (pyopencl: AMD, Intel and NVIDIA).

The kernels in :mod:`core.supersonic_gpu`, :mod:`core.nozzle_gpu` and :mod:`core.lbm2d_gpu` are written once
in the common subset of CUDA C++ and OpenCL C, with a few macros (``KERNEL``, ``DEVICE``, ``GMEM``,
``GID_X``, ``SYNC_LOCAL()``, ``ATOMIC_ADD_F64`` ...). :func:`prelude` defines them for each dialect.

:class:`OpenCLRuntime` is a small CuPy look-alike (just the part the solvers use: ``RawModule``,
``zeros``/``asarray``, ``.set``/``.get``/``.fill``, streams and events), so the solver host code runs
unchanged on either back end. CUDA graphs have no OpenCL equivalent; the solvers check ``GRAPHS``.
"""

from __future__ import annotations

import threading
from typing import Any, Optional, Sequence

import numpy as np

CUDA_PRELUDE = r"""
#define KERNEL extern "C" __global__ void
#define DEVICE __device__ __forceinline__
#define GMEM
#define RESTRICT __restrict__
#define SHARED __shared__
#define CONSTANT __constant__
#define GID_X ((int)(blockIdx.x * blockDim.x + threadIdx.x))
#define GID_Y ((int)(blockIdx.y * blockDim.y + threadIdx.y))
#define LID_X ((int)threadIdx.x)
#define LSIZE_X ((int)blockDim.x)
#define SYNC_LOCAL() __syncthreads()
typedef long long i64;
typedef unsigned long long u64;
#define DBL_AS_U64(x) ((u64)__double_as_longlong((double)(x)))
#define U64_AS_DBL(x) __longlong_as_double((long long)(x))
#define ATOMIC_MAX_U64(p, v) atomicMax((p), (v))
#define ATOMIC_ADD_F64(p, v) atomicAdd((p), (v))
"""

OPENCL_PRELUDE = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#define KERNEL __kernel void
#define DEVICE
#define GMEM __global
#define RESTRICT restrict
#define SHARED __local
#define CONSTANT __constant
#define GID_X ((int)get_global_id(0))
#define GID_Y ((int)get_global_id(1))
#define LID_X ((int)get_local_id(0))
#define LSIZE_X ((int)get_local_size(0))
#define SYNC_LOCAL() barrier(CLK_LOCAL_MEM_FENCE)
typedef long i64;
typedef ulong u64;
#define DBL_AS_U64(x) as_ulong((double)(x))
#define U64_AS_DBL(x) as_double((ulong)(x))
#define sqrtf sqrt
#define fminf fmin
#define fmaxf fmax
void atomic_max_u64(volatile __global ulong* p, ulong v) {
    ulong old = *p;
    while (old < v) {
        ulong prev = atom_cmpxchg(p, old, v);
        if (prev == old) break;
        old = prev;
    }
}
void atomic_add_f64(volatile __global double* p, double v) {
    volatile __global ulong* q = (volatile __global ulong*)p;
    ulong old = *q, prev;
    do {
        prev = old;
        old = atom_cmpxchg(q, prev, as_ulong(as_double(prev) + v));
    } while (old != prev);
}
#define ATOMIC_MAX_U64(p, v) atomic_max_u64((p), (v))
#define ATOMIC_ADD_F64(p, v) atomic_add_f64((p), (v))
"""


def prelude(runtime: Any) -> str:
    """The macro prelude for the kernels of the given runtime (the CuPy module or an OpenCLRuntime)."""
    return OPENCL_PRELUDE if isinstance(runtime, OpenCLRuntime) else CUDA_PRELUDE


# ---------------------------------------------------------------------------
# OpenCL runtime with the CuPy subset the solvers use
# ---------------------------------------------------------------------------
class _Ptr:
    def __init__(self, buf: Any) -> None:
        self.ptr = id(buf)


class DeviceArray:
    """A typed device buffer (like ``cupy.ndarray`` for what the solvers need)."""

    def __init__(self, rt: "OpenCLRuntime", shape: Sequence[int] | int, dtype: Any) -> None:
        import pyopencl as cl

        self.rt = rt
        self.shape = (int(shape),) if np.isscalar(shape) else tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(self.shape)) if self.shape else 1
        self.nbytes = self.size * self.dtype.itemsize
        self.buf = cl.Buffer(rt.ctx, cl.mem_flags.READ_WRITE, max(self.nbytes, 1))
        self.data = _Ptr(self.buf)

    def _queue(self, stream: Optional["Stream"]) -> Any:
        return (stream or self.rt.current_stream()).queue

    def set(self, host: np.ndarray, stream: Optional["Stream"] = None) -> None:
        import pyopencl as cl

        h = np.ascontiguousarray(host, dtype=self.dtype)
        if h.size != self.size:
            raise ValueError(f"size mismatch: {h.size} into {self.size}")
        if self.nbytes:
            cl.enqueue_copy(self._queue(stream), self.buf, h, is_blocking=True)

    def get(self, stream: Optional["Stream"] = None, out: Optional[np.ndarray] = None,
            blocking: bool = True) -> np.ndarray:
        import pyopencl as cl

        if out is None:
            out = np.empty(self.shape, dtype=self.dtype)
            blocking = True
        if self.nbytes:
            cl.enqueue_copy(self._queue(stream), out, self.buf, is_blocking=blocking)
        return out

    def fill(self, value: float) -> None:
        import pyopencl as cl

        if self.nbytes:
            cl.enqueue_fill_buffer(self._queue(None), self.buf, np.array(value, dtype=self.dtype), 0, self.nbytes)

    def sum(self) -> "_HostScalar":
        return _HostScalar(self.get().sum())


class _HostScalar:
    def __init__(self, v: Any) -> None:
        self.v = v

    def get(self) -> Any:
        return self.v


class Stream:
    """An in-order command queue; ``with stream:`` makes it current (like a CuPy stream)."""

    def __init__(self, rt: "OpenCLRuntime", non_blocking: bool = True) -> None:
        import pyopencl as cl

        self.rt = rt
        self.queue = cl.CommandQueue(rt.ctx, rt.device, properties=cl.command_queue_properties.PROFILING_ENABLE)

    def __enter__(self) -> "Stream":
        self.rt._stack().append(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.rt._stack().pop()
        self.queue.flush()                  # submit now: the GPU works while Python returns to the GUI

    def synchronize(self) -> None:
        self.queue.finish()

    @property
    def done(self) -> bool:
        import pyopencl as cl

        ev = cl.enqueue_marker(self.queue)
        self.queue.flush()
        return ev.command_execution_status == cl.command_execution_status.COMPLETE

    def begin_capture(self) -> None:
        raise RuntimeError("command graphs are not available on OpenCL")

    def end_capture(self) -> None:
        raise RuntimeError("command graphs are not available on OpenCL")


class Event:
    def __init__(self) -> None:
        self.ev: Any = None

    def record(self, stream: Stream) -> None:
        import pyopencl as cl

        self.ev = cl.enqueue_marker(stream.queue)


class _Kernel:
    def __init__(self, rt: "OpenCLRuntime", kernel: Any) -> None:
        self.rt = rt
        self.kernel = kernel
        self._last: list[Any] = []          # arguments already set (kernels keep them between launches)

    def __call__(self, grid: Sequence[int], block: Sequence[int], args: Sequence[Any]) -> None:
        import pyopencl as cl

        last = self._last
        if len(last) != len(args):
            last[:] = [None] * len(args)
        for i, a in enumerate(args):        # setting only the changed arguments saves most of the launch cost
            v = a.buf if isinstance(a, DeviceArray) else a
            p = last[i]
            if p is v or (p is not None and not isinstance(v, cl.MemoryObjectHolder)
                          and type(p) is type(v) and p == v):
                continue
            self.kernel.set_arg(i, v)
            last[i] = v
        gsize = tuple(int(g) * int(b) for g, b in zip(grid, block))
        cl.enqueue_nd_range_kernel(self.rt.current_stream().queue, self.kernel, gsize, tuple(int(b) for b in block))


class RawModule:
    def __init__(self, rt: "OpenCLRuntime", code: str) -> None:
        import pyopencl as cl

        import warnings

        self.rt = rt
        with warnings.catch_warnings():         # drivers print informational build logs; real errors still raise
            warnings.simplefilter("ignore", cl.CompilerWarning)
            self.program = cl.Program(rt.ctx, code).build()

    def get_function(self, name: str) -> _Kernel:
        import pyopencl as cl

        return _Kernel(self.rt, cl.Kernel(self.program, name))   # one kernel object per caller (own arguments)


class _CudaNamespace:
    """``cp.cuda.*`` look-alikes."""

    def __init__(self, rt: "OpenCLRuntime") -> None:
        self.rt = rt

    def Stream(self, non_blocking: bool = True) -> Stream:  # noqa: N802 - CuPy's spelling
        return Stream(self.rt, non_blocking)

    @staticmethod
    def Event() -> Event:  # noqa: N802
        return Event()

    @staticmethod
    def alloc_pinned_memory(nbytes: int) -> bytearray:
        return bytearray(int(nbytes))

    @staticmethod
    def get_elapsed_time(a: Event, b: Event) -> float:
        """Milliseconds between two recorded markers (0 when the driver gives no marker timestamps)."""
        try:
            return (b.ev.profile.end - a.ev.profile.end) * 1e-6
        except Exception:  # noqa: BLE001 - profiling of markers is optional in OpenCL
            return 0.0


class OpenCLRuntime:
    """The CuPy subset on one OpenCL GPU."""

    GRAPHS = False
    float32, float64, uint8, uint64, int32, int64 = np.float32, np.float64, np.uint8, np.uint64, np.int32, np.int64

    def __init__(self, device: Any) -> None:
        import pyopencl as cl

        self.device = device
        self.ctx = cl.Context([device])
        self.cuda = _CudaNamespace(self)
        self._local = threading.local()
        self._default = Stream(self)

    def _stack(self) -> list:
        st = getattr(self._local, "stack", None)
        if st is None:
            st = self._local.stack = []
        return st

    def current_stream(self) -> Stream:
        st = self._stack()
        return st[-1] if st else self._default

    def RawModule(self, code: str, options: Any = None) -> RawModule:  # noqa: N802
        return RawModule(self, code)

    def zeros(self, shape: Sequence[int] | int, dtype: Any = np.float64) -> DeviceArray:
        a = DeviceArray(self, shape, dtype)
        a.fill(0)
        return a

    def ones(self, shape: Sequence[int] | int, dtype: Any = np.float64) -> DeviceArray:
        a = DeviceArray(self, shape, dtype)
        a.fill(1)
        return a

    def zeros_like(self, a: DeviceArray) -> DeviceArray:
        return self.zeros(a.shape, a.dtype)

    def asarray(self, host: np.ndarray, dtype: Any = None) -> DeviceArray:
        h = np.ascontiguousarray(host, dtype=dtype)
        a = DeviceArray(self, h.shape, h.dtype)
        a.set(h)
        return a


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------
def _score(dev: Any) -> tuple:
    """Prefer discrete GPUs (more compute units, more memory) over integrated ones."""

    integrated = bool(getattr(dev, "host_unified_memory", False))
    return (not integrated, int(dev.max_compute_units) * int(dev.max_clock_frequency), int(dev.global_mem_size))


def opencl_devices() -> list:
    """Usable OpenCL GPUs (double precision, 64-bit atomics, work groups of 256), best first."""
    import pyopencl as cl

    out = []
    for plat in cl.get_platforms():
        try:
            devs = plat.get_devices(device_type=cl.device_type.GPU)
        except cl.Error:
            continue
        for d in devs:
            ext = d.extensions
            if "cl_khr_fp64" in ext and "cl_khr_int64_base_atomics" in ext and d.max_work_group_size >= 256:
                out.append(d)
    out.sort(key=_score, reverse=True)
    return out


_PROBE = r"""
__kernel void aeroflow_probe(__global double* x, __global double* s) {
    int i = get_global_id(0);
    x[i] = 2.0 * x[i] + 1.0;
    atomic_add_f64(s, x[i]);
}
"""


def probe_opencl() -> tuple[Optional[OpenCLRuntime], dict]:
    """``(runtime, info)`` for the best OpenCL GPU, or ``(None, {"reason": ...})``."""
    try:
        import pyopencl as cl  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return None, {"reason": f"pyopencl is not installed ({type(exc).__name__}): pip install pyopencl"}
    try:
        devs = opencl_devices()
    except Exception as exc:  # noqa: BLE001 - no ICD loader / broken driver
        return None, {"reason": f"no OpenCL driver ({str(exc).strip()[:120] or type(exc).__name__})"}
    if not devs:
        return None, {"reason": "no OpenCL GPU with double precision found"}
    errors = []
    for dev in devs:
        try:
            rt = OpenCLRuntime(dev)
            k = rt.RawModule(OPENCL_PRELUDE + _PROBE).get_function("aeroflow_probe")
            x = rt.asarray(np.ones(64))
            s = rt.zeros(1, np.float64)
            k((1,), (64,), (x, s))
            rt.current_stream().synchronize()
            if float(x.get().sum()) != 192.0 or float(s.get()[0]) != 192.0:
                errors.append(f"{dev.name.strip()}: the test kernel returned a wrong result")
                continue
            return rt, {"name": f"{dev.name.strip()} ({dev.platform.name.strip()})",
                        "memory_gb": dev.global_mem_size / 2 ** 30,
                        "version": dev.version.strip()}
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{dev.name.strip()}: {str(exc).strip().splitlines()[0][:160] if str(exc).strip() else type(exc).__name__}")
    return None, {"reason": "OpenCL initialisation failed: " + "; ".join(errors)}


__all__ = ["CUDA_PRELUDE", "OPENCL_PRELUDE", "prelude", "OpenCLRuntime", "DeviceArray", "opencl_devices",
           "probe_opencl"]
