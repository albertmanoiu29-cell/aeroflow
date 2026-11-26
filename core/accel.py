"""Fast field kernels for the Instant Aero Engine.

Two superposition kernels dominate a solve (about 99 % of the time before this module existed):

* :func:`source_velocity` - induced velocity of regularised point sources (Rankine surfels)
  ``v(x) = sum_j w_j (x - c_j) / (|x - c_j|^2 + a_j^2)^{3/2}``.
* :func:`horseshoe_velocity` - Biot-Savart velocity of K horseshoe vortices (bound segment A->B plus
  two semi-infinite trailing legs along ``dhat``), each carrying strength ``g_k``.

Each kernel has a numba implementation (parallel over evaluation points, compiled once and cached
on disk) and a NumPy fallback that writes cross products out per component and works in chunks,
so peak memory stays bounded. numba is optional: without it the fallback is used transparently.

Thread-safety: numba's default "workqueue" threading layer must not be entered from two Python
threads at the same time (it aborts the process), so every parallel kernel call is serialised by
:data:`_KERNEL_LOCK`.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import Final

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger("aeroflow.core.accel")

FloatArray = NDArray[np.float64]

INV_FOUR_PI: Final[float] = 1.0 / (4.0 * math.pi)
_KERNEL_LOCK = threading.Lock()
_FALLBACK_BLOCK_ELEMENTS: Final[int] = 2_000_000    # (points x sources) per NumPy chunk

#: Set ``AEROFLOW_NO_NUMBA=1`` to force the NumPy path (testing / troubleshooting).
_DISABLED = os.environ.get("AEROFLOW_NO_NUMBA", "").strip() not in ("", "0", "false", "False")

try:
    if _DISABLED:
        raise ImportError("disabled by AEROFLOW_NO_NUMBA")
    import numba
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception as _exc:  # noqa: BLE001 - ImportError, or a broken llvmlite binary
    HAVE_NUMBA = False
    _NUMBA_REASON = str(_exc)
else:
    _NUMBA_REASON = ""


# ---------------------------------------------------------------------------
# numba kernels
# ---------------------------------------------------------------------------
if HAVE_NUMBA:

    # nogil=True: the GIL is released while a kernel runs, so the GUI thread keeps painting while a
    # background solve evaluates the field (without it the UI froze for the kernel's duration).
    @njit(parallel=True, fastmath=True, cache=True, nogil=True)
    def _source_nb(nodes, pos, a2, w, out):  # pragma: no cover - compiled
        n = nodes.shape[0]
        m = pos.shape[0]
        for i in prange(n):
            x0 = nodes[i, 0]
            x1 = nodes[i, 1]
            x2 = nodes[i, 2]
            vx = 0.0
            vy = 0.0
            vz = 0.0
            for j in range(m):
                dx = x0 - pos[j, 0]
                dy = x1 - pos[j, 1]
                dz = x2 - pos[j, 2]
                d2 = dx * dx + dy * dy + dz * dz + a2[j]
                inv = w[j] / (d2 * math.sqrt(d2))
                vx += inv * dx
                vy += inv * dy
                vz += inv * dz
            out[i, 0] += vx
            out[i, 1] += vy
            out[i, 2] += vz

    @njit(parallel=True, fastmath=True, cache=True, nogil=True)
    def _horseshoe_nb(nodes, a, b, g, d, eps2, out):  # pragma: no cover - compiled
        n = nodes.shape[0]
        k_count = a.shape[0]
        d0 = d[0]
        d1 = d[1]
        d2_ = d[2]
        for i in prange(n):
            x0 = nodes[i, 0]
            x1 = nodes[i, 1]
            x2 = nodes[i, 2]
            vx = 0.0
            vy = 0.0
            vz = 0.0
            for k in range(k_count):
                gk = g[k]
                if gk == 0.0:
                    continue
                r1x = x0 - a[k, 0]
                r1y = x1 - a[k, 1]
                r1z = x2 - a[k, 2]
                r2x = x0 - b[k, 0]
                r2y = x1 - b[k, 1]
                r2z = x2 - b[k, 2]
                r0x = b[k, 0] - a[k, 0]
                r0y = b[k, 1] - a[k, 1]
                r0z = b[k, 2] - a[k, 2]
                n1 = math.sqrt(r1x * r1x + r1y * r1y + r1z * r1z)
                n2 = math.sqrt(r2x * r2x + r2y * r2y + r2z * r2z)
                if n1 < 1e-12:
                    n1 = 1e-12
                if n2 < 1e-12:
                    n2 = 1e-12
                # bound segment
                cx = r1y * r2z - r1z * r2y
                cy = r1z * r2x - r1x * r2z
                cz = r1x * r2y - r1y * r2x
                cr2 = cx * cx + cy * cy + cz * cz + eps2 * (r0x * r0x + r0y * r0y + r0z * r0z)
                dot = (r0x * (r1x / n1 - r2x / n2) + r0y * (r1y / n1 - r2y / n2)
                       + r0z * (r1z / n1 - r2z / n2))
                fb_ = dot / cr2
                vx += gk * cx * fb_
                vy += gk * cy * fb_
                vz += gk * cz * fb_
                # trailing legs (semi-infinite along d): leg from A (inbound) and from B (outbound)
                ax = d1 * r1z - d2_ * r1y
                ay = d2_ * r1x - d0 * r1z
                az = d0 * r1y - d1 * r1x
                bx = d1 * r2z - d2_ * r2y
                by = d2_ * r2x - d0 * r2z
                bz = d0 * r2y - d1 * r2x
                fa = (1.0 + (r1x * d0 + r1y * d1 + r1z * d2_) / n1) / (ax * ax + ay * ay + az * az + eps2)
                fb = (1.0 + (r2x * d0 + r2y * d1 + r2z * d2_) / n2) / (bx * bx + by * by + bz * bz + eps2)
                vx += gk * (bx * fb - ax * fa)
                vy += gk * (by * fb - ay * fa)
                vz += gk * (bz * fb - az * fa)
            out[i, 0] += vx * 0.07957747154594767
            out[i, 1] += vy * 0.07957747154594767
            out[i, 2] += vz * 0.07957747154594767


# ---------------------------------------------------------------------------
# NumPy fallbacks
# ---------------------------------------------------------------------------
def _source_np(nodes: FloatArray, pos: FloatArray, a2: FloatArray, w: FloatArray, out: FloatArray) -> None:
    m = max(len(pos), 1)
    chunk = int(np.clip(_FALLBACK_BLOCK_ELEMENTS // m, 64, 8192))
    px, py, pz = pos[:, 0], pos[:, 1], pos[:, 2]
    for s in range(0, len(nodes), chunk):
        e = min(s + chunk, len(nodes))
        dx = nodes[s:e, 0, None] - px[None, :]
        dy = nodes[s:e, 1, None] - py[None, :]
        dz = nodes[s:e, 2, None] - pz[None, :]
        d2 = dx * dx
        d2 += dy * dy
        d2 += dz * dz
        d2 += a2[None, :]
        inv = w[None, :] / (d2 * np.sqrt(d2))
        out[s:e, 0] += (inv * dx).sum(1)
        out[s:e, 1] += (inv * dy).sum(1)
        out[s:e, 2] += (inv * dz).sum(1)


def horseshoe_terms(x: FloatArray, a: FloatArray, b: FloatArray, d: FloatArray,
                    eps2: float) -> tuple[FloatArray, FloatArray]:
    """Per-horseshoe unit-strength velocities ``(bound, legs)``, each (P, K, 3).

    Used where the individual influences are needed (the vortex-lattice matrix). Cross products
    are written per component, which is about 3x faster than ``np.cross`` on (P, K, 3) arrays.
    """
    r1x = x[:, 0, None] - a[None, :, 0]
    r1y = x[:, 1, None] - a[None, :, 1]
    r1z = x[:, 2, None] - a[None, :, 2]
    r2x = x[:, 0, None] - b[None, :, 0]
    r2y = x[:, 1, None] - b[None, :, 1]
    r2z = x[:, 2, None] - b[None, :, 2]
    r0 = b - a
    r0x, r0y, r0z = r0[None, :, 0], r0[None, :, 1], r0[None, :, 2]
    n1 = np.maximum(np.sqrt(r1x * r1x + r1y * r1y + r1z * r1z), 1e-12)
    n2 = np.maximum(np.sqrt(r2x * r2x + r2y * r2y + r2z * r2z), 1e-12)
    cx = r1y * r2z - r1z * r2y
    cy = r1z * r2x - r1x * r2z
    cz = r1x * r2y - r1y * r2x
    cr2 = cx * cx + cy * cy + cz * cz + eps2 * (r0x * r0x + r0y * r0y + r0z * r0z)
    dot = r0x * (r1x / n1 - r2x / n2) + r0y * (r1y / n1 - r2y / n2) + r0z * (r1z / n1 - r2z / n2)
    f = dot / cr2 * INV_FOUR_PI
    bound = np.stack([cx * f, cy * f, cz * f], axis=-1)
    d0, d1, d2 = float(d[0]), float(d[1]), float(d[2])
    ax, ay, az = d1 * r1z - d2 * r1y, d2 * r1x - d0 * r1z, d0 * r1y - d1 * r1x
    bx, by, bz = d1 * r2z - d2 * r2y, d2 * r2x - d0 * r2z, d0 * r2y - d1 * r2x
    fa = (1.0 + (r1x * d0 + r1y * d1 + r1z * d2) / n1) / (ax * ax + ay * ay + az * az + eps2) * INV_FOUR_PI
    fb = (1.0 + (r2x * d0 + r2y * d1 + r2z * d2) / n2) / (bx * bx + by * by + bz * bz + eps2) * INV_FOUR_PI
    legs = np.stack([bx * fb - ax * fa, by * fb - ay * fa, bz * fb - az * fa], axis=-1)
    return bound, legs


def _horseshoe_np(nodes: FloatArray, a: FloatArray, b: FloatArray, g: FloatArray, d: FloatArray,
                  eps2: float, out: FloatArray) -> None:
    k = max(len(a), 1)
    chunk = int(np.clip(_FALLBACK_BLOCK_ELEMENTS // (4 * k), 64, 8192))
    for s in range(0, len(nodes), chunk):
        e = min(s + chunk, len(nodes))
        bound, legs = horseshoe_terms(nodes[s:e], a, b, d, eps2)
        out[s:e] += np.einsum("pkc,k->pc", bound + legs, g)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _c64(a: np.ndarray) -> FloatArray:
    return np.ascontiguousarray(a, dtype=np.float64)


def source_velocity(nodes: FloatArray, pos: FloatArray, a2: FloatArray, w: FloatArray,
                    out: FloatArray | None = None) -> FloatArray:
    """Add the velocity of regularised point sources at ``nodes`` (N,3) into ``out`` and return it.

    Args:
        nodes: Evaluation points (N, 3).
        pos: Source positions (M, 3).
        a2: Core radius squared per source (M,).
        w: Strength per source, already including the ``1/(4 pi)`` and area factors (M,).
        out: Optional (N, 3) float64 accumulator; created (zeroed) when omitted.
    """
    nodes, pos, a2, w = _c64(nodes), _c64(pos), _c64(a2), _c64(w)
    if out is None:
        out = np.zeros((len(nodes), 3))
    if len(nodes) == 0 or len(pos) == 0:
        return out
    if HAVE_NUMBA:
        try:
            with _KERNEL_LOCK:
                _source_nb(nodes, pos, a2, w, out)
            return out
        except Exception as exc:  # noqa: BLE001 - never let the accelerator break a solve
            _disable_numba(exc)
    _source_np(nodes, pos, a2, w, out)
    return out


def horseshoe_velocity(nodes: FloatArray, a: FloatArray, b: FloatArray, g: FloatArray, d: FloatArray,
                       eps2: float, out: FloatArray | None = None) -> FloatArray:
    """Add the summed velocity of K horseshoe vortices (strengths ``g``) at ``nodes`` into ``out``."""
    nodes, a, b, g, d = _c64(nodes), _c64(a), _c64(b), _c64(g), _c64(d)
    if out is None:
        out = np.zeros((len(nodes), 3))
    if len(nodes) == 0 or len(a) == 0:
        return out
    if HAVE_NUMBA:
        try:
            with _KERNEL_LOCK:
                _horseshoe_nb(nodes, a, b, g, d, float(eps2), out)
            return out
        except Exception as exc:  # noqa: BLE001
            _disable_numba(exc)
    _horseshoe_np(nodes, a, b, g, d, float(eps2), out)
    return out


def _disable_numba(exc: BaseException) -> None:
    global HAVE_NUMBA, _NUMBA_REASON
    HAVE_NUMBA = False
    _NUMBA_REASON = f"kernel failed: {exc}"
    log.warning("numba acceleration disabled (%s); using the NumPy fallback.", exc)


def precompile(dispatcher, *example_args) -> None:
    """Compile (or load from cache) a numba kernel for the argument types of ``example_args`` without
    running it - so no kernel lock is needed and the 3-D solver is never blocked by a warm-up."""
    if not HAVE_NUMBA:
        return
    try:
        dispatcher.compile(tuple(numba.typeof(a) for a in example_args))
    except Exception as exc:  # noqa: BLE001
        log.debug("Pre-compilation of %s skipped: %s", getattr(dispatcher, "__name__", dispatcher), exc)


def backend() -> str:
    """Human-readable name of the active kernel backend."""
    if HAVE_NUMBA:
        return f"numba {numba.__version__} ({numba.config.NUMBA_NUM_THREADS} threads)"
    return "NumPy" + (f" (numba unavailable: {_NUMBA_REASON})" if _NUMBA_REASON else "")


_warm_lock = threading.Lock()
_warmed = False


def warmup() -> float:
    """Compile (or load from the on-disk cache) every kernel with tiny inputs. Returns seconds taken.

    Safe to call from any thread and more than once; only the first call does any work."""
    global _warmed
    with _warm_lock:
        if _warmed:
            return 0.0
        t0 = time.perf_counter()
        x = np.zeros((4, 3))
        p = np.ones((3, 3))
        source_velocity(x, p, np.ones(3), np.ones(3))
        horseshoe_velocity(x, p, 2.0 * p, np.ones(3), np.array([1.0, 0.0, 0.0]), 1e-4)
        try:   # the Live Flow Lab and Supersonic Lab kernels (imported lazily: they import this module)
            from core import lbm2d, nozzle, supersonic
            lbm2d.warmup()
            supersonic.warmup()
            nozzle.warmup()
        except Exception as exc:  # noqa: BLE001 - a warm-up failure must never break start-up
            log.debug("Lab kernel warm-up skipped: %s", exc)
        _warmed = True
        dt = time.perf_counter() - t0
        log.info("Field kernels ready: %s (%.2f s)", backend(), dt)
        return dt


def warmup_async() -> None:
    """Start :func:`warmup` on a daemon thread so the first solve does not pay the compile time."""
    threading.Thread(target=warmup, name="aeroflow-kernel-warmup", daemon=True).start()
