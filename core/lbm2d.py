"""Real-time 2-D wind tunnel: a Lattice-Boltzmann (D2Q9) flow solver for the Live Flow Lab.

Unlike the Instant Aero Engine (steady potential flow), this is a *transient* viscous solver:
vortex shedding, separation, wakes and stall emerge from the physics rather than from models.

Method
------
* **D2Q9 lattice, BGK collision with a Smagorinsky sub-grid model.** The relaxation time is raised
  locally from the resolved strain rate (Hou et al. 1996), which keeps the scheme stable at the
  high Reynolds numbers a small grid cannot resolve.
* **Fused pull-scheme stream + collide** in one pass (two buffers, swapped every step).
* **Boundaries:** equilibrium velocity inlet (left), zero-gradient outlet (right), free-slip
  (specular) tunnel walls (top / bottom), full-way bounce-back on the obstacle.
* **Forces** by the momentum-exchange method on every fluid-solid link, giving CL / CD in time.
* **Visual tracers:** a passive smoke-dye field (semi-Lagrangian advection, injected by an inlet
  rake) and massless particles advected with the local velocity.

Lattice units: the inflow speed is ``u_lattice`` (default 0.1, Mach ≈ 0.17), density ≈ 1. The
kinematic viscosity follows the requested Reynolds number: ``nu = U L / Re``, ``tau = 3 nu + 0.5``.

Performance: with numba the step kernel runs in parallel (~5-15 ms for 50 steps on a 400x160 grid
on 8 threads); without numba a vectorised NumPy step is used (several times slower). Kernel calls
share :data:`core.accel._KERNEL_LOCK` with the 3-D solver, because numba's default threading layer
must not be entered from two threads at once; :meth:`FlowLab2D.step` never blocks on it.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Final, Optional

import numpy as np
from numpy.typing import NDArray

from core import accel

log = logging.getLogger("aeroflow.core.lbm2d")

F32 = NDArray[np.float32]

# D2Q9 velocity set: rest, E, N, W, S, NE, NW, SW, SE
CX: Final = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
CY: Final = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
W: Final = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=np.float32)
OPP: Final = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
REFY: Final = np.array([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=np.int64)     # mirror in y (free-slip walls)
SMAGORINSKY_CS: Final[float] = 0.14
U_MAX_GUARD: Final[float] = 0.35            # lattice speed clamp inside the equilibrium (stability)


# ---------------------------------------------------------------------------
# numba kernels
# ---------------------------------------------------------------------------
if accel.HAVE_NUMBA:
    from numba import njit, prange

    @njit(parallel=True, fastmath=True, cache=True, nogil=True)
    def _lbm_step_nb(f_src, f_dst, solid, special, u_in, tau0, smag, ux, uy, rho_out, force,
                     cx, cy, w, opp, refy, sponge, ftarget):  # pragma: no cover - compiled
        ny, nx = solid.shape
        fx_tot = 0.0
        fy_tot = 0.0
        for yy in prange(ny):
            y = np.int64(yy)          # prange indices are unsigned; y - cy[i] must stay a signed int
            fin = np.empty(9, dtype=np.float32)
            for x in range(nx):
                if not special[y, x]:
                    # ---- fast path: interior fluid cell, no wall or body among its neighbours ----
                    f0 = f_src[0, y, x]
                    f1 = f_src[1, y, x - 1]
                    f2 = f_src[2, y - 1, x]
                    f3 = f_src[3, y, x + 1]
                    f4 = f_src[4, y + 1, x]
                    f5 = f_src[5, y - 1, x - 1]
                    f6 = f_src[6, y - 1, x + 1]
                    f7 = f_src[7, y + 1, x + 1]
                    f8 = f_src[8, y + 1, x - 1]
                    rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
                    if not (rho > 0.2 and rho < 5.0):
                        rho = 1.0
                        f0 = f1 = f2 = f3 = f4 = f5 = f6 = f7 = f8 = 0.0
                        u = u_in
                        v = 0.0
                    else:
                        u = (f1 + f5 + f8 - f3 - f6 - f7) / rho
                        v = (f2 + f5 + f6 - f4 - f7 - f8) / rho
                    ux[y, x] = u
                    uy[y, x] = v
                    rho_out[y, x] = rho
                    sp = math.sqrt(u * u + v * v)
                    if sp > 0.35:
                        u = u * 0.35 / sp
                        v = v * 0.35 / sp
                    usq = 1.5 * (u * u + v * v)
                    a = rho / 9.0
                    b = rho / 36.0
                    upv = u + v
                    umv = u - v
                    e0 = 4.0 * a * (1.0 - usq)
                    e1 = a * (1.0 + 3.0 * u + 4.5 * u * u - usq)
                    e2 = a * (1.0 + 3.0 * v + 4.5 * v * v - usq)
                    e3 = a * (1.0 - 3.0 * u + 4.5 * u * u - usq)
                    e4 = a * (1.0 - 3.0 * v + 4.5 * v * v - usq)
                    e5 = b * (1.0 + 3.0 * upv + 4.5 * upv * upv - usq)
                    e6 = b * (1.0 - 3.0 * umv + 4.5 * umv * umv - usq)
                    e7 = b * (1.0 - 3.0 * upv + 4.5 * upv * upv - usq)
                    e8 = b * (1.0 + 3.0 * umv + 4.5 * umv * umv - usq)
                    if rho == 1.0 and f0 == 0.0:      # reset after a divergence: start at equilibrium
                        f0, f1, f2, f3, f4, f5, f6, f7, f8 = e0, e1, e2, e3, e4, e5, e6, e7, e8
                    d0 = f0 - e0
                    d1 = f1 - e1
                    d2 = f2 - e2
                    d3 = f3 - e3
                    d4 = f4 - e4
                    d5 = f5 - e5
                    d6 = f6 - e6
                    d7 = f7 - e7
                    d8 = f8 - e8
                    pxx = d1 + d3 + d5 + d6 + d7 + d8
                    pyy = d2 + d4 + d5 + d6 + d7 + d8
                    pxy = d5 - d6 + d7 - d8
                    q = math.sqrt(pxx * pxx + pyy * pyy + 2.0 * pxy * pxy)
                    tb = tau0[x]                       # per-column base tau (sponge layers at both ends)
                    om = 2.0 / (tb + math.sqrt(tb * tb + smag * q / rho))
                    f_dst[0, y, x] = f0 - om * d0
                    f_dst[1, y, x] = f1 - om * d1
                    f_dst[2, y, x] = f2 - om * d2
                    f_dst[3, y, x] = f3 - om * d3
                    f_dst[4, y, x] = f4 - om * d4
                    f_dst[5, y, x] = f5 - om * d5
                    f_dst[6, y, x] = f6 - om * d6
                    f_dst[7, y, x] = f7 - om * d7
                    f_dst[8, y, x] = f8 - om * d8
                    s = sponge[x]
                    if s > 0.0:                        # absorbing layer: relax toward the freestream
                        for i in range(9):
                            f_dst[i, y, x] += s * (ftarget[i] - f_dst[i, y, x])
                    continue
                # ---- general path: walls, inlet / outlet, body and cells next to it ----
                if solid[y, x]:
                    for i in range(9):
                        f_dst[i, y, x] = w[i]
                    ux[y, x] = 0.0
                    uy[y, x] = 0.0
                    rho_out[y, x] = 1.0
                    continue
                for i in range(9):
                    sx = x - cx[i]
                    sy = y - cy[i]
                    ii = i
                    if sy < 0 or sy >= ny:            # free-slip wall: specular reflection
                        sy = y
                        ii = refy[i]
                    if sx < 0:
                        # velocity inlet: equilibrium at the inflow speed with the local density, so the
                        # imposed velocity holds (a fixed density let mass pile up and the approach
                        # flow sag to ~0.7 U)
                        rl = rho_out[y, x]
                        if not (rl > 0.2 and rl < 5.0):
                            rl = 1.0
                        cu = 3.0 * (cx[i] * u_in)
                        fin[i] = w[i] * rl * (1.0 + cu + 0.5 * cu * cu - 1.5 * u_in * u_in)
                    elif sx >= nx:
                        # pressure outlet: equilibrium at rho = 1 with the local velocity
                        uo = ux[y, nx - 1]
                        vo = uy[y, nx - 1]
                        cu = 3.0 * (cx[i] * uo + cy[i] * vo)
                        fin[i] = w[i] * (1.0 + cu + 0.5 * cu * cu - 1.5 * (uo * uo + vo * vo))
                    elif solid[sy, sx]:               # bounce-back + momentum exchange
                        fb = f_src[opp[i], y, x]
                        fin[i] = fb
                        fx_tot += 2.0 * fb * cx[opp[i]]
                        fy_tot += 2.0 * fb * cy[opp[i]]
                    else:
                        fin[i] = f_src[ii, sy, sx]
                rho = 0.0
                mx = 0.0
                my = 0.0
                for i in range(9):
                    rho += fin[i]
                    mx += fin[i] * cx[i]
                    my += fin[i] * cy[i]
                if not (rho > 0.2 and rho < 5.0):     # divergence guard (also catches NaN)
                    rho = 1.0
                    mx = u_in
                    my = 0.0
                    for i in range(9):
                        cu = 3.0 * (cx[i] * u_in)
                        fin[i] = w[i] * (1.0 + cu + 0.5 * cu * cu - 1.5 * u_in * u_in)
                u = mx / rho
                v = my / rho
                sp = math.sqrt(u * u + v * v)
                ue = u
                ve = v
                if sp > 0.35:
                    ue = u * 0.35 / sp
                    ve = v * 0.35 / sp
                usq = 1.5 * (ue * ue + ve * ve)
                pxx = 0.0
                pyy = 0.0
                pxy = 0.0
                for i in range(9):
                    cu = 3.0 * (cx[i] * ue + cy[i] * ve)
                    feq = w[i] * rho * (1.0 + cu + 0.5 * cu * cu - usq)
                    d = fin[i] - feq
                    pxx += cx[i] * cx[i] * d
                    pyy += cy[i] * cy[i] * d
                    pxy += cx[i] * cy[i] * d
                q = math.sqrt(pxx * pxx + pyy * pyy + 2.0 * pxy * pxy)
                tb = tau0[x]
                tau = 0.5 * (tb + math.sqrt(tb * tb + smag * q / rho))
                omega = 1.0 / tau
                s = sponge[x]
                for i in range(9):
                    cu = 3.0 * (cx[i] * ue + cy[i] * ve)
                    feq = w[i] * rho * (1.0 + cu + 0.5 * cu * cu - usq)
                    fp = fin[i] - omega * (fin[i] - feq)
                    f_dst[i, y, x] = fp + s * (ftarget[i] - fp)
                ux[y, x] = u
                uy[y, x] = v
                rho_out[y, x] = rho
        force[0] += fx_tot
        force[1] += fy_tot

    @njit(parallel=True, fastmath=True, cache=True, nogil=True)
    def _advect_dye_nb(dye, out, ux, uy, solid, dt, decay):  # pragma: no cover - compiled
        ny, nx = dye.shape
        for y in prange(ny):
            for x in range(nx):
                if solid[y, x]:
                    out[y, x] = 0.0
                    continue
                px = x - ux[y, x] * dt
                py = y - uy[y, x] * dt
                if px < 0.0:
                    px = 0.0
                if px > nx - 1.001:
                    px = nx - 1.001
                if py < 0.0:
                    py = 0.0
                if py > ny - 1.001:
                    py = ny - 1.001
                i0 = int(px)
                j0 = int(py)
                fx = px - i0
                fy = py - j0
                v = ((1.0 - fx) * (1.0 - fy) * dye[j0, i0] + fx * (1.0 - fy) * dye[j0, i0 + 1]
                     + (1.0 - fx) * fy * dye[j0 + 1, i0] + fx * fy * dye[j0 + 1, i0 + 1])
                out[y, x] = v * decay


if accel.HAVE_NUMBA:

    # Parallel over rows (streamline tracing is a dependent load chain, ~30 ns per step serially), so it
    # runs under the shared kernel lock; see line_integral_convolution.
    @njit(parallel=True, fastmath=True, cache=True, nogil=True)
    def _lic_nb(dxf, dyf, solid, noise, wfwd, wbwd, out):  # pragma: no cover - compiled
        # dxf / dyf: unit flow direction x 0.8 px step (0 where the flow is at rest);
        # wfwd / wbwd: precomputed filter weights along the forward / backward streamline halves.
        ny, nx = dxf.shape
        half_len = wfwd.shape[0]
        for yy in prange(ny):
            y = np.int64(yy)
            for x in range(nx):
                if solid[y, x]:
                    out[y, x] = 0.0
                    continue
                acc = 0.0
                wsum = 0.0
                for side in range(2):
                    sgn = 1.0 if side == 0 else -1.0
                    px = x + 0.5
                    py = y + 0.5
                    for k in range(half_len):
                        i = int(px)
                        j = int(py)
                        if px < 0.0 or i >= nx or py < 0.0 or j >= ny or solid[j, i]:
                            break
                        wk = wfwd[k] if side == 0 else wbwd[k]
                        acc += wk * noise[j, i]
                        wsum += wk
                        ddx = dxf[j, i]
                        ddy = dyf[j, i]
                        if ddx == 0.0 and ddy == 0.0:
                            break
                        px += sgn * ddx
                        py += sgn * ddy
                out[y, x] = acc / wsum if wsum > 0.0 else 0.5


def _lic_np(ux: F32, uy: F32, solid: NDArray[np.bool_], noise: F32, half_len: int, phase: float,
            out: F32) -> None:
    """Vectorised NumPy line-integral convolution (all pixels advance together)."""
    ny, nx = ux.shape
    m = np.sqrt(ux * ux + uy * uy)
    dx = np.where(m > 1e-9, ux / np.maximum(m, 1e-9), 0.0)
    dy = np.where(m > 1e-9, uy / np.maximum(m, 1e-9), 0.0)
    jj, ii = np.mgrid[0:ny, 0:nx]
    acc = np.zeros((ny, nx))
    wsum = np.zeros((ny, nx))
    for direction in (1.0, -1.0):
        px, py = ii + 0.5, jj + 0.5
        alive = ~solid
        for k in range(half_len):
            i = np.clip(px.astype(np.int64), 0, nx - 1)
            j = np.clip(py.astype(np.int64), 0, ny - 1)
            alive &= (px >= 0) & (px < nx) & (py >= 0) & (py < ny) & ~solid[j, i]
            s = direction * k / half_len
            wk = 1.0 + 0.6 * math.cos(math.pi * s - phase)
            acc += np.where(alive, wk * noise[j, i], 0.0)
            wsum += np.where(alive, wk, 0.0)
            px = px + direction * 0.8 * dx[j, i]
            py = py + direction * 0.8 * dy[j, i]
    out[:] = np.where(wsum > 0, acc / np.maximum(wsum, 1e-12), 0.5)
    out[solid] = 0.0


def line_integral_convolution(ux: F32, uy: F32, solid: NDArray[np.bool_], noise: F32, half_len: int = 10,
                              phase: float = 0.0, out: Optional[F32] = None) -> F32:
    """Line-integral convolution of ``noise`` along the velocity field: every streamline at once.

    ``phase`` shifts a periodic filter along the streamlines, so advancing it each frame makes the
    texture visibly flow. Returns values around 0.5 (0 inside solids).

    With numba the parallel kernel runs only if the shared kernel lock is free at that moment; when
    the 3-D solver holds it, ``out`` is returned unchanged (the previous frame's texture) so the
    caller never blocks."""
    if out is None:
        out = np.full(ux.shape, 0.5, dtype=np.float32)
    if accel.HAVE_NUMBA:
        if not accel._KERNEL_LOCK.acquire(blocking=False):
            return out
        try:
            m = np.sqrt(ux * ux + uy * uy)
            scale = np.where(m > 1e-9, 0.8 / np.maximum(m, 1e-9), 0.0).astype(np.float32)
            k = np.arange(half_len, dtype=np.float64) / half_len
            wfwd = (1.0 + 0.6 * np.cos(np.pi * k - phase)).astype(np.float32)     # travelling filter
            wbwd = (1.0 + 0.6 * np.cos(-np.pi * k - phase)).astype(np.float32)
            _lic_nb((ux * scale).astype(np.float32), (uy * scale).astype(np.float32), solid, noise, wfwd, wbwd, out)
            return out
        except Exception as exc:  # noqa: BLE001
            log.debug("LIC kernel failed (%s); using NumPy.", exc)
        finally:
            accel._KERNEL_LOCK.release()
    _lic_np(ux, uy, solid, noise, int(half_len), float(phase), out)
    return out


# ---------------------------------------------------------------------------
# NumPy fallback
# ---------------------------------------------------------------------------
def _feq_np(rho: F32, ux: F32, uy: F32) -> F32:
    sp = np.sqrt(ux * ux + uy * uy)
    s = np.where(sp > U_MAX_GUARD, U_MAX_GUARD / np.maximum(sp, 1e-12), 1.0)
    ue, ve = ux * s, uy * s
    usq = 1.5 * (ue * ue + ve * ve)
    out = np.empty((9,) + rho.shape, dtype=np.float32)
    for i in range(9):
        cu = 3.0 * (CX[i] * ue + CY[i] * ve)
        out[i] = W[i] * rho * (1.0 + cu + 0.5 * cu * cu - usq)
    return out


def _lbm_step_np(f_src: F32, f_dst: F32, solid: NDArray[np.bool_], u_in: float, tau0: F32, smag: float,
                 ux: F32, uy: F32, rho_out: F32, force: NDArray[np.float64], sponge: F32, ftarget: F32) -> None:
    ny, nx = solid.shape
    fin = np.empty_like(f_src)
    fx = fy = 0.0
    for i in range(9):
        # pull: fin_i(x) = f_i(x - c_i) with free-slip y walls (specular) and inlet / outlet in x
        src = f_src[i]
        shifted = np.empty((ny, nx), dtype=np.float32)
        cx, cy = int(CX[i]), int(CY[i])
        # y shift with reflection at the walls
        if cy == 0:
            ys = src
        else:
            ys = np.empty_like(src)
            ref = f_src[REFY[i]]
            if cy > 0:
                ys[1:] = src[:-1]
                ys[0] = ref[0]
            else:
                ys[:-1] = src[1:]
                ys[-1] = ref[-1]
        if cx == 0:
            shifted[:] = ys
        elif cx > 0:      # velocity inlet with the local (previous-step) density
            shifted[:, 1:] = ys[:, :-1]
            cu = 3.0 * cx * u_in
            rl = np.where((rho_out[:, 0] > 0.2) & (rho_out[:, 0] < 5.0), rho_out[:, 0], 1.0)
            shifted[:, 0] = W[i] * rl * (1.0 + cu + 0.5 * cu * cu - 1.5 * u_in * u_in)
        else:             # pressure outlet: rho = 1 with the local velocity
            shifted[:, :-1] = ys[:, 1:]
            uo, vo = ux[:, -1], uy[:, -1]
            cu = 3.0 * (cx * uo + cy * vo)
            shifted[:, -1] = W[i] * (1.0 + cu + 0.5 * cu * cu - 1.5 * (uo * uo + vo * vo))
        # bounce-back where the source cell is solid
        src_solid = np.zeros((ny, nx), dtype=bool)
        sy0, sy1 = (1, ny) if cy > 0 else (0, ny - 1) if cy < 0 else (0, ny)
        sx0, sx1 = (1, nx) if cx > 0 else (0, nx - 1) if cx < 0 else (0, nx)
        src_solid[sy0:sy1, sx0:sx1] = solid[sy0 - cy:sy1 - cy, sx0 - cx:sx1 - cx]
        src_solid &= ~solid
        back = f_src[OPP[i]]
        shifted[src_solid] = back[src_solid]
        if src_solid.any():
            s = float(back[src_solid].sum())
            fx += 2.0 * s * CX[OPP[i]]
            fy += 2.0 * s * CY[OPP[i]]
        fin[i] = shifted
    rho = fin.sum(axis=0)
    bad = ~((rho > 0.2) & (rho < 5.0))
    rho = np.where(bad, 1.0, rho).astype(np.float32)
    mx = np.tensordot(CX.astype(np.float32), fin, axes=1)
    my = np.tensordot(CY.astype(np.float32), fin, axes=1)
    u = np.where(bad, u_in, mx / rho).astype(np.float32)
    v = np.where(bad, 0.0, my / rho).astype(np.float32)
    feq = _feq_np(rho, u, v)
    d = fin - feq
    pxx = np.tensordot((CX * CX).astype(np.float32), d, axes=1)
    pyy = np.tensordot((CY * CY).astype(np.float32), d, axes=1)
    pxy = np.tensordot((CX * CY).astype(np.float32), d, axes=1)
    q = np.sqrt(pxx * pxx + pyy * pyy + 2.0 * pxy * pxy)
    tau = 0.5 * (tau0 + np.sqrt(tau0 * tau0 + smag * q / rho))
    f_dst[:] = fin - d / tau
    f_dst += sponge[None, None, :] * (ftarget[:, None, None] - f_dst)
    f_dst[:, solid] = W[:, None]
    u[solid] = 0.0
    v[solid] = 0.0
    rho[solid] = 1.0
    ux[:] = u
    uy[:] = v
    rho_out[:] = rho
    force[0] += fx
    force[1] += fy


def _advect_dye_np(dye: F32, out: F32, ux: F32, uy: F32, solid: NDArray[np.bool_], dt: float,
                   decay: float) -> None:
    ny, nx = dye.shape
    jj, ii = np.mgrid[0:ny, 0:nx].astype(np.float32)
    px = np.clip(ii - ux * dt, 0.0, nx - 1.001)
    py = np.clip(jj - uy * dt, 0.0, ny - 1.001)
    i0, j0 = px.astype(np.int64), py.astype(np.int64)
    fx, fy = px - i0, py - j0
    v = ((1 - fx) * (1 - fy) * dye[j0, i0] + fx * (1 - fy) * dye[j0, i0 + 1]
         + (1 - fx) * fy * dye[j0 + 1, i0] + fx * fy * dye[j0 + 1, i0 + 1])
    out[:] = v * decay
    out[solid] = 0.0


# ---------------------------------------------------------------------------
# The simulation object
# ---------------------------------------------------------------------------
@dataclass
class ForceHistory:
    """Time history of the aerodynamic coefficients (convective time t* = t U / L)."""

    t: list[float] = field(default_factory=list)
    cl: list[float] = field(default_factory=list)
    cd: list[float] = field(default_factory=list)

    def clear(self) -> None:
        self.t.clear()
        self.cl.clear()
        self.cd.clear()

    def append(self, t: float, cl: float, cd: float, max_len: int = 6000) -> None:
        self.t.append(t)
        self.cl.append(cl)
        self.cd.append(cd)
        if len(self.t) > max_len:
            del self.t[:-max_len], self.cl[:-max_len], self.cd[:-max_len]


class FlowLab2D:
    """A real-time 2-D wind tunnel. All methods are meant for one (GUI) thread."""

    def __init__(self, nx: int = 400, ny: int = 160, u_lattice: float = 0.12) -> None:
        self.u_lattice = float(u_lattice)
        self.reynolds = 1000.0
        self.ref_length = 0.2 * ny                 # cells
        self.frontal_length = 0.2 * ny             # cells (for the Strouhal number)
        self.smag = 18.0 * math.sqrt(2.0) * SMAGORINSKY_CS ** 2
        self.ramp_steps = 600                      # soft start: no pressure shock
        self.dye_spacing = 10                      # rake: one smoke streak every n cells (3 cells wide)
        self.n_particles = 900
        self.history = ForceHistory()
        self.last_step_ms = 1.0
        self._ft_u = -1.0
        self._ftarget = np.zeros(9, dtype=np.float32)
        self._rng = np.random.default_rng(7)
        self.resize(nx, ny)

    # -- set-up ------------------------------------------------------------
    def resize(self, nx: int, ny: int) -> None:
        """Allocate a new grid (keeps the obstacle mask only if the size is unchanged)."""
        self.nx, self.ny = int(nx), int(ny)
        shape = (self.ny, self.nx)
        self.solid = np.zeros(shape, dtype=np.bool_)
        self.f = np.empty((9,) + shape, dtype=np.float32)
        self.f2 = np.empty_like(self.f)
        self.ux = np.zeros(shape, dtype=np.float32)
        self.uy = np.zeros(shape, dtype=np.float32)
        self.rho = np.ones(shape, dtype=np.float32)
        self.dye = np.zeros(shape, dtype=np.float32)
        self._dye2 = np.zeros_like(self.dye)
        self._update_special()
        self.reset()

    def _update_special(self) -> None:
        """Cells that need the general (boundary-aware) kernel path: the body, its 8 neighbours and
        the tunnel edges. Everything else takes the unrolled fast path."""
        s = self.solid
        sp = s.copy()
        sp[1:, :] |= s[:-1, :]
        sp[:-1, :] |= s[1:, :]
        sp2 = sp.copy()
        sp2[:, 1:] |= sp[:, :-1]
        sp2[:, :-1] |= sp[:, 1:]
        sp2[0, :] = sp2[-1, :] = True
        sp2[:, 0] = sp2[:, -1] = True
        self.special = sp2

    def set_obstacle(self, mask: NDArray[np.bool_], ref_length_cells: float) -> None:
        """Replace the obstacle (the flow keeps running around the new shape)."""
        mask = np.asarray(mask, dtype=np.bool_)
        if mask.shape != (self.ny, self.nx):
            raise ValueError(f"mask shape {mask.shape} != grid {(self.ny, self.nx)}")
        mask = mask.copy()
        mask[:, :2] = False                       # keep the inlet column clear
        mask[:, -2:] = False
        newly = mask & ~self.solid
        self.solid = mask
        self._update_special()
        self.f[:, newly] = W[:, None]             # cells that became solid start at rest
        self.f2[:, newly] = W[:, None]
        new_ref = max(float(ref_length_cells), 2.0)
        if abs(new_ref - self.ref_length) > 1e-9:
            # t* and the coefficients are scaled by L: a new reference length starts a new history
            # (mixing the two made false jumps in the chart and polluted the Strouhal spectrum).
            self.history.clear()
        self.ref_length = new_ref
        rows = np.where(mask.any(axis=1))[0]
        self.frontal_length = float(rows[-1] - rows[0] + 1) if len(rows) else self.ref_length
        self._update_tau()

    def paint(self, cx: float, cy: float, radius: float, solid: bool) -> None:
        """Add (``solid=True``) or erase a disc of obstacle cells - interactive drawing."""
        jj, ii = np.ogrid[0:self.ny, 0:self.nx]
        disc = (ii - cx) ** 2 + (jj - cy) ** 2 <= radius * radius
        disc[:, :2] = False
        disc[:, -2:] = False
        if solid:
            mask = self.solid | disc
        else:
            mask = self.solid & ~disc
        self.set_obstacle(mask, self.ref_length)

    def set_reynolds(self, re: float) -> None:
        """Change the Reynolds number based on the reference length (applies immediately)."""
        self.reynolds = float(np.clip(re, 10.0, 1.0e6))
        self._update_tau()

    def _update_tau(self) -> None:
        nu = self.u_lattice * self.ref_length / self.reynolds
        self.tau0 = max(3.0 * nu + 0.5, 0.5005)
        # Absorbing sponge layers over the first 8 % and last 12 % of the tunnel. There the
        # post-collision state is relaxed toward the uniform freestream (a "relaxation zone"), which
        # absorbs pressure waves of any wavelength. Without it a domain-sized acoustic mode bounced
        # between inlet and outlet, made the speed field pulse by ~±9 % and swamped the shedding
        # peak in the Strouhal spectrum. The outlet layer also gets extra viscosity so wake vortices
        # leave without reflecting.
        x = np.arange(self.nx, dtype=np.float32)
        w_in, w_out = max(4.0, 0.08 * self.nx), max(8.0, 0.12 * self.nx)
        s_in = np.clip((w_in - x) / w_in, 0.0, 1.0)
        s_out = np.clip((x - (self.nx - 1 - w_out)) / w_out, 0.0, 1.0)
        self.sponge = (0.08 * (s_in ** 2 + s_out ** 2)).astype(np.float32)
        self.tau_x = (self.tau0 + 0.5 * s_out ** 2).astype(np.float32)

    @property
    def nu(self) -> float:
        """Molecular lattice viscosity in effect (after the stability floor)."""
        return (self.tau0 - 0.5) / 3.0

    def reset(self) -> None:
        """Restart from uniform flow (the obstacle is kept)."""
        u0 = 0.0
        self.f[:] = _feq_np(np.ones((self.ny, self.nx), np.float32), np.full((self.ny, self.nx), u0, np.float32),
                            np.zeros((self.ny, self.nx), np.float32))
        self.f[:, self.solid] = W[:, None]
        self.f2[:] = self.f
        self.ux[:] = 0.0
        self.uy[:] = 0.0
        self.rho[:] = 1.0
        self.dye[:] = 0.0
        self.steps = 0
        self.history.clear()
        self._force = np.zeros(2)
        self.cl = 0.0
        self.cd = 0.0
        self._spawn_particles(all_new=True)
        self._update_tau()

    # -- stepping ----------------------------------------------------------
    @property
    def time(self) -> float:
        """Convective time t* = t U / L (number of reference lengths the flow has travelled)."""
        return self.steps * self.u_lattice / max(self.ref_length, 1e-9)

    def _inlet_speed(self) -> float:
        ramp = min(1.0, (self.steps + 1) / self.ramp_steps)
        return self.u_lattice * (3.0 * ramp * ramp - 2.0 * ramp ** 3)   # smoothstep ramp

    def step(self, n: int, dye: bool = True, particles: bool = True) -> int:
        """Advance ``n`` lattice steps (plus the tracers). Returns the number of steps done.

        Returns 0 without blocking when the 3-D solver is using the numba kernels at that moment.
        """
        n = max(int(n), 0)
        if n == 0:
            return 0
        use_nb = accel.HAVE_NUMBA
        locked = False
        if use_nb:
            locked = accel._KERNEL_LOCK.acquire(blocking=False)
            if not locked:
                return 0          # the 3-D solver is in a kernel right now; try again next frame
        try:
            force = np.zeros(2)
            t_start = time.perf_counter()
            for _ in range(n):
                u_in = self._inlet_speed()
                if u_in != self._ft_u:                 # constant once the start-up ramp is over
                    cu = 3.0 * CX * u_in
                    self._ftarget = (W * (1.0 + cu + 0.5 * cu * cu - 1.5 * u_in * u_in)).astype(np.float32)
                    self._ft_u = u_in
                ftarget = self._ftarget
                if use_nb:
                    try:
                        _lbm_step_nb(self.f, self.f2, self.solid, self.special, np.float32(u_in), self.tau_x,
                                     np.float32(self.smag), self.ux, self.uy, self.rho, force, CX, CY, W, OPP, REFY,
                                     self.sponge, ftarget)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("Live Flow Lab: numba kernel failed (%s); using NumPy.", exc)
                        use_nb = False
                        _lbm_step_np(self.f, self.f2, self.solid, u_in, self.tau_x, self.smag, self.ux, self.uy,
                                     self.rho, force, self.sponge, ftarget)
                else:
                    _lbm_step_np(self.f, self.f2, self.solid, u_in, self.tau_x, self.smag, self.ux, self.uy,
                                 self.rho, force, self.sponge, ftarget)
                self.f, self.f2 = self.f2, self.f
                self.steps += 1
            # Cost of the lattice steps alone (tracers and checks excluded), for adaptive frame budgets.
            self.last_step_ms = (time.perf_counter() - t_start) * 1000.0 / n
            if dye:
                self._advect_dye(float(n), use_nb)
        finally:
            if locked:
                accel._KERNEL_LOCK.release()
        if not np.isfinite(self.f).all():
            log.warning("Live Flow Lab diverged; restarting from uniform flow (try a lower Reynolds number).")
            self.reset()
            return n
        q = 0.5 * self.u_lattice ** 2 * self.ref_length * n
        self.cd = float(force[0] / q)
        self.cl = float(force[1] / q)
        self.history.append(self.time, self.cl, self.cd)
        if particles:
            self._advect_particles(float(n))
        return n

    # -- tracers -----------------------------------------------------------
    def _advect_dye(self, dt: float, use_nb: bool) -> None:
        if use_nb:
            _advect_dye_nb(self.dye, self._dye2, self.ux, self.uy, self.solid, np.float32(dt), np.float32(0.9995))
        else:
            _advect_dye_np(self.dye, self._dye2, self.ux, self.uy, self.solid, dt, 0.9995)
        self.dye, self._dye2 = self._dye2, self.dye
        rows = np.arange(self.ny)
        on = ((rows + self.dye_spacing // 2) % self.dye_spacing) < max(1, (3 * self.dye_spacing) // 10)
        self.dye[on, 0:3] = 1.0
        self.dye[~on, 0:3] = 0.0

    def _spawn_particles(self, all_new: bool = False) -> None:
        n = self.n_particles
        if all_new or not hasattr(self, "px"):
            self.px = self._rng.uniform(0.0, self.nx - 1.0, n).astype(np.float32)
            self.py = self._rng.uniform(0.0, self.ny - 1.0, n).astype(np.float32)
            self.page = np.zeros(n, dtype=np.float32)

    def _advect_particles(self, dt: float) -> None:
        nx, ny = self.nx, self.ny
        for _ in range(2):                         # two half steps (midpoint) for smoother paths
            i0 = np.clip(self.px.astype(np.int64), 0, nx - 2)
            j0 = np.clip(self.py.astype(np.int64), 0, ny - 2)
            fx = np.clip(self.px - i0, 0.0, 1.0)
            fy = np.clip(self.py - j0, 0.0, 1.0)

            def bil(a: F32) -> F32:
                return ((1 - fx) * (1 - fy) * a[j0, i0] + fx * (1 - fy) * a[j0, i0 + 1]
                        + (1 - fx) * fy * a[j0 + 1, i0] + fx * fy * a[j0 + 1, i0 + 1])
            self.px += bil(self.ux) * dt * 0.5
            self.py += bil(self.uy) * dt * 0.5
        self.page += dt
        ix = np.clip(self.px.astype(np.int64), 0, nx - 1)
        iy = np.clip(self.py.astype(np.int64), 0, ny - 1)
        dead = ((self.px < 0) | (self.px >= nx - 1) | (self.py < 0) | (self.py >= ny - 1)
                | self.solid[iy, ix] | (self.page > 40.0 * nx / max(self.u_lattice, 1e-6)))
        k = int(dead.sum())
        if k:
            self.px[dead] = self._rng.uniform(0.0, 2.0, k)
            self.py[dead] = self._rng.uniform(0.0, ny - 1.0, k)
            self.page[dead] = 0.0

    # -- derived fields ----------------------------------------------------
    def speed(self) -> F32:
        """|u| / U_inf (1 = freestream)."""
        return np.sqrt(self.ux * self.ux + self.uy * self.uy) / self.u_lattice

    def vorticity(self) -> F32:
        """Out-of-plane vorticity scaled by U/L (positive = counter-clockwise)."""
        dvdx = np.gradient(self.uy, axis=1)
        dudy = np.gradient(self.ux, axis=0)
        w = (dvdx - dudy) * (self.ref_length / self.u_lattice)
        w[self.solid] = 0.0
        return w

    def pressure_cp(self) -> F32:
        """Pressure coefficient from the lattice density (p = rho c_s^2), referenced to the inlet."""
        p_ref = float(np.mean(self.rho[:, 2:6]))
        cp = (self.rho - p_ref) / 3.0 / (0.5 * self.u_lattice ** 2)
        cp[self.solid] = 0.0
        return cp

    def mean_coefficients(self, window_time: float = 10.0) -> tuple[float, float]:
        """Time-averaged (CL, CD) over the last ``window_time`` convective units."""
        h = self.history
        if not h.t:
            return 0.0, 0.0
        t = np.asarray(h.t)
        sel = t >= t[-1] - window_time
        return float(np.mean(np.asarray(h.cl)[sel])), float(np.mean(np.asarray(h.cd)[sel]))

    def strouhal(self, min_time: float = 10.0) -> Optional[float]:
        """Shedding Strouhal number St = f D / U from the CL signal (None until it is periodic)."""
        h = self.history
        if len(h.t) < 64 or h.t[-1] - h.t[0] < min_time:
            return None
        t = np.asarray(h.t)
        sel = t >= t[-1] - 0.6 * (t[-1] - t[0])            # skip the start-up transient
        t, cl = t[sel], np.asarray(h.cl)[sel]
        if len(t) < 64:
            return None
        tu = np.linspace(t[0], t[-1], 1024)
        s = np.interp(tu, t, cl)
        s = s - s.mean()
        if np.std(s) < 1e-3:
            return None                                      # steady (no shedding)
        spec = np.abs(np.fft.rfft(s * np.hanning(len(s))))
        freq = np.fft.rfftfreq(len(s), d=tu[1] - tu[0])     # per convective time unit (L/U)
        k = int(np.argmax(spec[1:]) + 1)
        f = freq[k]                                          # = f_phys * L / U
        return float(f * self.frontal_length / max(self.ref_length, 1e-9))


def warmup() -> None:
    """Compile the Flow Lab kernels (or load them from the on-disk cache) without running them, so the
    kernel lock is never held during a (possibly long) first-time compilation."""
    if not accel.HAVE_NUMBA:
        return
    lab = FlowLab2D(16, 12)
    ft = np.zeros(9, np.float32)
    accel.precompile(_lbm_step_nb, lab.f, lab.f2, lab.solid, lab.special, np.float32(0.05), lab.tau_x,
                     np.float32(lab.smag), lab.ux, lab.uy, lab.rho, np.zeros(2), CX, CY, W, OPP, REFY, lab.sponge, ft)
    accel.precompile(_advect_dye_nb, lab.dye, lab._dye2, lab.ux, lab.uy, lab.solid, np.float32(1.0),
                     np.float32(1.0))
    z = np.zeros(lab.ux.shape, np.float32)
    w4 = np.ones(4, np.float32)
    accel.precompile(_lic_nb, z, z, lab.solid, z, w4, w4, np.empty_like(z))
