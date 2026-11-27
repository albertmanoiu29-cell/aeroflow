"""Supersonic Lab solver: 2-D compressible Euler equations (shock-capturing finite volume).

Method
------
* Conservative variables ``U = (rho, rho u, rho v, E)`` of a perfect gas (gamma = 1.4) on a uniform
  Cartesian grid of unit cells, non-dimensionalised by the freestream: ``rho_inf = 1``,
  ``a_inf = 1`` (so ``p_inf = 1/gamma`` and ``u_inf = M_inf``).
* **HLLC approximate Riemann solver** (Toro) for the face fluxes, with an automatic switch to the
  more dissipative **HLL** flux at strong-shock faces (pressure ratio > 2). The hybrid suppresses
  the "carbuncle" instability that pure HLLC shows in front of blunt bodies at high Mach.
* **MUSCL** reconstruction of primitive variables with the minmod limiter (second order, no new
  extrema); reconstruction drops to first order next to walls and wherever it would give a
  non-positive density or pressure.
* **SSP Runge-Kutta 2** time integration with a global CFL time step.
* **Bodies** are cell masks with reflective (slip) walls: on every fluid-solid face the flux is the
  HLLC flux between the fluid state and its mirror image (normal velocity reversed), i.e. an exact
  impermeable inviscid wall.
* Boundaries: supersonic inflow (fixed freestream) on the left, zero-gradient outflow elsewhere.

The inviscid model is the standard tool for shock structure: shock positions, angles and jumps,
expansion fans and the sonic line are captured. Viscous effects (boundary layers, shock / boundary-
layer interaction) are not modelled.

Compute devices (see :mod:`core.compute`):

* **GPU** — CUDA port of the same kernels (:mod:`core.supersonic_gpu`), double precision. The state
  stays on the GPU; each batch of steps is queued asynchronously and only the fields are copied back.
* **CPU** — numba-compiled parallel kernels; calls are serialised with the shared kernel lock of
  :mod:`core.accel` (non-blocking from the GUI). Without numba a vectorised first-order HLL fallback
  keeps the Lab usable (slower and more diffusive).

:meth:`Supersonic2D.begin_step` / :meth:`Supersonic2D.finish_step` split a batch so the caller can draw
while the solver works (on the GPU, or on a background thread for the CPU). Readers always see the last
*finished* state: :meth:`Supersonic2D.primitives` returns a published copy that is only replaced in
:meth:`finish_step`, so an in-flight batch never tears a frame.
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import Future
from typing import Final, Optional

import numpy as np
from numpy.typing import NDArray

from core import accel, compute

log = logging.getLogger("aeroflow.core.supersonic")

F64 = NDArray[np.float64]
GAMMA: Final[float] = 1.4
RHO_FLOOR: Final[float] = 1e-6
P_FLOOR: Final[float] = 1e-7
CFL: Final[float] = 0.5            # of the 2-D sum (|u|+c) + (|v|+c); stability bound is 1 (2x margin)
G = 2                              # ghost layers


# ---------------------------------------------------------------------------
# numba kernels
# ---------------------------------------------------------------------------
if accel.HAVE_NUMBA:
    from numba import njit, prange

    @njit(inline="always", fastmath=False, cache=True)
    def _minmod(a, b):  # pragma: no cover - compiled
        if a * b <= 0.0:
            return 0.0
        return a if abs(a) < abs(b) else b

    @njit(cache=True, fastmath=False)
    def _riemann_flux(rl, ul, vl, pl, rr, ur, vr, pr, g):  # pragma: no cover - compiled
        """Normal-direction flux between left/right primitive states (u normal, v tangential).

        HLLC, switching to HLL across strong shocks (anti-carbuncle)."""
        cl = math.sqrt(g * pl / rl)
        cr = math.sqrt(g * pr / rr)
        el = pl / (g - 1.0) + 0.5 * rl * (ul * ul + vl * vl)
        er = pr / (g - 1.0) + 0.5 * rr * (ur * ur + vr * vr)
        sl = min(ul - cl, ur - cr)
        sr = max(ul + cl, ur + cr)
        fl0, fl1, fl2, fl3 = rl * ul, rl * ul * ul + pl, rl * ul * vl, ul * (el + pl)
        fr0, fr1, fr2, fr3 = rr * ur, rr * ur * ur + pr, rr * ur * vr, ur * (er + pr)
        if sl >= 0.0:
            return fl0, fl1, fl2, fl3
        if sr <= 0.0:
            return fr0, fr1, fr2, fr3
        strong = max(pl, pr) > 2.0 * min(pl, pr)
        if strong:                                       # HLL
            inv = 1.0 / (sr - sl)
            return ((sr * fl0 - sl * fr0 + sl * sr * (rr - rl)) * inv,
                    (sr * fl1 - sl * fr1 + sl * sr * (rr * ur - rl * ul)) * inv,
                    (sr * fl2 - sl * fr2 + sl * sr * (rr * vr - rl * vl)) * inv,
                    (sr * fl3 - sl * fr3 + sl * sr * (er - el)) * inv)
        den = rl * (sl - ul) - rr * (sr - ur)
        sm = (pr - pl + rl * ul * (sl - ul) - rr * ur * (sr - ur)) / den
        if sm >= 0.0:
            c = rl * (sl - ul) / (sl - sm)
            u0, u1, u2 = c, c * sm, c * vl
            u3 = c * (el / rl + (sm - ul) * (sm + pl / (rl * (sl - ul))))
            return (fl0 + sl * (u0 - rl), fl1 + sl * (u1 - rl * ul), fl2 + sl * (u2 - rl * vl),
                    fl3 + sl * (u3 - el))
        c = rr * (sr - ur) / (sr - sm)
        u0, u1, u2 = c, c * sm, c * vr
        u3 = c * (er / rr + (sm - ur) * (sm + pr / (rr * (sr - ur))))
        return (fr0 + sr * (u0 - rr), fr1 + sr * (u1 - rr * ur), fr2 + sr * (u2 - rr * vr),
                fr3 + sr * (u3 - er))

    @njit(inline="always", cache=True)
    def _muscl(a0, a1, a2, a3):  # pragma: no cover - compiled
        """Left / right face values between cells a1 | a2 (minmod-limited linear reconstruction)."""
        return a1 + 0.5 * _minmod(a1 - a0, a2 - a1), a2 - 0.5 * _minmod(a2 - a1, a3 - a2)

    @njit(cache=True, nogil=True)
    def _fill_ib_nb(W, gj, gi, sj, si, sw, nrm):  # pragma: no cover - compiled
        """Immersed-boundary ghost states: interpolate the flow at each ghost cell's mirror image
        across the true wall, keep rho and p, reflect the velocity about the wall normal."""
        for k in range(gj.shape[0]):
            r = 0.0
            u = 0.0
            v = 0.0
            p = 0.0
            for q in range(4):
                w = sw[k, q]
                if w == 0.0:
                    continue
                r += w * W[0, sj[k, q], si[k, q]]
                u += w * W[1, sj[k, q], si[k, q]]
                v += w * W[2, sj[k, q], si[k, q]]
                p += w * W[3, sj[k, q], si[k, q]]
            nx_ = nrm[k, 0]
            ny_ = nrm[k, 1]
            un = u * nx_ + v * ny_
            W[0, gj[k], gi[k]] = r
            W[1, gj[k], gi[k]] = u - 2.0 * un * nx_
            W[2, gj[k], gi[k]] = v - 2.0 * un * ny_
            W[3, gj[k], gi[k]] = p

    @njit(cache=True, fastmath=True)
    def _face_flux(W, solid, lj, li, rj, ri, dj, di, nc, tc, g):  # pragma: no cover - compiled
        """Flux through the face between padded cells L=(lj,li) and R=(rj,ri) in the (dj,di) direction.

        ``nc`` / ``tc`` are the indices of the normal / tangential velocity components in ``W``.
        ``solid`` here is the *wall* mask: solid cells handled by face reflection. Immersed-boundary
        ghost cells are not in it - they carry mirrored states in ``W`` and are treated like fluid.
        Returns (mass, normal momentum, tangential momentum, energy) or zeros between two walls."""
        sl_ = solid[lj, li]
        sr_ = solid[rj, ri]
        if sl_ and sr_:
            return 0.0, 0.0, 0.0, 0.0
        rl = W[0, lj, li]
        pl = W[3, lj, li]
        nl = W[nc, lj, li]
        tl = W[tc, lj, li]
        rr = W[0, rj, ri]
        pr = W[3, rj, ri]
        nr = W[nc, rj, ri]
        tr = W[tc, rj, ri]
        # Wall faces reflect about the *face* normal: that is what makes the face exactly impermeable
        # (reflecting about the true surface normal instead lets mass leak through grid faces).
        if sl_:                                   # slip wall on the left: mirror the fluid state
            rl = rr
            nl = -nr
            tl = tr
            pl = pr
        elif sr_:                                 # slip wall on the right
            rr = rl
            nr = -nl
            tr = tl
            pr = pl
        else:
            llj = lj - dj
            lli = li - di
            rrj = rj + dj
            rri = ri + di
            if not (solid[llj, lli] or solid[rrj, rri]):          # second order away from walls
                a, b = _muscl(W[0, llj, lli], rl, rr, W[0, rrj, rri])
                c, d = _muscl(W[3, llj, lli], pl, pr, W[3, rrj, rri])
                if a > RHO_FLOOR and b > RHO_FLOOR and c > P_FLOOR and d > P_FLOOR:
                    rl = a
                    rr = b
                    pl = c
                    pr = d
                    nl, nr = _muscl(W[nc, llj, lli], nl, nr, W[nc, rrj, rri])
                    tl, tr = _muscl(W[tc, llj, lli], tl, tr, W[tc, rrj, rri])
        return _riemann_flux(rl, nl, tl, pl, rr, nr, tr, pr, g)

    @njit(parallel=True, cache=True, nogil=True)
    def _flux_x_nb(W, solid, g, F):  # pragma: no cover - compiled
        """x-face fluxes F (4, ny, nx+1); face f lies between interior cells f-1 and f."""
        _, ny, nf = F.shape
        for jj in prange(ny):
            j = np.int64(jj) + 2
            for f in range(nf):
                li = np.int64(f) + 1
                f0, f1, f2, f3 = _face_flux(W, solid, j, li, j, li + 1, 0, 1, 1, 2, g)
                F[0, jj, f] = f0
                F[1, jj, f] = f1          # normal momentum = x
                F[2, jj, f] = f2          # tangential = y
                F[3, jj, f] = f3

    @njit(parallel=True, cache=True, nogil=True)
    def _flux_y_nb(W, solid, g, Gf):  # pragma: no cover - compiled
        """y-face fluxes G (4, ny+1, nx); face f lies between interior rows f-1 and f."""
        _, nf, nx = Gf.shape
        for ff in prange(nf):
            lj = np.int64(ff) + 1
            for ii in range(nx):
                i = np.int64(ii) + 2
                f0, f1, f2, f3 = _face_flux(W, solid, lj, i, lj + 1, i, 1, 0, 2, 1, g)
                Gf[0, ff, ii] = f0
                Gf[1, ff, ii] = f2        # tangential = x
                Gf[2, ff, ii] = f1        # normal momentum = y
                Gf[3, ff, ii] = f3

    @njit(parallel=True, cache=True, nogil=True)
    def _divergence_nb(F, Gf, solid, R):  # pragma: no cover - compiled
        _, ny, nx = R.shape
        for jj in prange(ny):
            for ii in range(nx):
                if solid[jj + 2, ii + 2]:
                    for k in range(4):
                        R[k, jj, ii] = 0.0
                    continue
                for k in range(4):
                    R[k, jj, ii] = F[k, jj, ii] - F[k, jj, ii + 1] + Gf[k, jj, ii] - Gf[k, jj + 1, ii]

    @njit(parallel=True, cache=True, nogil=True)
    def _rk_update_nb(Ubase, Ucur, F, Gf, solid, dt, a, out):  # pragma: no cover - compiled
        """Fused SSP-RK stage: out = a*Ubase + (1-a)*(Ucur - dt*div F) - no temporaries. ``out`` may be
        ``Ubase`` (each element is read before it is written). Body cells keep their (rest) state."""
        _, ny, nx = out.shape
        b = 1.0 - a
        for jj in prange(ny):
            for ii in range(nx):
                if solid[jj + 2, ii + 2]:
                    continue
                for k in range(4):
                    r = F[k, jj, ii] - F[k, jj, ii + 1] + Gf[k, jj, ii] - Gf[k, jj + 1, ii]
                    out[k, jj, ii] = a * Ubase[k, jj, ii] + b * (Ucur[k, jj, ii] + dt * r)

    def _residual_nb(W, wall, solid, g, R, F, Gf):  # pragma: no cover - thin wrapper
        """R = -div F for every fluid cell (each face flux is computed once). ``wall``: face-reflection
        solids; ``solid``: every body cell (zero residual)."""
        _flux_x_nb(W, wall, g, F)
        _flux_y_nb(W, wall, g, Gf)
        _divergence_nb(F, Gf, solid, R)

    @njit(parallel=True, cache=True, nogil=True)
    def _primitives_nb(U, solid, g, W):  # pragma: no cover - compiled
        """Interior primitives (with positivity floors) into the padded array ``W``."""
        _, ny, nx = U.shape
        for jj in prange(ny):
            for ii in range(nx):
                j = jj + 2
                i = ii + 2
                r = U[0, jj, ii]
                if r < RHO_FLOOR:
                    r = RHO_FLOOR
                u = U[1, jj, ii] / r
                v = U[2, jj, ii] / r
                p = (g - 1.0) * (U[3, jj, ii] - 0.5 * r * (u * u + v * v))
                if p < P_FLOOR:
                    p = P_FLOOR
                W[0, j, i] = r
                W[1, j, i] = u
                W[2, j, i] = v
                W[3, j, i] = p

    @njit(parallel=True, cache=True, nogil=True)
    def _max_speed_nb(W, solid, g):  # pragma: no cover - compiled
        # Per-row maxima in parallel, then one serial max (a parallel max-reduction trips numba's
        # parfor analysis).
        _, nyp, nxp = W.shape
        rows = np.zeros(nyp - 4)
        for jj in prange(nyp - 4):
            j = np.int64(jj) + 2
            row = 0.0
            for i in range(2, nxp - 2):
                if solid[j, i]:
                    continue
                c = math.sqrt(g * W[3, j, i] / W[0, j, i])
                s = abs(W[1, j, i]) + abs(W[2, j, i]) + 2.0 * c
                if s > row:
                    row = s
            rows[jj] = row
        return rows.max()


# ---------------------------------------------------------------------------
# NumPy fallback (first-order HLL, vectorised) - used only without numba
# ---------------------------------------------------------------------------
def _hll_np(rl, nl, tl, pl, rr, nr, tr, pr, g):
    cl = np.sqrt(g * pl / rl)
    cr = np.sqrt(g * pr / rr)
    el = pl / (g - 1) + 0.5 * rl * (nl * nl + tl * tl)
    er = pr / (g - 1) + 0.5 * rr * (nr * nr + tr * tr)
    sl = np.minimum(nl - cl, nr - cr)
    sr = np.maximum(nl + cl, nr + cr)
    fl = np.stack([rl * nl, rl * nl * nl + pl, rl * nl * tl, nl * (el + pl)])
    fr = np.stack([rr * nr, rr * nr * nr + pr, rr * nr * tr, nr * (er + pr)])
    ul = np.stack([rl, rl * nl, rl * tl, el])
    ur = np.stack([rr, rr * nr, rr * tr, er])
    mid = (sr * fl - sl * fr + sl * sr * (ur - ul)) / np.where(sr - sl == 0, 1e-12, sr - sl)
    return np.where(sl >= 0, fl, np.where(sr <= 0, fr, mid))


def _residual_np(W: F64, wall: NDArray[np.bool_], solid: NDArray[np.bool_], g: float, R: F64) -> None:
    ny, nx = R.shape[1:]
    R[:] = 0.0
    for axis in (0, 1):
        if axis == 0:   # faces between padded columns c and c+1 for c = 1..nx+1
            Lw, Rw = W[:, 2:-2, 1:-2], W[:, 2:-2, 2:-1]
            Ls, Rs = wall[2:-2, 1:-2], wall[2:-2, 2:-1]
            nl, tl, nr, tr = Lw[1], Lw[2], Rw[1], Rw[2]
        else:
            Lw, Rw = W[:, 1:-2, 2:-2], W[:, 2:-1, 2:-2]
            Ls, Rs = wall[1:-2, 2:-2], wall[2:-1, 2:-2]
            nl, tl, nr, tr = Lw[2], Lw[1], Rw[2], Rw[1]
        rl, pl, rr, pr = Lw[0].copy(), Lw[3].copy(), Rw[0].copy(), Rw[3].copy()
        nl, tl, nr, tr = nl.copy(), tl.copy(), nr.copy(), tr.copy()
        m = Ls & ~Rs
        rl[m], nl[m], tl[m], pl[m] = rr[m], -nr[m], tr[m], pr[m]
        m = Rs & ~Ls
        rr[m], nr[m], tr[m], pr[m] = rl[m], -nl[m], tl[m], pl[m]
        f = _hll_np(rl, nl, tl, pl, rr, nr, tr, pr, g)
        f[:, Ls & Rs] = 0.0
        fx, fy = (f[1], f[2]) if axis == 0 else (f[2], f[1])
        flux = np.stack([f[0], fx, fy, f[3]])
        if axis == 0:
            R += flux[:, :, :-1] - flux[:, :, 1:]
        else:
            R += flux[:, :-1, :] - flux[:, 1:, :]
    R[:, solid[2:-2, 2:-2]] = 0.0


def _nearest_on_segments(points: F64, segs: F64) -> F64:
    """Nearest point on a segment soup for each of ``points`` (P, 2) (chunked to bound memory)."""
    a, ab = segs[:, 0, :], segs[:, 1, :] - segs[:, 0, :]
    denom = np.maximum((ab * ab).sum(1), 1e-30)
    nearest = np.empty_like(points)
    for k0 in range(0, len(points), 2048):
        cc = points[k0:k0 + 2048]
        t = np.clip(((cc[:, None, :] - a[None]) * ab[None]).sum(2) / denom[None], 0.0, 1.0)
        proj = a[None] + t[..., None] * ab[None]
        d2 = ((cc[:, None, :] - proj) ** 2).sum(2)
        k = np.argmin(d2, axis=1)
        nearest[k0:k0 + 2048] = proj[np.arange(len(cc)), k]
    return nearest


# ---------------------------------------------------------------------------
# Solver object
# ---------------------------------------------------------------------------
class Supersonic2D:
    """Real-time 2-D compressible Euler solver around a cell-mask body."""

    def __init__(self, nx: int = 400, ny: int = 200, mach: float = 2.0, gamma: float = GAMMA) -> None:
        self.gamma = float(gamma)
        self.mach = float(mach)
        self.ref_length = 0.25 * ny
        self.last_step_ms = 1.0
        self.device = "CPU"                 # where the last batch ran: "GPU" or "CPU"
        self._gpu = None                    # core.supersonic_gpu.SupersonicGPU while on the GPU
        self._gpu_dirty = True              # host state changed: upload before the next GPU batch
        self._future: Optional[Future] = None
        self._pending_gpu = 0
        self._U_stale = False               # the GPU holds a newer U than the host (fetched on demand)
        self._mach_pub: Optional[F64] = None   # Mach field computed on the GPU for the published state
        self.resize(nx, ny)

    #: GPU field precision: float64 (identical to the CPU) by default; float32 is several times faster.
    gpu_single = False

    def set_gpu_precision(self, single: bool) -> None:
        """Float32 (fast) or float64 (exact) GPU fields; the state carries over."""
        if bool(single) == self.gpu_single:
            return
        self._drain()
        self._leave_gpu()                    # fetch the state, free the old device arrays
        self.gpu_single = bool(single)

    @property
    def U(self) -> F64:
        """Conserved variables ``(rho, rho u, rho v, E)`` (ny, nx); fetched from the GPU when needed."""
        if self._U_stale:
            self._U_stale = False
            if self._gpu is not None:
                try:
                    self._gpu.download_U(self._U)
                except Exception as exc:  # noqa: BLE001 - keep the last host copy
                    compute.gpu_failed(exc)
                    self._gpu = None
                    self._gpu_dirty = True
        return self._U

    @U.setter
    def U(self, value: F64) -> None:
        self._U = value
        self._U_stale = False

    # -- set-up ------------------------------------------------------------
    def resize(self, nx: int, ny: int) -> None:
        self._drain()
        self._gpu = None
        self.nx, self.ny = int(nx), int(ny)
        self.solid_int = np.zeros((self.ny, self.nx), dtype=np.bool_)
        self.U = np.zeros((4, self.ny, self.nx))
        self._U1 = np.zeros_like(self.U)
        self._R = np.zeros_like(self.U)
        self._F = np.zeros((4, self.ny, self.nx + 1))        # x-face fluxes
        self._Gf = np.zeros((4, self.ny + 1, self.nx))       # y-face fluxes
        self.W = np.zeros((4, self.ny + 2 * G, self.nx + 2 * G))
        self._Wpub = np.zeros_like(self.W)                   # last finished state, read by primitives()
        self._solid_pad = np.zeros((self.ny + 2 * G, self.nx + 2 * G), dtype=np.bool_)
        self._wall_pad = np.zeros_like(self._solid_pad)     # solids handled by face reflection
        self._clear_ib()
        self.reset()

    def _clear_ib(self) -> None:
        self.ib_count = 0
        self._ib_gj = np.zeros(0, np.int64)
        self._ib_gi = np.zeros(0, np.int64)
        self._ib_sj = np.zeros((0, 4), np.int64)
        self._ib_si = np.zeros((0, 4), np.int64)
        self._ib_sw = np.zeros((0, 4))
        self._ib_n = np.zeros((0, 2))

    @property
    def freestream(self) -> tuple[float, float, float, float]:
        """``(rho, u, v, p)`` of the oncoming flow (non-dimensional)."""
        return 1.0, self.mach, 0.0, 1.0 / self.gamma

    def _cons(self, r: float, u: float, v: float, p: float) -> tuple[float, float, float, float]:
        return r, r * u, r * v, p / (self.gamma - 1.0) + 0.5 * r * (u * u + v * v)

    def reset(self) -> None:
        """Impulsive start: uniform freestream everywhere outside the body."""
        self._drain()
        self._gpu_dirty = True
        self._U_stale = False                            # U is overwritten entirely: no need to fetch it
        c = self._cons(*self.freestream)
        for k in range(4):
            self.U[k] = c[k]
        rest = self._cons(1.0, 0.0, 0.0, 1.0 / self.gamma)
        for k in range(4):
            self.U[k][self.solid_int] = rest[k]
        self.time = 0.0
        self.steps = 0
        self._update_primitives()

    def set_mach(self, mach: float) -> None:
        """Change the freestream Mach number (the inflow changes; the field keeps evolving)."""
        self.mach = float(np.clip(mach, 0.3, 8.0))

    def set_obstacle(self, mask: NDArray[np.bool_], ref_length_cells: float,
                     segments: Optional[NDArray[np.float64]] = None) -> None:
        """Set the body. ``segments`` (M, 2, 2) is the exact outline in cell coordinates (cell (j, i)
        centred at (i, j)); with it, walls use the immersed-boundary ghost-cell method (true surface
        position and normal). Without it - or for cells in regions too thin for a ghost state - the
        walls reflect at the cell faces."""
        mask = np.asarray(mask, dtype=np.bool_).copy()
        if mask.shape != (self.ny, self.nx):
            raise ValueError(f"mask shape {mask.shape} != grid {(self.ny, self.nx)}")
        self._drain()                                    # host U must be current before it is edited
        self._gpu_dirty = True
        mask[:, :3] = False                              # keep the inflow region clear
        mask[:, -3:] = False
        mask[:3, :] = False
        mask[-3:, :] = False
        newly_fluid = self.solid_int & ~mask
        self.solid_int = mask
        self._solid_pad[:] = False
        self._solid_pad[G:-G, G:-G] = mask
        self.segments = None if segments is None else np.asarray(segments, dtype=float)
        self._build_ib(segments)
        self._build_surface_probes()
        if newly_fluid.any():                            # uncovered cells start from the freestream
            c = self._cons(*self.freestream)
            for k in range(4):
                self.U[k][newly_fluid] = c[k]
        self.ref_length = max(float(ref_length_cells), 2.0)
        self._update_primitives()

    def _build_ib(self, segments: Optional[NDArray[np.float64]]) -> None:
        """Pre-compute the immersed-boundary ghost cells: mirror point, interpolation stencil, normal."""
        self._clear_ib()
        s = self.solid_int
        self._wall_pad[:] = self._solid_pad
        if segments is None or len(segments) == 0 or not s.any():
            return
        seg = np.asarray(segments, dtype=float)
        fluid = ~s
        ny, nx = s.shape

        def shifted(a: NDArray[np.bool_], dj: int, di: int) -> NDArray[np.bool_]:
            out = np.zeros_like(a)
            ys = slice(max(dj, 0), ny + min(dj, 0))
            yd = slice(max(-dj, 0), ny + min(-dj, 0))
            xs = slice(max(di, 0), nx + min(di, 0))
            xd = slice(max(-di, 0), nx + min(-di, 0))
            out[yd, xd] = a[ys, xs]
            return out                                        # out[j, i] = a[j + dj, i + di]

        f_xp = shifted(fluid, 0, 1) | shifted(fluid, 0, 2)
        f_xm = shifted(fluid, 0, -1) | shifted(fluid, 0, -2)
        f_yp = shifted(fluid, 1, 0) | shifted(fluid, 2, 0)
        f_ym = shifted(fluid, -1, 0) | shifted(fluid, -2, 0)
        cand = s & (f_xp | f_xm | f_yp | f_ym)                # solid cells inside some fluid face stencil
        thin = (f_xp & f_xm) | (f_yp & f_ym)                  # fluid on opposite sides: no single ghost state
        js, iis = np.nonzero(cand & ~thin)
        if js.size == 0:
            return
        c = np.column_stack([iis, js]).astype(float)         # cell centres (x = i, y = j)
        nearest = _nearest_on_segments(c, seg)                # nearest point on the true outline
        nvec = nearest - c
        dist = np.linalg.norm(nvec, axis=1)
        img = 2.0 * nearest - c                               # mirror image of the ghost centre in the fluid
        n_hat = nvec / np.maximum(dist, 1e-12)[:, None]       # wall normal, pointing into the fluid
        i0 = np.floor(img[:, 0]).astype(np.int64)
        j0 = np.floor(img[:, 1]).astype(np.int64)
        fx, fy = img[:, 0] - i0, img[:, 1] - j0
        sj = np.stack([j0, j0, j0 + 1, j0 + 1], axis=1)
        si = np.stack([i0, i0 + 1, i0, i0 + 1], axis=1)
        sw = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], axis=1)
        inside = (sj >= 0) & (sj < ny) & (si >= 0) & (si < nx)
        sjc, sic = np.clip(sj, 0, ny - 1), np.clip(si, 0, nx - 1)
        sw = np.where(inside & fluid[sjc, sic], sw, 0.0)      # interpolate from fluid cells only
        wsum = sw.sum(1)
        ok = (wsum > 1e-6) & (dist > 1e-9)
        if not ok.any():
            return
        sw = sw[ok] / wsum[ok, None]
        self._ib_gj = (js[ok] + G).astype(np.int64)
        self._ib_gi = (iis[ok] + G).astype(np.int64)
        self._ib_sj = (sjc[ok] + G).astype(np.int64)
        self._ib_si = (sic[ok] + G).astype(np.int64)
        self._ib_sw = np.ascontiguousarray(sw)
        self._ib_n = np.ascontiguousarray(n_hat[ok])
        self.ib_count = int(ok.sum())
        self._wall_pad[self._ib_gj, self._ib_gi] = False      # these walls are represented by ghost states

    def _build_surface_probes(self, offset: float = 1.25) -> None:
        """Probe points ``offset`` cells off the true surface (outward), one per outline segment, used for
        accurate surface pressure, forces and the stagnation pressure (no staircase error)."""
        self._probe = None
        segs = self.segments
        if segs is None or len(segs) == 0:
            return
        # Subdivide long segments into ~1-cell pieces: a 4-segment diamond must still get dozens of
        # pressure samples per face (one sample per face made forces noisy and resolution-dependent).
        pieces = []
        for (p0, p1) in segs:
            n_sub = max(1, int(math.ceil(float(np.linalg.norm(p1 - p0)) / 1.0)))
            t = np.linspace(0.0, 1.0, n_sub + 1)
            pts_ = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
            pieces.append(np.stack([pts_[:-1], pts_[1:]], axis=1))
        sub = np.concatenate(pieces) if pieces else segs
        a, b = sub[:, 0, :], sub[:, 1, :]
        d = b - a
        length = np.linalg.norm(d, axis=1)
        keep = length > 1e-9
        a, d, length = a[keep], d[keep], length[keep]
        mid = a + 0.5 * d
        n = np.column_stack([d[:, 1], -d[:, 0]]) / length[:, None]
        # orient outward: the point just inside along -n must be solid
        test = mid - 0.75 * n
        ti = np.clip(np.round(test[:, 0]).astype(np.int64), 0, self.nx - 1)
        tj = np.clip(np.round(test[:, 1]).astype(np.int64), 0, self.ny - 1)
        flip = ~self.solid_int[tj, ti]
        n[flip] *= -1.0
        pts = mid + offset * n
        i0 = np.floor(pts[:, 0]).astype(np.int64)
        j0 = np.floor(pts[:, 1]).astype(np.int64)
        fx, fy = pts[:, 0] - i0, pts[:, 1] - j0
        sj = np.stack([j0, j0, j0 + 1, j0 + 1], axis=1)
        si = np.stack([i0, i0 + 1, i0, i0 + 1], axis=1)
        sw = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], axis=1)
        inside = (sj >= 0) & (sj < self.ny) & (si >= 0) & (si < self.nx)
        sj, si = np.clip(sj, 0, self.ny - 1), np.clip(si, 0, self.nx - 1)
        sw = np.where(inside & ~self.solid_int[sj, si], sw, 0.0)
        wsum = sw.sum(1)
        ok = wsum > 1e-6
        if not ok.any():
            return
        self._probe = dict(mid=mid[ok], n=n[ok], length=length[ok], sj=sj[ok], si=si[ok],
                           sw=sw[ok] / wsum[ok, None])

    def surface_pressure(self) -> Optional[dict[str, F64]]:
        """Surface samples: ``x, y`` (cells), outward normal ``nx, ny``, length, and ``p`` = p/p_inf."""
        pr = self._probe
        if pr is None:
            return None
        p = self.primitives()[3]
        vals = (p[pr["sj"], pr["si"]] * pr["sw"]).sum(1) * self.gamma
        return {"x": pr["mid"][:, 0], "y": pr["mid"][:, 1], "nx": pr["n"][:, 0], "ny": pr["n"][:, 1],
                "length": pr["length"], "p": vals}

    def _fill_ib(self, use_nb: bool) -> None:
        if self.ib_count == 0:
            return
        if use_nb:
            _fill_ib_nb(self.W, self._ib_gj, self._ib_gi, self._ib_sj, self._ib_si, self._ib_sw, self._ib_n)
            return
        vals = self.W[:, self._ib_sj, self._ib_si]                 # (4, G, 4)
        r, u, v, p = (vals * self._ib_sw[None]).sum(-1)
        un = u * self._ib_n[:, 0] + v * self._ib_n[:, 1]
        self.W[0, self._ib_gj, self._ib_gi] = r
        self.W[1, self._ib_gj, self._ib_gi] = u - 2.0 * un * self._ib_n[:, 0]
        self.W[2, self._ib_gj, self._ib_gi] = v - 2.0 * un * self._ib_n[:, 1]
        self.W[3, self._ib_gj, self._ib_gi] = p

    # -- stepping ------------------------------------------------------------
    def _fill_ghosts(self, W: F64) -> None:
        r, u, v, p = self.freestream
        g = G
        W[:, :, :g] = np.array([r, u, v, p])[:, None, None]            # supersonic inflow
        W[:, :, -g:] = W[:, :, -g - 1:-g]                               # outflow: zero gradient
        W[:, :g, :] = W[:, g:g + 1, :]                                  # far field top / bottom
        W[:, -g:, :] = W[:, -g - 1:-g, :]

    def _primitives(self, U: F64, use_nb: bool) -> None:
        if use_nb:
            _primitives_nb(U, self._solid_pad, self.gamma, self.W)
        else:
            r = np.maximum(U[0], RHO_FLOOR)
            u, v = U[1] / r, U[2] / r
            p = np.maximum((self.gamma - 1.0) * (U[3] - 0.5 * r * (u * u + v * v)), P_FLOOR)
            self.W[:, G:-G, G:-G] = np.stack([r, u, v, p])
        self._fill_ghosts(self.W)
        self._fill_ib(use_nb)

    def _update_primitives(self) -> None:
        """Host W from U, then publish it to readers (GUI thread only)."""
        self._primitives(self.U, False)
        np.copyto(self._Wpub, self.W)
        self._mach_pub = None

    def _residual(self, use_nb: bool) -> F64:
        if use_nb:
            _residual_nb(self.W, self._wall_pad, self._solid_pad, self.gamma, self._R, self._F, self._Gf)
        else:
            _residual_np(self.W, self._wall_pad, self._solid_pad, self.gamma, self._R)
        return self._R

    def _dt(self, use_nb: bool) -> float:
        if use_nb:
            s = float(_max_speed_nb(self.W, self._solid_pad, self.gamma))
        else:
            W = self.W[:, G:-G, G:-G]
            c = np.sqrt(self.gamma * W[3] / W[0])
            s = float(np.max((np.abs(W[1]) + np.abs(W[2]) + 2.0 * c)[~self.solid_int]))
        return CFL / max(s, 1e-9)

    def step(self, n: int) -> int:
        """Advance ``n`` time steps (SSP-RK2) and wait for them. Returns the steps done (0 if the CPU
        kernels are busy with another solver)."""
        self.begin_step(n)
        return self.finish_step()

    # -- asynchronous stepping ---------------------------------------------------
    @property
    def busy(self) -> bool:
        """True while a batch started by :meth:`begin_step` is still running."""
        if self._future is not None:
            return not self._future.done()
        return self._gpu is not None and self._gpu.busy()

    def begin_step(self, n: int, background: bool = False) -> None:
        """Start ``n`` steps without waiting: queued on the GPU, or (``background=True``) run on a worker
        thread for the CPU; otherwise the CPU steps run right here. Collect with :meth:`finish_step`."""
        self._drain()
        n = max(int(n), 0)
        if n == 0:
            return
        if compute.use_gpu(wait=not background) and self._begin_gpu(n):
            return
        self._leave_gpu()
        self.device = "CPU"
        if background:
            self._future = compute.cpu_executor().submit(self._step_cpu, n)
        else:
            fut: Future = Future()
            try:
                fut.set_result(self._step_cpu(n))
            except BaseException as exc:  # noqa: BLE001 - re-raised by finish_step
                fut.set_exception(exc)
            self._future = fut

    def finish_step(self) -> int:
        """Wait for the batch from :meth:`begin_step` and publish its result. Returns the steps done."""
        if self._pending_gpu:
            done = self._finish_gpu()
            # Any NaN anywhere reaches the CFL reduction, so a finite time means a finite field.
            ok = math.isfinite(self.time)
        elif self._future is not None:
            fut, self._future = self._future, None
            done = int(fut.result())
            ok = bool(np.isfinite(self.U).all()) if done else True
            if done and ok:
                np.copyto(self._Wpub, self.W)
                self._mach_pub = None
        else:
            return 0
        if done and not ok:
            log.warning("Supersonic solver diverged; restarting from the freestream.")
            self.reset()
        return done

    def _drain(self) -> None:
        """Finish any batch in flight (host arrays are current afterwards)."""
        if getattr(self, "_pending_gpu", 0) or getattr(self, "_future", None) is not None:
            try:
                self.finish_step()
            except Exception as exc:  # noqa: BLE001 - a failed batch must not block a reset
                log.debug("Supersonic batch failed while draining: %s", exc)

    # -- GPU path ------------------------------------------------------------------
    def _ib_arrays(self) -> dict:
        return {"count": self.ib_count, "gj": self._ib_gj, "gi": self._ib_gi, "sj": self._ib_sj,
                "si": self._ib_si, "sw": self._ib_sw, "n": self._ib_n}

    def _begin_gpu(self, n: int) -> bool:
        try:
            if (self._gpu is None or (self._gpu.nx, self._gpu.ny) != (self.nx, self.ny)
                    or self._gpu.single != self.gpu_single):
                from core.supersonic_gpu import SupersonicGPU

                self._gpu = SupersonicGPU(self.nx, self.ny, single=self.gpu_single)
                self._gpu_dirty = True
            if self._gpu_dirty:
                self._gpu.upload(self.U, self._solid_pad, self._wall_pad, self._ib_arrays())
                self._gpu_dirty = False
            self._gpu.enqueue(n, self.gamma, self.freestream, self.time)
        except Exception as exc:  # noqa: BLE001 - fall back to the CPU, never crash the Lab
            compute.gpu_failed(exc)
            self._gpu = None
            self._gpu_dirty = True
            return False
        self._pending_gpu = n
        self.device = "GPU"
        return True

    def _finish_gpu(self) -> int:
        self._pending_gpu = 0
        try:
            done, t, ms = self._gpu.wait()
        except Exception as exc:  # noqa: BLE001
            compute.gpu_failed(exc)
            self._gpu = None
            self._gpu_dirty = True
            return 0
        np.copyto(self._Wpub, self._gpu.h_W)             # published directly (host W is CPU scratch)
        # A copy, not the pinned buffer itself: the next batch's asynchronous download overwrites that.
        if self._mach_pub is None or self._mach_pub.shape != self._gpu.h_M.shape:
            self._mach_pub = np.empty(self._gpu.h_M.shape, dtype=np.float32)
        np.copyto(self._mach_pub, self._gpu.h_M)
        self._U_stale = True
        self.time = t if math.isfinite(t) else float("nan")
        self.steps += done
        if ms > 0.0:
            self.last_step_ms = ms
        return done

    def _leave_gpu(self) -> None:
        """Switching to the CPU: fetch the state from the GPU, then free the GPU memory."""
        if self._gpu is not None:
            _ = self.U                                    # pulls U if the GPU copy is newer
            self._gpu = None
            self._gpu_dirty = True
            self._mach_pub = None

    # -- CPU path ------------------------------------------------------------------
    def _step_cpu(self, n: int) -> int:
        """``n`` steps on the CPU (may run on a worker thread: touches no published state)."""
        use_nb = accel.HAVE_NUMBA
        locked = False
        if use_nb:
            locked = accel._KERNEL_LOCK.acquire(blocking=False)
            if not locked:
                return 0
        t0 = time.perf_counter()
        try:
            for _ in range(n):
                self._primitives(self.U, use_nb)
                dt = self._dt(use_nb)
                if use_nb:
                    wall, solid = self._wall_pad, self._solid_pad
                    # stage 1: U1 = U + dt L(U)
                    _flux_x_nb(self.W, wall, self.gamma, self._F)
                    _flux_y_nb(self.W, wall, self.gamma, self._Gf)
                    self._U1[:, solid[G:-G, G:-G]] = self.U[:, solid[G:-G, G:-G]]
                    _rk_update_nb(self.U, self.U, self._F, self._Gf, solid, dt, 0.0, self._U1)
                    # stage 2: U = 0.5 U + 0.5 (U1 + dt L(U1))
                    self._primitives(self._U1, use_nb)
                    _flux_x_nb(self.W, wall, self.gamma, self._F)
                    _flux_y_nb(self.W, wall, self.gamma, self._Gf)
                    _rk_update_nb(self.U, self._U1, self._F, self._Gf, solid, dt, 0.5, self.U)
                else:
                    R = self._residual(use_nb)
                    np.multiply(R, dt, out=self._U1)
                    self._U1 += self.U                               # stage 1: U1 = U + dt L(U)
                    self._primitives(self._U1, use_nb)
                    R = self._residual(use_nb)
                    self.U *= 0.5                                    # stage 2
                    self.U += 0.5 * (self._U1 + dt * R)
                self.time += dt
                self.steps += 1
        except Exception as exc:  # noqa: BLE001
            if use_nb:
                log.warning("Supersonic kernel failed (%s); switching to the NumPy fallback.", exc)
                accel._disable_numba(exc)
            raise
        finally:
            if locked:
                accel._KERNEL_LOCK.release()
        self.last_step_ms = (time.perf_counter() - t0) * 1000.0 / n
        self._primitives(self.U, False)                      # W consistent with the final U
        return n

    # -- derived fields --------------------------------------------------------
    def primitives(self) -> tuple[F64, F64, F64, F64]:
        """Interior ``(rho, u, v, p)`` views of the last finished state."""
        W = self._Wpub[:, G:-G, G:-G]
        return W[0], W[1], W[2], W[3]

    def mach_field(self) -> F64:
        if self._mach_pub is not None:                   # already computed on the GPU for this state
            return self._mach_pub.astype(np.float64)
        r, u, v, p = self.primitives()
        m = np.sqrt(u * u + v * v) / np.sqrt(self.gamma * p / r)
        m[self.solid_int] = 0.0
        return m

    def pressure_ratio(self) -> F64:
        """p / p_inf."""
        out = self.primitives()[3] * self.gamma
        out = out.copy()
        out[self.solid_int] = np.nan
        return out

    def density_ratio(self) -> F64:
        out = self.primitives()[0].copy()
        out[self.solid_int] = np.nan
        return out

    def temperature_ratio(self) -> F64:
        """T / T_inf = gamma p / rho."""
        r, _, _, p = self.primitives()
        out = self.gamma * p / r
        out[self.solid_int] = np.nan
        return out

    def total_pressure_ratio(self) -> F64:
        """p0 / p0_inf: 1 in loss-free flow, below 1 behind shocks (the entropy each shock creates)."""
        g = self.gamma
        m = self.mach_field()
        p = self.primitives()[3]
        p0 = p * (1.0 + 0.5 * (g - 1.0) * m * m) ** (g / (g - 1.0))
        p0_inf = (1.0 / g) * (1.0 + 0.5 * (g - 1.0) * self.mach ** 2) ** (g / (g - 1.0))
        out = p0 / p0_inf
        out[self.solid_int] = np.nan
        return out

    def schlieren(self, k: float = 12.0) -> F64:
        """Numerical schlieren: exp(-k |grad rho| / max) — shocks dark on a light background."""
        r = self.primitives()[0]
        gy, gx = np.gradient(r)
        mag = np.sqrt(gx * gx + gy * gy)
        mag[self.solid_int] = 0.0
        mx = float(np.percentile(mag[~self.solid_int], 99.8)) if (~self.solid_int).any() else 1.0
        return np.exp(-k * mag / max(mx, 1e-12))

    def shadowgraph(self) -> F64:
        """Laplacian of density (what a shadowgraph shows): bright/dark pairs at shocks."""
        r = self.primitives()[0]
        lap = (np.roll(r, 1, 0) + np.roll(r, -1, 0) + np.roll(r, 1, 1) + np.roll(r, -1, 1) - 4.0 * r)
        lap[self.solid_int] = 0.0
        return lap

    def shock_sensor(self) -> NDArray[np.bool_]:
        """Cells inside a shock: strong compression (div u < 0) with a large normalised pressure jump."""
        _, u, v, p = self.primitives()
        div = np.gradient(u, axis=1) + np.gradient(v, axis=0)
        gy, gx = np.gradient(p)
        jump = np.sqrt(gx * gx + gy * gy) / np.maximum(p, P_FLOOR)
        s = (div < -0.02 * max(self.mach, 0.5)) & (jump > 0.08)
        s[self.solid_int] = False
        return s

    # -- measurements ------------------------------------------------------------
    def probe_line(self, row: Optional[int] = None) -> dict[str, F64]:
        """Fields along a horizontal line (default: the tunnel centre line)."""
        j = self.ny // 2 if row is None else int(np.clip(row, 0, self.ny - 1))
        x = np.arange(self.nx, dtype=float)
        return {"x": x, "p": self.pressure_ratio()[j], "M": self.mach_field()[j],
                "rho": self.density_ratio()[j], "T": self.temperature_ratio()[j],
                "p0": self.total_pressure_ratio()[j]}

    def wall_faces(self) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
        """Fluid cells touching the body with the outward (fluid -> body) face normal: (j, i, nx, ny)."""
        s = self.solid_int
        js, iis, nxs, nys = [], [], [], []
        for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb = np.zeros_like(s)
            ys = slice(max(dj, 0), s.shape[0] + min(dj, 0))
            yd = slice(max(-dj, 0), s.shape[0] + min(-dj, 0))
            xs = slice(max(di, 0), s.shape[1] + min(di, 0))
            xd = slice(max(-di, 0), s.shape[1] + min(-di, 0))
            nb[yd, xd] = s[ys, xs]                  # neighbour (j+dj, i+di) is solid
            j, i = np.nonzero(nb & ~s)
            js.append(j)
            iis.append(i)
            nxs.append(np.full(j.size, float(di)))
            nys.append(np.full(j.size, float(dj)))
        return (np.concatenate(js), np.concatenate(iis), np.concatenate(nxs), np.concatenate(nys))

    def force_coefficients(self) -> tuple[float, float]:
        """Inviscid (wave + pressure) drag and lift coefficients ``(Cd, Cl)`` from the wall pressures,
        referenced to the body length and the freestream dynamic pressure."""
        if not self.solid_int.any():
            return 0.0, 0.0
        sp = self.surface_pressure()
        q = 0.5 * self.mach ** 2                                  # rho_inf = 1, u_inf = M
        if sp is not None:
            # Integrate along the true outline: force on the body = -(p - p_inf) n_out dl.
            gauge = (sp["p"] - 1.0) / self.gamma
            fx = float(-np.sum(gauge * sp["nx"] * sp["length"]))
            fy = float(-np.sum(gauge * sp["ny"] * sp["length"]))
            return fx / (q * self.ref_length), fy / (q * self.ref_length)
        j, i, nx_, ny_ = self.wall_faces()
        p = self.primitives()[3][j, i] - 1.0 / self.gamma        # gauge pressure (closed body: exact)
        fx = float(np.sum(p * nx_))
        fy = float(np.sum(p * ny_))
        return fx / (q * self.ref_length), fy / (q * self.ref_length)

    def standoff_distance(self, row: Optional[int] = None) -> Optional[float]:
        """Bow-shock stand-off (cells) ahead of the body along ``row`` (default: the stagnation row)."""
        if self.mach <= 1.0 or not self.solid_int.any():
            return None
        if row is None:                                   # the stagnation streamline: rows of the foremost point
            js, iis = np.nonzero(self.solid_int)
            row = int(np.median(js[iis == iis.min()]))
        line = np.nan_to_num(self.pressure_ratio()[row], nan=np.inf)
        body = np.where(self.solid_int[row])[0]
        if body.size == 0:
            return None
        nose = int(body.min())
        from core.gasdynamics import normal_shock
        half = 1.0 + 0.5 * (normal_shock(self.mach, self.gamma).p_ratio - 1.0)
        ahead = np.where(line[:nose] > half)[0]
        if ahead.size == 0:
            return None
        return float(nose - ahead.min())

    def stagnation_pressure(self) -> float:
        """Highest surface pressure p/p_inf (the stagnation point of a blunt body)."""
        s = self.solid_int
        if not s.any():
            return float("nan")
        sp = self.surface_pressure()
        if sp is not None:
            return float(np.max(sp["p"]))
        near = np.zeros_like(s)
        near[:, 1:] |= s[:, :-1]
        near[:, :-1] |= s[:, 1:]
        near[1:, :] |= s[:-1, :]
        near[:-1, :] |= s[1:, :]
        near &= ~s
        return float(np.nanmax(self.pressure_ratio()[near]))

    def measured_shock_angle(self, apex: tuple[float, float], length: float) -> Optional[float]:
        """Angle (deg) of the upper attached shock leaving ``apex`` (x, y in cells), fitted over
        ``length`` cells downstream: for each column the first cell above the body where the pressure
        has risen half-way to its post-shock value."""
        p = self.pressure_ratio()
        x0, y0 = apex
        xs, ys = [], []
        j0 = int(math.floor(y0))
        # Fit only the downstream half: right at a (grid-resolved, hence slightly rounded) tip the shock
        # curves, so including it biases the angle low by ~1-2°.
        for dx in np.linspace(0.5 * length, length, 12):
            i = int(round(x0 + dx))
            if not 0 <= i < self.nx or j0 >= self.ny - 4:
                continue
            vals = np.nan_to_num(p[j0:, i], nan=0.0)          # from the apex height up to the far field
            p_post = float(vals.max())
            if p_post < 1.05:
                continue
            half = 1.0 + 0.5 * (p_post - 1.0)
            above = np.where(vals > half)[0]
            if above.size == 0:
                continue
            k = int(above.max())                              # the outermost compressed cell = shock front
            if k >= len(vals) - 3:                            # the shock has left through the tunnel top
                continue
            # sub-cell position: interpolate the half-way crossing between k and k+1
            frac = 0.0
            if k + 1 < len(vals) and vals[k] > vals[k + 1]:
                frac = (vals[k] - half) / (vals[k] - vals[k + 1])
            ys.append(j0 + k + frac - y0)
            xs.append(i - x0)
        if len(xs) < 4:
            return None
        slope = float(np.polyfit(np.asarray(xs), np.asarray(ys), 1)[0])
        return math.degrees(math.atan(slope))


def warmup() -> None:
    """Compile the kernels (or load them from cache)."""
    if not accel.HAVE_NUMBA:
        return
    s = Supersonic2D(24, 16, 2.0)
    m = np.zeros((16, 24), bool)
    m[6:10, 8:12] = True
    s.set_obstacle(m, 4)
    g = float(s.gamma)
    accel.precompile(_primitives_nb, s.U, s._solid_pad, g, s.W)
    accel.precompile(_flux_x_nb, s.W, s._wall_pad, g, s._F)
    accel.precompile(_flux_y_nb, s.W, s._wall_pad, g, s._Gf)
    accel.precompile(_divergence_nb, s._F, s._Gf, s._solid_pad, s._R)
    accel.precompile(_rk_update_nb, s.U, s.U, s._F, s._Gf, s._solid_pad, 0.1, 0.5, s._U1)
    accel.precompile(_max_speed_nb, s.W, s._solid_pad, g)
    accel.precompile(_fill_ib_nb, s.W, s._ib_gj, s._ib_gi, s._ib_sj, s._ib_si, s._ib_sw, s._ib_n)
