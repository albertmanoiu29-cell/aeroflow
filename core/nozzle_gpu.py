"""CUDA backend of the Rocket Nozzle Lab solver (:class:`core.nozzle.NozzleFlow`).

It reuses the Supersonic Lab kernels (HLLC/HLL fluxes, MUSCL, immersed-boundary walls, CFL reduction) and
adds two of its own: the nozzle boundary conditions (injector inlet with the ignition ramp, axis, far
field, pressure outlet) and the cut-cell (embedded-boundary) axisymmetric Runge-Kutta update. Like the CPU solver, the boundary
parameters live in a small device array rather than being passed by value, so throttling, altitude
changes, ignition and shut-down never invalidate the captured CUDA graphs. The ramp is evaluated on the
device from the simulated time.
"""

from __future__ import annotations

import numpy as np

from core.supersonic_gpu import _BLOCK, SupersonicGPU, _grid, _module, _pinned

#: Time-step factor of the nozzle solver (of the 2-D sum (|u| + c) + (|v| + c); the stability bound is 1).
NOZZLE_CFL = 0.7


def load_module(single: bool = True):
    """Compile (or fetch) the nozzle kernels in single or double precision."""
    return _module("nozzle", _NOZZLE_SRC, single=single, cfl=NOZZLE_CFL)

_NOZZLE_SRC = r"""
__device__ __forceinline__ real ramp_f(const treal* __restrict__ P, treal tt) {
    real t = (real)tt;
    real ti = (real)P[5], tau = (real)P[6], toff = (real)P[7], tauoff = (real)P[8], foff = (real)P[9];
    if (t < ti) return 0.0;
    real s = fmin((t - ti) / fmax(tau, 1e-12), 1.0);
    real f = s * s * (3.0 - 2.0 * s);
    if (t >= toff) {
        real s2 = fmin((t - toff) / fmax(tauoff, 1e-12), 1.0);
        f = foff * (1.0 - s2 * s2 * (3.0 - 2.0 * s2));
    }
    return f;
}

// Far-field ghost state (see NozzleFlow._far): free stream in flight; in still air, outflow at the ambient static
// pressure and inflow from air at rest (ambient total pressure and temperature).
__device__ __forceinline__ void far_state(real r, real u, real v, real p, real un, real pa, real ra, real uf,
                                          bool outlet, real g, real &o0, real &o1, real &o2, real &o3) {
    if (uf > 0.0 && !outlet) { o0 = ra; o1 = uf; o2 = 0.0; o3 = pa; return; }
    if (un >= 0.0) { o0 = r; o1 = u; o2 = v; o3 = pa; return; }
    real ta = g * pa / ra;
    real q2 = u * u + v * v;
    real q2c = fmin(q2, 0.9 * 2.0 * ta / (g + 1.0));
    real scale = sqrt(q2c / fmax(q2, 1e-30));
    real tt = ta - 0.5 * (g - 1.0) * q2c;
    real pin = pa * pow(tt / ta, g / (g - 1.0));
    o0 = g * pin / tt; o1 = u * scale; o2 = v * scale; o3 = pin;
}

// Ghost layers of the nozzle domain (see NozzleFlow._fill_ghosts, which this mirrors exactly).
extern "C" __global__ void ghosts_nozzle(real* __restrict__ W, int ny, int nx, const treal* __restrict__ P,
                                         const treal* __restrict__ tsum, real g) {
    int nyp = ny + 2 * G, nxp = nx + 2 * G;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nyp * nxp) return;
    int j = t / nxp, i = t - j * nxp;
    bool gx = i < G || i >= nx + G;
    bool gy = j < G || j >= ny + G;
    if (!gx && !gy) return;
    real pa = (real)P[0], ra = (real)P[1], uf = (real)P[2];
    real o0, o1, o2, o3;
    if (gx && gy) {                       // corners: read by no stencil
        o0 = ra; o1 = uf; o2 = 0.0; o3 = pa;
    } else if (i < G) {                   // injector face / far field upstream
        real r = WI(0, j, G), u = WI(1, j, G), v = WI(2, j, G), p = WI(3, j, G);
        far_state(r, u, v, p, -u, pa, ra, uf, false, g, o0, o1, o2, o3);
        if (j - G < (int)P[10] && j - G >= (int)P[12]) {   // injector rows (annular for an aerospike)
            real f = ramp_f(P, *tsum);
            real p0 = pa + f * ((real)P[3] - pa), T0 = (real)P[4];
            // non-reflecting characteristic injector (see NozzleFlow._fill_ghosts)
            real mc = (real)P[11];
            real tt = T0 / (1.0 + 0.5 * (g - 1.0) * mc * mc);
            real ut = mc * sqrt(tt);
            real pt = p0 * pow(tt / T0, g / (g - 1.0));
            real z = sqrt(g * p * r);
            real jp = pt + z * ut, jm = p - z * u;
            real pg = 0.5 * (jp + jm);
            real ug = (jp - jm) / (2.0 * z);
            if (f > 0.0 && ug > 0.0 && pg > 1e-7) {
                ug = fmin(ug, 0.98 * sqrt(2.0 * T0 / (g + 1.0)));
                real tg = T0 - 0.5 * (g - 1.0) * ug * ug;
                o0 = g * pg / tg; o1 = ug; o2 = 0.0; o3 = pg;
            } else {                      // closed injector: a wall
                o0 = r; o1 = -u; o2 = v; o3 = p;
            }
        }
    } else if (i >= nx + G) {             // outflow
        real r = WI(0, j, nx + G - 1), u = WI(1, j, nx + G - 1), v = WI(2, j, nx + G - 1), p = WI(3, j, nx + G - 1);
        if (u >= sqrt(g * p / r)) { o0 = r; o1 = u; o2 = v; o3 = p; }
        else far_state(r, u, v, p, u, pa, ra, uf, true, g, o0, o1, o2, o3);
    } else if (j >= ny + G) {             // far field above
        real r = WI(0, ny + G - 1, i), u = WI(1, ny + G - 1, i), v = WI(2, ny + G - 1, i), p = WI(3, ny + G - 1, i);
        far_state(r, u, v, p, v, pa, ra, uf, false, g, o0, o1, o2, o3);
    } else {                              // axis / symmetry plane: mirror
        int jm = 2 * G - 1 - j;
        o0 = WI(0, jm, i); o1 = WI(1, jm, i); o2 = -WI(2, jm, i); o3 = WI(3, jm, i);
    }
    WI(0, j, i) = o0;
    WI(1, j, i) = o1;
    WI(2, j, i) = o2;
    WI(3, j, i) = o3;
}

// Wall pressure: exact piston solution normal to the wall (see core.nozzle._wall_pressure).
__device__ __forceinline__ real wall_pressure(real r, real u, real v, real p, real nx_, real ny_, real g) {
    real un = u * nx_ + v * ny_;
    real c = sqrt(g * p / r);
    if (un > 0.0) {
        real k = 0.25 * (g + 1.0) * un;
        return p + r * un * (k + sqrt(k * k + c * c));
    }
    real x = 1.0 + 0.5 * (g - 1.0) * un / c;
    if (x <= 0.0) return 0.0;
    return p * pow(x, 2.0 * g / (g - 1.0));
}

// SSP-RK stage of the cut-cell scheme (see core.nozzle._rk_cut_nb, which this mirrors line by line).
extern "C" __global__ void rk_update_cut(const real* Ubase, const real* __restrict__ Ucur,
                                         const real* __restrict__ F, const real* __restrict__ Gf,
                                         const real* __restrict__ W, const unsigned char* __restrict__ solid,
                                         const real* __restrict__ ax, const real* __restrict__ ay,
                                         const real* __restrict__ wx, const real* __restrict__ wr,
                                         const real* __restrict__ nwx, const real* __restrict__ nwr,
                                         const real* __restrict__ ks, const treal* __restrict__ dtp, real a,
                                         int axi, real g, int ny, int nx, real* out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ny * nx) return;
    int jj = t / nx, ii = t - jj * nx;
    int nyp = ny + 2 * G, nxp = nx + 2 * G;
    size_t n = (size_t)ny * nx;
    if (solid[(jj + 2) * nxp + ii + 2]) {
        for (int k = 0; k < 4; ++k) out[k * n + t] = Ubase[k * n + t];
        return;
    }
    real rm = 1.0, rp = 1.0;
    if (axi) {
        real rc = jj + 0.5;
        rm = jj / rc;
        rp = (jj + 1.0) / rc;
    }
    real aL = ax[(size_t)jj * (nx + 1) + ii], aR = ax[(size_t)jj * (nx + 1) + ii + 1];
    real aB = ay[(size_t)jj * nx + ii] * rm, aT = ay[(size_t)(jj + 1) * nx + ii] * rp;
    real wxc = wx[t], wrc = wr[t];
    real pw = 0.0;
    if (wxc != 0.0 || wrc != 0.0)
        pw = wall_pressure(WI(0, jj + 2, ii + 2), WI(1, jj + 2, ii + 2), WI(2, jj + 2, ii + 2), WI(3, jj + 2, ii + 2),
                           nwx[t], nwr[t], g);
    real dt = (real)*dtp;
    real b = 1.0 - a;
    size_t fp = (size_t)ny * (nx + 1), gp = (size_t)(ny + 1) * nx;
    for (int k = 0; k < 4; ++k) {
        size_t fi = k * fp + (size_t)jj * (nx + 1) + ii;
        size_t gi = k * gp + (size_t)jj * nx + ii;
        real r = aL * F[fi] - aR * F[fi + 1] + aB * Gf[gi] - aT * Gf[gi + nx];
        if (k == 1) r += pw * wxc;
        else if (k == 2) r += ks[t] * WI(3, jj + 2, ii + 2) + pw * wrc;
        out[k * n + t] = a * Ubase[k * n + t] + b * (Ucur[k * n + t] + dt * r);
    }
}
"""


class NozzleGPU(SupersonicGPU):
    """Device state of one nozzle grid (see :class:`core.supersonic_gpu.SupersonicGPU`)."""

    KERNELS = SupersonicGPU.KERNELS + ("ghosts_nozzle", "rk_update_cut")

    def _load_module(self):
        return load_module(self.single)

    def __init__(self, nx: int, ny: int, axisymmetric: bool = True, single: bool = True) -> None:
        self.single = bool(single)
        self.dtype = np.float32 if self.single else np.float64
        super().__init__(nx, ny)
        self.axi = np.int32(1 if axisymmetric else 0)
        with self.stream:
            self.P = self.cp.zeros(16, dtype=self.cp.float64)
        self.h_P = _pinned((16,))
        self.cut = None

    def upload_cut(self, ax, ay, wx, wr, nwx, nwr, ks) -> None:
        """Copy the cut-cell geometry (apertures, wall areas and normals, radial source weights)."""
        cp = self.cp
        with self.stream:
            self.cut = tuple(cp.asarray(np.ascontiguousarray(a, dtype=self.dtype)) for a in (ax, ay, wx, wr, nwx, nwr, ks))
        self.stream.synchronize()
        self.ib_version += 1                 # captured graphs hold the old pointers

    def set_params(self, bc: np.ndarray) -> None:
        """Queue the boundary parameters for the next batch (the previous batch has been collected)."""
        np.copyto(self.h_P, np.asarray(bc, dtype=np.float64)[:16])
        self.P.set(self.h_P, stream=self.stream)

    def _primitives(self, U, g: float, fs) -> None:
        ny, nx = self.ny, self.nx
        self.k["prim"](_grid(ny * nx), (_BLOCK,), (U, self.W, self._r(g), np.int32(ny), np.int32(nx)))
        npad = (ny + 4) * (nx + 4)
        self.k["ghosts_nozzle"](_grid(npad), (_BLOCK,), (self.W, np.int32(ny), np.int32(nx), self.P, self.tsum,
                                                         self._r(g)))
        if self.ib_count:
            gj, gi, sj, si, sw, nrm = self.ib
            self.k["fill_ib"](_grid(self.ib_count), (_BLOCK,), (self.W, np.int32(ny + 4), np.int32(nx + 4),
                                                                np.int32(self.ib_count), gj, gi, sj, si, sw, nrm))

    def _rk(self, Ucur, a: float, out, g: float) -> None:
        ny, nx = self.ny, self.nx
        self.k["rk_update_cut"](_grid(ny * nx), (_BLOCK,), (self.U, Ucur, self.F, self.Gf, self.W, self.solid, *self.cut,
                                                            self.dt, self._r(a), self.axi, self._r(g),
                                                            np.int32(ny), np.int32(nx), out))

    def _one_step(self, g: float, fs) -> None:
        ny, nx = self.ny, self.nx
        cells = ny * nx
        gg, iny, inx = self._r(g), np.int32(ny), np.int32(nx)
        self._primitives(self.U, g, fs)
        self.k["max_speed"](_grid(cells), (_BLOCK,), (self.W, self.solid, gg, iny, inx, self.smax))
        self.k["finish_dt"]((1,), (1,), (self.smax, self.dt, self.tsum))
        self._fluxes(g)
        self._rk(self.U, 0.0, self.U1, g)
        self._primitives(self.U1, g, fs)
        self._fluxes(g)
        self._rk(self.U1, 0.5, self.U, g)
