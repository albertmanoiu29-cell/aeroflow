"""CUDA backend of the Supersonic Lab solver (:class:`core.supersonic.Supersonic2D`).

A line-by-line port of the numba kernels — HLLC/HLL hybrid fluxes, MUSCL-minmod reconstruction,
immersed-boundary ghost cells, SSP-RK2 — in double precision, so GPU and CPU results agree to round-off.

The whole state lives in GPU memory; a batch of ``n`` steps is queued without any host round trip
(the CFL time step is reduced and applied on the device), and only the primitive field ``W``, the
conserved field ``U`` and the simulated time are copied back, asynchronously, into page-locked host
buffers. :meth:`SupersonicGPU.enqueue` returns immediately, so the GUI renders the previous frame while
the GPU computes the next one; :meth:`SupersonicGPU.wait` collects the result.
"""

from __future__ import annotations

import threading

import numpy as np

from core import compute

_SRC = r"""
typedef double real;
#define G 2
#define RHO_FLOOR 1e-6
#define P_FLOOR 1e-7
#define CFL 0.5

__device__ __forceinline__ real minmod(real a, real b) {
    if (a * b <= 0.0) return 0.0;
    return fabs(a) < fabs(b) ? a : b;
}

__device__ __forceinline__ void muscl(real a0, real a1, real a2, real a3, real &l, real &r) {
    l = a1 + 0.5 * minmod(a1 - a0, a2 - a1);
    r = a2 - 0.5 * minmod(a2 - a1, a3 - a2);
}

// Normal-direction flux between left/right primitive states (u normal, v tangential):
// HLLC, switching to HLL across strong shocks (anti-carbuncle).
__device__ void riemann(real rl, real ul, real vl, real pl, real rr, real ur, real vr, real pr, real g,
                        real &f0, real &f1, real &f2, real &f3) {
    real cl = sqrt(g * pl / rl);
    real cr = sqrt(g * pr / rr);
    real el = pl / (g - 1.0) + 0.5 * rl * (ul * ul + vl * vl);
    real er = pr / (g - 1.0) + 0.5 * rr * (ur * ur + vr * vr);
    real sl = fmin(ul - cl, ur - cr);
    real sr = fmax(ul + cl, ur + cr);
    real fl0 = rl * ul, fl1 = rl * ul * ul + pl, fl2 = rl * ul * vl, fl3 = ul * (el + pl);
    real fr0 = rr * ur, fr1 = rr * ur * ur + pr, fr2 = rr * ur * vr, fr3 = ur * (er + pr);
    if (sl >= 0.0) { f0 = fl0; f1 = fl1; f2 = fl2; f3 = fl3; return; }
    if (sr <= 0.0) { f0 = fr0; f1 = fr1; f2 = fr2; f3 = fr3; return; }
    if (fmax(pl, pr) > 2.0 * fmin(pl, pr)) {                       // HLL
        real inv = 1.0 / (sr - sl);
        f0 = (sr * fl0 - sl * fr0 + sl * sr * (rr - rl)) * inv;
        f1 = (sr * fl1 - sl * fr1 + sl * sr * (rr * ur - rl * ul)) * inv;
        f2 = (sr * fl2 - sl * fr2 + sl * sr * (rr * vr - rl * vl)) * inv;
        f3 = (sr * fl3 - sl * fr3 + sl * sr * (er - el)) * inv;
        return;
    }
    real den = rl * (sl - ul) - rr * (sr - ur);
    real sm = (pr - pl + rl * ul * (sl - ul) - rr * ur * (sr - ur)) / den;
    if (sm >= 0.0) {
        real c = rl * (sl - ul) / (sl - sm);
        real u3 = c * (el / rl + (sm - ul) * (sm + pl / (rl * (sl - ul))));
        f0 = fl0 + sl * (c - rl);
        f1 = fl1 + sl * (c * sm - rl * ul);
        f2 = fl2 + sl * (c * vl - rl * vl);
        f3 = fl3 + sl * (u3 - el);
        return;
    }
    real c = rr * (sr - ur) / (sr - sm);
    real u3 = c * (er / rr + (sm - ur) * (sm + pr / (rr * (sr - ur))));
    f0 = fr0 + sr * (c - rr);
    f1 = fr1 + sr * (c * sm - rr * ur);
    f2 = fr2 + sr * (c * vr - rr * vr);
    f3 = fr3 + sr * (u3 - er);
}

#define WI(k, j, i) W[((size_t)(k) * nyp + (j)) * nxp + (i)]

// Flux through the face between padded cells L=(lj,li) and R=(rj,ri) in the (dj,di) direction.
// nc / tc: indices of the normal / tangential velocity in W. `wall`: face-reflection solids.
__device__ void face_flux(const real* __restrict__ W, const unsigned char* __restrict__ wall,
                          int nyp, int nxp, int lj, int li, int rj, int ri, int dj, int di, int nc, int tc,
                          real g, real &f0, real &f1, real &f2, real &f3) {
    bool sl_ = wall[lj * nxp + li] != 0;
    bool sr_ = wall[rj * nxp + ri] != 0;
    if (sl_ && sr_) { f0 = f1 = f2 = f3 = 0.0; return; }
    real rl = WI(0, lj, li), pl = WI(3, lj, li), nl = WI(nc, lj, li), tl = WI(tc, lj, li);
    real rr = WI(0, rj, ri), pr = WI(3, rj, ri), nr = WI(nc, rj, ri), tr = WI(tc, rj, ri);
    if (sl_) {                        // slip wall on the left: mirror the fluid state
        rl = rr; nl = -nr; tl = tr; pl = pr;
    } else if (sr_) {                 // slip wall on the right
        rr = rl; nr = -nl; tr = tl; pr = pl;
    } else {
        int llj = lj - dj, lli = li - di, rrj = rj + dj, rri = ri + di;
        if (!(wall[llj * nxp + lli] || wall[rrj * nxp + rri])) {   // second order away from walls
            real a, b, c, d;
            muscl(WI(0, llj, lli), rl, rr, WI(0, rrj, rri), a, b);
            muscl(WI(3, llj, lli), pl, pr, WI(3, rrj, rri), c, d);
            if (a > RHO_FLOOR && b > RHO_FLOOR && c > P_FLOOR && d > P_FLOOR) {
                rl = a; rr = b; pl = c; pr = d;
                real x, y;
                muscl(WI(nc, llj, lli), nl, nr, WI(nc, rrj, rri), x, y); nl = x; nr = y;
                muscl(WI(tc, llj, lli), tl, tr, WI(tc, rrj, rri), x, y); tl = x; tr = y;
            }
        }
    }
    riemann(rl, nl, tl, pl, rr, nr, tr, pr, g, f0, f1, f2, f3);
}

// Interior primitives (with positivity floors) into the padded W.
extern "C" __global__ void prim(const real* __restrict__ U, real* __restrict__ W, real g, int ny, int nx) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ny * nx) return;
    int jj = t / nx, ii = t - jj * nx;
    int nyp = ny + 2 * G, nxp = nx + 2 * G;
    size_t n = (size_t)ny * nx;
    real r = U[t];
    if (r < RHO_FLOOR) r = RHO_FLOOR;
    real u = U[n + t] / r;
    real v = U[2 * n + t] / r;
    real p = (g - 1.0) * (U[3 * n + t] - 0.5 * r * (u * u + v * v));
    if (p < P_FLOOR) p = P_FLOOR;
    WI(0, jj + G, ii + G) = r;
    WI(1, jj + G, ii + G) = u;
    WI(2, jj + G, ii + G) = v;
    WI(3, jj + G, ii + G) = p;
}

// Ghost layers: supersonic inflow (left), zero-gradient outflow (right), far field (top / bottom).
extern "C" __global__ void ghosts(real* __restrict__ W, int ny, int nx, real r0, real u0, real v0, real p0) {
    int nyp = ny + 2 * G, nxp = nx + 2 * G;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nyp * nxp) return;
    int j = t / nxp, i = t - j * nxp;
    bool gx = i < G || i >= nx + G;
    bool gy = j < G || j >= ny + G;
    if (!gx && !gy) return;
    int js = j < G ? G : (j >= ny + G ? ny + G - 1 : j);
    real fs[4] = {r0, u0, v0, p0};
    for (int k = 0; k < 4; ++k) {
        real val;
        if (i < G) val = fs[k];
        else if (i >= nx + G) val = WI(k, js, nx + G - 1);
        else val = WI(k, js, i);
        WI(k, j, i) = val;
    }
}

// Immersed-boundary ghost states: the flow at each ghost cell's mirror image across the true wall,
// keeping rho and p and reflecting the velocity about the wall normal.
extern "C" __global__ void fill_ib(real* __restrict__ W, int nyp, int nxp, int count,
                                   const long long* __restrict__ gj, const long long* __restrict__ gi,
                                   const long long* __restrict__ sj, const long long* __restrict__ si,
                                   const real* __restrict__ sw, const real* __restrict__ nrm) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    real r = 0.0, u = 0.0, v = 0.0, p = 0.0;
    for (int q = 0; q < 4; ++q) {
        real w = sw[4 * k + q];
        if (w == 0.0) continue;
        long long a = sj[4 * k + q], b = si[4 * k + q];
        r += w * WI(0, a, b);
        u += w * WI(1, a, b);
        v += w * WI(2, a, b);
        p += w * WI(3, a, b);
    }
    real nx_ = nrm[2 * k], ny_ = nrm[2 * k + 1];
    real un = u * nx_ + v * ny_;
    WI(0, gj[k], gi[k]) = r;
    WI(1, gj[k], gi[k]) = u - 2.0 * un * nx_;
    WI(2, gj[k], gi[k]) = v - 2.0 * un * ny_;
    WI(3, gj[k], gi[k]) = p;
}

// Largest (|u| + |v| + 2c) over the fluid cells: block reduction, then one atomic per block. Speeds are
// non-negative, so their IEEE bit patterns order like unsigned integers.
extern "C" __global__ void max_speed(const real* __restrict__ W, const unsigned char* __restrict__ solid,
                                     real g, int ny, int nx, unsigned long long* smax) {
    __shared__ real sh[256];
    int nyp = ny + 2 * G, nxp = nx + 2 * G;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    real s = 0.0;
    if (t < ny * nx) {
        int j = t / nx + G, i = t % nx + G;
        if (!solid[j * nxp + i]) {
            real c = sqrt(g * WI(3, j, i) / WI(0, j, i));
            s = fabs(WI(1, j, i)) + fabs(WI(2, j, i)) + 2.0 * c;
        }
    }
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) sh[threadIdx.x] = fmax(sh[threadIdx.x], sh[threadIdx.x + o]);
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(smax, (unsigned long long)__double_as_longlong(sh[0]));
}

// Local Mach number of the interior cells (0 inside the body), for display: saves the host recomputing it.
extern "C" __global__ void mach_out(const real* __restrict__ W, const unsigned char* __restrict__ solid,
                                    real g, int ny, int nx, float* __restrict__ M) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ny * nx) return;
    int nyp = ny + 2 * G, nxp = nx + 2 * G;
    int j = t / nx + G, i = t % nx + G;
    if (solid[j * nxp + i]) { M[t] = 0.0f; return; }
    real u = WI(1, j, i), v = WI(2, j, i);
    M[t] = (float)(sqrt(u * u + v * v) / sqrt(g * WI(3, j, i) / WI(0, j, i)));
}

// dt = CFL / max speed; accumulate the simulated time; clear the reduction for the next step.
extern "C" __global__ void finish_dt(unsigned long long* smax, real* dt, real* tsum) {
    real s = __longlong_as_double((long long)(*smax));
    real d = CFL / fmax(s, 1e-9);
    *dt = d;
    *tsum += d;
    *smax = 0ULL;
}

// x-face fluxes F (4, ny, nx+1); face f lies between interior cells f-1 and f.
extern "C" __global__ void flux_x(const real* __restrict__ W, const unsigned char* __restrict__ wall, real g,
                                  int ny, int nx, real* __restrict__ F) {
    int nf = nx + 1;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ny * nf) return;
    int jj = t / nf, f = t - jj * nf;
    int nyp = ny + 2 * G, nxp = nx + 2 * G;
    int j = jj + 2, li = f + 1;
    real f0, f1, f2, f3;
    face_flux(W, wall, nyp, nxp, j, li, j, li + 1, 0, 1, 1, 2, g, f0, f1, f2, f3);
    size_t plane = (size_t)ny * nf;
    F[t] = f0;
    F[plane + t] = f1;          // normal momentum = x
    F[2 * plane + t] = f2;      // tangential = y
    F[3 * plane + t] = f3;
}

// y-face fluxes G (4, ny+1, nx); face f lies between interior rows f-1 and f.
extern "C" __global__ void flux_y(const real* __restrict__ W, const unsigned char* __restrict__ wall, real g,
                                  int ny, int nx, real* __restrict__ Gf) {
    int nf = ny + 1;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nf * nx) return;
    int ff = t / nx, ii = t - ff * nx;
    int nyp = ny + 2 * G, nxp = nx + 2 * G;
    int lj = ff + 1, i = ii + 2;
    real f0, f1, f2, f3;
    face_flux(W, wall, nyp, nxp, lj, i, lj + 1, i, 1, 0, 2, 1, g, f0, f1, f2, f3);
    size_t plane = (size_t)nf * nx;
    Gf[t] = f0;
    Gf[plane + t] = f2;         // tangential = x
    Gf[2 * plane + t] = f1;     // normal momentum = y
    Gf[3 * plane + t] = f3;
}

// Fused SSP-RK stage: out = a*Ubase + (1-a)*(Ucur + dt*R). `out` may alias Ubase. Body cells copy Ubase.
extern "C" __global__ void rk_update(const real* Ubase, const real* __restrict__ Ucur,
                                     const real* __restrict__ F, const real* __restrict__ Gf,
                                     const unsigned char* __restrict__ solid, const real* __restrict__ dtp,
                                     real a, int ny, int nx, real* out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ny * nx) return;
    int jj = t / nx, ii = t - jj * nx;
    int nxp = nx + 2 * G;
    size_t n = (size_t)ny * nx;
    if (solid[(jj + 2) * nxp + ii + 2]) {
        for (int k = 0; k < 4; ++k) out[k * n + t] = Ubase[k * n + t];
        return;
    }
    real dt = *dtp;
    real b = 1.0 - a;
    size_t fp = (size_t)ny * (nx + 1), gp = (size_t)(ny + 1) * nx;
    for (int k = 0; k < 4; ++k) {
        size_t fi = k * fp + (size_t)jj * (nx + 1) + ii;
        size_t gi = k * gp + (size_t)jj * nx + ii;
        real r = F[fi] - F[fi + 1] + Gf[gi] - Gf[gi + nx];
        out[k * n + t] = a * Ubase[k * n + t] + b * (Ucur[k * n + t] + dt * r);
    }
}
"""

_BLOCK = 256
_module_cache: dict[str, object] = {}


_module_lock = threading.Lock()


def _module(key: str = "m", extra_src: str = ""):
    """The compiled CUDA module (built once; CuPy also caches the binary on disk between sessions).
    ``extra_src`` appends more kernels (the Nozzle Lab's boundary conditions) under its own ``key``."""
    cp = compute.cupy()
    with _module_lock:                     # the start-up warm-up thread may be compiling it right now
        mod = _module_cache.get(key)
        if mod is None:
            mod = cp.RawModule(code=_SRC + extra_src, options=("--std=c++14",))
            mod.get_function("prim")    # RawModule compiles lazily: force it here
            _module_cache[key] = mod
    return mod


def _pinned(shape: tuple[int, ...], dtype=np.float64) -> np.ndarray:
    """A NumPy array in page-locked host memory (required for truly asynchronous copies)."""
    cp = compute.cupy()
    n = int(np.prod(shape)) * np.dtype(dtype).itemsize
    mem = cp.cuda.alloc_pinned_memory(max(n, 1))
    return np.frombuffer(mem, dtype=dtype, count=int(np.prod(shape))).reshape(shape)


def _grid(n: int) -> tuple[int]:
    return ((n + _BLOCK - 1) // _BLOCK,)


class SupersonicGPU:
    """Device state and kernel launches for one grid size."""

    #: Kernels looked up in the compiled module (subclasses add their own).
    KERNELS: tuple[str, ...] = ("prim", "ghosts", "fill_ib", "max_speed", "finish_dt", "flux_x", "flux_y",
                                "rk_update", "mach_out")

    @staticmethod
    def _load_module():
        return _module()

    def __init__(self, nx: int, ny: int) -> None:
        cp = compute.cupy()
        self.cp = cp
        self.nx, self.ny = int(nx), int(ny)
        nyp, nxp = self.ny + 4, self.nx + 4
        mod = self._load_module()
        self.k = {name: mod.get_function(name) for name in self.KERNELS}
        self.stream = cp.cuda.Stream(non_blocking=True)
        with self.stream:
            self.U = cp.zeros((4, self.ny, self.nx), dtype=cp.float64)
            self.U1 = cp.zeros_like(self.U)
            self.W = cp.zeros((4, nyp, nxp), dtype=cp.float64)
            self.F = cp.zeros((4, self.ny, self.nx + 1), dtype=cp.float64)
            self.Gf = cp.zeros((4, self.ny + 1, self.nx), dtype=cp.float64)
            self.solid = cp.zeros((nyp, nxp), dtype=cp.uint8)
            self.wall = cp.zeros((nyp, nxp), dtype=cp.uint8)
            self.smax = cp.zeros(1, dtype=cp.uint64)
            self.dt = cp.zeros(1, dtype=cp.float64)
            self.tsum = cp.zeros(1, dtype=cp.float64)
            self.M = cp.zeros((self.ny, self.nx), dtype=cp.float32)
        self.ib_count = 0
        self.ib = None
        self.ib_version = 0                  # bumped on every upload (graphs hold the ghost-cell pointers)
        self.use_graphs = True
        self._graphs: dict[int, object] = {}
        self._graph_key: tuple = ()
        self.h_W = _pinned((4, nyp, nxp))
        self.h_M = _pinned((self.ny, self.nx), np.float32)
        self.h_t = _pinned((1,))
        self.ev_start = cp.cuda.Event()
        self.ev_end = cp.cuda.Event()
        self.pending = 0

    # -- host -> device ----------------------------------------------------
    def upload(self, U: np.ndarray, solid_pad: np.ndarray, wall_pad: np.ndarray, ib: dict) -> None:
        """Copy the full solver state (after a reset, a new body or a switch from the CPU)."""
        cp = self.cp
        with self.stream:
            self.U.set(np.ascontiguousarray(U, dtype=np.float64), stream=self.stream)
            self.solid.set(np.ascontiguousarray(solid_pad, dtype=np.uint8), stream=self.stream)
            self.wall.set(np.ascontiguousarray(wall_pad, dtype=np.uint8), stream=self.stream)
            self.ib_count = int(ib["count"])
            if self.ib_count:
                self.ib = tuple(cp.asarray(np.ascontiguousarray(ib[k])) for k in ("gj", "gi", "sj", "si", "sw", "n"))
            else:
                self.ib = None
            self.ib_version += 1
        self.stream.synchronize()            # the host arrays may be changed right after this returns

    # -- kernels -------------------------------------------------------------
    def _primitives(self, U, g: float, fs: tuple[float, float, float, float]) -> None:
        ny, nx = self.ny, self.nx
        n = ny * nx
        self.k["prim"](_grid(n), (_BLOCK,), (U, self.W, np.float64(g), np.int32(ny), np.int32(nx)))
        npad = (ny + 4) * (nx + 4)
        self.k["ghosts"](_grid(npad), (_BLOCK,), (self.W, np.int32(ny), np.int32(nx), np.float64(fs[0]),
                                                  np.float64(fs[1]), np.float64(fs[2]), np.float64(fs[3])))
        if self.ib_count:
            gj, gi, sj, si, sw, nrm = self.ib
            self.k["fill_ib"](_grid(self.ib_count), (_BLOCK,), (self.W, np.int32(ny + 4), np.int32(nx + 4),
                                                                np.int32(self.ib_count), gj, gi, sj, si, sw, nrm))

    def _fluxes(self, g: float) -> None:
        ny, nx = self.ny, self.nx
        gg, iny, inx = np.float64(g), np.int32(ny), np.int32(nx)
        self.k["flux_x"](_grid(ny * (nx + 1)), (_BLOCK,), (self.W, self.wall, gg, iny, inx, self.F))
        self.k["flux_y"](_grid((ny + 1) * nx), (_BLOCK,), (self.W, self.wall, gg, iny, inx, self.Gf))

    def _one_step(self, g: float, fs: tuple[float, float, float, float]) -> None:
        """Launch the 13 kernels of one SSP-RK2 step (no host synchronisation anywhere)."""
        ny, nx = self.ny, self.nx
        cells = ny * nx
        gg, iny, inx = np.float64(g), np.int32(ny), np.int32(nx)
        self._primitives(self.U, g, fs)
        self.k["max_speed"](_grid(cells), (_BLOCK,), (self.W, self.solid, gg, iny, inx, self.smax))
        self.k["finish_dt"]((1,), (1,), (self.smax, self.dt, self.tsum))
        # stage 1: U1 = U + dt L(U)
        self._fluxes(g)
        self.k["rk_update"](_grid(cells), (_BLOCK,), (self.U, self.U, self.F, self.Gf, self.solid, self.dt,
                                                      np.float64(0.0), iny, inx, self.U1))
        # stage 2: U = 0.5 U + 0.5 (U1 + dt L(U1))
        self._primitives(self.U1, g, fs)
        self._fluxes(g)
        self.k["rk_update"](_grid(cells), (_BLOCK,), (self.U, self.U1, self.F, self.Gf, self.solid, self.dt,
                                                      np.float64(0.5), iny, inx, self.U))

    def _graph(self, k: int, g: float, fs: tuple[float, float, float, float]):
        """A CUDA graph of ``k`` whole steps: replaying it costs one launch instead of 13 k. Graphs bake in
        their arguments, so they are rebuilt when the Mach number or the body (ghost-cell list) changes."""
        key = (float(g), tuple(float(v) for v in fs), self.ib_version)
        if key != self._graph_key:
            self._graphs.clear()
            self._graph_key = key
        graph = self._graphs.get(k)
        if graph is None:
            self.stream.begin_capture()
            try:
                for _ in range(k):
                    self._one_step(g, fs)
            finally:
                graph = self.stream.end_capture()
            self._graphs[k] = graph
        return graph

    def enqueue(self, n: int, g: float, fs: tuple[float, float, float, float], t0: float) -> None:
        """Queue ``n`` SSP-RK2 steps and the asynchronous copy-back; returns without waiting."""
        remaining = int(n)
        with self.stream:
            self.tsum.fill(t0)
            self.ev_start.record(self.stream)
            if self.use_graphs:
                try:
                    for k in (64, 8):
                        while remaining >= k:
                            self._graph(k, g, fs).launch(self.stream)
                            remaining -= k
                except Exception as exc:  # noqa: BLE001 - graphs are an optimisation only
                    import logging

                    # `remaining` counts only replays that were launched: finish with plain launches.
                    logging.getLogger("aeroflow.core.supersonic").info("CUDA graphs disabled: %s", exc)
                    self.use_graphs = False
                    self._graphs.clear()
            for _ in range(remaining):
                self._one_step(g, fs)
            self._primitives(self.U, g, fs)          # W consistent with the final U
            self.ev_end.record(self.stream)
            cells = self.ny * self.nx
            self.k["mach_out"](_grid(cells), (_BLOCK,), (self.W, self.solid, np.float64(g), np.int32(self.ny),
                                                         np.int32(self.nx), self.M))
            # U stays on the GPU (fetched by download_U only when the host needs it).
            self.W.get(stream=self.stream, out=self.h_W, blocking=False)
            self.M.get(stream=self.stream, out=self.h_M, blocking=False)
            self.tsum.get(stream=self.stream, out=self.h_t, blocking=False)
        self.pending = int(n)

    def download_U(self, out: np.ndarray) -> None:
        """Copy the conserved state into ``out`` (blocking)."""
        self.U.get(stream=self.stream, out=out)
        self.stream.synchronize()

    def wait(self) -> tuple[int, float, float]:
        """Block until the queued batch is done. Returns ``(steps, simulated time, GPU ms per step)``."""
        n = self.pending
        self.pending = 0
        self.stream.synchronize()
        ms = self.cp.cuda.get_elapsed_time(self.ev_start, self.ev_end) / max(n, 1) if n else 0.0
        return n, float(self.h_t[0]), float(ms)

    def busy(self) -> bool:
        return self.pending > 0 and not self.stream.done
