"""GPU backend (CUDA or OpenCL, see :mod:`core.gpu_backend`) of the Live Flow Lab (:class:`core.lbm2d.FlowLab2D`): D2Q9 BGK-Smagorinsky lattice
Boltzmann with the same boundaries, sponge layers and momentum-exchange forces as the numba kernel.

One thread per lattice cell (fused pull-stream + collide), float32 like the CPU solver. The populations
stay in GPU memory; a batch of steps plus the smoke-dye advection is queued without host round trips,
and only the macroscopic fields (velocity, density, dye) and the force sum are copied back,
asynchronously, into page-locked buffers.
"""

from __future__ import annotations

import math
import threading

import numpy as np

from core import compute, gpu_backend

_SRC = r"""
CONSTANT int CX[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
CONSTANT int CY[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
CONSTANT float WT[9] = {4.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
                            1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f};
CONSTANT int OPP[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
CONSTANT int REFY[9] = {0, 1, 4, 3, 2, 8, 7, 6, 5};

#define F(a, i, y, x) a[((size_t)(i) * ny + (y)) * nx + (x)]

KERNEL lbm_step(GMEM const float* RESTRICT fs, GMEM float* RESTRICT fd,
                                    GMEM const unsigned char* RESTRICT solid,
                                    GMEM const unsigned char* RESTRICT special,
                                    float u_in, GMEM const float* RESTRICT tau0, float smag,
                                    GMEM float* RESTRICT uxa, GMEM float* RESTRICT uya, GMEM float* RESTRICT rhoa,
                                    GMEM double* force, GMEM const float* RESTRICT sponge, int ny, int nx) {
    int x = GID_X;
    int y = GID_Y;
    if (x >= nx || y >= ny) return;
    size_t c = (size_t)y * nx + x;
    float s = sponge[x];
    float tb = tau0[x];
    float ft[9];
    if (s > 0.0f) {                                   // freestream target of the absorbing layers
        for (int i = 0; i < 9; ++i) {
            float cu = 3.0f * (CX[i] * u_in);
            ft[i] = WT[i] * (1.0f + cu + 0.5f * cu * cu - 1.5f * u_in * u_in);
        }
    }
    if (!special[c]) {
        // ---- fast path: interior fluid cell, no wall or body among its neighbours ----
        float f0 = F(fs, 0, y, x), f1 = F(fs, 1, y, x - 1), f2 = F(fs, 2, y - 1, x);
        float f3 = F(fs, 3, y, x + 1), f4 = F(fs, 4, y + 1, x), f5 = F(fs, 5, y - 1, x - 1);
        float f6 = F(fs, 6, y - 1, x + 1), f7 = F(fs, 7, y + 1, x + 1), f8 = F(fs, 8, y + 1, x - 1);
        float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
        float u, v;
        bool bad = !(rho > 0.2f && rho < 5.0f);
        if (bad) {
            rho = 1.0f;
            f0 = f1 = f2 = f3 = f4 = f5 = f6 = f7 = f8 = 0.0f;
            u = u_in; v = 0.0f;
        } else {
            u = (f1 + f5 + f8 - f3 - f6 - f7) / rho;
            v = (f2 + f5 + f6 - f4 - f7 - f8) / rho;
        }
        uxa[c] = u; uya[c] = v; rhoa[c] = rho;
        float sp = sqrtf(u * u + v * v);
        if (sp > 0.35f) { u = u * 0.35f / sp; v = v * 0.35f / sp; }
        float usq = 1.5f * (u * u + v * v);
        float a = rho / 9.0f, b = rho / 36.0f, upv = u + v, umv = u - v;
        float e0 = 4.0f * a * (1.0f - usq);
        float e1 = a * (1.0f + 3.0f * u + 4.5f * u * u - usq);
        float e2 = a * (1.0f + 3.0f * v + 4.5f * v * v - usq);
        float e3 = a * (1.0f - 3.0f * u + 4.5f * u * u - usq);
        float e4 = a * (1.0f - 3.0f * v + 4.5f * v * v - usq);
        float e5 = b * (1.0f + 3.0f * upv + 4.5f * upv * upv - usq);
        float e6 = b * (1.0f - 3.0f * umv + 4.5f * umv * umv - usq);
        float e7 = b * (1.0f - 3.0f * upv + 4.5f * upv * upv - usq);
        float e8 = b * (1.0f + 3.0f * umv + 4.5f * umv * umv - usq);
        if (bad) { f0 = e0; f1 = e1; f2 = e2; f3 = e3; f4 = e4; f5 = e5; f6 = e6; f7 = e7; f8 = e8; }
        float d0 = f0 - e0, d1 = f1 - e1, d2 = f2 - e2, d3 = f3 - e3, d4 = f4 - e4;
        float d5 = f5 - e5, d6 = f6 - e6, d7 = f7 - e7, d8 = f8 - e8;
        float pxx = d1 + d3 + d5 + d6 + d7 + d8;
        float pyy = d2 + d4 + d5 + d6 + d7 + d8;
        float pxy = d5 - d6 + d7 - d8;
        float q = sqrtf(pxx * pxx + pyy * pyy + 2.0f * pxy * pxy);
        float om = 2.0f / (tb + sqrtf(tb * tb + smag * q / rho));
        float o[9] = {f0 - om * d0, f1 - om * d1, f2 - om * d2, f3 - om * d3, f4 - om * d4,
                      f5 - om * d5, f6 - om * d6, f7 - om * d7, f8 - om * d8};
        for (int i = 0; i < 9; ++i) {
            float val = o[i];
            if (s > 0.0f) val += s * (ft[i] - val);
            F(fd, i, y, x) = val;
        }
        return;
    }
    // ---- general path: walls, inlet / outlet, body and cells next to it ----
    if (solid[c]) {
        for (int i = 0; i < 9; ++i) F(fd, i, y, x) = WT[i];
        uxa[c] = 0.0f; uya[c] = 0.0f; rhoa[c] = 1.0f;
        return;
    }
    float fin[9];
    double fx = 0.0, fy = 0.0;
    for (int i = 0; i < 9; ++i) {
        int sx = x - CX[i], sy = y - CY[i], ii = i;
        if (sy < 0 || sy >= ny) { sy = y; ii = REFY[i]; }      // free-slip wall: specular reflection
        if (sx < 0) {                                          // velocity inlet (local density)
            float rl = rhoa[c];
            if (!(rl > 0.2f && rl < 5.0f)) rl = 1.0f;
            float cu = 3.0f * (CX[i] * u_in);
            fin[i] = WT[i] * rl * (1.0f + cu + 0.5f * cu * cu - 1.5f * u_in * u_in);
        } else if (sx >= nx) {                                 // pressure outlet: rho = 1, local velocity
            float uo = uxa[(size_t)y * nx + nx - 1], vo = uya[(size_t)y * nx + nx - 1];
            float cu = 3.0f * (CX[i] * uo + CY[i] * vo);
            fin[i] = WT[i] * (1.0f + cu + 0.5f * cu * cu - 1.5f * (uo * uo + vo * vo));
        } else if (solid[(size_t)sy * nx + sx]) {              // bounce-back + momentum exchange
            float fb = F(fs, OPP[i], y, x);
            fin[i] = fb;
            fx += 2.0 * fb * CX[OPP[i]];
            fy += 2.0 * fb * CY[OPP[i]];
        } else {
            fin[i] = F(fs, ii, sy, sx);
        }
    }
    float rho = 0.0f, mx = 0.0f, my = 0.0f;
    for (int i = 0; i < 9; ++i) { rho += fin[i]; mx += fin[i] * CX[i]; my += fin[i] * CY[i]; }
    if (!(rho > 0.2f && rho < 5.0f)) {                        // divergence guard (also catches NaN)
        rho = 1.0f; mx = u_in; my = 0.0f;
        for (int i = 0; i < 9; ++i) {
            float cu = 3.0f * (CX[i] * u_in);
            fin[i] = WT[i] * (1.0f + cu + 0.5f * cu * cu - 1.5f * u_in * u_in);
        }
    }
    float u = mx / rho, v = my / rho;
    float sp = sqrtf(u * u + v * v);
    float ue = u, ve = v;
    if (sp > 0.35f) { ue = u * 0.35f / sp; ve = v * 0.35f / sp; }
    float usq = 1.5f * (ue * ue + ve * ve);
    float pxx = 0.0f, pyy = 0.0f, pxy = 0.0f;
    float feq[9];
    for (int i = 0; i < 9; ++i) {
        float cu = 3.0f * (CX[i] * ue + CY[i] * ve);
        feq[i] = WT[i] * rho * (1.0f + cu + 0.5f * cu * cu - usq);
        float d = fin[i] - feq[i];
        pxx += CX[i] * CX[i] * d;
        pyy += CY[i] * CY[i] * d;
        pxy += CX[i] * CY[i] * d;
    }
    float q = sqrtf(pxx * pxx + pyy * pyy + 2.0f * pxy * pxy);
    float tau = 0.5f * (tb + sqrtf(tb * tb + smag * q / rho));
    float omega = 1.0f / tau;
    for (int i = 0; i < 9; ++i) {
        float fp = fin[i] - omega * (fin[i] - feq[i]);
        F(fd, i, y, x) = (s > 0.0f) ? fp + s * (ft[i] - fp) : fp;
    }
    uxa[c] = u; uya[c] = v; rhoa[c] = rho;
    if (fx != 0.0) ATOMIC_ADD_F64(&force[0], fx);
    if (fy != 0.0) ATOMIC_ADD_F64(&force[1], fy);
}

// Semi-Lagrangian smoke advection over `dt` steps, then the inlet rake (streaks every `spacing` rows).
KERNEL advect_dye(GMEM const float* RESTRICT dye, GMEM float* RESTRICT out,
                                      GMEM const float* RESTRICT ux, GMEM const float* RESTRICT uy,
                                      GMEM const unsigned char* RESTRICT solid, float dt, float decay,
                                      int spacing, int ny, int nx) {
    int x = GID_X;
    int y = GID_Y;
    if (x >= nx || y >= ny) return;
    size_t c = (size_t)y * nx + x;
    if (x < 3) {
        int width = (3 * spacing) / 10;
        if (width < 1) width = 1;
        out[c] = (((y + spacing / 2) % spacing) < width) ? 1.0f : 0.0f;
        return;
    }
    if (solid[c]) { out[c] = 0.0f; return; }
    float px = x - ux[c] * dt, py = y - uy[c] * dt;
    px = fminf(fmaxf(px, 0.0f), nx - 1.001f);
    py = fminf(fmaxf(py, 0.0f), ny - 1.001f);
    int i0 = (int)px, j0 = (int)py;
    float fx = px - i0, fy = py - j0;
    float v = (1.0f - fx) * (1.0f - fy) * dye[(size_t)j0 * nx + i0] + fx * (1.0f - fy) * dye[(size_t)j0 * nx + i0 + 1]
            + (1.0f - fx) * fy * dye[(size_t)(j0 + 1) * nx + i0] + fx * fy * dye[(size_t)(j0 + 1) * nx + i0 + 1];
    out[c] = v * decay;
}
"""

_BLOCK = (32, 8)
_module_cache: dict[str, object] = {}


_module_lock = threading.Lock()


def _module():
    """The compiled GPU module (built once; CuPy also caches the binary on disk between sessions)."""
    cp = compute.cupy()
    with _module_lock:                     # the start-up warm-up thread may be compiling it right now
        mod = _module_cache.get("m")
        if mod is None:
            mod = cp.RawModule(code=gpu_backend.prelude(cp) + _SRC, options=("--std=c++14",))
            mod.get_function("lbm_step")    # RawModule compiles lazily: force it here
            _module_cache["m"] = mod
    return mod


def _pinned(shape: tuple[int, ...], dtype) -> np.ndarray:
    cp = compute.cupy()
    count = int(np.prod(shape))
    mem = cp.cuda.alloc_pinned_memory(max(count * np.dtype(dtype).itemsize, 1))
    return np.frombuffer(mem, dtype=dtype, count=count).reshape(shape)


class FlowLabGPU:
    """Device state and kernel launches for one grid size."""

    def __init__(self, nx: int, ny: int) -> None:
        cp = compute.cupy()
        self.cp = cp
        self.nx, self.ny = int(nx), int(ny)
        mod = _module()
        self.k_step = mod.get_function("lbm_step")
        self.k_dye = mod.get_function("advect_dye")
        self.stream = cp.cuda.Stream(non_blocking=True)
        shape = (self.ny, self.nx)
        with self.stream:
            self.f = cp.zeros((9,) + shape, dtype=cp.float32)
            self.f2 = cp.zeros_like(self.f)
            self.solid = cp.zeros(shape, dtype=cp.uint8)
            self.special = cp.zeros(shape, dtype=cp.uint8)
            self.ux = cp.zeros(shape, dtype=cp.float32)
            self.uy = cp.zeros(shape, dtype=cp.float32)
            self.rho = cp.ones(shape, dtype=cp.float32)
            self.dye = cp.zeros(shape, dtype=cp.float32)
            self.dye2 = cp.zeros(shape, dtype=cp.float32)
            self.tau = cp.zeros(self.nx, dtype=cp.float32)
            self.sponge = cp.zeros(self.nx, dtype=cp.float32)
            self.force = cp.zeros(2, dtype=cp.float64)
        self.h_ux = _pinned(shape, np.float32)
        self.h_uy = _pinned(shape, np.float32)
        self.h_rho = _pinned(shape, np.float32)
        self.h_dye = _pinned(shape, np.float32)
        self.h_force = _pinned((2,), np.float64)
        self.ev_start = cp.cuda.Event()
        self.ev_end = cp.cuda.Event()
        self.grid = (math.ceil(self.nx / _BLOCK[0]), math.ceil(self.ny / _BLOCK[1]))
        self.use_graphs = bool(getattr(cp, "GRAPHS", True))   # CUDA graphs (none on OpenCL)
        self._graphs: dict[tuple[int, int], object] = {}
        self._graph_key: tuple = ()
        self.pending = 0
        self.dye_pending = False

    # -- host -> device -------------------------------------------------------
    def upload_state(self, f: np.ndarray, ux: np.ndarray, uy: np.ndarray, rho: np.ndarray, dye: np.ndarray) -> None:
        s = self.stream
        self.f.set(np.ascontiguousarray(f, np.float32), stream=s)
        self.ux.set(np.ascontiguousarray(ux, np.float32), stream=s)
        self.uy.set(np.ascontiguousarray(uy, np.float32), stream=s)
        self.rho.set(np.ascontiguousarray(rho, np.float32), stream=s)
        self.dye.set(np.ascontiguousarray(dye, np.float32), stream=s)
        s.synchronize()

    def upload_geometry(self, solid: np.ndarray, special: np.ndarray, tau: np.ndarray, sponge: np.ndarray) -> None:
        s = self.stream
        self.solid.set(np.ascontiguousarray(solid, np.uint8), stream=s)
        self.special.set(np.ascontiguousarray(special, np.uint8), stream=s)
        self.tau.set(np.ascontiguousarray(tau, np.float32), stream=s)
        self.sponge.set(np.ascontiguousarray(sponge, np.float32), stream=s)
        s.synchronize()

    def download_f(self) -> np.ndarray:
        """The populations (blocking) - needed only to edit the obstacle or to move to the CPU."""
        out = self.f.get(stream=self.stream)
        self.stream.synchronize()
        return out

    # -- stepping ------------------------------------------------------------------
    def _one_step(self, u_in: float, smag: float) -> None:
        self.k_step(self.grid, _BLOCK, (self.f, self.f2, self.solid, self.special, np.float32(u_in), self.tau,
                                        np.float32(smag), self.ux, self.uy, self.rho, self.force, self.sponge,
                                        np.int32(self.ny), np.int32(self.nx)))
        self.f, self.f2 = self.f2, self.f

    def _graph(self, k: int, u_in: float, smag: float):
        """A CUDA graph of ``k`` (even) steps at a constant inlet speed: one launch replays them all. An
        even count leaves ``f`` in the same buffer, so the graph stays valid for the next replay."""
        key = (float(u_in), float(smag))
        if key != self._graph_key:
            self._graphs.clear()
            self._graph_key = key
        slot = (k, int(self.f.data.ptr))          # one graph per buffer orientation
        graph = self._graphs.get(slot)
        if graph is None:
            self.stream.begin_capture()
            try:
                for _ in range(k):
                    self._one_step(u_in, smag)
            finally:
                graph = self.stream.end_capture()
            self._graphs[slot] = graph
        return graph

    def enqueue(self, inlet_speeds: list[float], smag: float, dye: bool, dye_spacing: int) -> None:
        """Queue one lattice step per inlet speed, the dye advection and the copy-back."""
        ny, nx = np.int32(self.ny), np.int32(self.nx)
        n = len(inlet_speeds)
        # Steps after the start-up ramp share one inlet speed: those are replayed from CUDA graphs.
        const = 0
        while const < n and inlet_speeds[n - 1 - const] == inlet_speeds[-1]:
            const += 1
        with self.stream:
            self.force.fill(0.0)
            self.ev_start.record(self.stream)
            for u_in in inlet_speeds[:n - const]:
                self._one_step(u_in, smag)
            remaining = const
            if self.use_graphs and const:
                try:
                    for k in (64, 8):
                        while remaining >= k:
                            self._graph(k, inlet_speeds[-1], smag).launch(self.stream)
                            remaining -= k
                except Exception as exc:  # noqa: BLE001 - graphs are an optimisation only
                    import logging

                    logging.getLogger("aeroflow.core.lbm2d").info("CUDA graphs disabled: %s", exc)
                    self.use_graphs = False
                    self._graphs.clear()
            for _ in range(remaining):
                self._one_step(inlet_speeds[-1], smag)
            self.ev_end.record(self.stream)
            if dye:
                self.k_dye(self.grid, _BLOCK, (self.dye, self.dye2, self.ux, self.uy, self.solid,
                                               np.float32(len(inlet_speeds)), np.float32(0.9995),
                                               np.int32(dye_spacing), ny, nx))
                self.dye, self.dye2 = self.dye2, self.dye
            self.ux.get(stream=self.stream, out=self.h_ux, blocking=False)
            self.uy.get(stream=self.stream, out=self.h_uy, blocking=False)
            self.rho.get(stream=self.stream, out=self.h_rho, blocking=False)
            if dye:
                self.dye.get(stream=self.stream, out=self.h_dye, blocking=False)
            self.force.get(stream=self.stream, out=self.h_force, blocking=False)
        self.pending = len(inlet_speeds)
        self.dye_pending = dye

    def wait(self) -> tuple[int, float]:
        """Block until the batch is done. Returns ``(steps, GPU ms per lattice step)``."""
        n = self.pending
        self.pending = 0
        self.stream.synchronize()
        ms = self.cp.cuda.get_elapsed_time(self.ev_start, self.ev_end) / max(n, 1) if n else 0.0
        return n, float(ms)

    def busy(self) -> bool:
        return self.pending > 0 and not self.stream.done
