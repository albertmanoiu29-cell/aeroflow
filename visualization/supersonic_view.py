"""Supersonic Lab canvas: renders the 2-D compressible Euler solution (:mod:`core.supersonic`).

Fields: Mach number, pressure, density, temperature, total-pressure loss, numerical schlieren and
shadowgraph. Overlays: the sonic line (M = 1), detected shock cells, and the exact oblique-shock
lines from theory (wedge / diamond / plate), so simulation and theory can be compared by eye.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from config.theme import PALETTE, mono_font
from core import gasdynamics as gd
from core import compute
from core.geometry_manager import GeometryModel
from core.shapes2d import (
    mesh_section, naca_outline, polygon_segments, rasterize_segments, read_airfoil_dat, transform_segments,
)
from core.supersonic import Supersonic2D


class SBody(str, Enum):
    """Bodies of the Supersonic Lab."""

    WEDGE = "Wedge (oblique shock)"
    DIAMOND = "Diamond airfoil"
    PLATE = "Flat plate"
    AIRFOIL = "NACA airfoil"
    AIRFOIL_FILE = "Airfoil from file (.dat)"
    CYLINDER = "Cylinder (bow shock)"
    CAPSULE = "Blunt capsule"
    SPHERE_CONE = "Sphere-cone heat shield (70°)"
    BICONIC = "Biconic entry body"
    INLET = "Supersonic inlet (ramp + cowl)"
    MODEL = "Section of the 3-D model"


class SField(str, Enum):
    """What the Supersonic Lab colours."""

    MACH = "Mach number"
    PRESSURE = "Pressure p/p∞"
    DENSITY = "Density ρ/ρ∞"
    TEMPERATURE = "Temperature T/T∞"
    TOTAL_PRESSURE = "Total-pressure loss"
    SCHLIEREN = "Schlieren"
    SHADOWGRAPH = "Shadowgraph"


#: (cells along x, cells along y) for each resolution preset.
SUPER_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "Fast (240 × 140)": (240, 140),
    "Balanced (320 × 190)": (320, 190),
    "Detailed (440 × 260)": (440, 260),
    "Ultra (600 × 350)": (600, 350),
    "Extreme (960 × 560, GPU)": (960, 560),
}
DEFAULT_SUPER_RESOLUTION = "Balanced (320 × 190)"
#: Bodies with a sharp nose whose attached shocks follow oblique-shock theory.
SHARP_BODIES = {SBody.WEDGE, SBody.DIAMOND, SBody.PLATE, SBody.INLET, SBody.BICONIC}
BLUNT_BODIES = {SBody.CYLINDER, SBody.CAPSULE, SBody.SPHERE_CONE}


@dataclass
class BodyGeometry:
    """A placed body: cell mask, exact outline segments, key points and reference length."""

    mask: np.ndarray
    segments: Optional[np.ndarray]
    ref_length: float
    nose: tuple[float, float]           # leading point (x, y) in cells
    upper_deflection: Optional[float]   # flow deflection of the upper / lower front surfaces (deg)
    lower_deflection: Optional[float]


def _unit_outline(body: SBody, half_angle: float, naca: str, model: Optional[GeometryModel],
                  airfoil_file: str = "") -> tuple[Optional[np.ndarray], tuple[float, float], bool]:
    """Unit-chord outline segments, pivot, and whether it is blunt (diameter-referenced)."""
    t = math.tan(math.radians(half_angle))
    if body is SBody.WEDGE:
        return polygon_segments(np.array([[0.0, 0.0], [1.0, t], [1.0, -t]])), (0.0, 0.0), False
    if body is SBody.DIAMOND:
        return polygon_segments(np.array([[0.0, 0.0], [0.5, 0.5 * t], [1.0, 0.0], [0.5, -0.5 * t]])), (0.5, 0.0), False
    if body is SBody.PLATE:
        h = 0.015
        return polygon_segments(np.array([[0.0, -h], [1.0, -h], [1.0, h], [0.0, h]])), (0.5, 0.0), False
    if body is SBody.AIRFOIL:
        return polygon_segments(naca_outline(naca)), (0.25, 0.0), False
    if body is SBody.AIRFOIL_FILE:
        if not airfoil_file:
            return None, (0.0, 0.0), False
        try:
            return polygon_segments(read_airfoil_dat(airfoil_file)[1]), (0.25, 0.0), False
        except ValueError:
            return None, (0.0, 0.0), False
    if body is SBody.CYLINDER:
        a = np.linspace(0.0, 2.0 * math.pi, 120, endpoint=False)
        return polygon_segments(np.column_stack([0.5 * np.cos(a), 0.5 * np.sin(a)])), (0.0, 0.0), True
    if body is SBody.CAPSULE:
        # spherical-cap heat shield (radius 1, half-angle ~30°) with a conical after-body
        a = np.linspace(math.radians(150.0), math.radians(210.0), 40)
        cap = np.column_stack([1.0 + np.cos(a), np.sin(a)])          # x from 0 (nose) to ~0.134
        x_s = cap[-1, 0]
        back = np.array([[x_s + 0.55, -0.22], [x_s + 0.55, 0.22]])
        pts = np.vstack([cap, back])
        return polygon_segments(pts - np.array([0.0, 0.0])), (0.0, 0.0), True
    if body is SBody.SPHERE_CONE:
        # Viking / Mars-entry aeroshell: spherical nose (radius 0.25 D) blending into a 70° cone, then a short
        # back-shell; unit diameter, the nose at the origin
        rn, cone = 0.25, math.radians(70.0)               # nose radius = half the base radius (Viking)
        a = np.linspace(math.pi, math.pi - (0.5 * math.pi - cone), 16)
        cap = np.column_stack([rn + rn * np.cos(a), rn * np.sin(a)])
        tx, ty = cap[-1]
        run = (0.5 - ty) / math.sin(cone)
        sx, sy = tx + run * math.cos(cone), 0.5
        upper = np.vstack([cap, [[sx, sy], [sx + 0.05, sy - 0.02], [sx + 0.32, 0.22], [sx + 0.36, 0.0]]])
        lower = upper[::-1] * np.array([1.0, -1.0])
        pts = np.vstack([upper, lower[1:-1]])
        return polygon_segments(pts), (0.0, 0.0), True
    if body is SBody.BICONIC:
        # sharp biconic: a fore-cone of the chosen half-angle, then a cone of half that angle, flat base
        t2 = math.tan(math.radians(0.5 * half_angle))
        x1 = 0.35
        y1 = x1 * t
        y2 = y1 + (1.0 - x1) * t2
        return polygon_segments(np.array([[0.0, 0.0], [x1, y1], [1.0, y2], [1.0, -y2], [x1, -y1]])), (0.4, 0.0), False
    if body is SBody.MODEL and model is not None:
        sec = mesh_section(model.vertices, model.faces)
        if sec is not None:
            return sec.segments, sec.pivot, False
    return None, (0.0, 0.0), False


def _inlet(nx: int, ny: int, ramp_deg: float, size: float, design_mach: float,
           ramp_shift: float = 0.0) -> BodyGeometry:
    """External-compression supersonic inlet: a ramp turns the flow by ``ramp_deg`` and a cowl lip is
    placed where the ramp shock meets it at ``design_mach`` (shock-on-lip, the design condition).
    ``ramp_shift`` (-1 .. 1) then slides the ramp (the centre-body) forward or back against the fixed cowl:
    forward, the shock falls ahead of the lip and air spills around it; back, the shock is swallowed."""
    t = math.tan(math.radians(ramp_deg))
    x_design = 0.18 * nx
    x0, y0 = x_design, 0.22 * ny                         # ramp corner (bottom wall raised clear of the edge)
    run = 0.30 * nx * size
    y_top = y0 + run * t                                 # ramp crest, then a flat duct floor
    beta = gd.oblique_shock_beta(design_mach, ramp_deg) if design_mach > 1.0 else None
    beta = beta if beta is not None else 45.0
    h_cap = 0.30 * ny * size                             # capture height above the crest
    y_lip = y_top + h_cap
    x_lip = x0 + (y_lip - y0) / math.tan(math.radians(beta))
    x_lip = max(x_lip, x0 + run + 4.0)                   # lip at or behind the ramp crest
    x0 = x_design - float(np.clip(ramp_shift, -1.0, 1.0)) * 0.10 * nx   # the centre-body moves, the cowl stays
    lip_t = max(3.0, 0.012 * ny)
    end = nx + 2.0
    ramp = np.array([[x0, y0], [x0 + run, y_top], [end, y_top], [end, y0 - 3.0], [x0, y0 - 3.0]])
    cowl = np.array([[x_lip, y_lip], [end, y_lip + 0.2 * lip_t], [end, y_lip + lip_t + 2.0],
                     [x_lip + 6.0 * lip_t, y_lip + lip_t]])
    segs = np.concatenate([polygon_segments(ramp), polygon_segments(cowl)])
    mask = rasterize_segments(segs, nx, ny)
    return BodyGeometry(mask, segs, run / max(math.cos(math.radians(ramp_deg)), 1e-9), (x0, y0), ramp_deg, None)


def build_body(body: SBody, nx: int, ny: int, size: float, aoa: float, half_angle: float,
               naca: str = "0012", model: Optional[GeometryModel] = None, airfoil_file: str = "",
               design_mach: float = 2.5, ramp_shift: float = 0.0) -> BodyGeometry:
    """Place a body in the tunnel: nose at ~22 % of the length, centred vertically."""
    if body is SBody.INLET:
        return _inlet(nx, ny, half_angle, size, design_mach, ramp_shift)
    segs_u, pivot, blunt = _unit_outline(body, half_angle, naca, model, airfoil_file)
    if segs_u is None:
        return BodyGeometry(np.zeros((ny, nx), bool), None, 1.0, (0.0, 0.0), None, None)
    if blunt:
        ref = 0.26 * ny * size                       # diameter / height
    else:
        ref = min(0.42 * nx, 1.2 * ny) * size * 0.75  # chord
    angle = 0.0 if body is SBody.CYLINDER else aoa
    origin = (0.25 * nx if not blunt else 0.35 * nx, 0.5 * ny + 0.37)
    segs = transform_segments(segs_u, pivot, ref, angle, origin)
    mask = rasterize_segments(segs, nx, ny)
    pts = segs.reshape(-1, 2)
    k = int(np.argmin(pts[:, 0]))
    nose = (float(pts[k, 0]), float(pts[k, 1]))
    up = lo = None
    if body in (SBody.WEDGE, SBody.DIAMOND, SBody.BICONIC):
        up, lo = half_angle - aoa, half_angle + aoa
    elif body is SBody.PLATE:
        up, lo = -aoa, aoa
    return BodyGeometry(mask, segs, ref, nose, up, lo)


def iso_segments(field: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Contour line ``field == level`` as line-segment pairs (vectorised marching squares).

    ``field[j, i]`` sits at (x, y) = (i, j). Returns ``(x, y)`` with consecutive points forming one
    segment each, ready for ``PlotCurveItem(connect="pairs")``. Replaces pyqtgraph's pure-Python
    ``isocurve``, which cost ~25-95 ms per frame on the Lab grids; this takes ~1-3 ms."""
    a = np.asarray(field, dtype=np.float64) - float(level)
    a = np.where(np.isfinite(a), a, -1.0)
    s = a > 0.0
    s00, s01, s10, s11 = s[:-1, :-1], s[:-1, 1:], s[1:, :-1], s[1:, 1:]
    mixed = ~((s00 == s01) & (s00 == s10) & (s00 == s11))
    cj, ci = np.nonzero(mixed)
    if cj.size == 0:
        return np.zeros(0), np.zeros(0)
    a00, a01, a10, a11 = a[cj, ci], a[cj, ci + 1], a[cj + 1, ci], a[cj + 1, ci + 1]
    b00, b01, b10, b11 = s00[cj, ci], s01[cj, ci], s10[cj, ci], s11[cj, ci]
    x0, y0 = ci.astype(np.float64), cj.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        def frac(p: np.ndarray, q: np.ndarray) -> np.ndarray:
            return np.clip(np.nan_to_num(p / (p - q), nan=0.5), 0.0, 1.0)
        # crossing points on the four cell edges: bottom, right, top, left
        X = np.stack([x0 + frac(a00, a01), x0 + 1.0, x0 + frac(a10, a11), x0], axis=1)
        Y = np.stack([y0, y0 + frac(a01, a11), y0 + 1.0, y0 + frac(a00, a10)], axis=1)
    E = np.stack([b00 != b01, b01 != b11, b10 != b11, b00 != b10], axis=1)
    n_cross = E.sum(axis=1)
    xs, ys = [], []
    two = n_cross == 2
    if two.any():
        idx = np.argsort(~E[two], axis=1, kind="stable")[:, :2]      # the two crossed edges
        rows = np.nonzero(two)[0]
        xs.append(np.stack([X[rows, idx[:, 0]], X[rows, idx[:, 1]]], axis=1))
        ys.append(np.stack([Y[rows, idx[:, 0]], Y[rows, idx[:, 1]]], axis=1))
    four = np.nonzero(n_cross == 4)[0]
    if four.size:
        # Saddle: cut off the two corners whose side differs from the cell centre.
        centre = (a00[four] + a01[four] + a10[four] + a11[four]) > 0.0
        cut_01_10 = b01[four] != centre
        pa = np.where(cut_01_10, 0, 3)            # (bottom, right) + (top, left)   or   (left, bottom) + (right, top)
        pb = np.where(cut_01_10, 1, 0)
        pc = np.where(cut_01_10, 2, 1)
        pd = np.where(cut_01_10, 3, 2)
        for p, q in ((pa, pb), (pc, pd)):
            xs.append(np.stack([X[four, p], X[four, q]], axis=1))
            ys.append(np.stack([Y[four, p], Y[four, q]], axis=1))
    if not xs:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(xs).ravel(), np.concatenate(ys).ravel()


def _lut(name: str, n: int = 256) -> np.ndarray:
    from matplotlib import colormaps

    return (np.asarray(colormaps[name](np.linspace(0.0, 1.0, n)))[:, :3] * 255).astype(np.uint8)


class SupersonicView(QWidget):
    """Canvas plus live readout of the Supersonic Lab."""

    stats_changed = Signal(object)
    fps_changed = Signal(float)
    status_message = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Root")
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(PALETTE.bg_deep)
        self.vb = self.glw.addViewBox(lockAspect=True, enableMenu=False)
        self.vb.setMouseEnabled(False, False)
        self.img = pg.ImageItem(axisOrder="row-major")
        self.vb.addItem(self.img)
        # Sonic line (M = 1): segments from a vectorised marching-squares pass, drawn as one batched path.
        self.sonic = pg.PlotCurveItem(pen=pg.mkPen((255, 255, 255, 235), width=1.6), connect="pairs",
                                      skipFiniteCheck=True)
        self.vb.addItem(self.sonic)
        self.theory_lines = [pg.PlotCurveItem(pen=pg.mkPen("#22D3EE", width=1.8, style=Qt.PenStyle.DotLine))
                             for _ in range(2)]
        for c in self.theory_lines:
            self.vb.addItem(c)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.glw)
        self.readout = QLabel(self)
        self.readout.setObjectName("GlassCard")
        self.readout.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.readout.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.readout.setFont(mono_font(12))
        self.readout.setTextFormat(Qt.TextFormat.RichText)
        self.readout.setMargin(10)
        self.legend = QLabel(self)
        self.legend.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.legend.setProperty("role", "muted")
        self.legend.setFont(mono_font(11))

        self.resolution = DEFAULT_SUPER_RESOLUTION
        self.gpu_single = True
        self.solver = Supersonic2D(*SUPER_RESOLUTIONS[self.resolution], mach=2.0)
        self.solver.gpu_single = True
        self.body = SBody.WEDGE
        self.half_angle = 10.0
        self.aoa = 0.0
        self.size = 1.0
        self.ramp_shift = 0.0
        self.naca = "0012"
        self.airfoil_file = ""
        self.field = SField.MACH
        self.cmap = "turbo"
        self.show_sonic = True
        self.show_shocks = False
        self.show_theory = True
        self.speed = 1.0
        self._model: Optional[GeometryModel] = None
        self.geometry: Optional[BodyGeometry] = None
        self._luts: dict[str, np.ndarray] = {}
        self._step_ms = 2.0
        self._t_change = 0.0                 # solver time of the last change (Mach / body / grid)
        self._fps_frames = 0
        self._fps_t0 = time.perf_counter()
        self._steps_t0 = 0
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self.rebuild_body(reset=True)

    # -- configuration -----------------------------------------------------------
    def set_model(self, model: Optional[GeometryModel]) -> None:
        self._model = model
        if self.body is SBody.MODEL:
            self.rebuild_body()

    def set_body(self, body: SBody) -> None:
        self.body = SBody(body)
        self.rebuild_body(reset=True)

    def set_param(self, *, half_angle: Optional[float] = None, aoa: Optional[float] = None,
                  size: Optional[float] = None, naca: Optional[str] = None,
                  ramp_shift: Optional[float] = None) -> None:
        if ramp_shift is not None:
            self.ramp_shift = float(ramp_shift)
        if half_angle is not None:
            self.half_angle = float(half_angle)
        if aoa is not None:
            self.aoa = float(aoa)
        if size is not None:
            self.size = float(size)
        if naca is not None:
            self.naca = naca
        self.rebuild_body()

    def set_mach(self, mach: float) -> None:
        if abs(mach - self.solver.mach) > 1e-12:
            self._t_change = self.solver.time
        self.solver.set_mach(mach)
        self._render()

    def set_resolution(self, name: str) -> None:
        if name not in SUPER_RESOLUTIONS or name == self.resolution:
            return
        self.resolution = name
        if "GPU" in name and not compute.use_gpu(wait=False):
            self.status_message.emit("The Extreme grid is meant for the GPU - on the CPU it runs very slowly.")
        m = self.solver.mach
        self.solver.finish_step()                # let a batch in flight finish before dropping the solver
        self.solver = Supersonic2D(*SUPER_RESOLUTIONS[name], mach=m)
        self.solver.gpu_single = self.gpu_single
        self.rebuild_body(reset=True)

    def set_gpu_precision(self, single: bool) -> None:
        """Single (fast) or double (exact) precision for the GPU solver; the flow carries over."""
        self.gpu_single = bool(single)
        self.solver.set_gpu_precision(self.gpu_single)

    def set_field(self, field: SField) -> None:
        self.field = SField(field)
        self._render()

    def set_colormap(self, name: str) -> None:
        self.cmap = name
        self._render()

    def set_overlays(self, sonic: bool, shocks: bool, theory: bool) -> None:
        self.show_sonic, self.show_shocks, self.show_theory = sonic, shocks, theory
        self._render()

    def set_speed(self, factor: float) -> None:
        self.speed = float(np.clip(factor, 0.25, 8.0))

    def restart(self) -> None:
        self.solver.reset()
        self._t_change = 0.0
        self._render()

    def rebuild_body(self, reset: bool = False) -> None:
        s = self.solver
        g = build_body(self.body, s.nx, s.ny, self.size, self.aoa, self.half_angle, self.naca, self._model,
                       self.airfoil_file, design_mach=max(s.mach, 1.2), ramp_shift=self.ramp_shift)
        if self.body is SBody.AIRFOIL_FILE and g.segments is None:
            self.status_message.emit("Load an airfoil coordinate file (.dat) for this body.")
        if self.body is SBody.MODEL and g.segments is None:
            self.status_message.emit("Load a 3-D model in the Wind Tunnel first (its mid-span section is used).")
        self.geometry = g
        s.set_obstacle(g.mask, g.ref_length, g.segments)
        if reset:
            s.reset()
        self._t_change = s.time
        self._render()

    # -- run control -----------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self.timer.isActive()

    def start(self) -> None:
        if not self.timer.isActive():
            self._last_tick, self._wait_ms, self._gui_ms = 0.0, 0.0, 13.0
            self._launch_ms = 0.05
            self._fps_t0 = time.perf_counter()
            self._fps_frames = 0
            self._steps_t0 = self.solver.steps
            self.timer.start()

    def stop(self) -> None:
        self.timer.stop()
        if self.solver.finish_step():            # collect the batch still in flight
            self._render()

    def step_once(self) -> None:
        self.solver.step(10)
        self._render()
        self._emit_stats(0.0)

    def _tick(self) -> None:
        # Pipelined frame: collect the batch started last frame (on the GPU, or on a CPU worker thread),
        # draw it, then start the next batch, which runs while this frame is shown. At 1x a batch is
        # ~13 ms of solver time, so the solver is busy almost all the time and the display still keeps
        # ~60 FPS; faster settings trade frame rate for simulation speed.
        t_tick = time.perf_counter()
        done = self.solver.finish_step()
        t_ready = time.perf_counter()
        if done:
            self._step_ms = 0.7 * self._step_ms + 0.3 * self.solver.last_step_ms
        self._render()
        if self._recorder is not None:
            self._recorder.capture()
        # GUI time per frame, excluding the wait for the solver: when drawing takes longer than a 60 FPS
        # frame, the next batch is sized to the actual frame so the solver never idles between frames.
        if self._last_tick > 0.0:
            gui_ms = max((t_tick - self._last_tick) * 1000.0 - self._wait_ms, 0.0)
            self._gui_ms = 0.8 * self._gui_ms + 0.2 * gui_ms
        self._last_tick = t_tick
        budget = 13.0 * self.speed * max(1.0, 0.9 * self._gui_ms / 13.0)
        n = int(np.clip(budget / max(self._step_ms, 0.005), 1, 400))
        if self.solver.device == "GPU":
            # A GPU step is 13 kernel launches (~70 µs of GUI-thread time): keep launches ≤ ~3 ms/frame.
            n = int(np.clip(budget / max(self._step_ms, 0.005), 1, max(3.0 / max(self._launch_ms, 1e-3), 10)))
        t_begin = time.perf_counter()
        self.solver.begin_step(n, background=True)
        t_end = time.perf_counter()
        if self.solver.device == "GPU":
            self._launch_ms = 0.8 * self._launch_ms + 0.2 * (t_end - t_begin) * 1000.0 / max(n, 1)
        self._wait_ms = (t_ready - t_tick + t_end - t_begin) * 1000.0
        self._fps_frames += 1
        now = time.perf_counter()
        if now - self._fps_t0 >= 1.0:
            dt = now - self._fps_t0
            self.fps_changed.emit(self._fps_frames / dt)
            sps = (self.solver.steps - self._steps_t0) / dt
            self._fps_frames, self._fps_t0, self._steps_t0 = 0, now, self.solver.steps
            self._emit_stats(sps)

    # -- measurements -------------------------------------------------------------------
    def measurements(self) -> dict:
        """Theory and simulation side by side for the current case."""
        s, g = self.solver, self.geometry
        m = s.mach
        # Flow-throughs since the last change of Mach number, body or grid: only after ~2 of them is the
        # field steady, so comparing with theory earlier would compare against a transient.
        since = max(s.time - self._t_change, 0.0)
        out: dict = {"mach": m, "time": s.time, "flow_throughs": since * m / max(s.nx, 1)}
        if g is None or g.segments is None:
            return out
        cd, cl = s.force_coefficients()
        out.update(cd=cd, cl=cl)
        if m > 1.0:
            out["normal_shock"] = gd.normal_shock(m)
        if m > 1.0 and self.body in SHARP_BODIES and g.upper_deflection is not None:
            theta = g.upper_deflection
            out["theta_upper"] = theta
            out["theta_lower"] = g.lower_deflection
            out["theta_max"] = gd.max_deflection(m)[0]
            if theta > 0:
                os_ = gd.oblique_shock(m, theta)
                out["oblique_upper"] = os_
                if os_ is not None:
                    beta_geo = s.measured_shock_angle(g.nose, 0.45 * g.ref_length) if abs(self.aoa) < 1e-9 else None
                    out["beta_geometric"] = beta_geo
                    # post-shock pressure sampled just above the front surface (half-way along it)
                    sp = s.surface_pressure()
                    if sp is not None:
                        xm = g.nose[0] + 0.25 * g.ref_length * math.cos(math.radians(self.aoa))
                        sel = (np.abs(sp["x"] - xm) < 0.08 * g.ref_length) & (sp["ny"] > 0.0)
                        if sel.any():
                            p_up = float(np.median(sp["p"][sel]))
                            out["p_upper_sim"] = p_up
                            out["beta_from_p"] = gd.shock_angle_from_pressure_ratio(m, p_up)
            elif theta < 0:
                out["expansion_upper"] = gd.expansion(m, -theta)
            if g.lower_deflection is not None and g.lower_deflection > 0:
                out["oblique_lower"] = gd.oblique_shock(m, g.lower_deflection)
            # Ackeret (linear) thin-airfoil theory for comparison
            b = math.sqrt(m * m - 1.0)
            a = math.radians(self.aoa)
            tc = math.tan(math.radians(self.half_angle)) if self.body is SBody.DIAMOND else 0.0
            if self.body in (SBody.DIAMOND, SBody.PLATE):
                out["cl_linear"] = 4.0 * a / b
                out["cd_linear"] = (4.0 * a * a + 4.0 * tc * tc) / b
        if m > 1.0 and self.body in BLUNT_BODIES:
            out["p_stag_sim"] = s.stagnation_pressure()
            so = s.standoff_distance()
            r = 0.5 * g.ref_length
            out["standoff_sim"] = None if so is None else so / r
            if self.body is SBody.CYLINDER:
                out["standoff_billig"] = gd.billig_standoff_cylinder(m)
        return out

    def _emit_stats(self, steps_per_s: float) -> None:
        meas = self.measurements()
        meas["steps_per_s"] = steps_per_s
        self.stats_changed.emit(meas)
        m = meas["mach"]
        regime = ("subsonic" if m < 0.8 else "transonic" if m < 1.2 else "supersonic" if m < 5.0 else "hypersonic")
        lines = [f"<b>M∞ {m:.2f}</b> <span style='color:{PALETTE.text_muted}'>({regime})</span> &nbsp; "
                 f"<b>Cd</b> {meas.get('cd', 0.0):.4f} &nbsp; <b>Cl</b> {meas.get('cl', 0.0):+.4f}"]
        if "oblique_upper" in meas and meas["oblique_upper"] is not None:
            th = meas["oblique_upper"]
            sim = meas.get("beta_from_p")
            sim_txt = f"{sim:.1f}°" if sim is not None else "—"
            lines.append(f"Shock angle β: theory {th.beta_deg:.1f}° · simulated {sim_txt}")
        elif "oblique_upper" in meas:
            lines.append("<span style='color:#F59E0B'>θ > θmax: the shock detaches into a bow shock</span>")
        if "p_stag_sim" in meas and "normal_shock" in meas:
            lines.append(f"Stagnation p/p∞: theory {meas['normal_shock'].pitot_ratio:.2f} · "
                         f"simulated {meas['p_stag_sim']:.2f}")
        settle = meas["flow_throughs"]
        lines.append(f"<span style='color:{PALETTE.text_muted}'>{settle:.1f} flow-throughs"
                     f"{' (settling…)' if settle < 2.0 else ' (steady)'} · {steps_per_s:,.0f} steps/s"
                     f" on {self.solver.device}</span>")
        self.readout.setText("<br>".join(lines))
        self._layout_overlays()

    # -- rendering ----------------------------------------------------------------------
    def _lut_for(self, name: str) -> np.ndarray:
        if name not in self._luts:
            self._luts[name] = _lut(name)
        return self._luts[name]

    def _render(self) -> None:
        s = self.solver
        f = self.field
        m = s.mach
        gray = False
        want_sonic = self.show_sonic and f is not SField.SCHLIEREN and f is not SField.SHADOWGRAPH
        mach = s.mach_field() if (f is SField.MACH or want_sonic) else None
        if f is SField.MACH:
            val, lo, hi = mach, 0.0, max(1.6 * m, 1.2)
            name, legend = self.cmap, f"Mach 0 – {hi:.1f}   (white line = sonic line M = 1)"
        elif f is SField.PRESSURE:
            ns = gd.normal_shock(m).p_ratio if m > 1 else 1.0 + 0.7 * m * m
            val, lo, hi = s.pressure_ratio(), 0.0, max(ns * 1.15, 1.5)
            name, legend = "inferno", f"p/p∞ 0 – {hi:.1f}"
        elif f is SField.DENSITY:
            ns = gd.normal_shock(m).rho_ratio if m > 1 else 1.5
            val, lo, hi = s.density_ratio(), 0.0, max(ns * 1.15, 1.5)
            name, legend = "viridis", f"ρ/ρ∞ 0 – {hi:.1f}"
        elif f is SField.TEMPERATURE:
            t0 = gd.total_temperature_ratio(m)
            val, lo, hi = s.temperature_ratio(), 0.5, max(t0 * 1.02, 1.2)
            name, legend = "magma", f"T/T∞ 0.5 – {hi:.2f}  (stagnation T0/T∞ = {t0:.2f})"
        elif f is SField.TOTAL_PRESSURE:
            val, lo, hi = 1.0 - s.total_pressure_ratio(), 0.0, max(1.0 - (gd.normal_shock(m).p0_ratio if m > 1 else 0.98),
                                                                  0.02)
            name, legend = "plasma", f"Total-pressure loss 1 − p0/p0∞  0 – {hi:.2f} (entropy made by shocks)"
        elif f is SField.SCHLIEREN:
            val, lo, hi, gray = s.schlieren(), 0.0, 1.0, True
            name, legend = "gray", "Numerical schlieren |∇ρ| — shocks dark, expansions grey"
        else:
            lap = s.shadowgraph()
            k = float(np.percentile(np.abs(lap[~s.solid_int]), 99.5)) if (~s.solid_int).any() else 1.0
            val, lo, hi, gray = lap, -k, k, True
            name, legend = "gray", "Shadowgraph ∇²ρ — bright/dark pairs mark shocks"
        v = np.nan_to_num(val, nan=lo)
        idx = np.clip((v - lo) / max(hi - lo, 1e-12) * 255.0, 0, 255).astype(np.uint8)
        rgb = self._lut_for(name)[idx]
        if self.show_shocks:
            sh = s.shock_sensor()
            rgb[sh] = (rgb[sh].astype(np.uint16) // 3 + np.array([170, 20, 30], np.uint16)).clip(0, 255).astype(np.uint8)
        rgb[s.solid_int] = (174, 184, 196) if not gray else (60, 66, 78)
        self.img.setImage(rgb, autoLevels=False, levels=(0, 255))
        shape = (s.nx, s.ny)
        if getattr(self, "_shown_shape", None) != shape:
            self._shown_shape = shape
            self.img.setRect(QRectF(-0.5, -0.5, s.nx, s.ny))
            self.vb.setRange(QRectF(-0.5, -0.5, s.nx, s.ny), padding=0.0)
        if want_sonic:
            xs, ys = iso_segments(mach, 1.0)
            self.sonic.setData(xs, ys, connect="pairs")
            self.sonic.show()
        else:
            self.sonic.hide()
        self._draw_theory()
        if self.legend.text() != legend:
            self.legend.setText(legend)
            self._layout_overlays()

    def _draw_theory(self) -> None:
        g = self.geometry
        for c in self.theory_lines:
            c.setData([], [])
        if not self.show_theory or g is None or self.solver.mach <= 1.0 or self.body not in SHARP_BODIES:
            return
        m = self.solver.mach
        x0, y0 = g.nose
        length = 1.15 * g.ref_length
        for k, (theta, sign) in enumerate(((g.upper_deflection, 1.0), (g.lower_deflection, -1.0))):
            if theta is None:
                continue
            if theta > 0:
                beta = gd.oblique_shock_beta(m, theta)
                if beta is None:
                    continue
            else:
                beta = gd.mach_angle_deg(m)              # weak wave: the first Mach line of the fan
            # shock direction relative to the freestream; for the upper surface the shock goes up
            a = math.radians(beta)
            xs = np.array([x0, x0 + length * math.cos(a)])
            ys = np.array([y0, y0 + sign * length * math.sin(a)])
            self.theory_lines[k].setData(xs, ys)

    def save_png(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not self.grab().save(str(out)):
            raise OSError(f"Could not write {out}")
        return out

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._layout_overlays()

    def _layout_overlays(self) -> None:
        self.readout.adjustSize()
        self.readout.move(14, 14)
        self.legend.adjustSize()
        self.legend.move(16, max(0, self.height() - self.legend.height() - 12))
        self.readout.raise_()
        self.legend.raise_()

    # -- video recording --------------------------------------------------------
    _recorder = None

    @property
    def recording(self) -> bool:
        return self._recorder is not None

    def start_recording(self, path: str) -> None:
        """Record the running simulation to MP4 (starts the simulation if paused)."""
        from visualization.recorder import FrameRecorder

        self._recorder = FrameRecorder(self, path)
        if not self.running:
            self.start()

    def stop_recording(self):
        rec, self._recorder = self._recorder, None
        return rec.close() if rec is not None else None

    def shutdown(self) -> None:
        if self._recorder is not None:
            self.stop_recording()
        self.timer.stop()
