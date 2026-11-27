"""Rocket Nozzle Lab canvas: the live nozzle flow of :mod:`core.nozzle` as a full side view.

The axisymmetric solution (one half plane) is mirrored about the axis, so the engine and its plume are
seen whole, in metres, with zoom and pan. Fields: a synthetic *exhaust glow* (what the eye sees: hot exhaust
brightest where shocks re-heat it, i.e. the shock diamonds), Mach number, pressure relative to ambient
(log scale: blue below, red above ambient), temperature, velocity, density, numerical schlieren and the
exhaust-gas fraction. Overlays: the sonic line, the detected Mach disks and shock cells, the theoretical
shock-cell spacing and separation point, streamlines, and a hover probe.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QGraphicsPolygonItem, QLabel, QVBoxLayout, QWidget

from config.theme import PALETTE, mono_font
from core import compute
from core import nozzle as nz
from visualization.supersonic_view import iso_segments

R_AIR = 287.05


class NField(str, Enum):
    """What the Nozzle Lab colours."""

    GLOW = "Exhaust glow (visual)"
    MACH = "Mach number"
    PRESSURE = "Pressure p / pₐ (log)"
    TEMPERATURE = "Temperature (K)"
    VELOCITY = "Velocity (m/s)"
    DENSITY = "Density (log)"
    SCHLIEREN = "Schlieren"
    EXHAUST = "Exhaust gas fraction"


#: (cells along the axis, cells across the half plane) for each resolution preset.
NOZZLE_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "Fast (480 × 120)": (480, 120),
    "Balanced (640 × 160)": (640, 160),
    "Detailed (880 × 220)": (880, 220),
    "Ultra (1200 × 300)": (1200, 300),
    "Extreme (1600 × 400, GPU)": (1600, 400),
}


def default_resolution() -> str:
    return "Detailed (880 × 220)" if compute.gpu_info().available else "Balanced (640 × 160)"


def _stops_lut(stops: list[tuple[float, tuple[int, int, int]]], n: int = 256) -> np.ndarray:
    xs = np.array([s[0] for s in stops])
    cs = np.array([s[1] for s in stops], dtype=float)
    t = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(t, xs, cs[:, k]) for k in range(3)], axis=1).clip(0, 255).astype(np.uint8)


#: Flame colours of the exhaust-glow view (soot-rich kerosene and solids glow orange, hydrogen and methane blue).
FLAMES: dict[str, np.ndarray] = {
    "orange": _stops_lut([(0.0, (0, 0, 0)), (0.15, (70, 8, 0)), (0.35, (190, 45, 0)), (0.6, (255, 135, 25)),
                          (0.8, (255, 210, 100)), (1.0, (255, 255, 235))]),
    "blue": _stops_lut([(0.0, (0, 0, 0)), (0.2, (25, 10, 70)), (0.45, (85, 45, 190)), (0.7, (145, 125, 255)),
                        (0.9, (215, 205, 255)), (1.0, (255, 255, 255))]),
    "cyan": _stops_lut([(0.0, (0, 0, 0)), (0.2, (8, 25, 75)), (0.45, (30, 95, 205)), (0.7, (95, 175, 255)),
                        (0.9, (205, 232, 255)), (1.0, (255, 255, 255))]),
    "white": _stops_lut([(0.0, (0, 0, 0)), (0.25, (95, 45, 15)), (0.5, (235, 140, 55)), (0.8, (255, 230, 175)),
                         (1.0, (255, 255, 255))]),
}


def flame_for(propellant_name: str) -> str:
    n = propellant_name.lower()
    if "lh2" in n or "hydrogen)" in n or "hydrazine" in n:
        return "blue"
    if "ch4" in n or "methane" in n or "cold gas" in n or "peroxide" in n:
        return "cyan"
    if "solid" in n or "apcp" in n:
        return "white"
    return "orange"


def _lut(name: str, n: int = 256) -> np.ndarray:
    from matplotlib import colormaps

    return (np.asarray(colormaps[name](np.linspace(0.0, 1.0, n)))[:, :3] * 255).astype(np.uint8)


@dataclass
class EngineSetup:
    """Everything the user can set (SI units)."""

    geom: nz.NozzleGeometry = field(default_factory=nz.NozzleGeometry)
    gas: nz.Gas = field(default_factory=nz.Gas)
    pc: float = 70e5                     # chamber pressure, Pa
    pa: float = 101325.0                 # ambient pressure, Pa
    ta: float = 288.15                   # ambient temperature, K
    flight_mach: float = 0.0
    planar: bool = False
    plume_length: float = 10.0           # exit diameters downstream
    plume_radius: float = 3.5            # exit radii across
    flame: str = "orange"


class ColorBar(QWidget):
    """A small horizontal colour bar with its range and title (drawn over the canvas)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.lut = _lut("turbo")
        self.lo, self.hi, self.title = "", "", ""
        self.setFixedSize(300, 46)

    def set(self, lut: np.ndarray, lo: str, hi: str, title: str) -> None:
        self.lut, self.lo, self.hi, self.title = lut, lo, hi, title
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QColor(8, 11, 19, 190))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(self.rect(), 6, 6)
        grad = QLinearGradient(10, 0, self.width() - 10, 0)
        for k in range(0, 256, 16):
            c = self.lut[k]
            grad.setColorAt(k / 255.0, QColor(int(c[0]), int(c[1]), int(c[2])))
        c = self.lut[-1]
        grad.setColorAt(1.0, QColor(int(c[0]), int(c[1]), int(c[2])))
        p.setBrush(QBrush(grad))
        p.drawRect(10, 22, self.width() - 20, 9)
        p.setPen(QColor(PALETTE.text_muted))
        p.setFont(mono_font(9))
        p.drawText(10, 15, self.title)
        p.drawText(10, 43, self.lo)
        fm = p.fontMetrics()
        p.drawText(self.width() - 10 - fm.horizontalAdvance(self.hi), 43, self.hi)
        p.end()


class NozzleView(QWidget):
    """Canvas, live readout and run loop of the Rocket Nozzle Lab."""

    stats_changed = Signal(object)
    fps_changed = Signal(float)
    status_message = Signal(str)
    geometry_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Root")
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(PALETTE.bg_deep)
        self.plot = self.glw.addPlot()
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=False, y=False)
        self.plot.setLabel("bottom", "x (m)")
        self.plot.setLabel("left", "r (m)")
        self.plot.setMenuEnabled(False)
        for ax in ("bottom", "left"):
            self.plot.getAxis(ax).setTextPen(pg.mkPen(PALETTE.text_muted))
            self.plot.getAxis(ax).setPen(pg.mkPen(PALETTE.border_strong))
        self.img = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.img)
        self.walls = [QGraphicsPolygonItem() for _ in range(2)]
        for w in self.walls:
            w.setBrush(QBrush(QColor(140, 150, 166)))
            w.setPen(QPen(QColor(40, 46, 58), 0))
            w.setZValue(5)
            self.plot.addItem(w)
        self.sonic = pg.PlotCurveItem(pen=pg.mkPen((255, 255, 255, 200), width=1.3), connect="pairs",
                                      skipFiniteCheck=True)
        self.sonic.setZValue(6)
        self.plot.addItem(self.sonic)
        self.stream = pg.PlotCurveItem(pen=pg.mkPen((230, 240, 255, 120), width=1.0), connect="finite")
        self.stream.setZValue(6)
        self.plot.addItem(self.stream)
        self.disk_lines = pg.PlotCurveItem(pen=pg.mkPen("#FFFFFF", width=2.4), connect="pairs")
        self.disk_lines.setZValue(7)
        self.plot.addItem(self.disk_lines)
        self.cell_marks = pg.ScatterPlotItem(size=11, symbol="t1", brush=pg.mkBrush(PALETTE.cyan),
                                             pen=pg.mkPen("#04121A", width=1))
        self.cell_marks.setZValue(8)
        self.plot.addItem(self.cell_marks)
        self.theory_marks = pg.ScatterPlotItem(size=11, symbol="o", brush=pg.mkBrush(0, 0, 0, 0),
                                               pen=pg.mkPen(PALETTE.warning, width=1.6))
        self.theory_marks.setZValue(8)
        self.plot.addItem(self.theory_marks)
        self.sep_marks = pg.ScatterPlotItem(size=12, symbol="x", pen=pg.mkPen(PALETTE.danger, width=2.4))
        self.sep_marks.setZValue(9)
        self.plot.addItem(self.sep_marks)
        self.exit_line = pg.PlotCurveItem(pen=pg.mkPen((255, 255, 255, 90), width=1, style=Qt.PenStyle.DashLine))
        self.exit_line.setZValue(6)
        self.plot.addItem(self.exit_line)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.glw)
        self.readout = QLabel(self)
        self.readout.setObjectName("GlassCard")
        self.readout.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.readout.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.readout.setFont(mono_font(11))
        self.readout.setTextFormat(Qt.TextFormat.RichText)
        self.readout.setMargin(10)
        self.hover = QLabel(self)
        self.hover.setObjectName("GlassCard")
        self.hover.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.hover.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.hover.setFont(mono_font(10))
        self.hover.setMargin(6)
        self.hover.hide()
        self.colorbar = ColorBar(self)
        self._mouse_proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=30, slot=self._on_mouse)

        self.resolution = default_resolution()
        self.setup = EngineSetup()
        self.field = NField.GLOW
        self.cmap = "turbo"
        self.show_sonic = True
        self.show_markers = True
        self.show_theory = True
        self.show_streamlines = False
        self.speed = 1.0
        self.ramp_ms = 1.0
        self.shutdown_ms = 1.0
        self.auto_ignite = True
        self.gpu_single = True
        self.wall_temperature_profile: Optional[tuple] = None   # (x, T_wall) from the heat-transfer model
        self.solver: Optional[nz.NozzleFlow] = None
        self.layout_: Optional[nz.NozzleLayout] = None
        self.scales: Optional[nz.Scales] = None
        self.theory: Optional[nz.NozzlePerformance] = None
        self.meas: dict = {}
        self.features: dict = {}
        self.avg: dict[str, float] = {}
        self.history: dict[str, list[float]] = {k: [] for k in ("t", "thrust", "pc", "mdot", "mdot_throat", "throttle")}
        self._axis_avg: Optional[dict[str, np.ndarray]] = None
        self._luts: dict[str, np.ndarray] = {}
        self._overlay_cache: dict[str, tuple] = {}
        self._stream_cache: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._frame = 0
        self._step_ms = 2.0
        self._fps_frames = 0
        self._fps_t0 = time.perf_counter()
        self._steps_t0 = 0
        self._last_tick = 0.0
        self._wait_ms = 0.0
        self._gui_ms = 13.0
        self._launch_ms = 0.05
        self._sps = 0.0
        self._recorder = None
        self._fitted = False
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self.configure(self.setup, force=True)

    # -- configuration ---------------------------------------------------------------
    def configure(self, setup: EngineSetup, force: bool = False) -> None:
        """Apply a new setup: live (throttle, ambient, flight speed) when only operating conditions change;
        otherwise rebuild the grid and restart (geometry, gas, plume region, axisymmetric / planar)."""
        old = self.setup
        self.setup = setup
        rebuild = force or self.solver is None or any((
            old.geom != setup.geom, old.planar != setup.planar, old.gas.gamma != setup.gas.gamma,
            old.gas.molar_mass != setup.gas.molar_mass, old.plume_length != setup.plume_length,
            old.plume_radius != setup.plume_radius))
        self.theory = nz.nozzle_theory(setup.geom, setup.gas, setup.pc, setup.pa, setup.planar)
        if rebuild:
            self._rebuild()
        else:
            self._apply_bc()
            self._render()
            self._emit_stats()

    def set_resolution(self, name: str) -> None:
        if name not in NOZZLE_RESOLUTIONS or name == self.resolution:
            return
        self.resolution = name
        if "GPU" in name and not compute.use_gpu(wait=False):
            self.status_message.emit("The Extreme grid is meant for the GPU - on the CPU it runs very slowly.")
        self._rebuild()

    def _rebuild(self) -> None:
        s = self.setup
        nx, ny = NOZZLE_RESOLUTIONS[self.resolution]
        was_firing = self.solver is not None and self.solver.ignited
        lay = nz.layout_nozzle(s.geom, nx, ny, s.plume_length, s.plume_radius)
        need_new = (self.solver is None or (self.solver.nx, self.solver.ny) != (nx, ny)
                    or self.solver.axi == s.planar or abs(self.solver.gamma - s.gas.gamma) > 1e-12)
        if need_new:
            if self.solver is not None:
                self.solver.finish_step()
            self.solver = nz.NozzleFlow(nx, ny, s.gas.gamma, axisymmetric=not s.planar)
            self.solver.gpu_single = self.gpu_single
        self.layout_ = lay
        self.solver.set_geometry(lay)
        rg = s.gas.gas_constant
        self.scales = nz.Scales(lay.dx, s.pa / (rg * s.ta), math.sqrt(s.gas.gamma * rg * s.ta), s.ta)
        self._apply_bc()
        self.solver.reset()
        self._clear_history()
        self._draw_walls()
        self._fitted = False
        if lay.throat_cells < 5.0:
            self.status_message.emit(
                f"The throat is only {lay.throat_cells:.1f} cells across: mass flow and thrust will be off. Use a finer "
                "grid, a shorter plume region or a smaller expansion ratio.")
        else:
            self.status_message.emit(f"Grid: {nx} × {ny} cells of {lay.dx * 1e3:.2f} mm; throat "
                                     f"{lay.throat_cells:.0f} cells across.")
        if was_firing and self.auto_ignite:
            self.ignite()
        self._render()
        self._emit_stats()
        self.geometry_changed.emit()

    def _apply_bc(self) -> None:
        s, sc, solver = self.setup, self.scales, self.solver
        if solver is None or sc is None:
            return
        rg = s.gas.gas_constant
        v_flight = s.flight_mach * math.sqrt(1.4 * R_AIR * s.ta)
        solver.set_conditions(s.pa / sc.pressure, s.pa / (rg * s.ta) / sc.rho, v_flight / sc.a, s.pc / sc.pressure,
                              s.gas.chamber_temperature / sc.temperature)

    def _clear_history(self) -> None:
        for v in self.history.values():
            v.clear()
        self._axis_avg = None
        self.avg = {}
        self.meas = {}
        self.features = {}
        self._stream_cache = None

    # -- engine and run control -----------------------------------------------------------
    @property
    def running(self) -> bool:
        return self.timer.isActive()

    def engine_state(self) -> str:
        s = self.solver
        if s is None or s.bc[nz.BC_TIGN] >= nz.NEVER:
            return "READY"
        f = s.throttle_fraction()
        if s.bc[nz.BC_TOFF] < nz.NEVER:
            return "SHUT DOWN" if f <= 1e-6 else "SHUTTING DOWN"
        return "FIRING" if f >= 0.999 else "IGNITION"

    def ignite(self) -> None:
        if self.solver is None or self.scales is None:
            return
        self.solver.ignite(self.ramp_ms * 1e-3 / self.scales.time)
        if not self.running:
            self.start()
        self._emit_stats()

    def shutdown_engine(self) -> None:
        if self.solver is None or self.scales is None:
            return
        self.solver.shutdown(self.shutdown_ms * 1e-3 / self.scales.time)
        self._emit_stats()

    def restart(self) -> None:
        """Back to the quiescent pre-ignition state (re-ignites at once when auto-ignite is on)."""
        if self.solver is None:
            return
        self.solver.reset()
        self._clear_history()
        if self.auto_ignite and self.running:
            self.ignite()
        self._render()
        self._emit_stats()

    def set_gpu_precision(self, single: bool) -> None:
        """Single (fast) or double (exact) precision for the GPU solver; the flow carries over."""
        self.gpu_single = bool(single)
        if self.solver is not None:
            self.solver.set_gpu_precision(self.gpu_single)

    def set_speed(self, factor: float) -> None:
        self.speed = float(np.clip(factor, 0.25, 8.0))

    def start(self) -> None:
        if not self.timer.isActive():
            self._last_tick, self._wait_ms, self._gui_ms = 0.0, 0.0, 13.0
            self._launch_ms = 0.05
            self._fps_t0 = time.perf_counter()
            self._fps_frames = 0
            self._steps_t0 = self.solver.steps if self.solver is not None else 0
            self.timer.start()

    def stop(self) -> None:
        self.timer.stop()
        if self.solver is not None and self.solver.finish_step():
            self._after_batch()
            self._render()

    def step_once(self) -> None:
        if self.solver is None:
            return
        self.solver.step(20)
        self._after_batch()
        self._render()
        self._emit_stats()

    def _tick(self) -> None:
        # Pipelined frame (as in the Supersonic Lab): collect the batch started last frame, draw it, then start
        # the next one so the solver runs while this frame is on screen.
        solver = self.solver
        if solver is None:
            return
        t_tick = time.perf_counter()
        done = solver.finish_step()
        t_ready = time.perf_counter()
        if done:
            self._step_ms = 0.7 * self._step_ms + 0.3 * solver.last_step_ms
            self._after_batch()
        if self.isVisible():                          # hidden behind the 3-D view: skip the 2-D drawing
            self._render()
        if self._recorder is not None:
            self._recorder.capture()
        if self._last_tick > 0.0:
            gui_ms = max((t_tick - self._last_tick) * 1000.0 - self._wait_ms, 0.0)
            self._gui_ms = 0.8 * self._gui_ms + 0.2 * gui_ms
        self._last_tick = t_tick
        budget = 13.0 * self.speed * max(1.0, 0.9 * self._gui_ms / 13.0)
        n = int(np.clip(budget / max(self._step_ms, 0.005), 1, 400))
        if solver.device == "GPU":
            n = int(np.clip(budget / max(self._step_ms, 0.005), 1, max(3.0 / max(self._launch_ms, 1e-3), 10)))
        t_begin = time.perf_counter()
        solver.begin_step(n, background=True)
        t_end = time.perf_counter()
        if solver.device == "GPU":
            self._launch_ms = 0.8 * self._launch_ms + 0.2 * (t_end - t_begin) * 1000.0 / max(n, 1)
        self._wait_ms = (t_ready - t_tick + t_end - t_begin) * 1000.0
        self._fps_frames += 1
        self._frame += 1
        now = time.perf_counter()
        if now - self._fps_t0 >= 0.25:
            dt = now - self._fps_t0
            self.fps_changed.emit(self._fps_frames / dt)
            self._sps = (solver.steps - self._steps_t0) / dt
            self._fps_frames, self._fps_t0, self._steps_t0 = 0, now, solver.steps
            self._emit_stats()

    # -- measurements -----------------------------------------------------------------------
    def _after_batch(self) -> None:
        """Measure the new state, update the time histories and the time-averaged axis profile."""
        solver, lay, sc = self.solver, self.layout_, self.scales
        if solver is None or lay is None or sc is None:
            return
        m = nz.measure(solver, lay, sc, self.setup.geom, features=False)
        self.meas = m
        h = self.history
        h["t"].append(m["time_s"] * 1e3)
        h["thrust"].append(m["thrust"])
        h["pc"].append(m["chamber_pressure"])
        h["mdot"].append(m["mdot_exit"])
        h["mdot_throat"].append(m["mdot_throat"])
        h["throttle"].append(m["throttle"])
        if len(h["t"]) > 12000:                       # keep long runs light: drop every other old point
            for k in h:
                h[k][:6000] = h[k][:6000:2]
        a = 0.08
        for key in ("thrust", "thrust_exit_plane", "base_force", "thrust_installed", "mdot_exit", "mdot_throat",
                    "exit_mach", "exit_pressure",
                    "exit_temperature", "exit_velocity", "chamber_pressure"):
            v = float(m.get(key, 0.0))
            self.avg[key] = v if key not in self.avg else (1 - a) * self.avg[key] + a * v
        # ratios from the averaged quantities (averaging the ratios themselves amplifies start-up spikes)
        g = self.setup.geom
        at = 2.0 * g.rt if self.setup.planar else math.pi * g.rt ** 2
        md, pc, f = self.avg["mdot_exit"], self.avg["chamber_pressure"], self.avg["thrust"]
        self.avg["isp"] = f / (md * nz.G0) if md > 1e-9 else 0.0
        self.avg["cf"] = f / (pc * at) if pc * at > 1e-9 else 0.0
        self.avg["cstar"] = pc * at / md if md > 1e-9 else 0.0
        prof = {"mach": m["axis_mach"], "p": m["axis_p"]}
        if self._axis_avg is None or self._axis_avg["p"].shape != prof["p"].shape:
            self._axis_avg = {k: v.copy() for k, v in prof.items()}
        else:
            for k, v in prof.items():
                self._axis_avg[k] = (1 - a) * self._axis_avg[k] + a * v
        self.features = nz.axis_features(m["axis_x"], self._axis_avg["mach"], self._axis_avg["p"], m["p_amb"],
                                         lay.x_exit, float(lay.r_inner(lay.x_exit)))

    def steady(self) -> bool:
        """Firing at full throttle, the thrust settled (< 0.5 % change over the last ~0.4 ms), and the nozzle
        flow established: the mass flow through the throat leaves through the exit, and the force on the walls
        equals the exit-plane momentum flux (both differ while a starting shock is still in the nozzle)."""
        h = self.history
        if self.engine_state() != "FIRING" or len(h["t"]) < 20:
            return False
        a = self.avg
        md, mt, f, fe = (a.get(k, 0.0) for k in ("mdot_exit", "mdot_throat", "thrust", "thrust_exit_plane"))
        if abs(md - mt) > 0.02 * max(abs(mt), 1e-12) or abs(f - fe) > 0.03 * max(abs(f), 1e-12):
            return False
        t = np.asarray(h["t"])
        f = np.asarray(h["thrust"])
        sel = t > t[-1] - 0.4
        if sel.sum() < 8:                              # coarse cells: each frame covers more time; use 8 frames
            sel = np.zeros_like(sel)
            sel[-8:] = True
        lo, hi = float(f[sel].min()), float(f[sel].max())
        return (hi - lo) < 0.005 * max(abs(float(f[sel].mean())), 1e-9)

    def summary(self) -> dict:
        """Everything the window's tables and charts show."""
        return {"meas": self.meas, "avg": self.avg, "features": self.features, "theory": self.theory,
                "setup": self.setup, "layout": self.layout_, "history": self.history, "state": self.engine_state(),
                "steady": self.steady(), "steps_per_s": self._sps,
                "device": self.solver.device if self.solver is not None else "CPU"}

    def _emit_stats(self) -> None:
        summ = self.summary()
        self.stats_changed.emit(summ)
        th, m, avg = self.theory, self.meas, self.avg
        state = summ["state"]
        colour = {"READY": PALETTE.text_muted, "IGNITION": PALETTE.warning, "FIRING": PALETTE.success,
                  "SHUTTING DOWN": PALETTE.warning, "SHUT DOWN": PALETTE.text_muted}[state]
        lines = []
        f = avg.get("thrust", 0.0)
        lines.append(f"<span style='color:{colour}'><b>{state}</b></span>"
                     + (" <span style='color:#10B981'>· steady</span>" if summ["steady"] else ""))
        if th is not None:
            lines.append(f"<b>Thrust {fmt_force(f)}</b> <span style='color:{PALETTE.text_muted}'>(theory "
                         f"{fmt_force(th.thrust)})</span> &nbsp; Isp {avg.get('isp', 0.0):.0f} s")
            lines.append(f"ṁ {avg.get('mdot_exit', 0.0):.3g} kg/s · p<sub>c</sub> {avg.get('chamber_pressure', 0.0) / 1e5:.1f} bar"
                         f" · M<sub>e</sub> {avg.get('exit_mach', 0.0):.2f}")
            feats = self.features
            nd = len(feats.get("mach_disks", []))
            ncell = feats.get("diamonds", 0)
            lines.append(f"{th.regime} · {ncell} shock cell{'s' if ncell != 1 else ''}"
                         + (f" · {nd} Mach disk{'s' if nd != 1 else ''}" if nd else ""))
        t_ms = m.get("time_s", 0.0) * 1e3
        lines.append(f"<span style='color:{PALETTE.text_muted}'>t = {t_ms:.2f} ms simulated · "
                     f"{self._sps:,.0f} steps/s on {summ['device']}</span>")
        self.readout.setText("<br>".join(lines))
        self._layout_overlays()

    # -- rendering ------------------------------------------------------------------------------
    def _lut_for(self, name: str) -> np.ndarray:
        if name in FLAMES:
            return FLAMES[name]
        if name not in self._luts:
            self._luts[name] = _lut(name)
        return self._luts[name]

    @staticmethod
    def _schlieren(r: np.ndarray, k: float = 12.0) -> np.ndarray:
        """Numerical schlieren exp(-k |grad ln rho|): scale-free (no percentile), so it is cheap and steady."""
        lr = np.log(np.maximum(r, 1e-30))
        gy, gx = np.gradient(lr)
        return np.exp(-k * np.sqrt(gx * gx + gy * gy))

    WALL_RGB = (20, 24, 32)

    def _lut_ext(self, name: str) -> np.ndarray:
        """The colour table plus one extra entry (index 256) for the walls."""
        key = "ext:" + name
        if key not in self._luts:
            self._luts[key] = np.vstack([self._lut_for(name), np.array([self.WALL_RGB], np.uint8)])
        return self._luts[key]

    def _glow_table(self, name: str) -> np.ndarray:
        """(256 flame levels × 33) colours: flame + (1 - flame) × a bluish-grey schlieren background for the 32
        schlieren levels; row 32 is the wall colour."""
        key = "glow:" + name
        if key not in self._luts:
            flame = self._lut_for(name).astype(np.float32)                       # (256, 3)
            level = np.linspace(0.0, 1.0, 256, dtype=np.float32)[:, None, None]
            sch = np.linspace(0.0, 1.0, 32, dtype=np.float32)[None, :, None]
            bg = (14.0 + 70.0 * (1.0 - sch)) * (1.0 - level) * np.array([0.75, 0.85, 1.0], np.float32)
            tab = np.clip(flame[:, None, :] + bg, 0, 255).astype(np.uint8)       # (256, 32, 3)
            wall = np.broadcast_to(np.array(self.WALL_RGB, np.uint8), (256, 1, 3))
            self._luts[key] = np.ascontiguousarray(np.concatenate([tab, wall], axis=1))
        return self._luts[key]

    def _exhaust_fraction(self, r: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Exhaust-gas fraction from the entropy, a tracer of inviscid flow: ambient air keeps the ambient
        entropy, exhaust gas the chamber's (shocks raise it a little, mixing blends the two)."""
        g = self.solver.gamma
        bc = self.solver.bc
        s = np.log(np.maximum(p, 1e-30)) - g * np.log(np.maximum(r, 1e-30))
        sa = math.log(bc[nz.BC_PA]) - g * math.log(bc[nz.BC_RHOA])
        p0, t0 = bc[nz.BC_P0], bc[nz.BC_T0]
        se = math.log(p0) - g * math.log(g * p0 / t0)
        d = se - sa
        if abs(d) < 1e-6:
            return np.zeros_like(r)
        return np.clip((s - sa) / d, 0.0, 1.0)

    def field_values(self, which: NField) -> tuple[np.ndarray, float, float, str, str, tuple[str, str]]:
        """``(values, lo, hi, lut, title, (lo label, hi label))`` of a field on the half plane."""
        solver, sc, th, s = self.solver, self.scales, self.theory, self.setup
        r, u, v, p = (a.astype(np.float32) for a in solver.primitives())     # display maths in single precision
        g = solver.gamma
        pa = solver.bc[nz.BC_PA]
        if which is NField.MACH:
            hi = max(1.2, 1.15 * max(th.exit_mach, th.jet_mach) if th is not None else 3.0)
            return (solver.mach_field().astype(np.float32), 0.0, hi, self.cmap, "Mach number (white line: M = 1)",
                    ("0", f"{hi:.1f}"))
        if which is NField.PRESSURE:
            span = 1.0
            val = np.log10(np.maximum(p, 1e-300) / pa)
            return val, -span, span, "RdBu_r", "Pressure / ambient (log scale)", ("0.1 ×", "10 ×")
        if which is NField.TEMPERATURE:
            tt = g * p / r * sc.temperature
            hi = max(s.gas.chamber_temperature * 1.02, s.ta * 1.5)
            lo = min(s.ta, th.exit_temperature if th is not None else s.ta) * 0.8
            return tt, lo, hi, "inferno", "Temperature (K)", (f"{lo:.0f} K", f"{hi:.0f} K")
        if which is NField.VELOCITY:
            vel = np.hypot(u, v) * sc.a
            hi = max(1.1 * (th.exit_velocity if th is not None else 1.0), 50.0)
            return vel, 0.0, hi, "viridis", "Velocity (m/s)", ("0", f"{hi:.0f} m/s")
        if which is NField.DENSITY:
            ra = solver.bc[nz.BC_RHOA]
            val = np.log10(np.maximum(r, 1e-300) / ra)
            hi = max(math.log10(max(solver.bc[nz.BC_P0] / max(pa, 1e-30), 1.0)) * 0.8, 1.0)
            return val, -1.5, hi, "cividis", "Density / ambient (log scale)", ("0.03 ×", f"{10 ** hi:.0f} ×")
        if which is NField.SCHLIEREN:
            return self._schlieren(r, 14.0), 0.0, 1.0, "gray", "Numerical schlieren |∇ ln ρ|", ("strong", "none")
        if which is NField.EXHAUST:
            return self._exhaust_fraction(r, p), 0.0, 1.0, "magma", "Exhaust gas fraction", ("ambient air", "exhaust")
        # exhaust glow
        tn = g * p / r * sc.temperature / max(s.gas.chamber_temperature, 1.0)
        y = self._exhaust_fraction(r, p)
        glow = y ** 0.7 * np.clip(tn, 0.0, 1.5) ** 3.0
        glow = 1.0 - np.exp(-2.4 * glow)
        return glow, 0.0, 1.0, s.flame, "Exhaust glow (hot gas, brightest behind shocks)", ("dark", "white-hot")

    def _render(self) -> None:
        solver, lay = self.solver, self.layout_
        if solver is None or lay is None:
            return
        val, lo, hi, lut_name, title, (lo_txt, hi_txt) = self.field_values(self.field)
        lut = self._lut_for(lut_name)
        scale = np.float32(255.0 / max(hi - lo, 1e-12))
        idx = np.clip((np.nan_to_num(val, nan=lo) - np.float32(lo)) * scale, 0, 255).astype(np.uint8)
        if self.field is NField.GLOW:
            # flame over a faint schlieren of the surrounding air (Mach waves, the starting shock): one gather
            # from a 2-D table (flame level × schlieren level; the last schlieren row is the wall colour)
            sch = self._schlieren(solver.primitives()[0].astype(np.float32), 10.0)
            sidx = (sch * np.float32(31.0)).astype(np.uint8)
            sidx[solver.solid_int] = 32
            rgb = self._glow_table(lut_name)[idx, sidx]
        else:
            idx16 = idx.astype(np.uint16)
            idx16[solver.solid_int] = 256
            rgb = self._lut_ext(lut_name)[idx16]
        full = np.concatenate([rgb[::-1], rgb], axis=0)
        self.img.setImage(full, autoLevels=False, levels=(0, 255))
        dx = lay.dx
        length, rad = solver.nx * dx, solver.ny * dx
        shape = (solver.nx, solver.ny, dx)
        if getattr(self, "_shown_shape", None) != shape:
            self._shown_shape = shape
            self.img.setRect(QRectF(0.0, -rad, length, 2.0 * rad))
        if not self._fitted:
            self._fitted = True
            self.plot.setRange(QRectF(0.0, -rad, length, 2.0 * rad), padding=0.0)
        self.colorbar.set(lut, lo_txt, hi_txt, title)
        self._draw_overlays(val if self.field is NField.MACH else None)

    def _draw_walls(self) -> None:
        lay = self.layout_
        if lay is None:
            return
        x = np.maximum(lay.wall_x, 0.0)
        for k, sign in enumerate((1.0, -1.0)):
            self.walls[k].setPolygon(QPolygonF([QPointF(float(a), sign * float(b)) for a, b in zip(x, lay.wall_r)]))
        re = float(lay.r_inner(lay.x_exit))
        self.exit_line.setData([lay.x_exit, lay.x_exit], [-re, re])

    def _set_if_changed(self, item, key: str, *data, **kw) -> None:
        sig = tuple(np.round(np.asarray(d, dtype=float), 6).tobytes() for d in data)
        if self._overlay_cache.get(key) != sig:
            self._overlay_cache[key] = sig
            item.setData(*data, **kw)

    def _draw_overlays(self, mach: Optional[np.ndarray]) -> None:
        solver, lay = self.solver, self.layout_
        dx = lay.dx
        if self.show_sonic and self.field not in (NField.SCHLIEREN, NField.GLOW):
            m = mach if mach is not None else solver.mach_field()
            m = np.where(solver.solid_int, np.nan, m)
            xs, ys = iso_segments(m, 1.0)                 # one half; the other is its mirror image
            xs = (xs + 0.5) * dx
            ys = (ys + 0.5) * dx
            self.sonic.setData(np.concatenate([xs, xs]), np.concatenate([ys, -ys]), connect="pairs")
            self.sonic.show()
        else:
            self.sonic.hide()
        re = float(lay.r_inner(lay.x_exit))
        feats = self.features
        if self.show_markers and feats:
            dx_ = [lay.x_exit + d for d in feats.get("mach_disks", [])]
            xs, ys = [], []
            for x in dx_:
                xs += [x, x]
                ys += [-0.45 * re, 0.45 * re]
            self._set_if_changed(self.disk_lines, "disks", np.asarray(xs, float), np.asarray(ys, float),
                                 connect="pairs")
            cx = [lay.x_exit + d for d in feats.get("diamond_x", [])]
            self._set_if_changed(self.cell_marks, "cells", cx, [0.0] * len(cx))
            self.disk_lines.show()
            self.cell_marks.show()
        else:
            self.disk_lines.hide()
            self.cell_marks.hide()
        th = self.theory
        if self.show_theory and th is not None:
            pts = []
            if th.shock_cell_spacing:
                n = int(min((solver.nx * dx - lay.x_exit) / th.shock_cell_spacing, 30))
                pts = [lay.x_exit + k * th.shock_cell_spacing for k in range(1, n + 1)]
            if th.mach_disk_distance:
                pts = [lay.x_exit + th.mach_disk_distance]
            self._set_if_changed(self.theory_marks, "theory", pts, [0.12 * re] * len(pts))
            sep = []
            if th.separation_x is not None:
                rs = float(lay.r_inner(th.separation_x))
                sep = [(th.separation_x, rs), (th.separation_x, -rs)]
            if th.shock_x is not None:
                rs = float(lay.r_inner(th.shock_x))
                sep += [(th.shock_x, rs), (th.shock_x, -rs)]
            self._set_if_changed(self.sep_marks, "sep", [a for a, _ in sep], [b for _, b in sep])
            self.theory_marks.show()
            self.sep_marks.show()
        else:
            self.theory_marks.hide()
            self.sep_marks.hide()
        if self.show_streamlines:
            if self._stream_cache is None or self._frame % 6 == 0:
                self._stream_cache = self._streamlines()
            xs, ys = self._stream_cache
            self.stream.setData(xs, ys, connect="finite")
            self.stream.show()
        else:
            self.stream.hide()

    def _streamlines(self) -> tuple[np.ndarray, np.ndarray]:
        """Streamlines from seeds in the chamber and in the surrounding air (RK2, bilinear velocity)."""
        solver, lay = self.solver, self.layout_
        _, u, v, _ = solver.primitives()
        u = np.where(solver.solid_int, 0.0, u)
        v = np.where(solver.solid_int, 0.0, v)
        ny, nx = u.shape
        rc = float(lay.contour_r[0]) / lay.dx
        re = float(lay.r_inner(lay.x_exit)) / lay.dx
        seeds = [(1.0, rc * f) for f in (0.15, 0.35, 0.55, 0.75, 0.9)]
        seeds += [(2.0, min(ny - 3.0, re * f + rc + 4.0)) for f in (0.5, 1.2, 2.0, 3.0)]
        seeds += [(min(nx - 3.0, lay.x_exit / lay.dx + k * re * 2.0), ny - 4.0) for k in (1, 3, 5)]
        pts = np.array(seeds, dtype=float)
        n_steps = 900
        path = np.full((n_steps, len(pts), 2), np.nan)
        speed_ref = max(float(np.nanpercentile(np.hypot(u, v), 99.0)), 1e-6)

        def vel(q: np.ndarray) -> np.ndarray:
            x = np.clip(q[:, 0] - 0.5, 0.0, nx - 1.001)
            y = np.clip(q[:, 1] - 0.5, 0.0, ny - 1.001)
            i0, j0 = x.astype(int), y.astype(int)
            fx, fy = x - i0, y - j0
            out = np.empty_like(q)
            for k, f in enumerate((u, v)):
                out[:, k] = ((1 - fx) * (1 - fy) * f[j0, i0] + fx * (1 - fy) * f[j0, i0 + 1]
                             + (1 - fx) * fy * f[j0 + 1, i0] + fx * fy * f[j0 + 1, i0 + 1])
            return out / speed_ref

        alive = np.ones(len(pts), bool)
        h = 0.8
        for k in range(n_steps):
            path[k, alive] = pts[alive]
            k1 = vel(pts)
            mid = pts + 0.5 * h * k1 / np.maximum(np.linalg.norm(k1, axis=1, keepdims=True), 0.05)
            k2 = vel(mid)
            step = h * k2 / np.maximum(np.linalg.norm(k2, axis=1, keepdims=True), 0.05)
            pts = pts + step
            alive &= (pts[:, 0] > 0) & (pts[:, 0] < nx - 1) & (pts[:, 1] >= 0) & (pts[:, 1] < ny - 1)
            alive &= np.linalg.norm(k2, axis=1) > 1e-3
            if not alive.any():
                break
        dx = lay.dx
        xs, ys = [], []
        for q in range(path.shape[1]):
            line = path[:, q]
            ok = np.isfinite(line[:, 0])
            if ok.sum() < 3:
                continue
            lx, ly = line[ok, 0] * dx, line[ok, 1] * dx
            for sign in (1.0, -1.0):
                xs += [lx, [np.nan]]
                ys += [sign * ly, [np.nan]]
        if not xs:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(xs), np.concatenate(ys)

    # -- hover probe ---------------------------------------------------------------------------------
    def _on_mouse(self, evt) -> None:
        solver, lay, sc = self.solver, self.layout_, self.scales
        if solver is None or lay is None or sc is None:
            return
        pos = evt[0]
        if not self.plot.sceneBoundingRect().contains(pos):
            self.hover.hide()
            return
        pt = self.plot.vb.mapSceneToView(pos)
        i = int(pt.x() / lay.dx)
        j = int(abs(pt.y()) / lay.dx)
        if not (0 <= i < solver.nx and 0 <= j < solver.ny) or solver.solid_int[j, i]:
            self.hover.hide()
            return
        r, u, v, p = solver.primitives()
        g = solver.gamma
        mach = math.hypot(u[j, i], v[j, i]) / math.sqrt(g * p[j, i] / r[j, i])
        self.hover.setText(
            f"x {pt.x():.3f} m  r {abs(pt.y()):.3f} m\n"
            f"M {mach:.2f}   p {p[j, i] * sc.pressure / 1e5:.3g} bar\n"
            f"T {g * p[j, i] / r[j, i] * sc.temperature:.0f} K   V {math.hypot(u[j, i], v[j, i]) * sc.a:.0f} m/s\n"
            f"ρ {r[j, i] * sc.rho:.3g} kg/m³")
        self.hover.adjustSize()
        local = self.mapFromGlobal(self.glw.mapToGlobal(self.glw.mapFromScene(pos)))
        self.hover.move(min(local.x() + 16, self.width() - self.hover.width() - 4),
                        min(local.y() + 16, self.height() - self.hover.height() - 4))
        self.hover.show()
        self.hover.raise_()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.hover.hide()
        super().leaveEvent(event)

    # -- display settings ------------------------------------------------------------------------------
    def set_field(self, f: NField) -> None:
        self.field = NField(f)
        self._render()

    def set_colormap(self, name: str) -> None:
        self.cmap = name
        self._render()

    def set_overlays(self, sonic: bool, markers: bool, theory: bool, streamlines: bool) -> None:
        self.show_sonic, self.show_markers, self.show_theory, self.show_streamlines = sonic, markers, theory, streamlines
        self._stream_cache = None
        self._render()

    def fit_view(self) -> None:
        self._fitted = False
        self._render()

    def zoom_nozzle(self) -> None:
        lay = self.layout_
        if lay is None:
            return
        r = float(max(np.max(lay.wall_r), 1e-6)) * 1.25
        self.plot.setRange(QRectF(0.0, -r, lay.x_exit * 1.6, 2.0 * r), padding=0.02)

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
        self.readout.move(70, 12)
        self.colorbar.move(70, max(0, self.height() - self.colorbar.height() - 46))
        self.readout.raise_()
        self.colorbar.raise_()

    # -- video recording ----------------------------------------------------------------------------------
    @property
    def recording(self) -> bool:
        return self._recorder is not None

    def start_recording(self, path: str) -> None:
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
        if self.solver is not None:
            self.solver.finish_step()


def fmt_force(f: float) -> str:
    """Thrust with a sensible unit."""
    a = abs(f)
    if a >= 1e6:
        return f"{f / 1e6:.3f} MN"
    if a >= 1e3:
        return f"{f / 1e3:.2f} kN"
    return f"{f:.1f} N"


def with_geom(setup: EngineSetup, **kw) -> EngineSetup:
    return replace(setup, geom=replace(setup.geom, **kw))
