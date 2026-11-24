"""Live Flow Lab viewport: renders the real-time 2-D wind tunnel (:mod:`core.lbm2d`) with pyqtgraph.

Each frame the simulation advances by an adaptive number of lattice steps (sized so stepping takes
about 9 ms of the 16 ms frame), then one RGB image is composed in NumPy:

* a colour-mapped field (speed, vorticity, pressure, or a dark background for pure smoke),
* white smoke streaks from the dye rake blended on top,
* the obstacle in a solid metallic grey,

and glowing tracer particles are drawn as a scatter layer. The left mouse button paints new walls
into the flow and the right button erases them (when drawing is enabled) - the flow reacts at once.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from config.theme import PALETTE, hex_to_rgb, mono_font
from core.geometry_manager import GeometryModel
from core.lbm2d import FlowLab2D, line_integral_convolution
from core.shapes2d import STREAMLINED, Obstacle2D, ShapeOutline, build_mask, builtin_outline, mesh_section

log = logging.getLogger("aeroflow.viz.flowlab")


class LabField(str, Enum):
    """What the Flow Lab colours the air by."""

    SPEED = "Speed"
    VORTICITY = "Vorticity"
    PRESSURE = "Pressure"
    SMOKE = "Smoke"
    TEXTURE = "Flow texture (LIC)"


#: (cells along x, cells along y) for each resolution preset.
LAB_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "Fast (300 × 120)": (300, 120),
    "Balanced (400 × 160)": (400, 160),
    "Detailed (560 × 224)": (560, 224),
    "Ultra (720 × 288)": (720, 288),
}
DEFAULT_RESOLUTION = "Balanced (400 × 160)"
LAB_COLORMAPS: tuple[str, ...] = ("turbo", "viridis", "inferno", "plasma", "cividis", "magma")


#: Diverging "ice → dark → fire" map for signed fields on the dark theme (zero = background colour).
_ICEFIRE: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.00, (200, 235, 255)), (0.22, (60, 150, 255)), (0.42, (24, 52, 120)), (0.50, (14, 20, 34)),
    (0.58, (120, 36, 30)), (0.78, (255, 96, 48)), (1.00, (255, 236, 170)),
)


def _lut(name: str, n: int = 256) -> np.ndarray:
    """(n, 3) uint8 colour table from a matplotlib colormap (or the built-in ``"icefire"``)."""
    x = np.linspace(0.0, 1.0, n)
    if name == "icefire":
        stops = np.array([s for s, _ in _ICEFIRE])
        cols = np.array([c for _, c in _ICEFIRE], dtype=float)
        return np.stack([np.interp(x, stops, cols[:, k]) for k in range(3)], axis=1).astype(np.uint8)
    from matplotlib import colormaps

    cmap = colormaps[name]
    return (np.asarray(cmap(x))[:, :3] * 255).astype(np.uint8)


class _PaintImage(pg.ImageItem):
    """Image item that reports mouse presses / drags in cell coordinates (for drawing walls)."""

    def __init__(self, owner: "FlowLabView") -> None:
        super().__init__(axisOrder="row-major")
        self._owner = owner

    def mouseClickEvent(self, ev) -> None:  # noqa: N802 (Qt API)
        if self._owner.drawing and ev.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            ev.accept()
            p = ev.pos()
            self._owner.paint_at(p.x(), p.y(), ev.button() == Qt.MouseButton.LeftButton)
        else:
            ev.ignore()

    def mouseDragEvent(self, ev) -> None:  # noqa: N802 (Qt API)
        if not self._owner.drawing or ev.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            ev.ignore()
            return
        ev.accept()
        p = ev.pos()
        self._owner.paint_at(p.x(), p.y(), ev.button() == Qt.MouseButton.LeftButton)


class FlowLabView(QWidget):
    """The Live Flow Lab canvas plus its overlay readout."""

    fps_changed = Signal(float)
    stats_changed = Signal(object)        # dict: cl, cd, cl_mean, cd_mean, st, t, re, steps_per_s
    status_message = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Root")
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(PALETTE.bg_deep)
        self.vb = self.glw.addViewBox(lockAspect=True, enableMenu=False)
        self.vb.setMouseEnabled(False, False)
        self.img = _PaintImage(self)
        self.vb.addItem(self.img)
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

        # Simulation state
        self.lab = FlowLab2D(*LAB_RESOLUTIONS[DEFAULT_RESOLUTION])
        self.resolution = DEFAULT_RESOLUTION
        self.kind = Obstacle2D.AIRFOIL
        self.naca = "2412"
        self.aoa = 8.0
        self.size = 1.0
        self.reynolds = 2000.0
        self.field = LabField.VORTICITY
        self.cmap = "turbo"
        self.show_smoke = False            # the Smoke field always shows it; as an overlay it is optional
        self.show_particles = True
        self.speed = 1.0
        self.drawing = False
        self.brush = 3.0
        self._model: Optional[GeometryModel] = None
        self._outline: Optional[ShapeOutline] = None
        self._luts: dict[str, np.ndarray] = {}
        self._solid_rgb = np.array(hex_to_rgb("#AEB8C4"), dtype=np.uint8)
        self._step_ms = 1.0
        self._fps_frames = 0
        self._fps_t0 = time.perf_counter()
        self._steps_t0 = 0
        self._stat_t = 0.0
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self._rebuild_obstacle()
        self.lab.set_reynolds(self.reynolds)
        self._render()

    # -- configuration -------------------------------------------------------
    def set_model(self, model: Optional[GeometryModel]) -> None:
        """The current 3-D geometry (used by the 'Section of the 3-D model' obstacle)."""
        self._model = model
        if self.kind is Obstacle2D.MODEL_SECTION:
            self._rebuild_obstacle()

    def set_obstacle(self, kind: Obstacle2D, naca: Optional[str] = None) -> None:
        self.kind = Obstacle2D(kind)
        if naca:
            self.naca = naca
        self._rebuild_obstacle()

    def set_aoa(self, deg: float) -> None:
        self.aoa = float(deg)
        self._rebuild_obstacle()

    def set_size(self, factor: float) -> None:
        self.size = float(factor)
        self._rebuild_obstacle()

    def set_reynolds(self, re: float) -> None:
        self.reynolds = float(re)
        self.lab.set_reynolds(self.reynolds)

    def set_resolution(self, name: str) -> None:
        if name not in LAB_RESOLUTIONS or name == self.resolution:
            return
        self.resolution = name
        self.lab.resize(*LAB_RESOLUTIONS[name])
        self._rebuild_obstacle()
        self.lab.set_reynolds(self.reynolds)
        self._render()

    def set_field(self, field: LabField) -> None:
        self.field = LabField(field)
        self._render()

    def set_colormap(self, name: str) -> None:
        self.cmap = name
        self._render()

    def set_overlays(self, smoke: bool, particles: bool) -> None:
        self.show_smoke, self.show_particles = bool(smoke), bool(particles)
        self._render()

    def set_speed(self, factor: float) -> None:
        self.speed = float(np.clip(factor, 0.1, 8.0))

    def set_drawing(self, enabled: bool, brush: Optional[float] = None) -> None:
        self.drawing = bool(enabled)
        if brush is not None:
            self.brush = float(brush)
        self.glw.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def clear_drawing(self) -> None:
        """Remove hand-drawn walls (rebuilds the chosen obstacle only)."""
        self._rebuild_obstacle(force_clear=True)
        self._render()

    def reset_flow(self) -> None:
        self.lab.reset()
        self._render()

    # -- run control ---------------------------------------------------------
    @property
    def running(self) -> bool:
        return self.timer.isActive()

    def start(self) -> None:
        if not self.timer.isActive():
            self._last_tick, self._wait_ms, self._gui_ms = 0.0, 0.0, 11.0
            self._launch_ms = 0.01
            self._fps_t0 = time.perf_counter()
            self._fps_frames = 0
            self._steps_t0 = self.lab.steps
            self.timer.start()

    def stop(self) -> None:
        self.timer.stop()
        if self.lab.finish_step():               # collect the batch still in flight
            self._render()

    def step_once(self) -> None:
        """Advance one frame's worth of steps while paused."""
        self.lab.step(20, self.show_smoke or self.field is LabField.SMOKE, self.show_particles)
        self._render()

    # -- obstacle ------------------------------------------------------------
    def _ref_cells(self) -> float:
        ny = self.lab.ny
        base = 0.42 * ny if self.kind in STREAMLINED else 0.16 * ny
        return float(base * self.size)

    def _rebuild_obstacle(self, force_clear: bool = False) -> None:
        nx, ny = self.lab.nx, self.lab.ny
        drawn = None if force_clear else getattr(self, "_drawn", None)
        if self.kind is Obstacle2D.EMPTY:
            outline = None
        elif self.kind is Obstacle2D.MODEL_SECTION:
            outline = None
            if self._model is not None:
                outline = mesh_section(self._model.vertices, self._model.faces)
            if outline is None:
                self.status_message.emit("The 3-D model has no cross-section at mid-span — showing an empty tunnel.")
        else:
            try:
                outline = builtin_outline(self.kind, self.naca)
            except ValueError as exc:
                self.status_message.emit(str(exc))
                outline = builtin_outline(self.kind, "2412") if self.kind is Obstacle2D.AIRFOIL else None
        self._outline = outline
        ref = self._ref_cells()
        streamlined = self.kind in STREAMLINED
        cx = 0.24 * nx + (0.25 * ref if streamlined else 0.0)
        center = (cx, 0.5 * ny + 0.37)        # slightly off-centre: lets vortex shedding start promptly
        mask = build_mask(outline, nx, ny, ref, self.aoa if streamlined or self.kind is Obstacle2D.SQUARE
                          or self.kind is Obstacle2D.WEDGE else 0.0, center)
        if drawn is not None and drawn.shape == mask.shape:
            mask |= drawn
        else:
            self._drawn = np.zeros_like(mask)
        self.lab.set_obstacle(mask, ref)
        if not self.running:
            self._render()

    def paint_at(self, x: float, y: float, solid: bool) -> None:
        """Paint (or erase) a wall disc at cell coordinates (x, y)."""
        if not (0 <= x < self.lab.nx and 0 <= y < self.lab.ny):
            return
        jj, ii = np.ogrid[0:self.lab.ny, 0:self.lab.nx]
        disc = (ii - x) ** 2 + (jj - y) ** 2 <= self.brush ** 2
        if not hasattr(self, "_drawn") or self._drawn.shape != disc.shape:
            self._drawn = np.zeros_like(disc)
        if solid:
            self._drawn |= disc
        else:
            self._drawn &= ~disc
        self.lab.paint(x, y, self.brush, solid)
        if not self.running:
            self._render()

    # -- frame loop ------------------------------------------------------------
    def _tick(self) -> None:
        # Pipelined frame: collect the batch started last frame (GPU, or a CPU worker thread), draw it,
        # then start the next one, which runs while this frame is on screen. ~11 ms of stepping per
        # frame at 1x; faster settings spend more solver time per frame rather than skipping physics.
        t_tick = time.perf_counter()
        done = self.lab.finish_step()
        t_ready = time.perf_counter()
        if done:
            # Smoothed cost of one lattice step alone; per-frame overheads (tracers, rendering) are
            # excluded so the budget cannot lock itself at one step per frame.
            self._step_ms = 0.7 * self._step_ms + 0.3 * self.lab.last_step_ms
        self._render()
        # GUI time per frame without the wait for the solver: slow frames get a bigger batch, so the
        # solver keeps working through the whole frame instead of idling.
        if self._last_tick > 0.0:
            gui_ms = max((t_tick - self._last_tick) * 1000.0 - self._wait_ms, 0.0)
            self._gui_ms = 0.8 * self._gui_ms + 0.2 * gui_ms
        self._last_tick = t_tick
        budget = 11.0 * self.speed * max(1.0, 0.9 * self._gui_ms / 11.0)
        n = int(np.clip(budget / max(self._step_ms, 0.005), 1, 400))
        if self.lab.device == "GPU":
            # Each GPU step is one kernel launch costing the GUI thread ~10 µs: keep launches ≤ ~3 ms/frame.
            n = int(np.clip(budget / max(self._step_ms, 0.005), 1, max(3.0 / max(self._launch_ms, 1e-4), 50)))
        t_begin = time.perf_counter()
        self.lab.begin_step(n, dye=self.show_smoke or self.field is LabField.SMOKE, particles=self.show_particles,
                            background=True)
        t_end = time.perf_counter()
        if self.lab.device == "GPU":
            self._launch_ms = 0.8 * self._launch_ms + 0.2 * (t_end - t_begin) * 1000.0 / max(n, 1)
        # Time spent waiting for or launching the solver is not drawing time.
        self._wait_ms = (t_ready - t_tick + t_end - t_begin) * 1000.0
        self._fps_frames += 1
        now = time.perf_counter()
        if now - self._fps_t0 >= 1.0:
            dt = now - self._fps_t0
            self.fps_changed.emit(self._fps_frames / dt)
            sps = (self.lab.steps - self._steps_t0) / dt
            self._fps_frames, self._fps_t0, self._steps_t0 = 0, now, self.lab.steps
            self._emit_stats(sps)

    def _emit_stats(self, steps_per_s: float) -> None:
        cl_m, cd_m = self.lab.mean_coefficients(10.0)
        st = self.lab.strouhal()
        stats = dict(cl=self.lab.cl, cd=self.lab.cd, cl_mean=cl_m, cd_mean=cd_m, st=st, t=self.lab.time,
                     re=self.lab.reynolds, steps_per_s=steps_per_s, history=self.lab.history)
        self.stats_changed.emit(stats)
        st_txt = f"{st:.3f}" if st is not None else "—"
        warn = ""
        if self.lab.tau0 <= 0.5006:
            warn = "<br><span style='color:#F59E0B'>Re above grid resolution: turbulence model active</span>"
        self.readout.setText(
            f"<b>CL</b> {cl_m:+.3f} &nbsp; <b>CD</b> {cd_m:.3f} &nbsp; <b>St</b> {st_txt}<br>"
            f"<span style='color:{PALETTE.text_muted}'>Re {self.lab.reynolds:,.0f} · t* {self.lab.time:.1f} · "
            f"{steps_per_s:,.0f} steps/s</span>{warn}")
        self._layout_overlays()

    # -- rendering ---------------------------------------------------------------
    def _lut_for(self, name: str) -> np.ndarray:
        if name not in self._luts:
            self._luts[name] = _lut(name)
        return self._luts[name]

    def _render(self) -> None:
        lab = self.lab
        f = self.field
        if f is LabField.SPEED:
            val = lab.speed()
            lo, hi = 0.0, 1.8
            lut = self._lut_for(self.cmap)
            legend = f"Speed  {lo:.1f} – {hi:.1f} × U∞   ({self.cmap})"
        elif f is LabField.VORTICITY:
            val = lab.vorticity()
            m = 6.0
            lo, hi = -m, m
            lut = self._lut_for("icefire")
            legend = "Vorticity  blue = clockwise · red = counter-clockwise"
        elif f is LabField.PRESSURE:
            val = lab.pressure_cp()
            lo, hi = -1.5, 1.5
            lut = self._lut_for("icefire")
            legend = "Pressure Cp  blue = suction · red = high pressure"
        elif f is LabField.TEXTURE:
            val = lab.speed()
            lo, hi = 0.0, 1.8
            lut = self._lut_for(self.cmap)
            legend = f"Flow texture: every streamline at once, tinted by speed ({self.cmap})"
        else:
            val = None
            lo, hi, lut = 0.0, 1.0, None
            legend = "Smoke streaks from the inlet rake"
        if val is not None:
            idx = np.clip((val - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
            rgb = lut[idx]
            if f is LabField.TEXTURE:
                rgb = self._apply_lic(rgb)
        else:
            rgb = np.empty((lab.ny, lab.nx, 3), dtype=np.uint8)
            rgb[:] = (14, 20, 34)
        if self.show_smoke or f is LabField.SMOKE:
            # integer alpha blend toward near-white smoke (much cheaper than float maths per frame)
            strength = 242 if f is LabField.SMOKE else 150
            a = (np.clip(lab.dye, 0.0, 1.0) * strength).astype(np.uint16)[..., None]
            rgb = ((rgb.astype(np.uint16) * (256 - a) + 245 * a) >> 8).astype(np.uint8)
        if self.show_particles:
            # tracer particles stamped straight into the image: crisp dots, no scatter-layer cost
            ix = np.clip(lab.px.astype(np.int64), 0, lab.nx - 1)
            iy = np.clip(lab.py.astype(np.int64), 0, lab.ny - 1)
            rgb[iy, ix] = ((rgb[iy, ix].astype(np.uint16) * 3 + 2 * 235) // 5).astype(np.uint8)
        rgb[lab.solid] = self._solid_rgb
        self.img.setImage(rgb, autoLevels=False, levels=(0, 255))
        shape = (lab.nx, lab.ny)
        if getattr(self, "_shown_shape", None) != shape:      # geometry of the view changes only on resize
            self._shown_shape = shape
            self.img.setRect(QRectF(-0.5, -0.5, lab.nx, lab.ny))
            self.vb.setRange(QRectF(-0.5, -0.5, lab.nx, lab.ny), padding=0.0)
        if self.legend.text() != legend:
            self.legend.setText(legend)
            self._layout_overlays()

    def _apply_lic(self, rgb: np.ndarray) -> np.ndarray:
        """Modulate the speed colours by a line-integral-convolution texture that flows with the air."""
        lab = self.lab
        if getattr(self, "_noise", None) is None or self._noise.shape != (lab.ny, lab.nx):
            rng = np.random.default_rng(11)
            raw = rng.random((lab.ny + 2, lab.nx + 2)).astype(np.float32)
            # A light 3x3 box blur: streaks become continuous strands instead of salt-and-pepper static.
            blur = sum(raw[dy:dy + lab.ny, dx:dx + lab.nx] for dy in range(3) for dx in range(3)) / 9.0
            self._noise = ((blur - blur.mean()) * 3.0 + 0.5).clip(0.0, 1.0).astype(np.float32)
            self._lic_buf = np.full((lab.ny, lab.nx), 0.5, dtype=np.float32)
        self._lic_phase = (getattr(self, "_lic_phase", 0.0) + 0.35) % (2.0 * np.pi) if self.running else \
            getattr(self, "_lic_phase", 0.0)
        tex = line_integral_convolution(lab.ux, lab.uy, lab.solid, self._noise, 12, self._lic_phase, self._lic_buf)
        # Stretch the (narrow, ~0.5-centred) LIC histogram for contrast, then shade the colours.
        t = np.clip((tex - 0.5) * 2.4 + 0.5, 0.0, 1.0)
        shade = (0.25 + 0.95 * t)[..., None]
        return np.clip(rgb * shade, 0, 255).astype(np.uint8)

    # -- export ----------------------------------------------------------------
    def save_png(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not self.grab().save(str(out)):
            raise OSError(f"Could not write {out}")
        return out

    # -- Qt events ---------------------------------------------------------------
    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._layout_overlays()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._layout_overlays()

    def _layout_overlays(self) -> None:
        self.readout.adjustSize()
        self.readout.move(14, 14)
        self.legend.adjustSize()
        self.legend.move(16, max(0, self.height() - self.legend.height() - 12))
        self.readout.raise_()
        self.legend.raise_()

    def shutdown(self) -> None:
        self.timer.stop()
