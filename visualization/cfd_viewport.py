"""3-D wind-tunnel viewport (PyVista / VTK embedded in PySide6).

Everything here runs on the GUI thread. Each visual layer is an actor with a stable name, and an
old actor is always removed before its replacement is added, so repeated runs and slider drags
never leak VTK objects.

Layers (all combinable):

* ``geometry``: PBR metallic/ceramic body, or the surface Cp heatmap.
* ``streamlines``: RK4/5 stream tubes from a line, plane or sphere seed rake, coloured by speed or Cp.
* ``particles``: animated glowing tracers advected along the computed streamlines (QTimer).
* ``slice`` / ``glyphs``: a movable cross-section with contours and velocity arrows.
* ``floor``: a translucent wind-tunnel floor grid, plus an axis triad in the corner.

The widget also hosts the floating KPI HUD (top-left), the camera pill toolbar (bottom-centre)
and the drag-and-drop overlay.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import QEvent, QObject, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QResizeEvent
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget,
)

from config.settings import (
    PARAM_LIMITS, SUPPORTED_MESH_EXTENSIONS, IsoField, ScalarField, SeedShape, SliceAxis,
)
from config.theme import (
    CFD_BACKGROUND_BOTTOM, CFD_BACKGROUND_TOP, PALETTE, make_icon, mono_font, set_property,
)
from core.geometry_manager import GeometryModel
from core.results import AeroKPIs, SolverResult

log = logging.getLogger("aeroflow.viz.cfd")

ACTOR_GEOMETRY = "geometry"
ACTOR_FLOOR = "floor"
ACTOR_FLOOR_GRID = "floor_grid"
ACTOR_STREAMLINES = "streamlines"
ACTOR_PARTICLES = "particles"
ACTOR_SLICE = "slice"
ACTOR_GLYPHS = "glyphs"
ACTOR_SEEDS = "seed_rake"
ACTOR_ISO = "isosurface"
ACTOR_VOLUME = "vortex_cloud"
#: The three planes of the "three orthogonal slices" mode.
ACTOR_SLICES_3 = ("slice_x", "slice_y", "slice_z")
ACTOR_PROBE, ACTOR_PROBE_B, ACTOR_PROBE_LINE = "probe_a", "probe_b", "probe_line"
#: Solid colours of the three isosurface kinds (they overlay the colour-mapped layers).
ISO_COLORS = {IsoField.QCRITERION: "#5EEAD4", IsoField.LAMBDA2: "#F472B6", IsoField.SPEED: "#FBBF24", IsoField.WAKE: "#A78BFA"}
# Scalar-bar titles must be unique per layer: PyVista keys bars by title, so two layers sharing a title
# used to delete each other's bar (and stack two different colour maps under one legend).
BAR_CP = "Cp (surface)"
BAR_STREAM_SPEED = "Speed m/s (lines)"
BAR_STREAM_CP = "Cp (lines)"
BAR_SLICE_SPEED = "Speed m/s (slice)"
BAR_SLICE_CP = "Cp (slice)"
_ALL_BARS = (BAR_CP, BAR_STREAM_SPEED, BAR_STREAM_CP, BAR_SLICE_SPEED, BAR_SLICE_CP)
#: Diverging map for every Cp display (surface, slice, streamlines): blue = suction, white = Cp 0, red = high.
CP_CMAP = "coolwarm"


def cp_clim(values: Optional[np.ndarray]) -> tuple[float, float]:
    """Symmetric Cp colour range so Cp = 0 is always the neutral (white) colour of the diverging map."""
    if values is None or len(values) == 0:
        return (-1.5, 1.5)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (-1.5, 1.5)
    m = float(max(abs(np.percentile(v, 1.0)), abs(np.percentile(v, 99.0)), 1.0))
    m = min(m, 6.0)
    return (-m, m)


def speed_clim(values: Optional[np.ndarray], v_inf: float) -> tuple[float, float]:
    """Speed range centred on the freestream speed, so V∞ is always the middle colour of the map:
    accelerated air (over a wing) reads hot, decelerated air (stagnation, wake) reads cold.

    The half-width is the larger 1st/99th-percentile excursion from V∞ (at least 5 %), so the
    colours use the whole map. A fixed 0–1.6·V∞ range used to paint ~95 % of every tube one colour,
    and a plain percentile range let a wake push the freestream to the top of the scale."""
    v_inf = max(float(v_inf), 1e-9)
    if values is None or len(values) == 0:
        return (0.0, 2.0 * v_inf)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0.0, 2.0 * v_inf)
    dev = max(abs(float(np.percentile(v, 1.0)) - v_inf), abs(float(np.percentile(v, 99.0)) - v_inf),
              0.05 * v_inf)
    dev = min(dev, v_inf)                     # never below zero speed
    return (v_inf - dev, v_inf + dev)
#: keys VTK's default interactor style would act on (q/e even close the window) - swallowed here.
_BLOCKED_VTK_KEYS = {Qt.Key.Key_Q, Qt.Key.Key_E, Qt.Key.Key_W, Qt.Key.Key_S, Qt.Key.Key_3, Qt.Key.Key_J,
                     Qt.Key.Key_T, Qt.Key.Key_P, Qt.Key.Key_U}


@dataclass(frozen=True)
class VisualSettings:
    """All user-adjustable visualisation options of the CFD viewport."""

    show_geometry: bool = True
    show_cp: bool = True
    show_streamlines: bool = True
    show_particles: bool = False
    show_slice: bool = False
    show_vectors: bool = False
    show_floor: bool = True
    show_seed_rake: bool = False
    colormap: str = "turbo"
    streamline_scalar: ScalarField = ScalarField.VELOCITY
    streamline_count: int = int(PARAM_LIMITS["streamline_count"].default)
    tube_radius: float = PARAM_LIMITS["tube_radius"].default
    seed_shape: SeedShape = SeedShape.PLANE
    seed_width: float = PARAM_LIMITS["seed_width"].default
    seed_height: float = PARAM_LIMITS["seed_height"].default
    seed_distance: float = PARAM_LIMITS["seed_distance"].default
    slice_axis: SliceAxis = SliceAxis.Y
    slice_position: float = PARAM_LIMITS["slice_position"].default
    slice_scalar: ScalarField = ScalarField.VELOCITY
    vector_density: int = int(PARAM_LIMITS["vector_density"].default)
    particle_count: int = int(PARAM_LIMITS["particle_count"].default)
    particle_speed: float = PARAM_LIMITS["particle_speed"].default
    ambient_occlusion: bool = True
    show_iso: bool = False
    show_volume: bool = False
    iso_field: IsoField = IsoField.QCRITERION
    iso_level: float = PARAM_LIMITS["iso_level"].default
    slice_three: bool = False          # three orthogonal planes through the slice point instead of one
    lock_ranges: bool = False          # keep the current colour ranges (fair comparisons between results)

    def clamped(self) -> "VisualSettings":
        """Copy with all numeric options clamped to their limits."""
        L = PARAM_LIMITS
        return replace(
            self,
            streamline_count=int(L["streamline_count"].clamp(self.streamline_count)),
            tube_radius=L["tube_radius"].clamp(self.tube_radius),
            seed_width=L["seed_width"].clamp(self.seed_width),
            seed_height=L["seed_height"].clamp(self.seed_height),
            seed_distance=L["seed_distance"].clamp(self.seed_distance),
            slice_position=L["slice_position"].clamp(self.slice_position),
            vector_density=int(L["vector_density"].clamp(self.vector_density)),
            particle_count=int(L["particle_count"].clamp(self.particle_count)),
            particle_speed=L["particle_speed"].clamp(self.particle_speed),
            iso_level=L["iso_level"].clamp(self.iso_level),
        )


# Which layers must be rebuilt when a given option changes.
_DEPENDS: dict[str, tuple[str, ...]] = {
    "show_geometry": ("geometry",), "show_cp": ("geometry",),
    "show_streamlines": ("streamlines", "particles"), "colormap": ("streamlines", "slice"),
    "streamline_scalar": ("streamlines",), "streamline_count": ("streamlines", "particles"),
    "tube_radius": ("streamlines",), "seed_shape": ("streamlines", "particles", "seeds"),
    "seed_width": ("streamlines", "particles", "seeds"), "seed_height": ("streamlines", "particles", "seeds"),
    "seed_distance": ("streamlines", "particles", "seeds"), "show_seed_rake": ("seeds",),
    # Particles ride on the traced lines, so turning them on must (re)trace even when tubes are hidden.
    "show_particles": ("streamlines", "particles"), "particle_count": ("particles",), "particle_speed": (),
    "show_slice": ("slice",), "show_vectors": ("slice",), "slice_axis": ("slice",),
    "slice_position": ("slice",), "slice_scalar": ("slice",), "vector_density": ("slice",),
    "show_floor": ("floor",), "ambient_occlusion": ("lighting",),
    "show_iso": ("iso",), "iso_field": ("iso",), "iso_level": ("iso",),
    "show_volume": ("volume",),
    "slice_three": ("slice",),
    "lock_ranges": ("geometry", "streamlines", "slice"),
}


def model_polydata(model: GeometryModel) -> pv.PolyData:
    """Build a ``pv.PolyData`` triangle surface from a :class:`GeometryModel`."""
    faces = np.column_stack([np.full(len(model.faces), 3, dtype=np.int64), model.faces.astype(np.int64)]).ravel()
    return pv.PolyData(np.asarray(model.vertices, dtype=np.float64), faces)


class _ParticleSystem:
    """Advects tracer particles along precomputed streamline polylines (vectorised)."""

    def __init__(self, lines: pv.PolyData, count: int, time_scale: float, seed: int = 3) -> None:
        self.valid = False
        pts = np.asarray(lines.points)
        conn = np.asarray(lines.lines)
        if len(pts) == 0 or len(conn) == 0:
            return
        times = (np.asarray(lines.point_data["IntegrationTime"], dtype=float)
                 if "IntegrationTime" in lines.point_data else None)
        segments: list[np.ndarray] = []
        i = 0
        while i < len(conn):
            n = int(conn[i])
            ids = conn[i + 1:i + 1 + n]
            if n >= 3:
                segments.append(ids)
            i += n + 1
        if not segments:
            return
        seg_pts, seg_t, starts, lengths, durations = [], [], [], [], []
        offset = 0
        for ids in segments:
            p = pts[ids]
            if times is not None:
                t = times[ids] - times[ids][0]
            else:
                t = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
            keep = np.concatenate([[True], np.diff(t) > 1e-12])
            p, t = p[keep], t[keep]
            if len(p) < 2 or t[-1] <= 0.0:
                continue
            seg_pts.append(p)
            seg_t.append(t)
            starts.append(offset)
            lengths.append(len(p))
            durations.append(t[-1])
            offset += len(p)
        if not seg_pts:
            return
        self.points = np.vstack(seg_pts)
        self.starts = np.asarray(starts)
        self.lengths = np.asarray(lengths)
        self.durations = np.asarray(durations)
        self.big = float(self.durations.max()) * 2.0 + 1.0
        line_of_point = np.repeat(np.arange(len(seg_pts)), self.lengths)
        self.keys = np.concatenate(seg_t) + line_of_point * self.big
        rng = np.random.default_rng(seed)
        # Weight lines by duration so particles are spread evenly in time.
        w = self.durations / self.durations.sum()
        self.line = rng.choice(len(seg_pts), size=count, p=w)
        self.phase = rng.random(count) * self.durations[self.line]
        self.time_scale = time_scale
        self.clock = 0.0
        self.valid = True

    def positions(self, dt_wall: float, speed: float) -> np.ndarray:
        """Advance the clock and return particle positions (N, 3)."""
        self.clock += dt_wall * self.time_scale * speed
        dur = self.durations[self.line]
        t = np.mod(self.phase + self.clock, dur)
        key = t + self.line * self.big
        idx = np.searchsorted(self.keys, key)
        lo = self.starts[self.line]
        hi = lo + self.lengths[self.line] - 1
        idx = np.clip(idx, lo + 1, hi)
        k0, k1 = self.keys[idx - 1], self.keys[idx]
        f = np.clip((key - k0) / np.maximum(k1 - k0, 1e-12), 0.0, 1.0)[:, None]
        return self.points[idx - 1] * (1.0 - f) + self.points[idx] * f


class _MetricCard(QFrame):
    """One glass KPI tile (caption + monospace value)."""

    def __init__(self, caption: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(0)
        cap = QLabel(caption)
        cap.setProperty("role", "metric-caption")
        self.value = QLabel("—")
        self.value.setProperty("role", "metric")
        self.value.setFont(mono_font(16, bold=True))
        lay.addWidget(cap)
        lay.addWidget(self.value)

    def set_value(self, text: str, level: Optional[str] = None) -> None:
        """Update the value (optionally colouring it by ``level``)."""
        self.value.setText(text)
        set_property(self.value, "level", level or "")
        # The stylesheet font (not the one set in __init__) decides the real text width; size from it,
        # so values like "5166.6" or "3.42e+06" are never clipped.
        self.value.ensurePolished()
        width = self.value.fontMetrics().horizontalAdvance(text + "00") + 6
        self.value.setMinimumWidth(max(width, 56))


class KpiHud(QFrame):
    """Floating glassmorphic KPI panel shown at the top-left of the viewport."""

    CAPTIONS = (("cl", "CL"), ("cd", "CD"), ("ld", "L/D"), ("lift", "LIFT  N"), ("drag", "DRAG  N"), ("re", "REYNOLDS"))

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("GlassCard")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        grid = QGridLayout(self)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(4)
        self.cards: dict[str, _MetricCard] = {}
        for i, (key, cap) in enumerate(self.CAPTIONS):
            card = _MetricCard(cap, self)
            self.cards[key] = card
            grid.addWidget(card, i // 3, i % 3)
        self.engine = QLabel("")
        self.engine.setProperty("role", "muted")
        grid.addWidget(self.engine, 2, 0, 1, 3)
        self.adjustSize()

    def update_kpis(self, kpis: Optional[AeroKPIs], engine: str = "") -> None:
        """Show the KPIs of a result (``None`` clears the panel)."""
        if kpis is None:
            for c in self.cards.values():
                c.set_value("—")
            self.engine.setText("No solution yet — press Run")
            return
        stall = "warning" if kpis.stalled else None
        self.cards["cl"].set_value(f"{kpis.cl:+.3f}", stall)
        self.cards["cd"].set_value(f"{kpis.cd:.4f}")
        self.cards["ld"].set_value(f"{kpis.l_over_d:.1f}", "success" if kpis.l_over_d > 10 else None)
        self.cards["lift"].set_value(_eng(kpis.lift_n))
        self.cards["drag"].set_value(_eng(kpis.drag_n))
        self.cards["re"].set_value(_eng(kpis.reynolds))          # "3.42M": compact, never clipped
        txt = f"{engine} · α {kpis.alpha_deg:+.1f}° · V {kpis.velocity_ms:.0f} m/s"
        if kpis.stalled:
            txt += " · STALL"
        self.engine.setText(txt)
        self.layout().activate()
        self.adjustSize()


def _eng(v: float) -> str:
    """Compact engineering formatting (1.23k, 4.5M)."""
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.2f}M"
    if a >= 1e4:
        return f"{v / 1e3:.1f}k"
    return f"{v:.1f}"


class CameraBar(QFrame):
    """Translucent pill toolbar with one-click camera views and a screenshot button."""

    view_requested = Signal(str)

    BUTTONS = (("iso", "iso", "Isometric view"), ("side", "side", "Side view (from the left wing)"),
               ("top", "top", "Top view"), ("front", "front", "Front view (looking downstream)"),
               ("reset", "reset", "Reset camera to fit the model"), ("screenshot", "camera", "Save 4K screenshot"))

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("FloatingBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 4, 10, 4)
        lay.setSpacing(2)
        for key, icon, tip in self.BUTTONS:
            b = QToolButton(self)
            b.setProperty("role", "camera")
            b.setIcon(make_icon(icon, 20))
            b.setIconSize(QSize(20, 20))
            b.setToolTip(tip)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _=False, k=key: self.view_requested.emit(k))
            lay.addWidget(b)
        self.adjustSize()


class DropOverlay(QFrame):
    """Cyan-bordered overlay shown while a mesh file is dragged over the viewport."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("DropOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        lay = QVBoxLayout(self)
        lbl = QLabel("Drop STL / OBJ / PLY / OFF / VTK to load")
        lbl.setProperty("role", "title")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(lbl)
        self.hide()


class _KeyFilter(QObject):
    """Swallows VTK's dangerous default key bindings (q/e close the window, w/s wireframe ...)."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 (Qt API)
        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            if event.key() in _BLOCKED_VTK_KEYS:  # type: ignore[attr-defined]
                return True
        return False


class CFDViewport(QWidget):
    """The CFD wind-tunnel viewport with HUD, camera toolbar and drag-and-drop."""

    file_dropped = Signal(str)
    fps_changed = Signal(float)
    status_message = Signal(str)
    screenshot_saved = Signal(str)
    screenshot_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.plotter = QtInteractor(self, multi_samples=8, auto_update=False)
        self.plotter.setAcceptDrops(False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plotter.interactor if hasattr(self.plotter, "interactor") else self.plotter)
        self._key_filter = _KeyFilter(self)
        self.plotter.installEventFilter(self._key_filter)
        if hasattr(self.plotter, "interactor"):
            self.plotter.interactor.installEventFilter(self._key_filter)

        self.hud = KpiHud(self)
        self.camera_bar = CameraBar(self)
        self.camera_bar.view_requested.connect(self._on_view)
        self.drop_overlay = DropOverlay(self)
        self.hud.update_kpis(None)
        self.probe_label = QLabel(self)
        self.probe_label.setObjectName("GlassCard")
        self.probe_label.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.probe_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.probe_label.setTextFormat(Qt.TextFormat.RichText)
        self.probe_label.setFont(mono_font(11))
        self.probe_label.setMargin(10)
        self.probe_label.hide()
        self._probe_mode = "off"
        self._probe_a: Optional[np.ndarray] = None

        self.settings = VisualSettings()
        self._model: Optional[GeometryModel] = None
        self._mesh: Optional[pv.PolyData] = None
        self._result: Optional[SolverResult] = None
        self._dataset: Optional[Any] = None
        self._surface: Optional[pv.PolyData] = None
        self._lines: Optional[pv.PolyData] = None
        self._particles: Optional[_ParticleSystem] = None
        self._particle_poly: Optional[pv.PolyData] = None
        self._pending: set[str] = set()
        self._closed = False
        self._bar_titles: dict[str, str] = {}   # layer -> its current scalar-bar title
        self._geo_poly: Optional[pv.PolyData] = None   # the drawn surface (normals precomputed)
        self._geo_mode: Optional[str] = None           # "cp" | "plain" | None
        self._volume_on = False                        # vortex-cloud volume currently drawn
        self._anim_paused = False          # paused by the main window (another mode is visible)
        self._ssao_on = False              # SSAO currently enabled in the renderer
        self._ssao_suspended = False       # SSAO temporarily off while particles animate
        self._locked: dict[str, tuple[float, float]] = {}   # colour ranges frozen by "Lock colour ranges"
        self._sweep_timer = QTimer(self)                     # slice sweep animation
        self._sweep_timer.setInterval(70)
        self._sweep_timer.timeout.connect(self._sweep_step)
        self._sweep_dir = 1.0

        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(70)
        self._rebuild_timer.timeout.connect(self._flush_rebuild)
        self._anim_timer = QTimer(self)
        self._anim_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._anim_timer.setInterval(16)
        self._anim_timer.timeout.connect(self._animate)
        self._last_tick = time.perf_counter()
        self._fps_frames = 0
        self._fps_t0 = time.perf_counter()

        self._setup_scene()

    # -- scene setup -----------------------------------------------------
    def _setup_scene(self) -> None:
        p = self.plotter
        p.set_background(CFD_BACKGROUND_BOTTOM, top=CFD_BACKGROUND_TOP)
        # MSAA x8 is already on (QtInteractor multi_samples); FXAA on top of it only blurred thin lines.
        self._set_depth_peeling(True)
        self._setup_lights()
        try:
            p.add_axes(interactive=False, line_width=2, color=PALETTE.text_muted, xlabel="X", ylabel="Y", zlabel="Z")
        except Exception as exc:
            log.debug("Axes widget unavailable: %s", exc)
        self._apply_ssao(self.settings.ambient_occlusion)

    def _setup_lights(self) -> None:
        """Three-point studio rig: warm key, cool fill, subtle rim."""
        p = self.plotter
        p.remove_all_lights()
        key = pv.Light(position=(-6.0, -8.0, 9.0), focal_point=(0.0, 0.0, 0.0), color="#FFF4E6", intensity=0.95)
        fill = pv.Light(position=(-5.0, 9.0, 3.0), focal_point=(0.0, 0.0, 0.0), color="#9EC9FF", intensity=0.45)
        rim = pv.Light(position=(9.0, 0.0, 6.0), focal_point=(0.0, 0.0, 0.0), color="#C7F7FF", intensity=0.55)
        for light in (key, fill, rim):
            light.positional = False
            p.add_light(light)
        p.add_light(pv.Light(light_type="headlight", intensity=0.18))

    def _set_depth_peeling(self, enabled: bool) -> None:
        """Correct ordering of translucent layers (floor, slice) for still images.

        It costs ~14 ms per frame, so it is switched off while the tracer particles animate
        (≈ 50 FPS → 160+ FPS) and back on when the animation stops."""
        try:
            if enabled:
                self.plotter.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.0)
            else:
                self.plotter.disable_depth_peeling()
        except Exception as exc:
            log.debug("Depth peeling unavailable: %s", exc)

    def _apply_ssao(self, enabled: bool) -> None:
        """Enable screen-space ambient occlusion where VTK supports it.

        The pass is always removed and re-created, because PyVista ignores ``enable_ssao`` when a pass
        already exists, which kept the radius sized for the first model forever. SSAO stays off while
        tracer particles animate (it costs ~28 ms per frame and capped the animation near 30 FPS)."""
        # VTK's SSAO pass cannot composite an unshaded volume (its shader fails to compile), so ambient
        # occlusion is also off while the vortex cloud is shown.
        want = enabled and not self._ssao_suspended and not self._volume_on
        try:
            if self._ssao_on:
                self.plotter.disable_ssao()
                self._ssao_on = False
            if want:
                length = self._ref_length()
                self.plotter.enable_ssao(radius=0.15 * length, bias=0.004 * length, kernel_size=64, blur=True)
                self._ssao_on = True
        except Exception as exc:
            log.debug("SSAO unavailable: %s", exc)
            self._ssao_on = False

    # -- geometry and result -------------------------------------------
    def _ref_length(self) -> float:
        if self._model is None:
            return 1.0
        i = self._model.info
        return float(max(i.chord_m, i.span_m, i.height_m, 1e-6))

    def set_geometry(self, model: Optional[GeometryModel], reset_camera: bool = True) -> None:
        """Show a new geometry. Any previous flow result is cleared (it belonged to the old shape)."""
        self._model = model
        self._mesh = model_polydata(model) if model is not None else None
        self._geo_poly, self._geo_mode = None, None        # a new mesh: never recolour the old actor
        self.clear_result(render=False, rebuild_geometry=False)
        self._apply_ssao(self.settings.ambient_occlusion)
        self._rebuild({"geometry", "floor", "seeds"})
        if reset_camera:
            self.view("iso")

    def clear_result(self, render: bool = True, rebuild_geometry: bool = True) -> None:
        """Remove all flow layers and forget the current result."""
        self._stop_animation()
        self._result = None
        self._dataset = None
        self._surface = None
        self._lines = None
        self._particles = None
        for name in (ACTOR_STREAMLINES, ACTOR_PARTICLES, ACTOR_SLICE, ACTOR_GLYPHS, ACTOR_ISO, ACTOR_VOLUME,
                     *ACTOR_SLICES_3, ACTOR_PROBE, ACTOR_PROBE_B, ACTOR_PROBE_LINE):
            self._remove(name)
        self._set_volume_on(False)
        for bar in _ALL_BARS:
            self._remove_bar(bar)
        for layer in list(self._bar_titles):
            self._remove_layer_bar(layer)
        self.hud.update_kpis(None)
        if self._mesh is not None and rebuild_geometry:
            self._rebuild({"geometry"}, render=render)
        elif render:
            self.plotter.render()

    def set_result(self, result: SolverResult) -> None:
        """Display a solver result (must be called on the GUI thread)."""
        self._stop_animation()
        self._result = result
        try:
            # Normally already built (with vorticity, Q-criterion, total pressure) by the worker thread.
            self._dataset = result.field.prepare_visual()
        except Exception as exc:
            log.error("Cannot build the flow-field dataset: %s", exc)
            self._dataset = None
        self._surface = result.surface.to_pyvista() if result.surface is not None else None
        self.hud.update_kpis(result.kpis, result.engine)
        self._rebuild({"geometry", "streamlines", "particles", "slice", "seeds", "iso", "volume"})

    @property
    def has_result(self) -> bool:
        """True when a flow field is displayed."""
        return self._dataset is not None

    # -- settings ----------------------------------------------------------
    def update_settings(self, settings: VisualSettings, immediate: bool = False) -> None:
        """Apply new visual settings; only affected layers are rebuilt (debounced for sliders)."""
        settings = settings.clamped()
        old = self.settings
        self.settings = settings
        if not settings.lock_ranges:
            self._locked.clear()
        changed = [k for k in _DEPENDS if getattr(old, k) != getattr(settings, k)]
        layers: set[str] = set()
        for k in changed:
            layers.update(_DEPENDS[k])
        if "lighting" in layers:
            self._apply_ssao(settings.ambient_occlusion)
            layers.discard("lighting")
            self.plotter.render()
        if not layers:
            if settings.show_particles and self._particles is not None and not self._anim_timer.isActive() \
                    and not self._anim_paused:
                self._start_animation()
            return
        self._pending |= layers
        if immediate:
            self._flush_rebuild()
        else:
            self._rebuild_timer.start()

    def _flush_rebuild(self) -> None:
        layers, self._pending = self._pending, set()
        if layers:
            self._rebuild(layers)

    def _rebuild(self, layers: set[str], render: bool = True) -> None:
        """Rebuild the given layers in dependency order."""
        if self._closed:
            return
        try:
            if "geometry" in layers:
                self._build_geometry()
            if "floor" in layers:
                self._build_floor()
            if "streamlines" in layers:
                self._build_streamlines()
            if "particles" in layers or "streamlines" in layers:
                self._build_particles()
            if "slice" in layers:
                self._build_slice()
            if "seeds" in layers:
                self._build_seed_rake()
            if "iso" in layers:
                self._build_iso()
            if "volume" in layers:
                self._build_volume()
        except Exception as exc:  # a failed layer must never take the viewport down
            log.exception("Visualisation update failed: %s", exc)
            self.status_message.emit(f"Visualisation update failed: {exc}")
        if render:
            self.plotter.render()

    # -- actor helpers -----------------------------------------------------
    def _remove(self, name: str) -> None:
        try:
            self.plotter.remove_actor(name, reset_camera=False, render=False)
        except Exception:
            pass

    def _remove_bar(self, title: str) -> None:
        try:
            bars = getattr(self.plotter, "scalar_bars", None)
            if bars is not None and title in list(bars.keys()):
                self.plotter.remove_scalar_bar(title, render=False)
        except Exception:
            pass

    def _remove_layer_bar(self, layer: str) -> None:
        title = self._bar_titles.pop(layer, None)
        if title:
            self._remove_bar(title)

    def _scalar_spec(self, field: ScalarField, mesh: pv.DataSet) -> tuple[str, str, tuple[float, float], str]:
        """``(array name, colour map, colour range, legend title)`` for colouring ``mesh`` by ``field``.

        Falls back to speed when the requested array is missing (e.g. an OpenFOAM result without
        vorticity)."""
        v_inf = self._result.field.freestream_ms if self._result is not None else 1.0
        pd = mesh.point_data

        def vals(name: str) -> np.ndarray:
            a = np.asarray(pd[name], dtype=float)
            if "solid" in pd:
                a = a[np.asarray(pd["solid"]) == 0]
            return a[np.isfinite(a)]

        if field is ScalarField.PRESSURE and "Cp" in pd:
            return "Cp", CP_CMAP, self._lock("Cp", lambda: cp_clim(vals("Cp"))), "Cp"
        if field is ScalarField.VORTICITY and "vorticity_mag" in pd:
            def vort() -> tuple[float, float]:
                v = vals("vorticity_mag")
                hi = float(np.percentile(v, 99.0)) if v.size else 1.0
                return (0.0, max(hi, 1e-6))
            return "vorticity_mag", "inferno", self._lock("vorticity_mag", vort), "Vorticity 1/s"
        if field is ScalarField.TOTAL_PRESSURE and "Cp0" in pd:
            def cp0() -> tuple[float, float]:
                v = vals("Cp0")
                lo = float(np.percentile(v, 1.0)) if v.size else 0.0
                return (min(lo, 0.9), 1.0)
            return "Cp0", "magma", self._lock("Cp0", cp0), "Total Cp0"
        return "speed", self.settings.colormap, \
            self._lock("speed", lambda: speed_clim(vals("speed") if "speed" in pd else None, v_inf)), "Speed m/s"

    def _lock(self, key: str, compute: Any) -> tuple[float, float]:
        """Colour range for ``key``: computed from the data, or - with *Lock colour ranges* on - the range
        in effect when the lock was switched on (so two results are coloured on the same scale)."""
        if self.settings.lock_ranges:
            if key not in self._locked:
                self._locked[key] = tuple(compute())
            return self._locked[key]
        return tuple(compute())

    @staticmethod
    def _bar_args(title: str, slot: int) -> dict[str, Any]:
        # Three stacked slots (surface, lines, slice) with a gap for each bar's title above it.
        return dict(title=title, vertical=True, position_x=0.885, position_y=0.07 + 0.30 * slot, height=0.19,
                    width=0.032, color=PALETTE.text_muted, title_font_size=12, label_font_size=11,
                    n_labels=5, fmt="%.2f", font_family="arial", shadow=False)

    # -- layers ------------------------------------------------------------
    def _geometry_mode(self) -> Optional[str]:
        s = self.settings
        if self._mesh is None or not s.show_geometry:
            return None
        if s.show_cp and self._surface is not None and "Cp" in self._surface.point_data:
            return "cp"
        return "plain"

    def _update_surface_cp(self, cp: np.ndarray) -> bool:
        """Recolour the existing Cp surface in place (≈1 ms instead of a ~200 ms actor rebuild).

        Returns False when there is no compatible Cp actor, so the caller must rebuild."""
        poly = self._geo_poly
        if (self._geo_mode != "cp" or poly is None or ACTOR_GEOMETRY not in self.plotter.actors
                or len(cp) != poly.n_points):
            return False
        poly.point_data["Cp"] = np.asarray(cp, dtype=np.float32)
        actor = self.plotter.actors[ACTOR_GEOMETRY]
        try:
            actor.mapper.scalar_range = self._lock("Cp_surface", lambda: cp_clim(cp))
        except Exception:  # noqa: BLE001
            return False
        return True

    def _build_geometry(self) -> None:
        mode = self._geometry_mode()
        if mode == "cp" and self._update_surface_cp(np.asarray(self._surface.point_data["Cp"])):
            return                                        # same mesh, new pressures: no rebuild needed
        self._remove(ACTOR_GEOMETRY)
        self._remove_bar(BAR_CP)
        self._geo_poly, self._geo_mode = None, None
        if mode is None:
            return
        # Normals are computed once here, so add_mesh needs no smooth_shading pass of its own.
        src = self._surface if mode == "cp" else self._mesh
        poly = src.compute_normals(cell_normals=False, split_vertices=False, auto_orient_normals=False)
        if mode == "cp":
            cp = np.asarray(poly.point_data["Cp"])
            self.plotter.add_mesh(poly, name=ACTOR_GEOMETRY, scalars="Cp", cmap=CP_CMAP,
                                  clim=self._lock("Cp_surface", lambda: cp_clim(cp)),
                                  smooth_shading=False, interpolation="phong", specular=0.35, specular_power=30,
                                  scalar_bar_args=self._bar_args(BAR_CP, 0), reset_camera=False)
        else:
            try:
                self.plotter.add_mesh(poly, name=ACTOR_GEOMETRY, color="#C8D1DC", pbr=True, metallic=0.55,
                                      roughness=0.32, smooth_shading=False, interpolation="pbr", reset_camera=False)
            except Exception:   # PBR needs VTK >= 9; fall back to Phong
                self.plotter.add_mesh(poly, name=ACTOR_GEOMETRY, color="#C8D1DC", specular=0.6,
                                      specular_power=40, smooth_shading=False, interpolation="phong",
                                      reset_camera=False)
        self._geo_poly, self._geo_mode = poly, mode

    def preview(self, kpis: AeroKPIs, cp_vertex: Optional[np.ndarray], label: str = "Preview") -> None:
        """Instant feedback while a slider is dragged: new KPIs and surface pressures, no volume rebuild."""
        self.hud.update_kpis(kpis, f"{label} — release to update the flow")
        if cp_vertex is not None and self._update_surface_cp(cp_vertex):
            pass
        self.plotter.render()

    def _build_floor(self) -> None:
        self._remove(ACTOR_FLOOR)
        self._remove(ACTOR_FLOOR_GRID)
        if self._model is None or not self.settings.show_floor:
            return
        info = self._model.info
        L = self._ref_length()
        cx = 0.5 * (info.bounds_min[0] + info.bounds_max[0]) + 1.0 * L
        cy = 0.5 * (info.bounds_min[1] + info.bounds_max[1])
        z = info.bounds_min[2] - max(0.35 * info.height_m, 0.25 * L)
        size_x, size_y = 6.0 * L, 4.0 * L
        plane = pv.Plane(center=(cx, cy, z), direction=(0, 0, 1), i_size=size_x, j_size=size_y,
                         i_resolution=1, j_resolution=1)
        grid = pv.Plane(center=(cx, cy, z + 1e-4 * L), direction=(0, 0, 1), i_size=size_x, j_size=size_y,
                        i_resolution=24, j_resolution=16)
        self.plotter.add_mesh(plane, name=ACTOR_FLOOR, color=PALETTE.surface, opacity=0.35, lighting=False,
                              reset_camera=False, pickable=False)
        self.plotter.add_mesh(grid, name=ACTOR_FLOOR_GRID, style="wireframe", color=PALETTE.border_strong,
                              opacity=0.45, line_width=1, lighting=False, reset_camera=False, pickable=False)

    def _seed_source(self) -> Optional[pv.DataSet]:
        """Seed points for the current rake shape, clipped to the flow domain."""
        if self._model is None:
            return None
        s, info = self.settings, self._model.info
        L = max(info.chord_m, info.span_m)
        cx = info.bounds_min[0] - s.seed_distance * L
        cy = 0.5 * (info.bounds_min[1] + info.bounds_max[1])
        cz = 0.5 * (info.bounds_min[2] + info.bounds_max[2])
        if self._dataset is not None:
            b = self._dataset.bounds
            cx = float(np.clip(cx, b[0] + 1e-3 * L, b[1] - 1e-3 * L))
        w, h, n = s.seed_width * L, s.seed_height * L, max(2, int(s.streamline_count))
        if s.seed_shape == SeedShape.LINE:
            return pv.Line((cx, cy - w / 2, cz), (cx, cy + w / 2, cz), resolution=n - 1)
        if s.seed_shape == SeedShape.SPHERE:
            k = np.arange(n) + 0.5
            phi = np.arccos(1 - 2 * k / n)
            th = math.pi * (1 + 5 ** 0.5) * k
            r = 0.5 * min(w, h)
            pts = np.column_stack([cx + r * np.cos(phi), cy + r * np.sin(phi) * np.cos(th),
                                   cz + r * np.sin(phi) * np.sin(th)])
            return pv.PolyData(pts)
        ny = max(2, int(round(math.sqrt(n * w / max(h, 1e-9)))))
        nz = max(2, int(math.ceil(n / ny)))
        ys = np.linspace(cy - w / 2, cy + w / 2, ny)
        zs = np.linspace(cz - h / 2, cz + h / 2, nz)
        yy, zz = np.meshgrid(ys, zs)
        pts = np.column_stack([np.full(yy.size, cx), yy.ravel(), zz.ravel()])[:n]
        return pv.PolyData(pts)

    def _build_seed_rake(self) -> None:
        self._remove(ACTOR_SEEDS)
        if not self.settings.show_seed_rake:
            return
        src = self._seed_source()
        if src is None:
            return
        self.plotter.add_mesh(src, name=ACTOR_SEEDS, color=PALETTE.indigo, point_size=6,
                              render_points_as_spheres=True, line_width=3, lighting=False, reset_camera=False)

    def _integrate(self, source: pv.DataSet) -> Optional[pv.PolyData]:
        """Integrate streamlines (RK4/5) through the dataset."""
        ds = self._dataset
        if ds is None:
            return None
        b = ds.bounds
        diag = math.sqrt((b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2)
        cell = diag / 120.0
        # Shorter steps (in cell lengths) give ~4x more points per line, so tubes are smooth, not faceted.
        kw: dict[str, Any] = dict(vectors="velocity", integrator_type=45, integration_direction="forward",
                                  initial_step_length=0.2, step_unit="cl", min_step_length=0.02,
                                  max_step_length=0.5, max_steps=12000, terminal_speed=1e-6, max_error=1e-6,
                                  compute_vorticity=False)
        try:
            lines = ds.streamlines_from_source(source, max_length=2.0 * diag, **kw)
        except TypeError:            # older PyVista: max_time instead of max_length
            lines = ds.streamlines_from_source(source, max_time=2.0 * diag, **kw)
        if lines is None or lines.n_points == 0:
            return None
        log.debug("Streamlines: %d points (cell ≈ %.3g)", lines.n_points, cell)
        return lines

    def _build_streamlines(self) -> None:
        self._remove(ACTOR_STREAMLINES)
        self._remove_layer_bar("streamlines")
        self._lines = None
        if self._dataset is None or self._model is None:
            return
        need_lines = self.settings.show_streamlines or self.settings.show_particles
        if not need_lines:
            return
        src = self._seed_source()
        if src is None:
            return
        lines = self._integrate(src)
        self._lines = lines
        if lines is None:
            self.status_message.emit("No streamlines could be traced from the seed rake — move it upstream.")
            return
        if not self.settings.show_streamlines:
            return
        s = self.settings
        radius = s.tube_radius * self._ref_length()
        try:
            tubes = lines.tube(radius=radius, n_sides=12, capping=False)
        except Exception:
            tubes = lines
        name, cmap, clim, title = self._scalar_spec(s.streamline_scalar, lines)
        if name not in tubes.point_data:
            name, cmap, clim, title = self._scalar_spec(ScalarField.VELOCITY, lines)
        bar = f"{title} (lines)"
        self._bar_titles["streamlines"] = bar
        # The tube filter already emits smooth point normals; re-smoothing cost ~90 ms per rebuild.
        self.plotter.add_mesh(tubes, name=ACTOR_STREAMLINES, scalars=name, cmap=cmap, clim=clim,
                              smooth_shading=False, interpolation="phong", specular=0.3,
                              scalar_bar_args=self._bar_args(bar, 1),
                              reset_camera=False)

    def _build_particles(self) -> None:
        self._stop_animation()
        self._remove(ACTOR_PARTICLES)
        self._particles = None
        self._particle_poly = None
        if not self.settings.show_particles or self._lines is None or self._result is None:
            return
        v_inf = max(self._result.field.freestream_ms, 1e-6)
        L = self._ref_length()
        # One second of wall time ≈ the flow travelling 1.2 model lengths at speed 1.0x.
        time_scale = 1.2 * L / v_inf
        ps = _ParticleSystem(self._lines, int(self.settings.particle_count), time_scale)
        if not ps.valid:
            return
        self._particles = ps
        self._particle_poly = pv.PolyData(ps.positions(0.0, 0.0))
        # Opaque sphere sprites: cheap, crisp at any zoom, and independent of translucency sorting.
        self.plotter.add_mesh(self._particle_poly, name=ACTOR_PARTICLES, color=PALETTE.cyan_hi,
                              point_size=7.0, render_points_as_spheres=True, lighting=False,
                              reset_camera=False, pickable=False)
        if not self._anim_paused:
            self._start_animation()

    def _slice_origin_normal(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        assert self._model is not None
        info = self._model.info
        c = [0.5 * (info.bounds_min[i] + info.bounds_max[i]) for i in range(3)]
        ext = [info.bounds_max[i] - info.bounds_min[i] for i in range(3)]
        axis = {SliceAxis.X: 0, SliceAxis.Y: 1, SliceAxis.Z: 2}[self.settings.slice_axis]
        normal = [0.0, 0.0, 0.0]
        normal[axis] = 1.0
        if axis == 0:   # streamwise: position measured in model lengths
            c[axis] += self.settings.slice_position * self._ref_length()
        else:           # spanwise / vertical: ±1 spans the body, ±1.5 goes beyond it
            c[axis] += self.settings.slice_position * 0.5 * max(ext[axis], 0.05 * self._ref_length())
        return (c[0], c[1], c[2]), (normal[0], normal[1], normal[2])

    def _build_slice(self) -> None:
        self._remove(ACTOR_SLICE)
        self._remove(ACTOR_GLYPHS)
        for name in ACTOR_SLICES_3:
            self._remove(name)
        self._remove_layer_bar("slice")
        if self._dataset is None or self._model is None or not (self.settings.show_slice or self.settings.show_vectors):
            return
        s = self.settings
        origin, normal = self._slice_origin_normal()
        b = self._dataset.bounds
        lo = np.array([b[0], b[2], b[4]])
        hi = np.array([b[1], b[3], b[5]])
        org = np.clip(np.asarray(origin), lo + 1e-6, hi - 1e-6)
        v_inf = self._result.field.freestream_ms if self._result is not None else 1.0
        if s.show_slice:
            if s.slice_three:
                # Three orthogonal planes through the slice point; one shared colour range and legend.
                planes = [(name, n) for name, n in zip(ACTOR_SLICES_3, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                                                        (0.0, 0.0, 1.0)))]
                cuts = [(name, self._dataset.slice(normal=n, origin=tuple(org))) for name, n in planes]
                cuts = [(name, c) for name, c in cuts if c.n_points]
                if cuts:
                    merged = pv.merge([c for _n, c in cuts]) if len(cuts) > 1 else cuts[0][1]
                    field, cmap, clim, title = self._scalar_spec(s.slice_scalar, merged)
                    bar = f"{title} (slice)"
                    self._bar_titles["slice"] = bar
                    for k, (name, c) in enumerate(cuts):
                        self._add_slice_mesh(c, name, field, cmap, clim, bar if k == 0 else None, opacity=0.85)
            else:
                sl = self._dataset.slice(normal=normal, origin=tuple(org))
                if sl.n_points:
                    field, cmap, clim, title = self._scalar_spec(s.slice_scalar, sl)
                    bar = f"{title} (slice)"
                    self._bar_titles["slice"] = bar
                    self._add_slice_mesh(sl, ACTOR_SLICE, field, cmap, clim, bar)
        if s.show_vectors:
            self._build_glyphs(org, np.asarray(normal), lo, hi, v_inf)

    def _add_slice_mesh(self, sl: Any, actor: str, field: str, cmap: str, clim: tuple[float, float],
                        bar: Optional[str], opacity: float = 0.92) -> None:
        view = np.asarray(sl.point_data[field], dtype=float).copy()
        if "solid" in sl.point_data:            # cut the body out of the slice
            view[np.asarray(sl.point_data["solid"]) > 0] = np.nan
        sl.point_data["slice_view"] = view
        kw: dict[str, Any] = dict(scalar_bar_args=self._bar_args(bar, 2)) if bar else dict(show_scalar_bar=False)
        self.plotter.add_mesh(sl, name=actor, scalars="slice_view", cmap=cmap, clim=clim,
                              nan_color=PALETTE.bg_deep, opacity=opacity, lighting=False, reset_camera=False, **kw)

    # -- slice sweep ----------------------------------------------------------
    slice_position_changed = Signal(float)

    def set_slice_sweep(self, on: bool) -> None:
        """Animate the slice plane back and forth through the flow (the slice is switched on if needed)."""
        if on and self._dataset is not None:
            if not self.settings.show_slice:
                self.settings = replace(self.settings, show_slice=True)
            self._sweep_timer.start()
        else:
            self._sweep_timer.stop()

    @property
    def sweeping(self) -> bool:
        return self._sweep_timer.isActive()

    def _sweep_step(self) -> None:
        if self._dataset is None or self._closed:
            self._sweep_timer.stop()
            return
        lim = PARAM_LIMITS["slice_position"]
        pos = self.settings.slice_position + self._sweep_dir * 0.03
        if pos >= lim.maximum or pos <= lim.minimum:
            self._sweep_dir = -self._sweep_dir
            pos = float(np.clip(pos, lim.minimum, lim.maximum))
        self.settings = replace(self.settings, slice_position=pos)
        try:
            self._build_slice()
            self.plotter.render()
        except Exception as exc:  # noqa: BLE001 - never take the viewport down
            log.debug("Slice sweep stopped: %s", exc)
            self._sweep_timer.stop()
            return
        self.slice_position_changed.emit(pos)

    # -- probes -----------------------------------------------------------------
    probe_line_ready = Signal(object)          # dict: distance + field arrays along a probe line

    def set_probe_mode(self, mode: str) -> None:
        """``"off"``, ``"point"`` (click: local flow readout) or ``"line"`` (two clicks: plot along a line)."""
        self._probe_mode = mode
        self._probe_a = None
        try:
            self.plotter.disable_picking()
        except Exception:  # noqa: BLE001 - nothing was enabled
            pass
        for name in (ACTOR_PROBE, ACTOR_PROBE_B, ACTOR_PROBE_LINE):
            self._remove(name)
        self.probe_label.hide()
        if mode in ("point", "line"):
            try:
                self.plotter.enable_point_picking(callback=self._on_pick, left_clicking=True, picker="cell",
                                                  show_message=False, show_point=False,
                                                  clear_on_no_selection=False)
            except Exception as exc:  # noqa: BLE001
                log.warning("Picking unavailable: %s", exc)
                self.status_message.emit(f"Probing is not available: {exc}")
                return
            self.status_message.emit("Probe: click on the slice, a streamline or the model." if mode == "point"
                                     else "Line probe: click the start point, then the end point.")
        self.plotter.render()

    def _sample(self, points: np.ndarray) -> dict[str, np.ndarray]:
        """Flow quantities at ``points`` (N, 3); NaN outside the flow domain or inside the body."""
        probe = pv.PolyData(np.asarray(points, dtype=float)).sample(self._dataset)
        pd = probe.point_data
        valid = np.ones(probe.n_points, dtype=bool)
        if "vtkValidPointMask" in pd:
            valid &= np.asarray(pd["vtkValidPointMask"]) > 0
        if "solid" in pd:
            valid &= np.asarray(pd["solid"]) == 0
        out: dict[str, np.ndarray] = {}
        for name in ("speed", "Cp", "Cp0", "vorticity_mag", "velocity"):
            if name in pd:
                a = np.asarray(pd[name], dtype=float).copy()
                a[~valid] = np.nan
                out[name] = a
        return out

    def _marker(self, name: str, p: np.ndarray, colour: str) -> None:
        r = 0.012 * self._ref_length()
        self.plotter.add_mesh(pv.Sphere(radius=r, center=tuple(p)), name=name, color=colour, lighting=False,
                              pickable=False, reset_camera=False)

    def _on_pick(self, point: Any, *_args: Any) -> None:
        if point is None or self._dataset is None:
            if self._dataset is None:
                self.status_message.emit("Run a simulation first - the probe reads the flow result.")
            return
        p = np.asarray(point, dtype=float).reshape(3)
        if self._probe_mode == "point":
            self._marker(ACTOR_PROBE, p, PALETTE.warning)
            s = self._sample(p[None, :])
            v_inf = self._result.field.freestream_ms if self._result is not None else 1.0
            lines = [f"<b>Probe</b> &nbsp;x {p[0]:+.3f} &nbsp;y {p[1]:+.3f} &nbsp;z {p[2]:+.3f} m"]
            if "speed" in s and np.isfinite(s["speed"][0]):
                sp = float(s["speed"][0])
                lines.append(f"Speed {sp:.2f} m/s &nbsp;({sp / max(v_inf, 1e-9):.3f} V∞)")
                if "velocity" in s:
                    vx, vy, vz = s["velocity"][0]
                    lines.append(f"u {vx:+.2f} &nbsp;v {vy:+.2f} &nbsp;w {vz:+.2f} m/s")
                if "Cp" in s:
                    lines.append(f"Cp {float(s['Cp'][0]):+.3f}")
                if "Cp0" in s:
                    lines.append(f"Total-pressure Cp0 {float(s['Cp0'][0]):.3f}")
                if "vorticity_mag" in s:
                    lines.append(f"Vorticity {float(s['vorticity_mag'][0]):.3g} 1/s")
            elif self._surface is not None and "Cp" in self._surface.point_data:
                k = int(self._surface.find_closest_point(p))
                lines.append(f"On the surface: Cp {float(self._surface.point_data['Cp'][k]):+.3f}")
            else:
                lines.append("Outside the flow field")
            self.probe_label.setText("<br>".join(lines))
            self.probe_label.show()
            self._layout_overlays()
        else:
            if self._probe_a is None:
                self._remove(ACTOR_PROBE_LINE)
                self._remove(ACTOR_PROBE_B)
                self._probe_a = p
                self._marker(ACTOR_PROBE, p, PALETTE.warning)
                self.status_message.emit("Line probe: now click the end point.")
            else:
                a, b = self._probe_a, p
                self._probe_a = None
                self._marker(ACTOR_PROBE_B, b, PALETTE.warning)
                self.plotter.add_mesh(pv.Line(tuple(a), tuple(b)), name=ACTOR_PROBE_LINE, color=PALETTE.warning,
                                      line_width=3, lighting=False, pickable=False, reset_camera=False)
                n = 300
                t = np.linspace(0.0, 1.0, n)
                pts = a[None, :] + t[:, None] * (b - a)[None, :]
                s = self._sample(pts)
                s["distance"] = t * float(np.linalg.norm(b - a))
                s["a"], s["b"] = a, b
                s["v_inf"] = np.array([self._result.field.freestream_ms if self._result is not None else 1.0])
                self.probe_line_ready.emit(s)
        self.plotter.render()

    # -- export / camera ------------------------------------------------------
    def export_vtk(self, path: str | Path) -> Path:
        """Save the flow field, the surface (with Cp) and the traced streamlines as one VTK multiblock
        (``.vtm``, opens in ParaView with each part as its own block)."""
        if self._dataset is None:
            raise RuntimeError("There is no flow result to export yet.")
        out = Path(path).with_suffix(".vtm")
        out.parent.mkdir(parents=True, exist_ok=True)
        mb = pv.MultiBlock()
        mb["flow_field"] = self._dataset
        if self._surface is not None:
            mb["surface"] = self._surface
        elif self._mesh is not None:
            mb["surface"] = self._mesh
        if self._lines is not None and self._lines.n_points:
            mb["streamlines"] = self._lines
        mb.save(str(out))
        log.info("Flow field exported: %s", out)
        return out

    def camera_state(self) -> list[list[float]]:
        """Camera position, focal point and view-up (JSON-friendly)."""
        return [list(map(float, v)) for v in self.plotter.camera_position]

    def set_camera_state(self, state: Any) -> None:
        try:
            self.plotter.camera_position = [tuple(float(x) for x in v) for v in state]
            self.plotter.render()
        except Exception as exc:  # noqa: BLE001
            log.debug("Ignoring an invalid camera bookmark: %s", exc)

    def _build_glyphs(self, origin: np.ndarray, normal: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                      v_inf: float) -> None:
        n = int(self.settings.vector_density)
        ax = int(np.argmax(np.abs(normal)))
        inplane = [i for i in range(3) if i != ax]
        size = float(max(hi[inplane[0]] - lo[inplane[0]], hi[inplane[1]] - lo[inplane[1]]))
        plane = pv.Plane(center=tuple(origin), direction=tuple(normal), i_size=size, j_size=size,
                         i_resolution=n * 2, j_resolution=n * 2)
        sampled = plane.sample(self._dataset)
        if "velocity" not in sampled.point_data:
            return
        vel = np.asarray(sampled.point_data["velocity"], dtype=float)
        speed = np.linalg.norm(vel, axis=1)
        mask = speed > 0.02 * v_inf
        if "vtkValidPointMask" in sampled.point_data:
            mask &= np.asarray(sampled.point_data["vtkValidPointMask"]) > 0
        if not mask.any():
            return
        cloud = pv.PolyData(np.asarray(sampled.points)[mask])
        cloud.point_data["velocity"] = vel[mask]
        cloud.point_data["speed"] = speed[mask]
        factor = 0.9 * size / (2 * n) / max(v_inf, 1e-9)
        # Low-poly arrows: one third of the triangles (faster to build and to draw), same look at slice scale.
        arrows = cloud.glyph(orient="velocity", scale="speed", factor=factor,
                             geom=pv.Arrow(tip_resolution=8, shaft_resolution=6))
        self.plotter.add_mesh(arrows, name=ACTOR_GLYPHS, scalars="speed", cmap=self.settings.colormap,
                              clim=(0.0, 1.6 * v_inf), show_scalar_bar=False, reset_camera=False)

    def _build_iso(self) -> None:
        """3-D isosurface: vortex cores (Q-criterion), a speed shell, or the viscous wake."""
        self._remove(ACTOR_ISO)
        s = self.settings
        ds = self._dataset
        if not s.show_iso or ds is None:
            return
        v_inf = self._result.field.freestream_ms if self._result is not None else 1.0
        kind = IsoField(s.iso_field)
        pd = ds.point_data
        if kind is IsoField.QCRITERION:
            if "qcriterion" not in pd:
                self.status_message.emit("This result has no vortex-core (Q-criterion) data.")
                return
            q = np.asarray(pd["qcriterion"], dtype=float)
            pos = q[q > 0.0]
            if pos.size < 20:
                self.status_message.emit("No significant vortex cores in this flow (try a higher angle of attack).")
                return
            array, level = "qcriterion", float(np.percentile(pos, 50.0 + 49.5 * s.iso_level))
            opacity, what = 0.85, "vortex cores"
        elif kind is IsoField.LAMBDA2:
            if "lambda2" not in pd:
                self.status_message.emit("This result has no λ2 data (re-run the solve to compute it).")
                return
            l2 = np.asarray(pd["lambda2"], dtype=float)
            if "solid" in pd:
                l2 = l2[np.asarray(pd["solid"]) == 0]
            neg = l2[l2 < 0.0]
            if neg.size < 20:
                self.status_message.emit("No significant vortex cores in this flow (try a higher angle of attack).")
                return
            # λ2 < 0 inside cores; the slider picks how strong a core must be to be drawn.
            array, level = "lambda2", float(np.percentile(neg, 50.0 - 49.5 * s.iso_level))
            opacity, what = 0.85, "vortex cores (λ2)"
        elif kind is IsoField.SPEED:
            array, level = "speed", float(v_inf * (0.5 + s.iso_level))
            opacity, what = 0.35, f"speed = {level:.1f} m/s"
        else:
            if "Cp0" not in pd:
                self.status_message.emit("This result has no total-pressure data.")
                return
            loss = 1.0 - np.asarray(pd["Cp0"], dtype=float)
            if "solid" in pd:
                loss = loss[np.asarray(pd["solid"]) == 0]
            l_max = float(np.percentile(loss, 99.9)) if loss.size else 0.0
            if l_max < 1e-3:
                self.status_message.emit("This flow has no measurable wake loss.")
                return
            # Relative to this result's strongest loss: a wing's wake loses only a few per cent,
            # a bluff body's most of its total pressure.
            threshold = l_max * (0.1 + 0.85 * s.iso_level)
            array, level = "Cp0", float(1.0 - threshold)
            opacity, what = 0.45, f"total-pressure loss {threshold:.3f}"
        surf = ds.contour([level], scalars=array)
        if surf.n_points == 0:
            self.status_message.emit(f"No {what} surface at this level — move the Iso level slider.")
            return
        try:
            surf = surf.smooth(n_iter=15, relaxation=0.1)        # take the voxel stair-steps off
        except Exception:
            pass
        self.plotter.add_mesh(surf, name=ACTOR_ISO, color=ISO_COLORS[kind], opacity=opacity, smooth_shading=True,
                              specular=0.4, specular_power=25, show_scalar_bar=False, reset_camera=False,
                              pickable=False)

    def _build_volume(self) -> None:
        """GPU volume rendering of vorticity: swirling air glows like a cloud (trailing / tip vortices)."""
        self._remove(ACTOR_VOLUME)
        self._set_volume_on(False)
        ds = self._dataset
        if not self.settings.show_volume or ds is None:
            return
        if not isinstance(ds, pv.ImageData) or "vorticity_mag" not in ds.point_data:
            self.status_message.emit("The vortex cloud needs a structured Instant-engine result.")
            return
        w = np.asarray(ds.point_data["vorticity_mag"], dtype=np.float32).copy()
        if "solid" in ds.point_data:
            w[np.asarray(ds.point_data["solid"]) > 0] = 0.0
        fluid = w[w > 0]
        if fluid.size < 100:
            return
        lo, hi = float(np.percentile(fluid, 80.0)), float(np.percentile(fluid, 99.7))
        if hi <= lo:
            return
        vol = pv.ImageData(dimensions=ds.dimensions, spacing=ds.spacing, origin=ds.origin)
        vol.point_data["swirl"] = w
        self._set_volume_on(True)                     # SSAO off before the volume is ever rendered
        try:
            self.plotter.add_volume(vol, scalars="swirl", name=ACTOR_VOLUME, cmap="plasma", clim=(lo, hi),
                                    opacity=[0.0, 0.0, 0.04, 0.12, 0.28, 0.5], shade=False,
                                    show_scalar_bar=False, mapper="smart", reset_camera=False)
        except Exception as exc:  # noqa: BLE001 - very old GPUs lack 3-D textures
            self._set_volume_on(False)
            self.status_message.emit(f"Volume rendering is not available on this GPU: {exc}")

    def _set_volume_on(self, on: bool) -> None:
        if on != self._volume_on:
            self._volume_on = on
            self._apply_ssao(self.settings.ambient_occlusion)

    # -- animation -----------------------------------------------------------
    def _start_animation(self) -> None:
        if self._particles is not None and not self._closed and not self._anim_timer.isActive():
            self._last_tick = time.perf_counter()
            self._fps_t0 = self._last_tick
            self._fps_frames = 0
            if not self._ssao_suspended:
                self._ssao_suspended = True          # heavy still-image effects off while animating
                self._set_depth_peeling(False)
                if self.settings.ambient_occlusion:
                    self._apply_ssao(self.settings.ambient_occlusion)
            self._anim_timer.start()

    def _stop_animation(self) -> None:
        if self._anim_timer.isActive():
            self._anim_timer.stop()
        if self._ssao_suspended:
            self._ssao_suspended = False
            if not self._closed:
                self._set_depth_peeling(True)
                self._apply_ssao(self.settings.ambient_occlusion)

    def _animate(self) -> None:
        if self._particles is None or self._particle_poly is None or self._closed:
            self._stop_animation()
            return
        now = time.perf_counter()
        dt = min(now - self._last_tick, 0.1)
        self._last_tick = now
        try:
            self._particle_poly.points = self._particles.positions(dt, self.settings.particle_speed)
            self.plotter.render()
        except Exception as exc:
            log.error("Particle animation stopped: %s", exc)
            self._stop_animation()
            return
        self._fps_frames += 1
        if now - self._fps_t0 >= 1.0:
            self.fps_changed.emit(self._fps_frames / (now - self._fps_t0))
            self._fps_frames = 0
            self._fps_t0 = now

    def set_animation_paused(self, paused: bool) -> None:
        """Pause/resume the tracer animation without discarding it."""
        self._anim_paused = paused
        if paused:
            self._stop_animation()
        else:
            self._start_animation()

    # -- camera ----------------------------------------------------------------
    def _on_view(self, key: str) -> None:
        if key == "screenshot":
            self.screenshot_requested.emit()   # the main window opens the save dialog
            return
        self.view(key)

    def view(self, key: str) -> None:
        """Apply a named camera preset: ``iso``, ``side``, ``top``, ``front`` or ``reset``."""
        p = self.plotter
        try:
            if key == "iso":
                p.view_vector((-1.0, -1.1, 0.75), viewup=(0, 0, 1))
            elif key == "side":
                p.view_vector((0.0, -1.0, 0.0), viewup=(0, 0, 1))
            elif key == "top":
                p.view_vector((0.0, 0.0, 1.0), viewup=(-1, 0, 0))
            elif key == "front":
                p.view_vector((-1.0, 0.0, 0.0), viewup=(0, 0, 1))
            if self._mesh is not None:
                b = np.asarray(self._mesh.bounds, dtype=float)
                pad = 0.15 * self._ref_length()
                p.reset_camera(bounds=(b[0] - pad, b[1] + pad, b[2] - pad, b[3] + pad, b[4] - pad, b[5] + pad))
            else:
                p.reset_camera()
            p.render()
        except Exception as exc:
            log.warning("Camera change failed: %s", exc)

    # -- export ----------------------------------------------------------------
    def save_screenshot(self, path: str | Path, target_width: int = 3840, transparent: bool = False) -> Path:
        """Save a high-resolution PNG (rendered at an integer up-scale of the window to ≈4K).

        ``transparent``: no background and no floor - for slides and papers."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        width = max(self.plotter.width(), 1)
        scale = max(1, int(math.ceil(target_width / width)))
        if transparent:
            self._transparent_screenshot(out, scale)
        else:
            try:
                self.plotter.screenshot(str(out), scale=scale, transparent_background=False)
            except TypeError:
                self.plotter.screenshot(str(out), transparent_background=False)
        log.info("Screenshot saved: %s", out)
        self.screenshot_saved.emit(str(out))
        return out

    def _transparent_screenshot(self, out: Path, scale: int) -> None:
        """PNG with a real alpha channel, by difference matting.

        The Qt OpenGL surface has no alpha channel, so VTK's ``transparent_background`` cannot work here.
        Instead the frame is rendered twice, on black and on white: where they agree the pixel is opaque;
        where they differ, ``alpha = 1 - (white - black)`` and the colour is recovered from the black
        render. This is exact for opaque and semi-transparent layers (slices, isosurfaces) alike."""
        from PIL import Image

        ren = self.plotter.renderer
        gradient = bool(ren.GetGradientBackground())
        bg = tuple(ren.GetBackground())
        hidden = []
        for name in (ACTOR_FLOOR, ACTOR_FLOOR_GRID):
            actor = self.plotter.actors.get(name)
            if actor is not None and actor.GetVisibility():
                actor.SetVisibility(False)
                hidden.append(actor)
        ssao = self._ssao_on
        if ssao:                                      # SSAO darkens the background differently per colour
            self.plotter.disable_ssao()
            self._ssao_on = False
        shots = []
        try:
            ren.SetGradientBackground(False)
            for colour in ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)):
                ren.SetBackground(*colour)
                self.plotter.render()                 # screenshot() alone may reuse the last frame
                try:
                    img = self.plotter.screenshot(None, scale=scale, return_img=True)
                except TypeError:
                    img = self.plotter.screenshot(None, return_img=True)
                shots.append(np.asarray(img)[..., :3].astype(np.float32) / 255.0)
        finally:
            ren.SetBackground(*bg)
            ren.SetGradientBackground(gradient)
            for actor in hidden:
                actor.SetVisibility(True)
            if ssao:
                self._apply_ssao(self.settings.ambient_occlusion)
            self.plotter.render()
        black, white = shots
        alpha = np.clip(1.0 - (white - black).mean(axis=2), 0.0, 1.0)
        rgb = np.where(alpha[..., None] > 1e-3, black / np.maximum(alpha[..., None], 1e-3), 0.0)
        rgba = np.dstack([np.clip(rgb, 0.0, 1.0), alpha])
        Image.fromarray((rgba * 255.0 + 0.5).astype(np.uint8), "RGBA").save(str(out))

    def grab_frame(self) -> np.ndarray:
        """Current frame as an RGB uint8 array."""
        self.plotter.render()
        img = self.plotter.screenshot(None, return_img=True)
        return np.asarray(img)[..., :3]

    def record_gif(self, path: str | Path, frames: int = 72, fps: int = 24, rotate: bool = True,
                   progress: Optional[Any] = None) -> Path:
        """Record a rotating (and, if enabled, flowing) animation to GIF."""
        from PIL import Image
        from PySide6.QtWidgets import QApplication

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        was_running = self._anim_timer.isActive()
        self._stop_animation()
        images: list[np.ndarray] = []
        cam = self.plotter.camera
        try:
            for i in range(frames):
                if rotate:
                    cam.azimuth = cam.azimuth + 360.0 / frames
                if self._particles is not None and self._particle_poly is not None:
                    self._particle_poly.points = self._particles.positions(1.0 / fps, self.settings.particle_speed)
                images.append(self.grab_frame())
                if progress:
                    progress(int(100 * (i + 1) / frames))
                QApplication.processEvents()
            if out.suffix.lower() == ".mp4":
                self._write_mp4(out, images, fps)
                log.info("Video saved: %s", out)
                return out
            h, w = images[0].shape[:2]
            pil = [Image.fromarray(np.ascontiguousarray(im)) for im in images]
            if w > 960:  # keep GIFs shareable
                size = (960, max(1, int(h * 960 / w)))
                pil = [im.resize(size, Image.LANCZOS) for im in pil]
            # One palette shared by every frame (built from a sample of frames), so colours do not flicker.
            sample = pil[::max(1, len(pil) // 6)][:6]
            sw, sh = sample[0].size
            montage = Image.new("RGB", (sw, sh * len(sample)))
            for k, im in enumerate(sample):
                montage.paste(im, (0, k * sh))
            palette_img = montage.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
            pal = [im.quantize(palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG) for im in pil]
            pal[0].save(str(out), save_all=True, append_images=pal[1:], duration=int(round(1000 / fps)),
                        loop=0, optimize=False, disposal=2)
        finally:
            if was_running:
                self._start_animation()
        log.info("Animation saved: %s", out)
        return out

    @staticmethod
    def _write_mp4(out: Path, images: list[np.ndarray], fps: int) -> None:
        """H.264 MP4 (plays everywhere; far smaller and sharper than a GIF). Needs imageio-ffmpeg."""
        try:
            import imageio.v2 as iio
        except ImportError as exc:
            raise RuntimeError("MP4 export needs imageio (pip install imageio imageio-ffmpeg).") from exc
        h, w = images[0].shape[:2]
        h2, w2 = h - h % 2, w - w % 2              # H.264 with yuv420p needs even dimensions
        try:
            writer = iio.get_writer(str(out), fps=fps, codec="libx264", quality=8, pixelformat="yuv420p",
                                    macro_block_size=1)
        except Exception as exc:  # noqa: BLE001 - imageio-ffmpeg missing
            raise RuntimeError("MP4 export needs the imageio-ffmpeg package (run setup_windows.bat again, or "
                               "pip install imageio-ffmpeg).") from exc
        try:
            for im in images:
                writer.append_data(np.ascontiguousarray(im[:h2, :w2]))
        finally:
            writer.close()

    # -- Qt events -------------------------------------------------------------
    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._layout_overlays()

    def showEvent(self, event: Any) -> None:  # noqa: N802
        super().showEvent(event)
        self._layout_overlays()

    def _layout_overlays(self) -> None:
        self.hud.adjustSize()
        self.hud.move(14, 14)
        self.camera_bar.adjustSize()
        bw, bh = self.camera_bar.width(), self.camera_bar.height()
        self.camera_bar.move(max(0, (self.width() - bw) // 2), max(0, self.height() - bh - 16))
        self.drop_overlay.setGeometry(self.rect().adjusted(8, 8, -8, -8))
        self.probe_label.adjustSize()
        self.probe_label.move(14, max(0, self.height() - self.probe_label.height() - bh - 28))
        for w in (self.hud, self.camera_bar, self.probe_label, self.drop_overlay):
            w.raise_()

    @staticmethod
    def _mesh_path_from(event: QDragEnterEvent | QDropEvent) -> Optional[str]:
        md = event.mimeData()
        if not md.hasUrls():
            return None
        for url in md.urls():
            if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in SUPPORTED_MESH_EXTENSIONS:
                return url.toLocalFile()
        return None

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if self._mesh_path_from(event):
            event.acceptProposedAction()
            self.drop_overlay.show()
            self.drop_overlay.raise_()
        else:
            event.ignore()

    def dragMoveEvent(self, event: Any) -> None:  # noqa: N802
        event.acceptProposedAction()

    def dragLeaveEvent(self, event: Any) -> None:  # noqa: N802
        self.drop_overlay.hide()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        self.drop_overlay.hide()
        path = self._mesh_path_from(event)
        if path:
            event.acceptProposedAction()
            self.file_dropped.emit(path)
        else:
            event.ignore()

    # -- shutdown ----------------------------------------------------------------
    def shutdown(self) -> None:
        """Stop timers and release VTK resources (call from the main window's closeEvent)."""
        if self._closed:
            return
        self._closed = True
        self._rebuild_timer.stop()
        self._stop_animation()
        try:
            self.plotter.close()
        except Exception as exc:
            log.debug("Plotter close: %s", exc)
