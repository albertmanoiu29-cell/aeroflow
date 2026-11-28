"""The Supersonic Lab: a separate window for detailed compressible-flow (Mach > 1) simulation.

Left: controls (Mach number, body, angles, resolution, view, overlays). Centre: the live solution
(:class:`visualization.supersonic_view.SupersonicView`). Right: analysis tabs —

* **Theory vs simulation**: exact gas-dynamics results (normal / oblique shock, Prandtl-Meyer,
  isentropic stagnation values, linear airfoil theory) next to what the solver measures.
* **Line probe**: p/p∞ and Mach number along a horizontal line through the body (the jumps are the
  shocks; dashed lines are the theoretical post-shock levels).
* **Surface**: pressure along the body surface.
* **θ-β-M chart**: the classic oblique-shock design chart with the current case marked.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QSlider,
    QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from config.settings import APP_NAME, get_paths
from config.theme import PALETTE, make_logo_icon, mono_font
from core import gasdynamics as gd
from core.geometry_manager import GeometryModel
from ui.cfd_panel import AccordionCard, ToggleSwitch, button, compact_combo, labeled_row, scroll_wrap
from ui.device_selector import DeviceSelector
from ui.widgets2d import AirfoilFileRow
from utils.validators import validate_naca_code
from visualization.supersonic_view import (
    BLUNT_BODIES, DEFAULT_SUPER_RESOLUTION, SUPER_RESOLUTIONS, SBody, SField, SupersonicView,
)
from visualization.telemetry_charts import _card, _style_plot


class ValueSlider(QWidget):
    """Labelled slider with a numeric readout (linear)."""

    def __init__(self, label: str, lo: float, hi: float, value: float, step: float, unit: str = "",
                 decimals: int = 2, tooltip: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.lo, self.hi, self.step, self.decimals, self.unit = lo, hi, step, decimals, unit
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setVerticalSpacing(2)
        cap = QLabel(label)
        cap.setToolTip(tooltip)
        self.val = QLabel("")
        self.val.setFont(mono_font(12))
        self.val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        n = int(round((hi - lo) / step))
        self.slider.setRange(0, n)
        self.slider.setToolTip(tooltip)
        lay.addWidget(cap, 0, 0)
        lay.addWidget(self.val, 0, 1)
        lay.addWidget(self.slider, 1, 0, 1, 2)
        self.slider.valueChanged.connect(lambda _v: self._show())
        self.set_value(value)

    def value(self) -> float:
        return self.lo + self.slider.value() * self.step

    def set_value(self, v: float) -> None:
        self.slider.setValue(int(round((min(max(v, self.lo), self.hi) - self.lo) / self.step)))
        self._show()

    def _show(self) -> None:
        self.val.setText(f"{self.value():.{self.decimals}f} {self.unit}".rstrip())


class SupersonicWindow(QMainWindow):
    """Separate top-level window for the Supersonic Lab."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Supersonic Lab — {APP_NAME}")
        self.setWindowIcon(make_logo_icon(64))
        self.resize(1560, 920)
        self.view = SupersonicView()
        self._last_stats: dict = {}

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scroll_wrap(self._controls()))
        splitter.addWidget(self.view)
        splitter.addWidget(self._analysis())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([370, 820, 420])
        self.setCentralWidget(splitter)
        self.status = self.statusBar()
        self.status_label = QLabel("")
        self.status_label.setProperty("role", "muted")
        self.status.addWidget(self.status_label, 1)
        self.fps_label = QLabel("— FPS")
        self.fps_label.setFont(mono_font(11))
        self.status.addPermanentWidget(self.fps_label)
        self.view.stats_changed.connect(self._on_stats)
        self.view.fps_changed.connect(lambda f: self.fps_label.setText(f"{f:5.1f} FPS"))
        self.view.status_message.connect(self.status_label.setText)
        self._sync_body_ui()
        self._refresh_theory_chart()

    # -- controls ----------------------------------------------------------------
    def _controls(self) -> QWidget:
        root = QWidget()
        root.setObjectName("Root")
        lay = QVBoxLayout(root)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        sim = AccordionCard("Simulation")
        intro = QLabel("Compressible (Euler) flow solved live: shock waves, expansion fans and bow shocks form "
                       "from the physics. Results are compared with exact gas-dynamics theory on the right.")
        intro.setWordWrap(True)
        intro.setProperty("role", "muted")
        sim.add(intro)
        row = QHBoxLayout()
        self.btn_run = button("Run", "primary", "play", "Start / pause the simulation")
        self.btn_step = button("Step", "ghost", "", "Advance a few steps while paused")
        self.btn_reset = button("Restart", "", "reset", "Restart from uniform flow")
        row.addWidget(self.btn_run, 2)
        row.addWidget(self.btn_step, 1)
        row.addWidget(self.btn_reset, 2)
        sim.add_layout(row)
        self.btn_run.clicked.connect(self.toggle_run)
        self.btn_step.clicked.connect(self.view.step_once)
        self.btn_reset.clicked.connect(self.view.restart)
        self.speed = ValueSlider("Simulation speed", 0.25, 4.0, 1.0, 0.25, "x", 2,
                                 "Steps per frame. Higher converges faster but lowers the frame rate.")
        self.speed.slider.valueChanged.connect(lambda _v: self.view.set_speed(self.speed.value()))
        sim.add(self.speed)
        self.resolution = QComboBox()
        for name in SUPER_RESOLUTIONS:
            self.resolution.addItem(name)
        self.resolution.setCurrentText(DEFAULT_SUPER_RESOLUTION)
        self.resolution.currentTextChanged.connect(self.view.set_resolution)
        sim.add(labeled_row("Grid", self.resolution, "Finer grids give sharper shocks but run slower."))
        self.device = DeviceSelector()
        sim.add(self.device)
        self.precision = QComboBox()
        self.precision.addItem("Single (fast)", True)
        self.precision.addItem("Double (exact)", False)
        self.precision.currentIndexChanged.connect(
            lambda _i: self.view.set_gpu_precision(bool(self.precision.currentData())))
        sim.add(labeled_row("GPU precision", self.precision,
                            "Single precision runs several times faster on consumer GPUs (they have few "
                            "double-precision units); shock positions and pressures agree with double precision "
                            "to about 0.1 %. Double precision matches the CPU solver exactly."))
        snap = button("Save snapshot…", "ghost", "camera", "Save the current view as PNG")
        snap.clicked.connect(self._snapshot)
        self.btn_rec = button("Record video…", "ghost", "camera", "Record the running simulation to an MP4 video")
        self.btn_rec.setCheckable(True)
        self.btn_rec.toggled.connect(self._toggle_recording)
        row_media = QHBoxLayout()
        row_media.addWidget(snap)
        row_media.addWidget(self.btn_rec)
        sim.add_layout(row_media)
        lay.addWidget(sim)

        flow = AccordionCard("Free stream")
        self.mach = ValueSlider("Mach number M∞", 0.5, 6.0, 2.0, 0.05, "", 2,
                                "Speed relative to the speed of sound. Above 1: supersonic (shocks); "
                                "0.8-1.2: transonic; above 5: hypersonic.")
        self.mach.slider.valueChanged.connect(lambda _v: self._mach_changed())
        flow.add(self.mach)
        presets = QHBoxLayout()
        for m in (0.8, 1.5, 2.0, 3.0, 5.0):
            b = button(f"M {m:g}", "ghost", tooltip=f"Set M∞ = {m:g}")
            b.clicked.connect(lambda _=False, v=m: self.mach.set_value(v))
            presets.addWidget(b)
        flow.add_layout(presets)
        self.regime = QLabel("")
        self.regime.setWordWrap(True)
        self.regime.setProperty("role", "muted")
        flow.add(self.regime)
        lay.addWidget(flow)

        body = AccordionCard("Body")
        self.body = QComboBox()
        for b_ in SBody:
            self.body.addItem(b_.value, b_)
        self.body.currentIndexChanged.connect(self._body_changed)
        body.add(self.body)
        self.half = ValueSlider("Half-angle θ", 2.0, 35.0, 10.0, 0.5, "°", 1,
                                "Wedge / diamond half-angle: how sharply the surface turns the flow. Beyond θmax "
                                "the shock detaches from the nose.")
        self.half.slider.sliderReleased.connect(self._geometry_changed)
        self.half.slider.valueChanged.connect(lambda _v: self._geometry_changed() if not self.half.slider.isSliderDown()
                                              else None)
        body.add(self.half)
        self.aoa = ValueSlider("Angle of attack α", -15.0, 15.0, 0.0, 0.5, "°", 1,
                               "Nose-up angle. It strengthens the lower-surface shock and turns the upper surface "
                               "into an expansion.")
        self.aoa.slider.sliderReleased.connect(self._geometry_changed)
        self.aoa.slider.valueChanged.connect(lambda _v: self._geometry_changed() if not self.aoa.slider.isSliderDown()
                                             else None)
        body.add(self.aoa)
        self.size = ValueSlider("Size", 0.5, 1.4, 1.0, 0.05, "x", 2, "Body size relative to the tunnel")
        self.size.slider.sliderReleased.connect(self._geometry_changed)
        self.size.slider.valueChanged.connect(lambda _v: self._geometry_changed() if not self.size.slider.isSliderDown()
                                              else None)
        body.add(self.size)
        self.ramp = ValueSlider("Centre-body position", -1.0, 1.0, 0.0, 0.05, "", 2,
                                "Inlet: slide the ramp forward (the shock falls ahead of the cowl lip: spillage) or "
                                "back (the shock is swallowed into the duct). 0 = shock on the lip at this Mach.")
        self.ramp.slider.sliderReleased.connect(self._geometry_changed)
        self.ramp.slider.valueChanged.connect(lambda _v: self._geometry_changed() if not self.ramp.slider.isSliderDown()
                                              else None)
        body.add(self.ramp)
        self.naca_row = QWidget()
        h = QHBoxLayout(self.naca_row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel("NACA"))
        self.naca = QLineEdit("0012")
        self.naca.setMaxLength(4)
        self.naca.setFont(mono_font(12))
        self.naca.editingFinished.connect(self._geometry_changed)
        h.addWidget(self.naca, 1)
        body.add(self.naca_row)
        self.file_row = AirfoilFileRow(self._load_airfoil_file)
        body.add(self.file_row)
        lay.addWidget(body)

        disp = AccordionCard("Display")
        self.field = QComboBox()
        for f in SField:
            self.field.addItem(f.value, f)
        self.field.currentIndexChanged.connect(lambda i: self.view.set_field(list(SField)[i]))
        disp.add(labeled_row("Show", self.field))
        self.cmap = QComboBox()
        for c in ("turbo", "viridis", "plasma", "inferno", "cividis"):
            self.cmap.addItem(c)
        self.cmap.currentTextChanged.connect(self.view.set_colormap)
        disp.add(labeled_row("Mach colours", self.cmap))
        grid = QGridLayout()
        self.t_sonic = ToggleSwitch("Sonic line", True, "White contour where the local Mach number is 1")
        self.t_shock = ToggleSwitch("Shock cells", False, "Highlight cells detected inside shock waves")
        self.t_theory = ToggleSwitch("Theory lines", True,
                                     "Dotted cyan lines: exact oblique-shock angles from theory (sharp bodies)")
        grid.addWidget(self.t_sonic, 0, 0)
        grid.addWidget(self.t_shock, 0, 1)
        grid.addWidget(self.t_theory, 1, 0)
        for t in (self.t_sonic, self.t_shock, self.t_theory):
            t.toggled.connect(lambda _c: self.view.set_overlays(self.t_sonic.isChecked(), self.t_shock.isChecked(),
                                                                self.t_theory.isChecked()))
        disp.add_layout(grid)
        self.probe_row = ValueSlider("Line probe height", 0.05, 0.95, 0.5, 0.01, "", 2,
                                     "Height (fraction of the tunnel) of the probe line on the right")
        self.probe_row.slider.valueChanged.connect(lambda _v: self._refresh_probe())
        disp.add(self.probe_row)
        lay.addWidget(disp)
        lay.addStretch(1)
        for combo in root.findChildren(QComboBox):
            compact_combo(combo)
        self._mach_changed()
        return root

    # -- analysis ------------------------------------------------------------------
    def _analysis(self) -> QWidget:
        tabs = QTabWidget()
        self.tabs = tabs
        # Theory table
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Quantity", "Theory", "Simulation"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setFont(mono_font(11))
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        theory = QWidget()
        tl = QVBoxLayout(theory)
        tl.setContentsMargins(6, 6, 6, 6)
        self.theory_note = QLabel("")
        self.theory_note.setWordWrap(True)
        self.theory_note.setProperty("role", "muted")
        tl.addWidget(self.table, 1)
        tl.addWidget(self.theory_note)
        tabs.addTab(theory, "Theory vs sim")
        # Line probe
        self.probe_plot = pg.PlotWidget()
        _style_plot(self.probe_plot, "x (cells, flow →)", "value")
        self.probe_plot.addLegend(offset=(-8, 8), labelTextColor=PALETTE.text_muted,
                                  brush=pg.mkBrush(PALETTE.surface_alt + "B0"))
        self.c_p = self.probe_plot.plot(name="p/p∞", pen=pg.mkPen(PALETTE.warning, width=2))
        self.c_m = self.probe_plot.plot(name="Mach", pen=pg.mkPen(PALETTE.cyan, width=2))
        self.l_ns = pg.InfiniteLine(angle=0, pen=pg.mkPen(PALETTE.warning, width=1, style=Qt.PenStyle.DashLine))
        self.l_sonic = pg.InfiniteLine(pos=1.0, angle=0, pen=pg.mkPen("#FFFFFF", width=1, style=Qt.PenStyle.DotLine))
        self.probe_plot.addItem(self.l_ns, ignoreBounds=True)
        self.probe_plot.addItem(self.l_sonic, ignoreBounds=True)
        self.probe_info = QLabel("")
        self.probe_info.setProperty("role", "muted")
        self.probe_info.setWordWrap(True)
        pw = QWidget()
        pl = QVBoxLayout(pw)
        pl.setContentsMargins(6, 6, 6, 6)
        pl.addWidget(_card("Along the probe line", self.probe_plot), 1)
        pl.addWidget(self.probe_info)
        tabs.addTab(pw, "Line probe")
        # Surface pressure
        self.surf_plot = pg.PlotWidget()
        _style_plot(self.surf_plot, "x along the body (fraction of length)", "p/p∞ on the surface")
        self.surf_plot.addLegend(offset=(-8, 8), labelTextColor=PALETTE.text_muted,
                                 brush=pg.mkBrush(PALETTE.surface_alt + "B0"))
        self.c_up = self.surf_plot.plot(name="Upper", pen=None, symbol="o", symbolSize=4,
                                        symbolBrush=pg.mkBrush(PALETTE.cyan), symbolPen=None)
        self.c_lo = self.surf_plot.plot(name="Lower", pen=None, symbol="o", symbolSize=4,
                                        symbolBrush=pg.mkBrush(PALETTE.warning), symbolPen=None)
        self.l_up = pg.InfiniteLine(angle=0, pen=pg.mkPen(PALETTE.cyan, width=1, style=Qt.PenStyle.DashLine))
        self.l_lo = pg.InfiniteLine(angle=0, pen=pg.mkPen(PALETTE.warning, width=1, style=Qt.PenStyle.DashLine))
        self.surf_plot.addItem(self.l_up, ignoreBounds=True)
        self.surf_plot.addItem(self.l_lo, ignoreBounds=True)
        sw = QWidget()
        sl = QVBoxLayout(sw)
        sl.setContentsMargins(6, 6, 6, 6)
        sl.addWidget(_card("Surface pressure (dashed: oblique-shock theory)", self.surf_plot))
        tabs.addTab(sw, "Surface")
        # theta-beta-M
        self.tbm = pg.PlotWidget()
        _style_plot(self.tbm, "Deflection θ (deg)", "Shock angle β (deg)")
        self.tbm.addLegend(offset=(-8, -8), labelTextColor=PALETTE.text_muted,
                           brush=pg.mkBrush(PALETTE.surface_alt + "B0"))
        self.tbm_curves = []
        self.tbm_current = self.tbm.plot(pen=pg.mkPen(PALETTE.cyan, width=3))
        self.tbm_pt = pg.ScatterPlotItem(size=12, brush=pg.mkBrush("#FFFFFF"), pen=pg.mkPen(PALETTE.cyan, width=2))
        self.tbm_sim = pg.ScatterPlotItem(size=11, symbol="x", brush=pg.mkBrush(PALETTE.warning),
                                          pen=pg.mkPen(PALETTE.warning, width=2))
        self.tbm.addItem(self.tbm_pt)
        self.tbm.addItem(self.tbm_sim)
        for m, col in ((1.2, "#475569"), (1.5, "#64748B"), (2.0, "#94A3B8"), (3.0, "#A78BFA"), (5.0, "#F472B6"),
                       (10.0, "#FB923C")):
            th, be = self._tbm_curve(m)
            self.tbm.plot(th, be, pen=pg.mkPen(col, width=1.2), name=f"M {m:g}")
        tw = QWidget()
        tlay = QVBoxLayout(tw)
        tlay.setContentsMargins(6, 6, 6, 6)
        tlay.addWidget(_card("θ-β-M: shock angle vs deflection", self.tbm), 1)
        self.tbm_info = QLabel("Thick cyan: current M∞ · white dot: theory · orange ×: simulation. "
                               "Each curve's right end is θmax (beyond it the shock detaches).")
        self.tbm_info.setWordWrap(True)
        self.tbm_info.setProperty("role", "muted")
        tlay.addWidget(self.tbm_info)
        tabs.addTab(tw, "θ-β-M")
        return tabs

    @staticmethod
    def _tbm_curve(m: float) -> tuple[np.ndarray, np.ndarray]:
        betas = np.linspace(gd.mach_angle_deg(m) + 1e-3, 90.0, 240)
        thetas = np.array([gd.theta_from_beta(m, b) for b in betas])
        return thetas, betas

    # -- behaviour --------------------------------------------------------------------
    def _selected_body(self) -> SBody:
        d = self.body.currentData()
        return d if isinstance(d, SBody) else SBody(d)

    def _sync_body_ui(self) -> None:
        b = self._selected_body()
        self.half.setEnabled(b in (SBody.WEDGE, SBody.DIAMOND, SBody.INLET, SBody.BICONIC))
        self.aoa.setEnabled(b not in (SBody.CYLINDER, SBody.INLET))
        self.ramp.setVisible(b is SBody.INLET)
        self.naca_row.setVisible(b is SBody.AIRFOIL)
        self.file_row.setVisible(b is SBody.AIRFOIL_FILE)

    def _load_airfoil_file(self, path: str) -> None:
        self.view.airfoil_file = path
        self._geometry_changed()

    def _body_changed(self, *_a: object) -> None:
        self._sync_body_ui()
        if self._selected_body() is SBody.AIRFOIL_FILE and not self.view.airfoil_file:
            self.file_row.browse()
        self._apply_params()
        self.view.set_body(self._selected_body())
        self._refresh_all()

    def _apply_params(self) -> None:
        naca, _ = validate_naca_code(self.naca.text().strip())
        self.view.half_angle = self.half.value()
        self.view.aoa = self.aoa.value()
        self.view.size = self.size.value()
        self.view.ramp_shift = self.ramp.value()
        if naca is not None:
            self.view.naca = naca.code

    def restore_state(self, st: dict) -> None:
        from ui.state_io import apply_widget_state

        apply_widget_state(self, st, "_sync_body_ui")
        self.view.set_body(self._selected_body())
        self._geometry_changed()
        self._mach_changed()

    def _geometry_changed(self) -> None:
        self._apply_params()
        self.view.rebuild_body()
        self._refresh_all()

    def _mach_changed(self) -> None:
        m = self.mach.value()
        if hasattr(self, "view"):
            self.view.set_mach(m)
        if m < 0.8:
            txt = "Subsonic: pressure waves travel upstream, so the flow adjusts smoothly — no shocks."
        elif m < 1.0:
            txt = ("Transonic: the air speeds up over the body past Mach 1 (inside the dashed sonic line) and "
                   "a shock usually terminates that supersonic pocket.")
        elif m < 1.2:
            txt = "Low supersonic: strong, nearly normal shocks; blunt bodies carry a detached bow shock."
        elif m < 5.0:
            txt = (f"Supersonic: disturbances are confined to the Mach cone (μ = {gd.mach_angle_deg(m):.1f}°). "
                   "Sharp noses carry attached oblique shocks; corners expand the flow in Prandtl-Meyer fans.")
        else:
            txt = ("Hypersonic: very strong, thin shocks lie close to the body; stagnation temperature reaches "
                   f"{gd.total_temperature_ratio(m):.1f}× the free-stream value.")
        self.regime.setText(txt)
        if hasattr(self, "tbm_current"):
            self._refresh_all()

    def toggle_run(self) -> None:
        from config.theme import make_icon

        if self.view.running:
            self.view.stop()
        else:
            self.view.start()
        running = self.view.running
        self.btn_run.setText("Pause" if running else "Run")
        self.btn_run.setIcon(make_icon("pause" if running else "play", 16, PALETTE.on_accent))
        self.btn_step.setEnabled(not running)


    def _toggle_recording(self, on: bool) -> None:
        import time

        from PySide6.QtWidgets import QFileDialog

        if on:
            default = get_paths().exports / f"supersonic_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            path, _ = QFileDialog.getSaveFileName(self, "Record video", str(default), "MP4 video (*.mp4)")
            if not path:
                self.btn_rec.setChecked(False)
                return
            try:
                self.view.start_recording(path)
            except RuntimeError as exc:
                self.btn_rec.setChecked(False)
                self.status_label.setText(str(exc))
                return
            self.btn_rec.setText("Stop recording")
            self.status_label.setText(f"Recording to {path} - press Stop recording to finish.")
        else:
            out = self.view.stop_recording()
            self.btn_rec.setText("Record video…")
            if out is not None:
                self.status_label.setText(f"Video saved: {out}")

    def _snapshot(self) -> None:
        default = get_paths().exports / f"supersonic_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path, _ = QFileDialog.getSaveFileName(self, "Save snapshot", str(default), "PNG image (*.png)")
        if path:
            try:
                self.view.save_png(Path(path))
                self.status_label.setText(f"Snapshot saved: {path}")
            except OSError as exc:
                self.status_label.setText(f"Snapshot failed: {exc}")

    def set_model(self, model: Optional[GeometryModel]) -> None:
        self.view.set_model(model)

    # -- refresh -----------------------------------------------------------------------
    def _on_stats(self, stats: dict) -> None:
        self._last_stats = stats
        self._refresh_all()

    def _refresh_all(self) -> None:
        stats = self.view.measurements()
        self._refresh_table(stats)
        self._refresh_probe()
        self._refresh_surface(stats)
        self._refresh_theory_chart(stats)

    def _refresh_table(self, s: dict) -> None:
        rows: list[tuple[str, str, str]] = []
        m = s.get("mach", self.mach.value())

        def f(v: Optional[float], d: int = 3) -> str:
            return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{d}f}"

        rows.append(("Free-stream Mach M∞", f(m, 2), ""))
        if m >= 1.0:
            rows.append(("Mach angle μ (deg)", f(gd.mach_angle_deg(m), 2), ""))
            rows.append(("Prandtl-Meyer angle ν (deg)", f(gd.prandtl_meyer_deg(m), 2), ""))
        rows.append(("Stagnation T0/T∞", f(gd.total_temperature_ratio(m)), ""))
        rows.append(("Stagnation p0/p∞ (isentropic)", f(gd.total_pressure_ratio(m)), ""))
        ns = s.get("normal_shock")
        if ns is not None:
            rows.append(("Normal shock: M behind", f(ns.m2), ""))
            rows.append(("Normal shock: pressure ratio p2/p1", f(ns.p_ratio), ""))
            rows.append(("Normal shock: density ratio ρ2/ρ1", f(ns.rho_ratio), ""))
            rows.append(("Normal shock: temperature ratio T2/T1", f(ns.t_ratio), ""))
            rows.append(("Normal shock: total-pressure recovery", f(ns.p0_ratio), ""))
            if self.view.body in BLUNT_BODIES:
                rows.append(("Stagnation pressure p/p∞ (pitot)", f(ns.pitot_ratio), f(s.get("p_stag_sim"))))
        if "theta_upper" in s:
            rows.append(("Max. deflection θmax (deg)", f(s.get("theta_max"), 2), ""))
            rows.append(("Upper-surface deflection θ (deg)", f(s["theta_upper"], 2), ""))
            os_ = s.get("oblique_upper")
            if os_ is not None:
                rows.append(("Upper shock angle β (deg)", f(os_.beta_deg, 2),
                             f(s.get("beta_from_p"), 2) + " (from p)"))
                if s.get("beta_geometric") is not None:
                    rows.append(("  … β fitted to shock position (±1.5°)", f(os_.beta_deg, 2),
                                 f(s["beta_geometric"], 2)))
                rows.append(("Upper surface p/p∞", f(os_.p_ratio), f(s.get("p_upper_sim"))))
                rows.append(("Mach behind upper shock", f(os_.m2), ""))
            elif "expansion_upper" in s:
                m2, pr = s["expansion_upper"]
                rows.append(("Upper surface: expansion M2", f(m2), ""))
                rows.append(("Upper surface: expansion p2/p1", f(pr), ""))
            elif s["theta_upper"] > 0:
                rows.append(("Upper shock", "detached (bow shock)", ""))
            ol = s.get("oblique_lower")
            if ol is not None:
                rows.append(("Lower shock angle β (deg)", f(ol.beta_deg, 2), ""))
                rows.append(("Lower surface p/p∞", f(ol.p_ratio), ""))
        if "standoff_sim" in s:
            rows.append(("Bow-shock stand-off Δ/R", f(s.get("standoff_billig")) + (" (Billig)" if "standoff_billig"
                                                                                in s else ""),
                         f(s.get("standoff_sim"))))
        if "cd" in s:
            rows.append(("Wave-drag coefficient Cd", f(s.get("cd_linear"), 4) + (" (linear)" if "cd_linear" in s
                                                                               else ""), f(s.get("cd"), 4)))
            rows.append(("Lift coefficient Cl", f(s.get("cl_linear"), 4) + (" (linear)" if "cl_linear" in s
                                                                          else ""), f(s.get("cl"), 4)))
        self.table.setRowCount(len(rows))
        for r, (a, b, c) in enumerate(rows):
            for col, text in enumerate((a, b, c)):
                item = QTableWidgetItem(text)
                if col:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, col, item)
        ft = s.get("flow_throughs", 0.0)
        self.theory_note.setText(
            f"Simulation time: {ft:.1f} flow-throughs of the tunnel "
            f"({'still settling — compare once ≥ 2' if ft < 2.0 else 'steady state reached'}). "
            "Simulated values are inviscid (Euler); theory assumes a calorically perfect gas, γ = 1.4. "
            "Linear (Ackeret) theory is a thin-airfoil approximation, shown for reference. Accuracy: shock "
            "pressures and angles within ~1-2 %, stagnation pressure within ~3 %. Knife-thin edges only a few "
            "cells thick act slightly blunt, so the wave drag of thin airfoils is over-predicted on coarse "
            "grids — use the Detailed or Ultra grid for them.")

    def _refresh_probe(self) -> None:
        s = self.view.solver
        row = int(self.probe_row.value() * (s.ny - 1))
        line = s.probe_line(row)
        self.c_p.setData(line["x"], np.nan_to_num(line["p"], nan=0.0))
        self.c_m.setData(line["x"], line["M"])
        if s.mach > 1.0:
            self.l_ns.setPos(gd.normal_shock(s.mach).p_ratio)
            self.l_ns.show()
        else:
            self.l_ns.hide()
        ns_txt = (f"Dashed orange: pressure behind a normal shock at M∞ {s.mach:.2f}; " if s.mach > 1.0
                  else "Subsonic free stream: no normal-shock reference; ")
        self.probe_info.setText(f"Row {row} of {s.ny}. {ns_txt}dotted white: Mach 1. Jumps in the curves are "
                                "shocks; smooth drops are expansion fans.")

    def _refresh_surface(self, s: dict) -> None:
        sp = self.view.solver.surface_pressure()
        g = self.view.geometry
        if sp is None or g is None:
            self.c_up.setData([], [])
            self.c_lo.setData([], [])
            return
        xs = (sp["x"] - g.nose[0]) / max(g.ref_length, 1e-9)
        up = sp["ny"] > 0
        self.c_up.setData(xs[up], sp["p"][up])
        self.c_lo.setData(xs[~up], sp["p"][~up])
        ou, ol = s.get("oblique_upper"), s.get("oblique_lower")
        self.l_up.setVisible(ou is not None)
        self.l_lo.setVisible(ol is not None)
        if ou is not None:
            self.l_up.setPos(ou.p_ratio)
        if ol is not None:
            self.l_lo.setPos(ol.p_ratio)

    def _refresh_theory_chart(self, s: Optional[dict] = None) -> None:
        m = self.mach.value()
        if m <= 1.0:
            self.tbm_current.setData([], [])
            self.tbm_pt.setData([], [])
            self.tbm_sim.setData([], [])
            return
        th, be = self._tbm_curve(m)
        self.tbm_current.setData(th, be)
        s = s or {}
        os_ = s.get("oblique_upper")
        if os_ is not None:
            self.tbm_pt.setData([os_.theta_deg], [os_.beta_deg])
            bs = s.get("beta_from_p")
            if bs is not None:
                self.tbm_sim.setData([os_.theta_deg], [bs])
            else:
                self.tbm_sim.setData([], [])
        else:
            self.tbm_pt.setData([], [])
            self.tbm_sim.setData([], [])

    # -- window events ---------------------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._refresh_all()

    def pause(self) -> None:
        """Stop the simulation (the state is kept)."""
        self.view.stop()
        self.btn_run.setText("Run")
        self.btn_step.setEnabled(True)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Closing only hides the window (the state is kept); the simulation pauses."""
        self.pause()
        event.accept()

    def shutdown(self) -> None:
        self.view.shutdown()
