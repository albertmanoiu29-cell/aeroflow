"""The Rocket Nozzle Lab: a separate window for live rocket-engine nozzle and exhaust-plume simulation.

Left: every input is manual: engine control (ignite, shut down), presets, propellant (γ, molar mass,
chamber temperature), chamber pressure or mass flow, the full nozzle geometry (throat, exit, chamber,
contour, angles, lip, vehicle body), the environment (altitude or ambient pressure and temperature, flight
Mach) and the simulation (grid, plume region, device). Centre: the live solution
(:class:`visualization.nozzle_view.NozzleView`). Right: analysis tabs:

* **Performance**: 1-D rocket-nozzle theory next to the simulated thrust, Isp, mass flow, exit state,
  regime, separation and shock-structure measurements.
* **Live**: thrust, chamber pressure and mass flow against time, from ignition.
* **Centre line**, **Exit plane**, **Wall**: profiles through the plume, across the exit and along the
  nozzle wall, with the theoretical values.
* **Altitude**: thrust and Isp from sea level to near vacuum, with separation and the design altitude.
* **Nozzle**: the contour, to scale, with its dimensions.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QGuiApplication
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QSplitter,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from config.settings import APP_NAME, ORG_NAME, get_paths, isa_atmosphere
from config.theme import PALETTE, make_icon, make_logo_icon, mono_font
from core import nozzle as nz
from core import nozzle_design as nd
from ui.cfd_panel import AccordionCard, SegmentedControl, ToggleSwitch, button, compact_combo, labeled_row, scroll_wrap
from ui.device_selector import DeviceSelector
from ui.nozzle_tools import (
    CompareTab, DesignTab, HeatTab, NoiseTab, altitude_curve, build_report_html, plot_image, widget_image, write_report,
)
from ui.supersonic_window import ValueSlider
from visualization.nozzle_view import (
    FLAMES, NOZZLE_RESOLUTIONS, EngineSetup, NField, NozzleView, default_resolution, flame_for, fmt_force,
)
from visualization.telemetry_charts import _card as _plain_card
from visualization.telemetry_charts import _style_plot

B = nz.Contour.BELL
C = nz.Contour.CONICAL
S = nz.Contour.SONIC

#: Ready-made engines (approximate, public-domain class values). Dimensions in metres, pressure in bar.
PRESETS: dict[str, dict] = {
    "Sea-level booster · kerolox (Merlin class)": dict(
        prop="LOX / RP-1 (kerosene)", pc=97.0, alt=0.0, plume=(9.0, 3.0),
        geom=dict(throat_diameter=0.26, exit_diameter=1.04, chamber_diameter=0.42, chamber_length=0.15,
                  contour=B, wall_thickness=0.012)),
    "Methalox booster · 300 bar (Raptor class)": dict(
        prop="LOX / CH4 (methane)", pc=300.0, alt=0.0, plume=(8.0, 3.0),
        geom=dict(throat_diameter=0.22, exit_diameter=1.28, chamber_diameter=0.34, chamber_length=0.12,
                  contour=B, wall_thickness=0.012)),
    "Hydrolox core stage (RS-25 class)": dict(
        prop="LOX / LH2 (hydrogen)", pc=206.0, alt=0.0, plume=(6.0, 2.6),
        geom=dict(throat_diameter=0.262, exit_diameter=2.18, chamber_diameter=0.45, chamber_length=0.15,
                  contour=B, wall_thickness=0.02)),
    "Vacuum upper stage · hydrolox at 40 km": dict(
        prop="LOX / LH2 (hydrogen)", pc=60.0, alt=40.0, plume=(6.0, 4.0),
        geom=dict(throat_diameter=0.12, exit_diameter=1.07, chamber_diameter=0.22, chamber_length=0.08,
                  contour=B, wall_thickness=0.01)),
    "Solid rocket motor (conical)": dict(
        prop="APCP solid (AP / Al / HTPB)", pc=70.0, alt=0.0, plume=(10.0, 3.5),
        geom=dict(throat_diameter=0.15, exit_diameter=0.45, chamber_diameter=0.30, chamber_length=0.10,
                  contour=C, divergent_angle=15.0, wall_thickness=0.015)),
    "Cold-gas thruster at 35 km": dict(
        prop="Cold gas: nitrogen", pc=10.0, alt=35.0, plume=(10.0, 6.0),
        geom=dict(throat_diameter=0.004, exit_diameter=0.02, chamber_diameter=0.01, chamber_length=0.004,
                  contour=C, divergent_angle=15.0, wall_thickness=0.0008)),
    "Underexpanded sonic jet (Mach disk)": dict(
        prop="Cold gas: air", pc=20.0, alt=0.0, plume=(14.0, 6.0),
        geom=dict(throat_diameter=0.02, exit_diameter=0.02, chamber_diameter=0.05, chamber_length=0.02,
                  contour=S, wall_thickness=0.002)),
    "Overexpanded test nozzle (shock diamonds)": dict(
        prop="Cold gas: air", pc=16.0, alt=0.0, plume=(12.0, 3.0),
        geom=dict(throat_diameter=0.05, exit_diameter=0.10, chamber_diameter=0.10, chamber_length=0.03,
                  contour=C, divergent_angle=15.0, wall_thickness=0.004)),
}
DEFAULT_PRESET = "Sea-level booster · kerolox (Merlin class)"
CUSTOM = "Custom"


def _dspin(lo: float, hi: float, val: float, dec: int = 2, suffix: str = "", step: float = 1.0,
           tip: str = "") -> QDoubleSpinBox:
    sp = QDoubleSpinBox()
    sp.setRange(lo, hi)
    sp.setDecimals(dec)
    sp.setSingleStep(step)
    sp.setValue(val)
    sp.setKeyboardTracking(False)            # commit on Enter / focus-out / arrows, not on every keystroke
    sp.setAccelerated(True)
    if suffix:
        sp.setSuffix(" " + suffix)
    if tip:
        sp.setToolTip(tip)
    return sp


def _card(title: str, content: QWidget) -> QWidget:
    """A titled chart card whose title wraps, so a long title never forces the panel wider."""
    frame = _plain_card(title, content)
    from PySide6.QtWidgets import QSizePolicy

    for lbl in frame.findChildren(QLabel):
        lbl.setWordWrap(True)
        lbl.setMinimumWidth(1)
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    return frame


def _muted(text: str = "") -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setProperty("role", "muted")
    return lbl


class NozzleWindow(QMainWindow):
    """Separate top-level window for the Rocket Nozzle Lab."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Rocket Nozzle Lab — {APP_NAME}")
        self.setWindowIcon(make_logo_icon(64))
        screen = self.screen() or QGuiApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else None
        width = min(1680, int(avail.width() * 0.96)) if avail is not None else 1640
        height = min(980, int(avail.height() * 0.92)) if avail is not None else 960
        self.resize(width, height)
        self._loading = False
        self.view = NozzleView()
        self.splitter = splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        controls = self._controls()
        self.left = scroll_wrap(controls)
        # never narrower than the controls need (no clipped inputs); drag the handle to make it wider
        self.left.setMinimumWidth(controls.minimumSizeHint().width() + 18)
        splitter.addWidget(self.left)
        splitter.addWidget(self._centre())
        self.right = self._analysis()
        self.right.setMinimumWidth(260)
        splitter.addWidget(self.right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        left_w = max(self.left.minimumWidth() + 40, int(0.29 * width))
        right_w = max(320, int(0.27 * width))
        splitter.setSizes([left_w, max(width - left_w - right_w, 300), right_w])
        self.setCentralWidget(splitter)
        self.status_label = QLabel("")
        self.status_label.setProperty("role", "muted")
        self.statusBar().addWidget(self.status_label, 1)
        self.fps_label = QLabel("— FPS")
        self.fps_label.setFont(mono_font(11))
        self.statusBar().addPermanentWidget(self.fps_label)
        self.view.stats_changed.connect(self._on_stats)
        self.view.fps_changed.connect(lambda f: self.fps_label.setText(f"{f:5.1f} FPS"))
        self.view.status_message.connect(self.status_label.setText)
        self.view.geometry_changed.connect(self._refresh_static)
        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(200)
        self._apply_timer.timeout.connect(self._apply)
        self._last_refresh = 0.0
        self._restore_layout()
        self.load_preset(DEFAULT_PRESET)

    # -- centre: 2-D section or 3-D view --------------------------------------------------------
    def _centre(self) -> QWidget:
        from PySide6.QtWidgets import QStackedWidget

        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.view_mode = SegmentedControl(["2-D section", "3-D view"],
                                          ["The solution in the r-x plane, mirrored about the axis",
                                           "The solution revolved about the axis: a glowing 3-D plume or a cut-away"])
        self.view_mode.changed.connect(self._set_view_mode)
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 0, 4, 0)
        self.btn_left = button("Hide controls", "ghost", "", "Show or hide the control panel (more room for the flow view)")
        self.btn_right = button("Hide analysis", "ghost", "", "Show or hide the analysis panel (more room for the controls)")
        for b, which in ((self.btn_left, "left"), (self.btn_right, "right")):
            b.setCheckable(True)
            b.setChecked(True)
            b.toggled.connect(lambda on, w=which: self._show_panel(w, on))
        bar.addWidget(self.btn_left)
        bar.addStretch(1)
        bar.addWidget(self.view_mode)
        bar.addStretch(1)
        bar.addWidget(self.btn_right)
        lay.addLayout(bar)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.view)
        lay.addWidget(self.stack, 1)
        self.view3d = None
        return box

    def _show_panel(self, which: str, on: bool) -> None:
        panel = self.left if which == "left" else self.right
        panel.setVisible(on)
        btn = self.btn_left if which == "left" else self.btn_right
        btn.setText(("Hide " if on else "Show ") + ("controls" if which == "left" else "analysis"))

    def _settings(self) -> QSettings:
        return QSettings(ORG_NAME, APP_NAME)

    def _restore_layout(self) -> None:
        """Panel widths and visibility from the last session."""
        st = self._settings()
        state = st.value("nozzle_lab/splitter")
        if state is not None:
            self.splitter.restoreState(state)
        for b, key in ((self.btn_left, "nozzle_lab/show_controls"), (self.btn_right, "nozzle_lab/show_analysis")):
            v = st.value(key)
            if v is not None:
                b.setChecked(str(v).lower() in ("true", "1"))

    def _save_layout(self) -> None:
        st = self._settings()
        st.setValue("nozzle_lab/splitter", self.splitter.saveState())
        st.setValue("nozzle_lab/show_controls", self.btn_left.isChecked())
        st.setValue("nozzle_lab/show_analysis", self.btn_right.isChecked())

    def _set_view_mode(self, i: int) -> None:
        if i == 1 and self.view3d is None:
            try:
                from visualization.nozzle_3d import Nozzle3DView

                self.view3d = Nozzle3DView(self.view)
            except Exception as exc:  # noqa: BLE001 - no OpenGL / VTK: stay in 2-D
                self.status_label.setText(f"The 3-D view is not available: {exc}")
                self.view_mode.set_index(0)
                return
            self.stack.addWidget(self.view3d)
        self.stack.setCurrentIndex(i)
        if i == 0:
            self.view._render()

    # -- controls -------------------------------------------------------------------------------
    def _controls(self) -> QWidget:
        root = QWidget()
        root.setObjectName("Root")
        lay = QVBoxLayout(root)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)
        lay.addWidget(self._card_engine())
        lay.addWidget(self._card_presets())
        lay.addWidget(self._card_propellant())
        lay.addWidget(self._card_feed())
        lay.addWidget(self._card_nozzle())
        lay.addWidget(self._card_environment())
        lay.addWidget(self._card_simulation())
        lay.addWidget(self._card_display())
        lay.addStretch(1)
        for combo in root.findChildren(QComboBox):
            compact_combo(combo)
        return root

    def _card_engine(self) -> QWidget:
        card = AccordionCard("Engine")
        card.add(_muted("Compressible flow through the nozzle and into the plume, solved live from ignition: "
                        "shock diamonds, Mach disks, expansion fans and the starting vortex form from the physics. "
                        "Thrust is measured across the exit plane and compared with 1-D theory on the right."))
        row = QHBoxLayout()
        self.btn_ignite = button("Ignite", "primary", "play", "Open the propellant valves: the chamber pressure "
                                                             "ramps up over the ignition time")
        self.btn_ignite.setMinimumHeight(38)
        self.btn_shutdown = button("Shut down", "", "stop", "Close the valves: the chamber pressure ramps down")
        self.btn_shutdown.setMinimumHeight(38)
        row.addWidget(self.btn_ignite, 1)
        row.addWidget(self.btn_shutdown, 1)
        card.add_layout(row)
        self.btn_ignite.clicked.connect(self._ignite)
        self.btn_shutdown.clicked.connect(self.view.shutdown_engine)
        row = QHBoxLayout()
        self.btn_run = button("Run", "", "play", "Run / pause the simulation (time stops while paused)")
        self.btn_step = button("Step", "ghost", "", "Advance a few time steps while paused")
        self.btn_reset = button("Restart", "", "reset", "Back to a cold engine in still air")
        row.addWidget(self.btn_run, 2)
        row.addWidget(self.btn_step, 1)
        row.addWidget(self.btn_reset, 2)
        card.add_layout(row)
        self.btn_run.clicked.connect(self.toggle_run)
        self.btn_step.clicked.connect(self.view.step_once)
        self.btn_reset.clicked.connect(self.view.restart)
        self.state_label = QLabel("READY")
        self.state_label.setFont(mono_font(13))
        self.state_label.setTextFormat(Qt.TextFormat.RichText)
        card.add(self.state_label)
        self.ramp = _dspin(0.01, 200.0, 1.0, 2, "ms", 0.1, "Time for the chamber pressure to rise to its set value "
                           "(simulated). Real engines take 0.1-1 s; a short ramp keeps the start-up watchable.")
        self.ramp_off = _dspin(0.01, 200.0, 1.0, 2, "ms", 0.1, "Time for the chamber pressure to fall at shut-down")
        card.add(labeled_row("Ignition ramp", self.ramp))
        card.add(labeled_row("Shut-down ramp", self.ramp_off))
        self.ramp.valueChanged.connect(lambda v: setattr(self.view, "ramp_ms", v))
        self.ramp_off.valueChanged.connect(lambda v: setattr(self.view, "shutdown_ms", v))
        self.t_auto = ToggleSwitch("Re-ignite after changes", True,
                                   "Geometry, gas and grid changes restart the flow; re-ignite automatically if the "
                                   "engine was firing")
        self.t_auto.toggled.connect(lambda c: setattr(self.view, "auto_ignite", c))
        card.add(self.t_auto)
        self.speed = ValueSlider("Simulation speed", 0.25, 8.0, 1.0, 0.25, "x", 2,
                                 "Steps per frame. Higher reaches steady state faster but lowers the frame rate.")
        self.speed.slider.valueChanged.connect(lambda _v: self.view.set_speed(self.speed.value()))
        card.add(self.speed)
        # ascent: the engine climbs through the standard atmosphere while it fires
        self.asc_top = _dspin(1.0, 80.0, 40.0, 1, "km", 5.0, "Altitude at the end of the climb")
        self.asc_time = _dspin(0.5, 500.0, 20.0, 1, "ms", 5.0, "Simulated time of the climb (compressed: a real "
                               "ascent takes minutes; the plume adapts within milliseconds)")
        grid = QGridLayout()
        grid.addWidget(labeled_row("Climb to", self.asc_top), 0, 0)
        grid.addWidget(labeled_row("over", self.asc_time), 0, 1)
        card.add_layout(grid)
        self.btn_ascent = button("Fly an ascent", "ghost", "plane", "Ignite (if needed) and climb from the current "
                                 "altitude: watch the plume widen and the thrust grow as the air thins")
        self.btn_ascent.setCheckable(True)
        self.btn_ascent.toggled.connect(self._toggle_ascent)
        card.add(self.btn_ascent)
        self._ascent: Optional[tuple[float, float, float, float]] = None
        self._ascent_pts: list[tuple[float, float]] = []
        return card

    def _toggle_ascent(self, on: bool) -> None:
        if not on:
            self._ascent = None
            self.btn_ascent.setText("Fly an ascent")
            return
        self.t_alt.setChecked(True)
        if self.view.engine_state() in ("READY", "SHUT DOWN"):
            self._ignite()
        elif not self.view.running:
            self.view.start()
        t_now = self.view.meas.get("time_s", 0.0) * 1e3
        start_after = self.ramp.value() * 1.5 if self.view.engine_state() == "IGNITION" else 0.0
        self._ascent = (t_now + start_after, self.asc_time.value(), self.alt.value(), self.asc_top.value())
        self._ascent_pts = []
        self.btn_ascent.setText("Stop the ascent")
        self.status_label.setText(f"Ascent: {self.alt.value():g} → {self.asc_top.value():g} km over "
                                  f"{self.asc_time.value():g} ms of simulated time.")

    def _advance_ascent(self, summ: dict) -> None:
        if self._ascent is None:
            return
        t0, dur, h0, h1 = self._ascent
        t = summ["meas"].get("time_s", 0.0) * 1e3
        if t < t0:
            return
        frac = min((t - t0) / max(dur, 1e-9), 1.0)
        h = h0 + (h1 - h0) * frac
        if abs(h - self.alt.value()) > 0.05:
            self.alt.setValue(h)
        avg = summ["avg"]
        if avg.get("thrust") and summ["state"] == "FIRING":
            self._ascent_pts.append((h, avg["thrust"] / 1e3))
            pts = np.asarray(self._ascent_pts)
            self.c_alt_sim.setData(pts[:, 0], pts[:, 1])
        if frac >= 1.0:
            self.btn_ascent.setChecked(False)
            self.status_label.setText(f"Ascent complete at {h1:g} km. The simulated thrust points are on the "
                                      "Altitude tab.")

    def _card_presets(self) -> QWidget:
        card = AccordionCard("Presets")
        self.preset = QComboBox()
        for name in PRESETS:
            self.preset.addItem(name)
        self.preset.addItem(CUSTOM)
        self.preset.activated.connect(lambda _i: self.load_preset(self.preset.currentText()))
        card.add(self.preset)
        card.add(_muted("Class-typical values for well-known engine types. Every field below stays editable."))
        return card

    def _card_propellant(self) -> QWidget:
        card = AccordionCard("Propellant")
        self.prop = QComboBox()
        for name in nz.PROPELLANTS:
            self.prop.addItem(name)
        self.prop.addItem(CUSTOM)
        self.prop.activated.connect(lambda _i: self._propellant_selected())
        card.add(self.prop)
        self.of = _dspin(0.5, 12.0, 2.6, 2, "", 0.1, "Oxidiser-to-fuel mass ratio. γ, molar mass and chamber "
                         "temperature follow from approximate equilibrium chemistry tables.")
        self.row_of = labeled_row("Mixture ratio O/F", self.of)
        card.add(self.row_of)
        self.of.valueChanged.connect(lambda _v: self._of_edited())
        self.gamma = _dspin(1.05, 1.67, 1.22, 3, "", 0.01, "Ratio of specific heats of the exhaust (frozen)")
        self.molar = _dspin(2.0, 60.0, 22.5, 2, "kg/kmol", 0.5, "Mean molar mass of the exhaust gas")
        self.tc = _dspin(100.0, 4500.0, 3650.0, 0, "K", 25.0, "Combustion (chamber stagnation) temperature")
        card.add(labeled_row("γ (cp/cv)", self.gamma))
        card.add(labeled_row("Molar mass", self.molar))
        card.add(labeled_row("Chamber temperature", self.tc))
        for w in (self.gamma, self.molar, self.tc):
            w.valueChanged.connect(self._propellant_edited)
        self.prop_note = _muted("")
        card.add(self.prop_note)
        return card

    def _card_feed(self) -> QWidget:
        card = AccordionCard("Chamber and flow rate")
        self.feed_mode = SegmentedControl(["Chamber pressure", "Mass flow"],
                                          ["Set the chamber pressure; the mass flow follows from the throat",
                                           "Set the propellant mass flow; the chamber pressure follows (ṁ c* / A_t)"])
        self.feed_mode.changed.connect(lambda _i: self._feed_mode_changed())
        card.add(self.feed_mode)
        self.pc = _dspin(0.5, 500.0, 97.0, 2, "bar", 1.0, "Chamber stagnation pressure (absolute). Changing it while "
                         "firing throttles the engine live.")
        self.mdot = _dspin(0.0001, 5000.0, 100.0, 4, "kg/s", 1.0, "Propellant mass flow rate")
        card.add(labeled_row("Chamber pressure", self.pc))
        card.add(labeled_row("Mass flow", self.mdot))
        self.pc.valueChanged.connect(lambda _v: self._feed_edited("pc"))
        self.mdot.valueChanged.connect(lambda _v: self._feed_edited("mdot"))
        self.feed_note = _muted("")
        card.add(self.feed_note)
        return card

    def _card_nozzle(self) -> QWidget:
        card = AccordionCard("Nozzle geometry")
        self.shape = SegmentedControl(["Round (axisymmetric)", "Slot (planar 2-D)"],
                                      ["A normal round rocket nozzle (solved in the r-x plane)",
                                       "A 2-D slot nozzle of unit width (thrust per metre of width)"])
        self.shape.changed.connect(lambda _i: self._schedule())
        card.add(self.shape)
        self.contour = QComboBox()
        for c in nz.Contour:
            self.contour.addItem(c.value, c)
        self.contour.activated.connect(lambda _i: (self._sync_contour_ui(), self._schedule()))
        card.add(labeled_row("Divergent section", self.contour))
        self.dt = _dspin(0.5, 5000.0, 260.0, 1, "mm", 5.0, "Throat diameter")
        self.de = _dspin(0.5, 20000.0, 1040.0, 1, "mm", 10.0, "Exit diameter")
        self.eps = _dspin(1.0, 500.0, 16.0, 2, "", 0.5, "Expansion ratio ε = exit area / throat area")
        self.dc = _dspin(0.5, 10000.0, 420.0, 1, "mm", 5.0, "Combustion chamber diameter")
        self.lc = _dspin(0.0, 20000.0, 150.0, 1, "mm", 5.0, "Length of the cylindrical chamber")
        self.conv = _dspin(10.0, 75.0, 35.0, 1, "°", 1.0, "Half-angle of the convergent cone")
        self.div = _dspin(2.0, 45.0, 15.0, 1, "°", 0.5, "Half-angle of a conical divergent section")
        self.bell = _dspin(30.0, 120.0, 80.0, 0, "%", 5.0, "Bell length as a percentage of a 15° cone of the same ε")
        self.tn = _dspin(0.0, 50.0, 0.0, 1, "°", 0.5, "Wall angle where the bell starts (0 = typical Rao value)")
        self.te = _dspin(0.0, 30.0, 0.0, 1, "°", 0.5, "Wall angle at the exit lip (0 = typical Rao value)")
        self.ru = _dspin(0.05, 5.0, 1.5, 3, "× Rt", 0.1, "Radius of curvature of the wall upstream of the throat")
        self.rd = _dspin(0.05, 5.0, 0.382, 3, "× Rt", 0.05, "Radius of curvature of the wall downstream of the throat")
        self.wall = _dspin(0.05, 500.0, 12.0, 2, "mm", 1.0, "Nozzle wall thickness (thick lips change the base flow)")
        self.body = _dspin(0.0, 20000.0, 0.0, 1, "mm", 50.0, "Diameter of a rocket body around the engine (0 = a bare "
                           "engine). The plume then interacts with the vehicle base.")
        self.row_div = labeled_row("Divergent half-angle", self.div)
        self.row_bell = labeled_row("Bell length", self.bell)
        self.row_tn = labeled_row("Bell start angle θn", self.tn)
        self.row_te = labeled_row("Bell exit angle θe", self.te)
        for text, w in (("Throat diameter", self.dt), ("Exit diameter", self.de), ("Expansion ratio ε", self.eps),
                        ("Chamber diameter", self.dc), ("Chamber length", self.lc), ("Convergent half-angle", self.conv)):
            card.add(labeled_row(text, w))
        for row in (self.row_div, self.row_bell, self.row_tn, self.row_te):
            card.add(row)
        for text, w in (("Throat radius upstream", self.ru), ("Throat radius downstream", self.rd),
                        ("Wall thickness", self.wall), ("Vehicle body diameter", self.body)):
            card.add(labeled_row(text, w))
        for w in (self.dt, self.dc, self.lc, self.conv, self.div, self.bell, self.tn, self.te, self.ru, self.rd,
                  self.wall, self.body):
            w.valueChanged.connect(lambda _v: self._schedule())
        self.de.valueChanged.connect(lambda _v: self._exit_edited("de"))
        self.eps.valueChanged.connect(lambda _v: self._exit_edited("eps"))
        self.dt.valueChanged.connect(lambda _v: self._exit_edited("dt"))
        self.geom_note = _muted("")
        card.add(self.geom_note)
        return card

    def _card_environment(self) -> QWidget:
        card = AccordionCard("Environment")
        self.t_alt = ToggleSwitch("Standard atmosphere (ISA)", True, "Ambient pressure and temperature from the altitude")
        self.t_alt.toggled.connect(lambda _c: self._env_edited())
        card.add(self.t_alt)
        self.alt = _dspin(-0.5, 80.0, 0.0, 2, "km", 1.0, "Altitude (ISA standard day). Change it while firing to "
                          "watch the plume grow as the engine climbs.")
        self.pa = _dspin(0.001, 2000.0, 101.325, 3, "kPa", 1.0, "Ambient (back) pressure")
        self.ta = _dspin(50.0, 1000.0, 288.15, 1, "K", 5.0, "Ambient temperature")
        self.fm = _dspin(0.0, 6.0, 0.0, 2, "", 0.1, "Flight Mach number of the vehicle (air flowing past the engine)")
        card.add(labeled_row("Altitude", self.alt))
        card.add(labeled_row("Ambient pressure", self.pa))
        card.add(labeled_row("Ambient temperature", self.ta))
        card.add(labeled_row("Flight Mach number", self.fm))
        for w in (self.alt, self.pa, self.ta, self.fm):
            w.valueChanged.connect(lambda _v: self._env_edited())
        row = QHBoxLayout()
        for km in (0.0, 10.0, 20.0, 40.0, 60.0):
            b = button(f"{km:g} km", "ghost", tooltip=f"Altitude {km:g} km")
            b.clicked.connect(lambda _=False, v=km: (self.t_alt.setChecked(True), self.alt.setValue(v)))
            row.addWidget(b)
        card.add_layout(row)
        self.env_note = _muted("")
        card.add(self.env_note)
        return card

    def _card_simulation(self) -> QWidget:
        card = AccordionCard("Simulation")
        self.resolution = QComboBox()
        for name in NOZZLE_RESOLUTIONS:
            self.resolution.addItem(name)
        self.resolution.setCurrentText(default_resolution())
        self.view.resolution = default_resolution()
        self.resolution.currentTextChanged.connect(self.view.set_resolution)
        card.add(labeled_row("Grid", self.resolution, "Finer grids resolve thinner shocks but run slower"))
        self.plume_l = _dspin(1.0, 40.0, 9.0, 1, "× De", 1.0, "Length of the plume region behind the exit")
        self.plume_r = _dspin(1.3, 20.0, 3.0, 1, "× Re", 0.5, "Radius of the region around the plume")
        card.add(labeled_row("Plume region length", self.plume_l))
        card.add(labeled_row("Plume region radius", self.plume_r))
        self.plume_l.valueChanged.connect(lambda _v: self._schedule())
        self.plume_r.valueChanged.connect(lambda _v: self._schedule())
        self.device = DeviceSelector()
        card.add(self.device)
        self.precision = QComboBox()
        self.precision.addItem("Single (fast)", True)
        self.precision.addItem("Double (exact)", False)
        self.precision.currentIndexChanged.connect(
            lambda _i: self.view.set_gpu_precision(bool(self.precision.currentData())))
        card.add(labeled_row("GPU precision", self.precision,
                             "Single precision runs ~6× faster on consumer GPUs (they have few double-precision "
                             "units); results agree with double precision to about 0.1 %. Double precision matches "
                             "the CPU solver exactly."))
        grid = QGridLayout()
        snap = button("Snapshot…", "ghost", "camera", "Save the current view as PNG")
        snap.clicked.connect(self._snapshot)
        self.btn_rec = button("Record video…", "ghost", "camera", "Record the running simulation to MP4")
        self.btn_rec.setCheckable(True)
        self.btn_rec.toggled.connect(self._toggle_recording)
        exp = button("Export results…", "ghost", "export", "Write the performance table, time histories and "
                                                         "profiles to CSV")
        exp.clicked.connect(self._export_csv)
        fit = button("Fit view", "ghost", "", "Show the whole domain")
        fit.clicked.connect(self.view.fit_view)
        zoom = button("Zoom on nozzle", "ghost", "", "Zoom in on the engine")
        zoom.clicked.connect(self.view.zoom_nozzle)
        grid.addWidget(snap, 0, 0)
        grid.addWidget(self.btn_rec, 0, 1)
        grid.addWidget(exp, 1, 0)
        grid.addWidget(fit, 1, 1)
        grid.addWidget(zoom, 2, 0)
        pdf = button("PDF report…", "ghost", "export", "A report with the inputs, results table, the flow views and "
                                                     "every chart")
        pdf.clicked.connect(self._export_pdf)
        grid.addWidget(pdf, 2, 1)
        card.add_layout(grid)
        card.add(_muted("Scroll to zoom, drag to pan; hover the flow for local values."))
        return card

    def _card_display(self) -> QWidget:
        card = AccordionCard("Display")
        self.field = QComboBox()
        for f in NField:
            self.field.addItem(f.value, f)
        self.field.currentIndexChanged.connect(lambda i: self.view.set_field(list(NField)[i]))
        card.add(labeled_row("Show", self.field))
        self.cmap = QComboBox()
        for c in ("turbo", "viridis", "plasma", "inferno", "cividis"):
            self.cmap.addItem(c)
        self.cmap.currentTextChanged.connect(self.view.set_colormap)
        card.add(labeled_row("Mach colours", self.cmap))
        self.flame = QComboBox()
        self.flame.addItem("Auto (from propellant)", "auto")
        for name in FLAMES:
            self.flame.addItem(name.capitalize(), name)
        self.flame.currentIndexChanged.connect(lambda _i: self._schedule())
        card.add(labeled_row("Flame colour", self.flame))
        grid = QGridLayout()
        self.t_sonic = ToggleSwitch("Sonic line", True, "White contour where the Mach number is 1")
        self.t_marks = ToggleSwitch("Shock markers", True, "Detected Mach disks (white bars) and shock cells (cyan)")
        self.t_theory = ToggleSwitch("Theory marks", True, "Amber circles: theoretical shock-cell spacing or Mach-disk "
                                     "position; red ×: 1-D separation / shock position in the nozzle")
        self.t_stream = ToggleSwitch("Streamlines", False, "Streamlines of the exhaust and of the entrained air")
        grid.addWidget(self.t_sonic, 0, 0)
        grid.addWidget(self.t_marks, 0, 1)
        grid.addWidget(self.t_theory, 1, 0)
        grid.addWidget(self.t_stream, 1, 1)
        for t in (self.t_sonic, self.t_marks, self.t_theory, self.t_stream):
            t.toggled.connect(lambda _c: self.view.set_overlays(self.t_sonic.isChecked(), self.t_marks.isChecked(),
                                                                self.t_theory.isChecked(), self.t_stream.isChecked()))
        card.add_layout(grid)
        fx = QGridLayout()
        self.t_after = ToggleSwitch("Afterburning", True, "Fuel-rich exhaust burning again with the entrained air "
                                    "in the mixing layer (glow view)")
        self.t_bloom = ToggleSwitch("Bloom", True, "Bright gas bleeds light into its surroundings, as on a camera")
        self.t_smoke = ToggleSwitch("Smoke", True, "The alumina smoke trail of solid motors")
        fx.addWidget(self.t_after, 0, 0)
        fx.addWidget(self.t_bloom, 0, 1)
        fx.addWidget(self.t_smoke, 1, 0)
        for tg in (self.t_after, self.t_bloom, self.t_smoke):
            tg.toggled.connect(lambda _c: self.view.set_flame_effects(self.t_after.isChecked(), self.t_bloom.isChecked(),
                                                                      self.t_smoke.isChecked()))
        card.add_layout(fx)
        return card

    # -- analysis tabs --------------------------------------------------------------------------------
    def _plot(self, xlabel: str, ylabel: str, legend: bool = True) -> pg.PlotWidget:
        w = pg.PlotWidget()
        _style_plot(w, xlabel, ylabel)
        if legend:
            w.addLegend(offset=(-8, 8), labelTextColor=PALETTE.text_muted, brush=pg.mkBrush(PALETTE.surface_alt + "B0"))
        return w

    def _tab(self, *widgets: QWidget) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        for x in widgets:
            lay.addWidget(x, 0 if isinstance(x, QLabel) else 1)        # charts stretch, notes keep their height
        return w

    def _analysis(self) -> QWidget:
        tabs = QTabWidget()
        self.tabs = tabs
        tabs.currentChanged.connect(lambda _i: self._refresh(force=True))
        # performance table
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Quantity", "Theory (1-D)", "Simulation"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setFont(mono_font(10))
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.regime = QLabel("")
        self.regime.setWordWrap(True)
        self.regime.setObjectName("GlassCard")
        self.regime.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.regime.setMargin(8)
        self.perf_note = _muted("")
        perf = QWidget()
        pl = QVBoxLayout(perf)
        pl.setContentsMargins(6, 6, 6, 6)
        pl.addWidget(self.regime)
        pl.addWidget(self.table, 1)
        pl.addWidget(self.perf_note)
        tabs.addTab(perf, "Performance")
        # live histories
        self.p_thrust = self._plot("time (ms)", "thrust (kN)")
        self.c_thrust = self.p_thrust.plot(name="simulated", pen=pg.mkPen(PALETTE.cyan, width=2))
        self.l_thrust = pg.InfiniteLine(angle=0, pen=pg.mkPen(PALETTE.warning, width=1, style=Qt.PenStyle.DashLine))
        self.p_thrust.addItem(self.l_thrust, ignoreBounds=True)
        self.p_pc = self._plot("time (ms)", "chamber pressure (bar)", legend=False)
        self.c_pc = self.p_pc.plot(pen=pg.mkPen(PALETTE.warning, width=2))
        self.p_md = self._plot("time (ms)", "mass flow (kg/s)")
        self.c_md = self.p_md.plot(name="exit", pen=pg.mkPen(PALETTE.success, width=2))
        self.c_mdt = self.p_md.plot(name="throat", pen=pg.mkPen(PALETTE.indigo, width=1.5))
        self.l_md = pg.InfiniteLine(angle=0, pen=pg.mkPen(PALETTE.warning, width=1, style=Qt.PenStyle.DashLine))
        self.p_md.addItem(self.l_md, ignoreBounds=True)
        for pw in (self.p_pc, self.p_md):
            pw.setXLink(self.p_thrust)
        live = QWidget()
        ll = QVBoxLayout(live)
        ll.setContentsMargins(6, 6, 6, 6)
        ll.addWidget(_card("Thrust", self.p_thrust), 3)
        ll.addWidget(_card("Chamber pressure", self.p_pc), 2)
        ll.addWidget(_card("Mass flow", self.p_md), 2)
        ll.addWidget(_muted("Dashed lines: 1-D theory at the set conditions."))
        tabs.addTab(live, "Live")
        # centre line
        self.p_axp = self._plot("x − x_exit (exit diameters)", "p / pₐ (log)")
        self.p_axp.setLogMode(y=True)
        self.c_axp = self.p_axp.plot(name="p / pₐ", pen=pg.mkPen(PALETTE.warning, width=2))
        self.p_axm = self._plot("x − x_exit (exit diameters)", "Mach, T / Tc")
        self.c_axm = self.p_axm.plot(name="Mach", pen=pg.mkPen(PALETTE.cyan, width=2))
        self.c_axt = self.p_axm.plot(name="T / Tc", pen=pg.mkPen("#F472B6", width=1.5))
        self.p_axm.setXLink(self.p_axp)
        self.ax_marks: list[pg.InfiniteLine] = []
        self.ax_info = _muted("")
        cl = QWidget()
        cll = QVBoxLayout(cl)
        cll.setContentsMargins(6, 6, 6, 6)
        cll.addWidget(_card("Pressure along the axis", self.p_axp), 1)
        cll.addWidget(_card("Mach number and temperature along the axis", self.p_axm), 1)
        cll.addWidget(_muted("Each pressure peak is a shock crossing the axis (one shock cell); white lines are "
                             "Mach disks."))
        cll.addWidget(self.ax_info)
        self.profiles = QTabWidget()
        self.profiles.currentChanged.connect(lambda _i: self._refresh(force=True))
        self.profiles.addTab(cl, "Centre line")
        # exit plane
        self.p_ex = self._plot("r / R_exit", "value")
        self.c_exm = self.p_ex.plot(name="Mach", pen=pg.mkPen(PALETTE.cyan, width=2))
        self.c_exu = self.p_ex.plot(name="u / v_e (theory)", pen=pg.mkPen(PALETTE.success, width=2))
        self.c_exp = self.p_ex.plot(name="p / p_e (theory)", pen=pg.mkPen(PALETTE.warning, width=2))
        self.l_exm = pg.InfiniteLine(angle=0, pen=pg.mkPen(PALETTE.cyan, width=1, style=Qt.PenStyle.DashLine))
        self.p_ex.addItem(self.l_exm, ignoreBounds=True)
        self.ex_info = _muted("Dashed: 1-D exit Mach number. A real nozzle's exit flow is not uniform: the Mach number "
                              "and pressure vary across the exit (the 1-D theory gives their average).")
        self.profiles.addTab(self._tab(_card("Across the exit plane", self.p_ex), self.ex_info), "Exit plane")
        # wall pressure
        self.p_wall = self._plot("x / R_throat (from the injector face)", "p_wall / p_c (log)")
        self.p_wall.setLogMode(y=True)
        self.c_wall1d = self.p_wall.plot(name="1-D isentropic", pen=pg.mkPen(PALETTE.text_muted, width=2))
        self.c_wall = self.p_wall.plot(name="simulated", pen=None, symbol="o", symbolSize=4,
                                       symbolBrush=pg.mkBrush(PALETTE.cyan), symbolPen=None)
        self.l_sum = pg.InfiniteLine(angle=0, pen=pg.mkPen(PALETTE.danger, width=1, style=Qt.PenStyle.DashLine),
                                     label="0.4 pₐ (Summerfield)", labelOpts={"color": PALETTE.danger, "position": 0.2})
        self.l_amb = pg.InfiniteLine(angle=0, pen=pg.mkPen("#FFFFFF", width=1, style=Qt.PenStyle.DotLine),
                                     label="pₐ", labelOpts={"color": "#FFFFFF", "position": 0.1})
        self.p_wall.addItem(self.l_sum, ignoreBounds=True)
        self.p_wall.addItem(self.l_amb, ignoreBounds=True)
        self.c_sch = self.p_wall.plot(name="separation limit (Schmucker)",
                                      pen=pg.mkPen(PALETTE.warning, width=1.5, style=Qt.PenStyle.DashLine))
        self.wall_info = _muted("Where the wall pressure falls well below the ambient, a real (viscous) nozzle separates "
                                "from the wall; the inviscid simulation shows the shock that a separation would start.")
        self.sep_info = _muted("")
        self.profiles.addTab(self._tab(_card("Pressure along the inner wall", self.p_wall), self.sep_info,
                                       self.wall_info), "Wall")
        tabs.addTab(self.profiles, "Profiles")
        # altitude
        self.p_altf = self._plot("altitude (km)", "thrust (kN)")
        self.c_altf = self.p_altf.plot(name="thrust", pen=pg.mkPen(PALETTE.cyan, width=2))
        self.c_altsep = self.p_altf.plot(name="separation likely", pen=pg.mkPen(PALETTE.danger, width=3))
        self.c_alt_sim = self.p_altf.plot(name="simulated (ascent)", pen=None, symbol="o", symbolSize=6,
                                          symbolBrush=pg.mkBrush(PALETTE.warning), symbolPen=None)
        self.p_alti = self._plot("altitude (km)", "Isp (s)", legend=False)
        self.c_alti = self.p_alti.plot(pen=pg.mkPen(PALETTE.success, width=2))
        self.p_alti.setXLink(self.p_altf)
        self.alt_now = [pg.InfiniteLine(angle=90, pen=pg.mkPen("#FFFFFF", width=1, style=Qt.PenStyle.DashLine))
                        for _ in range(2)]
        self.alt_opt = [pg.InfiniteLine(angle=90, pen=pg.mkPen(PALETTE.warning, width=1, style=Qt.PenStyle.DotLine))
                        for _ in range(2)]
        for pw, a, b in ((self.p_altf, self.alt_now[0], self.alt_opt[0]), (self.p_alti, self.alt_now[1], self.alt_opt[1])):
            pw.addItem(a, ignoreBounds=True)
            pw.addItem(b, ignoreBounds=True)
        self.alt_info = _muted("")
        al = QWidget()
        all_ = QVBoxLayout(al)
        all_.setContentsMargins(6, 6, 6, 6)
        all_.addWidget(_card("Thrust vs altitude", self.p_altf), 1)
        all_.addWidget(_card("Specific impulse vs altitude", self.p_alti), 1)
        all_.addWidget(self.alt_info)
        tabs.addTab(al, "Altitude")
        self.heat = HeatTab()
        self.heat.wall_profile_changed.connect(self._wall_profile)
        tabs.addTab(self.heat, "Heat")
        self.design = DesignTab()
        self.design.use_mixture_ratio.connect(lambda of: self.of.setValue(of))
        self.design.apply_expansion_ratio.connect(self._apply_eps)
        tabs.addTab(self.design, "Design")
        self.noise = NoiseTab()
        tabs.addTab(self.noise, "Noise")
        self.compare = CompareTab(self._snapshot_engine)
        tabs.addTab(self.compare, "Compare")
        self._static_key = None
        # contour
        self.p_geo = self._plot("x (mm)", "r (mm)", legend=False)
        self.p_geo.setAspectLocked(True)
        self.c_geo = [self.p_geo.plot(pen=pg.mkPen(PALETTE.cyan, width=2)) for _ in range(2)]
        self.c_geo_out = [self.p_geo.plot(pen=pg.mkPen(PALETTE.text_muted, width=1)) for _ in range(2)]
        self.geo_info = QLabel("")
        self.geo_info.setFont(mono_font(10))
        self.geo_info.setWordWrap(True)
        tabs.addTab(self._tab(_card("Nozzle contour (to scale)", self.p_geo), self.geo_info), "Nozzle")
        return tabs

    # -- presets and inputs -------------------------------------------------------------------------------
    def load_preset(self, name: str) -> None:
        if name not in PRESETS:
            return
        p = PRESETS[name]
        self._loading = True
        try:
            self.preset.setCurrentText(name)
            self.prop.setCurrentText(p["prop"])
            self._set_propellant(p["prop"])
            self.feed_mode.set_index(0)
            self.pc.setValue(p["pc"])
            g = replace(nz.NozzleGeometry(), **p["geom"])
            self.contour.setCurrentIndex(list(nz.Contour).index(g.contour))
            self.dt.setValue(g.throat_diameter * 1e3)
            self.de.setValue((g.exit_diameter if g.contour is not S else g.throat_diameter) * 1e3)
            self.eps.setValue(max((self.de.value() / self.dt.value()) ** 2, 1.0))
            self.dc.setValue(g.chamber_diameter * 1e3)
            self.lc.setValue(g.chamber_length * 1e3)
            self.conv.setValue(g.convergent_angle)
            self.div.setValue(g.divergent_angle)
            self.bell.setValue(g.bell_fraction * 100.0)
            self.tn.setValue(g.theta_n)
            self.te.setValue(g.theta_e)
            self.ru.setValue(g.throat_upstream_radius)
            self.rd.setValue(g.throat_downstream_radius)
            self.wall.setValue(g.wall_thickness * 1e3)
            self.body.setValue(g.vehicle_diameter * 1e3)
            self.shape.set_index(0)
            self.t_alt.setChecked(True)
            self.alt.setValue(p["alt"])
            self.fm.setValue(0.0)
            self.plume_l.setValue(p["plume"][0])
            self.plume_r.setValue(p["plume"][1])
            field = NField.SCHLIEREN if "Cold gas" in p["prop"] else NField.GLOW
            self.field.setCurrentIndex(list(NField).index(field))
        finally:
            self._loading = False
        self._sync_contour_ui()
        self._env_edited(apply=False)
        self._feed_edited("pc", apply=False)
        self._apply()

    def _propellant_selected(self) -> None:
        name = self.prop.currentText()
        if name in nz.PROPELLANTS:
            self._loading = True
            self._set_propellant(name)
            self._loading = False
        self._mark_custom()
        self._schedule()

    def _set_propellant(self, name: str) -> None:
        """Fill γ, molar mass and chamber temperature for preset propellant ``name`` (at its usual O/F)."""
        rng = nd.of_range(name)
        self.row_of.setVisible(rng is not None)
        if rng is not None:
            self.of.setRange(*rng)
            self.of.setValue(nd.DEFAULT_OF.get(name, 0.5 * (rng[0] + rng[1])))
            gas = nd.gas_at_mixture_ratio(name, self.of.value())
            self.gamma.setValue(gas.gamma)
            self.molar.setValue(gas.molar_mass)
            self.tc.setValue(gas.chamber_temperature)
        else:
            p = nz.PROPELLANTS[name]
            self.gamma.setValue(p.gamma)
            self.molar.setValue(p.molar_mass)
            self.tc.setValue(p.chamber_temperature)

    def _of_edited(self) -> None:
        if self._loading:
            return
        name = self.prop.currentText()
        if nd.of_range(name) is None:
            return
        gas = nd.gas_at_mixture_ratio(name, self.of.value())
        self._loading = True
        self.gamma.setValue(gas.gamma)
        self.molar.setValue(gas.molar_mass)
        self.tc.setValue(gas.chamber_temperature)
        self._loading = False
        self._mark_custom()
        self._schedule()

    def _current_of(self) -> Optional[float]:
        return self.of.value() if self.row_of.isVisible() or nd.of_range(self.prop.currentText()) else None

    def _propellant_edited(self, *_a: object) -> None:
        if self._loading:
            return
        name = self.prop.currentText()
        p = nz.PROPELLANTS.get(name)
        if p is not None:                            # a hand-edited gas is no longer the named propellant
            ref = nd.gas_at_mixture_ratio(name, self.of.value()) if nd.of_range(name) else nz.Gas.of(p)
            if (abs(ref.gamma - self.gamma.value()) > 1e-6 or abs(ref.molar_mass - self.molar.value()) > 1e-6
                    or abs(ref.chamber_temperature - self.tc.value()) > 1e-6):
                self.prop.setCurrentText(CUSTOM)
                self.row_of.setVisible(False)
        self._mark_custom()
        self._schedule()

    def _mark_custom(self) -> None:
        if not self._loading and self.preset.currentText() != CUSTOM:
            self.preset.setCurrentText(CUSTOM)

    def _gas(self) -> nz.Gas:
        return nz.Gas(self.gamma.value(), self.molar.value(), self.tc.value())

    def _geometry(self) -> nz.NozzleGeometry:
        contour = self.contour.currentData()
        contour = contour if isinstance(contour, nz.Contour) else nz.Contour(contour)
        return nz.NozzleGeometry(
            throat_diameter=self.dt.value() * 1e-3, exit_diameter=max(self.de.value(), self.dt.value()) * 1e-3,
            chamber_diameter=self.dc.value() * 1e-3, chamber_length=self.lc.value() * 1e-3,
            convergent_angle=self.conv.value(), contour=contour, divergent_angle=self.div.value(),
            bell_fraction=self.bell.value() / 100.0, theta_n=self.tn.value(), theta_e=self.te.value(),
            throat_upstream_radius=self.ru.value(), throat_downstream_radius=self.rd.value(),
            wall_thickness=self.wall.value() * 1e-3, vehicle_diameter=self.body.value() * 1e-3)

    def _planar(self) -> bool:
        return self.shape.index() == 1

    def _exit_edited(self, which: str) -> None:
        if self._loading:
            return
        self._loading = True
        try:
            planar = self._planar()
            if which == "eps":
                q = self.eps.value() if planar else math.sqrt(self.eps.value())
                self.de.setValue(self.dt.value() * q)
            else:
                q = max(self.de.value() / max(self.dt.value(), 1e-9), 1.0)
                self.eps.setValue(q if planar else q * q)
        finally:
            self._loading = False
        self._schedule()

    def _sync_contour_ui(self) -> None:
        c = self.contour.currentData()
        c = c if isinstance(c, nz.Contour) else nz.Contour(c)
        self.row_div.setVisible(c is C)
        for row in (self.row_bell, self.row_tn, self.row_te):
            row.setVisible(c is B)
        self.de.setEnabled(c is not S)
        self.eps.setEnabled(c is not S)

    def _feed_mode_changed(self) -> None:
        mass = self.feed_mode.index() == 1
        self.pc.setEnabled(not mass)
        self.mdot.setEnabled(mass)
        self._feed_edited("mdot" if mass else "pc")

    def _feed_edited(self, which: str, apply: bool = True) -> None:
        if self._loading:
            return
        geom, gas = self._geometry(), self._gas()
        planar = self._planar()
        self._loading = True
        try:
            if self.feed_mode.index() == 1:
                pc = nz.chamber_pressure_for_mass_flow(self.mdot.value(), geom, gas, planar)
                self.pc.setValue(pc / 1e5)
            else:
                at = 2.0 * geom.rt if planar else math.pi * geom.rt ** 2
                self.mdot.setValue(self.pc.value() * 1e5 * at / nz.characteristic_velocity(gas))
        finally:
            self._loading = False
        unit = "kg/s per metre of width" if planar else "kg/s"
        self.feed_note.setText(f"c* = {nz.characteristic_velocity(gas):.0f} m/s · ṁ = {self.mdot.value():.4g} {unit} at "
                               f"p_c = {self.pc.value():.4g} bar (choked throat).")
        if apply:
            self._schedule()

    def _env_edited(self, apply: bool = True) -> None:
        if self._loading:
            return
        isa = self.t_alt.isChecked()
        self.alt.setEnabled(isa)
        self.pa.setEnabled(not isa)
        self.ta.setEnabled(not isa)
        if isa:
            st = isa_atmosphere(self.alt.value() * 1e3)
            self._loading = True
            self.pa.setValue(st.pressure_pa / 1e3)
            self.ta.setValue(st.temperature_k)
            self._loading = False
        self.env_note.setText(f"pₐ = {self.pa.value():.4g} kPa, Tₐ = {self.ta.value():.1f} K. Altitude, ambient and "
                              "flight speed change live while the engine runs.")
        if apply:
            self._schedule()

    def _schedule(self) -> None:
        if self._loading:
            return
        if self.feed_mode.index() == 1:
            self._feed_edited("mdot", apply=False)
        else:
            self._feed_edited("pc", apply=False)
        self._apply_timer.start()

    def _setup(self) -> EngineSetup:
        geom = self._geometry()
        flame = self.flame.currentData()
        if flame in (None, "auto"):
            flame = flame_for(self.prop.currentText())
        return EngineSetup(geom=geom, gas=self._gas(), pc=self.pc.value() * 1e5, pa=self.pa.value() * 1e3,
                           ta=self.ta.value(), flight_mach=self.fm.value(), planar=self._planar(),
                           plume_length=self.plume_l.value(), plume_radius=self.plume_r.value(), flame=flame)

    def _apply(self) -> None:
        if self._loading:
            return
        setup = self._setup()
        p = nz.PROPELLANTS.get(self.prop.currentText())
        of_txt = (f"At O/F {self.of.value():.2f}. " if nd.of_range(self.prop.currentText()) else "")
        note = p.note.split(",")[-1].strip().capitalize() if p is not None and nd.of_range(self.prop.currentText())             else (p.note if p is not None else "")
        self.prop_note.setText(of_txt + (note + ". " if p is not None else "Custom gas. ")
                               + f"R = {setup.gas.gas_constant:.1f} J/(kg·K), c* = "
                               f"{nz.characteristic_velocity(setup.gas):.0f} m/s.")
        self.view.configure(setup)
        self._refresh_static()
        self._refresh(force=True)

    # -- run control ----------------------------------------------------------------------------------
    def _ignite(self) -> None:
        self.view.ramp_ms = self.ramp.value()
        self.view.ignite()
        self._sync_run_button()

    def toggle_run(self) -> None:
        if self.view.running:
            self.view.stop()
        else:
            self.view.start()
        self._sync_run_button()

    def _sync_run_button(self) -> None:
        running = self.view.running
        self.btn_run.setText("Pause" if running else "Run")
        self.btn_run.setIcon(make_icon("pause" if running else "play", 16))
        self.btn_step.setEnabled(not running)

    # -- refresh ----------------------------------------------------------------------------------------
    def _on_stats(self, summ: dict) -> None:
        state = summ["state"]
        colour = {"READY": PALETTE.text_muted, "IGNITION": PALETTE.warning, "FIRING": PALETTE.success,
                  "SHUTTING DOWN": PALETTE.warning, "SHUT DOWN": PALETTE.text_muted}[state]
        self.state_label.setText(f"<span style='color:{colour}'>● {state}</span>"
                                 + ("  <span style='color:#10B981'>steady</span>" if summ["steady"] else ""))
        self._sync_run_button()
        self._advance_ascent(summ)
        self._refresh()

    def _refresh(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._last_refresh < 0.2:
            return
        self._last_refresh = now
        summ = self.view.summary()
        self._refresh_table(summ)
        if not hasattr(self, "compare"):              # still building the tabs
            return
        page = self.tabs.currentWidget()
        name = self.tabs.tabText(self.tabs.currentIndex())
        if name == "Live":
            self._refresh_live(summ)
        elif page is self.profiles:
            sub = self.profiles.tabText(self.profiles.currentIndex())
            if sub == "Centre line":
                self._refresh_axis(summ)
            elif sub == "Exit plane":
                self._refresh_exit(summ)
            else:
                self._refresh_wall(summ)
        elif page is self.compare and force:
            self.compare.refresh()
        elif page is self.design and force:
            self._refresh_design()

    def _refresh_table(self, summ: dict) -> None:
        th: Optional[nz.NozzlePerformance] = summ["theory"]
        avg, feats, s = summ["avg"], summ["features"], summ["setup"]
        if th is None:
            return
        have = bool(avg) and summ["state"] in ("FIRING", "IGNITION", "SHUTTING DOWN")

        def f(v: Optional[float], d: int = 3, scale: float = 1.0) -> str:
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                return "—"
            return f"{v * scale:.{d}f}"

        def sim(key: str, d: int = 3, scale: float = 1.0) -> str:
            return f(avg.get(key), d, scale) if have else "—"

        planar = s.planar
        per = " /m" if planar else ""
        de = 2.0 * s.geom.re
        rows: list[tuple[str, str, str]] = [
            ("Chamber pressure (bar)", f(th.chamber_pressure, 2, 1e-5), sim("chamber_pressure", 2, 1e-5)),
            ("Ambient pressure (kPa)", f(th.ambient_pressure, 3, 1e-3), ""),
            ("Thrust (force on chamber + nozzle)" + per, fmt_force(th.thrust), fmt_force(avg["thrust"]) if have else "—"),
            ("  … exit-plane momentum + pressure" + per, "", fmt_force(avg["thrust_exit_plane"]) if have else "—"),
            ("  … outer-surface / base pressure" + per, "", fmt_force(avg["base_force"]) if have else "—"),
            ("  … installed (thrust + base)" + per, "", fmt_force(avg["thrust_installed"]) if have else "—"),
            ("  … 1-D ideal (λ = 1)" + per, fmt_force(th.thrust_ideal), ""),
            ("  … in vacuum" + per, fmt_force(th.thrust_vacuum) if math.isfinite(th.thrust_vacuum) else "—", ""),
            ("Specific impulse Isp (s)", f(th.isp, 1), sim("isp", 1)),
            ("  … in vacuum (s)", f(th.isp_vacuum, 1), ""),
            ("Thrust coefficient CF", f(th.cf, 4), sim("cf", 4)),
            ("Mass flow (kg/s" + (" /m" if planar else "") + ")", f(th.mass_flow, 4), sim("mdot_exit", 4)),
            ("  … through the throat", "", sim("mdot_throat", 4)),
            ("Characteristic velocity c* (m/s)", f(th.cstar, 0), sim("cstar", 0)),
            ("Exit Mach number", f(th.exit_mach, 3), sim("exit_mach", 3)),
            ("Exit pressure (kPa)", f(th.exit_pressure, 2, 1e-3), sim("exit_pressure", 2, 1e-3)),
            ("Exit pressure / ambient", f(th.exit_pressure / th.ambient_pressure, 3),
             f(avg.get("exit_pressure", 0.0) / th.ambient_pressure, 3) if have else "—"),
            ("Exit temperature (K)", f(th.exit_temperature, 0), sim("exit_temperature", 0)),
            ("Exit velocity (m/s)", f(th.exit_velocity, 0), sim("exit_velocity", 0)),
            ("Expansion ratio ε", f(th.expansion_ratio, 2), ""),
            ("Optimum ε for this pₐ", f(th.optimum_expansion_ratio, 2), ""),
            ("Divergence factor λ", f(th.divergence_factor, 4), ""),
            ("Fully expanded jet Mach", f(th.jet_mach, 3), ""),
            ("Shock-cell length (m)", f(th.shock_cell_spacing, 3), f(feats.get("diamond_spacing"), 3) if have else "—"),
            ("  … in exit diameters", f(th.shock_cell_spacing / de if th.shock_cell_spacing else None, 2),
             f(feats["diamond_spacing"] / de if have and feats.get("diamond_spacing") else None, 2)),
            ("Shock cells in view", "", str(feats.get("diamonds", 0)) if have else "—"),
            ("First Mach disk (exit diameters)", f(th.mach_disk_distance / de if th.mach_disk_distance else None, 2),
             f(feats["mach_disk_x"] / de if have and feats.get("mach_disk_x") is not None else None, 2)),
            ("Supersonic core length (m)", "", f(feats.get("core_length"), 3) if have else "—"),
            ("Flow separation (Summerfield)", (f"yes, at x = {th.separation_x:.3f} m" if th.separation_x is not None
                                               else "yes") if th.separation else "no", ""),
            ("Nozzle efficiency (sim / 1-D ideal)", "", f(avg["thrust"] / th.thrust_ideal, 4)
             if have and abs(th.thrust_ideal) > 1e-9 else "—"),
        ]
        if self.table.rowCount() != len(rows):
            self.table.setRowCount(len(rows))
        for r_, (a, b, c) in enumerate(rows):
            for col, text in enumerate((a, b, c)):
                item = self.table.item(r_, col)
                if item is None:
                    item = QTableWidgetItem(text)
                    if col:
                        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    self.table.setItem(r_, col, item)
                elif item.text() != text:
                    item.setText(text)
        col = {"Overexpanded": PALETTE.warning, "Underexpanded": PALETTE.sky, "Ideally expanded": PALETTE.success,
               "Shock inside the nozzle": PALETTE.danger, "Unchoked (subsonic)": PALETTE.danger}.get(th.regime, PALETTE.text)
        self.regime.setText(f"<b style='color:{col}'>{th.regime}</b> · p<sub>e</sub>/p<sub>a</sub> = "
                            f"{th.exit_pressure / th.ambient_pressure:.3f}<br>{th.regime_note}")
        steady = "steady state reached" if summ["steady"] else ("settling — compare once the thrust is steady"
                                                                 if have else "press Ignite")
        self.perf_note.setText(
            f"Simulation: {steady}. Simulated values are averaged over the last few frames. Theory: quasi-1-D "
            "isentropic flow of a frozen perfect gas; the thrust includes the divergence loss λ. The simulation is "
            "inviscid (no boundary layer, no heat loss): expect it 1-3 % below the ideal thrust from 2-D effects and "
            "a discharge coefficient below 1. Shock-cell length: Prandtl vortex-sheet model (Tam & Tanna); Mach disk: "
            "Ashkenas & Sherman (sonic nozzles).")

    def _refresh_live(self, summ: dict) -> None:
        h = summ["history"]
        th = summ["theory"]
        if not h["t"]:
            for c in (self.c_thrust, self.c_pc, self.c_md, self.c_mdt):
                c.setData([], [])
            return
        t = np.asarray(h["t"])
        self.c_thrust.setData(t, np.asarray(h["thrust"]) / 1e3)
        self.c_pc.setData(t, np.asarray(h["pc"]) / 1e5)
        self.c_md.setData(t, np.asarray(h["mdot"]))
        self.c_mdt.setData(t, np.asarray(h["mdot_throat"]))
        if th is not None:
            self.l_thrust.setPos(th.thrust / 1e3)
            self.l_md.setPos(th.mass_flow)

    def _refresh_axis(self, summ: dict) -> None:
        m, lay, s, th = summ["meas"], summ["layout"], summ["setup"], summ["theory"]
        for line in self.ax_marks:
            self.p_axm.removeItem(line)
        self.ax_marks = []
        if not m or lay is None:
            return
        de = 2.0 * s.geom.re
        x = (m["axis_x"] - lay.x_exit) / de
        self.c_axp.setData(x, np.maximum(m["axis_p"] / max(m["p_amb"], 1e-300), 1e-4))
        self.c_axm.setData(x, m["axis_mach"])
        self.c_axt.setData(x, m["axis_T"] / max(s.gas.chamber_temperature, 1.0))
        feats = summ["features"]
        for d in feats.get("mach_disks", []):
            line = pg.InfiniteLine(pos=d / de, angle=90, pen=pg.mkPen("#FFFFFF", width=1))
            self.p_axm.addItem(line, ignoreBounds=True)
            self.ax_marks.append(line)
        txt = []
        if feats.get("diamond_spacing"):
            txt.append(f"Shock-cell length {feats['diamond_spacing'] / de:.2f} De (simulated)")
        if th is not None and th.shock_cell_spacing:
            txt.append(f"{th.shock_cell_spacing / de:.2f} De (Prandtl vortex-sheet theory)")
        if feats.get("mach_disk_x") is not None:
            txt.append(f"first Mach disk {feats['mach_disk_x'] / de:.2f} De from the exit"
                       + (f" (Ashkenas-Sherman {th.mach_disk_distance / de:.2f} De)" if th and th.mach_disk_distance else ""))
        self.ax_info.setText(" · ".join(txt) or "No shock structure yet: ignite the engine and let the plume settle.")

    def _refresh_exit(self, summ: dict) -> None:
        m, s, th = summ["meas"], summ["setup"], summ["theory"]
        if not m or "exit_profile" not in m or th is None:
            return
        rr = m["exit_r"] / max(s.geom.re, 1e-12)
        pr = m["exit_profile"]
        self.c_exm.setData(rr, pr["mach"])
        self.c_exu.setData(rr, pr["u"] / max(th.exit_velocity, 1e-9))
        self.c_exp.setData(rr, pr["p"] / max(th.exit_pressure, 1e-9))
        self.l_exm.setPos(th.exit_mach)

    def _refresh_wall(self, summ: dict) -> None:
        m, s = summ["meas"], summ["setup"]
        rt = s.geom.rt
        x1, p1 = nz.wall_pressure_1d(s.geom, s.gas, s.pc, s.planar)
        self.c_wall1d.setData(x1 / rt, p1 / s.pc)
        if m and "wall_x" in m and len(m["wall_x"]):
            pc = max(m.get("chamber_pressure", s.pc), 1e-9)
            self.c_wall.setData(m["wall_x"] / rt, np.maximum(m["wall_p"] / pc, 1e-6))
        else:
            self.c_wall.setData([], [])
        self.l_sum.setPos(math.log10(max(0.4 * s.pa / s.pc, 1e-12)))
        self.l_amb.setPos(math.log10(max(s.pa / s.pc, 1e-12)))
        g = s.gas.gamma
        k_t = int(np.argmin(np.abs(x1 - s.geom.throat_x())))
        mach = np.array([nz.mach_from_pressure_ratio(pp / s.pc, g) for pp in p1])
        lim = np.array([nd.schmucker_separation_ratio(m, g) for m in mach]) * s.pa / s.pc
        self.c_sch.setData(x1[k_t:] / rt, lim[k_t:])
        theo = {c: nd.separation_point(s.geom, s.gas, s.pc, s.pa, c, s.planar) for c in ("summerfield", "schmucker")}
        sim = None
        if m and "wall_x" in m and len(m["wall_x"]):
            xs, ps = m["wall_x"], m["wall_p"]
            lim_sim = np.interp(xs, x1, lim * s.pc)
            bad = np.nonzero((xs > s.geom.throat_x()) & (ps < lim_sim))[0]
            sim = float(xs[bad[0]]) if bad.size else None

        def where(x: Optional[float]) -> str:
            return "attached to the exit" if x is None else f"separates at x = {x * 1e3:.0f} mm ({x / rt:.1f} Rt)"

        self.sep_info.setText(f"Flow separation — 1-D: Summerfield {where(theo['summerfield'])}; Schmucker "
                              f"{where(theo['schmucker'])}. Simulated wall pressure: {where(sim)} (Schmucker limit).")

    def _refresh_static(self) -> None:
        """Charts that depend only on the setup: altitude performance and the contour."""
        s = self.view.setup
        geom = s.geom
        cx, cr = geom.inner_contour()
        for k, sign in enumerate((1.0, -1.0)):
            self.c_geo[k].setData(cx * 1e3, sign * cr * 1e3)
            self.c_geo_out[k].setData(cx * 1e3, sign * (cr + geom.wall_thickness) * 1e3)
        tn, te = geom.bell_angles()
        lay = self.view.layout_
        lines = [f"Length (injector face → exit): {cx[-1] * 1e3:.1f} mm   throat at x = {geom.throat_x() * 1e3:.1f} mm",
                 f"Throat Ø {geom.throat_diameter * 1e3:.1f} mm · exit Ø {2 * geom.re * 1e3:.1f} mm · chamber Ø "
                 f"{2 * geom.rc * 1e3:.1f} mm",
                 f"Expansion ratio ε = {geom.area_ratio(s.planar):.2f} · contraction ratio = "
                 f"{(geom.rc / geom.rt) ** (1 if s.planar else 2):.2f}"]
        if geom.contour is B:
            lines.append(f"Bell {geom.bell_fraction * 100:.0f} %: θn = {tn:.1f}°, θe = {te:.1f}°"
                         + (" (typical Rao values)" if geom.theta_n <= 0 or geom.theta_e <= 0 else ""))
        elif geom.contour is C:
            lines.append(f"Conical, half-angle {geom.divergent_angle:.1f}°")
        if lay is not None:
            lines.append(f"Grid cell {lay.dx * 1e3:.2f} mm · throat {lay.throat_cells:.1f} cells across")
        self.geo_info.setText("\n".join(lines))
        self.geom_note.setText(lines[0] + "\n" + lines[2])
        key = (geom, s.gas, s.pc, s.planar)
        if key != self._static_key:                  # heat transfer depends on the engine, not on the altitude
            self._static_key = key
            self.heat.set_setup(s)
        th0 = self.view.theory
        if th0 is not None:
            self.noise.set_inputs(th0, 2.0 * geom.re, s.ta, th0.shock_cell_spacing)
        if self.tabs.currentWidget() is self.design:
            self._refresh_design()
        if self.tabs.currentWidget() is self.compare:
            self.compare.refresh()
        # altitude sweep
        alts = np.linspace(0.0, 60.0, 61)
        f, isp, sep = [], [], []
        for a in alts:
            st = isa_atmosphere(a * 1e3)
            t = nz.nozzle_theory(geom, s.gas, s.pc, st.pressure_pa, s.planar)
            f.append(t.thrust / 1e3)
            isp.append(t.isp)
            sep.append(t.separation or t.regime == "Shock inside the nozzle")
        f, isp, sep = np.asarray(f), np.asarray(isp), np.asarray(sep)
        self.c_altf.setData(alts, f)
        self.c_alti.setData(alts, isp)
        self.c_altsep.setData(alts[sep], f[sep])
        now = self.alt.value() if self.t_alt.isChecked() else None
        for line in self.alt_now:
            line.setVisible(now is not None)
            if now is not None:
                line.setPos(now)
        th = self.view.theory
        pe = th.exit_pressure if th is not None and th.choked else None
        design = None
        if pe is not None:
            pres = np.array([isa_atmosphere(a * 1e3).pressure_pa for a in alts])
            k = int(np.argmin(np.abs(np.log(pres / max(pe, 1e-9)))))
            design = float(alts[k])
        for line in self.alt_opt:
            line.setVisible(design is not None)
            if design is not None:
                line.setPos(design)
        info = []
        if design is not None:
            info.append(f"Design altitude (p_e = pₐ) ≈ {design:.0f} km")
        if sep.any():
            info.append(f"1-D separation likely below {alts[sep].max():.0f} km (red)")
        self.alt_info.setText(" · ".join(info) + ". 1-D theory in the standard atmosphere; white line: the current "
                              "altitude, amber: the design altitude (pₑ = pₐ). Thrust rises with altitude as the ambient "
                              "pressure falls; below the design altitude the nozzle is overexpanded, above it "
                              "underexpanded.")

    def _refresh_design(self) -> None:
        self.design.set_setup(self.view.setup, self.prop.currentText(), self._current_of())

    def _wall_profile(self, prof) -> None:
        self.view.wall_temperature_profile = prof
        if self.view3d is not None:
            self.view3d._update_walls()

    def _apply_eps(self, eps: float) -> None:
        if self.contour.currentData() in (S, S.value):
            self.contour.setCurrentIndex(list(nz.Contour).index(B))
            self._sync_contour_ui()
        self.eps.setValue(eps)
        self.status_label.setText(f"Expansion ratio set to {eps:.1f}.")

    def _snapshot_engine(self) -> dict:
        """Settings, theory and the latest simulated values of the current engine (for the Compare tab)."""
        s, th, avg = self.view.setup, self.view.theory, self.view.avg
        sim_ok = bool(avg) and self.view.engine_state() in ("FIRING", "IGNITION")
        rows = [("Propellant", self.prop.currentText()), ("Chamber pressure (bar)", s.pc / 1e5),
                ("Throat Ø (mm)", s.geom.throat_diameter * 1e3), ("Exit Ø (mm)", 2.0 * s.geom.re * 1e3),
                ("Expansion ratio ε", s.geom.area_ratio(s.planar)), ("Altitude / pₐ (kPa)", s.pa / 1e3),
                ("Thrust, theory (kN)", th.thrust / 1e3 if th else None),
                ("Thrust, simulated (kN)", avg["thrust"] / 1e3 if sim_ok else None),
                ("Isp, theory (s)", th.isp if th else None), ("Isp, simulated (s)", avg["isp"] if sim_ok else None),
                ("Vacuum Isp, theory (s)", th.isp_vacuum if th else None),
                ("Mass flow (kg/s)", th.mass_flow if th else None), ("Exit Mach", th.exit_mach if th else None),
                ("Exit pressure (kPa)", th.exit_pressure / 1e3 if th else None),
                ("Regime", th.regime if th else None)]
        label = f"{self.prop.currentText().split(' (')[0]}, ε {s.geom.area_ratio(s.planar):.1f}, {s.pc / 1e5:.0f} bar"
        return {"label": label, "rows": rows, "values": dict(rows), "altitude_curve": altitude_curve(s)}

    def _export_pdf(self) -> None:
        default = get_paths().exports / f"nozzle_report_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
        path, _ = QFileDialog.getSaveFileName(self, "PDF report", str(default), "PDF (*.pdf)")
        if not path:
            return
        self._refresh(force=True)
        summ = self.view.summary()
        s, th = summ["setup"], summ["theory"]
        g = s.geom
        tn, te = g.bell_angles()
        inputs = [["Input", "Value"], ["Propellant", self.prop.currentText()],
                  ["Mixture ratio O/F", f"{self.of.value():.2f}" if nd.of_range(self.prop.currentText()) else "—"],
                  ["γ, molar mass, chamber T", f"{s.gas.gamma:.3f}, {s.gas.molar_mass:.2f} kg/kmol, "
                                              f"{s.gas.chamber_temperature:.0f} K"],
                  ["Chamber pressure", f"{s.pc / 1e5:.2f} bar"],
                  ["Ambient", f"{s.pa / 1e3:.3f} kPa, {s.ta:.1f} K, flight Mach {s.flight_mach:.2f}"],
                  ["Nozzle", f"{g.contour.value}; throat Ø {g.throat_diameter * 1e3:.1f} mm, exit Ø "
                             f"{2 * g.re * 1e3:.1f} mm (ε {g.area_ratio(s.planar):.2f}), chamber Ø {2 * g.rc * 1e3:.1f} mm"],
                  ["Contour angles", f"convergent {g.convergent_angle:.1f}°; "
                                     + (f"bell θn {tn:.1f}°, θe {te:.1f}°" if g.contour is B
                                        else f"cone {g.divergent_angle:.1f}°")],
                  ["Grid", f"{self.resolution.currentText()}; {'planar slot' if s.planar else 'axisymmetric'}"]]
        perf = [["Quantity", "Theory (1-D)", "Simulation"]]
        for r in range(self.table.rowCount()):
            perf.append([self.table.item(r, c).text() if self.table.item(r, c) else "" for c in range(3)])
        regime = (f"<p><b>{th.regime}</b>: {th.regime_note}</p><p>State: {summ['state'].lower()}"
                  f"{', steady' if summ['steady'] else ''}; simulated time "
                  f"{summ['meas'].get('time_s', 0.0) * 1e3:.2f} ms.</p>") if th else ""
        images = [("Flow (2-D section)", widget_image(self.view))]
        if self.view3d is not None:
            try:
                from PySide6.QtGui import QImage as _QI

                arr = self.view3d.plotter.screenshot(return_img=True)
                h, w_ = arr.shape[:2]
                img = _QI(arr.copy().data, w_, h, 3 * w_, _QI.Format.Format_RGB888).copy()
                from ui.nozzle_tools import _png_b64

                images.append(("Flow (3-D view)", _png_b64(img)))
            except Exception:  # noqa: BLE001 - the 3-D view may not be available
                pass
        for cap, pw in (("Thrust, chamber pressure and mass flow from ignition", self.p_thrust),
                        ("Chamber pressure", self.p_pc), ("Mass flow", self.p_md),
                        ("Pressure along the axis", self.p_axp), ("Mach number and temperature along the axis", self.p_axm),
                        ("Across the exit plane", self.p_ex), ("Pressure along the nozzle wall", self.p_wall),
                        ("Thrust vs altitude", self.p_altf), ("Isp vs altitude", self.p_alti),
                        ("Heat flux into the wall", self.heat.p_q), ("Wall temperature", self.heat.p_t),
                        ("Isp vs mixture ratio", self.design.p_of), ("Exhaust noise directivity", self.noise.p),
                        ("Nozzle contour", self.p_geo)):
            images.append((cap, plot_image(pw)))
        sections = [("Operating point", regime),
                    ("Heat transfer", f"<p>{self.heat.info.text()}</p>"),
                    ("Exhaust noise", "<p>" + self.noise.info.text().replace(chr(10), "<br>") + "</p>")]
        html_text = build_report_html(f"Rocket engine report — {self.prop.currentText()}", sections, images,
                                      [("Inputs", inputs), ("Performance: theory and simulation", perf)])

        def done(out: Optional[Path], err: str) -> None:
            self.status_label.setText(f"PDF report saved: {out}" if out else f"PDF failed ({err}); the HTML report "
                                      f"was saved next to it.")

        try:
            write_report(Path(path), html_text, done)
            self.status_label.setText("Writing the PDF report…")
        except OSError as exc:
            self.status_label.setText(f"Report failed: {exc}")

    # -- media and export -----------------------------------------------------------------------------
    def _snapshot(self) -> None:
        default = get_paths().exports / f"nozzle_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path, _ = QFileDialog.getSaveFileName(self, "Save snapshot", str(default), "PNG image (*.png)")
        if path:
            try:
                target = self.view3d if self.stack.currentIndex() == 1 and self.view3d is not None else self.view
                target.save_png(Path(path))
                self.status_label.setText(f"Snapshot saved: {path}")
            except OSError as exc:
                self.status_label.setText(f"Snapshot failed: {exc}")

    def _toggle_recording(self, on: bool) -> None:
        if on:
            default = get_paths().exports / f"nozzle_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
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
            self.status_label.setText(f"Recording to {path} — press Stop recording to finish.")
        else:
            out = self.view.stop_recording()
            self.btn_rec.setText("Record video…")
            if out is not None:
                self.status_label.setText(f"Video saved: {out}")

    def _export_csv(self) -> None:
        default = get_paths().exports / f"nozzle_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export results", str(default), "CSV (*.csv)")
        if not path:
            return
        try:
            export_results(Path(path), self.view.summary(), self.table)
            self.status_label.setText(f"Results saved: {path}")
        except OSError as exc:
            self.status_label.setText(f"Export failed: {exc}")

    # -- window events ---------------------------------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._refresh(force=True)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Closing hides the window (the state is kept); the simulation pauses."""
        self._save_layout()
        self.view.stop()
        self._sync_run_button()
        event.accept()

    def shutdown(self) -> None:
        self._save_layout()
        if self.view3d is not None:
            self.view3d.shutdown()
        self.view.shutdown()


def export_results(path: Path, summ: dict, table: Optional[QTableWidget] = None,
                   writer_factory: Callable = csv.writer) -> Path:
    """Write the performance table, the time histories and the axis / exit / wall profiles to one CSV file."""
    s, m, h = summ["setup"], summ["meas"], summ["history"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = writer_factory(fh)
        w.writerow(["# Rocket Nozzle Lab results", APP_NAME, time.strftime("%Y-%m-%d %H:%M:%S")])
        g = s.geom
        w.writerow(["# setup", f"gamma={s.gas.gamma}", f"M={s.gas.molar_mass} kg/kmol", f"Tc={s.gas.chamber_temperature} K",
                    f"pc={s.pc:.6g} Pa", f"pa={s.pa:.6g} Pa", f"Ta={s.ta:.6g} K", f"flight Mach={s.flight_mach}",
                    f"contour={g.contour.value}", f"Dt={g.throat_diameter} m", f"De={2 * g.re} m",
                    f"Dc={2 * g.rc} m", f"planar={s.planar}"])
        if table is not None:
            w.writerow([])
            w.writerow(["quantity", "theory (1-D)", "simulation"])
            for r in range(table.rowCount()):
                w.writerow([table.item(r, c).text() if table.item(r, c) else "" for c in range(3)])
        if h["t"]:
            w.writerow([])
            w.writerow(["time_ms", "thrust_N", "chamber_pressure_Pa", "mdot_exit_kg_s", "mdot_throat_kg_s", "throttle"])
            for row in zip(h["t"], h["thrust"], h["pc"], h["mdot"], h["mdot_throat"], h["throttle"]):
                w.writerow([f"{v:.6g}" for v in row])
        if m:
            w.writerow([])
            w.writerow(["axis_x_m", "mach", "p_Pa", "T_K", "u_m_s"])
            for row in zip(m["axis_x"], m["axis_mach"], m["axis_p"], m["axis_T"], m["axis_u"]):
                w.writerow([f"{v:.6g}" for v in row])
            w.writerow([])
            w.writerow(["exit_r_m", "mach", "u_m_s", "p_Pa", "T_K"])
            pr = m["exit_profile"]
            for row in zip(m["exit_r"], pr["mach"], pr["u"], pr["p"], pr["T"]):
                w.writerow([f"{v:.6g}" for v in row])
            if "wall_x" in m:
                w.writerow([])
                w.writerow(["wall_x_m", "wall_p_Pa"])
                for row in zip(m["wall_x"], m["wall_p"]):
                    w.writerow([f"{v:.6g}" for v in row])
    return path
