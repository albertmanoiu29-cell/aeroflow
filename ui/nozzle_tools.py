"""Design and analysis tabs of the Rocket Nozzle Lab (heat transfer, mixture ratio and expansion-ratio
optimisation, noise, engine comparison) and its PDF report."""

from __future__ import annotations

import base64
from dataclasses import replace
import html
import math
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QBuffer, QIODevice, Qt, QTimer, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QScrollArea,
    QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from config.settings import APP_NAME, APP_VERSION, get_paths, isa_atmosphere
from config.theme import PALETTE, mono_font
from core import nozzle as nz
from core import nozzle_design as nd
from core import solid_motor as sm
from core import trajectory as tj
from ui.cfd_panel import ToggleSwitch, button, labeled_row
from visualization.telemetry_charts import _style_plot


def _card(title: str, content: QWidget) -> QWidget:
    from ui.nozzle_window import _card as card

    return card(title, content)


def _muted(text: str = "") -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setProperty("role", "muted")
    return lbl


def _spin(lo: float, hi: float, val: float, dec: int, suffix: str, step: float, tip: str = "") -> QDoubleSpinBox:
    sp = QDoubleSpinBox()
    sp.setRange(lo, hi)
    sp.setDecimals(dec)
    sp.setSingleStep(step)
    sp.setValue(val)
    sp.setKeyboardTracking(False)
    if suffix:
        sp.setSuffix(" " + suffix)
    if tip:
        sp.setToolTip(tip)
    return sp


def _plot(xlabel: str, ylabel: str, legend: bool = True) -> pg.PlotWidget:
    w = pg.PlotWidget()
    _style_plot(w, xlabel, ylabel)
    if legend:
        w.addLegend(offset=(-8, 8), labelTextColor=PALETTE.text_muted, brush=pg.mkBrush(PALETTE.surface_alt + "B0"))
    return w


def _page(*widgets: QWidget) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(6, 6, 6, 6)
    for x in widgets:
        if isinstance(x, QWidget):
            lay.addWidget(x, 0 if isinstance(x, QLabel) or x.property("fixed") else 1)
        else:
            lay.addLayout(x)
    return w


# ---------------------------------------------------------------------------
# Heat transfer
# ---------------------------------------------------------------------------
class HeatTab(QWidget):
    """Bartz heat flux and the wall temperature for the chosen cooling."""

    wall_profile_changed = Signal(object)          # (x array, wall temperature array) or None

    MODES = (("Regenerative + radiation-cooled extension", "mixed"), ("Regenerative (wall held cool)", "regenerative"),
             ("Radiation-cooled throughout", "radiation"), ("Uncooled (adiabatic wall)", "adiabatic"))

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.mode = QComboBox()
        for text, key in self.MODES:
            self.mode.addItem(text, key)
        self.t_wall = _spin(300.0, 2500.0, 800.0, 0, "K", 50.0, "Gas-side wall temperature of a regeneratively "
                            "cooled wall (copper chambers run ~700-900 K)")
        self.emis = _spin(0.05, 1.0, 0.85, 2, "", 0.05, "Emissivity of a radiation-cooled wall (niobium / carbon: "
                          "0.6-0.9)")
        self.ext = _spin(1.5, 200.0, 8.0, 1, "", 1.0, "Area ratio where the regeneratively cooled nozzle ends and "
                         "the radiation-cooled extension begins")
        controls = QWidget()
        controls.setProperty("fixed", True)
        g = QGridLayout(controls)
        g.setContentsMargins(0, 0, 0, 0)
        g.addWidget(labeled_row("Cooling", self.mode), 0, 0, 1, 2)
        g.addWidget(labeled_row("Wall temperature", self.t_wall), 1, 0, 1, 2)
        g.addWidget(labeled_row("Emissivity", self.emis), 2, 0, 1, 2)
        g.addWidget(labeled_row("Extension from ε", self.ext), 3, 0, 1, 2)
        self.p_q = _plot("x / R_throat", "heat flux (MW/m²)", legend=False)
        self.c_q = self.p_q.plot(pen=pg.mkPen(PALETTE.danger, width=2))
        self.p_t = _plot("x / R_throat", "temperature (K)")
        self.c_taw = self.p_t.plot(name="gas recovery T", pen=pg.mkPen(PALETTE.warning, width=1.5))
        self.c_tw = self.p_t.plot(name="wall T", pen=pg.mkPen(PALETTE.cyan, width=2))
        self.p_t.setXLink(self.p_q)
        self.info = _muted("")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(controls)
        lay.addWidget(_card("Heat flux into the wall", self.p_q), 1)
        lay.addWidget(_card("Wall and gas recovery temperature", self.p_t), 1)
        lay.addWidget(self.info)
        for w in (self.t_wall, self.emis, self.ext):
            w.valueChanged.connect(lambda _v: self.refresh())
        self.mode.currentIndexChanged.connect(lambda _i: (self._sync(), self.refresh()))
        self._setup = None
        self.result: Optional[nd.HeatTransfer] = None
        self._sync()

    def _sync(self) -> None:
        key = self.mode.currentData()
        self.t_wall.setEnabled(key in ("regenerative", "mixed"))
        self.emis.setEnabled(key in ("radiation", "mixed"))
        self.ext.setEnabled(key == "mixed")

    def set_setup(self, setup) -> None:
        self._setup = setup
        self.refresh()

    def refresh(self) -> None:
        s = self._setup
        if s is None:
            return
        key = self.mode.currentData()
        ht = nd.bartz_heat_transfer(s.geom, s.gas, s.pc, key, self.t_wall.value(), self.emis.value(), s.planar,
                                    extension_ratio=self.ext.value())
        self.result = ht
        xr = ht.x / s.geom.rt
        self.c_q.setData(xr, ht.q / 1e6)
        self.c_taw.setData(xr, ht.t_aw)
        self.c_tw.setData(xr, ht.t_wall)
        unit = "MW per metre of width" if s.planar else "MW"
        self.info.setText(
            f"Peak heat flux {ht.q_peak / 1e6:.1f} MW/m² at x = {ht.x_peak * 1e3:.0f} mm (the throat); total heat into "
            f"the wall {ht.heat_load / 1e6:.2f} {unit}; hottest wall {float(ht.t_wall.max()):.0f} K. Bartz correlation "
            "(gas side, ±30 %); the radiation-cooled wall is in radiative equilibrium, εσT⁴ = convective heating.")
        self.wall_profile_changed.emit((ht.x, ht.t_wall))


# ---------------------------------------------------------------------------
# Design: mixture ratio and expansion-ratio optimiser
# ---------------------------------------------------------------------------
class DesignTab(QWidget):
    """Isp against mixture ratio, and the expansion ratio that maximises Isp over an altitude range."""

    use_mixture_ratio = Signal(float)
    apply_expansion_ratio = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.p_of = _plot("mixture ratio O/F", "Isp (s)")
        self.c_isp = self.p_of.plot(name="at the current altitude", pen=pg.mkPen(PALETTE.cyan, width=2))
        self.c_ispv = self.p_of.plot(name="in vacuum", pen=pg.mkPen(PALETTE.success, width=2))
        self.l_of = pg.InfiniteLine(angle=90, pen=pg.mkPen("#FFFFFF", width=1, style=Qt.PenStyle.DashLine))
        self.p_of.addItem(self.l_of, ignoreBounds=True)
        self.of_info = _muted("")
        self.btn_best = button("Use the best O/F", "ghost", "", "Set the mixture ratio with the highest Isp here")
        self.btn_best.clicked.connect(lambda: self.use_mixture_ratio.emit(self._best_of) if self._best_of else None)
        self.alt_lo = _spin(0.0, 80.0, 0.0, 1, "km", 1.0, "Lowest altitude the engine fires at")
        self.alt_hi = _spin(0.0, 80.0, 40.0, 1, "km", 5.0, "Highest altitude (equal to the lowest: one design point)")
        self.max_de = _spin(0.0, 20000.0, 0.0, 0, "mm", 50.0, "Largest exit diameter that fits (0 = no limit)")
        self.t_sep = ToggleSwitch("No separation at the lowest altitude", True,
                                  "Reject expansion ratios that would separate (Schmucker criterion) at the lowest "
                                  "altitude - an over-long nozzle at sea level")
        self.btn_opt = button("Optimise expansion ratio", "primary", "", "Search ε for the highest average Isp")
        self.btn_apply = button("Apply", "ghost", "", "Use the optimum exit diameter")
        self.btn_apply.setEnabled(False)
        self.p_opt = _plot("altitude (km)", "Isp (s)")
        self.c_cur = self.p_opt.plot(name="current ε", pen=pg.mkPen(PALETTE.text_muted, width=2))
        self.c_best = self.p_opt.plot(name="optimum ε", pen=pg.mkPen(PALETTE.warning, width=2))
        self.opt_info = _muted("Choose the altitude range the engine works over and press Optimise.")
        box = QWidget()
        box.setProperty("fixed", True)
        g = QGridLayout(box)
        g.setContentsMargins(0, 0, 0, 0)
        g.addWidget(labeled_row("From", self.alt_lo), 0, 0)
        g.addWidget(labeled_row("To", self.alt_hi), 0, 1)
        g.addWidget(labeled_row("Max. exit Ø", self.max_de), 1, 0)
        g.addWidget(self.t_sep, 1, 1)
        row = QHBoxLayout()
        row.addWidget(self.btn_opt, 2)
        row.addWidget(self.btn_apply, 1)
        g.addLayout(row, 2, 0, 1, 2)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(_card("Specific impulse vs mixture ratio", self.p_of), 1)
        top = QHBoxLayout()
        top.addWidget(self.of_info, 1)
        top.addWidget(self.btn_best)
        lay.addLayout(top)
        lay.addWidget(box)
        lay.addWidget(_card("Isp over the altitude range", self.p_opt), 1)
        lay.addWidget(self.opt_info)
        self.btn_opt.clicked.connect(self.optimise)
        self.btn_apply.clicked.connect(lambda: self.apply_expansion_ratio.emit(self._opt.expansion_ratio)
                                       if self._opt else None)
        self._setup = None
        self._prop = ""
        self._of: Optional[float] = None
        self._best_of: Optional[float] = None
        self._opt: Optional[nd.Optimum] = None

    def set_setup(self, setup, propellant: str, of: Optional[float]) -> None:
        self._setup, self._prop, self._of = setup, propellant, of
        self._opt = None
        self.btn_apply.setEnabled(False)
        if nd.of_range(propellant) is None:
            for c in (self.c_isp, self.c_ispv):
                c.setData([], [])
            self._best_of = None
            self.btn_best.setEnabled(False)
            self.of_info.setText("Mixture ratio: only for liquid bipropellants (solids, monopropellants and cold gas "
                                 "have a fixed composition).")
            return
        sw = nd.mixture_sweep(propellant, setup.geom, setup.pc, setup.pa, setup.planar)
        self.c_isp.setData(sw.of, sw.isp)
        self.c_ispv.setData(sw.of, sw.isp_vac)
        self._best_of = sw.best_of
        self.btn_best.setEnabled(True)
        if of is not None:
            self.l_of.setPos(of)
        self.of_info.setText(
            f"Best O/F here: {sw.best_of:.2f} (Isp {float(sw.isp.max()):.1f} s). Hottest at O/F "
            f"{float(sw.of[int(np.argmax(sw.tc))]):.2f} ({float(sw.tc.max()):.0f} K): the best Isp is fuel-rich of that, "
            "because a lighter exhaust expands faster. Approximate equilibrium chemistry tables.")

    def optimise(self) -> None:
        s = self._setup
        if s is None:
            return
        lo, hi = self.alt_lo.value(), max(self.alt_hi.value(), self.alt_lo.value())
        o = nd.optimise_expansion(s.geom, s.gas, s.pc, lo, hi, self.t_sep.isChecked(), s.planar,
                                  self.max_de.value() * 1e-3 if self.max_de.value() > 0 else None)
        self._opt = o
        self.c_cur.setData(o.altitudes, o.isp_current)
        self.c_best.setData(o.altitudes, o.isp_optimum)
        self.btn_apply.setEnabled(True)
        gain = (o.score / max(o.current_score, 1e-9) - 1.0) * 100.0
        where = f"at {lo:g} km" if hi == lo else f"averaged over {lo:g}-{hi:g} km"
        self.opt_info.setText(
            f"Optimum ε = {o.expansion_ratio:.1f} (exit Ø {o.exit_diameter * 1e3:.0f} mm): Isp {o.score:.1f} s {where}, "
            f"{gain:+.1f} % against the current ε = {s.geom.area_ratio(s.planar):.1f}."
            + (" The current nozzle separates at the lowest altitude." if o.sea_level_separation else "")
            + (" " + o.note if o.note else "") + " 1-D theory; press Apply to use it.")


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------
class NoiseTab(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.dist = _spin(5.0, 100000.0, 100.0, 0, "m", 10.0, "Distance from the engine")
        self.eff = _spin(0.05, 3.0, 0.5, 2, "%", 0.1, "Acoustic efficiency: the share of the jet power radiated as "
                         "sound (0.2-1 % for rockets)")
        box = QWidget()
        box.setProperty("fixed", True)
        g = QHBoxLayout(box)
        g.setContentsMargins(0, 0, 0, 0)
        g.addWidget(labeled_row("Distance", self.dist))
        g.addWidget(labeled_row("Efficiency", self.eff))
        self.p = _plot("angle from the exhaust axis (°)", "sound pressure level (dB)", legend=False)
        self.c = self.p.plot(pen=pg.mkPen(PALETTE.warning, width=2))
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setFont(mono_font(10))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(box)
        lay.addWidget(_card("Overall sound pressure level around the engine", self.p), 1)
        lay.addWidget(self.info)
        lay.addWidget(_muted("Engineering estimate (acoustic-efficiency method of NASA SP-8072): good to a few dB at "
                             "best. 0° is straight down the exhaust; the noise peaks about 45° off it."))
        for w in (self.dist, self.eff):
            w.valueChanged.connect(lambda _v: self.refresh())
        self._args = None

    def set_inputs(self, theory: nz.NozzlePerformance, exit_diameter: float, ta: float,
                   shock_cell: Optional[float]) -> None:
        self._args = (theory, exit_diameter, ta, shock_cell)
        self.refresh()

    def refresh(self) -> None:
        if self._args is None:
            return
        th, de, ta, cell = self._args
        ns = nd.jet_noise(th, de, math.sqrt(1.4 * 287.05 * ta), self.dist.value(), self.eff.value() / 100.0, cell)
        self.c.setData(ns.angles, ns.oaspl)
        self.info.setText(
            f"Sound power {ns.acoustic_power / 1e3:,.0f} kW ({ns.pwl:.0f} dB re 1 pW)\n"
            f"Loudest {float(ns.oaspl.max()):.0f} dB at {ns.peak_angle:.0f}° and {ns.distance:,.0f} m "
            f"(120 dB: pain threshold; 140 dB: hearing damage)\n"
            f"Spectrum peak ≈ {ns.mixing_peak_hz:,.0f} Hz (turbulent mixing)"
            + (f"\nShock-associated noise ≈ {ns.shock_peak_hz:,.0f} Hz · screech ≈ {ns.screech_hz:,.0f} Hz"
               if ns.shock_peak_hz else ""))


def _scroll_page(*items) -> QScrollArea:
    """A tab page that scrolls when the analysis panel is short (items: widgets or (widget, min height))."""
    inner = QWidget()
    lay = QVBoxLayout(inner)
    lay.setContentsMargins(6, 6, 6, 6)
    for it in items:
        w, h = it if isinstance(it, tuple) else (it, 0)
        if h:
            w.setMinimumHeight(h)
        lay.addWidget(w, 1 if h else 0)
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setWidget(inner)
    return area


def _grid(rows: list[tuple[str, QWidget]], cols: int = 2) -> QWidget:
    box = QWidget()
    box.setProperty("fixed", True)
    g = QGridLayout(box)
    g.setContentsMargins(0, 0, 0, 0)
    for k, (text, w) in enumerate(rows):
        g.addWidget(labeled_row(text, w), k // cols, k % cols)
    return box


# ---------------------------------------------------------------------------
# Solid motor (BATES grain burn-back)
# ---------------------------------------------------------------------------
class MotorTab(QWidget):
    """Thrust curve of a solid motor: a BATES grain burning through the current nozzle."""

    curve_changed = Signal(object)                 # (MotorResult, BatesGrain) or None

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.prop = QComboBox()
        for name in sm.SOLID_PROPELLANTS:
            self.prop.addItem(name)
        self.a = _spin(0.1, 100.0, 3.5, 2, "mm/s", 0.1, "Burn rate coefficient: r = a · p[MPa]^n")
        self.n = _spin(0.0, 0.95, 0.35, 3, "", 0.01, "Pressure exponent n (below ~0.6 for a stable motor)")
        self.rho = _spin(500.0, 3000.0, 1770.0, 0, "kg/m³", 10.0, "Propellant density as cast")
        self.segs = QSpinBox()
        self.segs.setRange(1, 40)
        self.segs.setValue(4)
        self.segs.setToolTip("Number of grain segments (BATES)")
        self.od = _spin(1.0, 5000.0, 100.0, 1, "mm", 1.0, "Grain outer diameter (inside the liner)")
        self.core = _spin(0.5, 4000.0, 35.0, 1, "mm", 1.0, "Core (port) diameter")
        self.seg_len = _spin(1.0, 10000.0, 150.0, 1, "mm", 5.0, "Length of one segment; about 1.5 × the outer "
                             "diameter gives the most neutral (flat) thrust curve")
        self.eff = _spin(50.0, 100.0, 95.0, 0, "%", 1.0, "c* efficiency (combustion losses; 90-97 %)")
        self.inhibit = ToggleSwitch("Inhibit the segment ends", False, "Only the core burns: a progressive "
                                    "(rising) thrust curve")
        self.btn_size = button("Size for the Lab's chamber pressure", "ghost", "", "Grain proportioned to the "
                               "nozzle's throat, with enough segments for the chamber pressure set in the Lab")
        self.btn_size.clicked.connect(self.auto_size)
        self.btn_eng = button("Export .eng…", "ghost", "", "Save the thrust curve as a RASP .eng motor file for "
                              "OpenRocket / RockSim")
        self.btn_eng.clicked.connect(self.export_eng)
        btns = QWidget()
        bl = QHBoxLayout(btns)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(self.btn_size)
        bl.addWidget(self.btn_eng)
        controls = _grid([("Propellant", self.prop), ("Burn rate a", self.a), ("Exponent n", self.n),
                          ("Density", self.rho), ("Segments", self.segs), ("Grain Ø", self.od), ("Core Ø", self.core),
                          ("Segment length", self.seg_len), ("c* efficiency", self.eff)])
        self.p_f = _plot("time (s)", "thrust (kN)", legend=False)
        self.c_f = self.p_f.plot(pen=pg.mkPen(PALETTE.warning, width=2), fillLevel=0.0,
                                 brush=pg.mkBrush(PALETTE.warning + "30"))
        self.p_p = _plot("time (s)", "chamber pressure (bar)")
        self.c_p = self.p_p.plot(name="pressure", pen=pg.mkPen(PALETTE.cyan, width=2))
        self.c_kn = self.p_p.plot(name="Kn / 10", pen=pg.mkPen(PALETTE.text_muted, width=1,
                                                               style=Qt.PenStyle.DashLine))
        self.p_p.setXLink(self.p_f)
        self.info = QLabel("")
        self.info.setFont(mono_font(10))
        self.info.setWordWrap(True)
        self.note = _muted("Quasi-steady internal ballistics: p_c = (Kn a ρ c*)^(1/(1-n)) at each step of the burn, "
                           "thrust from the nozzle theory at that pressure (over-expansion and separation included). "
                           "No erosive burning, throat erosion or ignition / tail-off transients.")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(_scroll_page(controls, self.inhibit, btns, (_card("Thrust curve", self.p_f), 200),
                                   (_card("Chamber pressure and Kn", self.p_p), 200), self.info, self.note))
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self.refresh)
        for w in (self.a, self.n, self.rho, self.od, self.core, self.seg_len, self.eff):
            w.valueChanged.connect(lambda _v: self._edited())
        self.segs.valueChanged.connect(lambda _v: self._edited())
        self.inhibit.toggled.connect(lambda _v: self._edited())
        self.prop.currentTextChanged.connect(self._prop_changed)
        self._setup = None
        self._sized_for = None
        self._loading = False
        self.result: Optional[sm.MotorResult] = None
        self._prop_changed(self.prop.currentText())

    def _edited(self) -> None:
        if not self._loading:
            self._timer.start()

    def _prop_changed(self, name: str) -> None:
        pr = sm.SOLID_PROPELLANTS.get(name)
        if pr is None:
            return
        self._loading = True
        try:
            self.a.setValue(pr.a_mm)
            self.n.setValue(pr.n)
            self.rho.setValue(pr.density)
        finally:
            self._loading = False
        self._timer.start()

    def propellant(self) -> sm.SolidPropellant:
        base = sm.SOLID_PROPELLANTS[self.prop.currentText()]
        return sm.SolidPropellant(base.name, self.a.value(), self.n.value(), self.rho.value(), base.gamma,
                                  base.molar_mass, base.chamber_temperature, base.note)

    def grain(self) -> sm.BatesGrain:
        od = self.od.value() * 1e-3
        return sm.BatesGrain(od, min(self.core.value() * 1e-3, 0.95 * od), self.seg_len.value() * 1e-3,
                             self.segs.value(), self.inhibit.isChecked())

    def set_setup(self, setup) -> None:
        self._setup = setup
        if setup is None:
            return
        if self._sized_for != setup.geom.throat_diameter:          # a new nozzle: re-proportion the grain
            self.auto_size()
        elif self.isVisible():
            self._timer.start()

    def auto_size(self) -> None:
        s = self._setup
        if s is None or s.planar:
            return
        g = s.geom
        dt = 2.0 * g.rt
        od = max(2.0 * g.rc, 3.0 * dt)
        core = min(max(1.6 * dt, 0.3 * od), 0.6 * od)
        self._loading = True
        try:
            self.od.setValue(od * 1e3)
            self.core.setValue(core * 1e3)
            self.seg_len.setValue(1.5 * od * 1e3)
            n = sm.segments_for_pressure(self.grain(), self.propellant(), g, None, s.pc, self.eff.value() / 100.0)
            self.segs.setValue(min(max(n, 1), self.segs.maximum()))
        finally:
            self._loading = False
        self._sized_for = g.throat_diameter
        self.refresh()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._timer.start()

    def refresh(self) -> None:
        s = self._setup
        if s is None:
            return
        if s.planar:
            self.info.setText("The solid-motor model needs a round (axisymmetric) nozzle.")
            self.result = None
            self.curve_changed.emit(None)
            return
        grain, prop = self.grain(), self.propellant()
        r = sm.simulate(grain, prop, s.geom, None, s.pa, self.eff.value() / 100.0)
        self.result = r
        self.c_f.setData(r.t, r.thrust / 1e3)
        self.c_p.setData(r.t, r.pc / 1e5)
        self.c_kn.setData(r.t, r.kn / 10.0)
        lines = [f"Motor {r.designation}: total impulse {r.total_impulse:,.1f} N·s, burn {r.burn_time:.2f} s",
                 f"Thrust: average {r.average_thrust:,.1f} N, peak {r.peak_thrust:,.1f} N",
                 f"Pressure: peak {r.peak_pressure / 1e5:.1f} bar; Kn {r.kn[0]:.0f} → {r.kn.max():.0f} (max)",
                 f"Propellant {r.propellant_mass:,.3f} kg; delivered Isp {r.isp:.0f} s at {s.pa / 1e3:.1f} kPa",
                 f"Port / throat {r.port_to_throat:.2f}; volumetric loading {r.volumetric_loading * 100:.0f} %"]
        lines += ["⚠ " + w for w in r.warnings]
        self.info.setText("\n".join(lines))
        self.curve_changed.emit((r, grain))

    def export_eng(self) -> None:
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export motor file", str(get_paths().exports /
                                                                              f"{self.result.designation}.eng"),
                                              "RASP motor file (*.eng)")
        if path:
            sm.write_eng(self.result, self.grain(), path)


# ---------------------------------------------------------------------------
# Flight (single-stage ascent)
# ---------------------------------------------------------------------------
class FlightTab(QWidget):
    """Fly a single-stage rocket on the Lab's engine (or the Motor tab's thrust curve)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.source = QComboBox()
        self.source.addItem("Nozzle Lab engine (steady, thrust follows the altitude)", "engine")
        self.source.addItem("Solid motor (Motor tab thrust curve)", "motor")
        self.engines = QSpinBox()
        self.engines.setRange(1, 40)
        self.engines.setValue(1)
        self.dry = _spin(0.001, 1e7, 1000.0, 3, "kg", 10.0, "Dry mass: structure, engines and payload")
        self.prop_m = _spin(0.0, 1e8, 9000.0, 3, "kg", 100.0, "Propellant mass (a solid motor sets its own)")
        self.dia = _spin(0.005, 20.0, 1.0, 3, "m", 0.05, "Body diameter (drag reference area)")
        self.cd = _spin(0.05, 2.0, 0.35, 2, "", 0.01, "Subsonic drag coefficient; a transonic rise is added")
        self.angle = _spin(10.0, 90.0, 89.0, 1, "°", 0.5, "Launch angle above the horizon, held until the pitch-over "
                           "speed")
        self.turn = _spin(0.0, 2000.0, 120.0, 0, "m/s", 10.0, "Speed at which the gravity turn starts (thrust then "
                          "follows the velocity). With a low thrust-to-weight ratio small changes matter a lot.")
        self.btn_size = button("Size the rocket", "ghost", "", "Masses for a lift-off thrust-to-weight ratio of 1.5 "
                               "and a 90 % propellant fraction (liquid engine), or 5 (solid motor, like a sounding "
                               "rocket)")
        self.btn_size.clicked.connect(lambda: (setattr(self, "_user_mass", False), self.auto_size()))
        controls = _grid([("Engines", self.engines), ("Dry mass", self.dry), ("Propellant", self.prop_m),
                          ("Diameter", self.dia), ("Drag Cd", self.cd), ("Launch angle", self.angle),
                          ("Pitch-over at", self.turn)])
        # optional upper stage: the Lab's engine with a vacuum nozzle
        self.t_upper = ToggleSwitch("Upper stage", False, "Add a second stage that fires after the first burns out "
                                    "(the Lab's engine with a larger vacuum nozzle)")
        self.up_dry = _spin(0.001, 1e7, 1000.0, 3, "kg", 10.0, "Upper stage dry mass, payload included")
        self.up_prop = _spin(0.0, 1e8, 5000.0, 3, "kg", 100.0, "Upper stage propellant")
        self.up_eps = _spin(2.0, 400.0, 100.0, 0, "", 5.0, "Expansion ratio of the upper stage's vacuum nozzle")
        self.up_engines = QSpinBox()
        self.up_engines.setRange(1, 10)
        self.t_horiz = ToggleSwitch("Steer horizontally", True, "The upper stage thrusts along the local horizontal "
                                    "to build orbital speed (otherwise along its velocity)")
        self.upper_box = _grid([("Dry + payload", self.up_dry), ("Propellant", self.up_prop),
                                ("Vacuum ε", self.up_eps), ("Engines", self.up_engines)])
        self.t_upper.toggled.connect(lambda on: (self.upper_box.setVisible(on), self.t_horiz.setVisible(on),
                                                 self._resize_or_run()))
        self.upper_box.setVisible(False)
        self.t_horiz.setVisible(False)
        self.p_h = _plot("time (s)", "altitude (km)", legend=False)
        self.c_h = self.p_h.plot(pen=pg.mkPen(PALETTE.cyan, width=2))
        self.p_v = _plot("time (s)", "speed (m/s)", legend=False)
        self.c_v = self.p_v.plot(pen=pg.mkPen(PALETTE.warning, width=2))
        self.p_q = _plot("time (s)", "dynamic pressure (kPa)", legend=False)
        self.c_q = self.p_q.plot(pen=pg.mkPen(PALETTE.danger, width=2))
        self.p_path = _plot("downrange (km)", "altitude (km)", legend=False)
        self.c_path = self.p_path.plot(pen=pg.mkPen(PALETTE.success, width=2))
        for pw in (self.p_v, self.p_q):
            pw.setXLink(self.p_h)
        self.burn_lines = [pg.InfiniteLine(angle=90, pen=pg.mkPen("#FFFFFF", width=1, style=Qt.PenStyle.DotLine))
                           for _ in range(3)]
        self.stage_lines = [pg.InfiniteLine(angle=90, pen=pg.mkPen(PALETTE.warning, width=1,
                                                                   style=Qt.PenStyle.DashLine)) for _ in range(3)]
        for pw, ln, sl in zip((self.p_h, self.p_v, self.p_q), self.burn_lines, self.stage_lines):
            pw.addItem(ln, ignoreBounds=True)
            pw.addItem(sl, ignoreBounds=True)
        charts = QWidget()
        cg = QGridLayout(charts)
        cg.setContentsMargins(0, 0, 0, 0)
        for k, (title, pw) in enumerate((("Altitude", self.p_h), ("Speed", self.p_v), ("Dynamic pressure", self.p_q),
                                         ("Flight path", self.p_path))):
            cg.addWidget(_card(title, pw), k // 2, k % 2)
        self.info = QLabel("")
        self.info.setFont(mono_font(10))
        self.info.setWordWrap(True)
        note = _muted("Point-mass flight over a round, non-rotating Earth, ISA atmosphere, drag with a transonic "
                      "rise, thrust at each altitude from the nozzle theory. A gravity turn after pitch-over, no "
                      "throttling; an optional upper stage. A quick feasibility check, not a mission design.")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(_scroll_page(labeled_row("Thrust from", self.source), controls, self.t_upper, self.upper_box,
                                   self.t_horiz, self.btn_size, (charts, 420), self.info, note))
        for w in (self.up_dry, self.up_prop):
            w.valueChanged.connect(lambda _v: (setattr(self, "_user_mass", True), self._timer.start()))
        self.up_eps.valueChanged.connect(lambda _v: self._timer.start())
        self.up_engines.valueChanged.connect(lambda _v: self._timer.start())
        self.t_horiz.toggled.connect(lambda _v: self._timer.start())
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self.refresh)
        for w in (self.dia, self.cd, self.angle, self.turn):
            w.valueChanged.connect(lambda _v: self._timer.start())
        for w in (self.dry, self.prop_m):                      # the masses are the user's from now on
            w.valueChanged.connect(lambda _v: (setattr(self, "_user_mass", True), self._timer.start()))
        self.engines.valueChanged.connect(lambda _v: self._resize_or_run())
        self.source.currentIndexChanged.connect(lambda _i: (self._sync(), self._resize_or_run()))
        self._user_mass = False
        self._setup = None
        self._motor = None
        self._sized_for = None
        self.flight: Optional[tj.Flight] = None
        self._sync()

    def _sync(self) -> None:
        motor = self.source.currentData() == "motor"
        self.prop_m.setEnabled(not motor)

    def _resize_or_run(self) -> None:
        if not self._user_mass:
            self.auto_size()
        self._timer.start()

    def set_setup(self, setup) -> None:
        self._setup = setup
        key = None if setup is None else (setup.geom, setup.gas, setup.pc)
        if setup is not None and key != self._sized_for and not self._user_mass:
            self.auto_size()                                     # a new engine: a rocket sized for it
        self._sized_for = key
        if self.isVisible():
            self._timer.start()

    def set_motor(self, motor) -> None:
        self._motor = motor
        if self.source.currentData() == "motor":
            self._resize_or_run()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._timer.start()

    def _thrust_source(self):
        """(thrust_at(t, h), lift-off thrust per engine, propellant mass or None)."""
        s = self._setup
        if self.source.currentData() == "motor":
            if self._motor is None:
                return None
            r, _grain = self._motor
            ae = math.pi * s.geom.re ** 2
            p0 = isa_atmosphere(0.0).pressure_pa
            t_c, f_c, md_c = r.t, r.thrust, r.mdot

            def thrust_at(t: float, h: float) -> tuple[float, float]:
                if t > t_c[-1]:
                    return 0.0, 0.0
                f = float(np.interp(t, t_c, f_c))
                if f <= 0.0:
                    return 0.0, float(np.interp(t, t_c, md_c))
                return f + (p0 - tj.atmosphere(h)[1]) * ae, float(np.interp(t, t_c, md_c))

            return thrust_at, float(f_c.max()), r.propellant_mass

        def theory(pa: float) -> tuple[float, float]:
            th = nz.nozzle_theory(s.geom, s.gas, s.pc, pa, False)
            return max(th.thrust, 0.0), th.mass_flow

        f0 = theory(isa_atmosphere(0.0).pressure_pa)[0]
        return tj.engine_thrust_table(theory), f0, None

    def auto_size(self) -> None:
        s = self._setup
        if s is None or s.planar:
            return
        src = self._thrust_source()
        if src is None:
            return
        _f, f0, m_prop = src
        m0 = self.engines.value() * f0 / ((1.5 if m_prop is None else 5.0) * tj.G0)
        self.dry.blockSignals(True)
        self.prop_m.blockSignals(True)
        if m_prop is None and self.t_upper.isChecked():
            # a typical two-stage split: the upper stage is about a fifth of the lift-off mass
            m_up = 0.2 * m0
            self.up_dry.blockSignals(True)
            self.up_prop.blockSignals(True)
            self.up_prop.setValue(0.85 * m_up)
            self.up_dry.setValue(0.15 * m_up)
            self.up_dry.blockSignals(False)
            self.up_prop.blockSignals(False)
            self.prop_m.setValue(0.94 * (m0 - m_up))
            self.dry.setValue(0.06 * (m0 - m_up))
        elif m_prop is None:
            self.prop_m.setValue(0.9 * m0)
            self.dry.setValue(0.1 * m0)
        else:
            self.dry.setValue(max(m0 - m_prop * self.engines.value(), 0.3 * m_prop * self.engines.value()))
        self.dia.setValue(max(round(2.2 * s.geom.re * max(self.engines.value(), 1) ** 0.5, 3), 0.005))
        self.dry.blockSignals(False)
        self.prop_m.blockSignals(False)
        self._timer.start()

    def refresh(self) -> None:
        s = self._setup
        if s is None or not self.isVisible():
            return
        if s.planar:
            self.info.setText("The flight model needs a round (axisymmetric) engine.")
            return
        src = self._thrust_source()
        if src is None:
            self.info.setText("Open the Motor tab first: it makes the solid motor's thrust curve.")
            return
        thrust_at, _f0, m_prop = src
        n = self.engines.value()
        prop = m_prop * n if m_prop is not None else self.prop_m.value()
        v = tj.Vehicle(self.dry.value(), prop, self.dia.value(), self.cd.value(), n, self.angle.value(),
                       turn_speed=self.turn.value())
        upper = None
        if self.t_upper.isChecked():
            kind = s.geom.contour
            if kind in (nz.Contour.SONIC, nz.Contour.AEROSPIKE, nz.Contour.CUSTOM):
                kind = nz.Contour.BELL
            vac = replace(s.geom, contour=kind, exit_diameter=2.0 * s.geom.rt * math.sqrt(self.up_eps.value()))

            def vac_theory(pa: float) -> tuple[float, float]:
                th = nz.nozzle_theory(vac, s.gas, s.pc, pa, False)
                return max(th.thrust, 0.0), th.mass_flow

            upper = tj.UpperStage(self.up_dry.value(), self.up_prop.value(), tj.engine_thrust_table(vac_theory),
                                  self.up_engines.value(), horizontal=self.t_horiz.isChecked())
        fl = tj.fly(v, thrust_at, upper)
        self.flight = fl
        self.c_h.setData(fl.t, fl.altitude / 1e3)
        self.c_v.setData(fl.t, fl.speed)
        self.c_q.setData(fl.t, fl.q / 1e3)
        self.c_path.setData(fl.downrange / 1e3, fl.altitude / 1e3)
        for ln in self.burn_lines:
            ln.setPos(fl.burnout_time)
        for ln in self.stage_lines:
            ln.setVisible(fl.staging is not None)
            if fl.staging is not None:
                ln.setPos(fl.staging[0])
        m0 = v.dry_mass + v.propellant_mass + (upper.dry_mass + upper.propellant_mass if upper else 0.0)
        lines = [fl.message,
                 f"Lift-off mass {m0:,.1f} kg, thrust-to-weight {fl.liftoff_twr:.2f}",
                 f"Burnout at {fl.burnout_time:.1f} s: {fl.burnout_altitude / 1e3:,.2f} km, {fl.burnout_speed:,.0f} m/s",
                 f"Apogee {fl.apogee / 1e3:,.2f} km at {fl.apogee_time:.0f} s, {fl.downrange_at_apogee / 1e3:,.1f} km "
                 "downrange",
                 f"Max-Q {fl.max_q / 1e3:,.1f} kPa at {fl.max_q_altitude / 1e3:.1f} km; peak acceleration "
                 f"{fl.max_accel_g:.1f} g",
                 f"Δv: ideal {fl.dv_ideal:,.0f} m/s - gravity loss {fl.gravity_loss:,.0f} - drag loss "
                 f"{fl.drag_loss:,.0f}"]
        if fl.staging is not None:
            lines.insert(3, f"Staging at {fl.staging[0]:.1f} s: {fl.staging[1] / 1e3:,.1f} km, {fl.staging[2]:,.0f} m/s")
        if fl.orbit is None and fl.burnout_speed > 7600.0 and fl.burnout_altitude > 150e3:
            lines.append("Fast enough for orbit at burnout, but not on a closed orbit above 100 km: steer flatter.")
        self.info.setText("\n".join(lines))


# ---------------------------------------------------------------------------
# Compare two engines
# ---------------------------------------------------------------------------
class CompareTab(QWidget):
    """Pin an engine; then compare it with the current one (theory and the last simulated values)."""

    def __init__(self, get_snapshot: Callable[[], dict], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._get = get_snapshot
        self.pinned: Optional[dict] = None
        self.btn_pin = button("Pin this engine as A", "primary", "", "Keep the current engine (settings, theory and "
                                                                    "simulated results) for comparison")
        self.btn_pin.clicked.connect(self.pin)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["", "A (pinned)", "B (current)", "B − A"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setFont(mono_font(10))
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for k in (1, 2, 3):
            hh.setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
        self.p = _plot("altitude (km)", "thrust (kN)")
        self.c_a = self.p.plot(name="A", pen=pg.mkPen(PALETTE.warning, width=2))
        self.c_b = self.p.plot(name="B", pen=pg.mkPen(PALETTE.cyan, width=2))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self.btn_pin)
        lay.addWidget(self.table, 2)
        lay.addWidget(_card("Thrust vs altitude (1-D theory)", self.p), 1)
        lay.addWidget(_muted("Pin an engine, change anything (propellant, nozzle, chamber pressure...) and compare. "
                             "Simulated values are those shown when each was taken."))

    def pin(self) -> None:
        self.pinned = self._get()
        self.btn_pin.setText(f"Pin this engine as A (pinned: {self.pinned['label']})")
        self.refresh()

    def refresh(self) -> None:
        cur = self._get()
        a = self.pinned
        rows = cur["rows"]
        self.table.setRowCount(len(rows))
        for r, (name, vb) in enumerate(rows):
            va = a["values"].get(name) if a else None
            cells = [name, _fmt(va), _fmt(vb), _fmt(vb - va) if isinstance(va, float) and isinstance(vb, float) else ""]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, item)
        self.c_b.setData(*cur["altitude_curve"])
        if a:
            self.c_a.setData(*a["altitude_curve"])
        else:
            self.c_a.setData([], [])


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "—"
        return f"{v:,.4g}" if abs(v) < 1e5 else f"{v:,.0f}"
    return str(v)


def altitude_curve(setup) -> tuple[np.ndarray, np.ndarray]:
    alts = np.linspace(0.0, 60.0, 41)
    f = [nz.nozzle_theory(setup.geom, setup.gas, setup.pc, isa_atmosphere(a * 1e3).pressure_pa, setup.planar).thrust
         / 1e3 for a in alts]
    return alts, np.asarray(f)


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------
def _png_b64(img: QImage) -> str:
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    return base64.b64encode(bytes(buf.data())).decode("ascii")


def plot_image(plot: pg.PlotWidget, width: int = 900) -> Optional[str]:
    """A chart as base64 PNG (works for charts on hidden tabs too)."""
    try:
        from pyqtgraph.exporters import ImageExporter

        ex = ImageExporter(plot.plotItem)
        ex.parameters()["width"] = width
        img = ex.export(toBytes=True)
        return _png_b64(img) if isinstance(img, QImage) and not img.isNull() else None
    except Exception:  # noqa: BLE001 - a chart that cannot be exported is left out
        return None


def widget_image(widget: QWidget) -> Optional[str]:
    try:
        img = widget.grab().toImage()
        return None if img.isNull() else _png_b64(img)
    except Exception:  # noqa: BLE001
        return None


def build_report_html(title: str, sections: list[tuple[str, str]], images: list[tuple[str, Optional[str]]],
                      tables: list[tuple[str, list[list[str]]]]) -> str:
    """A self-contained HTML report (inline CSS, base64 images) that Qt WebEngine prints to PDF."""
    css = """
    body { font-family: 'Segoe UI', Arial, sans-serif; color: #111827; margin: 0; font-size: 10.5pt; }
    h1 { font-size: 19pt; margin: 0 0 2px 0; } h2 { font-size: 13pt; margin: 18px 0 6px 0; color: #0E7490;
    border-bottom: 1px solid #CBD5E1; padding-bottom: 3px; }
    .meta { color: #64748B; font-size: 9pt; margin-bottom: 10px; }
    table { border-collapse: collapse; width: 100%; font-size: 9.5pt; }
    td, th { border-bottom: 1px solid #E2E8F0; padding: 3px 6px; text-align: right; }
    td:first-child, th:first-child { text-align: left; } th { background: #F1F5F9; }
    img { width: 100%; border: 1px solid #E2E8F0; border-radius: 4px; margin: 4px 0 2px 0; }
    .cap { color: #475569; font-size: 9pt; margin-bottom: 10px; } p { margin: 4px 0; }
    .pb { page-break-before: always; }
    """
    out = [f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>",
           f"<h1>{html.escape(title)}</h1>",
           f"<div class='meta'>{html.escape(APP_NAME)} {html.escape(APP_VERSION)} · "
           f"{time.strftime('%Y-%m-%d %H:%M')}</div>"]
    for head, body in sections:
        out.append(f"<h2>{html.escape(head)}</h2>{body}")
    for name, rows in tables:
        out.append(f"<h2>{html.escape(name)}</h2><table>")
        for k, row in enumerate(rows):
            tag = "th" if k == 0 else "td"
            out.append("<tr>" + "".join(f"<{tag}>{html.escape(str(c))}</{tag}>" for c in row) + "</tr>")
        out.append("</table>")
    first = True
    for cap, b64 in images:
        if not b64:
            continue
        cls = " class='pb'" if first else ""
        first = False
        out.append(f"<h2{cls}>{html.escape(cap)}</h2><img src='data:image/png;base64,{b64}'>")
    out.append("</body></html>")
    return "\n".join(out)


def write_report(path_pdf: Path, html_text: str, done: Callable[[Optional[Path], str], None]) -> Path:
    """Save the HTML next to the PDF and print it to PDF (asynchronously; ``done`` reports the result)."""
    from ui.pdf_export import html_to_pdf

    html_path = path_pdf.with_suffix(".html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_text, encoding="utf-8")
    html_to_pdf(html_path, path_pdf, done)
    return html_path


__all__ = ["HeatTab", "DesignTab", "NoiseTab", "CompareTab", "altitude_curve", "plot_image", "widget_image",
           "build_report_html", "write_report"]
