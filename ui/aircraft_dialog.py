"""Aircraft design workbench: sizing, tails and stability, performance with a real propeller (or a jet)."""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Optional

import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QScrollArea, QSpinBox,
    QVBoxLayout, QWidget,
)

from config.settings import APP_NAME
from config.theme import PALETTE, make_section_label, mono_font
from core import aircraft_design as ad
from visualization.telemetry_charts import _card, _style_plot


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


PRESETS = {
    "Light single (Cessna 172 class)": ad.AircraftSpec(),
    "Trainer / aerobatic": ad.AircraftSpec(mass=900, wing_loading=70, aspect_ratio=6.5, power_kw=150,
                                           prop_diameter=1.85, cd0=0.030, fuel_mass=110),
    "Motor glider": ad.AircraftSpec(mass=800, wing_loading=40, aspect_ratio=18, taper=0.45, cd0=0.016, cl_max=1.4,
                                    power_kw=60, prop_diameter=1.6, fuel_mass=60, tail_arm_h=5.5, tail_arm_v=5.3),
    "Business jet": ad.AircraftSpec(mass=9000, wing_loading=320, aspect_ratio=8.2, taper=0.35, cd0=0.020,
                                    cl_max=1.9, engine="jet", jet_thrust=2 * 15500, fuel_mass=2600, sfc=0.07,
                                    altitude=12000, tail_arm_h=7.0, tail_arm_v=6.5),
    "Large model aircraft (RC)": ad.AircraftSpec(mass=6.0, wing_loading=6.0, aspect_ratio=7.0, cd0=0.035,
                                                 cl_max=1.3, power_kw=1.5, prop_diameter=0.4, prop_pitch_ratio=0.6,
                                                 prop_rpm=7500, fuel_mass=0.5, sfc=0.8, altitude=100,
                                                 tail_arm_h=0.75, tail_arm_v=0.72),
}


class AircraftDialog(QDialog):
    FIELDS = (  # (attribute, label, lo, hi, decimals, suffix, step, tip)
        ("mass", "Take-off mass", 0.1, 600000.0, 1, "kg", 10.0, ""),
        ("wing_loading", "Wing loading", 1.0, 900.0, 1, "kg/m²", 5.0, "Mass per wing area: low for gliders, high for jets"),
        ("aspect_ratio", "Aspect ratio", 2.0, 40.0, 1, "", 0.5, "span² / area"),
        ("taper", "Taper ratio", 0.1, 1.0, 2, "", 0.05, "tip chord / root chord"),
        ("cd0", "Zero-lift drag CD0", 0.008, 0.1, 4, "", 0.001, "Whole aircraft (0.02-0.035 for light aircraft)"),
        ("cl_max", "CL,max (landing)", 0.8, 3.5, 2, "", 0.05, "Maximum lift coefficient with flaps"),
        ("tail_arm_h", "Horizontal tail arm", 0.1, 60.0, 2, "m", 0.1, "Wing a.c. to tail a.c."),
        ("tail_arm_v", "Vertical tail arm", 0.1, 60.0, 2, "m", 0.1, ""),
        ("v_h", "Horizontal tail volume", 0.2, 1.5, 2, "", 0.05, "V_H = S_h l_h / (S c̄): 0.5-0.8 typical"),
        ("v_v", "Vertical tail volume", 0.01, 0.15, 3, "", 0.005, "V_V = S_v l_v / (S b): 0.03-0.09 typical"),
        ("x_cg", "CG position", 0.0, 1.0, 2, "× MAC", 0.01, "From the leading edge of the mean aerodynamic chord"),
        ("power_kw", "Engine power", 0.05, 20000.0, 1, "kW", 5.0, "Propeller: shaft power"),
        ("prop_diameter", "Propeller diameter", 0.05, 8.0, 2, "m", 0.05, ""),
        ("prop_pitch_ratio", "Propeller P/D", 0.2, 2.0, 2, "", 0.05, ""),
        ("prop_rpm", "Propeller rpm", 100.0, 30000.0, 0, "rpm", 50.0, ""),
        ("jet_thrust", "Jet thrust (sea level)", 1.0, 1e6, 0, "N", 100.0, "Total static thrust (jet)"),
        ("fuel_mass", "Fuel", 0.0, 300000.0, 1, "kg", 5.0, ""),
        ("sfc", "Fuel consumption", 0.01, 2.0, 3, "", 0.01, "Piston: kg/(kW·h), about 0.3. Jet: kg/(N·h), about 0.07"),
        ("altitude", "Cruise altitude", 0.0, 15000.0, 0, "m", 250.0, ""),
    )

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Aircraft design — {APP_NAME}")
        self.resize(1320, 860)
        self.preset = QComboBox()
        for name in PRESETS:
            self.preset.addItem(name)
        self.engine = QComboBox()
        self.engine.addItem("Propeller (piston / electric)", "propeller")
        self.engine.addItem("Jet", "jet")
        self.blades = QSpinBox()
        self.blades.setRange(2, 6)
        self.spins: dict[str, QDoubleSpinBox] = {}
        form = QFormLayout()
        form.addRow("Start from", self.preset)
        form.addRow("Engine", self.engine)
        form.addRow("Propeller blades", self.blades)
        for attr, label, lo, hi, dec, suf, step, tip in self.FIELDS:
            sp = _spin(lo, hi, 0.0, dec, suf, step, tip)
            self.spins[attr] = sp
            form.addRow(label, sp)
        fw = QWidget()
        fw.setLayout(form)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(fw)
        scroll.setFixedWidth(370)
        self.out = QLabel("")
        self.out.setFont(mono_font(11))
        self.out.setWordWrap(True)
        self.out.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        # charts
        self.p_plan = pg.PlotWidget()
        _style_plot(self.p_plan, "y (m)", "x (m)")
        self.p_plan.setAspectLocked(True)
        self.p_plan.getPlotItem().invertY(True)
        self.c_wing = self.p_plan.plot(pen=pg.mkPen(PALETTE.text, width=1.5))
        self.c_tail = self.p_plan.plot(pen=pg.mkPen(PALETTE.cyan, width=1.5))
        self.c_cg = self.p_plan.plot(pen=None, symbol="o", symbolSize=10, symbolBrush=pg.mkBrush(PALETTE.warning),
                                     symbolPen=None)
        self.c_np = self.p_plan.plot(pen=None, symbol="x", symbolSize=12, symbolPen=pg.mkPen(PALETTE.success, width=2))
        self.p_t = pg.PlotWidget()
        _style_plot(self.p_t, "airspeed (m/s)", "thrust (N)")
        self.p_t.addLegend(offset=(-8, 8), labelTextColor=PALETTE.text_muted, brush=pg.mkBrush(PALETTE.surface_alt + "B0"))
        self.c_req = self.p_t.plot(name="required (drag)", pen=pg.mkPen(PALETTE.danger, width=2))
        self.c_av = self.p_t.plot(name="available", pen=pg.mkPen(PALETTE.success, width=2))
        self.p_roc = pg.PlotWidget()
        _style_plot(self.p_roc, "airspeed (m/s)", "rate of climb (m/s)")
        self.c_roc = self.p_roc.plot(pen=pg.mkPen(PALETTE.cyan, width=2))
        self.p_roc.setXLink(self.p_t)
        grid = QGridLayout()
        grid.addWidget(_card("Planform, tail, CG (●) and neutral point (×)", self.p_plan), 0, 0)
        grid.addWidget(_card("Results", self.out), 0, 1)
        grid.addWidget(_card("Thrust required and available (cruise altitude)", self.p_t), 1, 0)
        grid.addWidget(_card("Rate of climb", self.p_roc), 1, 1)
        note = QLabel("Conceptual sizing (Raymer / Anderson): parabolic polar, tail volume coefficients, neutral point "
                      "with downwash, BEM propeller thrust at the power limit, ideal Breguet range and endurance (best "
                      "speed, no reserves).")
        note.setWordWrap(True)
        note.setProperty("role", "muted")
        right = QVBoxLayout()
        right.addLayout(grid, 1)
        right.addWidget(note)
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        left.addWidget(make_section_label("Aircraft"))
        left.addWidget(scroll, 1)
        root.addLayout(left)
        root.addLayout(right, 1)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self.recompute)
        self._loading = False
        self.preset.currentTextChanged.connect(self._load_preset)
        for sp in self.spins.values():
            sp.valueChanged.connect(lambda _v: self._edited())
        self.engine.currentIndexChanged.connect(lambda _i: self._edited())
        self.blades.valueChanged.connect(lambda _v: self._edited())
        self.design: Optional[ad.AircraftDesign] = None
        self._load_preset(self.preset.currentText())

    def _edited(self) -> None:
        if not self._loading:
            self._sync()
            self._timer.start()

    def _sync(self) -> None:
        prop = self.engine.currentData() == "propeller"
        for k in ("power_kw", "prop_diameter", "prop_pitch_ratio", "prop_rpm"):
            self.spins[k].setEnabled(prop)
        self.blades.setEnabled(prop)
        self.spins["jet_thrust"].setEnabled(not prop)

    def _load_preset(self, name: str) -> None:
        spec = PRESETS.get(name)
        if spec is None:
            return
        self._loading = True
        try:
            for f in fields(spec):
                if f.name in self.spins:
                    self.spins[f.name].setValue(float(getattr(spec, f.name)))
            self.engine.setCurrentIndex(0 if spec.engine == "propeller" else 1)
            self.blades.setValue(spec.prop_blades)
        finally:
            self._loading = False
        self._sync()
        self.recompute()

    def spec(self) -> ad.AircraftSpec:
        kw = {k: sp.value() for k, sp in self.spins.items()}
        return ad.AircraftSpec(engine=self.engine.currentData(), prop_blades=self.blades.value(), **kw)

    def recompute(self) -> None:
        d = ad.design(self.spec())
        self.design = d
        # planform (top view, x aft): straight quarter-chord line
        half, cr = 0.5 * d.span, d.root_chord
        ct = cr * d.spec.taper
        xw = [-0.25 * cr, -0.25 * ct, 0.75 * ct, 0.75 * cr, 0.75 * ct, -0.25 * ct, -0.25 * cr]
        yw = [0.0, half, half, 0.0, -half, -half, 0.0]
        self.c_wing.setData(yw, xw)
        bh = math.sqrt(d.spec.ar_h * d.tail_h_area)
        ch = d.tail_h_area / max(bh, 1e-9)
        xt = d.spec.tail_arm_h
        self.c_tail.setData([0.0, 0.5 * bh, 0.5 * bh, 0.0, -0.5 * bh, -0.5 * bh, 0.0],
                            [xt - 0.25 * ch, xt - 0.25 * ch * 0.7, xt + 0.75 * ch * 0.7, xt + 0.75 * ch,
                             xt + 0.75 * ch * 0.7, xt - 0.25 * ch * 0.7, xt - 0.25 * ch])
        x_le_mac = -0.25 * d.mac
        self.c_cg.setData([0.0], [x_le_mac + d.spec.x_cg * d.mac])
        self.c_np.setData([0.0], [x_le_mac + d.neutral_point * d.mac])
        self.c_req.setData(d.speeds, d.thrust_required)
        self.c_av.setData(d.speeds, d.thrust_available)
        self.c_roc.setData(d.speeds, d.climb_rate)
        kt = 1.0 / 0.514444
        lines = [
            f"Wing   S {d.wing_area:.2f} m²  b {d.span:.2f} m  MAC {d.mac:.2f} m  root {d.root_chord:.2f} m",
            f"Tails  S_h {d.tail_h_area:.2f} m²  S_v {d.tail_v_area:.2f} m²",
            f"Polar  e {d.oswald:.3f}  K {d.k_induced:.4f}  (L/D)max {d.ld_max:.1f} at CL {d.cl_ld_max:.2f}",
            "",
            f"Stall  {d.stall_speed:.1f} m/s ({d.stall_speed * kt:.0f} kt) at sea level, CL,max {d.spec.cl_max:.2f}",
            f"Vmax   {d.v_max:.1f} m/s ({d.v_max * kt:.0f} kt) at {d.spec.altitude:.0f} m",
            f"Climb  {d.best_climb:.2f} m/s ({d.best_climb * 196.85:.0f} ft/min) at {d.best_climb_speed:.0f} m/s",
            f"Range  {d.range_km:,.0f} km   Endurance {d.endurance_h:.1f} h  (ideal Breguet)",
            "",
            f"Stability  neutral point {d.neutral_point * 100:.1f} % MAC, CG {d.spec.x_cg * 100:.1f} % MAC",
            f"           static margin {d.static_margin * 100:.1f} %",
            f"           CLα {d.a_wing:.2f}/rad  Cmα {d.cm_alpha:+.3f}/rad  Cm_q {d.cm_q:+.1f}",
            f"           dε/dα {d.downwash:.3f}",
            "",
        ] + d.notes
        self.out.setText("\n".join(lines))


__all__ = ["AircraftDialog", "PRESETS"]

