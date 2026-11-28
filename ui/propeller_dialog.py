"""Propeller analysis by blade-element momentum theory: performance at an operating point, CT / CP / efficiency
curves against advance ratio, and the load along the blade."""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDoubleSpinBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from config.settings import APP_NAME, isa_atmosphere
from config.theme import PALETTE, mono_font
from core import bem
from ui.cfd_panel import button
from visualization.telemetry_charts import _card, _style_plot


def _spin(lo: float, hi: float, val: float, dec: int, suffix: str, step: float) -> QDoubleSpinBox:
    sp = QDoubleSpinBox()
    sp.setRange(lo, hi)
    sp.setDecimals(dec)
    sp.setSingleStep(step)
    sp.setValue(val)
    sp.setKeyboardTracking(False)
    if suffix:
        sp.setSuffix(" " + suffix)
    return sp


class PropellerDialog(QDialog):
    """Fast propeller analysis (BEM) - the same geometry as the program's propeller preset."""

    def __init__(self, build_model: Optional[Callable[[float, int, float], None]] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Propeller analysis (blade-element momentum) — {APP_NAME}")
        self.resize(1180, 760)
        self._build_model = build_model
        self.d = _spin(0.05, 20.0, 1.0, 3, "m", 0.05)
        self.blades = QSpinBox()
        self.blades.setRange(2, 6)
        self.blades.setValue(2)
        self.pd = _spin(0.2, 2.5, 0.8, 2, "", 0.05)
        self.rpm = _spin(10.0, 60000.0, 2000.0, 0, "rpm", 100.0)
        self.v = _spin(0.0, 300.0, 15.0, 1, "m/s", 1.0)
        self.alt = _spin(0.0, 20.0, 0.0, 1, "km", 0.5)
        self.camber = _spin(0.0, 0.09, 0.04, 3, "", 0.005)
        self.chord = _spin(0.3, 3.0, 1.0, 2, "×", 0.1)
        form = QFormLayout()
        form.addRow("Diameter", self.d)
        form.addRow("Blades", self.blades)
        form.addRow("Pitch / diameter", self.pd)
        form.addRow("Chord scale", self.chord)
        form.addRow("Section camber", self.camber)
        form.addRow("Rotation speed", self.rpm)
        form.addRow("Flight speed", self.v)
        form.addRow("Altitude (ISA)", self.alt)
        self.result = QLabel("")
        self.result.setFont(mono_font(11))
        self.result.setWordWrap(True)
        left = QVBoxLayout()
        left.addLayout(form)
        left.addWidget(self.result)
        if build_model is not None:
            b = button("Build this propeller in 3-D", "ghost", "", "Load the same propeller into the wind tunnel "
                       "(for an OpenFOAM rotating-propeller run)")
            b.clicked.connect(lambda: self._build_model(self.d.value(), self.blades.value(), self.pd.value()))
            left.addWidget(b)
        note = QLabel("Blade-element momentum theory with Prandtl tip/hub losses and a generic cambered-airfoil model: "
                      "~10 % against measured data in the working range; no hub or spinner drag. For the full flow "
                      "use the wind tunnel with OpenFOAM and a rotation speed (MRF).")
        note.setWordWrap(True)
        note.setProperty("role", "muted")
        left.addWidget(note)
        left.addStretch(1)
        lw = QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(360)
        self.p_c = pg.PlotWidget()
        _style_plot(self.p_c, "advance ratio J = V / (n D)", "coefficient")
        self.p_c.addLegend(offset=(-8, 8), labelTextColor=PALETTE.text_muted, brush=pg.mkBrush(PALETTE.surface_alt + "B0"))
        self.c_ct = self.p_c.plot(name="CT", pen=pg.mkPen(PALETTE.cyan, width=2))
        self.c_cp = self.p_c.plot(name="CP", pen=pg.mkPen(PALETTE.warning, width=2))
        self.p_e = pg.PlotWidget()
        _style_plot(self.p_e, "advance ratio J", "efficiency η")
        self.c_eta = self.p_e.plot(pen=pg.mkPen(PALETTE.success, width=2))
        self.l_j = [pg.InfiniteLine(angle=90, pen=pg.mkPen("#FFFFFF", width=1, style=Qt.PenStyle.DashLine))
                    for _ in range(2)]
        self.p_c.addItem(self.l_j[0], ignoreBounds=True)
        self.p_e.addItem(self.l_j[1], ignoreBounds=True)
        self.p_s = pg.PlotWidget()
        _style_plot(self.p_s, "r / R", "thrust per metre of radius (N/m)")
        self.c_dt = self.p_s.plot(pen=pg.mkPen(PALETTE.cyan, width=2))
        self.p_a = pg.PlotWidget()
        _style_plot(self.p_a, "r / R", "angle of attack (°)")
        self.c_al = self.p_a.plot(pen=pg.mkPen(PALETTE.indigo, width=2))
        grid = QGridLayout()
        grid.addWidget(_card("CT and CP against advance ratio", self.p_c), 0, 0)
        grid.addWidget(_card("Propeller efficiency", self.p_e), 0, 1)
        grid.addWidget(_card("Thrust along the blade", self.p_s), 1, 0)
        grid.addWidget(_card("Section angle of attack", self.p_a), 1, 1)
        charts = QWidget()
        charts.setLayout(grid)
        self.tabs = QTabWidget()
        self.tabs.addTab(charts, "Performance")
        self.view_host = QWidget()
        QVBoxLayout(self.view_host).setContentsMargins(0, 0, 0, 0)
        self.tabs.addTab(self.view_host, "3-D: rotation and airflow")
        self.view3d = None
        self.tabs.currentChanged.connect(self._tab_changed)
        root = QHBoxLayout(self)
        root.addWidget(lw)
        root.addWidget(self.tabs, 1)
        for w in (self.d, self.pd, self.rpm, self.v, self.alt, self.camber, self.chord):
            w.valueChanged.connect(lambda _v: self.recompute())
        self.blades.valueChanged.connect(lambda _v: self.recompute())
        self.recompute()

    def _tab_changed(self, index: int) -> None:
        if index == 1 and self.view3d is None:
            from visualization.propeller_view import Propeller3DView

            self.view3d = Propeller3DView(self.view_host)
            self.view_host.layout().addWidget(self.view3d)
            self.recompute()
            self.view3d.start()

    def show_3d(self) -> None:
        self.tabs.setCurrentIndex(1)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.view3d is not None:
            self.view3d.shutdown()
        super().closeEvent(event)

    def _prop(self) -> bem.Propeller:
        return bem.Propeller(self.d.value(), self.blades.value(), self.pd.value(), camber=self.camber.value(),
                             chord_scale=self.chord.value())

    def recompute(self) -> None:
        import warnings

        prop = self._prop()
        rho = isa_atmosphere(self.alt.value() * 1e3).density
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = bem.solve(prop, self.rpm.value(), self.v.value(), rho)
            sw = bem.sweep_advance_ratio(prop, self.rpm.value(), rho)
        self.c_ct.setData(sw["J"], sw["CT"])
        self.c_cp.setData(sw["J"], sw["CP"])
        self.c_eta.setData(sw["J"], np.clip(sw["eta"], 0.0, 1.0))
        for line in self.l_j:
            line.setPos(res.J)
        rad = 0.5 * prop.diameter
        self.c_dt.setData(res.r / rad, res.dT_dr)
        if self.view3d is not None:
            from core.propeller_flow import slipstream

            self.view3d.set_propeller(prop.diameter, prop.blades, prop.pitch_ratio, slipstream(res, prop.diameter))
        self.c_al.setData(res.r / rad, res.alpha_deg)
        tip = math.pi * prop.diameter * self.rpm.value() / 60.0
        tip_mach = math.hypot(tip, self.v.value()) / isa_atmosphere(self.alt.value() * 1e3).speed_of_sound
        ideal = bem.ideal_efficiency(res.thrust, self.v.value(), rho, prop.diameter) if self.v.value() > 0.1 else 0.0
        k = int(np.argmax(sw["eta"]))
        self.result.setText(
            f"Thrust     {res.thrust:10.1f} N\nTorque     {res.torque:10.2f} N·m\nPower      {res.power / 1e3:10.2f} kW\n"
            f"Efficiency {res.efficiency:10.3f}  (ideal {ideal:.3f})\nJ          {res.J:10.3f}\n"
            f"CT / CP    {res.CT:.4f} / {res.CP:.4f}\nTip Mach   {tip_mach:10.2f}\n"
            f"Best η {float(sw['eta'][k]):.3f} at J {float(sw['J'][k]):.2f}"
            + ("\n⚠ part of the blade is stalled" if res.stalled_fraction > 0.05 else "")
            + ("\n⚠ tip Mach above 0.85: compressibility losses" if tip_mach > 0.85 else ""))
