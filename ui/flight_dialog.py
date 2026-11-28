"""Flight dynamics of the current aircraft (from its polar): trim, dynamic modes and a response simulation."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)

from config.settings import APP_NAME
from config.theme import PALETTE, mono_font
from core.flight import Aircraft, FlightModel, default_radius_of_gyration, handling_notes
from core.results import PolarData
from ui.cfd_panel import button
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


class FlightDialog(QDialog):
    def __init__(self, polar: PolarData, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Flight dynamics — {polar.geometry_name or 'aircraft'} — {APP_NAME}")
        self.resize(1200, 780)
        self.polar = polar
        v0 = polar.velocity_ms if polar.velocity_ms > 1 else 30.0
        # a mass that trims near the polar's speed at a moderate CL
        mass0 = max(0.5 * 1.225 * v0 * v0 * polar.ref_area_m2 * 0.4 / 9.80665, 0.1)
        self.mass = _spin(0.01, 1e6, round(mass0, 2), 2, "kg", 10.0, "Take-off mass")
        self.speed = _spin(1.0, 400.0, v0, 1, "m/s", 1.0, "Airspeed to trim at")
        self.alt = _spin(0.0, 15000.0, 0.0, 0, "m", 100.0)
        self.ky = _spin(0.01, 100.0, round(default_radius_of_gyration(polar), 2), 2, "m", 0.1,
                        "Pitch radius of gyration (I_yy = m k²); about 0.3 × span for light aircraft")
        self.cg = _spin(-50.0, 50.0, 0.0, 1, "% chord", 1.0, "CG position behind the polar's moment reference "
                        "(+ aft: less stable)")
        self.cmd = _spin(-0.2, -0.001, -0.02, 3, "/°", 0.002, "Elevator effectiveness Cm per degree")
        self.cmq = _spin(-100.0, 0.0, -15.0, 1, "", 1.0, "Pitch damping Cm_q (mostly from the tail)")
        # derivatives from the tail geometry (instead of typed-in values)
        from ui.cfd_panel import ToggleSwitch

        self.t_est = ToggleSwitch("Estimate Cm_δe and Cm_q from the tail", True,
                                  "Tail-volume method: Cm_q = −2 η V_H a_t l_t / c̄, Cm_δe = −η V_H a_t τ")
        self.vh = _spin(0.2, 1.5, 0.6, 2, "", 0.05, "Horizontal tail volume coefficient V_H = S_h l_t / (S c̄)")
        self.lt = _spin(1.0, 12.0, 3.5, 1, "c̄", 0.25, "Tail arm in mean chords")
        self.ce = _spin(0.1, 0.6, 0.3, 2, "", 0.05, "Elevator chord / tail chord")
        self.t_est.toggled.connect(lambda _v: self._estimate())
        for w in (self.vh, self.lt, self.ce):
            w.valueChanged.connect(lambda _v: self._estimate())
        self.kind = QComboBox()
        self.kind.addItem("Elevator pulse", "elevator")
        self.kind.addItem("Vertical gust", "gust")
        self.size = _spin(-15.0, 15.0, -2.0, 1, "°", 0.5, "Elevator deflection of the pulse, or the gust's change "
                          "of angle of attack")
        self.dur = _spin(5.0, 600.0, 60.0, 0, "s", 10.0)
        form = QFormLayout()
        for text, w in (("Mass", self.mass), ("Airspeed", self.speed), ("Altitude", self.alt),
                        ("Radius of gyration", self.ky), ("CG shift", self.cg), ("Elevator power", self.cmd),
                        ("Pitch damping", self.cmq), ("Tail volume V_H", self.vh), ("Tail arm", self.lt),
                        ("Elevator chord", self.ce), ("Disturbance", self.kind), ("Size", self.size),
                        ("Duration", self.dur)):
            form.addRow(text, w)
        self.out = QLabel("")
        self.out.setFont(mono_font(10))
        self.out.setWordWrap(True)
        form.insertRow(7, self.t_est)
        run = button("Simulate", "primary", "play", "Trim, find the modes and fly the response")
        run.clicked.connect(self.recompute)
        left = QVBoxLayout()
        left.addLayout(form)
        left.addWidget(run)
        left.addWidget(self.out, 1)
        note = QLabel("Longitudinal motion from the computed polar (CL, CD, Cm vs α, moved to the CG). Elevator power and "
                      "pitch damping are inputs (the polar holds neither). Thrust balances the trim drag, then stays "
                      "fixed.")
        note.setWordWrap(True)
        note.setProperty("role", "muted")
        left.addWidget(note)
        lw = QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(380)
        self.plots = {}
        grid = QGridLayout()
        for k, (key, title, ylab, colour) in enumerate((
                ("alpha", "Angle of attack and pitch attitude", "degrees", PALETTE.cyan),
                ("V", "Airspeed", "m/s", PALETTE.warning),
                ("h", "Altitude change", "m", PALETTE.success),
                ("q", "Pitch rate", "°/s", PALETTE.indigo))):
            pw = pg.PlotWidget()
            _style_plot(pw, "time (s)", ylab)
            if key == "alpha":
                pw.addLegend(offset=(-8, 8), labelTextColor=PALETTE.text_muted,
                             brush=pg.mkBrush(PALETTE.surface_alt + "B0"))
            self.plots[key] = (pw, pw.plot(name="α" if key == "alpha" else None, pen=pg.mkPen(colour, width=2)))
            grid.addWidget(_card(title, pw), k // 2, k % 2)
        self.c_theta = self.plots["alpha"][0].plot(name="θ", pen=pg.mkPen("#F472B6", width=1.5))
        for key in ("V", "h", "q"):
            self.plots[key][0].setXLink(self.plots["alpha"][0])
        root = QHBoxLayout(self)
        root.addWidget(lw)
        root.addLayout(grid, 1)
        self._estimate()
        self.recompute()

    def _estimate(self) -> None:
        """Cm_q and Cm_δe from the tail volume (tail lift slope from its aspect ratio ~4.5; elevator
        effectiveness τ from the elevator chord ratio, Raymer)."""
        on = self.t_est.isChecked()
        for w in (self.vh, self.lt, self.ce):
            w.setEnabled(on)
        self.cmd.setEnabled(not on)
        self.cmq.setEnabled(not on)
        if not on:
            return
        from core.aircraft_design import lift_slope

        a_t = lift_slope(4.5)
        eta = 0.9
        tau = min(1.0, 1.55 * self.ce.value() ** 0.5 - 0.2)            # elevator effectiveness (fit)
        cmq = -2.0 * eta * self.vh.value() * a_t * self.lt.value()
        cmd = -eta * self.vh.value() * a_t * tau * math.pi / 180.0
        self.cmq.blockSignals(True)
        self.cmd.blockSignals(True)
        self.cmq.setValue(max(min(cmq, 0.0), self.cmq.minimum()))
        self.cmd.setValue(min(max(cmd, self.cmd.minimum()), self.cmd.maximum()))
        self.cmq.blockSignals(False)
        self.cmd.blockSignals(False)

    def recompute(self) -> None:
        ac = Aircraft(self.mass.value(), self.ky.value(), self.cg.value() / 100.0, self.cmd.value(), self.cmq.value())
        fm = FlightModel(self.polar, ac, self.alt.value())
        tr = fm.trim(self.speed.value())
        if not tr.ok and tr.speed == 0.0:
            self.out.setText(tr.message)
            for _pw, c in self.plots.values():
                c.setData([], [])
            self.c_theta.setData([], [])
            return
        modes = fm.modes(tr)
        sim = fm.simulate(tr, self.dur.value(), 0.01 if self.dur.value() <= 120 else 0.02, self.kind.currentData(),
                          self.size.value())
        t = sim["t"]
        self.plots["alpha"][1].setData(t, sim["alpha_deg"])
        self.c_theta.setData(t, sim["theta_deg"])
        self.plots["V"][1].setData(t, sim["V"])
        self.plots["h"][1].setData(t, sim["h"])
        self.plots["q"][1].setData(t, sim["q_deg_s"])
        lines = [tr.message, "",
                 f"Trim  α {tr.alpha_deg:+.2f}°  elevator {tr.elevator_deg:+.2f}°",
                 f"      CL {tr.cl:.3f}  L/D {tr.cl / max(tr.cd, 1e-9):.1f}  thrust {tr.thrust:,.0f} N", ""]
        for m in modes:
            per = f"{m.period:6.1f} s" if np.isfinite(m.period) else "   —  "
            lines.append(f"{m.name:14s} ωn {m.omega_n:6.3f} rad/s  ζ {m.damping:+.3f}  period {per}")
        lines += [""] + handling_notes(modes)
        if len(t) and t[-1] < self.dur.value() - 1e-6:
            lines.append("The simulation diverged (stall or loss of speed).")
        self.out.setText("\n".join(lines))
