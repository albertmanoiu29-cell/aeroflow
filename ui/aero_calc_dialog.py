"""Aero calculator: the standard atmosphere at an altitude, flight-condition numbers (Mach, Reynolds, dynamic
pressure, equivalent airspeed) and the lift / stall speed of a wing."""

from __future__ import annotations

import math
from typing import Optional

from PySide6.QtWidgets import QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from config.settings import APP_NAME, isa_atmosphere
from config.theme import mono_font

G0 = 9.80665
SPEED_UNITS = {"m/s": 1.0, "km/h": 1.0 / 3.6, "knots": 0.514444, "mph": 0.44704, "Mach": None}


def flight_numbers(altitude_m: float, speed_ms: float, length_m: float, dt_isa: float = 0.0) -> dict:
    """Atmosphere and flight-condition numbers (SI)."""
    st = isa_atmosphere(altitude_m, dt_isa)
    sl = isa_atmosphere(0.0)
    q = 0.5 * st.density * speed_ms ** 2
    return {"T": st.temperature_k, "p": st.pressure_pa, "rho": st.density, "a": st.speed_of_sound,
            "mu": st.dynamic_viscosity, "nu": st.kinematic_viscosity, "mach": speed_ms / st.speed_of_sound,
            "re": speed_ms * length_m / st.kinematic_viscosity, "q": q,
            "eas": speed_ms * math.sqrt(st.density / sl.density), "sigma": st.density / sl.density}


def _spin(lo: float, hi: float, val: float, dec: int, suffix: str, step: float) -> QDoubleSpinBox:
    sp = QDoubleSpinBox()
    sp.setRange(lo, hi)
    sp.setDecimals(dec)
    sp.setSingleStep(step)
    sp.setValue(val)
    if suffix:
        sp.setSuffix(" " + suffix)
    return sp


class AeroCalculator(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Aero calculator — {APP_NAME}")
        self.resize(560, 640)
        self.alt = _spin(-500.0, 80000.0, 0.0, 0, "m", 100.0)
        self.dt = _spin(-60.0, 60.0, 0.0, 1, "K", 1.0)
        self.dt.setToolTip("Temperature offset from the standard day (ISA + ΔT)")
        self.speed = _spin(0.0, 100000.0, 50.0, 2, "", 1.0)
        self.unit = QComboBox()
        self.unit.addItems(list(SPEED_UNITS))
        sp = QHBoxLayout()
        sp.addWidget(self.speed, 1)
        sp.addWidget(self.unit)
        spw = QWidget()
        spw.setLayout(sp)
        self.length = _spin(0.0001, 1000.0, 1.0, 4, "m", 0.1)
        self.length.setToolTip("Reference length for the Reynolds number (chord, body length…)")
        self.area = _spin(0.0001, 10000.0, 16.0, 3, "m²", 0.5)
        self.mass = _spin(0.001, 1e7, 1000.0, 3, "kg", 10.0)
        self.clmax = _spin(0.1, 5.0, 1.5, 2, "", 0.05)
        self.cl = _spin(-2.0, 5.0, 0.5, 3, "", 0.05)
        form = QFormLayout()
        form.addRow("Altitude", self.alt)
        form.addRow("ISA + ΔT", self.dt)
        form.addRow("Speed", spw)
        form.addRow("Reference length", self.length)
        form.addRow("Wing area", self.area)
        form.addRow("Mass", self.mass)
        form.addRow("CL", self.cl)
        form.addRow("CL,max", self.clmax)
        self.out = QLabel("")
        self.out.setFont(mono_font(10))
        self.out.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.out, 1)
        for w in (self.alt, self.dt, self.speed, self.length, self.area, self.mass, self.clmax, self.cl):
            w.valueChanged.connect(lambda _v: self.recompute())
        self.unit.currentTextChanged.connect(lambda _t: self.recompute())
        self.recompute()

    def speed_ms(self) -> float:
        f = SPEED_UNITS[self.unit.currentText()]
        if f is None:
            return self.speed.value() * isa_atmosphere(self.alt.value(), self.dt.value()).speed_of_sound
        return self.speed.value() * f

    def recompute(self) -> None:
        v = self.speed_ms()
        n = flight_numbers(self.alt.value(), v, self.length.value(), self.dt.value())
        w = self.mass.value() * G0
        s = self.area.value()
        lift = n["q"] * s * self.cl.value()
        v_stall = math.sqrt(2.0 * w / (n["rho"] * s * self.clmax.value()))
        cl_req = w / max(n["q"] * s, 1e-9)
        regime = ("incompressible" if n["mach"] < 0.3 else "subsonic, compressible" if n["mach"] < 0.8 else
                  "transonic" if n["mach"] < 1.2 else "supersonic" if n["mach"] < 5 else "hypersonic")
        lines = [
            "Atmosphere (ISA)",
            f"  temperature   {n['T']:8.2f} K   ({n['T'] - 273.15:+.1f} °C)",
            f"  pressure      {n['p']:10.1f} Pa  ({n['p'] / 101325:.4f} atm)",
            f"  density       {n['rho']:10.5f} kg/m³  (σ = {n['sigma']:.4f})",
            f"  speed of sound {n['a']:8.2f} m/s",
            f"  viscosity     {n['mu']:.4e} Pa·s · ν {n['nu']:.4e} m²/s",
            "",
            "Flight condition",
            f"  true airspeed {v:9.2f} m/s = {v * 3.6:,.1f} km/h = {v / 0.514444:,.1f} kt",
            f"  equivalent    {n['eas']:9.2f} m/s EAS",
            f"  Mach          {n['mach']:9.4f}  ({regime})",
            f"  Reynolds      {n['re']:12,.0f}",
            f"  dynamic pressure q = {n['q']:,.1f} Pa",
            "",
            "Wing",
            f"  lift at CL {self.cl.value():.2f}: {lift:,.1f} N ({lift / G0:,.1f} kgf); weight {w:,.1f} N",
            f"  CL needed for level flight: {cl_req:.3f}",
            f"  stall speed at CL,max {self.clmax.value():.2f}: {v_stall:.2f} m/s TAS = {v_stall * 3.6:.1f} km/h",
            f"  wing loading {w / s:,.1f} N/m² ({self.mass.value() / s:,.1f} kg/m²)",
        ]
        self.out.setText("\n".join(lines))
