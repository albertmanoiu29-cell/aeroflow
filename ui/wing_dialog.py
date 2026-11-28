"""Wing analysis: Prandtl lifting line with section data from the panel method."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog, QDoubleSpinBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QSlider, QVBoxLayout, QWidget,
)

from config.settings import APP_NAME
from config.theme import PALETTE, mono_font
from core import lifting_line as ll
from core import panel as pm
from core.shapes2d import naca_outline
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


def _plot(xl: str, yl: str, legend: bool = False) -> pg.PlotWidget:
    w = pg.PlotWidget()
    _style_plot(w, xl, yl)
    if legend:
        w.addLegend(offset=(-8, -8), labelTextColor=PALETTE.text_muted, brush=pg.mkBrush(PALETTE.surface_alt + "B0"))
    return w


class WingDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Wing analysis (lifting line) — {APP_NAME}")
        self.resize(1240, 800)
        self.span = _spin(0.05, 200.0, 10.0, 3, "m", 0.5)
        self.root = _spin(0.01, 50.0, 1.6, 3, "m", 0.05, "Root chord")
        self.taper = _spin(0.0, 1.0, 0.45, 2, "", 0.05, "Tip chord / root chord (0.4 is close to elliptic loading)")
        self.washout = _spin(-10.0, 15.0, 2.0, 1, "°", 0.5, "Tip twisted nose-down relative to the root (moves "
                             "the stall inboard, keeps the ailerons working)")
        self.code = QLineEdit("2412")
        self.code.setToolTip("NACA 4-digit section (its lift slope, zero-lift angle, CL,max and drag come from the "
                             "panel method)")
        self.re = _spin(1e4, 1e9, 3e6, 0, "", 1e5, "Chord Reynolds number for the section data")
        self.mach = _spin(0.0, 0.9, 0.0, 2, "", 0.05, "Flight Mach number (compressible section data)")
        self.alpha = QSlider(Qt.Orientation.Horizontal)
        self.alpha.setRange(-50, 200)
        self.alpha.setValue(60)
        self.alpha_lbl = QLabel("α = 6.0°")
        form = QFormLayout()
        for text, w in (("Span", self.span), ("Root chord", self.root), ("Taper ratio", self.taper),
                        ("Washout", self.washout), ("Section (NACA)", self.code), ("Reynolds number", self.re),
                        ("Mach number", self.mach)):
            form.addRow(text, w)
        form.addRow(self.alpha_lbl, self.alpha)
        self.info = QLabel("")
        self.info.setFont(mono_font(10))
        self.info.setWordWrap(True)
        note = QLabel("Prandtl lifting line (Glauert's Fourier solution): straight, unswept wings of moderate to high "
                      "aspect ratio (> 4). The wing stalls where the local Cl first reaches the section CL,max.")
        note.setWordWrap(True)
        note.setProperty("role", "muted")
        left = QVBoxLayout()
        left.addLayout(form)
        left.addWidget(self.info, 1)
        left.addWidget(note)
        lw = QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(380)
        self.p_plan = _plot("y (m)", "x (m)")
        self.p_plan.setAspectLocked(True)
        self.p_plan.getPlotItem().invertY(True)
        self.c_plan = self.p_plan.plot(pen=pg.mkPen(PALETTE.text, width=1.5), fillLevel=None)
        self.c_stall_mark = self.p_plan.plot(pen=None, symbol="x", symbolSize=14,
                                             symbolPen=pg.mkPen(PALETTE.danger, width=2))
        self.p_span = _plot("y (m)", "Cl", legend=True)
        self.c_local = self.p_span.plot(name="local Cl", pen=pg.mkPen(PALETTE.cyan, width=2))
        self.c_limit = self.p_span.plot(name="section CL,max", pen=pg.mkPen(PALETTE.danger, width=1,
                                                                            style=Qt.PenStyle.DashLine))
        self.c_ellip = self.p_span.plot(name="elliptic, same CL", pen=pg.mkPen(PALETTE.text_muted, width=1,
                                                                               style=Qt.PenStyle.DotLine))
        self.p_cl = _plot("α (°)", "CL", legend=True)
        self.c_cl = self.p_cl.plot(name="wing", pen=pg.mkPen(PALETTE.cyan, width=2))
        self.c_sec = self.p_cl.plot(name="section (2-D)", pen=pg.mkPen(PALETTE.text_muted, width=1,
                                                                       style=Qt.PenStyle.DashLine))
        self.c_st = self.p_cl.plot(name="wing stall", pen=None, symbol="x", symbolSize=12,
                                   symbolPen=pg.mkPen(PALETTE.danger, width=2))
        self.now = pg.InfiniteLine(angle=90, pen=pg.mkPen("#FFFFFF", width=1, style=Qt.PenStyle.DotLine))
        self.p_cl.addItem(self.now, ignoreBounds=True)
        self.p_pol = _plot("CD", "CL", legend=True)
        self.c_pol = self.p_pol.plot(name="total", pen=pg.mkPen(PALETTE.success, width=2))
        self.c_cdi = self.p_pol.plot(name="induced", pen=pg.mkPen(PALETTE.warning, width=1.5))
        grid = QGridLayout()
        grid.addWidget(_card("Planform", self.p_plan), 0, 0)
        grid.addWidget(_card("Spanwise lift", self.p_span), 0, 1)
        grid.addWidget(_card("Lift curve", self.p_cl), 1, 0)
        grid.addWidget(_card("Drag polar", self.p_pol), 1, 1)
        root = QHBoxLayout(self)
        root.addWidget(lw)
        root.addLayout(grid, 1)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self.recompute)
        for w in (self.span, self.root, self.taper, self.washout, self.re, self.mach):
            w.valueChanged.connect(lambda _v: self._timer.start())
        self.code.editingFinished.connect(self._timer.start)
        self.alpha.valueChanged.connect(lambda _v: self._show_alpha())
        self.wing: Optional[ll.Wing] = None
        self.section: Optional[pm.Polar] = None
        self._section_key = None
        self.recompute()

    def _section(self) -> Optional[pm.Polar]:
        key = (self.code.text().strip(), self.re.value(), self.mach.value())
        if key != self._section_key:
            try:
                nodes = pm.prepare(naca_outline(key[0], 200), 140)
            except Exception as exc:  # noqa: BLE001 - a malformed code
                self.info.setText(f"Not a NACA 4-digit code: {exc}")
                return None
            self.section = pm.polar(nodes, np.arange(-8.0, 24.01, 0.5), key[1], mach=key[2])
            self._section_key = key
        return self.section

    def recompute(self) -> None:
        sec = self._section()
        if sec is None:
            return
        cl_max = sec.cl_max if math.isfinite(sec.cl_max) else float(sec.cl.max())
        self.wing = w = ll.Wing(self.span.value(), self.root.value(), self.taper.value(), self.washout.value(),
                                sec.cl_alpha, sec.alpha_zero_lift, cl_max, float(sec.cd.min()))
        half, c_r, c_t = 0.5 * w.span, w.root_chord, w.tip_chord
        # unswept quarter-chord line: leading edge at -c/4
        xs = [-0.25 * c_r, -0.25 * c_t, 0.75 * c_t, 0.75 * c_r, 0.75 * c_t, -0.25 * c_t, -0.25 * c_r]
        ys = [0.0, half, half, 0.0, -half, -half, 0.0]
        self.c_plan.setData(ys, xs)
        alphas = np.arange(-6.0, 22.01, 0.5)
        p = ll.polar(w, alphas)
        self.polar = p
        att = alphas <= (p.stall_alpha if math.isfinite(p.stall_alpha) else alphas[-1])
        self.c_cl.setData(alphas[att], p.cl[att])
        sec_att = sec.alpha <= (sec.stall_alpha if math.isfinite(sec.stall_alpha) else sec.alpha[-1])
        self.c_sec.setData(sec.alpha[sec_att], sec.cl[sec_att])
        self.c_pol.setData(p.cd[att], p.cl[att])
        self.c_cdi.setData(p.cdi[att], p.cl[att])
        if math.isfinite(p.stall_alpha):
            self.c_st.setData([p.stall_alpha], [p.cl_max])
            self.c_stall_mark.setData([p.stall_y, -p.stall_y], [0.25 * float(w.chord(np.array([p.stall_y]))[0])] * 2)
        else:
            self.c_st.setData([], [])
            self.c_stall_mark.setData([], [])
        ld = p.cl[att] / np.maximum(p.cd[att], 1e-9)
        k = int(np.argmax(ld))
        frac = p.stall_y / half if math.isfinite(p.stall_y) else math.nan
        where = ("at the root" if frac < 0.2 else "mid-span" if frac < 0.6 else "near the tips (ailerons!)") \
            if math.isfinite(frac) else "-"
        self.info.setText("\n".join([
            f"Area {w.area:.3f} m², aspect ratio {w.aspect_ratio:.2f}, MAC {w.mean_aerodynamic_chord:.3f} m",
            f"Section {self.code.text().strip()}: CLα {sec.cl_alpha:.2f}/rad, α0 {sec.alpha_zero_lift:+.2f}°, "
            f"CL,max {cl_max:.2f}, CD0 {float(sec.cd.min()):.4f}",
            f"Wing CLα {p.cl_alpha:.2f}/rad ({p.cl_alpha * math.pi / 180:.4f}/°), span efficiency e {p.e:.3f}",
            (f"Wing stall ≈ {p.stall_alpha:.1f}°, CL,max ≈ {p.cl_max:.2f}, first {where} (|y| ≈ {p.stall_y:.2f} m)"
             if math.isfinite(p.stall_alpha) else "No stall in the α range."),
            f"Best L/D ≈ {ld[k]:.1f} at CL {p.cl[att][k]:.2f} (α {alphas[att][k]:.1f}°)"]))
        self._show_alpha()

    def _show_alpha(self) -> None:
        a = self.alpha.value() / 10.0
        self.alpha_lbl.setText(f"α = {a:.1f}°")
        self.now.setPos(a)
        if self.wing is None:
            return
        s = ll.solve(self.wing, a)
        self.c_local.setData(s.y, s.cl_local)
        half = 0.5 * self.wing.span
        self.c_limit.setData([-half, half], [self.wing.cl_max] * 2)
        yy = np.linspace(-half, half, 101)
        # elliptic loading of the same CL: Cl c = (4 CL S / (π b)) sqrt(1 - (2y/b)²)
        cc = 4.0 * s.cl * self.wing.area / (math.pi * self.wing.span) * np.sqrt(np.clip(1.0 - (yy / half) ** 2, 0, 1))
        self.c_ellip.setData(yy, cc / self.wing.chord(yy))
        self.p_span.setTitle(f"CL {s.cl:.3f} · CDi {s.cdi:.4f} · e {s.e:.3f}", color=PALETTE.text_muted, size="9pt")
