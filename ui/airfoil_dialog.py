"""Airfoil analysis: Hess-Smith panel method (Cp distribution, lift, moment), stall estimate and drag polar."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QSlider,
    QSpinBox, QVBoxLayout, QWidget,
)

from config.settings import APP_NAME, get_paths
from config.theme import PALETTE, mono_font
from core import panel as pm
from core.shapes2d import naca_outline, read_airfoil_dat
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


def _plot(xl: str, yl: str, legend: bool = False, corner: tuple[int, int] = (-8, 8)) -> pg.PlotWidget:
    w = pg.PlotWidget()
    _style_plot(w, xl, yl)
    if legend:
        w.addLegend(offset=corner, labelTextColor=PALETTE.text_muted, brush=pg.mkBrush(PALETTE.surface_alt + "B0"))
    return w


class AirfoilDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Airfoil analysis (panel method) — {APP_NAME}")
        self.resize(1280, 820)
        self.code = QLineEdit("2412")
        self.code.setToolTip("NACA 4-digit code (e.g. 0012, 2412, 4415)")
        self.code.editingFinished.connect(self._from_code)
        load = button("Load .dat…", "ghost", "folder", "Airfoil coordinates (UIUC / XFOIL Selig or Lednicer format)")
        load.clicked.connect(self._load)
        row = QHBoxLayout()
        row.addWidget(self.code, 1)
        row.addWidget(load)
        rw = QWidget()
        rw.setLayout(row)
        self.panels = QSpinBox()
        self.panels.setRange(40, 400)
        self.panels.setValue(160)
        self.panels.setSingleStep(20)
        self.re = _spin(1e4, 1e9, 3e6, 0, "", 1e5, "Chord Reynolds number (boundary layer, drag and stall)")
        self.mach = _spin(0.0, 0.95, 0.0, 2, "", 0.05, "Free-stream Mach number: Kármán-Tsien pressures, "
                          "Prandtl-Glauert lift, wave drag past the critical Mach number")
        self.trip_u = _spin(0.0, 1.0, 0.0, 2, "x/c", 0.05, "Transition strip on the upper surface (0 = free "
                            "transition, Michel's criterion)")
        self.trip_l = _spin(0.0, 1.0, 0.0, 2, "x/c", 0.05, "Transition strip on the lower surface (0 = free)")
        for sp in (self.trip_u, self.trip_l):
            sp.setSpecialValueText("free")
        self.a_lo = _spin(-30.0, 0.0, -10.0, 1, "°", 1.0)
        self.a_hi = _spin(0.0, 30.0, 20.0, 1, "°", 1.0)
        self.a_step = _spin(0.1, 5.0, 0.5, 1, "°", 0.1)
        self.alpha = QSlider(Qt.Orientation.Horizontal)
        self.alpha.setRange(-200, 250)
        self.alpha.setValue(50)
        self.alpha_lbl = QLabel("α = 5.0°")
        form = QFormLayout()
        form.addRow("Airfoil", rw)
        form.addRow("Panels", self.panels)
        form.addRow("Reynolds number", self.re)
        form.addRow("Mach number", self.mach)
        form.addRow("Trip, upper", self.trip_u)
        form.addRow("Trip, lower", self.trip_l)
        self.flap_c = _spin(0.0, 45.0, 0.0, 0, "% c", 5.0, "Slotted flap chord (0 = no flap; 25-35 % is typical)")
        self.flap_c.setSpecialValueText("none")
        self.flap_d = _spin(0.0, 50.0, 25.0, 1, "°", 2.5, "Flap deflection, trailing edge down")
        self.flap_gap = _spin(0.2, 5.0, 1.5, 1, "% c", 0.25, "Slot gap between the main trailing edge and the flap")
        self.slat_c = _spin(0.0, 25.0, 0.0, 0, "% c", 5.0, "Leading-edge slat chord (0 = no slat; 12-18 % is typical)")
        self.slat_c.setSpecialValueText("none")
        self.slat_d = _spin(0.0, 60.0, 40.0, 1, "°", 2.5, "Slat deflection, leading edge down")
        form.addRow("Flap chord", self.flap_c)
        form.addRow("Flap deflection", self.flap_d)
        form.addRow("Flap gap", self.flap_gap)
        form.addRow("Slat chord", self.slat_c)
        form.addRow("Slat deflection", self.slat_d)
        form.addRow("α from", self.a_lo)
        form.addRow("α to", self.a_hi)
        form.addRow("α step", self.a_step)
        form.addRow(self.alpha_lbl, self.alpha)
        self.info = QLabel("")
        self.info.setFont(mono_font(10))
        self.info.setWordWrap(True)
        exp = button("Export polar CSV…", "ghost", "", "α, CL (viscous estimate and inviscid), CD, Cm")
        exp.clicked.connect(self._export)
        note = QLabel("Hess-Smith panel method + integral boundary layer (Thwaites laminar, Michel transition, "
                      "Head turbulent; Squire-Young drag). Turbulent separation lowers the lift (Kirchhoff "
                      "trailing-edge stall); the Valarezo-Chin limit adds leading-edge stall. One-way coupling: "
                      "expect ±10-15 % on CL,max, more for thick cambered sections.")
        note.setWordWrap(True)
        note.setProperty("role", "muted")
        left = QVBoxLayout()
        left.addLayout(form)
        left.addWidget(exp)
        left.addWidget(self.info, 1)
        left.addWidget(note)
        lw = QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(380)
        self.p_cp = _plot("x / c", "Cp", legend=True)
        self.p_cp.getPlotItem().invertY(True)
        self.c_up = self.p_cp.plot(name="upper", pen=pg.mkPen(PALETTE.cyan, width=2))
        self.c_lo = self.p_cp.plot(name="lower", pen=pg.mkPen(PALETTE.warning, width=2))
        self.c_slat = self.p_cp.plot(name="slat", pen=pg.mkPen(PALETTE.success, width=2))
        self.c_flap = self.p_cp.plot(name="flap", pen=pg.mkPen(PALETTE.indigo, width=2))
        self.l_sonic = pg.InfiniteLine(angle=0, pen=pg.mkPen(PALETTE.danger, width=1, style=Qt.PenStyle.DashLine),
                                       label="Cp*  (sonic)", labelOpts={"color": PALETTE.danger, "position": 0.08})
        self.p_cp.addItem(self.l_sonic, ignoreBounds=True)
        self.p_shape = _plot("x / c", "y / c")
        self.p_shape.setAspectLocked(True)
        self.c_shape = self.p_shape.plot(pen=pg.mkPen(PALETTE.text, width=1.5), fillLevel=None)
        self.p_cl = _plot("α (°)", "CL", legend=True, corner=(-8, -8))
        self.c_cl = self.p_cl.plot(name="viscous estimate", pen=pg.mkPen(PALETTE.cyan, width=2))
        self.c_cli = self.p_cl.plot(name="inviscid", pen=pg.mkPen(PALETTE.text_muted, width=1,
                                                                 style=Qt.PenStyle.DashLine))
        self.c_stall = self.p_cl.plot(name="estimated stall", pen=None, symbol="x", symbolSize=12,
                                      symbolPen=pg.mkPen(PALETTE.danger, width=2))
        self.now = [pg.InfiniteLine(angle=90, pen=pg.mkPen("#FFFFFF", width=1, style=Qt.PenStyle.DotLine))
                    for _ in range(2)]
        self.p_cl.addItem(self.now[0], ignoreBounds=True)
        self.p_polar = _plot("CD", "CL")
        self.c_polar = self.p_polar.plot(pen=pg.mkPen(PALETTE.success, width=2))
        self.p_cm = _plot("α (°)", "Cm (¼ chord)")
        self.c_cm = self.p_cm.plot(pen=pg.mkPen(PALETTE.indigo, width=2))
        self.p_cm.addItem(self.now[1], ignoreBounds=True)
        grid = QGridLayout()
        grid.addWidget(_card("Pressure distribution", self.p_cp), 0, 0)
        grid.addWidget(_card("Section", self.p_shape), 0, 1)
        grid.addWidget(_card("Lift curve", self.p_cl), 1, 0)
        grid.addWidget(_card("Drag polar", self.p_polar), 1, 1)
        self.p_bl = _plot("x / c", "skin friction Cf × 1000", legend=True)
        self.c_cfu = self.p_bl.plot(name="upper", pen=pg.mkPen(PALETTE.cyan, width=2))
        self.c_cfl = self.p_bl.plot(name="lower", pen=pg.mkPen(PALETTE.warning, width=2))
        self.c_marks = self.p_bl.plot(name="transition / separation", pen=None, symbol="o", symbolSize=8,
                                      symbolBrush=pg.mkBrush(PALETTE.danger), symbolPen=None)
        self.c_cp_marks = self.p_cp.plot(pen=None, symbol="t", symbolSize=10, symbolBrush=pg.mkBrush(PALETTE.danger),
                                         symbolPen=None)
        grid.addWidget(_card("Pitching moment", self.p_cm), 2, 0)
        grid.addWidget(_card("Boundary layer", self.p_bl), 2, 1)
        root = QHBoxLayout(self)
        root.addWidget(lw)
        root.addLayout(grid, 1)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self.recompute)
        for w in (self.re, self.a_lo, self.a_hi, self.a_step, self.trip_u, self.trip_l, self.mach, self.flap_c,
                  self.flap_d, self.flap_gap, self.slat_c, self.slat_d):
            w.valueChanged.connect(lambda _v: self._timer.start())
        self.panels.valueChanged.connect(lambda _v: (self._prepare(), self._timer.start()))
        self.alpha.valueChanged.connect(lambda _v: self._show_alpha())
        self.outline: Optional[np.ndarray] = None
        self.name = ""
        self.nodes: Optional[np.ndarray] = None
        self.polar: Optional[pm.Polar] = None
        self._from_code()

    # -- input -----------------------------------------------------------------------------------
    def _from_code(self) -> None:
        code = self.code.text().strip()
        try:
            self.outline = naca_outline(code, 240)
        except Exception as exc:  # noqa: BLE001 - a malformed code
            self.info.setText(f"Not a NACA 4-digit code: {exc}")
            return
        self.name = f"NACA {code}"
        self._prepare()
        self.recompute()

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load airfoil", str(get_paths().root),
                                              "Airfoil coordinates (*.dat *.txt *.csv);;All files (*)")
        if not path:
            return
        try:
            self.name, self.outline = read_airfoil_dat(path)
        except Exception as exc:  # noqa: BLE001
            self.info.setText(f"Could not read the file: {exc}")
            return
        self.code.setText(self.name)
        self._prepare()
        self.recompute()

    def _prepare(self) -> None:
        if self.outline is not None:
            self.nodes = pm.prepare(self.outline, self.panels.value())
            self.c_shape.setData(self.nodes[:, 0], self.nodes[:, 1])

    # -- results -----------------------------------------------------------------------------------
    def recompute(self) -> None:
        if self.nodes is None:
            return
        lo, hi = self.a_lo.value(), self.a_hi.value()
        alphas = np.arange(lo, hi + 1e-9, max(self.a_step.value(), 0.1))
        if len(alphas) < 3:
            return
        hl = self._hl()
        for w in (self.flap_d, self.flap_gap):
            w.setEnabled(self.flap_c.value() > 0)
        self.slat_d.setEnabled(self.slat_c.value() > 0)
        if hl is not None:
            self._recompute_multi(alphas, hl)
            return
        self.c_slat.setData([], [])
        self.c_flap.setData([], [])
        self.c_shape.setData(self.nodes[:, 0], self.nodes[:, 1])
        p = pm.polar(self.nodes, alphas, self.re.value(), True, self._trip(self.trip_u), self._trip(self.trip_l),
                     self.mach.value())
        self.polar = p
        self.c_cli.setData(p.alpha, p.cl_inviscid)
        self.c_cm.setData(p.alpha, p.cm)
        attached = np.ones_like(p.alpha, dtype=bool)
        if math.isfinite(p.stall_alpha):
            attached &= p.alpha <= p.stall_alpha
            self.c_stall.setData([p.stall_alpha], [p.cl_max])
        else:
            self.c_stall.setData([], [])
        if math.isfinite(p.stall_negative):
            attached &= p.alpha >= p.stall_negative
        self.c_cl.setData(p.alpha[attached], p.cl[attached])          # the viscous estimate ends at stall
        self.c_polar.setData(p.cd[attached], p.cl[attached])
        ld = p.cl[attached] / np.maximum(p.cd[attached], 1e-9)
        k = int(np.argmax(ld)) if ld.size else 0
        cm0 = float(np.interp(p.alpha_zero_lift, p.alpha, p.cm))
        mc0 = float(np.interp(0.0, p.alpha, p.m_crit))
        lines = [f"{self.name}: t/c {p.t_c * 100:.1f} %, Re {self.re.value():,.0f}"
                 + (f", M {self.mach.value():.2f}" if self.mach.value() > 1e-3 else ""),
                 f"Critical Mach (α = 0°) {mc0:.3f}",
                 f"CLα {p.cl_alpha * math.pi / 180:.4f} /° ({p.cl_alpha:.2f} /rad; thin-airfoil 2π = 6.28 inviscid)",
                 f"Zero-lift α {p.alpha_zero_lift:+.2f}°, Cm0 {cm0:+.4f}"]
        if math.isfinite(p.stall_alpha):
            lines.append(f"Stall ≈ {p.stall_alpha:.1f}°, CL,max ≈ {p.cl_max:.2f}"
                         + (f"; negative stall ≈ {p.stall_negative:.1f}°" if math.isfinite(p.stall_negative) else ""))
        else:
            lines.append("No stall within the α range (extend it).")
        if ld.size:
            lines.append(f"Best L/D ≈ {ld[k]:.0f} at α {p.alpha[attached][k]:.1f}° (CL {p.cl[attached][k]:.2f}, "
                         f"CD {p.cd[attached][k]:.4f})")
        self.info.setText("\n".join(lines))
        self._show_alpha()

    def _hl(self):
        from core import multi_element as me

        if self.flap_c.value() <= 0 and self.slat_c.value() <= 0:
            return None
        return me.HighLift(flap_chord=self.flap_c.value() / 100.0, flap_deflection=self.flap_d.value(),
                           flap_gap=self.flap_gap.value() / 100.0, slat_chord=self.slat_c.value() / 100.0,
                           slat_deflection=self.slat_d.value())

    def _recompute_multi(self, alphas: np.ndarray, hl) -> None:
        """Slat / flap configuration: coupled multi-element panel solution."""
        from core import multi_element as me

        self._elements = me.build_elements(self.outline, hl, max(self.panels.value(), 100))
        xs, ys = [], []
        for _n, nodes in self._elements:
            xs += list(nodes[:, 0]) + [np.nan]
            ys += list(nodes[:, 1]) + [np.nan]
        self.c_shape.setData(np.asarray(xs), np.asarray(ys), connect="finite")
        al = alphas if len(alphas) <= 40 else np.linspace(alphas[0], alphas[-1], 40)
        mp = me.polar(self._elements, al, self.re.value())
        clean = pm.polar(self.nodes, al, self.re.value())
        self.polar = None
        self._mpolar = mp
        self.c_cli.setData(mp.alpha, mp.cl_inviscid)
        self.c_cm.setData(mp.alpha, mp.cm)
        att = mp.alpha <= (mp.stall_alpha if math.isfinite(mp.stall_alpha) else mp.alpha[-1])
        self.c_cl.setData(mp.alpha[att], mp.cl[att])
        pos = att & (mp.alpha >= 0.0)                       # high-lift settings are used at positive α
        self.c_polar.setData(mp.cd[pos], mp.cl[pos])
        if math.isfinite(mp.stall_alpha):
            self.c_stall.setData([mp.stall_alpha], [mp.cl_max])
        else:
            self.c_stall.setData([], [])
        clm0 = clean.cl_max if math.isfinite(clean.cl_max) else float(clean.cl.max())
        parts = ", ".join(n for n, _ in self._elements)
        lines = [f"{self.name} with {parts} (Re {self.re.value():,.0f})"]
        if math.isfinite(mp.stall_alpha):
            lines.append(f"Stall ≈ {mp.stall_alpha:.1f}°, CL,max ≈ {mp.cl_max:.2f}  (clean {clm0:.2f}: ΔCL,max "
                         f"{mp.cl_max - clm0:+.2f}); set by the {mp.critical_element}")
        else:
            lines.append("No stall within the α range (extend it).")
        lines.append(f"CL at 0°: {float(np.interp(0.0, mp.alpha, mp.cl)):.2f} (clean "
                     f"{float(np.interp(0.0, clean.alpha, clean.cl)):.2f})")
        lines.append("Multi-element panel method; stall from the Valarezo-Chin limit on each element's suction peak. "
                     "Drag: each element's boundary layer (no confluent-wake model).")
        self.info.setText("\n".join(lines))
        self._show_alpha()

    def _show_multi(self, a: float) -> None:
        from core import multi_element as me

        ms = me.solve(self._elements, a)
        curves = {"main": None, "slat": self.c_slat, "flap": self.c_flap}
        self.c_slat.setData([], [])
        self.c_flap.setData([], [])
        for name, el in zip(ms.names, ms.elements):
            if name == "main":
                up, lo = el.upper, ~el.upper
                self.c_up.setData(el.x[up], el.cp[up])
                self.c_lo.setData(el.x[lo][::-1], el.cp[lo][::-1])
            else:
                order = np.concatenate([np.arange(len(el.x)), [0]])
                curves[name].setData(el.x[order], el.cp[order])
        self.c_cfu.setData([], [])
        self.c_cfl.setData([], [])
        self.c_marks.setData([], [])
        self.c_cp_marks.setData([], [])
        self.p_bl.setTitle("Boundary-layer chart: clean airfoil only", color=PALETTE.text_muted, size="9pt")
        self.p_cp.setTitle(f"CL {ms.cl:.3f} (inviscid) · " + " · ".join(f"{n} {c:.2f}" for n, c in
                                                                         zip(ms.names, ms.cl_parts)),
                           color=PALETTE.text_muted, size="9pt")

    @staticmethod
    def _trip(sp) -> float | None:
        return sp.value() if sp.value() > 0.0 else None

    def _show_alpha(self) -> None:
        a = self.alpha.value() / 10.0
        self.alpha_lbl.setText(f"α = {a:.1f}°")
        for ln in self.now:
            ln.setPos(a)
        if self.nodes is None:
            return
        if self._hl() is not None and getattr(self, "_elements", None):
            self._show_multi(a)
            return
        s = pm.solve(self.nodes, a)
        up, lo = s.upper, ~s.upper
        m = self.mach.value()
        cp = pm.karman_tsien(s.cp, m) if m > 1e-3 else s.cp
        self.c_up.setData(s.x[up], cp[up])
        self.c_lo.setData(s.x[lo][::-1], cp[lo][::-1])
        self.l_sonic.setVisible(m > 1e-3)
        if m > 1e-3:
            self.l_sonic.setPos(pm.sonic_cp(m))
        from core import boundary_layer as bl

        b = bl.analyse(s, self.re.value(), self._trip(self.trip_u), self._trip(self.trip_l))
        self.c_cfu.setData(b.upper.x, b.upper.cf * 1e3)
        self.c_cfl.setData(b.lower.x, b.lower.cf * 1e3)
        mx, my, cx, cy = [], [], [], []
        for surf, ys in ((b.upper, s.cp[up]), (b.lower, s.cp[lo])):
            for xm in (surf.transition_x, surf.separation_x if surf.separation_x < 0.999 else float("nan")):
                if xm == xm:
                    mx.append(xm)
                    my.append(float(np.interp(xm, surf.x, surf.cf * 1e3)))
                    xs_s = s.x[up] if surf is b.upper else s.x[lo][::-1]
                    ys_s = ys if surf is b.upper else ys[::-1]
                    order = np.argsort(xs_s)
                    cx.append(xm)
                    cy.append(float(np.interp(xm, xs_s[order], ys_s[order])))
        self.c_marks.setData(mx, my)
        self.c_cp_marks.setData(cx, cy)

        def where(v: float) -> str:
            return "—" if v != v else f"{v:.2f}"

        self.p_bl.setTitle(f"transition x/c: upper {where(b.upper.transition_x)}, lower {where(b.lower.transition_x)}"
                           f" · separation {where(b.upper.separation_x) if b.upper.separation_x < 0.999 else 'none'}"
                           f" · Cd {b.cd:.5f}", color=PALETTE.text_muted, size="9pt")
        cl_v = (float(np.interp(a, self.polar.alpha, self.polar.cl)) if self.polar is not None
                else pm.VISCOUS_LIFT_FACTOR * s.cl)
        mc = pm.critical_mach(float(s.cp.min()))
        self.p_cp.setTitle(f"CL {cl_v:.3f} (inviscid {s.cl:.3f}) · Cm {s.cm_quarter:+.4f} · M_crit {mc:.2f}"
                           + (" · SUPERCRITICAL (shock)" if m > mc else ""),
                           color=PALETTE.text_muted, size="9pt")

    def _export(self) -> None:
        if self.polar is None:
            return
        name = "".join(ch if ch.isalnum() else "_" for ch in self.name) or "airfoil"
        path, _ = QFileDialog.getSaveFileName(self, "Export polar", str(get_paths().exports / f"{name}_polar.csv"),
                                              "CSV (*.csv)")
        if not path:
            return
        p = self.polar
        with Path(path).open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["alpha_deg", "CL", "CL_inviscid", "CD", "Cm_quarter", "dCp"])
            for row in zip(p.alpha, p.cl, p.cl_inviscid, p.cd, p.cm, p.delta_cp):
                w.writerow([f"{v:.6g}" for v in row])
