"""Design and analysis tabs of the Rocket Nozzle Lab (heat transfer, mixture ratio and expansion-ratio
optimisation, noise, engine comparison) and its PDF report."""

from __future__ import annotations

import base64
import html
import math
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QBuffer, QIODevice, Qt, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from config.settings import APP_NAME, APP_VERSION, isa_atmosphere
from config.theme import PALETTE, mono_font
from core import nozzle as nz
from core import nozzle_design as nd
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
        g.addWidget(labeled_row("Wall temperature", self.t_wall), 1, 0)
        g.addWidget(labeled_row("Emissivity", self.emis), 1, 1)
        g.addWidget(labeled_row("Extension from ε", self.ext), 2, 0)
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
