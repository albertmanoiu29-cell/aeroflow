"""Real-time 2-D charts (pyqtgraph, dark theme).

* :class:`ResidualChart`: live OpenFOAM convergence (log scale).
* :class:`ChordCpChart`: chordwise pressure distribution (Cp axis inverted, suction up).
* :class:`PolarCharts`: lift curve, drag polar and L/D, with stall and best-L/D markers.

Updates are cheap: residuals are appended to lists and the curves are redrawn by a short
single-shot timer that only runs while new data is pending, so a fast producer never stalls the
GUI and an idle chart costs nothing.

Text annotations and empty-chart hints are added with ``ignoreBounds=True`` so they never take
part in auto-ranging (otherwise an empty chart auto-ranges to ±1e-305 garbage ticks).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from config.theme import PALETTE, make_section_label, mono_font
from core.results import ChordCp, PolarData

log = logging.getLogger("aeroflow.viz.charts")

pg.setConfigOptions(antialias=True, background=PALETTE.surface, foreground=PALETTE.text_muted)

RESIDUAL_COLORS: dict[str, str] = {
    "Ux": "#00E5FF", "Uy": "#38BDF8", "Uz": "#6366F1", "p": "#F59E0B", "k": "#10B981", "omega": "#EC4899",
    "nuTilda": "#A3E635", "epsilon": "#F97316",
}


def _style_plot(pw: pg.PlotWidget, x_label: str = "", y_label: str = "") -> None:
    """Apply the dark aerospace style to a plot widget."""
    pw.setBackground(PALETTE.surface)
    item = pw.getPlotItem()
    item.showGrid(x=True, y=True, alpha=0.12)
    font = mono_font(10)
    for name in ("left", "bottom"):
        ax = item.getAxis(name)
        ax.setPen(pg.mkPen(PALETTE.border_strong))
        ax.setTextPen(pg.mkPen(PALETTE.text_muted))
        ax.setTickFont(font)
        ax.enableAutoSIPrefix(False)          # never rewrite labels as "α (deg) (x0.001)"
    style = {"color": PALETTE.text_muted, "font-size": "10px"}
    if x_label:
        item.setLabel("bottom", x_label, **style)
    if y_label:
        item.setLabel("left", y_label, **style)
    item.setMenuEnabled(False)
    item.hideButtons()
    pw.setMinimumHeight(140)


def _card(title: str, content: QWidget, extra: Optional[QWidget] = None) -> QFrame:
    """Wrap a chart in a titled card."""
    frame = QFrame()
    frame.setObjectName("Card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(10, 8, 10, 10)
    lay.setSpacing(6)
    head = QHBoxLayout()
    head.addWidget(make_section_label(title))
    head.addStretch(1)
    if extra is not None:
        head.addWidget(extra)
    lay.addLayout(head)
    lay.addWidget(content, 1)
    return frame


def export_plot_png(widget: pg.PlotWidget, path: str | Path, width: int = 1400) -> Path:
    """Export a pyqtgraph plot to PNG (used for report snapshots)."""
    import pyqtgraph.exporters as exporters

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    exp = exporters.ImageExporter(widget.getPlotItem())
    exp.parameters()["width"] = int(width)
    exp.export(str(out))
    return out


class _Placeholder(pg.TextItem):
    """Centred hint text shown on an empty chart."""

    def __init__(self, text: str) -> None:
        super().__init__(text=text, color=PALETTE.text_dim, anchor=(0.5, 0.5))


def _show_empty(plot: pg.PlotWidget, hint: pg.TextItem, x_range: tuple[float, float],
                y_range: tuple[float, float]) -> None:
    """Give an empty chart a sensible fixed range and centre its hint text in it."""
    plot.disableAutoRange()
    plot.setXRange(*x_range, padding=0.0)
    plot.setYRange(*y_range, padding=0.0)
    hint.setPos(0.5 * (x_range[0] + x_range[1]), 0.5 * (y_range[0] + y_range[1]))
    hint.show()


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------
class ResidualChart(QWidget):
    """Live log-scale convergence chart of the initial residuals."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.plot = pg.PlotWidget()
        _style_plot(self.plot, "Iteration", "Initial residual")
        self.plot.setLogMode(x=False, y=True)
        self.plot.addLegend(offset=(-8, 8), labelTextColor=PALETTE.text_muted, brush=pg.mkBrush(PALETTE.surface_alt))
        self.status = QLabel("Residuals appear here while OpenFOAM runs.")
        self.status.setProperty("role", "muted")
        self.status.setWordWrap(True)
        # Force coefficients against iteration: a converged run has flat CL / CD (residuals alone can mislead).
        self.coef_plot = pg.PlotWidget()
        _style_plot(self.coef_plot, "Iteration", "CL  ·  10 × CD")
        self.coef_plot.addLegend(offset=(-8, 8), labelTextColor=PALETTE.text_muted,
                                 brush=pg.mkBrush(PALETTE.surface_alt))
        self.c_cl = self.coef_plot.plot(name="CL", pen=pg.mkPen(PALETTE.cyan, width=1.8))
        self.c_cd = self.coef_plot.plot(name="10 × CD", pen=pg.mkPen(PALETTE.warning, width=1.8))
        self.coef_card = _card("Force coefficients", self.coef_plot)
        self.coef_card.hide()
        self._coef_it: list[float] = []
        self._coef_cl: list[float] = []
        self._coef_cd: list[float] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(_card("Convergence", self.plot))
        lay.addWidget(self.coef_card)
        lay.addWidget(self.status)
        self._iters: list[float] = []
        self._data: dict[str, list[float]] = {}
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._dirty = False
        self._timer = QTimer(self)                 # single-shot: runs only while data is pending
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._redraw)

    def _schedule(self) -> None:
        self._dirty = True
        if not self._timer.isActive():
            self._timer.start()

    def stop(self) -> None:
        """Stop the pending redraw (on shutdown)."""
        self._timer.stop()

    def reset(self, message: str = "Waiting for the first iteration…") -> None:
        """Clear all curves."""
        self._iters.clear()
        self._data.clear()
        for c in self._curves.values():
            self.plot.removeItem(c)
        self._curves.clear()
        self._coef_it.clear()
        self._coef_cl.clear()
        self._coef_cd.clear()
        self.c_cl.setData([], [])
        self.c_cd.setData([], [])
        self.coef_card.hide()
        self._dirty = False
        self.status.setText(message)

    def add_coefficients(self, iteration: int, coeffs: dict[str, float]) -> None:
        """Append the live CL / CD of one iteration (OpenFOAM forceCoeffs)."""
        if "Cl" not in coeffs and "Cd" not in coeffs:
            return
        if self._coef_it and self._coef_it[-1] == float(iteration):
            self._coef_cl[-1] = float(coeffs.get("Cl", math.nan))
            self._coef_cd[-1] = 10.0 * float(coeffs.get("Cd", math.nan))
        else:
            self._coef_it.append(float(iteration))
            self._coef_cl.append(float(coeffs.get("Cl", math.nan)))
            self._coef_cd.append(10.0 * float(coeffs.get("Cd", math.nan)))
        self.coef_card.show()
        self._schedule()

    def add_residuals(self, iteration: int, values: dict[str, float]) -> None:
        """Append one iteration's initial residuals."""
        self._iters.append(float(iteration))
        n = len(self._iters)
        for name, v in values.items():
            series = self._data.setdefault(name, [math.nan] * (n - 1))
            series.append(max(float(v), 1e-12))
        for series in self._data.values():
            while len(series) < n:
                series.append(math.nan)
        self._schedule()

    def set_history(self, residuals: Optional[dict[str, list[float]]]) -> None:
        """Replace the chart content with complete residual histories."""
        if not residuals:
            self.reset("Direct solve — the Instant Aero Engine has no iterations to converge.")
            return
        self.reset()
        n = max(len(v) for v in residuals.values())
        self._iters = [float(i + 1) for i in range(n)]
        self._data = {k: [max(x, 1e-12) for x in v] + [math.nan] * (n - len(v)) for k, v in residuals.items()}
        self._dirty = True
        self._redraw()

    def _redraw(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        if self._coef_it:
            xi = np.asarray(self._coef_it)
            self.c_cl.setData(xi, np.asarray(self._coef_cl), connect="finite")
            self.c_cd.setData(xi, np.asarray(self._coef_cd), connect="finite")
        x = np.asarray(self._iters)
        for name, series in self._data.items():
            y = np.asarray(series, dtype=float)
            curve = self._curves.get(name)
            if curve is None:
                color = RESIDUAL_COLORS.get(name, PALETTE.text)
                curve = self.plot.plot(name=name, pen=pg.mkPen(color, width=1.6), connect="finite")
                self._curves[name] = curve
            curve.setData(x, y, connect="finite")
        if len(x):
            last = {k: v[-1] for k, v in self._data.items() if v and not math.isnan(v[-1])}
            worst = max(last.values()) if last else float("nan")
            self.status.setText(f"Iteration {int(x[-1])} · max residual {worst:.2e}")


# ---------------------------------------------------------------------------
# Chordwise Cp
# ---------------------------------------------------------------------------
class ChordCpChart(QWidget):
    """Pressure coefficient along the chord (upper and lower surface)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.plot = pg.PlotWidget()
        _style_plot(self.plot, "x / c", "Cp")
        self.plot.invertY(True)
        self.plot.addLegend(offset=(-8, 8), labelTextColor=PALETTE.text_muted, brush=pg.mkBrush(PALETTE.surface_alt))
        self.upper = self.plot.plot(name="Upper", pen=None, symbol="o", symbolSize=4,
                                    symbolBrush=pg.mkBrush(PALETTE.cyan), symbolPen=None)
        self.lower = self.plot.plot(name="Lower", pen=None, symbol="o", symbolSize=4,
                                    symbolBrush=pg.mkBrush(PALETTE.warning), symbolPen=None)
        self.plot.addItem(pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen(PALETTE.border_strong, width=1)))
        self.hint = _Placeholder("Run a solve to see the chordwise Cp")
        self.plot.addItem(self.hint, ignoreBounds=True)
        self.station = QLabel("")
        self.station.setProperty("role", "muted")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(_card("Pressure distribution", self.plot, self.station))
        _show_empty(self.plot, self.hint, (0.0, 1.0), (-1.5, 1.0))

    def set_data(self, data: Optional[ChordCp]) -> None:
        """Show a chordwise Cp distribution (``None`` clears)."""
        if data is None or data.is_empty:
            self.upper.setData([], [])
            self.lower.setData([], [])
            _show_empty(self.plot, self.hint, (0.0, 1.0), (-1.5, 1.0))
            self.station.setText("")
            return
        self.hint.hide()
        self.upper.setData(np.asarray(data.x_over_c_upper), np.asarray(data.cp_upper))
        self.lower.setData(np.asarray(data.x_over_c_lower), np.asarray(data.cp_lower))
        self.station.setText(f"y = {data.y_position:+.3f} m")
        self.plot.enableAutoRange()


# ---------------------------------------------------------------------------
# Polars
# ---------------------------------------------------------------------------
class PolarCharts(QWidget):
    """Lift curve, drag polar and aerodynamic efficiency (side-by-side or stacked)."""

    def __init__(self, horizontal: bool = True, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.lift = pg.PlotWidget()
        self.drag = pg.PlotWidget()
        self.eff = pg.PlotWidget()
        self.mom = pg.PlotWidget()
        _style_plot(self.lift, "α (deg)", "CL")
        _style_plot(self.drag, "CD", "CL")
        _style_plot(self.eff, "α (deg)", "L/D")
        _style_plot(self.mom, "α (deg)", "CM")
        pen = lambda c: pg.mkPen(c, width=2.2)  # noqa: E731
        sym = dict(symbol="o", symbolSize=5, symbolPen=None)
        self.c_lift = self.lift.plot(pen=pen(PALETTE.cyan), symbolBrush=pg.mkBrush(PALETTE.cyan), **sym)
        self.c_drag = self.drag.plot(pen=pen(PALETTE.indigo), symbolBrush=pg.mkBrush(PALETTE.indigo), **sym)
        self.c_eff = self.eff.plot(pen=pen(PALETTE.success), symbolBrush=pg.mkBrush(PALETTE.success), **sym)
        self.c_mom = self.mom.plot(pen=pen(PALETTE.warning), symbolBrush=pg.mkBrush(PALETTE.warning), **sym)
        self.mom.addItem(pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen(PALETTE.border_strong, width=1)))
        mk = dict(size=12, pen=pg.mkPen("#FFFFFF", width=1.5))
        self.m_stall = pg.ScatterPlotItem(brush=pg.mkBrush(PALETTE.warning), symbol="star", **mk)
        self.m_best = pg.ScatterPlotItem(brush=pg.mkBrush(PALETTE.success), symbol="star", **mk)
        self.m_best_polar = pg.ScatterPlotItem(brush=pg.mkBrush(PALETTE.success), symbol="star", **mk)
        self.m_current = [pg.ScatterPlotItem(brush=pg.mkBrush("#FFFFFF"), symbol="o", size=9,
                                             pen=pg.mkPen(PALETTE.cyan, width=2)) for _ in range(3)]
        self.lift.addItem(self.m_stall)
        self.eff.addItem(self.m_best)
        self.drag.addItem(self.m_best_polar)
        for plot, m in zip((self.lift, self.drag, self.eff), self.m_current):
            plot.addItem(m)
        # Labels hang *below* the peak markers: above them they ran off the top of the auto-ranged chart.
        self.t_stall = pg.TextItem(color=PALETTE.warning, anchor=(0.5, -0.5))
        self.t_best = pg.TextItem(color=PALETTE.success, anchor=(0.5, -0.5))
        self.lift.addItem(self.t_stall, ignoreBounds=True)
        self.eff.addItem(self.t_best, ignoreBounds=True)
        self.hints = [_Placeholder("Run a polar sweep") for _ in range(4)]
        for plot, h in zip(self.plots(), self.hints):
            plot.addItem(h, ignoreBounds=True)
        cards = (_card("Lift curve", self.lift), _card("Drag polar", self.drag), _card("Efficiency", self.eff),
                 _card("Pitching moment", self.mom))
        if horizontal:            # the big polar page: a 2 x 2 grid
            grid = QGridLayout(self)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setSpacing(10)
            for i, c in enumerate(cards):
                grid.addWidget(c, i // 2, i % 2)
        else:                     # the narrow side panel: stacked
            lay = QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(10)
            for c in cards:
                lay.addWidget(c)
        self.polar: Optional[PolarData] = None
        self._show_all_empty()

    #: Fixed axis ranges used while there is no polar (lift, drag, efficiency, moment).
    _EMPTY_RANGES = (((-6.0, 18.0), (-0.5, 1.5)), ((0.0, 0.1), (-0.5, 1.5)), ((-6.0, 18.0), (-10.0, 30.0)),
                     ((-6.0, 18.0), (-0.3, 0.1)))

    def _show_all_empty(self) -> None:
        for plot, hint, (xr, yr) in zip(self.plots(), self.hints, self._EMPTY_RANGES):
            _show_empty(plot, hint, xr, yr)

    #: Colours of pinned reference polars (drawn dashed and faded under the current polar).
    REF_COLORS = ("#F472B6", "#A3E635", "#FB923C", "#C4B5FD")

    def set_references(self, refs: list[tuple[str, PolarData]]) -> None:
        """Overlay pinned polars ``[(label, polar), …]`` as dashed curves for comparison."""
        for item in getattr(self, "_ref_items", []):
            for plot in self.plots():
                try:
                    plot.removeItem(item)
                except Exception:  # noqa: BLE001 - item may belong to another plot
                    pass
        self._ref_items: list = []
        if refs and self.lift.getPlotItem().legend is None:      # legend first, so the curves register
            self.lift.addLegend(offset=(8, 8), labelTextColor=PALETTE.text_muted,
                                brush=pg.mkBrush(PALETTE.surface_alt + "B0"))
        for k, (label, p) in enumerate(refs[:len(self.REF_COLORS)]):
            pen = pg.mkPen(self.REF_COLORS[k], width=1.6, style=pg.QtCore.Qt.PenStyle.DashLine)
            for plot, (x, y) in zip(self.plots(), ((p.alpha_deg, p.cl), (p.cd, p.cl), (p.alpha_deg, p.l_over_d),
                                                   (p.alpha_deg, p.cm))):
                item = plot.plot(x, y, pen=pen, name=label if plot is self.lift else None)
                item.setZValue(-5)
                self._ref_items.append(item)
        if refs:
            for h in self.hints:
                h.hide()
            for plot in self.plots():
                plot.enableAutoRange()
        elif self.polar is None:
            self._show_all_empty()

    def plots(self) -> tuple[pg.PlotWidget, pg.PlotWidget, pg.PlotWidget, pg.PlotWidget]:
        """The four plot widgets: lift curve, drag polar, efficiency, pitching moment (for export)."""
        return self.lift, self.drag, self.eff, self.mom

    def save_png(self, path: str | Path, scale: float = 2.0) -> Path:
        """Save all charts as one image (the whole widget, rendered at ``scale``x)."""
        from PySide6.QtGui import QPixmap

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        size = self.size() * scale
        pix = QPixmap(size)
        pix.setDevicePixelRatio(scale)
        pix.fill()
        self.render(pix)
        if not pix.save(str(out)):
            raise OSError(f"Could not write {out}")
        return out

    def set_polar(self, polar: Optional[PolarData]) -> None:
        """Display a polar (``None`` clears)."""
        self.polar = polar
        if polar is None or len(polar.alpha_deg) == 0:
            for c in (self.c_lift, self.c_drag, self.c_eff, self.c_mom):
                c.setData([], [])
            for m in (self.m_stall, self.m_best, self.m_best_polar, *self.m_current):
                m.setData([], [])
            self.t_stall.setText("")
            self.t_best.setText("")
            self._show_all_empty()
            return
        for h in self.hints:
            h.hide()
        a, cl, cd, ld = polar.alpha_deg, polar.cl, polar.cd, polar.l_over_d
        self.c_lift.setData(a, cl)
        self.c_drag.setData(cd, cl)
        self.c_eff.setData(a, ld)
        self.c_mom.setData(a, polar.cm)
        i_st = int(np.argmax(cl))
        i_ld = int(np.argmax(ld))
        self.m_stall.setData([a[i_st]], [cl[i_st]])
        self.t_stall.setText(f"CL,max {cl[i_st]:.2f} @ {a[i_st]:.1f}°")
        self.t_stall.setPos(float(a[i_st]), float(cl[i_st]))
        self.m_best.setData([a[i_ld]], [ld[i_ld]])
        self.t_best.setText(f"L/D max {ld[i_ld]:.1f} @ {a[i_ld]:.1f}°")
        self.t_best.setPos(float(a[i_ld]), float(ld[i_ld]))
        self.m_best_polar.setData([cd[i_ld]], [cl[i_ld]])
        self._draw_theory(polar)
        for p in self.plots():
            p.enableAutoRange()

    #: Show lifting-line theory next to the computed polar (validation at a glance).
    show_theory = True

    def set_show_theory(self, on: bool) -> None:
        self.show_theory = bool(on)
        if self.polar is not None:
            self._draw_theory(self.polar)

    def _draw_theory(self, polar: PolarData) -> None:
        """Reference curves from classical theory for this wing's aspect ratio:

        * lift: Helmbold's lifting-line slope ``a = 2π AR / (2 + sqrt(AR² + 4))`` through the computed
          zero-lift angle (valid for attached flow, any aspect ratio);
        * drag: the ideal polar ``CD = CD0 + CL² / (π e AR)`` with the computed minimum drag and e = 0.9.
        """
        for item in getattr(self, "_theory_items", []):
            for plot in (self.lift, self.drag):
                try:
                    plot.removeItem(item)
                except Exception:  # noqa: BLE001
                    pass
        self._theory_items: list = []
        ar = float(polar.aspect_ratio)
        a0 = polar.alpha_zero_lift_deg
        if not self.show_theory or not (0.2 < ar < 60.0) or not np.isfinite(a0) or len(polar.alpha_deg) < 3:
            return
        slope = 2.0 * np.pi * ar / (2.0 + np.sqrt(ar * ar + 4.0))          # per radian
        a = np.linspace(float(polar.alpha_deg.min()), float(polar.alpha_stall_deg), 40)
        cl_t = slope * np.radians(a - a0)
        pen = pg.mkPen("#E2E8F0", width=1.4, style=pg.QtCore.Qt.PenStyle.DotLine)
        if self.lift.getPlotItem().legend is None:
            self.lift.addLegend(offset=(8, 8), labelTextColor=PALETTE.text_muted,
                                brush=pg.mkBrush(PALETTE.surface_alt + "B0"))
        item = self.lift.plot(a, cl_t, pen=pen, name=f"Lifting-line theory (AR {ar:.1f})")
        item.setZValue(-6)
        self._theory_items.append(item)
        cd0 = float(np.min(polar.cd))
        cl_r = np.linspace(float(np.min(polar.cl)), float(np.max(polar.cl)), 40)
        item = self.drag.plot(cd0 + cl_r ** 2 / (np.pi * 0.9 * ar), cl_r, pen=pen)
        item.setZValue(-6)
        self._theory_items.append(item)

    def set_current(self, alpha_deg: Optional[float], cl: Optional[float] = None, cd: Optional[float] = None) -> None:
        """Highlight the current operating point (e.g. the latest single solve)."""
        if alpha_deg is None or cl is None or cd is None:
            for m in self.m_current:
                m.setData([], [])
            return
        ld = cl / cd if abs(cd) > 1e-9 else 0.0
        self.m_current[0].setData([alpha_deg], [cl])
        self.m_current[1].setData([cd], [cl])
        self.m_current[2].setData([alpha_deg], [ld])



# ---------------------------------------------------------------------------
# Live Flow Lab force history
# ---------------------------------------------------------------------------
class LiveForceChart(QWidget):
    """Rolling CL / CD history of the Live Flow Lab with the shedding frequency (Strouhal number)."""

    WINDOW = 40.0          # convective time units shown

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.plot = pg.PlotWidget()
        _style_plot(self.plot, "t*  (chord lengths travelled)", "Coefficient")
        self.plot.addLegend(offset=(-8, 8), labelTextColor=PALETTE.text_muted, brush=pg.mkBrush(PALETTE.surface_alt))
        self.c_cl = self.plot.plot(name="CL", pen=pg.mkPen(PALETTE.cyan, width=1.8))
        self.c_cd = self.plot.plot(name="CD", pen=pg.mkPen(PALETTE.warning, width=1.8))
        self.plot.addItem(pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen(PALETTE.border_strong, width=1)))
        self.hint = _Placeholder("Open the Live Flow Lab to see forces in real time")
        self.plot.addItem(self.hint, ignoreBounds=True)
        _show_empty(self.plot, self.hint, (0.0, self.WINDOW), (-1.0, 2.0))
        self.summary = QLabel("")
        self.summary.setProperty("role", "muted")
        self.summary.setFont(mono_font(11))
        self.summary.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(_card("Live forces", self.plot))
        lay.addWidget(self.summary)

    def update_stats(self, stats: dict) -> None:
        """Redraw from a Flow Lab stats dict (``history``, means, Strouhal)."""
        h = stats.get("history")
        if h is None or not h.t:
            self.c_cl.setData([], [])
            self.c_cd.setData([], [])
            _show_empty(self.plot, self.hint, (0.0, self.WINDOW), (-1.0, 2.0))
            self.summary.setText("")
            return
        self.hint.hide()
        t = np.asarray(h.t)
        sel = t >= t[-1] - self.WINDOW
        cl, cd = np.asarray(h.cl)[sel], np.asarray(h.cd)[sel]
        self.c_cl.setData(t[sel], cl)
        self.c_cd.setData(t[sel], cd)
        # Robust y range (2nd-98th percentile): the impulsive start-up spike must not flatten the signal.
        both = np.concatenate([cl, cd])
        lo, hi = float(np.percentile(both, 2.0)), float(np.percentile(both, 98.0))
        pad = max(0.15 * (hi - lo), 0.05)
        self.plot.setXRange(float(t[sel][0]), float(max(t[-1], t[sel][0] + 1e-6)), padding=0.02)
        self.plot.setYRange(min(lo, 0.0) - pad, hi + pad, padding=0.0)
        st = stats.get("st")
        st_txt = f"{st:.3f}" if st is not None else "not periodic yet"
        self.summary.setText(f"Mean over the last 10 t*:  CL {stats['cl_mean']:+.3f} · CD {stats['cd_mean']:.3f}\n"
                             f"Shedding Strouhal number St = f·D/U: {st_txt}")



# ---------------------------------------------------------------------------
# Spanwise load distribution
# ---------------------------------------------------------------------------
class SpanLoadChart(QWidget):
    """Spanwise lift distribution c·cl / c_mean versus 2y/b, with the ideal elliptic loading."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.plot = pg.PlotWidget()
        _style_plot(self.plot, "2y / b  (tip … root … tip)", "c·cl / c̄")
        # Bottom-centre and translucent: the loading curve fills the top of the chart, so a top-right
        # opaque legend hid half of it.
        legend = self.plot.addLegend(offset=(0, -6), labelTextColor=PALETTE.text_muted,
                                     brush=pg.mkBrush(PALETTE.surface_alt + "B0"), colCount=2)
        legend.anchor(itemPos=(0.5, 1.0), parentPos=(0.5, 1.0), offset=(0, -6))
        self.c_load = self.plot.plot(name="Computed", pen=pg.mkPen(PALETTE.cyan, width=2.2), symbol="o",
                                     symbolSize=4, symbolBrush=pg.mkBrush(PALETTE.cyan), symbolPen=None)
        self.c_ell = self.plot.plot(name="Elliptic (ideal)",
                                    pen=pg.mkPen(PALETTE.text_muted, width=1.4, style=pg.QtCore.Qt.PenStyle.DashLine))
        self.hint = _Placeholder("Lifting surfaces only — run a solve on a wing")
        self.plot.addItem(self.hint, ignoreBounds=True)
        self.info = QLabel("")
        self.info.setProperty("role", "muted")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(_card("Spanwise loading", self.plot, self.info))
        self.set_data(None)

    def set_data(self, extra: Optional[dict]) -> None:
        """Show the span load from ``SolverResult.extra`` (``None`` or a non-lifting body clears)."""
        if not extra or "span_eta" not in extra:
            self.c_load.setData([], [])
            self.c_ell.setData([], [])
            _show_empty(self.plot, self.hint, (-1.0, 1.0), (0.0, 1.5))
            self.info.setText("")
            return
        eta = np.asarray(extra["span_eta"], dtype=float)
        cl = np.asarray(extra["span_cl"], dtype=float)
        chord = np.asarray(extra["span_chord"], dtype=float)
        y = np.asarray(extra["span_y"], dtype=float)
        trapz = getattr(np, "trapezoid", None) or np.trapz
        span = max(float(extra.get("span_b", np.ptp(y))), 1e-9)
        c_mean = max(float(trapz(chord, y)) / span, 1e-9)
        load = cl * chord / c_mean
        cl_tot = float(trapz(load, eta)) / 2.0
        e = np.linspace(-1.0, 1.0, 161)
        ell = 4.0 / math.pi * cl_tot * np.sqrt(np.clip(1.0 - e * e, 0.0, 1.0))
        self.hint.hide()
        self.c_load.setData(eta, load)
        self.c_ell.setData(e, ell)
        self.plot.enableAutoRange()
        self.info.setText(f"CL ≈ {cl_tot:+.3f}")
