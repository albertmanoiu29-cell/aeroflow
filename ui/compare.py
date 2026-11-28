"""Side-by-side comparison of two configurations: linked 3-D views, a coefficient table and a ΔCp surface;
and their polars overlaid, with the differences B − A and a table of key figures."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from config.settings import APP_NAME
from config.theme import PALETTE, mono_font
from core import polar_compare as pc
from core.geometry_manager import GeometryModel
from core.results import PolarData, SolverResult
from visualization.telemetry_charts import _style_plot


def _label(r: SolverResult) -> str:
    k = r.kpis
    return f"{r.engine} · α {k.alpha_deg:+.1f}° · β {k.beta_deg:+.1f}° · V {k.velocity_ms:.0f} m/s"


class CompareWindow(QMainWindow):
    """A (pinned) vs B (current): both views follow one camera."""

    def __init__(self, model_a: Optional[GeometryModel], a: Optional[SolverResult], model_b: Optional[GeometryModel],
                 b: Optional[SolverResult], parent: Optional[QWidget] = None,
                 polars: Optional[tuple[PolarData, PolarData]] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Compare A and B — {APP_NAME}")
        self.resize(1600, 900)
        self.a, self.b = a, b
        self.views: list = []
        tabs = QTabWidget()
        if a is not None and b is not None:
            tabs.addTab(self._flow_tab(model_a, a, model_b, b), "3-D flow and coefficients")
        if polars is not None:
            tabs.addTab(PolarCompare(*polars), "Polars")
        self.setCentralWidget(tabs)

    def _flow_tab(self, model_a: GeometryModel, a: SolverResult, model_b: GeometryModel, b: SolverResult) -> QWidget:
        from visualization.cfd_viewport import CFDViewport

        root = QWidget()
        lay = QVBoxLayout(root)
        lay.setContentsMargins(8, 8, 8, 8)
        split = QSplitter(Qt.Orientation.Horizontal)
        for title, model, res in (("A (pinned)", model_a, a), ("B (current)", model_b, b)):
            box = QWidget()
            bl = QVBoxLayout(box)
            bl.setContentsMargins(0, 0, 0, 0)
            cap = QLabel(f"<b>{title}</b> — {_label(res)}")
            cap.setTextFormat(Qt.TextFormat.RichText)
            bl.addWidget(cap)
            v = CFDViewport()
            v.update_settings(replace(v.settings, show_particles=False, lock_ranges=True), immediate=True)
            if self.views:                             # B uses A's colour ranges: identical scales
                v._locked = dict(self.views[0]._locked)
            v.set_geometry(model)
            v.set_result(res)
            bl.addWidget(v, 1)
            split.addWidget(box)
            self.views.append(v)
        lay.addWidget(split, 1)
        row = QHBoxLayout()
        self.diff = QCheckBox("Show ΔCp = Cp(B) − Cp(A) on B")
        self.diff.setToolTip("Colours B's surface by the pressure change (needs the same geometry mesh)")
        self.diff.toggled.connect(self._toggle_diff)
        row.addWidget(self.diff)
        self.note = QLabel("")
        self.note.setProperty("role", "muted")
        row.addWidget(self.note, 1)
        lay.addLayout(row)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Quantity", "A", "B", "B − A"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setFont(mono_font(11))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setMaximumHeight(230)
        lay.addWidget(self.table)
        self._fill_table()
        self._link_cameras()
        return root

    def _fill_table(self) -> None:
        ka, kb = self.a.kpis, self.b.kpis
        rows = [("CL", ka.cl, kb.cl, 4), ("CD", ka.cd, kb.cd, 5), ("CM", ka.cm, kb.cm, 4),
                ("L/D", ka.l_over_d, kb.l_over_d, 2), ("Lift (N)", ka.lift_n, kb.lift_n, 1),
                ("Drag (N)", ka.drag_n, kb.drag_n, 1), ("Cp min", ka.cp_min, kb.cp_min, 3)]
        self.table.setRowCount(len(rows))
        for i, (name, va, vb, d) in enumerate(rows):
            for j, text in enumerate((name, f"{va:.{d}f}", f"{vb:.{d}f}", f"{vb - va:+.{d}f}")):
                item = QTableWidgetItem(text)
                if j:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(i, j, item)

    def _link_cameras(self) -> None:
        """Copy A's camera to B whenever A's view changes (and B's to A), without feedback loops."""
        self._syncing = False

        def sync(src, dst) -> None:
            if self._syncing:
                return
            self._syncing = True
            try:
                dst.plotter.camera_position = src.plotter.camera_position
                dst.plotter.render()
            finally:
                self._syncing = False

        a, b = self.views
        a.plotter.iren.add_observer("EndInteractionEvent", lambda *_: sync(a, b))
        a.plotter.iren.add_observer("MouseWheelForwardEvent", lambda *_: sync(a, b))
        a.plotter.iren.add_observer("MouseWheelBackwardEvent", lambda *_: sync(a, b))
        b.plotter.iren.add_observer("EndInteractionEvent", lambda *_: sync(b, a))
        b.plotter.iren.add_observer("MouseWheelForwardEvent", lambda *_: sync(b, a))
        b.plotter.iren.add_observer("MouseWheelBackwardEvent", lambda *_: sync(b, a))
        sync(a, b)

    def _toggle_diff(self, on: bool) -> None:
        va, vb = self.views
        sa, sb = va._surface, vb._surface
        if on:
            if sa is None or sb is None or sa.n_points != sb.n_points or "Cp" not in sa.point_data:
                self.note.setText("ΔCp needs both results on the same surface mesh (same geometry and engine).")
                self.diff.setChecked(False)
                return
            sb.point_data["dCp"] = (np.asarray(sb.point_data["Cp"], dtype=np.float32)
                                    - np.asarray(sa.point_data["Cp"], dtype=np.float32))
            vb.update_settings(replace(vb.settings, surface_scalar="dCp"), immediate=True)
            self.note.setText("Red: B has higher pressure than A · blue: B has more suction.")
        else:
            vb.update_settings(replace(vb.settings, surface_scalar="Cp"), immediate=True)
            self.note.setText("")

    def closeEvent(self, event) -> None:  # noqa: N802
        for v in self.views:
            try:
                v.shutdown()
            except Exception:  # noqa: BLE001
                pass
        super().closeEvent(event)


class PolarCompare(QWidget):
    """Polars A and B overlaid, the differences B − A, and their key figures."""

    def __init__(self, a: PolarData, b: PolarData, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        ca = pg.mkPen("#9AA3AF", width=2, style=Qt.PenStyle.DashLine)
        cb = pg.mkPen(PALETTE.cyan, width=2)
        la = a.geometry_name or "pinned"
        lb = b.geometry_name or "current"
        grid = QGridLayout()
        lda, ldb = pc._ld(a), pc._ld(b)
        charts = (("Lift curve", "α (°)", "CL", (a.alpha_deg, a.cl), (b.alpha_deg, b.cl)),
                  ("Drag polar", "CD", "CL", (a.cd, a.cl), (b.cd, b.cl)),
                  ("Efficiency", "α (°)", "L/D", (a.alpha_deg, lda), (b.alpha_deg, ldb)),
                  ("Pitching moment", "α (°)", "CM", (a.alpha_deg, a.cm), (b.alpha_deg, b.cm)))
        for k, (title, xl, yl, da, db) in enumerate(charts):
            pw = pg.PlotWidget(title=title)
            _style_plot(pw, xl, yl)
            pw.addLegend(offset=(8, 8))
            pw.plot(*da, pen=ca, name="A")
            pw.plot(*db, pen=cb, name="B")
            grid.addWidget(pw, 0, k)
        d = pc.difference(a, b)
        for k, (title, y) in enumerate((("ΔCL = B − A", d.d_cl), ("ΔCD = B − A", d.d_cd), ("ΔCM = B − A", d.d_cm),
                                        ("Δ(L/D) = B − A", d.d_ld))):
            pw = pg.PlotWidget(title=title)
            _style_plot(pw, "α (°)", title.split(" =")[0])
            pw.addLine(y=0, pen=pg.mkPen("#6F7682", width=1))
            pw.plot(d.alpha, np.where(y >= 0, y, 0.0), pen=None, fillLevel=0, brush=pg.mkBrush(52, 211, 153, 70))
            pw.plot(d.alpha, np.where(y < 0, y, 0.0), pen=None, fillLevel=0, brush=pg.mkBrush(248, 113, 113, 70))
            pw.plot(d.alpha, y, pen=pg.mkPen(PALETTE.cyan, width=2))
            grid.addWidget(pw, 1, k)
        ma, mb = pc.metrics(a), pc.metrics(b)
        table = QTableWidget(len(ma), 4)
        table.setHorizontalHeaderLabels(["Key figure", "A", "B", "B − A"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setFont(mono_font(10))
        for i, ((name, va, dec), (_, vb, _)) in enumerate(zip(ma, mb)):
            for j, text in enumerate((name, f"{va:.{dec}f}", f"{vb:.{dec}f}", f"{vb - va:+.{dec}f}")):
                it = QTableWidgetItem(text)
                if j:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(i, j, it)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table.setMinimumWidth(520)
        table.setMaximumWidth(600)
        cap = QLabel(f"<b>A</b> (dashed): {la} · {a.engine} · Re {a.reynolds:.3g} &nbsp;&nbsp; "
                     f"<b>B</b>: {lb} · {b.engine} · Re {b.reynolds:.3g}. In the difference charts, green "
                     "shading is where B is higher.")
        cap.setTextFormat(Qt.TextFormat.RichText)
        cap.setWordWrap(True)
        top = QWidget()
        top.setLayout(grid)
        body = QHBoxLayout()
        body.addWidget(top, 1)
        body.addWidget(table)
        lay = QVBoxLayout(self)
        lay.addWidget(cap)
        lay.addLayout(body, 1)
