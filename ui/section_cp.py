"""Spanwise Cp sections: chordwise pressure distributions at several stations along the span."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)

from config.settings import APP_NAME, get_paths
from config.theme import PALETTE
from visualization.telemetry_charts import _style_plot

_COLOURS = ("#22D3EE", "#F59E0B", "#A78BFA", "#34D399", "#F472B6", "#60A5FA", "#FBBF24", "#F87171")


class SectionCpDialog(QDialog):
    """Cp vs x/c at user-chosen spanwise stations of the current result (suction plotted upward)."""

    def __init__(self, viewport: Any, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.view = viewport
        self.setWindowTitle(f"Spanwise Cp sections — {APP_NAME}")
        self.resize(920, 620)
        self.sections: list[dict[str, Any]] = []
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        row = QHBoxLayout()
        row.addWidget(QLabel("Stations (% of semi-span from the centre):"))
        self.stations = QLineEdit("10, 30, 50, 70, 90")
        self.stations.setToolTip("Comma-separated percentages: 0 = centre line, 100 = wing tip (+y side)")
        self.stations.editingFinished.connect(self.refresh)
        row.addWidget(self.stations, 1)
        self.cuts = QCheckBox("Show cuts on the model")
        self.cuts.setChecked(True)
        self.cuts.toggled.connect(self.refresh)
        row.addWidget(self.cuts)
        btn = QPushButton("Update")
        btn.clicked.connect(self.refresh)
        row.addWidget(btn)
        self.btn_csv = QPushButton("Export CSV…")
        self.btn_csv.clicked.connect(self._export)
        row.addWidget(self.btn_csv)
        root.addLayout(row)
        self.plot = pg.PlotWidget()
        _style_plot(self.plot, "x / c  (local chord)", "Cp")
        self.plot.invertY(True)                   # aerodynamic convention: suction (negative Cp) upward
        self.plot.addLegend(offset=(-8, 8), labelTextColor=PALETTE.text_muted,
                            brush=pg.mkBrush(PALETTE.surface_alt + "B0"))
        root.addWidget(self.plot, 1)
        self.note = QLabel("")
        self.note.setProperty("role", "muted")
        self.note.setWordWrap(True)
        root.addWidget(self.note)
        self.refresh()

    def _fractions(self) -> list[float]:
        out = []
        for part in self.stations.text().replace(";", ",").split(","):
            try:
                out.append(float(part.strip().rstrip("%")) / 100.0)
            except ValueError:
                continue
        return [f for f in out if 0.0 <= f <= 1.0][:8] or [0.5]

    def refresh(self) -> None:
        self.plot.clear()
        self.sections = self.view.section_cp(self._fractions(), show_cuts=self.cuts.isChecked())
        if not self.sections:
            self.note.setText("No surface pressures to cut - run a simulation first (Surface Cp must be available).")
            return
        for k, s in enumerate(self.sections):
            col = _COLOURS[k % len(_COLOURS)]
            for surf, style in ((s["upper"], None), (~s["upper"], "dash")):
                order = np.argsort(s["x_c"][surf])
                pen = pg.mkPen(col, width=2, style=pg.QtCore.Qt.PenStyle.DashLine if style else
                               pg.QtCore.Qt.PenStyle.SolidLine)
                self.plot.plot(s["x_c"][surf][order], s["cp"][surf][order], pen=pen,
                               name=f"{100 * s['frac']:.0f} % span" if style is None else None)
        self.note.setText("Solid: upper surface · dashed: lower surface. The area between the curves is the "
                          "section lift; a flat, merging upper curve near the trailing edge signals separation.")

    def _export(self) -> None:
        if not self.sections:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export sections", str(get_paths().exports / "section_cp.csv"),
                                              "CSV (*.csv)")
        if not path:
            return
        with Path(path).open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["span_fraction", "y_m", "chord_m", "x_over_c", "cp", "surface"])
            for s in self.sections:
                for x, cp, up in zip(s["x_c"], s["cp"], s["upper"]):
                    w.writerow([f"{s['frac']:.4f}", f"{s['y']:.6g}", f"{s['chord']:.6g}", f"{x:.6g}", f"{cp:.6g}",
                                "upper" if up else "lower"])
        self.note.setText(f"Saved {path}")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.view.clear_sections()
        super().closeEvent(event)
