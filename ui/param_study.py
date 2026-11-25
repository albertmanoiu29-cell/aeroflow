"""Parametric study: sweep one flight condition (α, β, speed / Reynolds / Mach, altitude) with the Instant
engine and plot CL, CD, CM and L/D against it, with a table and CSV export."""

from __future__ import annotations

import csv
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QProgressBar,
    QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from config.settings import APP_NAME, FlowConditions, GridQuality, get_paths
from config.theme import PALETTE, mono_font
from core.geometry_manager import GeometryModel
from core.instant_solver import InstantSolver
from core.results import AeroKPIs
from visualization.telemetry_charts import _style_plot


@dataclass(frozen=True)
class StudyVariable:
    key: str            # FlowConditions field
    label: str
    unit: str
    lo: float
    hi: float
    default_from: float
    default_to: float


VARIABLES = (
    StudyVariable("beta_deg", "Sideslip β", "deg", -20.0, 20.0, -15.0, 15.0),
    StudyVariable("velocity_ms", "Speed V (Reynolds and Mach follow)", "m/s", 1.0, 340.0, 10.0, 250.0),
    StudyVariable("altitude_m", "Altitude", "m", 0.0, 15000.0, 0.0, 12000.0),
    StudyVariable("alpha_deg", "Angle of attack α", "deg", -20.0, 30.0, -6.0, 16.0),
)


class _Runner(QObject):
    progress = Signal(int, int)
    done = Signal(object, str)          # (list of (value, AeroKPIs)), error


def run_study(solver: InstantSolver, model: GeometryModel, base: FlowConditions, quality: GridQuality,
              var: StudyVariable, values: np.ndarray, progress=None, cancelled=None) -> list[tuple[float, AeroKPIs]]:
    """Evaluate the coefficients at each value (the geometry preparation is shared, so this is fast)."""
    prep = solver.cached_preparation(model, quality) or solver.prepare(model, quality)
    out = []
    for i, v in enumerate(values):
        if cancelled is not None and cancelled():
            break
        out.append((float(v), solver.coefficients(prep, replace(base, **{var.key: float(v)}))))
        if progress is not None:
            progress(i + 1, len(values))
    return out


class ParamStudyDialog(QDialog):
    """Non-modal study window."""

    def __init__(self, solver: InstantSolver, model: GeometryModel, base: FlowConditions, quality: GridQuality,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Parametric study — {model.name} — {APP_NAME}")
        self.resize(1100, 720)
        self.solver, self.model, self.base, self.quality = solver, model, base, quality
        self.rows: list[tuple[float, AeroKPIs]] = []
        self._cancel = threading.Event()
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        # controls
        ctl = QHBoxLayout()
        self.var = QComboBox()
        for v in VARIABLES:
            self.var.addItem(v.label, v)
        self.var.currentIndexChanged.connect(self._var_changed)
        self.v_from, self.v_to = QDoubleSpinBox(), QDoubleSpinBox()
        for sb in (self.v_from, self.v_to):
            sb.setDecimals(1)
            sb.setKeyboardTracking(False)
        self.n = QSpinBox()
        self.n.setRange(3, 200)
        self.n.setValue(25)
        self.btn_run = QPushButton("Run study")
        self.btn_run.setProperty("role", "primary")
        self.btn_run.clicked.connect(self._run)
        self.btn_csv = QPushButton("Export CSV…")
        self.btn_csv.setEnabled(False)
        self.btn_csv.clicked.connect(self._export)
        for w in (QLabel("Vary"), self.var, QLabel("from"), self.v_from, QLabel("to"), self.v_to, QLabel("points"),
                  self.n, self.btn_run, self.btn_csv):
            ctl.addWidget(w)
        ctl.addStretch(1)
        root.addLayout(ctl)
        fixed = QLabel(f"Other conditions fixed at the current values: V {base.velocity_ms:.1f} m/s, "
                       f"α {base.alpha_deg:.1f}°, β {base.beta_deg:.1f}°, altitude {base.altitude_m:.0f} m · "
                       f"Instant Aero Engine, {quality.value} grid")
        fixed.setProperty("role", "muted")
        fixed.setWordWrap(True)
        root.addWidget(fixed)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setFixedHeight(10)
        self.bar.setTextVisible(False)
        self.bar.hide()
        root.addWidget(self.bar)
        # charts + table
        split = QSplitter(Qt.Orientation.Vertical)
        charts = QWidget()
        grid = QGridLayout(charts)
        grid.setContentsMargins(0, 0, 0, 0)
        self.plots: dict[str, pg.PlotWidget] = {}
        self.curves: dict[str, pg.PlotDataItem] = {}
        colours = {"CL": PALETTE.cyan, "CD": PALETTE.warning, "CM": "#A78BFA", "L/D": "#34D399"}
        for k, name in enumerate(("CL", "CD", "CM", "L/D")):
            pw = pg.PlotWidget()
            _style_plot(pw, "", name)
            self.curves[name] = pw.plot(pen=pg.mkPen(colours[name], width=2), symbol="o", symbolSize=5,
                                        symbolBrush=pg.mkBrush(colours[name]), symbolPen=None)
            self.plots[name] = pw
            grid.addWidget(pw, k // 2, k % 2)
        split.addWidget(charts)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["Value", "Re", "Mach", "CL", "CD", "CM", "L/D", "Lift N"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setFont(mono_font(11))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        split.addWidget(self.table)
        split.setSizes([460, 220])
        root.addWidget(split, 1)
        self.note = QLabel("")
        self.note.setProperty("role", "muted")
        self.note.setWordWrap(True)
        root.addWidget(self.note)
        self._var_changed()

    def _variable(self) -> StudyVariable:
        return self.var.currentData()

    def _var_changed(self) -> None:
        v = self._variable()
        for sb in (self.v_from, self.v_to):
            sb.setRange(v.lo, v.hi)
            sb.setSuffix(f" {v.unit}")
        self.v_from.setValue(v.default_from)
        self.v_to.setValue(v.default_to)
        for pw in self.plots.values():
            pw.setLabel("bottom", f"{v.label.split(' (')[0]} ({v.unit})")

    def _run(self) -> None:
        if self.btn_run.text() == "Cancel":
            self._cancel.set()
            return
        var = self._variable()
        values = np.linspace(self.v_from.value(), self.v_to.value(), int(self.n.value()))
        self._cancel.clear()
        self.btn_run.setText("Cancel")
        self.bar.setValue(0)
        self.bar.show()
        self.note.setText("Computing…")
        runner = _Runner(self)
        runner.progress.connect(lambda i, n: self.bar.setValue(int(1000 * i / max(n, 1))))
        runner.done.connect(lambda rows, err, v=var: self._done(v, rows, err))

        def work() -> None:
            try:
                rows = run_study(self.solver, self.model, self.base, self.quality, var, values,
                                 progress=lambda i, n: runner.progress.emit(i, n), cancelled=self._cancel.is_set)
                runner.done.emit(rows, "")
            except Exception as exc:  # noqa: BLE001
                runner.done.emit([], str(exc))

        threading.Thread(target=work, name="aeroflow-param-study", daemon=True).start()

    def _done(self, var: StudyVariable, rows: list[tuple[float, AeroKPIs]], error: str) -> None:
        self.btn_run.setText("Run study")
        self.bar.hide()
        if error:
            self.note.setText(f"The study failed: {error}")
            return
        self.rows, self._var_done = rows, var
        x = np.array([r[0] for r in rows])
        k = [r[1] for r in rows]
        cl = np.array([a.cl for a in k])
        cd = np.array([a.cd for a in k])
        cm = np.array([a.cm for a in k])
        ld = np.where(np.abs(cd) > 1e-9, cl / np.where(np.abs(cd) > 1e-9, cd, 1.0), np.nan)
        for name, y in (("CL", cl), ("CD", cd), ("CM", cm), ("L/D", ld)):
            self.curves[name].setData(x, y)
        self.table.setRowCount(len(rows))
        for i, (v, a) in enumerate(rows):
            vals = (f"{v:.2f}", f"{a.reynolds:.3g}", f"{a.mach:.3f}", f"{a.cl:+.4f}", f"{a.cd:.5f}", f"{a.cm:+.4f}",
                    f"{ld[i]:.2f}", f"{a.lift_n:,.1f}")
            for j, t in enumerate(vals):
                item = QTableWidgetItem(t)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(i, j, item)
        self.btn_csv.setEnabled(bool(rows))
        if len(ld) and np.isfinite(ld).any():
            j = int(np.nanargmax(ld))
            self.note.setText(f"{len(rows)} points. Best L/D {ld[j]:.2f} at {var.label.split(' (')[0]} = "
                              f"{x[j]:.2f} {var.unit}.")
        else:
            self.note.setText(f"{len(rows)} points.")

    def _export(self) -> None:
        if not self.rows:
            return
        var = self._var_done
        default = get_paths().exports / f"{self.model.name}_study_{var.key}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export study", str(default), "CSV (*.csv)")
        if not path:
            return
        with Path(path).open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow([f"{var.key}", "reynolds", "mach", "cl", "cd", "cm", "l_over_d", "lift_n", "drag_n"])
            for v, a in self.rows:
                w.writerow([f"{v:.6g}", f"{a.reynolds:.6g}", f"{a.mach:.6g}", f"{a.cl:.6g}", f"{a.cd:.6g}",
                            f"{a.cm:.6g}", f"{a.cl / a.cd if abs(a.cd) > 1e-12 else float('nan'):.6g}",
                            f"{a.lift_n:.6g}", f"{a.drag_n:.6g}"])
        self.note.setText(f"Saved {path}")
