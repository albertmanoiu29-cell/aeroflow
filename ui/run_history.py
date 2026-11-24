"""Run history: every solve is logged in a sortable table that can be exported and re-applied.

Double-clicking a row (or *Apply conditions*) sends that run's flow conditions back to the controls,
so earlier operating points can be revisited and compared quickly.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from config.settings import FlowConditions, get_paths
from config.theme import mono_font
from core.results import SolverResult

MAX_RUNS = 500


@dataclass(frozen=True)
class RunRecord:
    """One logged solve."""

    stamp: str
    geometry: str
    engine: str
    conditions: FlowConditions
    cl: float
    cd: float
    cm: float
    l_over_d: float
    reynolds: float
    seconds: float

    def cells(self) -> list[str]:
        # Key numbers first so they are visible in the narrow side panel; descriptive columns after.
        c = self.conditions
        return [self.stamp, f"{c.alpha_deg:+.1f}", f"{c.velocity_ms:.1f}", f"{self.cl:+.4f}", f"{self.cd:.5f}",
                f"{self.l_over_d:.1f}", f"{self.cm:+.4f}", self.geometry, self.engine.replace(" Aero Engine", ""),
                f"{c.beta_deg:+.1f}", f"{self.reynolds:.2e}", f"{self.seconds:.2f}"]


HEADERS = ("Time", "α °", "V m/s", "CL", "CD", "L/D", "CM", "Geometry", "Engine", "β °", "Re", "Solve s")
_NUMERIC = {1, 2, 3, 4, 5, 6, 9, 10, 11}


class _NumItem(QTableWidgetItem):
    """Table item that sorts numerically."""

    def __lt__(self, other: QTableWidgetItem) -> bool:  # noqa: D105
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return super().__lt__(other)


class RunHistoryWidget(QWidget):
    """Table of all solves in this session."""

    apply_requested = Signal(object)          # FlowConditions

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Root")
        self.records: list[RunRecord] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self.summary = QLabel("Every solve is listed here. Double-click a run to restore its conditions.")
        self.summary.setProperty("role", "muted")
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        self.table = QTableWidget(0, len(HEADERS))
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setSortingEnabled(True)
        self.table.setFont(mono_font(11))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(lambda r, _c: self._apply_row(r))
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.btn_apply = QPushButton("Apply conditions")
        self.btn_apply.setToolTip("Set the flow sliders to the selected run's conditions")
        self.btn_export = QPushButton("Export CSV…")
        self.btn_clear = QPushButton("Clear")
        for b in (self.btn_export, self.btn_clear):
            b.setProperty("role", "ghost")
        row.addWidget(self.btn_apply)
        row.addStretch(1)
        row.addWidget(self.btn_export)
        row.addWidget(self.btn_clear)
        lay.addLayout(row)
        self.btn_apply.clicked.connect(lambda: self._apply_row(self.table.currentRow()))
        self.btn_export.clicked.connect(self.export_dialog)
        self.btn_clear.clicked.connect(self.clear)

    # -- data ----------------------------------------------------------------
    def add_result(self, result: SolverResult, geometry: str) -> None:
        """Log one solver result."""
        k = result.kpis
        rec = RunRecord(time.strftime("%H:%M:%S"), geometry, result.engine, result.conditions, k.cl, k.cd, k.cm,
                        k.l_over_d, k.reynolds, result.elapsed_s)
        self.records.append(rec)
        if len(self.records) > MAX_RUNS:
            self.records = self.records[-MAX_RUNS:]
            self._refill()
            return
        self._append_row(rec, len(self.records) - 1)
        self._update_summary()

    def _append_row(self, rec: RunRecord, index: int) -> None:
        sorting = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, text in enumerate(rec.cells()):
            item = _NumItem(text) if c in _NUMERIC else QTableWidgetItem(text)
            if c in _NUMERIC:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            item.setData(Qt.ItemDataRole.UserRole, index)
            self.table.setItem(r, c, item)
        self.table.setSortingEnabled(sorting)
        self.table.scrollToBottom()

    def _refill(self) -> None:
        self.table.setRowCount(0)
        for i, rec in enumerate(self.records):
            self._append_row(rec, i)
        self._update_summary()

    def _update_summary(self) -> None:
        n = len(self.records)
        if not n:
            self.summary.setText("Every solve is listed here. Double-click a run to restore its conditions.")
            return
        best = max(self.records, key=lambda r: r.l_over_d)
        self.summary.setText(f"{n} run{'s' if n != 1 else ''} · best L/D {best.l_over_d:.1f} "
                             f"({best.geometry}, α {best.conditions.alpha_deg:+.1f}°)")

    def clear(self) -> None:
        self.records.clear()
        self.table.setRowCount(0)
        self._update_summary()

    def _apply_row(self, row: int) -> None:
        if row < 0:
            return
        item = self.table.item(row, 0)
        if item is None:
            return
        idx = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(idx, int) and 0 <= idx < len(self.records):
            self.apply_requested.emit(self.records[idx].conditions)

    # -- export ----------------------------------------------------------------
    def export_csv(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["time", "geometry", "engine", "alpha_deg", "beta_deg", "velocity_ms", "altitude_m",
                        "CL", "CD", "CM", "L_over_D", "Re", "solve_s"])
            for r in self.records:
                c = r.conditions
                w.writerow([r.stamp, r.geometry, r.engine, f"{c.alpha_deg:.4f}", f"{c.beta_deg:.4f}",
                            f"{c.velocity_ms:.4f}", f"{c.altitude_m:.1f}", f"{r.cl:.6f}", f"{r.cd:.6f}",
                            f"{r.cm:.6f}", f"{r.l_over_d:.4f}", f"{r.reynolds:.6e}", f"{r.seconds:.3f}"])
        return out

    def export_dialog(self) -> Optional[Path]:
        if not self.records:
            return None
        default = get_paths().exports / f"runs_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export run history", str(default), "CSV (*.csv)")
        return self.export_csv(path) if path else None
