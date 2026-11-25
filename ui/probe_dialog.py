"""Line probe window: flow quantities along a line drawn through the 3-D result."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QDialog, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from config.settings import APP_NAME, get_paths
from config.theme import PALETTE
from visualization.telemetry_charts import _style_plot

_SERIES = (("speed_ratio", "Speed / V∞", PALETTE.cyan), ("Cp", "Pressure Cp", PALETTE.warning),
           ("Cp0", "Total pressure Cp0", "#A78BFA"), ("vorticity_mag", "Vorticity (1/s)", "#34D399"))


class LineProbeDialog(QDialog):
    """Four charts against the distance along the probe line; updated by each new line."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Line probe — {APP_NAME}")
        self.resize(900, 620)
        self.data: dict[str, np.ndarray] = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        self.info = QLabel("")
        self.info.setProperty("role", "muted")
        self.info.setWordWrap(True)
        root.addWidget(self.info)
        grid = QGridLayout()
        self.curves: dict[str, pg.PlotDataItem] = {}
        for k, (key, label, colour) in enumerate(_SERIES):
            pw = pg.PlotWidget()
            _style_plot(pw, "Distance along the line (m)", label)
            self.curves[key] = pw.plot(pen=pg.mkPen(colour, width=2), connect="finite")
            grid.addWidget(pw, k // 2, k % 2)
        root.addLayout(grid, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_csv = QPushButton("Export CSV…")
        self.btn_csv.clicked.connect(self._export)
        row.addWidget(self.btn_csv)
        root.addLayout(row)

    def set_data(self, data: dict[str, np.ndarray]) -> None:
        self.data = dict(data)
        d = self.data["distance"]
        v_inf = float(self.data.get("v_inf", np.array([1.0]))[0])
        if "speed" in self.data:
            self.data["speed_ratio"] = self.data["speed"] / max(v_inf, 1e-9)
        for key, _label, _c in _SERIES:
            y = self.data.get(key)
            self.curves[key].setData(d, y if y is not None else np.full_like(d, np.nan))
        a, b = self.data["a"], self.data["b"]
        inside = int(np.isnan(self.data.get("speed", np.zeros(1))).sum())
        self.info.setText(f"From ({a[0]:+.3f}, {a[1]:+.3f}, {a[2]:+.3f}) to ({b[0]:+.3f}, {b[1]:+.3f}, {b[2]:+.3f}) m, "
                          f"length {d[-1]:.3f} m, {len(d)} samples"
                          + (f" — {inside} samples inside the body or outside the domain are left blank." if inside else
                             "."))

    def _export(self) -> None:
        if not self.data:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export line probe", str(get_paths().exports / "line_probe.csv"),
                                              "CSV (*.csv)")
        if not path:
            return
        keys = ["distance"] + [k for k, _l, _c in _SERIES if k in self.data]
        with Path(path).open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(keys)
            for i in range(len(self.data["distance"])):
                w.writerow([f"{self.data[k][i]:.6g}" for k in keys])
        self.info.setText(f"Saved {path}")
