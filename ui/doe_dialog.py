"""Design of experiments window: vary several flight conditions together, run the batch in the background,
and read main effects, a response map, a fitted response surface and its optimum."""

from __future__ import annotations

import csv
import threading
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
    QProgressBar, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from config.settings import APP_NAME, FlowConditions, GridQuality, get_paths
from config.theme import PALETTE, mono_font
from core import doe
from visualization.telemetry_charts import _style_plot

#: (key, label, unit, lowest, highest, default from, default to, default levels, on by default)
FACTORS = (
    ("alpha_deg", "Angle of attack α", "deg", -20.0, 30.0, -4.0, 14.0, 7, True),
    ("velocity_ms", "Speed V", "m/s", 1.0, 340.0, 20.0, 120.0, 5, True),
    ("beta_deg", "Sideslip β", "deg", -20.0, 20.0, -10.0, 10.0, 5, False),
    ("altitude_m", "Altitude", "m", 0.0, 15000.0, 0.0, 10000.0, 5, False),
)


class _Runner(QObject):
    progress = Signal(int, int)
    done = Signal(object, str)


class _NumItem(QTableWidgetItem):
    def __lt__(self, other: QTableWidgetItem) -> bool:  # numeric sort
        try:
            return float(self.data(Qt.ItemDataRole.UserRole)) < float(other.data(Qt.ItemDataRole.UserRole))
        except (TypeError, ValueError):
            return super().__lt__(other)


def _spin(lo: float, hi: float, val: float, unit: str) -> QDoubleSpinBox:
    sp = QDoubleSpinBox()
    sp.setRange(lo, hi)
    sp.setDecimals(1)
    sp.setValue(val)
    sp.setSuffix(f" {unit}")
    sp.setKeyboardTracking(False)
    return sp


class DOEDialog(QDialog):
    def __init__(self, solver, model, base: FlowConditions, quality: GridQuality,
                 apply: Optional[Callable[[FlowConditions], None]] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Design of experiments — {model.name} — {APP_NAME}")
        self.resize(1320, 820)
        self.solver, self.model, self.base, self.quality, self._apply = solver, model, base, quality, apply
        self.result: Optional[doe.DOEResult] = None
        self.surface: Optional[doe.Surface] = None
        self._best: Optional[np.ndarray] = None
        self._cancel = threading.Event()

        # -- factors ------------------------------------------------------------------------------
        fgrid = QGridLayout()
        for c, h in enumerate(("Factor", "From", "To", "Levels")):
            lab = QLabel(h)
            lab.setProperty("role", "muted")
            fgrid.addWidget(lab, 0, c)
        self.rows: list[tuple[QCheckBox, QDoubleSpinBox, QDoubleSpinBox, QSpinBox, tuple]] = []
        for r, spec in enumerate(FACTORS, 1):
            key, label, unit, lo, hi, a, b, lv, on = spec
            cb = QCheckBox(label)
            cb.setChecked(on)
            s0, s1 = _spin(lo, hi, a, unit), _spin(lo, hi, b, unit)
            n = QSpinBox()
            n.setRange(2, 41)
            n.setValue(lv)
            for w in (cb, s0, s1, n):
                (w.toggled if isinstance(w, QCheckBox) else w.valueChanged).connect(lambda *_: self._count())
            fgrid.addWidget(cb, r, 0)
            fgrid.addWidget(s0, r, 1)
            fgrid.addWidget(s1, r, 2)
            fgrid.addWidget(n, r, 3)
            self.rows.append((cb, s0, s1, n, spec))
        self.kind = QComboBox()
        self.kind.addItems(doe.DESIGNS)
        self.kind.setToolTip("Full factorial: every combination of the levels. Latin hypercube: a space-filling "
                             "random design with any number of runs. Central composite: corners, face centres and "
                             "the centre point, the classic design for a quadratic response surface.")
        self.samples = QSpinBox()
        self.samples.setRange(4, 5000)
        self.samples.setValue(60)
        self.kind.currentIndexChanged.connect(lambda *_: self._count())
        self.samples.valueChanged.connect(lambda *_: self._count())
        self.count = QLabel("")
        self.count.setProperty("role", "muted")
        self.btn_run = QPushButton("Run batch")
        self.btn_run.setProperty("role", "primary")
        self.btn_run.clicked.connect(self._run)
        self.btn_csv = QPushButton("Export CSV…")
        self.btn_csv.setEnabled(False)
        self.btn_csv.clicked.connect(self._export)
        self.response = QComboBox()
        self.response.addItems(list(doe.RESPONSES))
        self.response.setCurrentText("L/D")
        self.goal = QComboBox()
        self.goal.addItems(["Maximise", "Minimise"])
        self.response.currentIndexChanged.connect(lambda *_: self._analyse())
        self.goal.currentIndexChanged.connect(lambda *_: self._analyse())
        self.btn_apply = QPushButton("Use the optimum in the wind tunnel")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._apply_best)

        left = QVBoxLayout()
        left.addLayout(fgrid)
        dl = QGridLayout()
        dl.addWidget(QLabel("Design"), 0, 0)
        dl.addWidget(self.kind, 0, 1)
        dl.addWidget(QLabel("Runs (hypercube)"), 1, 0)
        dl.addWidget(self.samples, 1, 1)
        left.addLayout(dl)
        left.addWidget(self.count)
        brow = QHBoxLayout()
        brow.addWidget(self.btn_run)
        brow.addWidget(self.btn_csv)
        left.addLayout(brow)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setFixedHeight(10)
        self.bar.setTextVisible(False)
        self.bar.hide()
        left.addWidget(self.bar)
        left.addSpacing(10)
        rl = QGridLayout()
        rl.addWidget(QLabel("Response"), 0, 0)
        rl.addWidget(self.response, 0, 1)
        rl.addWidget(QLabel("Goal"), 1, 0)
        rl.addWidget(self.goal, 1, 1)
        left.addLayout(rl)
        self.summary = QLabel("Choose the factors and press Run batch.")
        self.summary.setWordWrap(True)
        self.summary.setFont(mono_font(10))
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        left.addWidget(self.summary)
        left.addWidget(self.btn_apply)
        left.addStretch(1)
        fixed = QLabel(f"Factors not varied stay at the current values (V {base.velocity_ms:.1f} m/s, "
                       f"α {base.alpha_deg:.1f}°, β {base.beta_deg:.1f}°, {base.altitude_m:.0f} m). Instant Aero "
                       f"Engine, {quality.value} grid: the geometry is prepared once, so hundreds of runs take "
                       "seconds.")
        fixed.setWordWrap(True)
        fixed.setProperty("role", "muted")
        left.addWidget(fixed)
        lw = QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(470)

        # -- results ------------------------------------------------------------------------------
        self.tabs = QTabWidget()
        self.effects = pg.GraphicsLayoutWidget()
        self.tabs.addTab(self.effects, "Main effects")
        mapw = QWidget()
        ml = QVBoxLayout(mapw)
        mrow = QHBoxLayout()
        self.map_x, self.map_y = QComboBox(), QComboBox()
        for c in (self.map_x, self.map_y):
            c.currentIndexChanged.connect(lambda *_: self._draw_map())
        mrow.addWidget(QLabel("Horizontal"))
        mrow.addWidget(self.map_x)
        mrow.addWidget(QLabel("Vertical"))
        mrow.addWidget(self.map_y)
        mrow.addStretch(1)
        ml.addLayout(mrow)
        self.p_map = pg.PlotWidget()
        _style_plot(self.p_map, "", "")
        self.img = pg.ImageItem()
        self.p_map.addItem(self.img)
        self.scatter = pg.ScatterPlotItem(size=8, pen=pg.mkPen("#111", width=0.5))
        self.p_map.addItem(self.scatter)
        self.best_mark = pg.ScatterPlotItem(size=16, symbol="star", brush=pg.mkBrush(PALETTE.warning),
                                            pen=pg.mkPen("#111"))
        self.p_map.addItem(self.best_mark)
        self.cbar = pg.ColorBarItem(colorMap=pg.colormap.get("viridis"), interactive=False, width=14)
        self.cbar.setImageItem(self.img, insert_in=self.p_map.getPlotItem())
        ml.addWidget(self.p_map, 1)
        note = QLabel("Colours: the fitted quadratic surface (other factors at the optimum); dots: the runs.")
        note.setProperty("role", "muted")
        ml.addWidget(note)
        self.tabs.addTab(mapw, "Response map")
        self.table = QTableWidget(0, 0)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setFont(mono_font(10))
        self.tabs.addTab(self.table, "Runs")

        root = QHBoxLayout(self)
        root.addWidget(lw)
        root.addWidget(self.tabs, 1)
        self._count()

    # -- setup ------------------------------------------------------------------------------------
    def factors(self) -> list[doe.Factor]:
        out = []
        for cb, s0, s1, n, spec in self.rows:
            if cb.isChecked():
                a, b = sorted((s0.value(), s1.value()))
                if b - a < 1e-9:
                    b = a + 1e-3
                out.append(doe.Factor(spec[0], spec[1], spec[2], a, b, n.value()))
        return out

    def _matrix(self) -> np.ndarray:
        return doe.design(self.factors(), self.kind.currentText(), self.samples.value())

    def _count(self) -> None:
        hyper = self.kind.currentText() == "Latin hypercube"
        self.samples.setEnabled(hyper)
        for cb, s0, s1, n, _ in self.rows:
            n.setEnabled(cb.isChecked() and self.kind.currentText() == "Full factorial")
            s0.setEnabled(cb.isChecked())
            s1.setEnabled(cb.isChecked())
        f = self.factors()
        runs = len(self._matrix()) if f else 0
        self.count.setText(f"{len(f)} factors · {runs:,} runs" if f else "Tick at least one factor.")
        self.btn_run.setEnabled(bool(f) or self.btn_run.text() == "Cancel")

    # -- running ----------------------------------------------------------------------------------
    def _run(self) -> None:
        if self.btn_run.text() == "Cancel":
            self._cancel.set()
            return
        factors, X, kind = self.factors(), self._matrix(), self.kind.currentText()
        if not factors:
            return
        self._cancel.clear()
        self.btn_run.setText("Cancel")
        self.bar.setValue(0)
        self.bar.show()
        self.summary.setText(f"Running {len(X):,} cases…")
        runner = _Runner(self)
        runner.progress.connect(lambda i, n: self.bar.setValue(int(1000 * i / max(n, 1))))
        runner.done.connect(self._done)

        def work() -> None:
            try:
                res = doe.run(self.solver, self.model, self.base, self.quality, factors, X,
                              progress=lambda i, n: runner.progress.emit(i, n), cancelled=self._cancel.is_set,
                              kind=kind)
                runner.done.emit(res, "")
            except Exception as exc:  # noqa: BLE001
                runner.done.emit(None, str(exc))

        threading.Thread(target=work, name="aeroflow-doe", daemon=True).start()

    def _done(self, res: Optional[doe.DOEResult], error: str) -> None:
        self.btn_run.setText("Run batch")
        self.bar.hide()
        self._count()
        if error or res is None:
            self.summary.setText(f"The batch failed: {error}")
            return
        self.result = res
        self.btn_csv.setEnabled(res.runs > 0)
        names = [f.label for f in res.factors]
        for c in (self.map_x, self.map_y):
            c.blockSignals(True)
            c.clear()
            c.addItems(names)
            c.blockSignals(False)
        self.map_x.setCurrentIndex(0)
        self.map_y.setCurrentIndex(1 if len(names) > 1 else 0)
        self._fill_table()
        self._analyse()

    # -- analysis ---------------------------------------------------------------------------------
    def _analyse(self) -> None:
        res = self.result
        if res is None or res.runs == 0:
            return
        resp = self.response.currentText()
        maximise = self.goal.currentText() == "Maximise"
        y = res.Y[resp]
        lines = [f"{res.runs:,} runs ({res.design}).", ""]
        if np.isfinite(y).any():
            j = int(np.nanargmax(y) if maximise else np.nanargmin(y))
            pt = ", ".join(f"{f.label} {v:.2f} {f.unit}" for f, v in zip(res.factors, res.X[j]))
            lines.append(f"Best run: {resp} = {y[j]:.4g}")
            lines.append(f"  at {pt}")
        self.surface = doe.fit_quadratic(res, resp)
        self._best = None
        if self.surface is not None:
            best, val = doe.optimum(self.surface, maximise)
            self._best = best
            lines.append("")
            lines.append(f"Quadratic response surface: R² = {self.surface.r2:.4f}")
            lines.append(f"Predicted {'maximum' if maximise else 'minimum'}: {resp} ≈ {val:.4g} at")
            for f, v in zip(res.factors, best):
                lines.append(f"  {f.label}: {v:.2f} {f.unit}")
            if self.surface.r2 < 0.9:
                lines.append("(R² below 0.9: the response is not well described by a quadratic here; "
                             "narrow the ranges or trust the best run.)")
        else:
            lines.append("")
            lines.append("Too few runs for a quadratic surface (needs at least "
                         f"{(len(res.factors) + 1) * (len(res.factors) + 2) // 2}).")
        self.summary.setText("\n".join(lines))
        trust = self.surface is not None and self.surface.r2 >= 0.9
        self.btn_apply.setText("Use the predicted optimum in the wind tunnel" if trust
                               else "Use the best run in the wind tunnel")
        self.btn_apply.setEnabled(self._apply is not None and res.runs > 0)
        self._draw_effects()
        self._draw_map()

    def _draw_effects(self) -> None:
        res = self.result
        self.effects.clear()
        resp = self.response.currentText()
        eff = doe.main_effects(res, resp)
        col = 0
        first = None
        for (x, m), f in zip(eff, res.factors):
            p = self.effects.addPlot(row=0, col=col)
            p.showGrid(x=True, y=True, alpha=0.15)
            p.setLabel("bottom", f"{f.label} ({f.unit})")
            if col == 0:
                p.setLabel("left", f"mean {resp}")
                first = p
            else:
                p.setYLink(first)
            p.plot(x, m, pen=pg.mkPen(PALETTE.cyan, width=2), symbol="o", symbolSize=6,
                   symbolBrush=pg.mkBrush(PALETTE.cyan), symbolPen=None)
            col += 1

    def _draw_map(self) -> None:
        res = self.result
        if res is None or res.runs == 0:
            return
        ix, iy = self.map_x.currentIndex(), self.map_y.currentIndex()
        if ix < 0 or iy < 0:
            return
        fx, fy = res.factors[ix], res.factors[iy]
        resp = self.response.currentText()
        y = res.Y[resp]
        self.p_map.setLabel("bottom", f"{fx.label} ({fx.unit})")
        self.p_map.setLabel("left", f"{fy.label} ({fy.unit})")
        cmap = pg.colormap.get("viridis")
        fin = y[np.isfinite(y)]
        lo, hi = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
        if self.surface is not None and ix != iy:
            n = 80
            gx, gy = np.linspace(fx.lo, fx.hi, n), np.linspace(fy.lo, fy.hi, n)
            ref = self._best if self._best is not None else res.X.mean(axis=0)
            P = np.tile(ref, (n * n, 1))
            XX, YY = np.meshgrid(gx, gy, indexing="ij")
            P[:, ix], P[:, iy] = XX.ravel(), YY.ravel()
            Z = self.surface.predict(P).reshape(n, n)
            lo, hi = min(lo, float(np.nanmin(Z))), max(hi, float(np.nanmax(Z)))
            self.img.setImage(Z, levels=(lo, hi))
            self.img.setRect(fx.lo, fy.lo, fx.hi - fx.lo, fy.hi - fy.lo)
            self.img.show()
        else:
            self.img.hide()
        self.cbar.setLevels((lo, hi))
        norm = (y - lo) / (hi - lo if hi > lo else 1.0)
        brushes = [pg.mkBrush(cmap.map(float(np.clip(v, 0, 1)) if np.isfinite(v) else 0.0, mode="qcolor"))
                   for v in norm]
        self.scatter.setData(res.X[:, ix], res.X[:, iy], brush=brushes)
        if self._best is not None:
            self.best_mark.setData([self._best[ix]], [self._best[iy]])
        else:
            self.best_mark.setData([], [])

    def _fill_table(self) -> None:
        res = self.result
        heads = [f"{f.label} ({f.unit})" for f in res.factors] + list(doe.RESPONSES)
        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setColumnCount(len(heads))
        self.table.setHorizontalHeaderLabels(heads)
        self.table.setRowCount(res.runs)
        cols = [res.X[:, j] for j in range(len(res.factors))] + [res.Y[k] for k in doe.RESPONSES]
        for i in range(res.runs):
            for j, c in enumerate(cols):
                v = float(c[i])
                it = _NumItem(f"{v:.4g}" if abs(v) < 1e5 else f"{v:.4e}")
                it.setData(Qt.ItemDataRole.UserRole, v)
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(i, j, it)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSortingEnabled(True)

    # -- output -----------------------------------------------------------------------------------
    def _apply_best(self) -> None:
        res = self.result
        if res is None or self._apply is None:
            return
        pt = self._best if self.surface is not None and self.surface.r2 >= 0.9 else None
        if pt is None:
            y = res.Y[self.response.currentText()]
            pt = res.X[int(np.nanargmax(y) if self.goal.currentText() == "Maximise" else np.nanargmin(y))]
        self._apply(replace(self.base, **{f.key: float(v) for f, v in zip(res.factors, pt)}))

    def _export(self) -> None:
        res = self.result
        if res is None:
            return
        default = get_paths().exports / f"{self.model.name}_doe.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export the batch", str(default), "CSV (*.csv)")
        if not path:
            return
        export_csv(res, path)
        self.summary.setText(self.summary.text() + f"\n\nSaved {path}")


def export_csv(res: doe.DOEResult, path: str | Path) -> Path:
    p = Path(path)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([f.key for f in res.factors] + list(doe.RESPONSES))
        for i in range(res.runs):
            w.writerow([f"{v:.6g}" for v in res.X[i]] + [f"{res.Y[k][i]:.6g}" for k in doe.RESPONSES])
    return p


__all__ = ["DOEDialog", "export_csv"]
