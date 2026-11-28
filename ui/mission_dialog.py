"""Mission planner: stack stages with engines from the library, see the Δv budget and the payload to orbit,
optimise the propellant split and fly the vehicle."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QSpinBox,
    QTableWidget, QVBoxLayout, QWidget,
)

from config.settings import APP_NAME
from config.theme import PALETTE, mono_font
from core import engine_db as edb
from core import mission as ms
from ui.cfd_panel import button
from visualization.telemetry_charts import _card, _style_plot


def _spin(lo: float, hi: float, val: float, dec: int, suffix: str, step: float) -> QDoubleSpinBox:
    sp = QDoubleSpinBox()
    sp.setRange(lo, hi)
    sp.setDecimals(dec)
    sp.setSingleStep(step)
    sp.setValue(val)
    sp.setKeyboardTracking(False)
    if suffix:
        sp.setSuffix(" " + suffix)
    return sp


def _engines() -> list:
    """Library engines first, then the built-in presets; plus a vacuum version of each booster."""
    lib = edb.load()
    out = list(lib)
    for e in edb.builtin():
        out.append(e)
        g = e.geometry()
        if g.area_ratio() < 60 and not g.is_aerospike:
            vac = replace(g, exit_diameter=g.throat_diameter * math.sqrt(100.0))
            out.append(edb.record_from(e.name.split(" (")[0].split(" ·")[0] + " · vacuum nozzle (ε 100)", vac, e.gas(),
                                       e.pc, e.propellant, "built-in, vacuum version"))
    return out


class MissionDialog(QDialog):
    COLS = ("Engine", "Engines", "Propellant (t)", "Dry mass (t)")

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Mission planner — {APP_NAME}")
        self.resize(1280, 820)
        self.engines = _engines()
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for k in range(1, len(self.COLS)):
            hh.setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(True)
        self.payload = _spin(0.0, 1e6, 10.0, 2, "t", 0.5)
        self.orbit = _spin(150.0, 2000.0, 200.0, 0, "km", 50.0)
        self.losses = _spin(500.0, 4000.0, 1700.0, 0, "m/s", 100.0)
        self.losses.setToolTip("Gravity, drag and steering losses on the way to orbit (1.5-2 km/s typical)")
        self.diam = _spin(0.2, 15.0, 3.7, 2, "m", 0.1)
        form = QFormLayout()
        form.addRow("Payload", self.payload)
        form.addRow("Target orbit", self.orbit)
        form.addRow("Ascent losses", self.losses)
        form.addRow("Vehicle diameter", self.diam)
        b_add = button("Add stage", "ghost", "", "Add a stage on top")
        b_del = button("Remove top stage", "ghost", "")
        b_opt = button("Optimise propellant split", "", "", "Move propellant between stages (tanks scale with it) to "
                       "carry the most payload")
        b_max = button("Maximum payload", "", "", "The heaviest payload that still reaches the orbit")
        b_fly = button("Fly it", "primary", "play", "Simulate the ascent of the first two stages")
        b_add.clicked.connect(lambda: self._add_row())
        b_del.clicked.connect(self._remove_row)
        b_opt.clicked.connect(self._optimise)
        b_max.clicked.connect(self._max_payload)
        b_fly.clicked.connect(self._fly)
        row = QHBoxLayout()
        for b in (b_add, b_del, b_opt, b_max, b_fly):
            row.addWidget(b)
        row.addStretch(1)
        self.out = QLabel("")
        self.out.setFont(mono_font(11))
        self.out.setWordWrap(True)
        self.out.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.p_dv = pg.PlotWidget()
        _style_plot(self.p_dv, "stage", "Δv (km/s)")
        self.bars = pg.BarGraphItem(x=[], height=[], width=0.6, brush=pg.mkBrush(PALETTE.cyan))
        self.p_dv.addItem(self.bars)
        self.req_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(PALETTE.danger, width=1, style=Qt.PenStyle.DashLine),
                                        label="required (total)", labelOpts={"color": PALETTE.danger})
        self.p_dv.addItem(self.req_line, ignoreBounds=True)
        self.p_alt = pg.PlotWidget()
        _style_plot(self.p_alt, "downrange (km)", "altitude (km)")
        self.c_alt = self.p_alt.plot(pen=pg.mkPen(PALETTE.success, width=2))
        grid = QGridLayout()
        grid.addWidget(_card("Δv by stage (stacked total vs required)", self.p_dv), 0, 0)
        grid.addWidget(_card("Ascent (first two stages)", self.p_alt), 0, 1)
        left = QVBoxLayout()
        left.addWidget(QLabel("Stages, bottom to top"))
        left.addWidget(self.table, 1)
        left.addLayout(row)
        top = QHBoxLayout()
        top.addLayout(left, 3)
        fw = QWidget()
        fw.setLayout(form)
        top.addWidget(fw, 1)
        root = QVBoxLayout(self)
        root.addLayout(top, 2)
        root.addWidget(self.out)
        root.addLayout(grid, 3)
        note = QLabel("Tsiolkovsky Δv per stage (first stage: mean of sea-level and vacuum Isp; above: vacuum Isp); "
                      "orbit needs v_circ + the ascent losses. Engines come from the engine library (Nozzle Lab ▸ Save "
                      "to engine library) and the built-in presets. The flight check uses a simple gravity turn and "
                      "altitude-holding upper-stage steering, so it is more conservative than the Δv budget.")
        note.setWordWrap(True)
        note.setProperty("role", "muted")
        root.addWidget(note)
        for p in self.payload, self.orbit, self.losses:
            p.valueChanged.connect(lambda _v: self.recompute())
        names = [e.name for e in self.engines]
        first = next((i for i, n in enumerate(names) if n.startswith("Sea-level booster")), 0)
        vac = next((i for i, n in enumerate(names) if "vacuum nozzle" in n and "Sea-level" in n), None)
        if vac is None:
            vac = next((i for i, n in enumerate(names) if "vacuum" in n.lower()), 0)
        self._add_row(first, 9, 400.0, 25.0)
        self._add_row(vac, 1, 100.0, 4.5)
        self.recompute()

    # -- table ---------------------------------------------------------------------------------------
    def _add_row(self, engine_index: int = 0, count: int = 1, prop_t: float = 50.0, dry_t: float = 4.0) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        cb = QComboBox()
        for e in self.engines:
            cb.addItem(e.name)
        cb.setCurrentIndex(min(engine_index, len(self.engines) - 1))
        n = QSpinBox()
        n.setRange(1, 60)
        n.setValue(count)
        p = _spin(0.001, 1e5, prop_t, 2, "", 1.0)
        d = _spin(0.001, 1e4, dry_t, 3, "", 0.5)
        for w in (cb, n, p, d):
            (w.currentIndexChanged if isinstance(w, QComboBox) else w.valueChanged).connect(
                lambda *_a: self.recompute())
        self.table.setCellWidget(r, 0, cb)
        self.table.setCellWidget(r, 1, n)
        self.table.setCellWidget(r, 2, p)
        self.table.setCellWidget(r, 3, d)
        self.table.setVerticalHeaderItem(r, None)
        self.recompute()

    def get_state(self) -> dict:
        rows = []
        for r in range(self.table.rowCount()):
            cb, n, p, d = (self.table.cellWidget(r, c) for c in range(4))
            rows.append({"engine": cb.currentText(), "engines": n.value(), "propellant_t": p.value(), "dry_t": d.value()})
        return {"rows": rows, "payload": self.payload.value(), "orbit": self.orbit.value(),
                "losses": self.losses.value(), "diameter": self.diam.value()}

    def set_state(self, st: dict) -> None:
        if not isinstance(st, dict) or not st.get("rows"):
            return
        while self.table.rowCount():
            self.table.removeRow(0)
        names = [e.name for e in self.engines]
        for row in st["rows"]:
            k = names.index(row["engine"]) if row.get("engine") in names else 0
            self._add_row(k, int(row.get("engines", 1)), float(row.get("propellant_t", 50.0)),
                          float(row.get("dry_t", 4.0)))
        for w, key in ((self.payload, "payload"), (self.orbit, "orbit"), (self.losses, "losses"), (self.diam, "diameter")):
            if key in st:
                w.setValue(float(st[key]))
        self.recompute()

    def _remove_row(self) -> None:
        if self.table.rowCount() > 1:
            self.table.removeRow(self.table.rowCount() - 1)
            self.recompute()

    def stages(self) -> list[ms.StageSpec]:
        out = []
        for r in range(self.table.rowCount()):
            cb, n, p, d = (self.table.cellWidget(r, c) for c in range(4))
            if cb is None:
                continue
            out.append(ms.StageSpec(self.engines[cb.currentIndex()], n.value(), p.value() * 1e3, d.value() * 1e3))
        return out

    def _set_stages(self, stages: list[ms.StageSpec]) -> None:
        for r, st in enumerate(stages):
            p, d = self.table.cellWidget(r, 2), self.table.cellWidget(r, 3)
            for w, v in ((p, st.propellant / 1e3), (d, st.dry / 1e3)):
                w.blockSignals(True)
                w.setValue(v)
                w.blockSignals(False)
        self.recompute()

    # -- actions --------------------------------------------------------------------------------------
    def _optimise(self) -> None:
        st = ms.optimise_split(self.stages(), self.orbit.value(), self.losses.value())
        self._set_stages(st)

    def _max_payload(self) -> None:
        self.payload.setValue(ms.max_payload(self.stages(), self.orbit.value(), self.losses.value()) / 1e3)

    def _fly(self) -> None:
        fl = ms.fly(self.stages(), self.payload.value() * 1e3, self.diam.value())
        self.c_alt.setData(fl.downrange / 1e3, fl.altitude / 1e3)
        self._flight_msg = fl.message
        self.recompute()

    def recompute(self) -> None:
        stages = self.stages()
        if not stages:
            return
        r = ms.analyse(stages, self.payload.value() * 1e3, self.orbit.value(), self.losses.value())
        xs = np.arange(1, len(r.stages) + 1)
        self.bars.setOpts(x=xs, height=[s.dv / 1e3 for s in r.stages], width=0.6)
        self.req_line.setPos(r.dv_required / 1e3)
        lines = [f"Lift-off mass {r.liftoff_mass / 1e3:,.1f} t · payload {r.payload / 1e3:,.2f} t "
                 f"({r.payload_fraction * 100:.2f} % of lift-off)"]
        for k, s in enumerate(r.stages, 1):
            lines.append(f"Stage {k}: Δv {s.dv:,.0f} m/s · Isp {s.isp:.0f} s · burn {s.burn_time:.0f} s · "
                         f"T/W at ignition {s.twr:.2f}" + (" ⚠ below 1: cannot lift off" if k == 1 and s.twr < 1.0
                                                           else ""))
        lines.append(f"Total Δv {r.dv_total:,.0f} m/s against {r.dv_required:,.0f} required "
                     f"({'margin' if r.margin >= -1.0 else 'SHORT by'} {abs(r.margin):,.0f} m/s): "
                     + ("reaches orbit." if r.margin >= -1.0 else "does not reach orbit."))
        msg = getattr(self, "_flight_msg", "")
        if msg:
            lines.append(f"Flight simulation: {msg}")
        self.out.setText("\n".join(lines))


__all__ = ["MissionDialog"]
