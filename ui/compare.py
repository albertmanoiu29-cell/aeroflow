"""Side-by-side comparison of two results: linked 3-D views, a coefficient table and a ΔCp surface."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from config.settings import APP_NAME
from config.theme import mono_font
from core.geometry_manager import GeometryModel
from core.results import SolverResult
from visualization.cfd_viewport import CFDViewport


def _label(r: SolverResult) -> str:
    k = r.kpis
    return f"{r.engine} · α {k.alpha_deg:+.1f}° · β {k.beta_deg:+.1f}° · V {k.velocity_ms:.0f} m/s"


class CompareWindow(QMainWindow):
    """A (pinned) vs B (current): both views follow one camera."""

    def __init__(self, model_a: GeometryModel, a: SolverResult, model_b: GeometryModel, b: SolverResult,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Compare results — {APP_NAME}")
        self.resize(1600, 900)
        self.a, self.b = a, b
        root = QWidget()
        lay = QVBoxLayout(root)
        lay.setContentsMargins(8, 8, 8, 8)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.views: list[CFDViewport] = []
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
        self.setCentralWidget(root)
        self._fill_table()
        self._link_cameras()

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

        def sync(src: CFDViewport, dst: CFDViewport) -> None:
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
