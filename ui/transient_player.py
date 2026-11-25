"""Playback of a transient OpenFOAM result: frame slider / Play, CL & CD against time, Strouhal number."""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from config.settings import APP_NAME
from config.theme import PALETTE
from core import openfoam_bridge as ob
from visualization.telemetry_charts import _style_plot


def strouhal(t: np.ndarray, cl: np.ndarray, length: float, speed: float) -> Optional[float]:
    """Dominant lift frequency as St = f L / V (None without at least ~2 clear periods)."""
    if len(t) < 32:
        return None
    sel = t >= t[0] + 0.3 * (t[-1] - t[0])                   # skip the start-up transient
    t, cl = t[sel], cl[sel]
    if len(t) < 32 or np.std(cl) < 1e-4:
        return None
    tu = np.linspace(t[0], t[-1], 1024)
    s = np.interp(tu, t, cl)
    s = s - s.mean()
    spec = np.abs(np.fft.rfft(s * np.hanning(len(s))))
    freq = np.fft.rfftfreq(len(s), d=tu[1] - tu[0])
    k = int(np.argmax(spec[1:]) + 1)
    if freq[k] * (t[-1] - t[0]) < 2.0:                       # fewer than two periods: not trustworthy
        return None
    return float(freq[k] * length / max(speed, 1e-9))


class TransientPlayer(QDialog):
    """Steps the main 3-D view through the saved frames of a transient run."""

    def __init__(self, result: Any, show_frame: Callable[[Any], None], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Transient result — {APP_NAME}")
        self.resize(760, 520)
        self.base = result
        self._show = show_frame
        ex = result.extra
        self.model, self.layout, self.settings = ex["of_model"], ex["of_layout"], ex["of_settings"]
        self.frames = ob.transient_frames(self.layout.case_dir)
        self.cache: dict[int, Any] = {}
        speed = self.settings.conditions.velocity_ms
        self.length = max(float(self.model.info.chord_m), 1e-6)
        self.t_scale = speed / self.length                   # t* = t V / L
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        self.plot = pg.PlotWidget()
        _style_plot(self.plot, "Convective time t* = t·V/L", "Coefficient")
        self.plot.addLegend(offset=(-8, 8), labelTextColor=PALETTE.text_muted,
                            brush=pg.mkBrush(PALETTE.surface_alt + "B0"))
        hist = ob.force_history(self.layout.case_dir)
        info = []
        if hist:
            ts = hist["t"] * self.t_scale
            if "Cl" in hist:
                self.plot.plot(ts, hist["Cl"], pen=pg.mkPen(PALETTE.cyan, width=1.8), name="CL")
            if "Cd" in hist:
                self.plot.plot(ts, hist["Cd"], pen=pg.mkPen(PALETTE.warning, width=1.8), name="CD")
                n = len(hist["Cd"])
                tail = hist["Cd"][n // 2:]
                info.append(f"mean CD (2nd half) {np.mean(tail):.4f}")
            if "Cl" in hist:
                st = strouhal(hist["t"], hist["Cl"], self.length, speed)
                info.append(f"Strouhal St = f·L/V = {st:.3f}" if st is not None
                            else "no periodic shedding detected yet (run longer for a Strouhal number)")
        self.cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen("#FFFFFF", width=1, style=Qt.PenStyle.DashLine))
        self.plot.addItem(self.cursor, ignoreBounds=True)
        root.addWidget(self.plot, 1)
        self.info = QLabel(" · ".join(info) or "No force history found.")
        self.info.setProperty("role", "muted")
        self.info.setWordWrap(True)
        root.addWidget(self.info)
        row = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.setProperty("role", "primary")
        self.btn_play.clicked.connect(self._toggle)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, max(len(self.frames) - 1, 0))
        self.slider.valueChanged.connect(self._goto)
        self.label = QLabel("")
        self.label.setMinimumWidth(170)
        row.addWidget(self.btn_play)
        row.addWidget(self.slider, 1)
        row.addWidget(self.label)
        root.addLayout(row)
        self.timer = QTimer(self)
        self.timer.setInterval(350)
        self.timer.timeout.connect(self._next)
        if self.frames:
            self.slider.setValue(len(self.frames) - 1)
            self._goto(len(self.frames) - 1)

    def _frame_result(self, k: int) -> Any:
        if k not in self.cache:
            res = ob.load_openfoam_result(self.model, self.layout, self.settings,
                                          ob.RunReport(self.layout.case_dir, True), vtk_path=self.frames[k][1])
            res.extra.update({kk: v for kk, v in self.base.extra.items() if kk.startswith("of_")})
            if len(self.cache) > 40:                         # bounded memory on long runs
                self.cache.pop(next(iter(self.cache)))
            self.cache[k] = res
        return self.cache[k]

    def _goto(self, k: int) -> None:
        if not self.frames:
            return
        t = self.frames[k][0]
        self.label.setText(f"frame {k + 1}/{len(self.frames)} · t* {t * self.t_scale:.2f}")
        self.cursor.setPos(t * self.t_scale)
        try:
            self._show(self._frame_result(k))
        except Exception as exc:  # noqa: BLE001 - a missing frame must not stop the player
            self.label.setText(f"frame {k + 1}: {exc}")

    def _next(self) -> None:
        k = self.slider.value() + 1
        self.slider.setValue(0 if k >= len(self.frames) else k)

    def _toggle(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("Play")
        else:
            self.timer.start()
            self.btn_play.setText("Pause")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.timer.stop()
        super().closeEvent(event)


def finite(x: float) -> bool:
    return isinstance(x, float) and math.isfinite(x)
