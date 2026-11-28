"""Guided tour: the window dims, one part of it at a time is cut out and outlined, and a card next to it
explains what it is. Back / Next / Skip, the arrow keys, Enter and Esc work; a step can prepare the window
first (switch workspace, open a panel)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from config.theme import PALETTE


@dataclass
class Step:
    title: str
    text: str
    target: Optional[Callable[[], Optional[QWidget]]] = None     # None: a centred card, no cut-out
    prepare: Optional[Callable[[], None]] = None


class TourOverlay(QWidget):
    finished = Signal(bool)                 # True when the tour was completed, False when skipped

    def __init__(self, host: QWidget, steps: list[Step]) -> None:
        super().__init__(host)
        self.host, self.steps, self.i = host, steps, 0
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.hole = QRect()
        self.card = QFrame(self)
        self.card.setObjectName("TourCard")
        self.card.setStyleSheet(
            f"#TourCard{{background:{PALETTE.surface if hasattr(PALETTE, 'surface') else '#1A1D22'};"
            f"border:1px solid {PALETTE.cyan};border-radius:12px}}"
            "#TourCard QLabel{background:transparent}")
        self.card.setFixedWidth(380)
        self.count = QLabel()
        self.count.setProperty("role", "muted")
        self.title = QLabel()
        self.title.setStyleSheet("font-size:16px;font-weight:600")
        self.title.setWordWrap(True)
        self.body = QLabel()
        self.body.setWordWrap(True)
        self.body.setTextFormat(Qt.TextFormat.RichText)
        self.b_skip = QPushButton("Skip tour")
        self.b_back = QPushButton("Back")
        self.b_next = QPushButton("Next")
        self.b_next.setProperty("role", "primary")
        self.b_skip.clicked.connect(lambda: self.close_tour(False))
        self.b_back.clicked.connect(self.back)
        self.b_next.clicked.connect(self.next)
        row = QHBoxLayout()
        row.addWidget(self.b_skip)
        row.addStretch(1)
        row.addWidget(self.b_back)
        row.addWidget(self.b_next)
        lay = QVBoxLayout(self.card)
        lay.setContentsMargins(18, 16, 18, 14)
        lay.setSpacing(8)
        lay.addWidget(self.count)
        lay.addWidget(self.title)
        lay.addWidget(self.body)
        lay.addSpacing(4)
        lay.addLayout(row)
        host.installEventFilter(self)
        self.setGeometry(host.rect())
        self.show()
        self.raise_()
        self.setFocus()
        self._show_step()

    # -- geometry --------------------------------------------------------------------------------------
    def eventFilter(self, obj: QObject, ev: QEvent) -> bool:  # noqa: N802
        if obj is self.host and ev.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self.setGeometry(self.host.rect())
            self._place()
        return False

    def _target_rect(self) -> QRect:
        st = self.steps[self.i]
        w = st.target() if st.target else None
        if w is None or not w.isVisible():
            return QRect()
        top = w.mapTo(self.host, QPoint(0, 0))
        return QRect(top, w.size()).adjusted(-6, -6, 6, 6).intersected(self.rect().adjusted(4, 4, -4, -4))

    def _place(self) -> None:
        self.hole = self._target_rect()
        self.card.adjustSize()
        cw, ch = self.card.width(), self.card.height()
        r, m = self.hole, 18
        W, H = self.width(), self.height()
        if r.isNull():
            x, y = (W - cw) // 2, (H - ch) // 2
        elif r.right() + m + cw < W:                     # right of the target
            x, y = r.right() + m, r.top()
        elif r.left() - m - cw > 0:                      # left of it
            x, y = r.left() - m - cw, r.top()
        elif r.bottom() + m + ch < H:                    # below
            x, y = r.left(), r.bottom() + m
        else:                                            # above
            x, y = r.left(), r.top() - m - ch
        x = max(12, min(x, W - cw - 12))
        y = max(12, min(y, H - ch - 12))
        self.card.move(x, y)
        self.update()

    # -- steps -----------------------------------------------------------------------------------------
    def _show_step(self) -> None:
        st = self.steps[self.i]
        if st.prepare is not None:
            try:
                st.prepare()
            except Exception:  # noqa: BLE001 - a tour step must never break the window
                pass
        self.count.setText(f"{self.i + 1} of {len(self.steps)}")
        self.title.setText(st.title)
        self.body.setText(st.text)
        self.b_back.setEnabled(self.i > 0)
        last = self.i == len(self.steps) - 1
        self.b_next.setText("Finish" if last else "Next")
        self.b_skip.setVisible(not last)
        self.raise_()
        self._place()
        self.setFocus()

    def next(self) -> None:
        if self.i >= len(self.steps) - 1:
            self.close_tour(True)
            return
        self.i += 1
        self._show_step()

    def back(self) -> None:
        if self.i > 0:
            self.i -= 1
            self._show_step()

    def close_tour(self, completed: bool) -> None:
        self.host.removeEventFilter(self)
        self.hide()
        self.finished.emit(completed)
        self.deleteLater()

    # -- input and painting ----------------------------------------------------------------------------
    def keyPressEvent(self, ev: QKeyEvent) -> None:  # noqa: N802
        k = ev.key()
        if k in (Qt.Key.Key_Right, Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.next()
        elif k == Qt.Key.Key_Left:
            self.back()
        elif k == Qt.Key.Key_Escape:
            self.close_tour(False)
        else:
            super().keyPressEvent(ev)

    def mousePressEvent(self, ev) -> None:  # noqa: N802 - clicks outside the card are swallowed
        ev.accept()

    def paintEvent(self, _ev) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRect(QRectF(self.rect()))
        if not self.hole.isNull():
            hole = QPainterPath()
            hole.addRoundedRect(QRectF(self.hole), 10, 10)
            path = path.subtracted(hole)
        p.fillPath(path, QColor(5, 7, 10, 175))
        if not self.hole.isNull():
            p.setPen(QPen(QColor(PALETTE.cyan), 2))
            p.drawRoundedRect(QRectF(self.hole), 10, 10)


__all__ = ["Step", "TourOverlay"]
