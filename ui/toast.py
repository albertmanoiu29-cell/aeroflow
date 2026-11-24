"""Non-modal toast notifications that stack in the bottom-right corner of the main window.

Used for confirmations (file saved, polar complete) and for errors that should not interrupt the
user with a modal dialog (for example a failed live re-solve). Each toast fades out on its own;
clicking one dismisses it immediately.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer
from PySide6.QtWidgets import QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QWidget

from config.theme import PALETTE, make_icon

_LEVELS = {
    "info": (PALETTE.cyan, "info"),
    "success": (PALETTE.success, "check"),
    "warning": (PALETTE.warning, "warning"),
    "error": (PALETTE.danger, "warning"),
}
MAX_TOASTS = 4


class Toast(QFrame):
    """One notification card."""

    def __init__(self, host: "ToastManager", text: str, level: str, seconds: float) -> None:
        super().__init__(host.parent_widget)
        self._host = host
        color, icon = _LEVELS.get(level, _LEVELS["info"])
        self.setObjectName("Toast")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"#Toast {{ background: {PALETTE.surface_alt}; border: 1px solid {PALETTE.border_strong};"
            f" border-left: 4px solid {color}; border-radius: 10px; }}"
            f"#Toast QLabel {{ color: {PALETTE.text}; background: transparent; }}")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 10, 14, 10)
        lay.setSpacing(10)
        ic = QLabel()
        ic.setPixmap(make_icon(icon, 18, color).pixmap(18, 18))
        lay.addWidget(ic, 0, Qt.AlignmentFlag.AlignTop)
        msg = QLabel(text)
        msg.setWordWrap(True)
        msg.setMaximumWidth(380)
        msg.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        lay.addWidget(msg, 1)
        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(1.0)
        self.setGraphicsEffect(self._effect)
        self._fade = QPropertyAnimation(self._effect, b"opacity", self)
        self._fade.setDuration(350)
        self._fade.setStartValue(1.0)
        self._fade.setEndValue(0.0)
        self._fade.setEasingCurve(QEasingCurve.Type.InQuad)
        self._fade.finished.connect(self._done)
        QTimer.singleShot(int(seconds * 1000), self._fade.start)
        self.adjustSize()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._fade.stop()
        self._done()

    def _done(self) -> None:
        self._host.remove(self)


class ToastManager:
    """Owns the toasts of one window and keeps them stacked above the status bar."""

    def __init__(self, parent_widget: QWidget, bottom_margin: int = 40) -> None:
        self.parent_widget = parent_widget
        self.bottom_margin = bottom_margin
        self.toasts: list[Toast] = []

    def show(self, text: str, level: str = "info", seconds: Optional[float] = None) -> None:
        if seconds is None:
            seconds = 6.0 if level in ("warning", "error") else 3.2
        t = Toast(self, text, level, seconds)
        self.toasts.append(t)
        while len(self.toasts) > MAX_TOASTS:
            old = self.toasts.pop(0)
            old.deleteLater()
        t.show()
        t.raise_()
        self.relayout()

    def remove(self, toast: Toast) -> None:
        if toast in self.toasts:
            self.toasts.remove(toast)
        toast.hide()
        toast.deleteLater()
        self.relayout()

    def relayout(self) -> None:
        w, h = self.parent_widget.width(), self.parent_widget.height()
        y = h - self.bottom_margin
        for t in reversed(self.toasts):
            t.adjustSize()
            y -= t.height() + 8
            t.move(max(8, w - t.width() - 18), max(8, y))
            t.raise_()
