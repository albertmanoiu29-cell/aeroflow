"""Undo / redo for any tool window, built on the same input snapshots as the project files (:mod:`ui.state_io`).

:func:`install` watches every input of the window (spin boxes, combo boxes, text fields, check boxes and
sliders, including ones added later such as new table rows); a short pause after a change records a
snapshot. Ctrl+Z / Ctrl+Y (and Ctrl+Shift+Z) step through them, even while a field has the keyboard focus.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractButton, QAbstractItemView, QAbstractSlider, QAbstractSpinBox, QComboBox, QLineEdit, QWidget,
)

LIMIT = 100


class ToolUndo(QObject):
    """Snapshot history of one window's inputs."""

    changed = Signal()

    def __init__(self, win: QWidget, get: Callable[[], dict], put: Callable[[dict], None]) -> None:
        super().__init__(win)
        self.win, self._get, self._put = win, get, put
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []
        self._restoring = False
        self._watched: set[int] = set()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(350)
        self._timer.timeout.connect(self._record)
        self.current = self._safe_get()
        for seq, fn in ((QKeySequence.StandardKey.Undo, self.undo), (QKeySequence.StandardKey.Redo, self.redo),
                        (QKeySequence("Ctrl+Y"), self.redo)):
            sc = QShortcut(QKeySequence(seq), win)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(fn)
        self.watch()

    # -- watching -----------------------------------------------------------------------------------
    def watch(self) -> None:
        """Connect to every input not watched yet."""
        for w in self.win.findChildren(QWidget):
            key = id(w)
            if key in self._watched:
                continue
            if isinstance(w, QAbstractItemView) and w.model() is not None:   # rows added or removed
                self._watched.add(key)
                w.model().rowsInserted.connect(self._touched)
                w.model().rowsRemoved.connect(self._touched)
                continue
            sig = None
            if isinstance(w, QAbstractSpinBox) and hasattr(w, "valueChanged"):
                sig = w.valueChanged
            elif isinstance(w, QComboBox):
                sig = w.currentIndexChanged
            elif isinstance(w, QLineEdit) and not isinstance(w.parent(), (QAbstractSpinBox, QComboBox)):
                sig = w.editingFinished
            elif isinstance(w, QAbstractButton) and w.isCheckable():
                sig = w.toggled
            elif isinstance(w, QAbstractSlider):
                sig = w.valueChanged
            if sig is None:
                continue
            self._watched.add(key)
            sig.connect(self._touched)
            if isinstance(w, (QLineEdit, QAbstractSpinBox)):
                w.installEventFilter(self)
                le = w.lineEdit() if isinstance(w, QAbstractSpinBox) else None
                if le is not None:
                    le.installEventFilter(self)

    def eventFilter(self, obj: QObject, ev: QEvent) -> bool:  # noqa: N802
        # Text fields claim Ctrl+Z for their own one-field undo; let the window's undo have it instead.
        if ev.type() == QEvent.Type.ShortcutOverride:
            if ev.matches(QKeySequence.StandardKey.Undo) or ev.matches(QKeySequence.StandardKey.Redo) or (
                    ev.key() == Qt.Key.Key_Y and ev.modifiers() == Qt.KeyboardModifier.ControlModifier):
                ev.ignore()
                return True
        return False

    def _touched(self, *_: Any) -> None:
        if not self._restoring:
            self._timer.start()

    def _safe_get(self) -> Optional[dict]:
        try:
            return self._get()
        except Exception:  # noqa: BLE001 - a window being torn down
            return None

    def _record(self) -> None:
        st = self._safe_get()
        if st is None or st == self.current:
            return
        if self.current is not None:
            self.undo_stack.append(self.current)
            del self.undo_stack[:-LIMIT]
        self.current = st
        self.redo_stack.clear()
        self.watch()                                 # inputs added since (new table rows ...)
        self.changed.emit()

    # -- stepping -----------------------------------------------------------------------------------
    def _restore(self, st: dict) -> None:
        self._restoring = True
        try:
            self._put(st)
        finally:
            self._restoring = False
        self._timer.stop()
        self.current = self._safe_get()
        self.watch()
        self.changed.emit()

    def undo(self) -> None:
        if self._timer.isActive():                   # a change still pending: record it first
            self._timer.stop()
            self._record()
        if not self.undo_stack:
            return
        if self.current is not None:
            self.redo_stack.append(self.current)
        self._restore(self.undo_stack.pop())

    def redo(self) -> None:
        if not self.redo_stack:
            return
        if self.current is not None:
            self.undo_stack.append(self.current)
        self._restore(self.redo_stack.pop())

    def can_undo(self) -> bool:
        return bool(self.undo_stack) or self._timer.isActive()

    def can_redo(self) -> bool:
        return bool(self.redo_stack)


def install(win: QWidget, get: Callable[[], dict], put: Callable[[dict], None]) -> ToolUndo:
    u = ToolUndo(win, get, put)
    win._undo = u                                    # keep it with the window
    return u


__all__ = ["ToolUndo", "install"]
