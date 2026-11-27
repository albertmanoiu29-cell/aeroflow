"""Appearance settings: light or dark theme and the interface scale (applied at the next start)."""

from __future__ import annotations

import sys
from typing import Optional

from PySide6.QtCore import QProcess, QSettings
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QVBoxLayout, QWidget

from config.settings import APP_NAME, ORG_NAME
from config.theme import appearance

SCALES = (0.9, 1.0, 1.1, 1.25, 1.5, 1.75)


class AppearanceDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Appearance — {APP_NAME}")
        theme, scale = appearance()
        self.theme = QComboBox()
        self.theme.addItem("Dark", "dark")
        self.theme.addItem("Light", "light")
        self.theme.setCurrentIndex(1 if theme == "light" else 0)
        self.scale = QComboBox()
        for s in SCALES:
            self.scale.addItem(f"{s * 100:.0f} %", s)
        self.scale.setCurrentIndex(min(range(len(SCALES)), key=lambda k: abs(SCALES[k] - scale)))
        form = QFormLayout()
        form.addRow("Theme", self.theme)
        form.addRow("Interface size (text, controls, icons)", self.scale)
        note = QLabel("Applied when the program starts again. The interface size multiplies the Windows display "
                      "scaling. The flow views keep their dark backgrounds in both themes.")
        note.setWordWrap(True)
        note.setProperty("role", "muted")
        buttons = QDialogButtonBox()
        self.btn_restart = buttons.addButton("Save and restart now", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton("Save (apply at next start)", QDialogButtonBox.ButtonRole.ApplyRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.clicked.connect(self._clicked)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(note)
        lay.addWidget(buttons)
        self._restart = False

    def _clicked(self, btn) -> None:
        role = self.sender().buttonRole(btn)
        if role == QDialogButtonBox.ButtonRole.RejectRole:
            self.reject()
            return
        st = QSettings(ORG_NAME, APP_NAME)
        st.setValue("ui/theme", self.theme.currentData())
        st.setValue("ui/scale", float(self.scale.currentData()))
        st.sync()
        self._restart = btn is self.btn_restart
        self.accept()

    @property
    def restart_requested(self) -> bool:
        return self._restart


def restart_application(main_window: QWidget) -> None:
    """Start a new instance with the same arguments, then close this one (saving its state)."""
    QProcess.startDetached(sys.executable, sys.argv)
    main_window.close()
    QApplication.instance().quit()
