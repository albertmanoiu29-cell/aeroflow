"""Update check UI: background checker, the "update available" dialog and the in-app installer download."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QTextBrowser, QVBoxLayout, QWidget,
)

from config.settings import APP_NAME, APP_VERSION
from config.theme import make_logo_pixmap
from core import updates
from core.updates import ReleaseInfo, UpdateError


class UpdateChecker(QObject):
    """Runs :func:`core.updates.latest_release` on a worker thread; results arrive on the GUI thread."""

    finished = Signal(object, str)          # (ReleaseInfo or None, error message or "")

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    def start(self) -> None:
        if self._busy:
            return
        self._busy = True

        def run() -> None:
            try:
                info, err = updates.latest_release(), ""
            except UpdateError as exc:
                info, err = None, str(exc)
            except Exception as exc:  # noqa: BLE001 - never crash the app over an update check
                info, err = None, f"Unexpected error: {exc}"
            self._busy = False
            self.finished.emit(info, err)

        threading.Thread(target=run, name="aeroflow-update-check", daemon=True).start()


class _Downloader(QObject):
    progress = Signal(int, int)
    done = Signal(str, str)                 # (installer path or "", error or "")


class UpdateDialog(QDialog):
    """Shows the new version and its notes; downloads and starts the installer on request."""

    install_started = Signal()              # the installer is running: the app should quit

    def __init__(self, info: ReleaseInfo, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.info = info
        self.setWindowTitle(f"{APP_NAME} — Update available")
        self.setMinimumSize(560, 460)
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        head = QHBoxLayout()
        logo = QLabel()
        logo.setPixmap(make_logo_pixmap(48))
        head.addWidget(logo)
        title = QLabel(f"{APP_NAME} {info.version} is available")
        title.setProperty("role", "title")
        sub = QLabel(f"You have version {APP_VERSION}. Your cases, exports and settings are kept.")
        sub.setProperty("role", "muted")
        col = QVBoxLayout()
        col.addWidget(title)
        col.addWidget(sub)
        head.addLayout(col, 1)
        root.addLayout(head)
        notes = QTextBrowser()
        notes.setOpenExternalLinks(True)
        notes.setMarkdown(info.notes or "_No release notes._")
        root.addWidget(notes, 1)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.hide()
        self.status = QLabel("")
        self.status.setProperty("role", "muted")
        self.status.setWordWrap(True)
        root.addWidget(self.bar)
        root.addWidget(self.status)
        buttons = QDialogButtonBox()
        can_install = sys.platform == "win32" and bool(info.installer_url)
        self.btn_install = QPushButton("Download and install")
        self.btn_install.setProperty("role", "primary")
        self.btn_install.setVisible(can_install)
        self.btn_page = QPushButton("Open release page")
        self.btn_later = QPushButton("Later")
        buttons.addButton(self.btn_install, QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(self.btn_page, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(self.btn_later, QDialogButtonBox.ButtonRole.RejectRole)
        root.addWidget(buttons)
        self.btn_install.clicked.connect(self._download)
        self.btn_page.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(info.page_url)))
        self.btn_later.clicked.connect(self._cancel_or_close)
        self._cancel = threading.Event()
        self._dl: Optional[_Downloader] = None

    def _cancel_or_close(self) -> None:
        if self._dl is not None:
            self._cancel.set()
        else:
            self.reject()

    def _download(self) -> None:
        self.btn_install.setEnabled(False)
        self.btn_later.setText("Cancel")
        self.bar.show()
        self.status.setText("Downloading the installer…")
        self._cancel.clear()
        dl = _Downloader(self)
        dl.progress.connect(self._on_progress)
        dl.done.connect(self._on_done)
        self._dl = dl
        dest = Path(tempfile.gettempdir()) / "aeroflow_update"

        def run() -> None:
            try:
                path = updates.download(self.info, dest, progress=lambda d, t: dl.progress.emit(d, t),
                                        cancelled=self._cancel.is_set)
                dl.done.emit(str(path), "")
            except UpdateError as exc:
                dl.done.emit("", str(exc))
            except Exception as exc:  # noqa: BLE001
                dl.done.emit("", f"Unexpected error: {exc}")

        threading.Thread(target=run, name="aeroflow-update-download", daemon=True).start()

    def _on_progress(self, done: int, total: int) -> None:
        if total > 0:
            self.bar.setValue(int(1000 * done / total))
            self.status.setText(f"Downloading the installer… {done / 1e6:.1f} of {total / 1e6:.1f} MB")

    def _on_done(self, path: str, error: str) -> None:
        self._dl = None
        self.btn_later.setText("Later")
        if error:
            self.bar.hide()
            self.btn_install.setEnabled(True)
            self.status.setText(error)
            return
        self.bar.setValue(1000)
        self.status.setText("Starting the installer — AeroFlow Studio closes so it can be updated…")
        try:
            flags = 0
            if sys.platform == "win32":
                flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            # /CLOSEAPPLICATIONS lets the installer close anything still holding files of the old version.
            subprocess.Popen([path, "/CLOSEAPPLICATIONS"], creationflags=flags, close_fds=True,  # noqa: S603
                             cwd=os.path.dirname(path))
        except OSError as exc:
            self.btn_install.setEnabled(True)
            self.status.setText(f"Could not start the installer: {exc}. It was saved to {path}.")
            return
        self.install_started.emit()
        self.accept()
