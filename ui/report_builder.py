"""Combined report builder: pick any workspaces, labs and tool windows, add a title page with your logo and
notes, and write one HTML file and one PDF (see :mod:`core.report_book`)."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QSettings, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from config.settings import APP_NAME, ORG_NAME, get_paths
from core import report_book as rb


@dataclass
class Source:
    key: str
    title: str
    build: Callable[[], rb.Section]
    checked: bool = True


def png_bytes(pix: QPixmap) -> bytes:
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    pix.save(buf, "PNG")
    buf.close()
    return bytes(ba.data())


def capture(win: QWidget) -> bytes:
    """A picture of a window as shown on screen (includes 3-D OpenGL views), or rendered off-screen."""
    if win.isHidden():
        win.show()
    win.raise_()
    for _ in range(6):
        QApplication.processEvents()
    pix = QPixmap()
    scr = win.screen()
    if scr is not None and win.isWindow() and QApplication.platformName() not in ("offscreen", "minimal"):
        g = win.geometry()
        pix = scr.grabWindow(0, g.x() - scr.geometry().x(), g.y() - scr.geometry().y(), g.width(), g.height())
    if pix.isNull():
        pix = win.grab()
    return png_bytes(pix)


def capture_section(title: str, win: QWidget, text: str = "") -> rb.Section:
    return rb.Section(title, text=text, images=[(title, capture(win))])


def file_png(path: Path) -> bytes:
    return Path(path).read_bytes()


def tmpdir() -> Path:
    d = Path(tempfile.mkdtemp(prefix="aeroflow_report_"))
    return d


class ReportBuilderDialog(QDialog):
    def __init__(self, sources: list[Source], default_title: str, accent: str,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Combined report — {APP_NAME}")
        self.resize(720, 680)
        self.sources = sources
        self.accent = accent
        st = QSettings(ORG_NAME, APP_NAME)
        self.title = QLineEdit(default_title)
        self.subtitle = QLineEdit("")
        self.author = QLineEdit(str(st.value("report/author", "")))
        self.org = QLineEdit(str(st.value("report/organisation", "")))
        self.logo = QLineEdit(str(st.value("report/logo", "")))
        self.logo.setPlaceholderText("PNG, JPG or SVG (optional)")
        b_logo = QPushButton("Browse…")
        b_logo.clicked.connect(self._pick_logo)
        lrow = QHBoxLayout()
        lrow.addWidget(self.logo, 1)
        lrow.addWidget(b_logo)
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText("Summary, purpose, assumptions… (shown on the title page)")
        self.notes.setFixedHeight(110)
        form = QFormLayout()
        form.addRow("Title", self.title)
        form.addRow("Subtitle", self.subtitle)
        form.addRow("Author", self.author)
        form.addRow("Organisation", self.org)
        form.addRow("Logo", lrow)
        form.addRow("Notes", self.notes)
        self.list = QListWidget()
        for s in sources:
            it = QListWidgetItem(s.title)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if s.checked else Qt.CheckState.Unchecked)
            self.list.addItem(it)
        self.pdf = QCheckBox("Also write a PDF")
        self.pdf.setChecked(True)
        hint = QLabel("Sections come from what is open now: the wind-tunnel result and polar, the labs and every open "
                      "tool window. Open a tool first to include it. Each window is captured as it is shown.")
        hint.setWordWrap(True)
        hint.setProperty("role", "muted")
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(QLabel("Sections"))
        root.addWidget(self.list, 1)
        root.addWidget(hint)
        root.addWidget(self.pdf)
        root.addWidget(bb)

    def _pick_logo(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Logo", self.logo.text(), "Images (*.png *.jpg *.jpeg *.svg)")
        if p:
            self.logo.setText(p)

    def chosen(self) -> list[Source]:
        return [s for k, s in enumerate(self.sources) if self.list.item(k).checkState() == Qt.CheckState.Checked]

    def meta(self) -> rb.ReportMeta:
        st = QSettings(ORG_NAME, APP_NAME)
        st.setValue("report/author", self.author.text())
        st.setValue("report/organisation", self.org.text())
        st.setValue("report/logo", self.logo.text())
        logo = Path(self.logo.text()) if self.logo.text().strip() else None
        return rb.ReportMeta(self.title.text().strip() or "Engineering report", self.subtitle.text().strip(),
                             self.author.text().strip(), self.org.text().strip(), "",
                             self.notes.toPlainText(), logo)


def build_report(parent: QWidget, sources: list[Source], default_title: str, default_name: str, accent: str,
                 notify: Callable[[str, str], None]) -> Optional[Path]:
    """Run the dialog, ask for the file name, build every chosen section and write HTML (+ PDF)."""
    dlg = ReportBuilderDialog(sources, default_title, accent, parent)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    chosen = dlg.chosen()
    if not chosen:
        notify("No sections were selected.", "warning")
        return None
    path, _ = QFileDialog.getSaveFileName(parent, "Save the report", str(get_paths().exports / default_name),
                                          "HTML report (*.html)")
    if not path:
        return None
    sections, failed = [], []
    for s in chosen:
        try:
            sections.append(s.build())
        except Exception as exc:  # noqa: BLE001 - one broken section must not lose the report
            failed.append(f"{s.title}: {exc}")
    parent.raise_()
    out = rb.write(path, dlg.meta(), sections, accent)
    if failed:
        notify("Some sections were skipped: " + "; ".join(failed), "warning")
    if dlg.pdf.isChecked():
        from ui.pdf_export import html_to_pdf

        def done(p: Optional[Path], err: str) -> None:
            notify(f"PDF report saved: {p.name}" if p else err, "success" if p else "warning")

        html_to_pdf(out, out.with_suffix(".pdf"), done)
    notify(f"Report saved: {out.name}", "success")
    return out


__all__ = ["Source", "capture", "capture_section", "build_report", "ReportBuilderDialog", "png_bytes", "file_png",
           "tmpdir"]
