"""HTML -> PDF with Qt WebEngine (renders the report's inline SVG charts exactly like a browser)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QMarginsF, QObject, QUrl
from PySide6.QtGui import QPageLayout, QPageSize

log = logging.getLogger("aeroflow.ui.pdf")

_jobs: list[QObject] = []          # keep pages alive until printing finishes


def html_to_pdf(html_path: Path, pdf_path: Path, done: Callable[[Optional[Path], str], None]) -> None:
    """Load ``html_path`` off-screen and print it to an A4 PDF; ``done(path or None, error)`` follows."""
    try:
        from PySide6.QtWebEngineCore import QWebEnginePage
    except ImportError:
        done(None, "PDF export needs Qt WebEngine (PySide6-Addons); the HTML report was saved.")
        return
    page = QWebEnginePage()
    _jobs.append(page)
    layout = QPageLayout(QPageSize(QPageSize.PageSizeId.A4), QPageLayout.Orientation.Portrait,
                         QMarginsF(12, 12, 12, 12), QPageLayout.Unit.Millimeter)

    def finished(path: str, ok: bool) -> None:
        if page in _jobs:
            _jobs.remove(page)
        page.deleteLater()
        done(Path(path) if ok else None, "" if ok else "Printing the PDF failed.")

    def loaded(ok: bool) -> None:
        if not ok:
            if page in _jobs:
                _jobs.remove(page)
            page.deleteLater()
            done(None, "The report could not be loaded for PDF export.")
            return
        page.printToPdf(str(pdf_path), layout)

    page.pdfPrintingFinished.connect(finished)
    page.loadFinished.connect(loaded)
    page.load(QUrl.fromLocalFile(str(Path(html_path).resolve())))
