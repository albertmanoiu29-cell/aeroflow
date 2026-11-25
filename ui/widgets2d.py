"""Small widgets shared by the 2-D Labs (Live Flow Lab and Supersonic Lab)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QWidget

from config.settings import APP_NAME, ORG_NAME
from core.shapes2d import read_airfoil_dat
from ui.cfd_panel import button


class AirfoilFileRow(QWidget):
    """``[Load .dat…]  name`` - picks an airfoil coordinate file (UIUC / XFOIL format) and validates it."""

    def __init__(self, on_loaded: Callable[[str], None], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._on_loaded = on_loaded
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.btn = button("Load .dat…", "ghost", "folder",
                          "Airfoil coordinates in Selig or Lednicer format (e.g. from the UIUC airfoil database "
                          "or XFOIL)")
        self.btn.clicked.connect(self.browse)
        self.name = QLabel("No file loaded")
        self.name.setProperty("role", "muted")
        self.name.setWordWrap(True)
        lay.addWidget(self.btn)
        lay.addWidget(self.name, 1)

    def browse(self) -> None:
        s = QSettings(ORG_NAME, APP_NAME)
        start = str(s.value("labs/airfoil_dir", str(Path.home())))
        path, _ = QFileDialog.getOpenFileName(self, "Load airfoil coordinates", start,
                                              "Airfoil coordinates (*.dat *.txt *.csv);;All files (*)")
        if not path:
            return
        try:
            name, pts = read_airfoil_dat(path)
        except ValueError as exc:
            self.name.setText(str(exc))
            return
        s.setValue("labs/airfoil_dir", str(Path(path).parent))
        self.name.setText(f"{name} ({len(pts)} points)")
        self.name.setToolTip(path)
        self._on_loaded(path)
