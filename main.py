"""AeroFlow Studio entry point.

Usage::

    python main.py                 # start with the default NACA 2412 wing
    python main.py path/to/model.stl   # start and import a mesh
"""

from __future__ import annotations

import os

# Must be set before anything imports pyvistaqt / qtpy, so the PySide6 binding is used even when
# PyQt5 / PyQt6 are also installed.
os.environ["QT_API"] = "pyside6"

import importlib  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

REQUIRED_MODULES: tuple[tuple[str, str], ...] = (
    ("PySide6", "PySide6"), ("numpy", "numpy"), ("scipy", "scipy"), ("vtk", "vtk"), ("pyvista", "pyvista"),
    ("pyvistaqt", "pyvistaqt"), ("pyqtgraph", "pyqtgraph"), ("PIL", "Pillow"), ("matplotlib", "matplotlib"),
)
OPTIONAL_MODULES: tuple[tuple[str, str], ...] = (
    ("trimesh", "trimesh"), ("stl", "numpy-stl"), ("imageio", "imageio"), ("numba", "numba"),
)


def _prepare_environment() -> None:
    """Platform tweaks that must happen before the QApplication exists."""
    # VTK's Qt widget needs an X11/GLX (or EGL-on-X) surface; under Wayland fall back to XWayland.
    if sys.platform.startswith("linux") and os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    # Make sure the package directory is importable when launched from elsewhere.
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)


def _missing_modules() -> tuple[list[str], list[str]]:
    """Return ``(missing required, missing optional)`` pip package names."""
    missing, optional = [], []
    for module, package in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
        except Exception:  # ImportError, or a broken binary wheel
            missing.append(package)
    for module, package in OPTIONAL_MODULES:
        try:
            importlib.import_module(module)
        except Exception:
            optional.append(package)
    return missing, optional


def _report_missing(missing: list[str]) -> None:
    req = Path(__file__).resolve().parent / "requirements.txt"
    if sys.platform == "win32":
        fix = ("Easiest fix: double-click setup_windows.bat, then start the app with run_windows.bat.\n\n"
               "Or install into this exact Python:\n\n"
               f'    "{sys.executable}" -m pip install -r "{req}"')
    else:
        fix = f'Install them into this exact Python with:\n\n    "{sys.executable}" -m pip install -r "{req}"'
    msg = ("AeroFlow Studio cannot start because these packages are missing:\n\n    "
           + ", ".join(missing)
           + f"\n\nThis is the Python that is running the app:\n\n    {sys.executable}  (Python "
           + sys.version.split()[0] + ")\n\nIf you already installed the libraries, they went into a different "
           "Python.\n\n" + fix)
    sys.stderr.write(msg + "\n")
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "AeroFlow Studio — missing packages", msg)
        del app
    except Exception:
        pass


def _show_splash(app, name: str, version: str):  # noqa: ANN001, ANN202
    """A branded splash screen shown while the heavy libraries load (``None`` if it cannot be drawn)."""
    try:
        import math

        from PySide6.QtCore import QRectF, Qt
        from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPixmap
        from PySide6.QtWidgets import QSplashScreen

        from config.theme import PALETTE, make_logo_pixmap

        w, h = 560, 300
        pix = QPixmap(w, h)
        pix.fill(QColor(PALETTE.bg_deep))
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        grad = QLinearGradient(0, 0, w, h)
        grad.setColorAt(0.0, QColor(PALETTE.surface))
        grad.setColorAt(1.0, QColor(PALETTE.bg_deep))
        p.fillRect(0, 0, w, h, grad)
        # stylised streamlines
        p.setPen(QColor(PALETTE.cyan).darker(260))
        for k in range(9):
            y0 = 60 + k * 22
            path_pts = [(x, y0 + 14.0 * math.sin((x + 40 * k) / 70.0)) for x in range(0, w + 1, 8)]
            for (x1, y1), (x2, y2) in zip(path_pts, path_pts[1:]):
                p.drawLine(int(x1), int(y1), int(x2), int(y2))
        p.drawPixmap(40, 70, make_logo_pixmap(84))
        p.setPen(QColor(PALETTE.text))
        f = QFont()
        f.setPixelSize(34)
        f.setBold(True)
        p.setFont(f)
        p.drawText(QRectF(140, 72, w - 160, 44), Qt.AlignmentFlag.AlignVCenter, name)
        f.setPixelSize(14)
        f.setBold(False)
        p.setFont(f)
        p.setPen(QColor(PALETTE.text_muted))
        p.drawText(QRectF(142, 116, w - 160, 24), Qt.AlignmentFlag.AlignVCenter,
                   f"Version {version} · 3-D CFD · polars · live flow lab")
        p.setPen(QColor(PALETTE.border_strong))
        p.drawRect(0, 0, w - 1, h - 1)
        p.end()
        splash = QSplashScreen(pix)
        splash.show()
        app.processEvents()
        return splash
    except Exception:  # noqa: BLE001 - a splash must never stop the app from starting
        return None


def _splash_messenger(app, splash):  # noqa: ANN001, ANN202
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor

    def say(text: str) -> None:
        if splash is not None:
            splash.showMessage(text, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft, QColor("#94A3B8"))
            app.processEvents()
    return say


def main() -> int:
    """Start the application and return its exit code."""
    _prepare_environment()
    missing, optional = _missing_modules()
    if missing:
        _report_missing(missing)
        return 1

    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    # Qt 6 enables high-DPI scaling automatically; pass fractional factors (125 %, 150 %) straight through.
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)

    from config.settings import APP_ID, APP_NAME, APP_VERSION
    from config.theme import apply_theme, make_logo_icon
    from utils.logger import get_logger, install_exception_hooks, setup_logging

    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("AeroFlow")
    app.setDesktopFileName(APP_ID)
    apply_theme(app)
    app.setWindowIcon(make_logo_icon(64))
    splash = _show_splash(app, APP_NAME, APP_VERSION)
    splash_msg = _splash_messenger(app, splash)
    splash_msg("Starting…")
    setup_logging()
    install_exception_hooks()
    log = get_logger("main")
    for pkg in optional:
        log.info("Optional package '%s' not installed — built-in fallbacks will be used.", pkg)

    splash_msg("Loading the 3-D engine (VTK / PyVista)…")
    import pyvista as pv
    try:
        pv.global_theme.allow_empty_mesh = True      # an empty streamline set must never raise
    except AttributeError:
        pass
    try:
        pv.global_theme.smooth_shading = True
    except AttributeError:
        pass

    splash_msg("Loading solvers and the user interface…")
    from ui.main_window import MainWindow

    splash_msg("Building the workspace…")
    window = MainWindow()
    window.show()
    if splash is not None:
        splash.finish(window)
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
        path = sys.argv[1]
        QTimer.singleShot(800, lambda: window.load_file(path))
    log.info("%s %s started (Python %s, Qt binding PySide6, PyVista %s)", APP_NAME, APP_VERSION,
             sys.version.split()[0], pv.__version__)
    return int(app.exec())


if __name__ == "__main__":
    sys.exit(main())
