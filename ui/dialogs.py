"""Modal dialogs: OpenFOAM setup guide, report export, and Help / About."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
    QPushButton, QScrollArea, QTabWidget, QVBoxLayout, QWidget,
)

from config.settings import APP_NAME, APP_VERSION, OPENFOAM_SETUP_URLS, get_paths
from config.theme import make_logo_pixmap, make_section_label, mono_font, set_property
from core.openfoam_bridge import OpenFOAMEnvironment, setup_instructions


def _card() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("Card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(14, 12, 14, 12)
    lay.setSpacing(8)
    return frame, lay


class OpenFOAMSetupDialog(QDialog):
    """Explains the current OpenFOAM status and how to install it on this platform."""

    def __init__(self, env: OpenFOAMEnvironment, redetect: Optional[Callable[[], None]] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("OpenFOAM Setup Guide")
        self.setMinimumSize(640, 560)
        self._redetect = redetect
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)
        title = QLabel("High-fidelity CFD with OpenFOAM")
        title.setProperty("role", "title")
        root.addWidget(title)
        intro = QLabel(
            "AeroFlow Studio always works with its built-in Instant Aero Engine. Installing OpenFOAM adds an "
            "optional high-fidelity RANS solver (k-ω SST). AeroFlow writes every case file, meshes your geometry "
            "and runs the solver for you — you never touch a dictionary or a terminal after installation.")
        intro.setWordWrap(True)
        intro.setProperty("role", "muted")
        root.addWidget(intro)

        status_card, sl = _card()
        sl.addWidget(make_section_label("Current status"))
        self.status = QLabel()
        self.status.setProperty("role", "badge")
        self.detail = QLabel()
        self.detail.setWordWrap(True)
        self.detail.setProperty("role", "muted")
        sl.addWidget(self.status, alignment=Qt.AlignmentFlag.AlignLeft)
        sl.addWidget(self.detail)
        root.addWidget(status_card)
        self.set_environment(env)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        il = QVBoxLayout(inner)
        il.setContentsMargins(0, 0, 0, 0)
        il.setSpacing(10)
        il.addWidget(make_section_label(f"Install on {platform.system() or 'your system'}"))
        for step_title, commands in setup_instructions(env):
            card, cl = _card()
            t = QLabel(step_title)
            t.setStyleSheet("font-weight: 700;")
            cl.addWidget(t)
            box = QPlainTextEdit(commands)
            box.setReadOnly(True)
            box.setFont(mono_font(11))
            box.setFixedHeight(min(28 + 18 * commands.count("\n") + 18, 140))
            cl.addWidget(box)
            copy = QPushButton("Copy")
            copy.setProperty("role", "ghost")
            copy.clicked.connect(lambda _=False, c=commands: QGuiApplication.clipboard().setText(c))
            cl.addWidget(copy, alignment=Qt.AlignmentFlag.AlignRight)
            il.addWidget(card)
        links_card, ll = _card()
        ll.addWidget(make_section_label("Official resources"))
        for name, url in OPENFOAM_SETUP_URLS.items():
            b = QPushButton(name)
            b.setProperty("role", "ghost")
            b.setToolTip(url)
            b.clicked.connect(lambda _=False, u=url: QDesktopServices.openUrl(QUrl(u)))
            ll.addWidget(b, alignment=Qt.AlignmentFlag.AlignLeft)
        il.addWidget(links_card)
        il.addStretch(1)
        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        buttons = QDialogButtonBox()
        self.btn_redetect = buttons.addButton("Re-detect", QDialogButtonBox.ButtonRole.ActionRole)
        self.btn_redetect.setEnabled(redetect is not None)
        self.btn_redetect.clicked.connect(self._on_redetect)
        close = buttons.addButton(QDialogButtonBox.StandardButton.Close)
        close.setProperty("role", "primary")
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def set_environment(self, env: OpenFOAMEnvironment) -> None:
        """Update the status badge (e.g. after re-detection)."""
        self.status.setText(env.badge_text)
        set_property(self.status, "level", "success" if env.available else "warning")
        self.detail.setText(env.detail)

    def _on_redetect(self) -> None:
        if self._redetect is not None:
            self.status.setText("Detecting…")
            set_property(self.status, "level", "")
            self._redetect()


@dataclass(frozen=True)
class ReportOptions:
    """What the user chose in :class:`ExportReportDialog`."""

    path: Path
    include_screenshot: bool
    include_polar_charts: bool
    write_csv: bool
    notes: str


class ExportReportDialog(QDialog):
    """Choose what goes into the HTML engineering report."""

    def __init__(self, default_name: str, has_polar: bool, has_result: bool, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Engineering Report")
        self.setMinimumWidth(520)
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(10)
        title = QLabel("Export HTML report")
        title.setProperty("role", "title")
        root.addWidget(title)
        card, cl = _card()
        cl.addWidget(make_section_label("File"))
        row = QHBoxLayout()
        self.path = QLineEdit(str(get_paths().exports / default_name))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(self.path, 1)
        row.addWidget(browse)
        cl.addLayout(row)
        cl.addWidget(make_section_label("Contents"))
        self.c_shot = QCheckBox("3-D view snapshot (streamlines / pressure)")
        self.c_shot.setChecked(has_result)
        self.c_shot.setEnabled(has_result)
        self.c_polar = QCheckBox("Polar charts (lift curve, drag polar, L/D)")
        self.c_polar.setChecked(has_polar)
        self.c_polar.setEnabled(has_polar)
        self.c_csv = QCheckBox("Also write the polar data as CSV next to the report")
        self.c_csv.setChecked(has_polar)
        self.c_csv.setEnabled(has_polar)
        for c in (self.c_shot, self.c_polar, self.c_csv):
            cl.addWidget(c)
        cl.addWidget(make_section_label("Notes (optional)"))
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText("Test description, configuration, assumptions…")
        self.notes.setFixedHeight(80)
        cl.addWidget(self.notes)
        root.addWidget(card)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        ok = buttons.addButton("Export", QDialogButtonBox.ButtonRole.AcceptRole)
        ok.setProperty("role", "primary")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save report", self.path.text(), "HTML report (*.html)")
        if path:
            self.path.setText(path if path.lower().endswith(".html") else path + ".html")

    def options(self) -> ReportOptions:
        """The chosen options."""
        p = Path(self.path.text().strip() or str(get_paths().exports / "report.html"))
        if p.suffix.lower() != ".html":
            p = p.with_suffix(".html")
        return ReportOptions(p, self.c_shot.isChecked(), self.c_polar.isChecked(), self.c_csv.isChecked(),
                             self.notes.toPlainText().strip())


HELP_QUICKSTART = """\
1. Pick a built-in shape and press Generate, or drag an STL / OBJ file onto the 3-D view.
2. If the nose does not face the wind (from the left), use "Align nose → wind".
3. Press Run simulation. Streamlines, surface pressure and CL / CD / L/D appear within a second or two.
4. Move the sliders: with "Live update" on, the flow re-solves when you release a slider.
5. Switch to Polar Analysis and run a sweep to see the lift curve, drag polar, best L/D and CM.
6. Open Visualization → 3-D surfaces to show vortex cores (wing-tip vortices), a speed shell or
   the wake; colour slices and streamlines by vorticity or total pressure.
7. Live Flow Lab (Ctrl+3): a real-time 2-D wind tunnel. Pick an airfoil, cylinder or a section of
   your model, drag the angle of attack, change the Reynolds number, or paint walls with the mouse
   and watch vortices, wakes and stall form live, with lift / drag and the shedding frequency.
8. Supersonic Lab (Ctrl+4, its own window): compressible flow above Mach 1 — shock waves, expansion
   fans, bow shocks and the sonic line, with Mach / pressure / temperature / schlieren / shadowgraph
   views and exact gas-dynamics theory compared side by side. Wait ~2 flow-throughs after a change.
9. Export a self-contained HTML report (Ctrl+E), a CSV of the polar or run history, a screenshot or a GIF.
"""

HELP_SHORTCUTS = """\
FILE & EDIT
  Ctrl+O        Import geometry          Ctrl+E      Export report
  Ctrl+S        Save 4K screenshot       Ctrl+Z      Undo geometry change
  Ctrl+Q        Quit

SIMULATION
  Ctrl+R        Run simulation           Esc         Abort running task

VIEW
  Ctrl+1        Wind tunnel              Ctrl+2      Polar analysis
  Ctrl+3        Live Flow Lab            Ctrl+4      Supersonic Lab (window)
  F9 / F10      Controls / Analysis panel
  F1            Help

3-D VIEW (mouse)
  Left drag     Rotate                   Wheel       Zoom
  Shift + drag  Pan                      Ctrl + drag Spin about the view axis
"""

HELP_ENGINES = """\
Instant Aero Engine (default, built in)
  A 3-D source-panel potential-flow solver coupled to a vortex-lattice lifting model,
  with Prandtl-Schlichting skin friction, a Hoerner form factor, induced drag and a
  separation / stall model. It reconstructs a full 3-D velocity and pressure grid in
  about a second. Treat it as an engineering preview: it is accurate for attached,
  subsonic flow and approximate once the flow is massively separated or transonic.

Live Flow Lab (real-time 2-D)
  A D2Q9 Lattice-Boltzmann solver with a Smagorinsky sub-grid model, velocity inlet, pressure
  outlet, absorbing sponge layers and momentum-exchange forces, compiled with numba. It is
  genuinely transient, so vortex shedding, separation and stall emerge from the physics.
  Validated: a cylinder at Re = 100 sheds at St ≈ 0.17-0.18 (12 % blockage).

Supersonic Lab (compressible 2-D)
  Euler equations, HLLC/HLL fluxes, MUSCL reconstruction, SSP-RK2, immersed-boundary walls on the
  true body outline. Validated: wedge oblique-shock pressure and angle within 1-2 %, blunt-body
  stagnation (Rayleigh pitot) within ~3 %, diamond wave drag within 10 % (shock-expansion theory).
  Inviscid: no boundary layers.

OpenFOAM (optional)
  Steady incompressible RANS (simpleFoam / foamRun, k-ω SST) on an automatically
  generated snappyHexMesh grid with boundary layers. Use it to confirm final numbers.
"""


class HelpDialog(QDialog):
    """Quick start, keyboard shortcuts, engines and about."""

    def __init__(self, parent: Optional[QWidget] = None, log_dir: Optional[Path] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} — Help")
        self.setMinimumSize(680, 520)
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        head = QHBoxLayout()
        logo = QLabel()
        logo.setPixmap(make_logo_pixmap(48))
        head.addWidget(logo)
        tl = QVBoxLayout()
        t = QLabel(APP_NAME)
        t.setProperty("role", "title")
        s = QLabel(f"Version {APP_VERSION} · 3-D CFD, polar analysis and a real-time 2-D flow lab")
        s.setProperty("role", "muted")
        tl.addWidget(t)
        tl.addWidget(s)
        head.addLayout(tl, 1)
        root.addLayout(head)
        tabs = QTabWidget()
        for name, text in (("Quick start", HELP_QUICKSTART), ("Shortcuts", HELP_SHORTCUTS), ("Engines", HELP_ENGINES)):
            box = QPlainTextEdit(text)
            box.setReadOnly(True)
            box.setFont(mono_font(12))
            tabs.addTab(box, name)
        about = QLabel(
            f"<p><b>{APP_NAME}</b> {APP_VERSION}</p>"
            "<p>Built with PySide6, PyVista / VTK, NumPy, SciPy, trimesh and pyqtgraph.</p>"
            "<p>OpenFOAM® is a registered trademark of OpenCFD Ltd. AeroFlow Studio is not affiliated "
            "with or endorsed by OpenCFD or the OpenFOAM Foundation.</p>"
            f"<p>Log files: <code>{(log_dir or get_paths().logs)}</code></p>")
        about.setWordWrap(True)
        about.setTextFormat(Qt.TextFormat.RichText)
        about.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        about.setAlignment(Qt.AlignmentFlag.AlignTop)
        tabs.addTab(about, "About")
        root.addWidget(tabs, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
