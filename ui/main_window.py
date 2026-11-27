"""The AeroFlow Studio main window: layout, mode switching and all orchestration.

Layout: a 56 px top command bar (logo, menu bar, mode switcher, engine pill, export, help), a left
control dock, a central stacked view (CFD viewport / polar charts), a right tabbed dock (charts,
geometry info, activity log) and a 28 px status bar. Window geometry and dock layout persist
between sessions (``QSettings``); every dock can be re-opened from the View menu.

Threading: geometry loading, repair, all solves, sweeps and OpenFOAM detection run in
:mod:`ui.workers`. This class only reacts to their queued results on the GUI thread, where it
updates widgets and VTK.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Optional

from PySide6.QtCore import QByteArray, QSettings, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QDesktopServices, QDragEnterEvent, QDropEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDockWidget, QFileDialog, QFrame, QHBoxLayout, QHeaderView, QLabel,
    QMainWindow,
    QMenuBar, QMessageBox, QProgressBar, QPushButton, QStackedWidget, QTableWidget, QTableWidgetItem, QTabWidget,
    QToolButton, QVBoxLayout, QWidget,
)

from config.settings import (
    APP_NAME, APP_VERSION, DEFAULT_WINDOW_SIZE, LEFT_DOCK_WIDTH, MAX_TRIANGLES_WARN, MESH_FILE_FILTER,
    MIN_WINDOW_SIZE, ORG_NAME, RIGHT_DOCK_WIDTH, STATUS_BAR_HEIGHT, SUPPORTED_MESH_EXTENSIONS, TOP_BAR_HEIGHT,
    AppMode, FlowConditions, GeometryPreset, GridQuality, ReferenceAreaMode, SolverEngine, get_paths,
)
from config.theme import PALETTE, make_icon, make_logo_pixmap, make_section_label, mono_font, set_property
from core import compute
from core.compute import Device
from core.geometry_manager import (
    GeometryError, GeometryManager, GeometryModel, LoadResult, align_nose, auto_orient, build_preset,
    center_model, convert_mm_to_m, deflect_flap, rotate_90, scale_to_length, scale_to_span, suggest_scale,
)
from core.instant_solver import InstantSolver
from core.openfoam_bridge import CaseSettings, OpenFOAMEnvironment
from core.polar_generator import default_export_name, export_html_report, export_polar_csv
from core.results import PolarData, SolverResult
from ui.cfd_panel import CFDPanel, SegmentedControl, scroll_wrap
from core.updates import ReleaseInfo, is_newer
from ui.device_selector import compute_relay
from ui.dialogs import ExportReportDialog, HelpDialog, OpenFOAMSetupDialog, open_issue_page
from ui.flowlab_panel import FlowLabPanel
from ui.run_history import RunHistoryWidget
from ui.toast import ToastManager
from ui.updates_ui import UpdateChecker, UpdateDialog
from ui.workers import (
    DetectOpenFOAMWorker, GridCheckWorker, InstantSolveWorker, LoadGeometryWorker, OpenFOAMPolarWorker, OpenFOAMWorker,
    PolarSweepWorker, RepairWorker, SweepRequest, TaskManager,
)
from utils.logger import ConsoleLogWidget, get_logger
from utils.sysinfo import system_report
from utils.validators import sanitize_filename, validate_sweep
from visualization.cfd_viewport import CFDViewport
from visualization.flowlab_view import FlowLabView
from visualization.telemetry_charts import (
    ChordCpChart, LiveForceChart, PolarCharts, ResidualChart, SpanLoadChart, export_plot_png,
)

log = get_logger("ui.main")

MODES: tuple[AppMode, ...] = (AppMode.CFD, AppMode.POLAR, AppMode.LIVE)

#: Worker types that "Abort" (Esc) may cancel. OpenFOAM *detection* is deliberately excluded, so an
#: abort never leaves the engine pill stuck on "detecting…".
_ABORTABLE = (InstantSolveWorker, PolarSweepWorker, OpenFOAMWorker, OpenFOAMPolarWorker, GridCheckWorker,
              LoadGeometryWorker, RepairWorker)

_SETTINGS_VERSION = 4         # bump when the dock layout changes so stale layouts are ignored


class MainWindow(QMainWindow):
    """Application main window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(*MIN_WINDOW_SIZE)
        self.resize(*DEFAULT_WINDOW_SIZE)
        self.setAcceptDrops(True)

        self.geometry_mgr = GeometryManager()
        self.solver = InstantSolver()
        self.tasks = TaskManager(self)
        self.of_env = OpenFOAMEnvironment(detail="Detecting OpenFOAM…")
        self.engine = SolverEngine.INSTANT
        self.mode = AppMode.CFD
        self.result: Optional[SolverResult] = None
        self.polar: Optional[PolarData] = None
        self._polar_model: Optional[GeometryModel] = None
        self._solve_worker: Optional[Any] = None
        self._sweep_worker: Optional[Any] = None
        self._resolve_pending = False
        self._setup_dialog: Optional[OpenFOAMSetupDialog] = None
        self._error_box_open = False
        self._recording = False
        self._closing = False
        self._lab_user_paused = False      # remembers a manual pause of the Flow Lab across mode switches
        self._pinned: list[tuple[str, PolarData]] = []     # reference polars shown dashed
        self._supersonic: Optional[Any] = None              # the Supersonic Lab window (created on demand)
        self._nozzle: Optional[Any] = None                  # the Rocket Nozzle Lab window (created on demand)
        self.toasts = ToastManager(self, bottom_margin=STATUS_BAR_HEIGHT + 14)
        self._restore_compute_device()
        self._update_checker = UpdateChecker(self)
        self._update_checker.finished.connect(self._on_update_result)
        self._update_manual = False
        self._latest_release: Optional[ReleaseInfo] = None

        self._build_top_bar()
        self._build_central()
        self._build_left_dock()
        self._build_right_dock()
        self._build_status_bar()
        self._build_menus()
        self._connect()
        self._restore_layout()
        self._set_mode(AppMode.CFD)
        QTimer.singleShot(120, self._startup)

    # =====================================================================
    # Layout
    # =====================================================================
    def _build_top_bar(self) -> None:
        bar = QFrame()
        bar.setObjectName("TopBar")
        bar.setFixedHeight(TOP_BAR_HEIGHT)
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 0, 12, 0)
        lay.setSpacing(10)
        logo = QLabel()
        logo.setPixmap(make_logo_pixmap(30))
        title = QLabel(APP_NAME)
        title.setObjectName("AppTitle")
        ver = QLabel(f"v{APP_VERSION}")
        ver.setProperty("role", "muted")
        lay.addWidget(logo)
        lay.addWidget(title)
        lay.addWidget(ver)
        self.menu_bar = QMenuBar()
        self.menu_bar.setNativeMenuBar(False)
        lay.addSpacing(8)
        lay.addWidget(self.menu_bar, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addStretch(1)
        self.mode_switch = SegmentedControl([m.value for m in MODES],
                                            ["3-D streamlines, pressure and forces (Ctrl+1)",
                                             "CL / CD / L/D across angles of attack (Ctrl+2)",
                                             "Real-time 2-D wind tunnel: watch vortices, wakes and stall live (Ctrl+3)"])
        for b in self.mode_switch.buttons:
            b.setMinimumWidth(140)
        lay.addWidget(self.mode_switch)
        lay.addStretch(1)
        self.btn_super = QPushButton("Supersonic Lab")
        self.btn_super.setIcon(make_icon("wind", 16))
        self.btn_super.setToolTip("Open the Supersonic Lab in its own window (Ctrl+4): shock waves above Mach 1")
        self.btn_super.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_super.clicked.connect(lambda: self.open_supersonic())
        lay.addWidget(self.btn_super)
        self.btn_nozzle = QPushButton("Nozzle Lab")
        self.btn_nozzle.setIcon(make_icon("play", 16))
        self.btn_nozzle.setToolTip("Open the Rocket Nozzle Lab (Ctrl+5): rocket engine nozzles, exhaust plumes, "
                                   "shock diamonds and thrust, live from ignition")
        self.btn_nozzle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_nozzle.clicked.connect(lambda: self.open_nozzle())
        lay.addWidget(self.btn_nozzle)
        self.engine_pill = QPushButton("Instant Aero Engine")
        self.engine_pill.setObjectName("EnginePill")
        self.engine_pill.setCursor(Qt.CursorShape.PointingHandCursor)
        self.engine_pill.setToolTip("Active solver engine — click to switch or to set up OpenFOAM")
        set_property(self.engine_pill, "state", "instant")
        lay.addWidget(self.engine_pill)
        self.btn_export = QToolButton()
        self.btn_export.setIcon(make_icon("export", 20))
        self.btn_export.setIconSize(QSize(20, 20))
        self.btn_export.setToolTip("Quick export: engineering report (Ctrl+E)")
        self.btn_help = QToolButton()
        self.btn_help.setIcon(make_icon("help", 20))
        self.btn_help.setIconSize(QSize(20, 20))
        self.btn_help.setToolTip("Help, shortcuts and about (F1)")
        lay.addWidget(self.btn_export)
        lay.addWidget(self.btn_help)
        self.setMenuWidget(bar)

    def _build_central(self) -> None:
        self.stack = QStackedWidget()
        self.cfd_view = CFDViewport()
        self.stack.addWidget(self.cfd_view)
        # Polar page
        polar_page = QWidget()
        polar_page.setObjectName("Root")
        pl = QVBoxLayout(polar_page)
        pl.setContentsMargins(16, 16, 16, 16)
        pl.setSpacing(12)
        head = QHBoxLayout()
        head.addWidget(make_section_label("Aerodynamic polar"))
        head.addStretch(1)
        self.polar_headline = QLabel("Run a polar sweep from the left panel (Polar sweep card).")
        self.polar_headline.setProperty("role", "muted")
        self.polar_headline.setFont(mono_font(12))
        self.btn_pin = QPushButton("Pin for comparison")
        self.btn_pin.setToolTip("Keep the current polar as a dashed reference curve (up to 4) to compare "
                                "other shapes or conditions against it")
        self.btn_unpin = QPushButton("Clear pinned")
        self.btn_unpin.setProperty("role", "ghost")
        self.btn_pin.clicked.connect(self._pin_polar)
        self.btn_unpin.clicked.connect(self._clear_pins)
        self.chk_theory = QCheckBox("Theory")
        self.chk_theory.setChecked(True)
        self.chk_theory.setToolTip("Dotted reference curves from classical theory for this wing's aspect ratio: "
                                   "Helmbold lifting-line lift slope and the ideal drag polar CD0 + CL²/(π·0.9·AR)")
        head.addWidget(self.chk_theory)
        head.addWidget(self.btn_pin)
        head.addWidget(self.btn_unpin)
        pl.addLayout(head)
        self.polar_headline.setWordWrap(True)
        pl.addWidget(self.polar_headline)              # own line: the summary never collides with buttons
        self.polar_big = PolarCharts(horizontal=True)
        self.chk_theory.toggled.connect(self.polar_big.set_show_theory)
        pl.addWidget(self.polar_big, 1)
        self.stack.addWidget(polar_page)
        # Live Flow Lab page
        self.lab_view = FlowLabView()
        self.stack.addWidget(self.lab_view)
        self.setCentralWidget(self.stack)

    def _build_left_dock(self) -> None:
        self.left_dock = QDockWidget("Controls", self)
        self.left_dock.setObjectName("LeftDock")
        self.left_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.left_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable |
                                   QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.left_dock.setTitleBarWidget(QWidget())
        self.cfd_panel = CFDPanel()
        self.lab_panel = FlowLabPanel(self.lab_view)
        self.left_scroll = scroll_wrap(self.cfd_panel)
        self.left_stack = QStackedWidget()
        self.left_stack.addWidget(self.left_scroll)
        self.left_stack.addWidget(scroll_wrap(self.lab_panel))
        self.left_stack.setMinimumWidth(LEFT_DOCK_WIDTH - 20)
        self.left_dock.setWidget(self.left_stack)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.left_dock)
        self.resizeDocks([self.left_dock], [LEFT_DOCK_WIDTH], Qt.Orientation.Horizontal)

    def _build_right_dock(self) -> None:
        self.right_dock = QDockWidget("Analysis", self)
        self.right_dock.setObjectName("RightDock")
        self.right_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable |
                                    QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.right_dock.setTitleBarWidget(QWidget())
        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setElideMode(Qt.TextElideMode.ElideNone)
        # Charts tab
        charts = QWidget()
        charts.setObjectName("Root")
        cl = QVBoxLayout(charts)
        cl.setContentsMargins(8, 8, 8, 8)
        cl.setSpacing(10)
        self.residual_chart = ResidualChart()
        self.cp_chart = ChordCpChart()
        self.polar_mini = PolarCharts(horizontal=False)
        self.live_chart = LiveForceChart()
        self.live_chart.setMinimumHeight(320)
        self.span_chart = SpanLoadChart()
        for w in (self.live_chart, self.residual_chart, self.cp_chart, self.span_chart, self.polar_mini):
            cl.addWidget(w)
        cl.addStretch(1)
        self.tabs.addTab(scroll_wrap(charts), make_icon("chart", 16), "Charts")
        # Geometry tab
        geo = QWidget()
        geo.setObjectName("Root")
        gl = QVBoxLayout(geo)
        gl.setContentsMargins(8, 8, 8, 8)
        self.geo_badge = QLabel("No geometry")
        self.geo_badge.setProperty("role", "badge")
        gl.addWidget(self.geo_badge, alignment=Qt.AlignmentFlag.AlignLeft)
        self.geo_table = QTableWidget(0, 2)
        self.geo_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.geo_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.geo_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.geo_table.verticalHeader().setVisible(False)
        self.geo_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.geo_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.geo_table.setFont(mono_font(12))
        gl.addWidget(self.geo_table, 1)
        self.geo_history = QLabel("")
        self.geo_history.setProperty("role", "muted")
        self.geo_history.setWordWrap(True)
        gl.addWidget(self.geo_history)
        self.tabs.addTab(geo, make_icon("info", 16), "Model")
        # Log tab
        logw = QWidget()
        logw.setObjectName("Root")
        ll = QVBoxLayout(logw)
        ll.setContentsMargins(8, 8, 8, 8)
        row = QHBoxLayout()
        self.log_level = QComboBox()
        for name, lvl in (("All", logging.DEBUG), ("Info", logging.INFO), ("Warnings", logging.WARNING),
                          ("Errors", logging.ERROR)):
            self.log_level.addItem(name, lvl)
        self.log_level.setCurrentIndex(1)
        clear = QPushButton("Clear")
        clear.setProperty("role", "ghost")
        save = QPushButton("Save…")
        save.setProperty("role", "ghost")
        row.addWidget(QLabel("Show"))
        row.addWidget(self.log_level)
        row.addStretch(1)
        row.addWidget(clear)
        row.addWidget(save)
        ll.addLayout(row)
        self.console = ConsoleLogWidget()
        self.console.setObjectName("Console")
        self.console.attach()
        self.console.set_min_level(logging.INFO)
        ll.addWidget(self.console, 1)
        self.log_level.currentIndexChanged.connect(lambda _i: self.console.set_min_level(self.log_level.currentData()))
        clear.clicked.connect(self.console.clear_log)
        save.clicked.connect(lambda: self.console.save_to_file())
        self.tabs.addTab(logw, make_icon("grid", 16), "Log")
        self.runs = RunHistoryWidget()
        self.tabs.insertTab(2, self.runs, make_icon("chart", 16), "Runs")
        self.tabs.setMinimumWidth(RIGHT_DOCK_WIDTH - 20)
        self.right_dock.setWidget(self.tabs)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.right_dock)
        self.resizeDocks([self.right_dock], [RIGHT_DOCK_WIDTH], Qt.Orientation.Horizontal)

    def _build_status_bar(self) -> None:
        sb = self.statusBar()
        sb.setFixedHeight(STATUS_BAR_HEIGHT)
        sb.setSizeGripEnabled(False)
        self.sb_mesh = QLabel("No geometry")
        self.sb_engine = QLabel("Instant Aero Engine")
        self.sb_engine.setProperty("role", "muted")
        self.sb_msg = QLabel("")
        self.sb_msg.setProperty("role", "muted")
        self.sb_progress = QProgressBar()
        self.sb_progress.setRange(0, 1000)
        self.sb_progress.setFixedWidth(220)
        self.sb_progress.setFixedHeight(12)
        self.sb_progress.setTextVisible(False)
        self.sb_progress.hide()
        self.sb_fps = QLabel("— FPS")
        self.sb_fps.setFont(mono_font(11))
        self.sb_fps.setProperty("role", "muted")
        sb.addWidget(self.sb_mesh)
        sb.addWidget(self._divider())
        sb.addWidget(self.sb_engine)
        sb.addWidget(self._divider())
        sb.addWidget(self.sb_msg, 1)
        self.sb_update = QPushButton("")
        self.sb_update.setProperty("role", "ghost")
        self.sb_update.setCursor(Qt.CursorShape.PointingHandCursor)
        self.sb_update.setToolTip("A newer version is available — click for details")
        self.sb_update.clicked.connect(self._show_update_dialog)
        self.sb_update.hide()
        sb.addPermanentWidget(self.sb_update)
        self.sb_device = QLabel("")
        self.sb_device.setFont(mono_font(11))
        self.sb_device.setProperty("role", "muted")
        self.sb_device.setToolTip("Where the Live Flow Lab and the Supersonic Lab run (Simulation ▸ Compute device)")
        sb.addPermanentWidget(self.sb_progress)
        sb.addPermanentWidget(self.sb_device)
        sb.addPermanentWidget(self._divider())
        sb.addPermanentWidget(self.sb_fps)

    @staticmethod
    def _divider() -> QFrame:
        d = QFrame()
        d.setFrameShape(QFrame.Shape.VLine)
        d.setStyleSheet(f"color: {PALETTE.border};")
        return d

    def _action(self, menu: Any, text: str, fn: Callable[[], Any], shortcut: str = "", tip: str = "") -> QAction:
        """Create a menu action that is also registered on the window, so its shortcut works everywhere."""
        act = QAction(text, self)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
            act.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        if tip:
            act.setStatusTip(tip)
            act.setToolTip(tip)
        act.triggered.connect(lambda _checked=False: fn())
        menu.addAction(act)
        self.addAction(act)
        return act

    def _build_menus(self) -> None:
        mb = self.menu_bar
        a = self._action
        m_file = mb.addMenu("&File")
        a(m_file, "Open project…", self.open_project_dialog, "Ctrl+Shift+O",
          "Reopen a saved session (.aeroflow): geometry, conditions, settings, camera and polar")
        a(m_file, "Save project", self.save_project, "Ctrl+Alt+S", "Save the whole session as one .aeroflow file")
        a(m_file, "Save project as…", lambda: self.save_project(ask=True))
        m_file.addSeparator()
        a(m_file, "Import geometry…", self.import_dialog, "Ctrl+O", "Load an STL / OBJ / PLY / OFF / VTK mesh")
        self.m_recent = m_file.addMenu("Open recent")
        self.m_recent.aboutToShow.connect(self._fill_recent_menu)
        m_file.addSeparator()
        a(m_file, "Export report…", self.export_report, "Ctrl+E", "Self-contained HTML engineering report")
        a(m_file, "Export polar CSV…", self.export_csv, "", "The last polar sweep as CSV")
        a(m_file, "Export run history CSV…", self._export_runs, "", "Every solve of this session as CSV")
        a(m_file, "Save screenshot…", self.screenshot_dialog, "Ctrl+S", "4K screenshot of the active view")
        a(m_file, "Save screenshot (transparent background)…", lambda: self.screenshot_dialog(transparent=True),
          "Ctrl+Shift+S", "4K PNG of the 3-D view without background or floor - for slides and papers")
        a(m_file, "Record animation (MP4 / GIF)…", self.record_gif, "", "Rotating flow animation")
        m_file.addSeparator()
        a(m_file, "Export flow field (ParaView / VTK)…", self.export_vtk, "",
          "Flow field, surface Cp and streamlines as a VTK multiblock (.vtm)")
        a(m_file, "Open result in ParaView", self.open_in_paraview, "", "Export and open the result in ParaView")
        m_file.addSeparator()
        a(m_file, "Open data folder", lambda: self._open_folder(get_paths().root), "", "Logs, cases and exports")
        m_file.addSeparator()
        a(m_file, "Quit", self.close, "Ctrl+Q")

        m_edit = mb.addMenu("&Edit")
        a(m_edit, "Undo geometry change", lambda: self._geometry_op("undo", None), "Ctrl+Z")
        a(m_edit, "Repair mesh", lambda: self._geometry_op("repair", None))
        a(m_edit, "Centre model", lambda: self._geometry_op("center", None))
        a(m_edit, "Auto-orient (nose → wind)", lambda: self._geometry_op("auto_orient", None))

        m_view = mb.addMenu("&View")
        a(m_view, "Wind tunnel", lambda: self._set_mode(AppMode.CFD), "Ctrl+1")
        a(m_view, "Polar analysis", lambda: self._set_mode(AppMode.POLAR), "Ctrl+2")
        a(m_view, "Live Flow Lab", lambda: self._set_mode(AppMode.LIVE), "Ctrl+3")
        a(m_view, "Supersonic Lab (separate window)…", self.open_supersonic, "Ctrl+4",
          "Detailed Mach > 1 simulation: shock waves, expansion fans, bow shocks and theory comparison")
        a(m_view, "Rocket Nozzle Lab (separate window)…", self.open_nozzle, "Ctrl+5",
          "Rocket engine nozzle and exhaust plume: ignition, shock diamonds, thrust and Isp vs 1-D theory")
        m_view.addSeparator()
        a(m_view, "Appearance (theme, interface size)…", self.open_appearance, "",
          "Light or dark theme and a larger or smaller interface")
        m_view.addSeparator()
        for key, label in (("iso", "Isometric view"), ("front", "Front view"), ("top", "Top view"),
                           ("side", "Side view"), ("reset", "Reset camera")):
            a(m_view, label, lambda k=key: self.cfd_view.view(k))
        m_view.addSeparator()
        a(m_view, "Spanwise Cp sections…", self.open_section_cp, "Ctrl+Shift+C",
          "Chordwise pressure distributions at several stations along the span")
        m_units = m_view.addMenu("Units")
        ugroup = QActionGroup(self)
        self._unit_actions: dict[str, QAction] = {}
        for name, label in (("SI", "SI (m/s, m, N)"), ("Imperial", "Imperial (kt, ft, lbf)")):
            act = QAction(label, self, checkable=True)
            act.triggered.connect(lambda _c=False, n=name: self._set_units(n))
            ugroup.addAction(act)
            m_units.addAction(act)
            self._unit_actions[name] = act
        m_bm = m_view.addMenu("Camera bookmarks")
        for i in range(1, 4):
            a(m_bm, f"Go to view {i}", lambda n=i: self._goto_bookmark(n), f"Alt+{i}")
        m_bm.addSeparator()
        for i in range(1, 4):
            a(m_bm, f"Save current view as {i}", lambda n=i: self._save_bookmark(n), f"Ctrl+Alt+{i}")
        m_view.addSeparator()
        left = self.left_dock.toggleViewAction()
        left.setText("Controls panel")
        left.setShortcut(QKeySequence("F9"))
        right = self.right_dock.toggleViewAction()
        right.setText("Analysis panel")
        right.setShortcut(QKeySequence("F10"))
        for act in (left, right):
            m_view.addAction(act)
            self.addAction(act)
        a(m_view, "Reset panel layout", self._reset_layout)

        m_sim = mb.addMenu("&Simulation")
        a(m_sim, "Run", self.run_simulation, "Ctrl+R", "Solve the current geometry")
        a(m_sim, "Abort", self.abort, "Esc", "Stop the running solve or sweep")
        a(m_sim, "Flight dynamics (trim, modes, response)…", self.open_flight_dynamics, "",
          "Trim the aircraft from its polar, find the short-period and phugoid modes, simulate gusts and inputs")
        a(m_sim, "Propeller analysis (BEM)…", self.open_propeller_tool, "",
          "Fast propeller thrust, torque, power and efficiency curves by blade-element momentum theory")
        a(m_sim, "Parametric study…", self.open_param_study, "Ctrl+Shift+P",
          "Sweep sideslip, speed (Reynolds / Mach), altitude or angle of attack and plot CL, CD, CM, L/D")
        a(m_sim, "Pin result for comparison", self.pin_result, "",
          "Remember the current result as 'A' to compare later results against")
        a(m_sim, "Compare with pinned result…", self.compare_results, "Ctrl+Shift+K",
          "Side-by-side 3-D views (linked cameras), coefficient differences and a ΔCp surface")
        a(m_sim, "Play transient result…", self.play_transient, "",
          "Step through the frames of a transient OpenFOAM run, with CL / CD against time")
        a(m_sim, "Grid convergence check…", self.grid_check, "",
          "Solve on the Coarse, Medium and Fine grids: Richardson extrapolation and the grid convergence index")
        m_sim.addSeparator()
        a(m_sim, "Use Instant Aero Engine", lambda: self._set_engine(SolverEngine.INSTANT))
        a(m_sim, "Use OpenFOAM", lambda: self._set_engine(SolverEngine.OPENFOAM))
        a(m_sim, "OpenFOAM setup guide…", self._show_setup_dialog)
        a(m_sim, "Re-detect OpenFOAM", self.detect_openfoam)
        m_sim.addSeparator()
        m_dev = m_sim.addMenu("Compute device (Live / Supersonic Labs)")
        group = QActionGroup(self)
        group.setExclusive(True)
        self._device_actions: dict[Device, QAction] = {}
        for dev in Device:
            act = QAction(dev.label, self, checkable=True)
            act.triggered.connect(lambda _c=False, d=dev: compute.set_preference(d))
            group.addAction(act)
            m_dev.addAction(act)
            self._device_actions[dev] = act

        m_help = mb.addMenu("&Help")
        a(m_help, "Help and shortcuts", self.show_help, "F1")
        a(m_help, "Guide (what everything means)…", lambda: __import__("ui.guide", fromlist=["show_guide"]).show_guide(self),
          "Shift+F1", "Explanations of each lab and of the physics on screen")
        a(m_help, "Open log folder", lambda: self._open_folder(get_paths().logs))
        m_help.addSeparator()
        a(m_help, "Check for updates…", lambda: self.check_for_updates(manual=True), "",
          "Look for a newer release on GitHub")
        self._act_auto_update = QAction("Check for updates automatically", self, checkable=True)
        self._act_auto_update.setChecked(self._auto_update_enabled())
        self._act_auto_update.toggled.connect(lambda on: self._settings().setValue("updates/auto", bool(on)))
        m_help.addAction(self._act_auto_update)
        a(m_help, "Report a problem…", lambda: open_issue_page(system_report(self._openfoam_summary())), "",
          "Open a pre-filled GitHub issue in your browser")
        m_help.addSeparator()
        a(m_help, f"About {APP_NAME}", lambda: self.show_help(tab="About"))

    def _connect(self) -> None:
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(30)
        self._preview_timer.timeout.connect(self._run_preview)
        self.mode_switch.changed.connect(lambda i: self._set_mode(MODES[i]))
        self.engine_pill.clicked.connect(self._on_engine_pill)
        self.btn_export.clicked.connect(self.export_report)
        self.btn_help.clicked.connect(self.show_help)
        p = self.cfd_panel
        p.preset_requested.connect(self._on_preset)
        p.import_requested.connect(self.import_dialog)
        p.geometry_op.connect(self._geometry_op)
        p.reference_mode_changed.connect(self._on_reference_mode)
        p.conditions_committed.connect(self._on_conditions_committed)
        p.conditions_changed.connect(self._on_conditions_live)
        p.run_requested.connect(self.run_simulation)
        p.abort_requested.connect(self.abort)
        p.engine_changed.connect(self._set_engine)
        p.visuals_changed.connect(self.cfd_view.update_settings)
        p.slice_sweep_requested.connect(self._on_slice_sweep)
        p.probe_mode_changed.connect(self.cfd_view.set_probe_mode)
        p.clear_seeds_requested.connect(self.cfd_view.clear_user_seeds)
        self.cfd_view.probe_line_ready.connect(self._show_line_probe)
        self.cfd_view.slice_position_changed.connect(p.show_slice_position)
        p.sweep_requested.connect(self.run_sweep)
        p.export_requested.connect(self._on_export)
        self.cfd_view.file_dropped.connect(self.load_file)
        self.cfd_view.fps_changed.connect(lambda f: self._show_fps(f, AppMode.CFD))
        self.cfd_view.status_message.connect(self._status)
        self.cfd_view.screenshot_requested.connect(self.screenshot_dialog)
        self.lab_view.fps_changed.connect(lambda f: self._show_fps(f, AppMode.LIVE))
        self.lab_view.stats_changed.connect(self.live_chart.update_stats)
        self.lab_view.status_message.connect(self._status)
        self.lab_panel.status_message.connect(self._status)
        self.runs.apply_requested.connect(self._apply_run_conditions)
        self.tasks.busy_changed.connect(self._on_busy)
        compute_relay().changed.connect(self._on_compute_changed)
        self._on_compute_changed()
        self._set_units(str(self._settings().value("ui/units", "SI")))

    # =====================================================================
    # Persistence
    # =====================================================================
    @staticmethod
    def _settings() -> QSettings:
        return QSettings(ORG_NAME, APP_NAME)

    def _restore_layout(self) -> None:
        s = self._settings()
        self._default_state = self.saveState(_SETTINGS_VERSION)
        try:
            geo = s.value("window/geometry")
            state = s.value("window/state")
            if isinstance(geo, QByteArray) and not geo.isEmpty():
                self.restoreGeometry(geo)
            if isinstance(state, QByteArray) and not state.isEmpty():
                self.restoreState(state, _SETTINGS_VERSION)
        except Exception as exc:  # noqa: BLE001 - a corrupt settings file must never block start-up
            log.debug("Ignoring saved window layout: %s", exc)
        # A dock that was closed last session would otherwise be invisible with no hint how to get it back.
        if not self.left_dock.isVisible() and not self.right_dock.isVisible():
            self._reset_layout()

    def _reset_layout(self) -> None:
        self.restoreState(self._default_state, _SETTINGS_VERSION)
        self.left_dock.show()
        self.right_dock.show()
        self.resizeDocks([self.left_dock, self.right_dock], [LEFT_DOCK_WIDTH, RIGHT_DOCK_WIDTH],
                         Qt.Orientation.Horizontal)

    def _save_layout(self) -> None:
        try:
            s = self._settings()
            s.setValue("window/geometry", self.saveGeometry())
            s.setValue("window/state", self.saveState(_SETTINGS_VERSION))
        except Exception as exc:  # noqa: BLE001
            log.debug("Could not save the window layout: %s", exc)

    def _restore_compute_device(self) -> None:
        try:
            saved = str(self._settings().value("compute/device", Device.AUTO.value))
            compute.set_preference(Device(saved))
        except (ValueError, TypeError):
            compute.set_preference(Device.AUTO)

    def _on_compute_changed(self) -> None:
        """Preference or GPU availability changed: sync the menu, the status bar and the saved setting."""
        pref = compute.preference()
        act = getattr(self, "_device_actions", {}).get(pref)
        if act is not None:
            act.setChecked(True)
        gpu_act = getattr(self, "_device_actions", {}).get(Device.GPU)
        if compute.probed():
            info = compute.gpu_info()
            if gpu_act is not None:
                gpu_act.setEnabled(info.available)
                gpu_act.setToolTip(info.summary())
            uses_gpu = compute.use_gpu()
            self.sb_device.setText(f"Labs: {'GPU' if uses_gpu else 'CPU'}")
            self.sb_device.setToolTip(f"Live Flow Lab / Supersonic Lab compute: {compute.describe()}")
        else:
            self.sb_device.setText("Labs: …")
        try:
            self._settings().setValue("compute/device", pref.value)
        except Exception as exc:  # noqa: BLE001
            log.debug("Could not save the compute device: %s", exc)

    @staticmethod
    def _open_folder(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    # =====================================================================
    # Startup & modes
    # =====================================================================
    def _startup(self) -> None:
        log.info("%s %s ready. Generating the default NACA 2412 wing…", APP_NAME, APP_VERSION)
        self._on_preset(GeometryPreset.NACA_2412, "2412", 6.0, 1.0, 0.0)
        self.detect_openfoam()
        compute.warmup_async()                 # probe CUDA + pre-compile the Lab GPU kernels in the background
        from core import accel

        if not accel.HAVE_NUMBA:
            log.warning("numba is not installed: the CPU solvers run 10-20x slower. Run setup_windows.bat "
                        "(or: pip install numba) to enable the compiled kernels.")
        QTimer.singleShot(4000, self._maybe_auto_check_updates)    # after start-up work has settled

    def _set_mode(self, mode: AppMode) -> None:
        if self._closing or self._recording:
            return
        previous = self.mode
        self.mode = mode
        self.mode_switch.set_index(MODES.index(mode))
        self.stack.setCurrentIndex(MODES.index(mode))
        live = mode == AppMode.LIVE
        self.left_stack.setCurrentIndex(1 if live else 0)
        self._update_chart_visibility()
        self.cfd_view.set_animation_paused(mode != AppMode.CFD)
        if live:
            if previous != AppMode.LIVE and not self._lab_user_paused:
                self.lab_view.start()
            self.lab_panel.sync_run_button()
            self._status("Live Flow Lab: change the angle, Reynolds number or body while the air flows.")
        else:
            if previous == AppMode.LIVE:
                self._lab_user_paused = not self.lab_view.running
            self.lab_view.stop()
        if mode == AppMode.POLAR:
            self.cfd_panel.focus_polar()
            if self.polar is None and self.geometry_mgr.has_model and not self._sweep_running():
                self._status("Tip: press 'Run polar sweep' to compute the polar.")
        self.sb_fps.setText("— FPS")

    def _update_chart_visibility(self) -> None:
        """Right-dock charts that make sense for the current mode (live forces only in the Flow Lab)."""
        live = self.mode == AppMode.LIVE
        self.live_chart.setVisible(live)
        self.cp_chart.setVisible(not live)
        self.span_chart.setVisible(not live)
        self.polar_mini.setVisible(not live and self.mode != AppMode.POLAR)
        self.residual_chart.setVisible(not live and (self.engine == SolverEngine.OPENFOAM or bool(
            self.result is not None and self.result.residuals)))

    # =====================================================================
    # Geometry
    # =====================================================================
    def _on_preset(self, preset: GeometryPreset, code: str, span: float, chord: float, tail: float) -> None:
        try:
            preset = GeometryPreset(preset)
            if preset is GeometryPreset.CUSTOM_NACA:
                model = build_preset(preset, code, span, chord)
            elif preset is GeometryPreset.SPHERE:
                model = build_preset(preset, tail_ratio=tail)
            else:
                model = build_preset(preset)
        except (GeometryError, ValueError) as exc:
            self._error("Could not build the geometry", str(exc))
            return
        log.info("Generated preset: %s (%d triangles)", model.name, model.triangle_count)
        self.geometry_mgr.set_model(model, keep_undo=True)
        self._model_changed(reset_camera=True)

    def import_dialog(self) -> None:
        """Open a file dialog and load a mesh."""
        path, _ = QFileDialog.getOpenFileName(self, "Import geometry", str(Path.home()), MESH_FILE_FILTER)
        if path:
            self.load_file(path)

    def load_file(self, path: str) -> None:
        """Load a mesh file in the background."""
        if Path(path).suffix.lower() not in SUPPORTED_MESH_EXTENSIONS:
            self._error("Unsupported file", f"{Path(path).name} is not one of: {', '.join(SUPPORTED_MESH_EXTENSIONS)}")
            return
        log.info("Importing %s …", path)
        if self.mode != AppMode.CFD:
            self._set_mode(AppMode.CFD)
        self.tasks.start(LoadGeometryWorker(path), on_result=self._on_loaded,
                         on_error=lambda m, tb: self._error("Import failed", m), on_progress=self._progress,
                         on_finished=self._progress_done)

    def _on_loaded(self, res: LoadResult) -> None:
        model = res.model
        if model.source is not None:
            self._add_recent(model.source)
        log.success("Loaded %s with %s: %d triangles", model.name, res.loader, model.triangle_count)
        for s in res.suggestions:
            log.warning(s)
        if model.triangle_count > MAX_TRIANGLES_WARN:
            log.warning("Large mesh (%d triangles): the Instant engine samples it, but the viewport may be slower.",
                        model.triangle_count)
        self.geometry_mgr.set_model(model, keep_undo=True)
        sugg = suggest_scale(model)
        if sugg is not None:
            ans = QMessageBox.question(self, "Convert units?", f"{sugg.reason}\n\nConvert millimetres to metres now?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ans == QMessageBox.StandardButton.Yes:
                self.geometry_mgr.apply(convert_mm_to_m)
                log.info("Converted mm → m.")
        model = self.geometry_mgr.require()
        degenerate = model.info.span_m < 1e-9 or model.info.chord_m < 1e-9
        self._model_changed(reset_camera=True, auto_solve=not degenerate)
        if degenerate:
            self._error("Degenerate geometry", "The model is flat in at least one direction and cannot be analysed.")
        elif not model.is_watertight:
            self._status("Mesh is not watertight — 'Repair mesh' is recommended before OpenFOAM.")

    def _geometry_op(self, op: str, arg: Any) -> None:
        if not self.geometry_mgr.has_model:
            self._status("Load or generate a geometry first.")
            return
        try:
            if op == "undo":
                if not self.geometry_mgr.can_undo:
                    self._status("Nothing to undo.")
                    return
                self.geometry_mgr.undo()
            elif op == "repair":
                if self.tasks.is_running(RepairWorker):
                    self._status("A repair is already running.")
                    return
                self.tasks.start(RepairWorker(self.geometry_mgr.require()), on_result=self._on_repaired,
                                 on_error=lambda m, tb: self._error("Repair failed", m), on_progress=self._progress,
                                 on_finished=self._progress_done)
                return
            elif op == "align_nose":
                self.geometry_mgr.apply(align_nose, str(arg))
            elif op == "rotate":
                axis, turns = arg
                self.geometry_mgr.apply(rotate_90, axis, turns)
            elif op == "flap":
                kind, delta, hinge, s_from, s_to = arg
                if s_to <= s_from:
                    self._status("The outboard end of the control surface must be beyond the inboard end.")
                    return
                if kind == "ailerons":
                    model = deflect_flap(self.geometry_mgr.require(), delta, hinge, s_from, s_to, "right")
                    self.geometry_mgr.apply(lambda _m: deflect_flap(model, -delta, hinge, s_from, s_to, "left"))
                else:
                    self.geometry_mgr.apply(deflect_flap, delta, hinge, s_from, s_to, "both")
            elif op == "center":
                self.geometry_mgr.apply(center_model)
            elif op == "auto_orient":
                self.geometry_mgr.apply(auto_orient)
            elif op == "mm_to_m":
                self.geometry_mgr.apply(convert_mm_to_m)
            elif op == "scale_span":
                self.geometry_mgr.apply(scale_to_span, float(arg))
            elif op == "scale_length":
                self.geometry_mgr.apply(scale_to_length, float(arg))
            else:
                log.warning("Unknown geometry operation: %s", op)
                return
        except (GeometryError, ValueError) as exc:
            self._error("Geometry operation failed", str(exc))
            return
        log.info("Geometry: %s", op.replace("_", " "))
        self._model_changed(reset_camera=op not in ("undo",))

    def _on_repaired(self, out: tuple[GeometryModel, Any]) -> None:
        model, report = out
        self.geometry_mgr.set_model(model, keep_undo=True)
        for m in getattr(report, "messages", []):
            log.info("Repair: %s", m)
        (log.success if report.watertight_after else log.warning)(
            "Repair finished — watertight: %s", "yes" if report.watertight_after else "no")
        self._model_changed(reset_camera=False)

    def _on_reference_mode(self, mode: ReferenceAreaMode) -> None:
        if self.geometry_mgr.has_model:
            self.geometry_mgr.set_reference_mode(mode)
            self._model_changed(reset_camera=False)

    def _model_changed(self, reset_camera: bool, auto_solve: bool = True) -> None:
        model = self.geometry_mgr.model
        self.result = None
        self.cfd_view.set_geometry(model, reset_camera=reset_camera)
        self.cp_chart.set_data(None)
        self.span_chart.set_data(None)
        self.residual_chart.reset("Run a solve to see the convergence history.")
        self._update_chart_visibility()
        self.lab_view.set_model(model)
        if self._supersonic is not None:
            self._supersonic.set_model(model)
        if self.polar is not None and self._polar_model is not model:
            self.polar = None
            self.polar_big.set_polar(None)
            self.polar_mini.set_polar(None)
            self.polar_headline.setText("Geometry changed — run the polar sweep again.")
            self.cfd_panel.set_polar_summary("No polar yet.")
        self._update_geometry_info(model)
        if model is not None:
            self.cfd_panel.set_reference_mode(model.reference_mode)
            self.setWindowTitle(f"{model.name} — {APP_NAME}")
            if auto_solve and self.engine == SolverEngine.INSTANT:
                self.run_simulation()

    def _update_geometry_info(self, model: Optional[GeometryModel]) -> None:
        self.geo_table.setRowCount(0)
        if model is None:
            self.geo_badge.setText("No geometry")
            self.sb_mesh.setText("No geometry")
            return
        info = model.info
        rows = info.rows()
        self.geo_table.setRowCount(len(rows))
        for r, (k, v) in enumerate(rows):
            self.geo_table.setItem(r, 0, QTableWidgetItem(k))
            self.geo_table.setItem(r, 1, QTableWidgetItem(v))
        ok = model.is_watertight
        self.geo_badge.setText("Watertight" if ok else "Not watertight")
        set_property(self.geo_badge, "level", "success" if ok else "warning")
        self.geo_history.setText("History: " + " → ".join(model.history[-6:]) if model.history else "")
        self.sb_mesh.setText(f"{model.name} · {model.triangle_count:,} tris")

    # =====================================================================
    # Solving
    # =====================================================================
    def _solve_running(self) -> bool:
        return self._solve_worker is not None

    def _sweep_running(self) -> bool:
        return self._sweep_worker is not None

    def _on_conditions_live(self, cond: FlowConditions) -> None:
        if self.polar is not None:
            self.polar_mini.set_current(cond.alpha_deg, self.polar.cl_at(cond.alpha_deg), self.polar.cd_at(cond.alpha_deg))
        self._preview_timer.start()           # coalesce slider ticks into one preview per ~30 ms

    def _run_preview(self) -> None:
        """Live feedback while a flow slider is dragged: KPIs and surface Cp from the cached geometry
        preparation (~5-20 ms on the GUI thread). The volume field updates when the slider is released."""
        model = self.geometry_mgr.model
        if (model is None or self.result is None or self.mode != AppMode.CFD
                or self.engine != SolverEngine.INSTANT or self._closing):
            return
        prep = self.solver.cached_preparation(model, self.cfd_panel.quality_value())
        if prep is None:
            return
        try:
            kpis, cp = self.solver.surface_preview(prep, self.cfd_panel.conditions())
        except Exception as exc:  # noqa: BLE001 - a preview must never interrupt dragging
            log.debug("Preview skipped: %s", exc)
            return
        self.cfd_view.preview(kpis, cp)

    def _on_conditions_committed(self, _cond: FlowConditions) -> None:
        if self.cfd_panel.live_update() and self.geometry_mgr.has_model and self.engine == SolverEngine.INSTANT:
            self.run_simulation(live=True)

    def run_simulation(self, live: bool = False) -> None:
        """Solve the current geometry with the selected engine.

        ``live`` marks a re-solve triggered by a slider: errors from it are reported in the status bar
        and the log instead of a modal dialog, and it never starts OpenFOAM.
        """
        model = self.geometry_mgr.model
        if model is None:
            self._status("Load or generate a geometry first.")
            return
        if self._solve_running():
            if self.engine == SolverEngine.INSTANT:
                self._resolve_pending = True           # re-solve with the newest inputs when this one ends
            else:
                self._status("OpenFOAM is running — abort it (Esc) before starting another run.")
            return
        if self._sweep_running():
            self._status("A polar sweep is running — please wait or abort it.")
            return
        cond = self.cfd_panel.conditions()
        quality = self.cfd_panel.quality_value()
        if self.engine == SolverEngine.OPENFOAM:
            if live:
                return
            self._run_openfoam(model, cond, quality)
            return
        self._resolve_pending = False
        worker = InstantSolveWorker(self.solver, model, cond, quality)
        self._solve_worker = worker
        self.cfd_panel.set_busy(True)
        self.tasks.start(worker, on_result=lambda r, m=model: self._on_result(r, m),
                         on_error=lambda m, tb: self._error("Solve failed", m, modal=not live),
                         on_progress=self._progress, on_cancelled=lambda: log.warning("Solve cancelled."),
                         on_finished=self._on_solve_finished)

    def _run_openfoam(self, model: GeometryModel, cond: FlowConditions, quality: GridQuality) -> None:
        if not self.of_env.available:
            self._show_setup_dialog()
            return
        if not model.is_watertight:
            ans = QMessageBox.question(
                self, "Geometry not watertight",
                "snappyHexMesh needs a closed surface. Repair the mesh first?\n\n"
                "Yes = repair now · No = try anyway",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            if ans == QMessageBox.StandardButton.Yes:
                self._geometry_op("repair", None)
                return
            if ans == QMessageBox.StandardButton.Cancel:
                return
        settings = CaseSettings(cond, quality, iterations=self.cfd_panel.openfoam_iterations(),
                                n_procs=self.cfd_panel.openfoam_procs(),
                                boundary_layers=self.cfd_panel.openfoam_layers(),
                                compressible=self.cfd_panel.openfoam_compressible(),
                                mrf_rpm=self.cfd_panel.openfoam_rpm(), esi=self.of_env.is_esi,
                                turbulence=self.cfd_panel.openfoam_turbulence(),
                                surface_refinement=self.cfd_panel.openfoam_refinement()[0],
                                wake_refinement=self.cfd_panel.openfoam_refinement()[1])
        transient, ctimes, frames = self.cfd_panel.openfoam_transient()
        if transient:
            if settings.compressible:
                self._status("Transient runs are incompressible only - running a steady compressible solve.")
            else:
                settings = replace(settings, transient=True, convective_times=ctimes, frames=frames)
        worker = OpenFOAMWorker(self.of_env, model, settings)
        worker.signals.residuals.connect(self._on_residuals)
        worker.signals.coefficients.connect(self._on_live_coefficients)
        worker.signals.message.connect(self._on_worker_message)
        self._solve_worker = worker
        self.residual_chart.reset()
        self._update_chart_visibility()
        self.tabs.setCurrentIndex(0)
        self.cfd_panel.set_busy(True)
        log.info("Starting OpenFOAM (%s, %s, %d iterations)…", self.of_env.mode.value, quality.value,
                 settings.iterations)
        self.tasks.start(worker, on_result=lambda r, m=model: self._on_result(r, m),
                         on_error=lambda m, tb: self._error("OpenFOAM failed", m),
                         on_progress=self._progress, on_cancelled=lambda: log.warning("OpenFOAM run aborted."),
                         on_finished=self._on_solve_finished)

    def _on_worker_message(self, text: str) -> None:
        log.info(text)

    def _on_residuals(self, iteration: int, values: dict[str, float]) -> None:
        self.residual_chart.add_residuals(iteration, values)

    def _on_live_coefficients(self, iteration: int, coeffs: dict[str, float]) -> None:
        self.residual_chart.add_coefficients(iteration, coeffs)
        self._status(f"Iteration {iteration}: CL {coeffs.get('Cl', math.nan):+.4f} · CD {coeffs.get('Cd', math.nan):.5f}")

    def _on_solve_finished(self) -> None:
        self._solve_worker = None
        self.cfd_panel.set_busy(self._sweep_running())
        self._progress_done()
        # Only the Instant engine queues re-solves; never let a queued live update start OpenFOAM.
        if self._resolve_pending and not self._closing and self.engine == SolverEngine.INSTANT:
            self._resolve_pending = False
            QTimer.singleShot(0, lambda: self.run_simulation(live=True))
        else:
            self._resolve_pending = False

    def _on_result(self, result: SolverResult, model: GeometryModel) -> None:
        if model is not self.geometry_mgr.model:
            log.debug("Discarding a result for a previous geometry.")
            return
        self.result = result
        k = result.kpis
        self.cfd_view.set_result(result)
        prop = result.extra.get("propeller") if result.extra else None
        if prop:
            QTimer.singleShot(300, lambda p=prop: self._show_propeller(p))
        if result.extra and result.extra.get("transient"):
            QTimer.singleShot(300, self.play_transient)       # step through the saved frames
        self.cp_chart.set_data(result.chord_cp)
        self.span_chart.set_data(result.extra)
        self.runs.add_result(result, model.name)
        self.residual_chart.set_history(result.residuals)
        # The convergence chart only means something for an iterative (OpenFOAM) result.
        self._update_chart_visibility()
        if self.polar is not None:
            self.polar_mini.set_current(k.alpha_deg, k.cl, k.cd)
        for w in result.warnings:
            log.warning(w)
        log.success("%s: CL %+.4f · CD %.5f · CM %+.4f · L/D %.1f · Re %.2e (%.2f s)", result.engine, k.cl, k.cd,
                    k.cm, k.l_over_d, k.reynolds, result.elapsed_s)
        self._status(f"Solved at α {k.alpha_deg:+.1f}° in {result.elapsed_s:.2f} s — CL {k.cl:+.3f} · "
                     f"CD {k.cd:.4f} · L/D {k.l_over_d:.1f}")
        if k.stalled:
            log.warning("The wing is stalled at α = %.1f° (stall ≈ %.1f°).", k.alpha_deg, k.stall_alpha_deg)

    def run_sweep(self, start: float, stop: float, step: float) -> None:
        """Run a polar sweep with the Instant engine."""
        model = self.geometry_mgr.model
        if model is None:
            self._status("Load or generate a geometry first.")
            return
        if self._sweep_running() or (self._solve_running() and self.engine == SolverEngine.OPENFOAM):
            self._status("Another simulation is running.")
            return
        angles, check = validate_sweep(start, stop, step)
        if not angles or not check.ok:
            self._error("Invalid polar sweep", check.summary() or "The sweep contains no angles of attack.")
            return
        if self.engine == SolverEngine.OPENFOAM:
            if not self.of_env.available:
                self._show_setup_dialog()
                return
            ans = QMessageBox.question(
                self, "OpenFOAM polar sweep",
                f"Run {len(angles)} OpenFOAM cases ({start:g}° to {stop:g}°)?\n\nThe mesh is built once and reused, "
                "so each angle costs one solver run (typically 1-5 minutes each on a Coarse / Medium grid). "
                "Progress and the coefficients of each angle appear in the status bar and the Log tab.")
            if ans != QMessageBox.StandardButton.Yes:
                return
            settings = CaseSettings(self.cfd_panel.conditions(), self.cfd_panel.quality_value(),
                                    iterations=self.cfd_panel.openfoam_iterations(),
                                    n_procs=self.cfd_panel.openfoam_procs(),
                                    boundary_layers=self.cfd_panel.openfoam_layers(),
                                    compressible=self.cfd_panel.openfoam_compressible(),
                                    mrf_rpm=self.cfd_panel.openfoam_rpm(), esi=self.of_env.is_esi,
                                    turbulence=self.cfd_panel.openfoam_turbulence(),
                                    surface_refinement=self.cfd_panel.openfoam_refinement()[0],
                                    wake_refinement=self.cfd_panel.openfoam_refinement()[1])
            worker = OpenFOAMPolarWorker(self.of_env, model, settings, angles)
            worker.signals.message.connect(self._on_worker_message)
        else:
            req = SweepRequest(self.cfd_panel.conditions(), start, stop, step, self.cfd_panel.quality_value())
            worker = PolarSweepWorker(self.solver, model, req)
        self._sweep_worker = worker
        self.cfd_panel.set_busy(True, sweeping=True)
        log.info("Polar sweep %.1f° → %.1f° (step %.2f°, %d points)…", start, stop, step, len(angles))

        def done() -> None:
            self._sweep_worker = None
            self.cfd_panel.set_busy(self._solve_running())
            self._progress_done()

        self.tasks.start(worker, on_result=lambda p: self._on_polar(p, model),
                         on_error=lambda m, tb: self._error("Polar sweep failed", m), on_progress=self._progress,
                         on_cancelled=lambda: self._status("Polar sweep cancelled."), on_finished=done)

    def _apply_run_conditions(self, cond: FlowConditions) -> None:
        """Restore a logged run's flow conditions (and re-solve)."""
        if self.mode != AppMode.CFD:
            self._set_mode(AppMode.CFD)
        self.cfd_panel.set_conditions(cond, commit=False)
        self.run_simulation()
        self.notify(f"Restored α {cond.alpha_deg:+.1f}°, V {cond.velocity_ms:.1f} m/s — solving…")

    def open_supersonic(self) -> None:
        """Show the Supersonic Lab window (created on first use, then kept with its state)."""
        if self._supersonic is None:
            from ui.supersonic_window import SupersonicWindow

            self._supersonic = SupersonicWindow(self)
            self._supersonic.set_model(self.geometry_mgr.model)
        win = self._supersonic
        win.show()
        win.raise_()
        win.activateWindow()
        if not win.view.running:
            win.toggle_run()

    def open_appearance(self) -> None:
        from ui.appearance import AppearanceDialog, restart_application

        dlg = AppearanceDialog(self)
        if dlg.exec() and dlg.restart_requested:
            restart_application(self)
        elif dlg.result():
            self.notify("Appearance saved: it applies the next time the program starts.")

    def open_nozzle(self) -> None:
        """Show the Rocket Nozzle Lab window (created on first use, then kept with its state)."""
        if self._nozzle is None:
            from ui.nozzle_window import NozzleWindow

            self._nozzle = NozzleWindow(self)
        win = self._nozzle
        win.show()
        win.raise_()
        win.activateWindow()

    def _export_runs(self) -> None:
        if not self.runs.records:
            self.notify("No runs yet — solve something first.", "warning")
            return
        try:
            out = self.runs.export_dialog()
        except OSError as exc:
            self._error("Export failed", str(exc))
            return
        if out is not None:
            self.notify(f"Run history saved: {out.name}", "success")

    def _pin_polar(self) -> None:
        if self.polar is None:
            self.notify("Run a polar sweep first, then pin it.", "warning")
            return
        label = f"{self.polar.geometry_name} · Re {self.polar.reynolds:.1e}"
        if len(self._pinned) >= len(PolarCharts.REF_COLORS):
            self._pinned.pop(0)
        self._pinned.append((label, self.polar))
        self.polar_big.set_references(self._pinned)
        self.notify(f"Pinned: {label}. Change the shape or conditions and sweep again to compare.", "success")

    def _clear_pins(self) -> None:
        self._pinned.clear()
        self.polar_big.set_references([])

    def _on_polar(self, polar: PolarData, model: GeometryModel) -> None:
        if model is not self.geometry_mgr.model:
            return
        self.polar = polar
        self._polar_model = model
        self.polar_big.set_polar(polar)
        self.polar_mini.set_polar(polar)
        za = polar.alpha_zero_lift_deg
        summary = (f"CL,max {polar.cl_max:.3f} @ {polar.alpha_stall_deg:.1f}° · L/D max {polar.ld_max:.1f} @ "
                   f"{polar.alpha_best_ld_deg:.1f}° · α0 {'n/a' if math.isnan(za) else f'{za:.2f}°'} · "
                   f"Re {polar.reynolds:.2e}")
        stab = polar.stability()
        self.polar_headline.setText(summary + ("" if stab is None else "\n" + stab.summary()))
        self.cfd_panel.set_polar_summary(summary.replace(" · ", "\n")
                                         + ("" if stab is None else "\n" + stab.summary().replace(" · ", "\n")))
        if stab is not None:
            log.info("Stability: %s", stab.summary())
        log.success("Polar complete: %s", summary)
        self.notify(f"Polar complete — CL,max {polar.cl_max:.2f}, best L/D {polar.ld_max:.1f}", "success")
        self._status(f"Polar complete ({len(polar.alpha_deg)} points).")
        cond = self.cfd_panel.conditions()
        self.polar_mini.set_current(cond.alpha_deg, polar.cl_at(cond.alpha_deg), polar.cd_at(cond.alpha_deg))

    def abort(self) -> None:
        """Abort the running solve / sweep / import (and kill OpenFOAM if running)."""
        n = self.tasks.cancel(_ABORTABLE)
        self._resolve_pending = False
        if n:
            log.warning("Abort requested (%d task%s).", n, "" if n == 1 else "s")
            self._status("Aborting…")

    # =====================================================================
    # Engines / OpenFOAM
    # =====================================================================
    def detect_openfoam(self) -> None:
        """Probe for OpenFOAM in the background."""
        if self.tasks.is_running(DetectOpenFOAMWorker):
            return
        self._update_engine_pill(detecting=True)
        self.tasks.start(DetectOpenFOAMWorker(), on_result=self._on_detected,
                         on_error=lambda m, tb: self._on_detect_failed(m),
                         on_cancelled=lambda: self._update_engine_pill())

    def _on_detect_failed(self, message: str) -> None:
        log.warning("OpenFOAM detection failed: %s", message)
        self._update_engine_pill()

    def _on_detected(self, env: OpenFOAMEnvironment) -> None:
        self.of_env = env
        self.cfd_panel.set_openfoam_available(env.available, env.detail)
        if self._setup_dialog is not None:
            self._setup_dialog.set_environment(env)
        if env.available:
            log.success("%s", env.badge_text)
        else:
            log.info(env.badge_text)
        if not env.available and self.engine == SolverEngine.OPENFOAM:
            self._set_engine(SolverEngine.INSTANT)
        self._update_engine_pill()

    def _set_engine(self, engine: SolverEngine) -> None:
        if engine == SolverEngine.OPENFOAM and not self.of_env.available:
            self.cfd_panel.set_engine(SolverEngine.INSTANT)
            self._show_setup_dialog()
            return
        if engine != self.engine:
            self._resolve_pending = False
        self.engine = engine
        self._update_chart_visibility()
        self.cfd_panel.set_engine(engine)
        self._update_engine_pill()
        log.info("Solver engine: %s", engine.value)

    def _update_engine_pill(self, detecting: bool = False) -> None:
        if self.engine == SolverEngine.OPENFOAM:
            text, state = self.of_env.badge_text, "openfoam"
        elif detecting:
            text, state = "Instant Aero Engine · detecting OpenFOAM…", "instant"
        elif self.of_env.available:
            text, state = f"Instant Aero Engine · {self.of_env.badge_text} available", "instant"
        else:
            # Not having OpenFOAM is a normal state, so it is shown neutrally rather than as an alarm.
            text, state = "Instant Aero Engine · OpenFOAM not installed", "instant"
        self.engine_pill.setText(text)
        set_property(self.engine_pill, "state", state)
        self.sb_engine.setText(self.engine.value)

    def _on_engine_pill(self) -> None:
        if not self.of_env.available:
            self._show_setup_dialog()
            return
        self._set_engine(SolverEngine.INSTANT if self.engine == SolverEngine.OPENFOAM else SolverEngine.OPENFOAM)

    def _show_setup_dialog(self) -> None:
        if self._setup_dialog is not None:
            self._setup_dialog.raise_()
            return
        dlg = OpenFOAMSetupDialog(self.of_env, redetect=self.detect_openfoam, parent=self)
        self._setup_dialog = dlg
        try:
            dlg.exec()
        finally:
            self._setup_dialog = None

    # =====================================================================
    # Export
    # =====================================================================
    def _on_export(self, kind: str) -> None:
        {"screenshot": self.screenshot_dialog, "gif": self.record_gif, "csv": self.export_csv,
         "report": self.export_report}.get(kind, lambda: None)()

    def _default_name(self, suffix: str) -> str:
        model = self.geometry_mgr.model
        base = sanitize_filename(model.name if model else "aeroflow")
        return f"{base}_{time.strftime('%Y%m%d_%H%M%S')}{suffix}"

    def screenshot_dialog(self, transparent: bool = False) -> None:
        """Save a screenshot of the active view (4K for the 3-D view, all three charts for the polar).
        ``transparent``: 3-D view without background or floor, for slides and papers."""
        if self._recording:
            return
        if transparent and self.mode != AppMode.CFD:
            self._set_mode(AppMode.CFD)
        path, _ = QFileDialog.getSaveFileName(self, "Save screenshot", str(get_paths().exports / self._default_name(".png")),
                                              "PNG image (*.png)")
        if not path:
            return
        try:
            if self.mode == AppMode.POLAR:
                self.polar_big.save_png(path)
            elif self.mode == AppMode.LIVE:
                self.lab_view.save_png(path)
            else:
                self.cfd_view.save_screenshot(path, transparent=transparent)
            log.success("Screenshot saved: %s", path)
            self.notify(f"Screenshot saved: {Path(path).name}", "success")
        except Exception as exc:  # noqa: BLE001
            self._error("Screenshot failed", str(exc))

    def record_gif(self) -> None:
        """Record a rotating flow animation."""
        if self._recording:
            return
        if self.mode != AppMode.CFD:
            self._set_mode(AppMode.CFD)
        path, chosen = QFileDialog.getSaveFileName(
            self, "Save animation", str(get_paths().exports / self._default_name(".mp4")),
            "MP4 video (*.mp4);;Animated GIF (*.gif)")
        if not path:
            return
        if Path(path).suffix.lower() not in (".mp4", ".gif"):
            path += ".gif" if "gif" in chosen.lower() else ".mp4"
        video = Path(path).suffix.lower() == ".mp4"
        # The recorder pumps the event loop between frames, so block re-entrant user actions meanwhile.
        self._recording = True
        self.left_dock.setEnabled(False)
        self.menu_bar.setEnabled(False)
        self.mode_switch.setEnabled(False)
        self.sb_progress.show()
        try:
            # MP4 is cheap per frame: twice the frames at 30 FPS for a smooth 5-second turn.
            self.cfd_view.record_gif(path, frames=150 if video else 72, fps=30 if video else 24,
                                     progress=lambda pct: self.sb_progress.setValue(int(pct * 10)))
            log.success("Animation saved: %s", path)
            self.notify(f"Animation saved: {Path(path).name}", "success")
        except Exception as exc:  # noqa: BLE001
            self._error("Recording failed", str(exc))
        finally:
            self._recording = False
            self.left_dock.setEnabled(True)
            self.menu_bar.setEnabled(True)
            self.mode_switch.setEnabled(True)
            self._progress_done()

    # =====================================================================
    # Projects
    # =====================================================================
    _project_path: Optional[Path] = None

    def save_project(self, ask: bool = False) -> None:
        from core.project import SUFFIX, ProjectState, save_project

        model = self.geometry_mgr.model
        if model is None:
            self._status("Nothing to save yet - load or generate a geometry first.")
            return
        path = self._project_path
        if ask or path is None:
            default = path or (get_paths().root / "projects" / f"{sanitize_filename(model.name)}{SUFFIX}")
            chosen, _ = QFileDialog.getSaveFileName(self, "Save project", str(default),
                                                    f"AeroFlow project (*{SUFFIX})")
            if not chosen:
                return
            path = Path(chosen)
        case = str(self.result.extra.get("case_dir", "")) if self.result is not None and self.result.extra else ""
        state = ProjectState(conditions=self.cfd_panel.conditions(), quality=self.cfd_panel.quality_value().value,
                             engine="openfoam" if self.engine == SolverEngine.OPENFOAM else "instant",
                             reference_mode=model.reference_mode.value, camera=self.cfd_view.camera_state(),
                             polar=self.polar if self._polar_model is model else None, openfoam_case=case)
        try:
            out = save_project(path, model, state)
        except OSError as exc:
            self._error("Saving the project failed", str(exc))
            return
        self._project_path = out
        self._add_recent(out)
        self.setWindowTitle(f"{out.stem} — {APP_NAME}")
        log.success("Project saved: %s", out)
        self.notify(f"Project saved: {out.name}", "success")

    def open_project_dialog(self) -> None:
        from core.project import SUFFIX

        start = str(self._project_path.parent if self._project_path else get_paths().root / "projects")
        path, _ = QFileDialog.getOpenFileName(self, "Open project", start, f"AeroFlow project (*{SUFFIX})")
        if path:
            self.open_project(path)

    def open_project(self, path: str | Path) -> None:
        from core.project import ProjectError, load_project

        try:
            model, state = load_project(path, get_paths().cache)
        except (ProjectError, GeometryError) as exc:
            self._error("Could not open the project", str(exc))
            return
        try:
            model = model.with_reference_mode(ReferenceAreaMode(state.reference_mode))
        except ValueError:
            pass
        if self.mode != AppMode.CFD:
            self._set_mode(AppMode.CFD)
        self.cfd_panel.set_conditions(state.conditions, commit=False)
        idx = self.cfd_panel.quality.findText(state.quality)
        if idx >= 0:
            self.cfd_panel.quality.setCurrentIndex(idx)
        engine = SolverEngine.OPENFOAM if state.engine == "openfoam" else SolverEngine.INSTANT
        self.cfd_panel.set_engine(engine if engine != SolverEngine.OPENFOAM or self.of_env.available
                                  else SolverEngine.INSTANT)
        self._set_engine(self.cfd_panel.selected_engine())
        self.geometry_mgr.set_model(model, keep_undo=False)
        self._model_changed(reset_camera=state.camera is None)
        if state.polar is not None:
            self._on_polar(state.polar, model)
        if state.camera:
            QTimer.singleShot(250, lambda c=state.camera: self.cfd_view.set_camera_state(c))
        self._project_path = Path(path)
        self._add_recent(path)
        self.setWindowTitle(f"{Path(path).stem} — {APP_NAME}")
        if engine == SolverEngine.OPENFOAM and state.openfoam_case:
            self._status(f"Project opened. Its OpenFOAM result is in {state.openfoam_case} - run again to view it.")
        else:
            self._status("Project opened.")
        log.success("Project opened: %s", path)

    def _show_line_probe(self, data: dict) -> None:
        from ui.probe_dialog import LineProbeDialog

        dlg = getattr(self, "_line_probe_dialog", None)
        if dlg is None:
            dlg = LineProbeDialog(self)
            self._line_probe_dialog = dlg
        dlg.set_data(data)
        dlg.show()
        dlg.raise_()

    def _set_units(self, name: str) -> None:
        from config import units

        units.set_system(name)
        self._settings().setValue("ui/units", name)
        for n, act in getattr(self, "_unit_actions", {}).items():
            act.setChecked(n == name)
        self.cfd_panel.apply_units()
        if self.result is not None:
            self.cfd_view.hud.update_kpis(self.result.kpis, self.result.engine)

    def _show_propeller(self, p: dict) -> None:
        from config import units

        lines = [
            f"Thrust {units.fmt(p['thrust_n'], 'force', 1)}  ·  torque {p['torque_nm']:.2f} N·m  ·  "
            f"power {p['power_w'] / 1000.0:.2f} kW",
            "",
            f"Advance ratio J = V/(nD) = {p['J']:.3f}",
            f"Thrust coefficient CT = {p['CT']:.4f}",
            f"Torque coefficient CQ = {p['CQ']:.5f}",
            f"Power coefficient CP = {p['CP']:.4f}",
            f"Efficiency η = J·CT/CP = {p['eta']:.3f}",
            "",
            f"{p['rpm']:.0f} rpm, diameter {p['diameter_m']:.3f} m (MRF steady rotating frame).",
        ]
        text = "\n".join(lines)
        log.success("Propeller: %s", " · ".join(ln for ln in lines if ln))
        QMessageBox.information(self, "Propeller performance", text)

    def play_transient(self) -> None:
        r = self.result
        if r is None or not r.extra or not r.extra.get("transient") or "of_layout" not in r.extra:
            self._status("Run a transient OpenFOAM simulation first (OpenFOAM ▸ Run type: Transient).")
            return
        from ui.transient_player import TransientPlayer

        if self.mode != AppMode.CFD:
            self._set_mode(AppMode.CFD)
        dlg = TransientPlayer(r, self.cfd_view.set_result, self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.show()

    _pinned_result: Optional[tuple[GeometryModel, SolverResult]] = None

    def pin_result(self) -> None:
        model = self.geometry_mgr.model
        if self.result is None or model is None:
            self._status("Run a simulation first - there is no result to pin.")
            return
        self._pinned_result = (model, self.result)
        self.notify("Result pinned as A - change something, solve, then Simulation ▸ Compare.", "info")

    def compare_results(self) -> None:
        model = self.geometry_mgr.model
        if self._pinned_result is None:
            self._status("Pin a result first (Simulation ▸ Pin result for comparison).")
            return
        if self.result is None or model is None:
            self._status("Run a simulation to compare against the pinned result.")
            return
        from ui.compare import CompareWindow

        win = CompareWindow(self._pinned_result[0], self._pinned_result[1], model, self.result, self)
        win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        win.show()

    def open_section_cp(self) -> None:
        if not self.cfd_view.has_result:
            self._status("Run a simulation first - the sections cut the surface pressure of the result.")
            return
        if self.mode != AppMode.CFD:
            self._set_mode(AppMode.CFD)
        from ui.section_cp import SectionCpDialog

        dlg = SectionCpDialog(self.cfd_view, self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.show()

    def grid_check(self) -> None:
        model = self.geometry_mgr.model
        if model is None:
            self._status("Load or generate a geometry first.")
            return
        self._status("Grid convergence check: solving on three grids…")
        self.tasks.start(GridCheckWorker(self.solver, model, self.cfd_panel.conditions()),
                         on_result=self._on_grid_check, on_error=lambda m, tb: self._error("Grid check failed", m),
                         on_progress=self._progress, on_finished=self._progress_done)

    def _on_grid_check(self, text: str) -> None:
        log.info("Grid convergence check:\n%s", text)
        box = QMessageBox(self)
        box.setWindowTitle("Grid convergence check")
        box.setText("Instant Aero Engine, current flow conditions.")
        box.setInformativeText(text + "\n\nGCI: estimated relative uncertainty of the fine-grid value "
                               "(Roache / Celik et al. 2008). The Instant engine's resolution is the number of surface "
                               "elements.")
        box.exec()

    def open_flight_dynamics(self) -> None:
        polar = getattr(self, "polar", None)
        if polar is None or getattr(self, "_polar_model", None) is not self.geometry_mgr.model:
            self._error("Flight dynamics", "Run a polar sweep of the current aircraft first (Polar analysis, Ctrl+2): "
                        "the flight model uses its lift, drag and pitching-moment curves.")
            return
        from ui.flight_dialog import FlightDialog

        dlg = FlightDialog(polar, self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.show()

    def open_propeller_tool(self) -> None:
        from ui.propeller_dialog import PropellerDialog

        def build(d: float, blades: int, pd: float) -> None:
            from core.geometry_manager import build_propeller

            self.geometry_mgr.set_model(build_propeller(d, blades, pd))
            self._model_changed(reset_camera=True)
            self.notify(f"Propeller D {d:g} m, {blades} blades, P/D {pd:g} loaded into the wind tunnel.")

        dlg = PropellerDialog(build, self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.show()

    def open_param_study(self) -> None:
        model = self.geometry_mgr.model
        if model is None:
            self._status("Load or generate a geometry first.")
            return
        from ui.param_study import ParamStudyDialog

        dlg = ParamStudyDialog(self.solver, model, self.cfd_panel.conditions(), self.cfd_panel.quality_value(), self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.show()

    def _on_slice_sweep(self, on: bool) -> None:
        if on and not self.cfd_view.has_result:
            self._status("Run a simulation first - the slice sweeps through the flow result.")
            self.cfd_panel.stop_slice_sweep()
            return
        self.cfd_view.update_settings(self.cfd_panel.visual_settings(), immediate=True)
        self.cfd_view.set_slice_sweep(on)

    # -- VTK / ParaView ------------------------------------------------------------
    def export_vtk(self, open_after: bool = False) -> Optional[Path]:
        """Save the current flow result as a VTK multiblock (.vtm) for ParaView / other post-processors."""
        if not self.cfd_view.has_result:
            self._status("Run a simulation first - there is no flow field to export.")
            return None
        if open_after:
            path = str(get_paths().exports / self._default_name(".vtm"))
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Export flow field", str(get_paths().exports /
                                                                                self._default_name(".vtm")),
                                                  "VTK multiblock (*.vtm)")
            if not path:
                return None
        try:
            out = self.cfd_view.export_vtk(path)
        except Exception as exc:  # noqa: BLE001
            self._error("Export failed", str(exc))
            return None
        log.success("Flow field exported: %s", out)
        if not open_after:
            self.notify(f"Flow field exported: {out.name}", "success")
        return out

    @staticmethod
    def _find_paraview() -> Optional[str]:
        import glob
        import shutil

        exe = shutil.which("paraview")
        if exe:
            return exe
        for root in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("LOCALAPPDATA", "")):
            hits = sorted(glob.glob(os.path.join(root, "ParaView*", "bin", "paraview.exe")))
            if hits:
                return hits[-1]                       # newest version (sorted by folder name)
        return None

    def open_in_paraview(self) -> None:
        exe = self._find_paraview()
        if exe is None:
            ans = QMessageBox.question(self, "ParaView not found",
                                       "ParaView (free, from Kitware) was not found on this PC.\n\n"
                                       "Open the ParaView download page?")
            if ans == QMessageBox.StandardButton.Yes:
                QDesktopServices.openUrl(QUrl("https://www.paraview.org/download/"))
            return
        out = self.export_vtk(open_after=True)
        if out is None:
            return
        import subprocess

        try:
            subprocess.Popen([exe, str(out)], close_fds=True)  # noqa: S603 - local ParaView executable
            self.notify("Opening the result in ParaView…", "info")
        except OSError as exc:
            self._error("Could not start ParaView", str(exc))

    # -- camera bookmarks --------------------------------------------------------------
    def _save_bookmark(self, n: int) -> None:
        import json

        self._settings().setValue(f"camera/bookmark{n}", json.dumps(self.cfd_view.camera_state()))
        self._status(f"Camera view {n} saved (Alt+{n} to return to it).")

    def _goto_bookmark(self, n: int) -> None:
        import json

        raw = self._settings().value(f"camera/bookmark{n}", "")
        if not raw:
            self._status(f"No view saved as {n} yet - use Ctrl+Alt+{n} to save the current view.")
            return
        if self.mode != AppMode.CFD:
            self._set_mode(AppMode.CFD)
        try:
            self.cfd_view.set_camera_state(json.loads(str(raw)))
        except ValueError:
            self._status(f"Camera view {n} is damaged - save it again.")

    def export_csv(self) -> None:
        """Export the polar as CSV."""
        if self.polar is None:
            self._status("Run a polar sweep first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export polar CSV",
                                              str(get_paths().exports / default_export_name(self.polar, ".csv")),
                                              "CSV (*.csv)")
        if path:
            try:
                export_polar_csv(self.polar, path)
                log.success("Polar CSV saved: %s", path)
                self.notify(f"Polar CSV saved: {Path(path).name}", "success")
            except OSError as exc:
                self._error("Export failed", str(exc))

    def export_report(self) -> None:
        """Export the HTML engineering report."""
        if self._recording:
            return
        model = self.geometry_mgr.model
        if model is None:
            self._status("Nothing to report yet.")
            return
        dlg = ExportReportDialog(self._default_name("_report.html"), self.polar is not None, self.result is not None, self)
        if dlg.exec() != ExportReportDialog.DialogCode.Accepted:
            return
        opt = dlg.options()
        images: list[tuple[str, str | Path]] = []
        try:
            snap_dir = opt.path.parent / (opt.path.stem + "_assets")
            if opt.include_screenshot and self.result is not None:
                previous = self.mode
                if previous != AppMode.CFD:
                    self._set_mode(AppMode.CFD)
                QApplication.processEvents()
                shot = self.cfd_view.save_screenshot(snap_dir / "flow.png", target_width=1920)
                images.append(("3-D flow field", shot))
                if previous != AppMode.CFD:
                    self._set_mode(previous)
            if opt.include_polar_charts and self.polar is not None:
                for name, plot in zip(("Lift curve", "Drag polar", "Efficiency", "Pitching moment"),
                                      self.polar_big.plots()):
                    images.append((name, export_plot_png(plot, snap_dir / f"{sanitize_filename(name)}.png")))
            cond = self.result.conditions if self.result is not None else self.cfd_panel.conditions()
            atm = cond.atmosphere
            engine = self.result.engine if self.result else self.engine.value
            cond_rows = [("Engine", engine),
                         ("Velocity", f"{cond.velocity_ms:.2f} m/s"), ("Angle of attack", f"{cond.alpha_deg:.2f}°"),
                         ("Sideslip", f"{cond.beta_deg:.2f}°"), ("Altitude", f"{cond.altitude_m:.0f} m"),
                         ("Density", f"{atm.density:.4f} kg/m³"), ("Mach", f"{cond.mach:.3f}")]
            out = export_html_report(opt.path, polar=self.polar, kpis=self.result.kpis if self.result else None,
                                     geometry_rows=model.info.rows(), title=f"{model.name} — Aerodynamic Report",
                                     notes=opt.notes, images=images, condition_rows=cond_rows, engine=engine)
            if opt.write_csv and self.polar is not None:
                export_polar_csv(self.polar, opt.path.with_suffix(".csv"))
            log.success("Report saved: %s", out)
            if opt.write_pdf:
                from ui.pdf_export import html_to_pdf

                def pdf_done(path: Optional[Path], error: str) -> None:
                    if path is not None:
                        log.success("PDF report saved: %s", path)
                        self.notify(f"PDF report saved: {path.name}", "success")
                    else:
                        log.warning(error)

                html_to_pdf(Path(out), Path(out).with_suffix(".pdf"), pdf_done)
            self.notify(f"Report saved: {Path(out).name}", "success")
        except Exception as exc:  # noqa: BLE001
            self._error("Report export failed", str(exc))

    def show_help(self, tab: str = "") -> None:
        """Help / shortcuts / about / system report."""
        HelpDialog(self, log_dir=get_paths().logs, tab=tab, openfoam=self._openfoam_summary()).exec()

    def _openfoam_summary(self) -> str:
        env = self.of_env
        if env.available:
            return f"OpenFOAM {env.version} ({env.mode.value})"
        return env.detail or "not detected"

    # =====================================================================
    # Recent files
    # =====================================================================
    _RECENT_MAX = 10

    def _recent_files(self) -> list[str]:
        try:
            v = self._settings().value("recent/files", [])
        except Exception:  # noqa: BLE001
            return []
        if isinstance(v, str):
            v = [v]
        return [str(p) for p in (v or []) if p]

    def _add_recent(self, path: str | Path) -> None:
        p = str(Path(path).resolve())
        items = [x for x in self._recent_files() if os.path.normcase(x) != os.path.normcase(p)]
        items.insert(0, p)
        self._settings().setValue("recent/files", items[:self._RECENT_MAX])

    def _fill_recent_menu(self) -> None:
        m = self.m_recent
        m.clear()
        files = self._recent_files()
        existing = [f for f in files if Path(f).is_file()]
        for i, f in enumerate(existing):
            act = m.addAction(f"&{(i + 1) % 10}  {Path(f).name}")
            act.setToolTip(f)
            act.setStatusTip(f)
            act.triggered.connect(lambda _c=False, p=f: self.open_project(p) if p.lower().endswith(".aeroflow")
                                  else self.load_file(p))
        if not existing:
            none = m.addAction("No recent files")
            none.setEnabled(False)
        m.addSeparator()
        clear = m.addAction("Clear list")
        clear.setEnabled(bool(files))
        clear.triggered.connect(lambda: self._settings().setValue("recent/files", []))

    # =====================================================================
    # Updates
    # =====================================================================
    def _auto_update_enabled(self) -> bool:
        v = self._settings().value("updates/auto", True)
        return v if isinstance(v, bool) else str(v).lower() not in ("false", "0")

    def _maybe_auto_check_updates(self) -> None:
        """At most one automatic check a day, in the background, only if enabled."""
        if not self._auto_update_enabled() or os.environ.get("AEROFLOW_NO_UPDATE_CHECK"):
            return
        try:
            last = float(self._settings().value("updates/last_check", 0.0) or 0.0)
        except (TypeError, ValueError):
            last = 0.0
        if time.time() - last < 24 * 3600:
            return
        self.check_for_updates(manual=False)

    def check_for_updates(self, manual: bool = True) -> None:
        if self._update_checker.busy:
            return
        self._update_manual = manual
        if manual:
            self._status("Checking for updates…")
        self._update_checker.start()

    def _on_update_result(self, info: Optional[ReleaseInfo], error: str) -> None:
        manual = self._update_manual
        if error:
            log.info("Update check: %s", error)
            if manual:
                QMessageBox.information(self, "Check for updates", f"Could not check for updates.\n\n{error}")
            return
        self._settings().setValue("updates/last_check", time.time())
        if info is None or not is_newer(info.version):
            self._latest_release = None
            self.sb_update.hide()
            if manual:
                QMessageBox.information(self, "Check for updates",
                                        f"You are running the latest version ({APP_NAME} {APP_VERSION}).")
            return
        self._latest_release = info
        self.sb_update.setText(f"⬆ Update {info.version} available")
        self.sb_update.show()
        log.info("A new version is available: %s (%s)", info.version, info.page_url)
        if manual:
            self._show_update_dialog()
        else:
            self.toasts.show(f"{APP_NAME} {info.version} is available — click the update button in the status bar.",
                             "info", 8.0)

    def _show_update_dialog(self) -> None:
        info = self._latest_release
        if info is None:
            return
        dlg = UpdateDialog(info, self)
        dlg.install_started.connect(lambda: QTimer.singleShot(300, self.close))
        dlg.exec()

    # =====================================================================
    # Status helpers
    # =====================================================================
    def _progress(self, fraction: float, message: str) -> None:
        self.sb_progress.show()
        self.sb_progress.setValue(int(fraction * 1000))
        self.sb_msg.setText(message)

    def _progress_done(self) -> None:
        # TaskManager removes a finished task *before* its on_finished callback runs, so the count here is
        # the number of tasks still running.
        if self.tasks.active_count == 0 and not self._recording:
            self.sb_progress.hide()
            self.sb_progress.setValue(0)

    def notify(self, text: str, level: str = "info") -> None:
        """Non-modal toast in the bottom-right corner (also mirrored to the status bar)."""
        self.sb_msg.setText(text)
        if not self._closing:
            self.toasts.show(text, level)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.toasts.relayout()

    def _status(self, text: str) -> None:
        self.sb_msg.setText(text)

    def _show_fps(self, fps: float, source: AppMode) -> None:
        if self.mode == source:
            self.sb_fps.setText(f"{fps:5.1f} FPS")

    def _on_busy(self, busy: bool) -> None:
        if not busy and not self._recording:
            self.sb_progress.hide()

    def _error(self, title: str, message: str, modal: bool = True) -> None:
        """Report an error: always logged and shown in the status bar; a dialog only when ``modal`` and
        no other error dialog is already open (so repeated failures never stack dialogs)."""
        log.error("%s: %s", title, message)
        first = message.splitlines()[0] if message else ""
        self._status(f"{title}: {first}")
        if not modal or self._error_box_open:
            if not self._closing:
                self.toasts.show(f"{title}: {first}", "error")
            return
        if self._closing:
            return
        box = QMessageBox(QMessageBox.Icon.Warning, title, message.split("\n--- last output ---")[0], parent=self)
        if "--- last output ---" in message:
            box.setDetailedText(message.split("--- last output ---", 1)[1].strip())
        self._error_box_open = True
        try:
            box.exec()
        finally:
            self._error_box_open = False

    # =====================================================================
    # Qt events
    # =====================================================================
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        md = event.mimeData()
        if md.hasUrls() and any(Path(u.toLocalFile()).suffix.lower() in SUPPORTED_MESH_EXTENSIONS for u in md.urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in SUPPORTED_MESH_EXTENSIONS:
                event.acceptProposedAction()
                self.load_file(path)
                return

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Stop timers, kill workers / OpenFOAM, and close the VTK interactor before exit."""
        if self._recording:
            event.ignore()
            return
        self._closing = True
        log.info("Shutting down…")
        self._save_layout()
        try:
            self.cfd_view.set_animation_paused(True)
            self.lab_view.shutdown()
            if self._supersonic is not None:
                self._supersonic.shutdown()
                self._supersonic.close()
            if self._nozzle is not None:
                self._nozzle.shutdown()
                self._nozzle.close()
            self.residual_chart.stop()
            self.tasks.shutdown(timeout_ms=5000)
        finally:
            self.cfd_view.shutdown()
        event.accept()
