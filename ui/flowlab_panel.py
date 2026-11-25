"""Left control panel of the Live Flow Lab (real-time 2-D wind tunnel).

The panel drives a :class:`visualization.flowlab_view.FlowLabView` directly: every control applies
immediately while the flow keeps running, so changes (angle of attack, Reynolds number, a wall drawn
with the mouse) can be watched as they happen.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QSlider, QVBoxLayout, QWidget,
)

from config.settings import LAB_RE_DEFAULT, LAB_RE_RANGE, PARAM_LIMITS, get_paths
from config.theme import PALETTE, make_icon, mono_font, set_property
from core.shapes2d import Obstacle2D
from ui.cfd_panel import AccordionCard, ParamSlider, ToggleSwitch, button, compact_combo, labeled_row
from ui.device_selector import DeviceSelector
from ui.widgets2d import AirfoilFileRow
from utils.validators import validate_naca_code
from visualization.flowlab_view import DEFAULT_RESOLUTION, LAB_COLORMAPS, LAB_RESOLUTIONS, FlowLabView, LabField


class LogSlider(QWidget):
    """Horizontal slider on a logarithmic scale with a live value label (for the Reynolds number)."""

    valueChanged = Signal(float)     # noqa: N815 (Qt naming)

    STEPS = 1000

    def __init__(self, label: str, lo: float, hi: float, value: float, tooltip: str = "",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.lo, self.hi = math.log10(lo), math.log10(hi)
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setVerticalSpacing(2)
        self.caption = QLabel(label)
        self.caption.setToolTip(tooltip)
        self.value_label = QLabel("")
        self.value_label.setFont(mono_font(12))
        self.value_label.setMinimumWidth(self.value_label.fontMetrics().horizontalAdvance("100,0000") + 8)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.STEPS)
        self.slider.setToolTip(tooltip)
        lay.addWidget(self.caption, 0, 0)
        lay.addWidget(self.value_label, 0, 1)
        lay.addWidget(self.slider, 1, 0, 1, 2)
        self.slider.valueChanged.connect(self._changed)
        self.set_value(value)

    def value(self) -> float:
        frac = self.slider.value() / self.STEPS
        v = 10 ** (self.lo + frac * (self.hi - self.lo))
        mag = 10 ** math.floor(math.log10(v))
        return float(round(v / mag * 4) / 4 * mag)       # 2 significant-ish digits: 1500, 2000, 25000

    def set_value(self, v: float) -> None:
        frac = (math.log10(max(v, 1e-12)) - self.lo) / (self.hi - self.lo)
        self.slider.setValue(int(round(min(max(frac, 0.0), 1.0) * self.STEPS)))
        self._show()

    def _show(self) -> None:
        self.value_label.setText(f"{self.value():,.0f}")

    def _changed(self, _pos: int) -> None:
        self._show()
        self.valueChanged.emit(self.value())


class FlowLabPanel(QWidget):
    """Controls for the Live Flow Lab."""

    status_message = Signal(str)

    def __init__(self, view: FlowLabView, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Root")
        self.view = view
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)
        root.addWidget(self._run_card())
        root.addWidget(self._body_card())
        root.addWidget(self._flow_card())
        root.addWidget(self._display_card())
        root.addWidget(self._draw_card())
        root.addStretch(1)
        for combo in self.findChildren(QComboBox):
            compact_combo(combo)
        self._sync_body_ui()

    # -- cards -----------------------------------------------------------------
    def _run_card(self) -> AccordionCard:
        card = AccordionCard("Simulation")
        intro = QLabel("A real-time 2-D wind tunnel (Lattice-Boltzmann). Everything you change applies "
                       "instantly while the air keeps flowing.")
        intro.setWordWrap(True)
        intro.setProperty("role", "muted")
        card.add(intro)
        row = QHBoxLayout()
        self.btn_play = button("Pause", "primary", "pause", "Pause / resume the simulation")
        self.btn_step = button("Step", "ghost", "", "Advance one frame while paused")
        self.btn_reset = button("Restart", "", "reset", "Restart the flow from rest (keeps the body)")
        row.addWidget(self.btn_play, 2)
        row.addWidget(self.btn_step, 1)
        row.addWidget(self.btn_reset, 2)
        card.add_layout(row)
        self.btn_play.clicked.connect(self.toggle_run)
        self.btn_step.clicked.connect(self.view.step_once)
        self.btn_reset.clicked.connect(self.view.reset_flow)
        self.speed = ParamSlider(PARAM_LIMITS["lab_speed"])
        self.speed.valueCommitted.connect(self.view.set_speed)
        card.add(self.speed)
        self.resolution = QComboBox()
        for name in LAB_RESOLUTIONS:
            self.resolution.addItem(name)
        self.resolution.setCurrentText(DEFAULT_RESOLUTION)
        self.resolution.currentTextChanged.connect(self.view.set_resolution)
        card.add(labeled_row("Grid", self.resolution, "More cells show finer vortices but run slower."))
        self.device = DeviceSelector()
        card.add(self.device)
        snap = button("Save snapshot…", "ghost", "camera", "Save the current view as a PNG image")
        snap.clicked.connect(self._snapshot)
        card.add(snap)
        return card

    def _body_card(self) -> AccordionCard:
        card = AccordionCard("Body in the tunnel")
        self.kind = QComboBox()
        for k in Obstacle2D:
            self.kind.addItem(k.value, k)
        self.kind.setCurrentIndex(list(Obstacle2D).index(Obstacle2D.AIRFOIL))
        self.kind.currentIndexChanged.connect(self._body_changed)
        card.add(self.kind)
        self.naca_row = QWidget()
        h = QHBoxLayout(self.naca_row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel("NACA"))
        self.naca = QLineEdit("2412")
        self.naca.setMaxLength(4)
        self.naca.setFont(mono_font(12))
        self.naca.setToolTip("Four-digit NACA code, e.g. 0012 (symmetric) or 4415 (high camber).")
        self.naca.editingFinished.connect(self._body_changed)
        h.addWidget(self.naca, 1)
        self.naca_status = QLabel("")
        self.naca_status.setProperty("role", "muted")
        h.addWidget(self.naca_status)
        card.add(self.naca_row)
        self.file_row = AirfoilFileRow(self._load_airfoil_file)
        card.add(self.file_row)
        self.aoa = ParamSlider(PARAM_LIMITS["lab_aoa"])
        self.aoa.valueChanged.connect(self.view.set_aoa)          # live while dragging
        card.add(self.aoa)
        self.size = ParamSlider(PARAM_LIMITS["lab_size"])
        self.size.valueCommitted.connect(self.view.set_size)
        card.add(self.size)
        self.section_note = QLabel("Uses the mid-span cross-section of the geometry loaded in the Wind Tunnel.")
        self.section_note.setWordWrap(True)
        self.section_note.setProperty("role", "muted")
        card.add(self.section_note)
        return card

    def _flow_card(self) -> AccordionCard:
        card = AccordionCard("Flow")
        self.re = LogSlider("Reynolds number", *LAB_RE_RANGE, LAB_RE_DEFAULT,
                            "Re = U·L/ν. Low values (< 50) give smooth, sticky flow; around 100 a cylinder sheds a "
                            "regular vortex street; thousands give turbulent, chaotic wakes.")
        self.re.valueChanged.connect(self.view.set_reynolds)
        card.add(self.re)
        presets = QHBoxLayout()
        for text, value, tip in (("Re 40", 40.0, "Creeping flow: smooth, attached, no shedding"),
                                 ("Re 150", 150.0, "Laminar vortex street behind bluff bodies"),
                                 ("Re 20k", 20_000.0, "Turbulent, chaotic wake (sub-grid model active)")):
            b = button(text, "ghost", tooltip=f"Set Re = {value:,.0f}: {tip}")
            b.clicked.connect(lambda _=False, v=value: self.re.set_value(v) or self.view.set_reynolds(v))
            presets.addWidget(b)
        card.add_layout(presets)
        return card

    def _display_card(self) -> AccordionCard:
        card = AccordionCard("Display")
        self.field = QComboBox()
        tips = {LabField.SPEED: "How fast the air moves", LabField.VORTICITY: "How the air spins (vortices)",
                LabField.PRESSURE: "Pressure coefficient: blue suction, red high pressure",
                LabField.SMOKE: "Only the smoke streaks, like a real smoke tunnel",
                LabField.TEXTURE: "Every streamline at once as a flowing texture (line-integral convolution)",
                LabField.MEAN_SPEED: "Speed averaged over time since the last restart / change: the mean wake "
                                     "that a wind-tunnel probe would measure",
                LabField.RMS: "How much the velocity fluctuates: bright where vortices shed or the flow is "
                              "turbulent, dark where it is steady"}
        for i, f in enumerate(LabField):
            self.field.addItem(f.value, f)
            self.field.setItemData(i, tips.get(f, ""), Qt.ItemDataRole.ToolTipRole)
        self.field.setCurrentIndex(list(LabField).index(LabField.VORTICITY))
        self.field.currentIndexChanged.connect(lambda i: self.view.set_field(list(LabField)[i]))
        card.add(labeled_row("Show", self.field, "Which property of the air is coloured"))
        self.cmap = QComboBox()
        for c in LAB_COLORMAPS:
            self.cmap.addItem(c)
        self.cmap.currentTextChanged.connect(self.view.set_colormap)
        card.add(labeled_row("Speed colours", self.cmap))
        grid = QGridLayout()
        self.t_smoke = ToggleSwitch("Smoke overlay", False,
                                    "White smoke streaks from an inlet rake drawn over the field "
                                    "(the Smoke view always shows them)")
        self.t_particles = ToggleSwitch("Particles", True, "Tiny tracer particles carried by the flow")
        grid.addWidget(self.t_smoke, 0, 0)
        grid.addWidget(self.t_particles, 0, 1)
        for t in (self.t_smoke, self.t_particles):
            t.toggled.connect(lambda _c: self.view.set_overlays(self.t_smoke.isChecked(), self.t_particles.isChecked()))
        card.add_layout(grid)
        return card

    def _draw_card(self) -> AccordionCard:
        card = AccordionCard("Draw obstacles", expanded=False)
        note = QLabel("Turn drawing on, then paint walls into the tunnel with the left mouse button and "
                      "erase them with the right. The flow reacts immediately.")
        note.setWordWrap(True)
        note.setProperty("role", "muted")
        card.add(note)
        self.t_draw = ToggleSwitch("Drawing enabled", False)
        self.t_draw.toggled.connect(lambda on: self.view.set_drawing(on, self.brush.value()))
        card.add(self.t_draw)
        self.brush = ParamSlider(PARAM_LIMITS["lab_brush"])
        self.brush.valueChanged.connect(lambda v: self.view.set_drawing(self.t_draw.isChecked(), v))
        card.add(self.brush)
        clear = button("Clear drawing", "ghost", "reset", "Remove everything you drew (keeps the chosen body)")
        clear.clicked.connect(self.view.clear_drawing)
        card.add(clear)
        return card

    # -- behaviour ---------------------------------------------------------------
    def _selected_kind(self) -> Obstacle2D:
        data = self.kind.currentData()
        return data if isinstance(data, Obstacle2D) else Obstacle2D(data)

    def _sync_body_ui(self) -> None:
        k = self._selected_kind()
        self.naca_row.setVisible(k is Obstacle2D.AIRFOIL)
        self.file_row.setVisible(k is Obstacle2D.AIRFOIL_FILE)
        self.section_note.setVisible(k is Obstacle2D.MODEL_SECTION)
        self.aoa.setEnabled(k not in (Obstacle2D.CYLINDER, Obstacle2D.EMPTY))
        self.size.setEnabled(k is not Obstacle2D.EMPTY)
        if k is Obstacle2D.EMPTY and not self.t_draw.isChecked():
            self.status_message.emit("Empty tunnel: open 'Draw obstacles' and paint your own shape with the mouse.")

    def _body_changed(self, *_args: object) -> None:
        k = self._selected_kind()
        code = self.naca.text().strip()
        if k is Obstacle2D.AIRFOIL:
            naca, res = validate_naca_code(code)
            if naca is None:
                self.naca_status.setText("invalid")
                set_property(self.naca_status, "level", "danger")
                self.status_message.emit(res.summary(1).lstrip("- "))
                return
            self.naca_status.setText("")
            set_property(self.naca_status, "level", "")
        self._sync_body_ui()
        if k is Obstacle2D.AIRFOIL_FILE and not self.view.airfoil_file:
            self.file_row.browse()                    # first time: ask for the file right away
            return
        self.view.set_obstacle(k, code if k is Obstacle2D.AIRFOIL else None)

    def _load_airfoil_file(self, path: str) -> None:
        self.view.airfoil_file = path
        self.view.set_obstacle(Obstacle2D.AIRFOIL_FILE)

    def toggle_run(self) -> None:
        if self.view.running:
            self.view.stop()
        else:
            self.view.start()
        self.sync_run_button()

    def sync_run_button(self) -> None:
        running = self.view.running
        self.btn_play.setText("Pause" if running else "Run")
        self.btn_play.setIcon(make_icon("pause" if running else "play", 16, PALETTE.on_accent))
        self.btn_step.setEnabled(not running)

    def _snapshot(self) -> None:
        import time

        default = get_paths().exports / f"flowlab_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path, _ = QFileDialog.getSaveFileName(self, "Save snapshot", str(default), "PNG image (*.png)")
        if path:
            try:
                self.view.save_png(Path(path))
                self.status_message.emit(f"Snapshot saved: {path}")
            except OSError as exc:
                self.status_message.emit(f"Snapshot failed: {exc}")
