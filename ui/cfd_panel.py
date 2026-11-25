"""Left control dock for the Wind Tunnel CFD and Polar Analysis modes, plus shared control widgets.

Shared widgets:

* :class:`AccordionCard`: collapsible titled card.
* :class:`ParamSlider`: plain-English label + unit badge + synchronised slider/spinbox + tooltip,
  driven entirely by :data:`config.settings.PARAM_LIMITS`.
* :class:`ToggleSwitch`: pill-style checkbox.
* :class:`SegmentedControl`: exclusive segmented buttons.

The panel never talks to solvers directly; it emits intent signals that the main window wires to
workers and viewports.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional, Sequence

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

from config import units
from config.settings import (
    DEFAULT_COLORMAPS, PARAM_LIMITS, FlowConditions, GeometryPreset, GridQuality, IsoField, ParamLimit,
    ReferenceAreaMode, ScalarField, SeedShape, SliceAxis, SolverEngine,
)
from config.theme import PALETTE, make_icon, make_section_label, mono_font, set_property
from utils.validators import validate_naca_code
from visualization.cfd_viewport import VisualSettings


# ---------------------------------------------------------------------------
# Shared widgets
# ---------------------------------------------------------------------------
class AccordionCard(QFrame):
    """A titled card whose body collapses when the header is clicked."""

    toggled = Signal(bool)

    def __init__(self, title: str, icon: str = "", expanded: bool = True, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 8)
        outer.setSpacing(2)
        self.header = QToolButton(self)
        self.header.setObjectName("CardToggle")
        self.header.setText("  " + title.upper())
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.header.setCursor(Qt.CursorShape.PointingHandCursor)
        self.header.setIconSize(QSize(14, 14))
        self.header.clicked.connect(self._on_click)
        outer.addWidget(self.header)
        self.body = QWidget(self)
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(10, 4, 10, 4)
        self.body_layout.setSpacing(8)
        outer.addWidget(self.body)
        self._icon = icon
        self.set_expanded(expanded)

    def _on_click(self, checked: bool) -> None:
        self.set_expanded(checked)
        self.toggled.emit(checked)

    def set_expanded(self, expanded: bool) -> None:
        """Expand or collapse the body."""
        self.header.setChecked(expanded)
        self.header.setIcon(make_icon("chevron_down" if expanded else "chevron_right", 14))
        self.body.setVisible(expanded)

    def add(self, widget: QWidget) -> QWidget:
        """Append a widget to the card body."""
        self.body_layout.addWidget(widget)
        return widget

    def add_layout(self, layout: QHBoxLayout | QVBoxLayout | QGridLayout) -> None:
        """Append a layout to the card body."""
        self.body_layout.addLayout(layout)


class ParamSlider(QWidget):
    """Label + unit badge + slider + spinbox for one :class:`ParamLimit` (kept in sync)."""

    valueChanged = Signal(float)      # live, while dragging   # noqa: N815 (Qt naming)
    valueCommitted = Signal(float)    # on release / spinbox edit finished   # noqa: N815

    STEPS = 1000

    def __init__(self, limit: ParamLimit, value: Optional[float] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.limit = limit
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setHorizontalSpacing(6)
        lay.setVerticalSpacing(2)
        self.label = QLabel(limit.label)
        self.label.setToolTip(limit.tooltip)
        self.label.setWordWrap(True)
        self.unit = QLabel(limit.unit or "—")
        self.unit.setProperty("role", "unit")
        self.unit.setVisible(bool(limit.unit))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.STEPS)
        self.slider.setToolTip(limit.tooltip)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(limit.minimum, limit.maximum)
        self.spin.setDecimals(limit.decimals)
        self.spin.setSingleStep(limit.step)
        self.spin.setFont(mono_font(12))
        self.spin.setFixedWidth(84)
        self.spin.setKeyboardTracking(False)
        self.spin.setToolTip(limit.tooltip)
        self.spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        lay.addWidget(self.label, 0, 0)
        lay.addWidget(self.unit, 0, 1, alignment=Qt.AlignmentFlag.AlignRight)
        lay.addWidget(self.slider, 1, 0)
        lay.addWidget(self.spin, 1, 1)
        lay.setColumnStretch(0, 1)
        self._updating = False
        self._k = 1.0                       # display = k * SI value (unit conversion; SI by default)
        self.slider.valueChanged.connect(self._from_slider)
        self.slider.sliderReleased.connect(lambda: self.valueCommitted.emit(self.value()))
        self.spin.valueChanged.connect(self._from_spin)
        self.set_value(limit.default if value is None else value)

    def _to_slider(self, v: float) -> int:
        span = self.limit.maximum - self.limit.minimum
        return int(round((v - self.limit.minimum) / span * self.STEPS)) if span > 0 else 0

    def _from_slider(self, pos: int) -> None:
        if self._updating:
            return
        v = self.limit.minimum + (self.limit.maximum - self.limit.minimum) * pos / self.STEPS
        step = self.limit.step
        if step > 0:
            v = round(v / step) * step
        v = self.limit.clamp(v)
        self._updating = True
        self.spin.setValue(v * self._k)
        self._updating = False
        self.valueChanged.emit(v)
        if not self.slider.isSliderDown():
            self.valueCommitted.emit(v)

    def _from_spin(self, d: float) -> None:
        if self._updating:
            return
        v = d / self._k
        self._updating = True
        self.slider.setValue(self._to_slider(v))
        self._updating = False
        self.valueChanged.emit(v)
        self.valueCommitted.emit(v)

    def value(self) -> float:
        """Current value (always in the parameter's SI unit)."""
        return self.limit.clamp(float(self.spin.value()) / self._k)

    def set_display_unit(self, scale: float, unit: str, decimals: Optional[int] = None) -> None:
        """Show (and accept typing in) another unit: display = ``scale`` x SI value. The value is kept."""
        v = self.value()
        self._k = float(scale) if scale > 0 else 1.0
        self._updating = True
        self.spin.setDecimals(self.limit.decimals if decimals is None else decimals)
        self.spin.setRange(self.limit.minimum * self._k, self.limit.maximum * self._k)
        self.spin.setSingleStep(self.limit.step * self._k)
        self.spin.setValue(v * self._k)
        self._updating = False
        self.unit.setText(unit)
        self.unit.setVisible(bool(unit))

    def set_value(self, v: float, emit: bool = False) -> None:
        """Set the value (clamped). Signals are only emitted when ``emit`` is True."""
        v = self.limit.clamp(v)
        self._updating = True
        self.spin.setValue(v * self._k)
        self.slider.setValue(self._to_slider(v))
        self._updating = False
        if emit:
            self.valueChanged.emit(v)
            self.valueCommitted.emit(v)


class ToggleSwitch(QCheckBox):
    """Pill-style on/off switch (styled by ``[toggle="true"]`` in the stylesheet)."""

    def __init__(self, text: str, checked: bool = False, tooltip: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setProperty("toggle", "true")
        self.setChecked(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if tooltip:
            self.setToolTip(tooltip)


class SegmentedControl(QFrame):
    """Exclusive segmented buttons. ``changed`` emits the selected index."""

    changed = Signal(int)

    def __init__(self, labels: Sequence[str], tooltips: Optional[Sequence[str]] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Segmented")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: list[QPushButton] = []
        for i, text in enumerate(labels):
            b = QPushButton(text)
            b.setProperty("role", "segment")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            if tooltips and i < len(tooltips):
                b.setToolTip(tooltips[i])
            self.group.addButton(b, i)
            self.buttons.append(b)
            lay.addWidget(b)
        if self.buttons:
            self.buttons[0].setChecked(True)
        self.group.idClicked.connect(self.changed.emit)

    def index(self) -> int:
        """Selected index."""
        return max(self.group.checkedId(), 0)

    def set_index(self, i: int, emit: bool = False) -> None:
        """Select button ``i``."""
        if 0 <= i < len(self.buttons):
            self.buttons[i].setChecked(True)
            if emit:
                self.changed.emit(i)

    def set_enabled_at(self, i: int, enabled: bool, tooltip: str = "") -> None:
        """Enable/disable one segment."""
        if 0 <= i < len(self.buttons):
            self.buttons[i].setEnabled(enabled)
            if tooltip:
                self.buttons[i].setToolTip(tooltip)


def combo_enum(combo: QComboBox, enum_cls: type[Enum]) -> Optional[Enum]:
    """Return a combo's current item data as a member of ``enum_cls``.

    PySide6 converts ``str``-based enum members stored as item data into plain ``str`` on the way
    out, so ``currentData()`` gives ``"NACA 2412 wing"`` rather than ``GeometryPreset.NACA_2412``.
    This maps the value back to the real member (identity checks with ``is`` then work).
    """
    data = combo.currentData()
    if data is None or isinstance(data, enum_cls):
        return data
    try:
        return enum_cls(data)
    except ValueError:
        return None


def compact_combo(combo: QComboBox, min_chars: int = 10) -> QComboBox:
    """Stop a combo box from demanding the width of its longest item (which clipped the dock).

    The popup list still shows the full text; the closed box elides to the available width."""
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(min_chars)
    combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    view = combo.view()
    if view is not None:
        view.setTextElideMode(Qt.TextElideMode.ElideNone)
        view.setMinimumWidth(combo.view().sizeHintForColumn(0) + 24)
    return combo


def labeled_row(text: str, widget: QWidget, tooltip: str = "") -> QWidget:
    """A ``label .... widget`` row. The label wraps instead of forcing the panel wider."""
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
    if tooltip:
        lbl.setToolTip(tooltip)
        widget.setToolTip(tooltip)
    if isinstance(widget, QComboBox):
        compact_combo(widget)
    h.addWidget(lbl, 2)
    h.addWidget(widget, 3)
    return row


def button(text: str, role: str = "", icon: str = "", tooltip: str = "") -> QPushButton:
    """Styled push button."""
    b = QPushButton(text)
    if role:
        b.setProperty("role", role)
    if icon:
        b.setIcon(make_icon(icon, 16, PALETTE.on_accent if role == "primary" else None))
    if tooltip:
        b.setToolTip(tooltip)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def scroll_wrap(content: QWidget) -> QScrollArea:
    """Put a panel in a frameless vertical scroll area."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    # AsNeeded (not AlwaysOff): if a panel is ever wider than the dock, it scrolls instead of clipping.
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setWidget(content)
    return area


# ---------------------------------------------------------------------------
# The CFD panel
# ---------------------------------------------------------------------------
class CFDPanel(QWidget):
    """All controls for geometry, flow conditions, solver, visualisation, polar sweep and export."""

    preset_requested = Signal(object, str, float, float, float)   # preset, naca, span, chord, tail
    import_requested = Signal()
    geometry_op = Signal(str, object)
    reference_mode_changed = Signal(object)
    conditions_changed = Signal(object)          # FlowConditions (live)
    conditions_committed = Signal(object)        # FlowConditions (slider released)
    run_requested = Signal()
    abort_requested = Signal()
    engine_changed = Signal(object)
    visuals_changed = Signal(object)             # VisualSettings
    sweep_requested = Signal(float, float, float)
    export_requested = Signal(str)               # screenshot | gif | csv | report

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Root")
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)
        self.cards: dict[str, AccordionCard] = {}
        root.addWidget(self._geometry_card())
        root.addWidget(self._flow_card())
        root.addWidget(self._solver_card())
        root.addWidget(self._visual_card())
        root.addWidget(self._polar_card())
        root.addWidget(self._export_card())
        root.addStretch(1)
        for combo in self.findChildren(QComboBox):
            if combo is not self.scale_kind:
                compact_combo(combo)
        self._visual_timer = QTimer(self)
        self._visual_timer.setSingleShot(True)
        self._visual_timer.setInterval(40)
        self._visual_timer.timeout.connect(lambda: self.visuals_changed.emit(self.visual_settings()))

    # -- cards -------------------------------------------------------------------
    def _geometry_card(self) -> AccordionCard:
        card = AccordionCard("Geometry", expanded=True)
        self.cards["geometry"] = card
        self.preset = QComboBox()
        for p in GeometryPreset:
            self.preset.addItem(p.value, p)
        self.preset.setCurrentIndex(list(GeometryPreset).index(GeometryPreset.NACA_2412))
        self.preset.setToolTip("Built-in shapes generated instantly — no files needed.")
        card.add(self.preset)
        # Custom NACA row
        self.custom_box = QWidget()
        g = QGridLayout(self.custom_box)
        g.setContentsMargins(0, 0, 0, 0)
        self.naca_code = QLineEdit("4415")
        self.naca_code.setMaxLength(4)
        self.naca_code.setFont(mono_font(12))
        self.naca_code.setToolTip("Four digits: max camber %, camber position (tenths), thickness %. Example: 4415.")
        self.span = QDoubleSpinBox()
        self.span.setRange(0.05, 200.0)
        self.span.setValue(6.0)
        self.span.setSuffix(" m")
        self.span.setToolTip("Wing span (tip to tip).")
        self.chord = QDoubleSpinBox()
        self.chord.setRange(0.01, 50.0)
        self.chord.setValue(1.0)
        self.chord.setSuffix(" m")
        self.chord.setToolTip("Wing chord (leading edge to trailing edge).")
        self.tail = QDoubleSpinBox()
        self.tail.setRange(0.0, 4.0)
        self.tail.setSingleStep(0.25)
        self.tail.setValue(1.5)
        self.tail.setToolTip("Teardrop tail length relative to the diameter (0 = perfect sphere).")
        self.naca_status = QLabel("")
        self.naca_status.setProperty("role", "muted")
        g.addWidget(QLabel("NACA"), 0, 0)
        g.addWidget(self.naca_code, 0, 1)
        g.addWidget(QLabel("Span"), 1, 0)
        g.addWidget(self.span, 1, 1)
        g.addWidget(QLabel("Chord"), 2, 0)
        g.addWidget(self.chord, 2, 1)
        g.addWidget(self.naca_status, 3, 0, 1, 2)
        self.tail_row = labeled_row("Teardrop tail", self.tail)
        card.add(self.custom_box)
        card.add(self.tail_row)
        row = QHBoxLayout()
        self.btn_generate = button("Generate", "primary", "play", "Build the selected preset")
        self.btn_import = button("Import…", "", "folder", "Load STL / OBJ / PLY / OFF / VTK (or drag a file onto the 3-D view)")
        row.addWidget(self.btn_generate, 1)
        row.addWidget(self.btn_import, 1)
        card.add_layout(row)
        self.btn_generate.clicked.connect(self._emit_preset)
        self.btn_import.clicked.connect(self.import_requested.emit)
        self.preset.currentIndexChanged.connect(self._preset_ui)
        self.naca_code.textChanged.connect(self._check_naca)
        self._preset_ui()

        card.add(make_section_label("Orientation & scale"))
        self.nose_dir = QComboBox()
        for d in ("+X", "-X", "+Y", "-Y", "+Z", "-Z"):
            self.nose_dir.addItem(f"Nose points {d}", d)
        self.nose_dir.setToolTip("Tell AeroFlow where the nose points now; it will be turned to face the wind (−X).")
        btn_align = button("Align nose → wind", "", "wind")
        btn_align.clicked.connect(lambda: self.geometry_op.emit("align_nose", self.nose_dir.currentData()))
        card.add(self.nose_dir)
        card.add(btn_align)
        rot = QGridLayout()
        rot.setHorizontalSpacing(4)
        for col, axis in enumerate(("X", "Y", "Z")):
            for row_i, turns in enumerate((1, -1)):
                b = button(f"{axis} {'+' if turns > 0 else '−'}90°", "ghost")
                b.setToolTip(f"Rotate {turns * 90:+d}° about the {axis} axis")
                b.clicked.connect(lambda _=False, a=axis, t=turns: self.geometry_op.emit("rotate", (a, t)))
                rot.addWidget(b, row_i, col)
        card.add_layout(rot)
        row2 = QHBoxLayout()
        for text, op, tip in (("Center", "center", "Move the model's centre to the origin"),
                              ("Auto-orient", "auto_orient", "Longest axis → X (fuselages, rockets, cars)"),
                              ("mm → m", "mm_to_m", "Convert a CAD model exported in millimetres")):
            b = button(text, "ghost", tooltip=tip)
            b.clicked.connect(lambda _=False, o=op: self.geometry_op.emit(o, None))
            row2.addWidget(b)
        card.add_layout(row2)
        scale_row = QHBoxLayout()
        self.scale_target = QDoubleSpinBox()
        self.scale_target.setRange(0.001, 10_000.0)
        self.scale_target.setDecimals(3)
        self.scale_target.setValue(6.0)
        self.scale_target.setSuffix(" m")
        self.scale_target.setFixedWidth(104)
        self.scale_target.setToolTip("Target size in metres")
        self.scale_kind = QComboBox()
        self.scale_kind.addItem("Span", "scale_span")
        self.scale_kind.addItem("Length", "scale_length")
        compact_combo(self.scale_kind, 5)
        b_scale = button("Scale", "ghost", tooltip="Scale uniformly so the chosen dimension matches")
        b_scale.clicked.connect(lambda: self.geometry_op.emit(self.scale_kind.currentData(), self.scale_target.value()))
        scale_row.addWidget(self.scale_kind)
        scale_row.addWidget(self.scale_target, 1)
        scale_row.addWidget(b_scale)
        card.add_layout(scale_row)
        row3 = QHBoxLayout()
        b_repair = button("Repair mesh", "", "settings", "Fix inverted normals, duplicate / degenerate faces, small holes")
        b_undo = button("Undo", "ghost", "reset", "Undo the last geometry change (Ctrl+Z)")
        b_repair.clicked.connect(lambda: self.geometry_op.emit("repair", None))
        b_undo.clicked.connect(lambda: self.geometry_op.emit("undo", None))
        row3.addWidget(b_repair, 1)
        row3.addWidget(b_undo)
        card.add_layout(row3)
        # Control surfaces: deflect a plain flap / ailerons on the current wing (undoable).
        card.add(make_section_label("Control surfaces"))
        cs = QGridLayout()
        self.cs_kind = QComboBox()
        self.cs_kind.addItem("Flaps (both sides)", "both")
        self.cs_kind.addItem("Ailerons (right down, left up)", "ailerons")
        compact_combo(self.cs_kind, 8)
        self.cs_delta, self.cs_hinge, self.cs_from, self.cs_to = (QDoubleSpinBox() for _ in range(4))
        for sb, lo, hi, val, suf, tip in (
                (self.cs_delta, -40.0, 40.0, 15.0, "°", "Deflection: positive = trailing edge down (more lift)"),
                (self.cs_hinge, 50.0, 95.0, 75.0, " % c", "Hinge line position along the local chord"),
                (self.cs_from, 0.0, 95.0, 10.0, " %", "Inboard end of the surface (% of the semi-span)"),
                (self.cs_to, 5.0, 100.0, 60.0, " %", "Outboard end of the surface (% of the semi-span)")):
            sb.setRange(lo, hi)
            sb.setValue(val)
            sb.setSuffix(suf)
            sb.setDecimals(1)
            sb.setToolTip(tip)
        cs.addWidget(self.cs_kind, 0, 0, 1, 2)
        cs.addWidget(QLabel("Deflection"), 1, 0)
        cs.addWidget(self.cs_delta, 1, 1)
        cs.addWidget(QLabel("Hinge"), 2, 0)
        cs.addWidget(self.cs_hinge, 2, 1)
        cs.addWidget(QLabel("Span from / to"), 3, 0)
        span_row = QHBoxLayout()
        span_row.addWidget(self.cs_from)
        span_row.addWidget(self.cs_to)
        cs.addLayout(span_row, 3, 1)
        card.add_layout(cs)
        b_cs = button("Deflect", "ghost", "", "Apply the deflection to the current geometry (Undo reverts it)")
        b_cs.clicked.connect(lambda: self.geometry_op.emit("flap", (
            self.cs_kind.currentData(), self.cs_delta.value(), self.cs_hinge.value() / 100.0,
            self.cs_from.value() / 100.0, self.cs_to.value() / 100.0)))
        card.add(b_cs)
        self.ref_mode = QComboBox()
        for m in ReferenceAreaMode:
            self.ref_mode.addItem(m.value, m)
        self.ref_mode.currentIndexChanged.connect(lambda _i: self.reference_mode_changed.emit(combo_enum(self.ref_mode, ReferenceAreaMode)))
        card.add(labeled_row("Reference area", self.ref_mode,
                             "Planform for wings and aircraft; frontal for cars, spheres and rockets."))
        return card

    def _flow_card(self) -> AccordionCard:
        card = AccordionCard("Flow conditions")
        self.cards["flow"] = card
        self.sliders: dict[str, ParamSlider] = {}
        for key in ("velocity", "alpha", "beta", "altitude", "temp_offset", "ground_h"):
            s = ParamSlider(PARAM_LIMITS[key])
            s.valueChanged.connect(lambda _v: self.conditions_changed.emit(self.conditions()))
            s.valueCommitted.connect(lambda _v: self.conditions_committed.emit(self.conditions()))
            self.sliders[key] = s
            card.add(s)
        self.atmo_label = QLabel("")
        self.atmo_label.setProperty("role", "muted")
        self.atmo_label.setFont(mono_font(11))
        self.atmo_label.setWordWrap(True)
        card.add(self.atmo_label)
        self.conditions_changed.connect(self._update_atmo_label)
        self._update_atmo_label(self.conditions())
        return card

    def _solver_card(self) -> AccordionCard:
        card = AccordionCard("Solver")
        self.cards["solver"] = card
        self.engine = SegmentedControl(["Instant", "OpenFOAM"],
                                       ["Built-in engine: results in about a second on any computer",
                                        "High-fidelity RANS k-ω SST (minutes to hours)"])
        self.engine.changed.connect(lambda i: self.engine_changed.emit(self.selected_engine()))
        card.add(self.engine)
        self.quality = QComboBox()
        for q in GridQuality:
            self.quality.addItem(q.value, q)
        self.quality.setCurrentIndex(1)
        card.add(labeled_row("Resolution", self.quality, "Coarse is fastest; Fine shows more detail."))
        self.of_box = QWidget()
        ob = QVBoxLayout(self.of_box)
        ob.setContentsMargins(0, 0, 0, 0)
        self.of_iters = ParamSlider(PARAM_LIMITS["openfoam_iterations"])
        ob.addWidget(self.of_iters)
        self.of_procs = QSpinBox()
        self.of_procs.setRange(1, 64)
        # Default: the physical cores minus one (logical / 2 with SMT), so the solve is fast and the
        # desktop stays responsive. Falls back to a serial run automatically when MPI is missing.
        logical = os.cpu_count() or 2
        self.of_procs.setValue(max(1, min(logical // 2 - 1, 16)) if logical >= 4 else 1)
        ob.addWidget(labeled_row("CPU cores", self.of_procs,
                                 "MPI ranks for OpenFOAM (snappyHexMesh still meshes on one core). "
                                 "OpenFOAM itself runs on the CPU only; the GPU is used by the Labs."))
        self.of_flow = QComboBox()
        self.of_flow.addItem("Auto (compressible from Mach 0.3)", "auto")
        self.of_flow.addItem("Incompressible (simpleFoam)", "incompressible")
        self.of_flow.addItem("Compressible air (rhoSimpleFoam / fluid)", "compressible")
        ob.addWidget(labeled_row("Flow model", self.of_flow,
                                 "Compressible solves density and temperature too (perfect-gas air): needed above "
                                 "about Mach 0.3 for accurate pressures; slices can then show the local Mach number."))
        self.of_run = QComboBox()
        self.of_run.addItem("Steady (RANS)", "steady")
        self.of_run.addItem("Transient (unsteady, PIMPLE)", "transient")
        ob.addWidget(labeled_row("Run type", self.of_run,
                                 "Transient resolves unsteady wakes and vortex shedding in time and saves frames to "
                                 "play back (Simulation ▸ Play transient result). Incompressible only."))
        self.of_ctimes = QDoubleSpinBox()
        self.of_ctimes.setRange(2.0, 200.0)
        self.of_ctimes.setValue(12.0)
        self.of_ctimes.setSuffix(" t*")
        self.of_frames = QSpinBox()
        self.of_frames.setRange(4, 200)
        self.of_frames.setValue(24)
        self.of_frames.setSuffix(" frames")
        trow = QHBoxLayout()
        trow.addWidget(self.of_ctimes)
        trow.addWidget(self.of_frames)
        self.of_trans_box = QWidget()
        self.of_trans_box.setLayout(trow)
        self.of_trans_box.setToolTip("Duration in convective times t* = t·V/L (how many body lengths the air "
                                     "travels) and the number of saved frames")
        ob.addWidget(self.of_trans_box)
        self.of_trans_box.setVisible(False)
        self.of_run.currentIndexChanged.connect(lambda _i: self.of_trans_box.setVisible(
            self.of_run.currentData() == "transient"))
        self.of_layers = QSpinBox()
        self.of_layers.setRange(-1, 15)
        self.of_layers.setValue(-1)
        self.of_layers.setSpecialValueText("Auto")
        ob.addWidget(labeled_row("Boundary layers", self.of_layers,
                                 "Prism-cell layers grown on the body (Auto: 2 / 3 / 4 for Coarse / Medium / Fine). "
                                 "More layers resolve skin friction better; check y+ afterwards (Surface shows: y+)."))
        card.add(self.of_box)
        self.of_box.setVisible(False)
        self.live = ToggleSwitch("Live update", True, "Re-solve automatically with the Instant engine when you release a slider")
        card.add(self.live)
        row = QHBoxLayout()
        self.btn_run = button("Run simulation", "primary", "play", "Solve the flow (Ctrl+R)")
        self.btn_abort = button("Abort", "danger", "stop", "Stop the running simulation (Esc)")
        self.btn_abort.setEnabled(False)
        self.btn_run.clicked.connect(self.run_requested.emit)
        self.btn_abort.clicked.connect(self.abort_requested.emit)
        row.addWidget(self.btn_run, 1)
        row.addWidget(self.btn_abort)
        card.add_layout(row)
        self.engine_note = QLabel("")
        self.engine_note.setProperty("role", "muted")
        self.engine_note.setWordWrap(True)
        card.add(self.engine_note)
        return card

    def _visual_card(self) -> AccordionCard:
        card = AccordionCard("Visualization", expanded=False)
        self.cards["visual"] = card
        d = VisualSettings()
        self.t_stream = ToggleSwitch("Streamlines", d.show_streamlines, "3-D stream tubes released upstream")
        self.t_particles = ToggleSwitch("Smoke", d.show_particles, "Animated glowing particles flowing along the streamlines")
        self.t_cp = ToggleSwitch("Surface Cp", d.show_cp,
                                 "Colour the model by pressure coefficient: red = high pressure, blue = suction")
        self.t_slice = ToggleSwitch("Slice", d.show_slice, "A 2-D cut through the flow field")
        self.t_vectors = ToggleSwitch("Arrows", d.show_vectors, "Velocity arrows on the slice plane showing flow direction")
        self.t_floor = ToggleSwitch("Floor grid", d.show_floor, "Wind-tunnel floor grid under the model")
        self.t_seeds = ToggleSwitch("Seed rake", d.show_seed_rake, "Show where the streamlines start")
        self.t_oil = ToggleSwitch("Oil flow", d.show_oilflow,
                                  "Surface oil-flow streaks: the direction of the air right at the skin - shows "
                                  "attachment and separation lines and the tip roll-up")
        self.t_ao = ToggleSwitch("Shadows", d.ambient_occlusion,
                                 "Ambient occlusion: soft contact shadows (needs a recent GPU driver; paused while animating)")
        grid = QGridLayout()
        for i, t in enumerate((self.t_stream, self.t_particles, self.t_cp, self.t_slice, self.t_vectors,
                               self.t_floor, self.t_seeds, self.t_ao, self.t_oil)):
            grid.addWidget(t, i // 2, i % 2)
            t.toggled.connect(self._visual_changed)
        card.add_layout(grid)
        self.probe = SegmentedControl(["Off", "Point", "Line", "Seed"],
                                      ["Normal mouse: rotate / zoom the view",
                                       "Click anywhere on the slice, a streamline or the model to read the local "
                                       "speed, Cp, total pressure and vorticity",
                                       "Click two points to plot the flow along the line between them",
                                       "Click on the slice or the model to release a bundle of streamlines from "
                                       "that spot (e.g. at a wing tip to see the tip vortex)"])
        self.probe.changed.connect(lambda i: self.probe_mode_changed.emit(("off", "point", "line", "seed")[i]))
        card.add(labeled_row("Mouse tool", self.probe, "Probe the flow or place streamline seeds with the mouse"))
        self.btn_clear_seeds = button("Clear seeds", "ghost", "", "Remove the streamlines placed with the Seed tool")
        self.btn_clear_seeds.clicked.connect(self.clear_seeds_requested.emit)
        card.add(self.btn_clear_seeds)
        self.cmap = QComboBox()
        for c in DEFAULT_COLORMAPS:
            self.cmap.addItem(c)
        self.cmap.currentIndexChanged.connect(self._visual_changed)
        card.add(labeled_row("Colour map", self.cmap))
        self.t_lock = ToggleSwitch("Lock colour ranges", d.lock_ranges,
                                   "Keep the current colour scales for new results and settings, so two runs (e.g. "
                                   "two angles of attack) are compared on the same scale")
        self.t_lock.toggled.connect(self._visual_changed)
        card.add(self.t_lock)
        self.surface_scalar = QComboBox()
        for key, label in (("Cp", "Pressure Cp"), ("Cf", "Skin friction Cf"), ("yPlus", "y+ (wall mesh)")):
            self.surface_scalar.addItem(label, key)
        self.surface_scalar.setToolTip("What colours the model surface. Skin friction and y+ come from OpenFOAM "
                                       "results; y+ between 30 and 300 suits the wall functions used.")
        self.surface_scalar.currentIndexChanged.connect(self._visual_changed)
        card.add(labeled_row("Surface shows", self.surface_scalar))
        self.stream_scalar = QComboBox()
        for f in ScalarField:
            self.stream_scalar.addItem(f.value, f)
        self.stream_scalar.currentIndexChanged.connect(self._visual_changed)
        card.add(labeled_row("Colour streamlines by", self.stream_scalar))
        card.add(make_section_label("Seed rake"))
        self.seed_shape = SegmentedControl([s.value for s in SeedShape])
        self.seed_shape.set_index(list(SeedShape).index(d.seed_shape))
        self.seed_shape.changed.connect(lambda _i: self._visual_changed())
        card.add(self.seed_shape)
        self.vs: dict[str, ParamSlider] = {}
        for key in ("streamline_count", "tube_radius", "seed_width", "seed_height", "seed_distance"):
            s = ParamSlider(PARAM_LIMITS[key])
            s.valueCommitted.connect(self._visual_changed)    # rebuild on release, not every drag tick
            self.vs[key] = s
            card.add(s)
        card.add(make_section_label("Slice"))
        self.slice_axis = SegmentedControl([f"⟂ {a.value}" for a in SliceAxis],
                                           [f"Slice plane normal to the {a.value} axis" for a in SliceAxis])
        self.slice_axis.set_index(list(SliceAxis).index(d.slice_axis))
        self.slice_axis.changed.connect(lambda _i: self._visual_changed())
        card.add(self.slice_axis)
        self.t_three = ToggleSwitch("3 planes", d.slice_three,
                                    "Three orthogonal slices (X, Y and Z) through the slice point at once")
        self.t_three.toggled.connect(self._visual_changed)
        self.btn_sweep_slice = button("Sweep", "ghost", "play",
                                      "Animate the slice plane back and forth through the flow")
        self.btn_sweep_slice.setCheckable(True)
        self.btn_sweep_slice.toggled.connect(self._on_sweep_toggled)
        row_s = QHBoxLayout()
        row_s.addWidget(self.t_three)
        row_s.addWidget(self.btn_sweep_slice)
        card.add_layout(row_s)
        self.slice_scalar = QComboBox()
        for f in ScalarField:
            self.slice_scalar.addItem(f.value, f)
        self.slice_scalar.currentIndexChanged.connect(self._visual_changed)
        card.add(labeled_row("Slice shows", self.slice_scalar))
        for key in ("slice_position", "vector_density", "particle_count", "particle_speed"):
            s = ParamSlider(PARAM_LIMITS[key])
            s.valueCommitted.connect(self._visual_changed)    # rebuild on release, not every drag tick
            self.vs[key] = s
            card.add(s)
        card.add(make_section_label("3-D surfaces"))
        self.t_iso = ToggleSwitch("Iso-surface", d.show_iso,
                                  "Draw a 3-D surface through the flow: vortex cores (tip vortices), a speed "
                                  "shell or the wake")
        self.t_iso.toggled.connect(self._visual_changed)
        self.t_volume = ToggleSwitch("Vortex cloud", d.show_volume,
                                     "Volume-render the vorticity: swirling air glows like smoke (tip and trailing "
                                     "vortices)")
        self.t_volume.toggled.connect(self._visual_changed)
        row_v = QHBoxLayout()
        row_v.addWidget(self.t_iso)
        row_v.addWidget(self.t_volume)
        card.add_layout(row_v)
        self.iso_field = QComboBox()
        for f in IsoField:
            self.iso_field.addItem(f.value, f)
        self.iso_field.currentIndexChanged.connect(self._visual_changed)
        card.add(labeled_row("Surface of", self.iso_field))
        s = ParamSlider(PARAM_LIMITS["iso_level"])
        s.valueCommitted.connect(self._visual_changed)
        self.vs["iso_level"] = s
        card.add(s)
        return card

    def _polar_card(self) -> AccordionCard:
        card = AccordionCard("Polar sweep", expanded=False)
        self.cards["polar"] = card
        self.polar_sliders = {k: ParamSlider(PARAM_LIMITS[k]) for k in ("polar_start", "polar_stop", "polar_step")}
        for s in self.polar_sliders.values():
            card.add(s)
        self.btn_sweep = button("Run polar sweep", "primary", "chart",
                                "Compute CL, CD, CM and L/D across the angle range")
        self.btn_sweep.clicked.connect(lambda: self.sweep_requested.emit(
            self.polar_sliders["polar_start"].value(), self.polar_sliders["polar_stop"].value(),
            self.polar_sliders["polar_step"].value()))
        card.add(self.btn_sweep)
        self.polar_summary = QLabel("No polar yet.")
        self.polar_summary.setProperty("role", "muted")
        self.polar_summary.setFont(mono_font(11))
        self.polar_summary.setWordWrap(True)
        card.add(self.polar_summary)
        return card

    def _export_card(self) -> AccordionCard:
        card = AccordionCard("Export", expanded=False)
        self.cards["export"] = card
        grid = QGridLayout()
        for i, (text, kind, icon, tip) in enumerate((
                ("Screenshot", "screenshot", "camera", "Save the 3-D view as a 4K PNG (Ctrl+S)"),
                ("GIF", "gif", "play", "Record a rotating / flowing animation"),
                ("Polar CSV", "csv", "export", "Save polar data as a spreadsheet-friendly CSV"),
                ("Report", "report", "export", "Styled HTML engineering report with snapshots and charts (Ctrl+E)"))):
            b = button(text, "", icon, tip)
            b.clicked.connect(lambda _=False, k=kind: self.export_requested.emit(k))
            grid.addWidget(b, i // 2, i % 2)
        card.add_layout(grid)
        return card

    # -- behaviour -----------------------------------------------------------------
    def _preset_ui(self) -> None:
        p = combo_enum(self.preset, GeometryPreset)
        self.custom_box.setVisible(p is GeometryPreset.CUSTOM_NACA)
        self.tail_row.setVisible(p is GeometryPreset.SPHERE)

    def _check_naca(self, text: str) -> None:
        naca, res = validate_naca_code(text)
        if naca is None:
            self.naca_status.setText(res.summary(1).lstrip("- "))
            set_property(self.naca_status, "level", "danger")
            self.btn_generate.setEnabled(combo_enum(self.preset, GeometryPreset) is not GeometryPreset.CUSTOM_NACA)
        else:
            self.naca_status.setText(f"Camber {naca.max_camber * 100:.0f}% at {naca.camber_position * 10:.0f}/10 · "
                                     f"thickness {naca.thickness * 100:.0f}%")
            set_property(self.naca_status, "level", "")
            self.btn_generate.setEnabled(True)

    def _emit_preset(self) -> None:
        p = combo_enum(self.preset, GeometryPreset)
        code = self.naca_code.text().strip()
        if p is GeometryPreset.CUSTOM_NACA:
            naca, _ = validate_naca_code(code)
            if naca is None:
                self._check_naca(code)
                return
        self.preset_requested.emit(p, code, self.span.value(), self.chord.value(), self.tail.value())

    def _update_atmo_label(self, cond: FlowConditions) -> None:
        atm = cond.atmosphere
        t_c = atm.temperature_k - 273.15
        temp = f"{t_c:.1f} °C" if units.system() == "SI" else f"{t_c * 1.8 + 32.0:.1f} °F"
        dens = units.fmt(atm.density, "density", 4 if units.system() == "SI" else 6)
        self.atmo_label.setText(f"ρ {dens} · T {temp} · Mach {cond.mach:.3f}")

    def apply_units(self, _system: str = "") -> None:
        """Show the flow-condition sliders in the current unit system (values stay the same)."""
        for key, qty, dec_si, dec_imp in (("velocity", "speed", 1, 1), ("altitude", "altitude", 0, 0),
                                          ("temp_offset", "temp_delta", 1, 1)):
            s = self.sliders.get(key)
            if s is not None:
                s.set_display_unit(units.scale(qty), units.unit(qty), dec_si if units.system() == "SI" else dec_imp)
        self._update_atmo_label(self.conditions())

    def _visual_changed(self, *_args: object) -> None:
        self._visual_timer.start()

    # -- getters / setters ------------------------------------------------------------
    def conditions(self) -> FlowConditions:
        """Current flow conditions."""
        s = self.sliders
        return FlowConditions(velocity_ms=s["velocity"].value(), alpha_deg=s["alpha"].value(),
                              beta_deg=s["beta"].value(), altitude_m=s["altitude"].value(),
                              temperature_offset_k=s["temp_offset"].value(),
                              ground_h_over_b=s["ground_h"].value())

    def set_conditions(self, cond: FlowConditions, commit: bool = True) -> None:
        """Move every flow slider to ``cond`` (e.g. from the run history); optionally trigger a re-solve."""
        for key, value in (("velocity", cond.velocity_ms), ("alpha", cond.alpha_deg), ("beta", cond.beta_deg),
                           ("altitude", cond.altitude_m), ("temp_offset", cond.temperature_offset_k),
                           ("ground_h", getattr(cond, "ground_h_over_b", 0.0))):
            self.sliders[key].set_value(value)
        self.conditions_changed.emit(self.conditions())
        if commit:
            self.conditions_committed.emit(self.conditions())

    def set_alpha(self, alpha_deg: float) -> None:
        """Set the angle of attack (without triggering a solve)."""
        self.sliders["alpha"].set_value(alpha_deg)

    def selected_engine(self) -> SolverEngine:
        """Engine chosen in the segmented control."""
        return SolverEngine.OPENFOAM if self.engine.index() == 1 else SolverEngine.INSTANT

    def set_engine(self, engine: SolverEngine) -> None:
        """Select an engine without emitting."""
        self.engine.set_index(1 if engine == SolverEngine.OPENFOAM else 0)
        self.of_box.setVisible(engine == SolverEngine.OPENFOAM)
        self.live.setEnabled(engine == SolverEngine.INSTANT)
        self.engine_note.setText("" if engine == SolverEngine.INSTANT else
                                 "OpenFOAM runs can take minutes or longer. Progress and residuals appear on the right.")

    def set_openfoam_available(self, available: bool, detail: str = "") -> None:
        """Enable/disable the OpenFOAM segment."""
        self.engine.set_enabled_at(1, available, "High-fidelity RANS k-ω SST" if available else
                                   (detail or "OpenFOAM not detected — click the status pill for the setup guide"))
        if not available and self.selected_engine() == SolverEngine.OPENFOAM:
            self.set_engine(SolverEngine.INSTANT)

    def quality_value(self) -> GridQuality:
        """Selected resolution preset."""
        return combo_enum(self.quality, GridQuality)

    def openfoam_iterations(self) -> int:
        """Max OpenFOAM iterations."""
        return int(self.of_iters.value())

    def openfoam_compressible(self) -> bool:
        """Whether OpenFOAM should solve compressible air for the current conditions."""
        mode = self.of_flow.currentData()
        return mode == "compressible" or (mode == "auto" and self.conditions().mach >= 0.3)

    def openfoam_transient(self) -> tuple[bool, float, int]:
        """(transient?, duration in convective times, number of frames)."""
        return (self.of_run.currentData() == "transient", float(self.of_ctimes.value()), int(self.of_frames.value()))

    def openfoam_layers(self) -> Optional[int]:
        """Requested prism layers (None: the grid preset's number)."""
        v = int(self.of_layers.value())
        return None if v < 0 else v

    def openfoam_procs(self) -> int:
        """Requested MPI ranks."""
        return int(self.of_procs.value())

    def live_update(self) -> bool:
        """Whether slider releases should re-solve with the Instant engine."""
        return self.live.isChecked() and self.selected_engine() == SolverEngine.INSTANT

    def set_busy(self, busy: bool, sweeping: bool = False) -> None:
        """Reflect a running task in the buttons."""
        self.btn_run.setEnabled(not busy)
        self.btn_sweep.setEnabled(not busy)
        self.btn_abort.setEnabled(busy)
        self.btn_run.setText("Solving…" if busy and not sweeping else "Run simulation")
        self.btn_sweep.setText("Sweeping…" if busy and sweeping else "Run polar sweep")

    def set_reference_mode(self, mode: ReferenceAreaMode) -> None:
        """Select the reference-area mode without emitting."""
        self.ref_mode.blockSignals(True)
        self.ref_mode.setCurrentIndex(list(ReferenceAreaMode).index(mode))
        self.ref_mode.blockSignals(False)

    def set_polar_summary(self, text: str) -> None:
        """Show the polar summary line."""
        self.polar_summary.setText(text)

    def visual_settings(self) -> VisualSettings:
        """Collect the visualisation options."""
        v = self.vs
        return VisualSettings(
            show_geometry=True, show_cp=self.t_cp.isChecked(), show_streamlines=self.t_stream.isChecked(),
            show_particles=self.t_particles.isChecked(), show_slice=self.t_slice.isChecked(),
            show_vectors=self.t_vectors.isChecked(), show_floor=self.t_floor.isChecked(),
            show_seed_rake=self.t_seeds.isChecked(), colormap=self.cmap.currentText(),
            streamline_scalar=combo_enum(self.stream_scalar, ScalarField),
            streamline_count=int(v["streamline_count"].value()), tube_radius=v["tube_radius"].value(),
            seed_shape=list(SeedShape)[self.seed_shape.index()], seed_width=v["seed_width"].value(),
            seed_height=v["seed_height"].value(), seed_distance=v["seed_distance"].value(),
            slice_axis=list(SliceAxis)[self.slice_axis.index()], slice_position=v["slice_position"].value(),
            slice_scalar=combo_enum(self.slice_scalar, ScalarField), vector_density=int(v["vector_density"].value()),
            particle_count=int(v["particle_count"].value()), particle_speed=v["particle_speed"].value(),
            ambient_occlusion=self.t_ao.isChecked(), show_iso=self.t_iso.isChecked(),
            show_volume=self.t_volume.isChecked(),
            iso_field=combo_enum(self.iso_field, IsoField) or IsoField.QCRITERION,
            iso_level=v["iso_level"].value(), slice_three=self.t_three.isChecked(),
            show_oilflow=self.t_oil.isChecked(), surface_scalar=self.surface_scalar.currentData() or "Cp",
            lock_ranges=self.t_lock.isChecked())

    slice_sweep_requested = Signal(bool)
    probe_mode_changed = Signal(str)
    clear_seeds_requested = Signal()

    def _on_sweep_toggled(self, on: bool) -> None:
        if on and not self.t_slice.isChecked():
            self.t_slice.setChecked(True)          # sweeping needs the slice
        self.btn_sweep_slice.setText("Stop" if on else "Sweep")
        self.slice_sweep_requested.emit(on)

    def show_slice_position(self, pos: float) -> None:
        """Mirror the sweeping slice position in the slider without triggering a rebuild."""
        self.vs["slice_position"].set_value(pos, emit=False)

    def stop_slice_sweep(self) -> None:
        if self.btn_sweep_slice.isChecked():
            self.btn_sweep_slice.setChecked(False)

    def focus_polar(self) -> None:
        """Expand the polar card (Polar Analysis mode)."""
        self.cards["polar"].set_expanded(True)
        self.cards["visual"].set_expanded(False)
