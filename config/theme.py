"""Visual design language for AeroFlow Studio.

Defines the colour palette, the complete QSS dark-mode stylesheet, generated
raster assets for toggles / checkboxes / arrows, vector-drawn icons, the app
logo, and small widget helpers (section labels, dynamic-property repolish).

Qt Style Sheets have no ``letter-spacing`` property, so uppercase section
headers must be created through :func:`make_section_label`.

Dynamic-property conventions used by the stylesheet
---------------------------------------------------
``role``   : "primary", "danger", "ghost", "segment" (QPushButton);
             "title", "subtitle", "muted", "section", "mono", "metric",
             "metric-caption", "unit", "badge" (QLabel);
             "camera" (QToolButton); "divider" (QFrame)
``level``  : "success", "warning", "danger", "info" (QLabel)
``state``  : "instant", "openfoam", "unavailable" (QPushButton#EnginePill)
``toggle`` : "true" turns a QCheckBox into a pill switch
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Template
from typing import Callable, Final, Optional

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor, QFont, QFontDatabase, QIcon, QImage, QLinearGradient, QPainter,
    QPainterPath, QPalette, QPen, QPixmap, QPolygonF,
)
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from config.settings import get_paths


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Palette:
    """All colours used by the application (single source of truth)."""

    bg: str = "#0B0F19"             # deep space background
    bg_deep: str = "#080B13"        # inputs, console, status bar
    surface: str = "#131A2A"        # elevated panel / card
    surface_alt: str = "#0F1524"
    surface_hover: str = "#1A2340"
    border: str = "#1E293B"
    border_strong: str = "#334155"
    text: str = "#E2E8F0"
    text_muted: str = "#94A3B8"
    text_dim: str = "#64748B"
    on_accent: str = "#04121A"      # text on top of cyan buttons
    cyan: str = "#00E5FF"           # primary accent
    cyan_hi: str = "#5CF2FF"
    sky: str = "#38BDF8"
    indigo: str = "#6366F1"         # secondary accent
    success: str = "#10B981"
    warning: str = "#F59E0B"
    danger: str = "#EF4444"
    viewport_bottom: str = "#0D1321"
    viewport_top: str = "#1B263B"


#: The light theme (Appearance setting; the flow views keep their dark scientific backgrounds).
LIGHT_PALETTE: Final[Palette] = Palette(
    bg="#F4F6FA", bg_deep="#FFFFFF", surface="#FFFFFF", surface_alt="#EEF2F7", surface_hover="#E2E8F0",
    border="#D7DEE8", border_strong="#B6C2D2", text="#0F172A", text_muted="#475569", text_dim="#64748B",
    on_accent="#FFFFFF", cyan="#0891B2", cyan_hi="#06B6D4", sky="#0284C7", indigo="#4F46E5", success="#059669",
    warning="#D97706", danger="#DC2626", viewport_bottom="#C9D3E0", viewport_top="#F1F5F9")


def appearance() -> tuple[str, float]:
    """The saved theme ("dark" / "light") and interface scale (1.0 = 100 %); read before the UI is built."""
    try:
        from PySide6.QtCore import QSettings

        from config.settings import APP_NAME, ORG_NAME

        st = QSettings(ORG_NAME, APP_NAME)
        theme = str(st.value("ui/theme", "dark") or "dark").lower()
        scale = float(st.value("ui/scale", 1.0) or 1.0)
        return ("light" if theme == "light" else "dark"), min(max(scale, 0.75), 2.0)
    except Exception:  # noqa: BLE001 - settings unreadable: defaults
        return "dark", 1.0


IS_LIGHT: Final[bool] = appearance()[0] == "light"
PALETTE: Final[Palette] = LIGHT_PALETTE if IS_LIGHT else Palette()

# --- Viewport / scene colours ---------------------------------------------
CFD_BACKGROUND_BOTTOM: Final[str] = PALETTE.viewport_bottom
CFD_BACKGROUND_TOP: Final[str] = PALETTE.viewport_top

#: Colours for console log levels (keyed by level name).
LOG_LEVEL_COLORS: Final[dict[str, str]] = {
    "DEBUG": PALETTE.text_dim, "INFO": PALETTE.text, "SUCCESS": PALETTE.success,
    "WARNING": PALETTE.warning, "ERROR": PALETTE.danger, "CRITICAL": PALETTE.danger,
}

_LEVEL_COLORS: Final[dict[str, str]] = {
    "success": PALETTE.success, "warning": PALETTE.warning, "danger": PALETTE.danger,
    "info": PALETTE.cyan, "muted": PALETTE.text_muted,
}


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
def hex_to_rgb(color: str) -> tuple[int, int, int]:
    """Convert ``#RGB`` / ``#RRGGBB`` to an ``(r, g, b)`` tuple of 0-255 ints."""
    c = color.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        raise ValueError(f"Invalid hex colour: {color!r}")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def hex_to_rgb01(color: str) -> tuple[float, float, float]:
    """Convert a hex colour to an ``(r, g, b)`` tuple of 0-1 floats (for VTK/PyVista)."""
    r, g, b = hex_to_rgb(color)
    return r / 255.0, g / 255.0, b / 255.0


def rgba(color: str, alpha: float) -> str:
    """Return a QSS ``rgba(r, g, b, a)`` string (alpha in 0-1, emitted as 0-255)."""
    r, g, b = hex_to_rgb(color)
    a = int(round(min(max(alpha, 0.0), 1.0) * 255))
    return f"rgba({r}, {g}, {b}, {a})"


def qcolor(color: str, alpha: float = 1.0) -> QColor:
    """Build a :class:`QColor` from a hex string and 0-1 alpha."""
    r, g, b = hex_to_rgb(color)
    return QColor(r, g, b, int(round(min(max(alpha, 0.0), 1.0) * 255)))


def lerp_color(c1: str, c2: str, t: float) -> str:
    """Linearly interpolate between two hex colours (``t`` clamped to 0-1)."""
    t = min(max(t, 0.0), 1.0)
    a, b = hex_to_rgb(c1), hex_to_rgb(c2)
    return "#{:02X}{:02X}{:02X}".format(*(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3)))


def status_color(level: str) -> str:
    """Map a semantic level ("success", "warning", "danger", "info", "muted") to a hex colour."""
    return _LEVEL_COLORS.get(level, PALETTE.text_muted)


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
UI_FONT_CANDIDATES: Final[tuple[str, ...]] = (
    "Inter", "Segoe UI", "SF Pro Display", ".AppleSystemUIFont", "Helvetica Neue",
    "Roboto", "Ubuntu", "Arial",
)
MONO_FONT_CANDIDATES: Final[tuple[str, ...]] = (
    "JetBrains Mono", "Consolas", "SF Mono", "Menlo", "DejaVu Sans Mono", "Courier New",
)


def _resolve_family(candidates: tuple[str, ...]) -> str:
    """Return the first installed family from ``candidates`` (else the last one)."""
    installed = set(QFontDatabase.families())
    for name in candidates:
        if name in installed:
            return name
    return candidates[-1]


def _font_stack(candidates: tuple[str, ...], generic: str) -> str:
    """Build a CSS font-family list with a generic fallback."""
    return ", ".join(f'"{c}"' for c in candidates) + f", {generic}"


def ui_font(pixel_size: int = 13, bold: bool = False) -> QFont:
    """Return the UI font (Inter / Segoe UI / SF Pro with fallbacks)."""
    font = QFont(_resolve_family(UI_FONT_CANDIDATES))
    font.setPixelSize(pixel_size)
    font.setBold(bold)
    return font


def mono_font(pixel_size: int = 12, bold: bool = False) -> QFont:
    """Return the fixed-pitch font for numeric readouts so digits never jitter."""
    font = QFont(_resolve_family(MONO_FONT_CANDIDATES))
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setFixedPitch(True)
    font.setPixelSize(pixel_size)
    font.setBold(bold)
    return font


# ---------------------------------------------------------------------------
# Widget helpers
# ---------------------------------------------------------------------------
def repolish(widget: QWidget) -> None:
    """Re-evaluate the stylesheet after a dynamic property changed."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def set_property(widget: QWidget, name: str, value: object) -> None:
    """Set a dynamic property and repolish only when the value actually changed."""
    if widget.property(name) != value:
        widget.setProperty(name, value)
        repolish(widget)


def make_section_label(text: str) -> QLabel:
    """Create an uppercase, letter-spaced, muted section header label."""
    label = QLabel(text.upper())
    label.setProperty("role", "section")
    font = label.font()
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.4)
    label.setFont(font)
    return label


# ---------------------------------------------------------------------------
# Generated raster assets (toggle switch, checkbox, radio, arrows)
# ---------------------------------------------------------------------------
def _save_asset(directory: Path, name: str, width: int, height: int,
                draw: Callable[[QPainter], None]) -> str:
    """Render ``draw`` at 1x and 2x (``name@2x.png``) and return the 1x posix path."""
    for scale, suffix in ((1, ""), (2, "@2x")):
        image = QImage(width * scale, height * scale, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.scale(scale, scale)
        draw(painter)
        painter.end()
        image.save(str(directory / f"{name}{suffix}.png"))
    return (directory / f"{name}.png").as_posix()


def generate_assets() -> dict[str, str]:
    """Create the PNG assets referenced by the stylesheet and return ``{key: path}``."""
    directory = get_paths().cache / "theme_assets"
    directory.mkdir(parents=True, exist_ok=True)
    p = PALETTE
    assets: dict[str, str] = {}

    def toggle(on: bool) -> Callable[[QPainter], None]:
        def draw(painter: QPainter) -> None:
            w, h = 38.0, 22.0
            track = QRectF(1, 1, w - 2, h - 2)
            if on:
                grad = QLinearGradient(0, 0, w, 0)
                grad.setColorAt(0.0, qcolor(p.cyan))
                grad.setColorAt(1.0, qcolor(p.sky))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(grad)
            else:
                painter.setPen(QPen(qcolor(p.border_strong), 1.2))
                painter.setBrush(qcolor(p.bg_deep))
            painter.drawRoundedRect(track, (h - 2) / 2, (h - 2) / 2)
            d = h - 8.0
            x = w - 4.0 - d if on else 4.0
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(qcolor("#FFFFFF") if on else qcolor(p.text_dim))
            painter.drawEllipse(QRectF(x, 4.0, d, d))
        return draw

    def checkbox(on: bool) -> Callable[[QPainter], None]:
        def draw(painter: QPainter) -> None:
            rect = QRectF(1, 1, 14, 14)
            if on:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(qcolor(p.cyan))
                painter.drawRoundedRect(rect, 4, 4)
                pen = QPen(qcolor(p.on_accent), 2.0)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPolyline(QPolygonF([QPointF(4.2, 8.2), QPointF(6.9, 10.9), QPointF(11.8, 5.2)]))
            else:
                painter.setPen(QPen(qcolor(p.border_strong), 1.2))
                painter.setBrush(qcolor(p.bg_deep))
                painter.drawRoundedRect(rect, 4, 4)
        return draw

    def radio(on: bool) -> Callable[[QPainter], None]:
        def draw(painter: QPainter) -> None:
            painter.setPen(QPen(qcolor(p.cyan if on else p.border_strong), 1.4))
            painter.setBrush(qcolor(p.bg_deep))
            painter.drawEllipse(QRectF(1, 1, 14, 14))
            if on:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(qcolor(p.cyan))
                painter.drawEllipse(QRectF(4.5, 4.5, 7, 7))
        return draw

    def arrow(up: bool) -> Callable[[QPainter], None]:
        def draw(painter: QPainter) -> None:
            pen = QPen(qcolor(p.text_muted), 1.6)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            pts = [(2.0, 6.5), (5.0, 3.5), (8.0, 6.5)] if up else [(2.0, 3.5), (5.0, 6.5), (8.0, 3.5)]
            painter.drawPolyline(QPolygonF([QPointF(x, y) for x, y in pts]))
        return draw

    assets["toggle_off"] = _save_asset(directory, "toggle_off", 38, 22, toggle(False))
    assets["toggle_on"] = _save_asset(directory, "toggle_on", 38, 22, toggle(True))
    assets["check_off"] = _save_asset(directory, "check_off", 16, 16, checkbox(False))
    assets["check_on"] = _save_asset(directory, "check_on", 16, 16, checkbox(True))
    assets["radio_off"] = _save_asset(directory, "radio_off", 16, 16, radio(False))
    assets["radio_on"] = _save_asset(directory, "radio_on", 16, 16, radio(True))
    assets["arrow_up"] = _save_asset(directory, "arrow_up", 10, 10, arrow(True))
    assets["arrow_down"] = _save_asset(directory, "arrow_down", 10, 10, arrow(False))
    return assets


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------
_QSS_TEMPLATE: Final[str] = """
* { outline: none; }
QWidget { color: $text; font-family: $ui_font; font-size: 13px; }
QMainWindow, QDialog, QMessageBox { background: $bg; }
QWidget#Root { background: $bg; }
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }

QToolTip { background: $surface; color: $text; border: 1px solid $cyan_a60;
           padding: 6px 8px; border-radius: 6px; }

/* ---------- Labels ---------- */
QLabel { background: transparent; }
QLabel[role="title"] { font-size: 15px; font-weight: 700; color: $text; }
QLabel[role="subtitle"], QLabel[role="muted"] { color: $text_muted; font-size: 12px; }
QLabel[role="section"] { color: $text_dim; font-size: 10px; font-weight: 700; }
QLabel[role="mono"] { font-family: $mono_font; }
QLabel[role="metric"] { font-family: $mono_font; font-size: 16px; font-weight: 700; color: $text; }
QLabel[role="metric-caption"] { color: $text_dim; font-size: 9px; font-weight: 700; }
QLabel[role="unit"] { background: $bg_deep; border: 1px solid $border; border-radius: 4px;
                      padding: 1px 6px; color: $text_muted; font-size: 10px; font-family: $mono_font; }
QLabel[role="badge"] { border-radius: 9px; padding: 2px 10px; font-size: 11px; font-weight: 700;
                       background: $cyan_a15; color: $cyan; }
QLabel[level="success"] { color: $success; }
QLabel[level="warning"] { color: $warning; }
QLabel[level="danger"] { color: $danger; }
QLabel[level="info"] { color: $cyan; }
QLabel[role="badge"][level="success"] { background: $success_a15; color: $success; }
QLabel[role="badge"][level="warning"] { background: $warning_a15; color: $warning; }
QLabel[role="badge"][level="danger"] { background: $danger_a15; color: $danger; }
QLabel#AppTitle { font-size: 15px; font-weight: 700; color: $text; }

/* ---------- Frames / cards ---------- */
QFrame#TopBar { background: $bg; border: none; border-bottom: 1px solid $border; }
QFrame#Card { background: $surface; border: 1px solid $border; border-radius: 8px; }
QFrame#GlassCard { background: $glass; border: 1px solid $glass_border; border-radius: 10px; }
QFrame#FloatingBar { background: $glass; border: 1px solid $glass_border; border-radius: 20px; }
QFrame#Segmented { background: $surface; border: 1px solid $border; border-radius: 12px; }
QFrame#DropOverlay { border: 2px dashed $cyan; border-radius: 12px; background: $cyan_a10; }
QFrame[role="divider"] { background: $border; border: none; max-height: 1px; min-height: 1px; }

/* ---------- Buttons ---------- */
QPushButton { background: $surface; border: 1px solid $border_strong; border-radius: 6px;
              padding: 7px 14px; color: $text; font-weight: 600; }
QPushButton:hover { background: $surface_hover; border-color: $cyan_a60; }
QPushButton:pressed { background: $cyan_a15; }
QPushButton:focus { border-color: $cyan; }
QPushButton:disabled { color: $text_dim; background: $surface_alt; border-color: $border; }
QPushButton[role="primary"] { color: $on_accent; border: none; font-weight: 700; padding: 9px 18px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 $cyan, stop:1 $sky); }
QPushButton[role="primary"]:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 $cyan_hi, stop:1 $sky); }
QPushButton[role="primary"]:pressed { background: $cyan; }
QPushButton[role="primary"]:disabled { background: $surface_hover; color: $text_dim; }
QPushButton[role="danger"] { background: $danger_a15; border: 1px solid $danger_a60; color: $danger; }
QPushButton[role="danger"]:hover { background: $danger_a25; border-color: $danger; }
QPushButton[role="ghost"] { background: transparent; border: 1px solid transparent; color: $text_muted; }
QPushButton[role="ghost"]:hover { background: $white_a06; color: $text; }
QPushButton[role="segment"] { background: transparent; border: none; border-radius: 9px;
    padding: 8px 18px; color: $text_muted; font-weight: 700; font-size: 12px; }
QPushButton[role="segment"]:hover { color: $text; background: $white_a06; }
QPushButton[role="segment"]:checked { color: $on_accent;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 $cyan, stop:1 $sky); }
QPushButton#EnginePill { border-radius: 15px; padding: 6px 14px; font-size: 11px; font-weight: 700;
                         background: $surface; border: 1px solid $border_strong; color: $text_muted; }
QPushButton#EnginePill[state="instant"] { color: $cyan; border-color: $cyan_a60; background: $cyan_a10; }
QPushButton#EnginePill[state="openfoam"] { color: $success; border-color: $success_a60; background: $success_a15; }
QPushButton#EnginePill[state="unavailable"] { color: $warning; border-color: $warning_a60; background: $warning_a15; }

QToolButton { background: transparent; border: 1px solid transparent; border-radius: 6px;
              padding: 5px; color: $text_muted; }
QToolButton:hover { background: $white_a06; color: $text; }
QToolButton:checked { background: $cyan_a15; border-color: $cyan_a40; color: $cyan; }
QToolButton:pressed { background: $cyan_a25; }
QToolButton::menu-indicator { image: none; }
QToolButton[role="camera"] { padding: 6px 10px; border-radius: 14px; font-size: 11px; font-weight: 600; }
QToolButton#CardToggle { text-align: left; font-size: 11px; font-weight: 700; color: $text_dim;
                         padding: 8px 10px; border-radius: 8px; }
QToolButton#CardToggle:hover { color: $text; background: $white_a06; }

/* ---------- Inputs ---------- */
QLineEdit { background: $bg_deep; border: 1px solid $border; border-radius: 6px; padding: 6px 10px;
            selection-background-color: $cyan_a40; }
QLineEdit:hover { border-color: $border_strong; }
QLineEdit:focus { border-color: $cyan; }
QLineEdit:disabled { color: $text_dim; }
QSpinBox, QDoubleSpinBox { background: $bg_deep; border: 1px solid $border; border-radius: 6px;
    padding: 4px 22px 4px 8px; font-family: $mono_font; font-size: 12px; min-height: 20px;
    selection-background-color: $cyan_a40; }
QSpinBox:hover, QDoubleSpinBox:hover { border-color: $border_strong; }
QSpinBox:focus, QDoubleSpinBox:focus { border-color: $cyan; }
QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right;
    width: 18px; border: none; background: transparent; }
QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-origin: border;
    subcontrol-position: bottom right; width: 18px; border: none; background: transparent; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url("$asset_arrow_up"); width: 8px; height: 8px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url("$asset_arrow_down"); width: 8px; height: 8px; }
QComboBox { background: $bg_deep; border: 1px solid $border; border-radius: 6px;
            padding: 6px 28px 6px 10px; min-height: 20px; }
QComboBox:hover { border-color: $cyan_a60; }
QComboBox:focus, QComboBox:on { border-color: $cyan; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: center right; width: 26px; border: none; }
QComboBox::down-arrow { image: url("$asset_arrow_down"); width: 10px; height: 10px; }
QComboBox QAbstractItemView { background: $surface; border: 1px solid $border_strong; border-radius: 6px;
    selection-background-color: $cyan_a25; selection-color: $text; outline: none; padding: 4px; }

/* ---------- Sliders ---------- */
QSlider::groove:horizontal { height: 4px; background: $border; border-radius: 2px; }
QSlider::sub-page:horizontal { border-radius: 2px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 $indigo, stop:1 $cyan); }
QSlider::add-page:horizontal { background: $border; border-radius: 2px; }
QSlider::handle:horizontal { background: $cyan; border: 3px solid $cyan_a40; background-clip: padding;
    width: 10px; height: 10px; margin: -6px 0; border-radius: 8px; }
QSlider::handle:horizontal:hover { background: $cyan_hi; border-color: $cyan_a60; }
QSlider::handle:horizontal:disabled { background: $text_dim; border-color: $border; }

/* ---------- Check / radio / toggle ---------- */
QCheckBox, QRadioButton { spacing: 8px; color: $text; }
QCheckBox:disabled, QRadioButton:disabled { color: $text_dim; }
QCheckBox::indicator { width: 16px; height: 16px; image: url("$asset_check_off"); }
QCheckBox::indicator:checked { image: url("$asset_check_on"); }
QCheckBox[toggle="true"]::indicator { width: 38px; height: 22px; image: url("$asset_toggle_off"); }
QCheckBox[toggle="true"]::indicator:checked { image: url("$asset_toggle_on"); }
QRadioButton::indicator { width: 16px; height: 16px; image: url("$asset_radio_off"); }
QRadioButton::indicator:checked { image: url("$asset_radio_on"); }

/* ---------- Progress ---------- */
QProgressBar { background: $bg; border: 1px solid $border; border-radius: 5px; text-align: center;
               color: $text; font-size: 10px; font-family: $mono_font; min-height: 10px; max-height: 14px; }
QProgressBar::chunk { border-radius: 4px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 $indigo, stop:1 $cyan); }

/* ---------- Scrollbars ---------- */
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: $border_strong; border-radius: 3px; min-height: 28px; }
QScrollBar::handle:vertical:hover { background: $text_dim; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; background: none; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:horizontal { background: $border_strong; border-radius: 3px; min-width: 28px; }
QScrollBar::handle:horizontal:hover { background: $text_dim; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; background: none; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: none; }

/* ---------- Tabs ---------- */
QTabWidget::pane { border: 1px solid $border; border-radius: 8px; background: $surface; top: -1px; }
QTabBar { background: transparent; }
QTabBar::tab { background: transparent; color: $text_muted; padding: 8px 14px; border: none;
               border-bottom: 2px solid transparent; font-weight: 600; margin-right: 2px; }
QTabBar::tab:selected { color: $cyan; border-bottom: 2px solid $cyan; }
QTabBar::tab:hover:!selected { color: $text; }

/* ---------- Docks / splitters / status bar ---------- */
QDockWidget { color: $text; }
QDockWidget::title { background: $bg; padding: 8px 12px; border-bottom: 1px solid $border;
                     text-align: left; font-weight: 700; }
QSplitter::handle { background: $border; }
QSplitter::handle:hover { background: $cyan_a60; }
QStatusBar { background: $bg_deep; border-top: 1px solid $border; color: $text_muted;
             font-family: $mono_font; font-size: 11px; min-height: 28px; }
QStatusBar::item { border: none; }

/* ---------- Console / text ---------- */
QTextEdit#Console, QPlainTextEdit#Console, QTextEdit, QPlainTextEdit {
    background: $bg_deep; border: 1px solid $border; border-radius: 8px; padding: 6px;
    selection-background-color: $cyan_a40; }
QTextEdit#Console, QPlainTextEdit#Console { font-family: $mono_font; font-size: 11px; }

/* ---------- Menus ---------- */
QMenu { background: $surface; border: 1px solid $border_strong; border-radius: 8px; padding: 6px; }
QMenu::item { padding: 7px 22px 7px 14px; border-radius: 6px; }
QMenu::item:selected { background: $cyan_a15; color: $cyan; }
QMenu::item:disabled { color: $text_dim; }
QMenu::separator { height: 1px; background: $border; margin: 5px 8px; }

/* ---------- Tables / lists ---------- */
QTableWidget, QTableView, QListWidget, QListView { background: $surface;
    alternate-background-color: $surface_alt; border: 1px solid $border; border-radius: 8px;
    gridline-color: $border; selection-background-color: $cyan_a18; selection-color: $text; }
QTableWidget::item, QListWidget::item { padding: 4px 6px; }
QHeaderView::section { background: $bg_deep; color: $text_dim; border: none;
    border-bottom: 1px solid $border; padding: 6px 8px; font-size: 10px; font-weight: 700; }
QGroupBox { border: 1px solid $border; border-radius: 8px; margin-top: 14px; padding: 12px 8px 8px 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: $text_dim;
                   font-size: 10px; font-weight: 700; }
QMessageBox QPushButton { min-width: 80px; }
"""


def build_stylesheet(assets: Optional[dict[str, str]] = None) -> str:
    """Render the complete QSS stylesheet.

    Args:
        assets: Mapping from asset key to PNG path. Generated on demand if omitted.
    """
    if assets is None:
        assets = generate_assets()
    tokens: dict[str, str] = dict(asdict(PALETTE))
    tokens["white"] = "#0F172A" if IS_LIGHT else "#FFFFFF"      # hover tints: dark on light, light on dark
    tokens["ui_font"] = _font_stack(UI_FONT_CANDIDATES, "sans-serif")
    tokens["mono_font"] = _font_stack(MONO_FONT_CANDIDATES, "monospace")
    for name in ("cyan", "indigo", "success", "warning", "danger", "white"):
        for pct in (6, 10, 15, 18, 25, 40, 60):
            tokens[f"{name}_a{pct:02d}"] = rgba(tokens[name], pct / 100.0)
    tokens["glass"] = rgba(PALETTE.surface, 0.72)
    tokens["glass_border"] = rgba("#0F172A" if IS_LIGHT else "#FFFFFF", 0.10)
    for key, path in assets.items():
        tokens[f"asset_{key}"] = path
    return Template(_QSS_TEMPLATE).substitute(tokens)


def build_palette() -> QPalette:
    """Return a dark :class:`QPalette` matching the stylesheet (for native dialogs)."""
    p = PALETTE
    pal = QPalette()
    roles = {
        QPalette.ColorRole.Window: p.bg, QPalette.ColorRole.WindowText: p.text,
        QPalette.ColorRole.Base: p.bg_deep, QPalette.ColorRole.AlternateBase: p.surface,
        QPalette.ColorRole.Text: p.text, QPalette.ColorRole.Button: p.surface,
        QPalette.ColorRole.ButtonText: p.text, QPalette.ColorRole.ToolTipBase: p.surface,
        QPalette.ColorRole.ToolTipText: p.text, QPalette.ColorRole.Highlight: p.cyan,
        QPalette.ColorRole.HighlightedText: p.on_accent, QPalette.ColorRole.PlaceholderText: p.text_dim,
        QPalette.ColorRole.Link: p.cyan,
    }
    for role, color in roles.items():
        pal.setColor(role, qcolor(color))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, qcolor(p.text_dim))
    return pal


def apply_theme(app: QApplication) -> None:
    """Apply Fusion style, dark palette, application font and the QSS stylesheet.

    Must be called after the :class:`QApplication` exists.
    """
    app.setStyle("Fusion")
    app.setPalette(build_palette())
    app.setFont(ui_font(13))
    app.setStyleSheet(build_stylesheet())


# ---------------------------------------------------------------------------
# Vector-drawn icons (24 x 24 design grid)
# ---------------------------------------------------------------------------
Drawer = Callable[[QPainter, QColor], None]


def _stroke(p: QPainter, c: QColor, width: float = 1.8) -> None:
    """Configure ``p`` for round-capped outline drawing."""
    pen = QPen(c, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)


def _fill(p: QPainter, c: QColor) -> None:
    """Configure ``p`` for filled, outline-free drawing."""
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(c)


def _poly(points: list[tuple[float, float]]) -> QPolygonF:
    return QPolygonF([QPointF(x, y) for x, y in points])


def _d_play(p: QPainter, c: QColor) -> None:
    _fill(p, c)
    p.drawPolygon(_poly([(7, 5), (19, 12), (7, 19)]))


def _d_pause(p: QPainter, c: QColor) -> None:
    _fill(p, c)
    p.drawRoundedRect(QRectF(6, 5, 4, 14), 1, 1)
    p.drawRoundedRect(QRectF(14, 5, 4, 14), 1, 1)


def _d_stop(p: QPainter, c: QColor) -> None:
    _fill(p, c)
    p.drawRoundedRect(QRectF(6, 6, 12, 12), 2, 2)


def _d_folder(p: QPainter, c: QColor) -> None:
    _stroke(p, c)
    path = QPainterPath()
    path.moveTo(3, 6)
    for x, y in ((9.5, 6), (11.5, 8.5), (21, 8.5), (21, 19), (3, 19)):
        path.lineTo(x, y)
    path.closeSubpath()
    p.drawPath(path)


def _d_camera(p: QPainter, c: QColor) -> None:
    _stroke(p, c)
    p.drawRoundedRect(QRectF(3, 7.5, 18, 12), 2.5, 2.5)
    p.drawEllipse(QPointF(12, 13.5), 3.4, 3.4)
    p.drawPolyline(_poly([(8.5, 7.5), (9.8, 5), (14.2, 5), (15.5, 7.5)]))


def _d_settings(p: QPainter, c: QColor) -> None:
    _stroke(p, c)
    p.drawEllipse(QPointF(12, 12), 3.0, 3.0)
    p.drawEllipse(QPointF(12, 12), 7.0, 7.0)
    for i in range(8):
        ang = i * math.pi / 4
        p.drawLine(QPointF(12 + 7.0 * math.cos(ang), 12 + 7.0 * math.sin(ang)),
                   QPointF(12 + 9.8 * math.cos(ang), 12 + 9.8 * math.sin(ang)))


def _d_help(p: QPainter, c: QColor) -> None:
    _stroke(p, c)
    p.drawEllipse(QPointF(12, 12), 9.2, 9.2)
    font = QFont()
    font.setPixelSize(13)
    font.setBold(True)
    p.setFont(font)
    p.drawText(QRectF(0, 0, 24, 24.5), Qt.AlignmentFlag.AlignCenter, "?")


def _d_export(p: QPainter, c: QColor) -> None:
    _stroke(p, c)
    p.drawLine(QPointF(12, 15), QPointF(12, 4))
    p.drawPolyline(_poly([(8, 8), (12, 4), (16, 8)]))
    p.drawPolyline(_poly([(5, 13), (5, 19), (19, 19), (19, 13)]))


def _d_reset(p: QPainter, c: QColor) -> None:
    _stroke(p, c)
    p.drawArc(QRectF(5, 5, 14, 14), 40 * 16, 280 * 16)
    p.drawPolyline(_poly([(18.5, 5.2), (18.6, 9.0), (14.8, 8.2)]))


def _d_iso(p: QPainter, c: QColor) -> None:
    _stroke(p, c, 1.6)
    p.drawPolygon(_poly([(12, 3), (20, 7.5), (20, 16.5), (12, 21), (4, 16.5), (4, 7.5)]))
    p.drawLine(QPointF(12, 12), QPointF(12, 21))
    p.drawLine(QPointF(12, 12), QPointF(4, 7.5))
    p.drawLine(QPointF(12, 12), QPointF(20, 7.5))


def _cube_face(p: QPainter, c: QColor, face: str) -> None:
    _stroke(p, c, 1.6)
    p.drawRoundedRect(QRectF(5, 5, 14, 14), 2, 2)
    hl = QColor(c)
    hl.setAlpha(110)
    _fill(p, hl)
    if face == "front":
        p.drawRect(QRectF(5, 5, 14, 14))
    elif face == "top":
        p.drawRect(QRectF(5, 5, 14, 5))
    else:
        p.drawRect(QRectF(14, 5, 5, 14))


def _d_front(p: QPainter, c: QColor) -> None:
    _cube_face(p, c, "front")


def _d_top(p: QPainter, c: QColor) -> None:
    _cube_face(p, c, "top")


def _d_side(p: QPainter, c: QColor) -> None:
    _cube_face(p, c, "side")


def _d_chevron_down(p: QPainter, c: QColor) -> None:
    _stroke(p, c, 2.0)
    p.drawPolyline(_poly([(6, 9), (12, 15), (18, 9)]))


def _d_chevron_right(p: QPainter, c: QColor) -> None:
    _stroke(p, c, 2.0)
    p.drawPolyline(_poly([(9, 6), (15, 12), (9, 18)]))


def _d_wind(p: QPainter, c: QColor) -> None:
    _stroke(p, c)
    for y, x_end, curl in ((9.0, 14.0, -4.5), (13.5, 19.0, 0.0), (17.5, 11.0, 4.0)):
        path = QPainterPath()
        path.moveTo(3, y)
        path.lineTo(x_end, y)
        if curl:
            path.cubicTo(x_end + 3.5, y, x_end + 3.5, y + curl, x_end + 0.5, y + curl)
        p.drawPath(path)


def _d_plane(p: QPainter, c: QColor) -> None:
    _fill(p, c)
    p.drawPolygon(_poly([
        (12, 2), (13.8, 9), (22, 14), (22, 16.2), (13.5, 14), (13, 19.5), (16, 21.5), (16, 22.8),
        (12, 21.7), (8, 22.8), (8, 21.5), (11, 19.5), (10.5, 14), (2, 16.2), (2, 14), (10.2, 9),
    ]))


def _d_chart(p: QPainter, c: QColor) -> None:
    _stroke(p, c)
    p.drawPolyline(_poly([(4, 4), (4, 20), (20, 20)]))
    p.drawPolyline(_poly([(7, 15), (11, 11), (14, 13), (19, 6)]))


def _d_check(p: QPainter, c: QColor) -> None:
    _stroke(p, c, 2.2)
    p.drawPolyline(_poly([(5, 12.5), (10, 17.5), (19, 7)]))


def _d_warning(p: QPainter, c: QColor) -> None:
    _stroke(p, c)
    p.drawPolygon(_poly([(12, 3), (22, 20), (2, 20)]))
    p.drawLine(QPointF(12, 9.5), QPointF(12, 14))
    _fill(p, c)
    p.drawEllipse(QPointF(12, 17.2), 0.9, 0.9)


def _d_close(p: QPainter, c: QColor) -> None:
    _stroke(p, c, 2.0)
    p.drawLine(QPointF(6, 6), QPointF(18, 18))
    p.drawLine(QPointF(18, 6), QPointF(6, 18))


def _d_info(p: QPainter, c: QColor) -> None:
    _stroke(p, c)
    p.drawEllipse(QPointF(12, 12), 9.2, 9.2)
    p.drawLine(QPointF(12, 11), QPointF(12, 16.5))
    _fill(p, c)
    p.drawEllipse(QPointF(12, 7.6), 0.9, 0.9)


def _d_grid(p: QPainter, c: QColor) -> None:
    _stroke(p, c, 1.5)
    p.drawRoundedRect(QRectF(4, 4, 16, 16), 2, 2)
    for v in (9.33, 14.67):
        p.drawLine(QPointF(v, 4), QPointF(v, 20))
        p.drawLine(QPointF(4, v), QPointF(20, v))


def _d_fallback(p: QPainter, c: QColor) -> None:
    _stroke(p, c)
    p.drawEllipse(QPointF(12, 12), 8, 8)


_ICON_DRAWERS: Final[dict[str, Drawer]] = {
    "play": _d_play, "pause": _d_pause, "stop": _d_stop, "folder": _d_folder, "camera": _d_camera,
    "settings": _d_settings, "help": _d_help, "export": _d_export, "reset": _d_reset, "iso": _d_iso,
    "front": _d_front, "top": _d_top, "side": _d_side, "chevron_down": _d_chevron_down,
    "chevron_right": _d_chevron_right, "wind": _d_wind, "plane": _d_plane, "chart": _d_chart,
    "check": _d_check, "warning": _d_warning, "close": _d_close, "info": _d_info, "grid": _d_grid,
}


def available_icons() -> tuple[str, ...]:
    """Names accepted by :func:`make_icon`."""
    return tuple(_ICON_DRAWERS)


def _render_icon_pixmap(name: str, size: int, color: str) -> QPixmap:
    """Render one icon at 2x device resolution."""
    scale = 2.0
    pix = QPixmap(int(size * scale), int(size * scale))
    pix.setDevicePixelRatio(scale)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(size / 24.0, size / 24.0)
    _ICON_DRAWERS.get(name, _d_fallback)(painter, qcolor(color))
    painter.end()
    return pix


def make_icon(name: str, size: int = 20, color: Optional[str] = None,
              active_color: Optional[str] = None) -> QIcon:
    """Create a crisp vector :class:`QIcon`.

    Args:
        name: One of :func:`available_icons` (unknown names render a circle).
        size: Logical pixel size.
        color: Normal colour (default: muted text).
        active_color: Colour for hover and checked states (default: cyan).
    """
    normal = color or PALETTE.text_muted
    active = active_color or PALETTE.cyan
    icon = QIcon()
    icon.addPixmap(_render_icon_pixmap(name, size, normal), QIcon.Mode.Normal, QIcon.State.Off)
    hot = _render_icon_pixmap(name, size, active)
    icon.addPixmap(hot, QIcon.Mode.Active, QIcon.State.Off)
    icon.addPixmap(hot, QIcon.Mode.Normal, QIcon.State.On)
    icon.addPixmap(hot, QIcon.Mode.Active, QIcon.State.On)
    return icon


# ---------------------------------------------------------------------------
# Logo
# ---------------------------------------------------------------------------
def make_logo_pixmap(size: int = 32) -> QPixmap:
    """Draw the AeroFlow logo: a gradient tile with an airfoil and streamlines."""
    scale = 2.0
    pix = QPixmap(int(size * scale), int(size * scale))
    pix.setDevicePixelRatio(scale)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(size / 32.0, size / 32.0)

    grad = QLinearGradient(0, 0, 32, 32)
    grad.setColorAt(0.0, qcolor(PALETTE.indigo))
    grad.setColorAt(1.0, qcolor(PALETTE.cyan))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(grad)
    painter.drawRoundedRect(QRectF(1, 1, 30, 30), 8, 8)

    pen = QPen(QColor(255, 255, 255, 175), 1.5)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for start, c1, c2, end in (
        ((3.5, 9.0), (10.0, 8.0), (18.0, 5.5), (28.5, 9.5)),
        ((3.5, 13.0), (9.0, 12.5), (15.0, 9.5), (28.5, 13.0)),
        ((3.5, 22.0), (12.0, 23.5), (20.0, 23.0), (28.5, 21.0)),
        ((3.5, 26.0), (12.0, 27.0), (20.0, 26.5), (28.5, 25.5)),
    ):
        path = QPainterPath()
        path.moveTo(*start)
        path.cubicTo(QPointF(*c1), QPointF(*c2), QPointF(*end))
        painter.drawPath(path)

    wing = QPainterPath()
    wing.moveTo(8, 17)
    wing.cubicTo(10, 11, 20, 11, 25, 16)
    wing.cubicTo(20, 18.5, 12, 19.5, 8, 17)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(qcolor("#06101C"))
    painter.drawPath(wing)
    painter.end()
    return pix


def make_logo_icon(size: int = 32) -> QIcon:
    """Return the logo as a :class:`QIcon` (window / taskbar icon)."""
    return QIcon(make_logo_pixmap(size))
