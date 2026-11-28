"""Preferences: appearance (theme, accent colour, interface size), start-up and compute device."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSettings, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QDialog, QFrame, QHBoxLayout, QLabel, QListWidget, QPushButton, QStackedWidget,
    QVBoxLayout, QWidget,
)

from config.settings import APP_NAME, ORG_NAME
from config.theme import ACCENT, ACCENTS, IS_LIGHT, accent_name, appearance, make_icon, make_section_label
from ui.cfd_panel import ToggleSwitch

SCALES = (0.85, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75)


def _swatch(color: str, size: int = 26) -> QIcon:
    pix = QPixmap(size * 2, size * 2)
    pix.setDevicePixelRatio(2.0)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(color))
    p.drawEllipse(2, 2, size - 4, size - 4)
    p.end()
    return QIcon(pix)


def _theme_preview(light: bool, accent: str) -> QIcon:
    """A miniature window in the given theme (for the theme picker)."""
    w, h = 132, 82
    pix = QPixmap(w * 2, h * 2)
    pix.setDevicePixelRatio(2.0)
    pix.fill(Qt.GlobalColor.transparent)
    bg, surf, line, text = (("#F4F5F7", "#FFFFFF", "#E1E4EA", "#C9CED7") if light else
                            ("#101216", "#16191E", "#252A31", "#3A404A"))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QColor(line))
    p.setBrush(QColor(bg))
    p.drawRoundedRect(1, 1, w - 2, h - 2, 8, 8)
    p.setBrush(QColor(surf))
    p.drawRoundedRect(8, 14, 34, h - 22, 4, 4)
    p.drawRoundedRect(48, 14, w - 56, h - 22, 4, 4)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(accent))
    p.drawRoundedRect(12, 20, 26, 7, 2, 2)
    p.setBrush(QColor(text))
    for k in range(4):
        p.drawRoundedRect(12, 33 + 9 * k, 22, 4, 2, 2)
    p.setPen(QColor(accent))
    for k in range(3):
        y = 50 + 6 * k
        p.drawLine(54, y, w - 14, y - 12 + 4 * k)
    p.end()
    return QIcon(pix)


class PreferencesDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Preferences — {APP_NAME}")
        self.resize(760, 520)
        st = QSettings(ORG_NAME, APP_NAME)
        theme, scale = appearance()
        self._initial = (theme, accent_name(), scale)
        self.nav = QListWidget()
        self.nav.setObjectName("NavList")
        self.nav.setFixedWidth(190)
        self.pages = QStackedWidget()
        for name, icon, page in (("Appearance", "sun", self._appearance(theme, scale)),
                                 ("Start-up", "home", self._startup(st)),
                                 ("Performance", "bolt", self._performance())):
            self.nav.addItem(name)
            self.nav.item(self.nav.count() - 1).setIcon(make_icon(icon, 16))
            self.pages.addWidget(page)
        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.nav.setCurrentRow(0)
        body = QHBoxLayout()
        body.setSpacing(0)
        side = QFrame()
        side.setObjectName("Card")
        sl = QVBoxLayout(side)
        sl.setContentsMargins(6, 10, 6, 10)
        sl.addWidget(self.nav)
        body.addWidget(side)
        body.addSpacing(16)
        body.addWidget(self.pages, 1)
        self.note = QLabel("")
        self.note.setProperty("role", "muted")
        self.btn_restart = QPushButton("Save and restart")
        self.btn_restart.setProperty("role", "primary")
        self.btn_save = QPushButton("Save")
        cancel = QPushButton("Cancel")
        cancel.setProperty("role", "ghost")
        cancel.clicked.connect(self.reject)
        self.btn_save.clicked.connect(lambda: self._save(False))
        self.btn_restart.clicked.connect(lambda: self._save(True))
        foot = QHBoxLayout()
        foot.addWidget(self.note, 1)
        foot.addWidget(cancel)
        foot.addWidget(self.btn_save)
        foot.addWidget(self.btn_restart)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 18, 18, 14)
        lay.addLayout(body, 1)
        lay.addSpacing(10)
        lay.addLayout(foot)
        self._restart = False
        self._update_note()

    # -- pages ------------------------------------------------------------------------------------
    def _page(self, title: str, subtitle: str) -> tuple[QWidget, QVBoxLayout]:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(10)
        t = QLabel(title)
        t.setProperty("role", "headline")
        s = QLabel(subtitle)
        s.setProperty("role", "muted")
        s.setWordWrap(True)
        lay.addWidget(t)
        lay.addWidget(s)
        lay.addSpacing(6)
        return w, lay

    def _appearance(self, theme: str, scale: float) -> QWidget:
        w, lay = self._page("Appearance", "Theme, accent colour and interface size. Theme and accent apply after a "
                                          "restart; the flow views keep their scientific dark backgrounds.")
        lay.addWidget(make_section_label("Theme"))
        row = QHBoxLayout()
        self.theme_group = QButtonGroup(self)
        accent_now = ACCENTS[ACCENT][1 if IS_LIGHT else 0]
        for k, (key, label) in enumerate((("dark", "Dark"), ("light", "Light"))):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setIcon(_theme_preview(key == "light", accent_now))
            b.setIconSize(QSize(132, 82))
            b.setStyleSheet("QPushButton { padding: 10px; } ")
            b.setProperty("theme", key)
            b.setChecked(key == theme)
            self.theme_group.addButton(b, k)
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addSpacing(6)
        lay.addWidget(make_section_label("Accent colour"))
        arow = QHBoxLayout()
        arow.setSpacing(6)
        self.accent_group = QButtonGroup(self)
        current = accent_name()
        for k, (name, (dark, light)) in enumerate(ACCENTS.items()):
            b = QPushButton(name)
            b.setCheckable(True)
            b.setIcon(_swatch(light if IS_LIGHT else dark))
            b.setIconSize(QSize(18, 18))
            b.setChecked(name == current)
            b.setProperty("accent", name)
            self.accent_group.addButton(b, k)
            arow.addWidget(b)
        arow.addStretch(1)
        lay.addLayout(arow)
        lay.addSpacing(6)
        lay.addWidget(make_section_label("Interface size"))
        self.scale = QComboBox()
        for s in SCALES:
            self.scale.addItem(f"{s * 100:.0f} %" + ("  (default)" if s == 1.0 else ""), s)
        self.scale.setCurrentIndex(min(range(len(SCALES)), key=lambda k: abs(SCALES[k] - scale)))
        self.scale.setMaximumWidth(220)
        lay.addWidget(self.scale)
        hint = QLabel("Multiplies the Windows display scaling: text, controls and icons.")
        hint.setProperty("role", "dim")
        lay.addWidget(hint)
        lay.addStretch(1)
        for grp in (self.theme_group, self.accent_group):
            grp.idClicked.connect(lambda _i: self._update_note())
        self.scale.currentIndexChanged.connect(lambda _i: self._update_note())
        return w

    def _startup(self, st: QSettings) -> QWidget:
        w, lay = self._page("Start-up", "What you see when AeroFlow Studio opens.")

        def flag(key: str, default: bool) -> bool:
            v = st.value(key, default)
            return v if isinstance(v, bool) else str(v).lower() not in ("false", "0")

        self.t_home = ToggleSwitch("Show the Home screen at start-up", flag("ui/show_home", True),
                                   "Otherwise the wind tunnel opens directly")
        self.t_updates = ToggleSwitch("Check for updates automatically (at most once a day)",
                                      flag("updates/auto", True))
        self.t_labs = ToggleSwitch("Open the Supersonic and Nozzle Labs inside the main window",
                                   flag("ui/labs_embedded", True),
                                   "As pages beside the wind tunnel (switch with the rail); off: separate windows. "
                                   "Applies to labs opened after the next start")
        lay.addWidget(self.t_home)
        lay.addWidget(self.t_updates)
        lay.addWidget(self.t_labs)
        lay.addStretch(1)
        return w

    def _performance(self) -> QWidget:
        from core import compute

        w, lay = self._page("Performance", "Where the Live Flow Lab, the Supersonic Lab and the Rocket Nozzle Lab "
                                           "run. The GPU is usually 5-20× faster.")
        self.device = QComboBox()
        for dev in compute.Device:
            self.device.addItem(dev.label, dev)
        cur = compute.preference()
        self.device.setCurrentIndex(list(compute.Device).index(cur))
        self.device.setMaximumWidth(320)
        lay.addWidget(make_section_label("Compute device"))
        lay.addWidget(self.device)
        info = compute.gpu_info()
        status = QLabel(f"GPU: {info.name} ({info.backend})" if info.available
                        else f"No usable GPU: the labs run on the CPU. {info.reason}")
        status.setWordWrap(True)
        status.setProperty("role", "muted")
        lay.addWidget(status)
        lay.addStretch(1)
        return w

    # -- saving ------------------------------------------------------------------------------------
    def _choice(self) -> tuple[str, str, float]:
        tb = self.theme_group.checkedButton()
        ab = self.accent_group.checkedButton()
        return (tb.property("theme") if tb else self._initial[0], ab.property("accent") if ab else self._initial[1],
                float(self.scale.currentData()))

    def _update_note(self) -> None:
        changed = self._choice() != self._initial
        self.note.setText("Theme, accent and size changes take effect after a restart." if changed else "")
        self.btn_restart.setEnabled(changed)

    def _save(self, restart: bool) -> None:
        from core import compute

        theme, accent, scale = self._choice()
        st = QSettings(ORG_NAME, APP_NAME)
        st.setValue("ui/theme", theme)
        st.setValue("ui/accent", accent)
        st.setValue("ui/scale", scale)
        st.setValue("ui/show_home", self.t_home.isChecked())
        st.setValue("updates/auto", self.t_updates.isChecked())
        st.setValue("ui/labs_embedded", self.t_labs.isChecked())
        st.sync()
        dev = self.device.currentData()
        if dev is not None and dev != compute.preference():
            compute.set_preference(dev)
        self._restart = restart
        self.accept()

    @property
    def restart_requested(self) -> bool:
        return self._restart


__all__ = ["PreferencesDialog"]

