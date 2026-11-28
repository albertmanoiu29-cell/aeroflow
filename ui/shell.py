"""The 2.0 application shell: navigation rail, Home screen and command palette."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QFont, QKeySequence, QPainter
from PySide6.QtWidgets import (
    QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton,
    QScrollArea, QSizePolicy, QStyle, QStyledItemDelegate, QStyleOptionViewItem, QToolBar, QToolButton, QVBoxLayout,
    QWidget,
)

from config.settings import APP_NAME, APP_VERSION
from config.theme import PALETTE, make_icon, make_logo_pixmap, make_section_label, qcolor


# ---------------------------------------------------------------------------
# Navigation rail
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NavItem:
    key: str
    label: str
    icon: str
    tip: str
    checkable: bool = True


class NavRail(QToolBar):
    """Vertical icon rail (VS Code / Fusion style). ``activated`` emits an item key."""

    activated = Signal(str)

    def __init__(self, groups: Sequence[Sequence[NavItem]], bottom: Sequence[NavItem] = (),
                 parent: Optional[QWidget] = None) -> None:
        super().__init__("Navigation", parent)
        self.setObjectName("NavRail")
        self.setMovable(False)
        self.setFloatable(False)
        self.setOrientation(Qt.Orientation.Vertical)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.setIconSize(QSize(22, 22))
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self._group = QActionGroup(self)
        self._group.setExclusive(True)
        self.actions_by_key: dict[str, QAction] = {}
        for i, group in enumerate(groups):
            if i:
                self._divider()
            for item in group:
                self._add(item)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.addWidget(spacer)
        for item in bottom:
            self._add(item)

    def _divider(self) -> None:
        line = QFrame()
        line.setProperty("role", "divider")
        line.setFixedHeight(1)
        self.addWidget(line)

    def _add(self, item: NavItem) -> None:
        act = QAction(make_icon(item.icon, 22, PALETTE.text_dim, PALETTE.text), item.label, self)
        act.setToolTip(item.tip)
        act.setCheckable(item.checkable)
        if item.checkable:
            self._group.addAction(act)
        act.triggered.connect(lambda _c=False, k=item.key: self.activated.emit(k))
        self.addAction(act)
        btn = self.widgetForAction(act)
        if isinstance(btn, QToolButton):
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.actions_by_key[item.key] = act

    def set_current(self, key: str) -> None:
        act = self.actions_by_key.get(key)
        if act is not None and act.isCheckable():
            act.setChecked(True)


# ---------------------------------------------------------------------------
# Home screen
# ---------------------------------------------------------------------------
class ClickableCard(QFrame):
    """A card that behaves like a button (hover highlight from the stylesheet, click signal)."""

    clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("HomeCard")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


def _icon_tile(icon: str, size: int = 40, tint: Optional[str] = None) -> QLabel:
    """A rounded tile with an icon (accent-tinted)."""
    from PySide6.QtGui import QPixmap

    tint = tint or PALETTE.cyan
    pix = QPixmap(size * 2, size * 2)
    pix.setDevicePixelRatio(2.0)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    c = qcolor(tint, 0.16)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(c)
    p.drawRoundedRect(0, 0, size, size, size * 0.28, size * 0.28)
    ic = make_icon(icon, int(size * 0.55), tint).pixmap(int(size * 0.55), int(size * 0.55))
    off = (size - int(size * 0.55)) / 2
    p.drawPixmap(int(off), int(off), ic)
    p.end()
    lbl = QLabel()
    lbl.setPixmap(pix)
    lbl.setFixedSize(size, size)
    return lbl


def _chip(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("role", "unit")
    return lbl


@dataclass(frozen=True)
class HomeEntry:
    key: str
    title: str
    text: str
    icon: str
    shortcut: str = ""
    tint: str = ""


class HomePage(QWidget):
    """Start screen: workspaces, tools, recent projects and what's new."""

    activated = Signal(str)                 # a workspace / tool key
    open_recent = Signal(str)               # a file path

    WORKSPACES = (
        HomeEntry("cfd", "Wind tunnel", "3-D streamlines, surface pressure and forces on any shape, solved in about "
                  "a second.", "wind", "Ctrl+1"),
        HomeEntry("polar", "Polar analysis", "Sweep the angle of attack: lift curve, drag polar, L/D and stall.",
                  "chart", "Ctrl+2"),
        HomeEntry("live", "Live Flow Lab", "A real-time 2-D wind tunnel: vortex streets, wakes and stall as they "
                  "happen.", "waves", "Ctrl+3"),
        HomeEntry("supersonic", "Supersonic Lab", "Shock waves, expansion fans and wave drag from Mach 0.8 to 5, "
                  "checked against theory.", "shock", "Ctrl+4"),
        HomeEntry("nozzle", "Rocket Nozzle Lab", "Engines from ignition: shock diamonds, thrust, aerospikes, solid "
                  "motors and flight.", "rocket", "Ctrl+5"),
        HomeEntry("import", "Your own model", "Import an STL, OBJ, PLY, STEP or IGES model and test it in the wind "
                  "tunnel.", "folder", "Ctrl+O"),
    )
    TOOLS = (
        HomeEntry("airfoil", "Airfoil analysis", "Panel method, stall estimate, drag polar", "airfoil", "Ctrl+Shift+A"),
        HomeEntry("wing", "Wing analysis", "Lifting line, span loading, induced drag", "wing", "Ctrl+Shift+W"),
        HomeEntry("calc", "Aero calculator", "Atmosphere, Mach, Reynolds, stall speed", "calculator", "Ctrl+Shift+U"),
        HomeEntry("aircraft", "Aircraft design", "Sizing, stability, climb, range", "plane", "Ctrl+Shift+D"),
        HomeEntry("mission", "Mission planner", "Stages, Δv, payload to orbit", "rocket", "Ctrl+Shift+M"),
        HomeEntry("propeller", "Propeller", "BEM, spinning 3-D slipstream", "propeller"),
        HomeEntry("flight", "Flight dynamics", "Trim, short period and phugoid", "plane"),
    )
    WHATS_NEW = (
        ("A new look", "A calm graphite design, a choice of accent colours and a light theme to match."),
        ("Home and navigation", "This start screen and a navigation rail to every workspace, lab and tool."),
        ("Command palette", "Press Ctrl+K and type: every command, workspace and tool is one search away."),
        ("Preferences", "Theme, accent, interface size, start-up and compute device in one place."),
    )

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Root")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        inner = QWidget()
        root = QHBoxLayout(inner)
        root.setContentsMargins(40, 32, 40, 32)
        root.setSpacing(28)
        main = QVBoxLayout()
        main.setSpacing(14)
        main.addWidget(self._hero())
        main.addSpacing(6)
        main.addWidget(make_section_label("Workspaces"))
        main.addLayout(self._grid(self.WORKSPACES, 3, big=True))
        main.addSpacing(6)
        main.addWidget(make_section_label("Engineering tools"))
        main.addLayout(self._grid(self.TOOLS, 3, big=False))
        main.addStretch(1)
        side = QVBoxLayout()
        side.setSpacing(12)
        side.addWidget(make_section_label("Recent"))
        self.recent_box = QVBoxLayout()
        self.recent_box.setSpacing(4)
        rb = QFrame()
        rb.setObjectName("Card")
        rb.setLayout(self.recent_box)
        self.recent_box.setContentsMargins(8, 8, 8, 8)
        side.addWidget(rb)
        side.addSpacing(8)
        side.addWidget(make_section_label(f"New in {APP_VERSION.split('.')[0]}.{APP_VERSION.split('.')[1]}"))
        news = QFrame()
        news.setObjectName("Card")
        nl = QVBoxLayout(news)
        nl.setContentsMargins(16, 14, 16, 14)
        nl.setSpacing(10)
        for title, text in self.WHATS_NEW:
            row = QHBoxLayout()
            row.setSpacing(10)
            dot = _icon_tile("sparkle", 22)
            row.addWidget(dot, 0, Qt.AlignmentFlag.AlignTop)
            col = QVBoxLayout()
            col.setSpacing(1)
            t = QLabel(title)
            t.setStyleSheet("font-weight: 600;")
            d = QLabel(text)
            d.setProperty("role", "muted")
            d.setWordWrap(True)
            col.addWidget(t)
            col.addWidget(d)
            row.addLayout(col, 1)
            nl.addLayout(row)
        side.addWidget(news)
        side.addSpacing(8)
        side.addWidget(make_section_label("Learn"))
        learn = QFrame()
        learn.setObjectName("Card")
        ll = QVBoxLayout(learn)
        ll.setContentsMargins(8, 8, 8, 8)
        ll.setSpacing(2)
        for key, text, icon in (("guide", "Guide: what everything means", "book"),
                                ("shortcuts", "Keyboard shortcuts", "keyboard"),
                                ("palette", "Command palette  (Ctrl+K)", "command")):
            b = QPushButton("  " + text)
            b.setProperty("role", "ghost")
            b.setIcon(make_icon(icon, 16))
            b.setStyleSheet("text-align: left; padding: 7px 8px;")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _c=False, k=key: self.activated.emit(k))
            ll.addWidget(b)
        side.addWidget(learn)
        side.addStretch(1)
        sw = QWidget()
        sw.setLayout(side)
        sw.setFixedWidth(340)
        root.addLayout(main, 1)
        root.addWidget(sw)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setWidget(inner)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(area)
        self.set_recent([])

    def _hero(self) -> QWidget:
        hero = QFrame()
        hero.setObjectName("Hero")
        hero.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        h = QHBoxLayout(hero)
        h.setContentsMargins(28, 26, 28, 26)
        h.setSpacing(22)
        logo = QLabel()
        logo.setPixmap(make_logo_pixmap(72))
        h.addWidget(logo, 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(6)
        title = QLabel(f"{APP_NAME}")
        title.setProperty("role", "display")
        badge = QLabel(f"Version {APP_VERSION}")
        badge.setProperty("role", "badge")
        top = QHBoxLayout()
        top.setSpacing(10)
        top.addWidget(title)
        top.addWidget(badge, 0, Qt.AlignmentFlag.AlignVCenter)
        top.addStretch(1)
        col.addLayout(top)
        sub = QLabel("Aerodynamics and propulsion, from a first sketch to a validated design: wind-tunnel CFD, live "
                     "flow labs, supersonic and rocket-nozzle simulation, and the classic engineering tools.")
        sub.setProperty("role", "muted")
        sub.setWordWrap(True)
        sub.setStyleSheet("font-size: 13px;")
        col.addWidget(sub)
        col.addSpacing(8)
        actions = QHBoxLayout()
        actions.setSpacing(8)
        for key, text, icon, role in (("cfd", "Open the wind tunnel", "play", "primary"),
                                      ("import", "Import geometry…", "folder", ""),
                                      ("open", "Open project…", "layers", ""),
                                      ("nozzle", "Rocket Nozzle Lab", "rocket", "")):
            b = QPushButton(text)
            if role:
                b.setProperty("role", role)
            b.setIcon(make_icon(icon, 16, PALETTE.on_accent if role == "primary" else None))
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setMinimumHeight(34)
            b.clicked.connect(lambda _c=False, k=key: self.activated.emit(k))
            actions.addWidget(b)
        actions.addStretch(1)
        col.addLayout(actions)
        h.addLayout(col, 1)
        return hero

    def _grid(self, entries: Sequence[HomeEntry], cols: int, big: bool) -> QGridLayout:
        g = QGridLayout()
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(12)
        for k, e in enumerate(entries):
            card = ClickableCard()
            card.setToolTip(e.text)
            lay = QVBoxLayout(card) if big else QHBoxLayout(card)
            lay.setContentsMargins(18, 16, 18, 16) if big else lay.setContentsMargins(14, 12, 14, 12)
            lay.setSpacing(10)
            if big:
                top = QHBoxLayout()
                top.addWidget(_icon_tile(e.icon, 40))
                top.addStretch(1)
                if e.shortcut:
                    top.addWidget(_chip(e.shortcut), 0, Qt.AlignmentFlag.AlignTop)
                lay.addLayout(top)
                t = QLabel(e.title)
                t.setProperty("role", "headline")
                d = QLabel(e.text)
                d.setProperty("role", "muted")
                d.setWordWrap(True)
                lay.addWidget(t)
                lay.addWidget(d)
                lay.addStretch(1)
                card.setMinimumHeight(150)
            else:
                lay.addWidget(_icon_tile(e.icon, 32))
                col = QVBoxLayout()
                col.setSpacing(1)
                t = QLabel(e.title)
                t.setStyleSheet("font-weight: 600;")
                d = QLabel(e.text)
                d.setProperty("role", "dim")
                d.setWordWrap(True)
                col.addWidget(t)
                col.addWidget(d)
                lay.addLayout(col, 1)
            card.clicked.connect(lambda k=e.key: self.activated.emit(k))
            g.addWidget(card, k // cols, k % cols)
        for c in range(cols):
            g.setColumnStretch(c, 1)
        return g

    def set_recent(self, files: Sequence[str]) -> None:
        while self.recent_box.count():
            it = self.recent_box.takeAt(0)
            if it.widget() is not None:
                it.widget().deleteLater()
        existing = [f for f in files if Path(f).is_file()][:7]
        if not existing:
            lbl = QLabel("Projects and models you open appear here.")
            lbl.setProperty("role", "muted")
            lbl.setWordWrap(True)
            lbl.setContentsMargins(8, 8, 8, 8)
            self.recent_box.addWidget(lbl)
            return
        for f in existing:
            p = Path(f)
            b = QPushButton(f"  {p.name}")
            b.setProperty("role", "ghost")
            b.setIcon(make_icon("layers" if p.suffix.lower() == ".aeroflow" else "folder", 16))
            b.setToolTip(str(p))
            b.setStyleSheet("text-align: left; padding: 7px 8px;")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _c=False, path=str(p): self.open_recent.emit(path))
            self.recent_box.addWidget(b)


# ---------------------------------------------------------------------------
# Command palette
# ---------------------------------------------------------------------------
@dataclass
class Command:
    title: str
    category: str
    run: Callable[[], object]
    shortcut: str = ""
    keywords: str = ""


def _clean(text: str) -> str:
    return text.replace("&", "").replace("…", "").strip()


def fuzzy_score(query: str, text: str) -> Optional[float]:
    """Higher is better; None if the characters of ``query`` do not appear in order in ``text``."""
    q = query.lower().strip()
    t = text.lower()
    if not q:
        return 0.0
    if q in t:
        return 100.0 - t.index(q) + (20.0 if t.startswith(q) else 0.0) + (10.0 if re.search(r"\b" + re.escape(q), t)
                                                                          else 0.0)
    pos, score, prev = 0, 0.0, -2
    for ch in q:
        if ch == " ":
            continue
        k = t.find(ch, pos)
        if k < 0:
            return None
        score += 3.0 if k == prev + 1 else (2.0 if k == 0 or t[k - 1] in " -/(" else 1.0)
        prev, pos = k, k + 1
    return score


class _CommandDelegate(QStyledItemDelegate):
    """Title on the left, category and shortcut in muted text on the right."""

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        cmd: Optional[Command] = index.data(Qt.ItemDataRole.UserRole)
        opt.text = ""
        style = opt.widget.style() if opt.widget else None
        if style is not None:
            style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)
        if cmd is None:
            return
        r = option.rect.adjusted(12, 0, -12, 0)
        painter.save()
        f = QFont(option.font)
        painter.setFont(f)
        painter.setPen(QColor(PALETTE.text))
        painter.drawText(r, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), cmd.title)
        f2 = QFont(option.font)
        f2.setPixelSize(11)
        painter.setFont(f2)
        right = (cmd.shortcut + "   " if cmd.shortcut else "") + cmd.category
        painter.setPen(QColor(PALETTE.text_dim))
        painter.drawText(r, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight), right)
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:  # noqa: N802
        return QSize(option.rect.width(), 36)


class CommandPalette(QDialog):
    """Ctrl+K: type to find any command, workspace or tool; Enter runs it."""

    def __init__(self, commands: Sequence[Command], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent, Qt.WindowType.FramelessWindowHint | Qt.WindowType.Popup)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.commands = list(commands)
        frame = QFrame(self)
        frame.setObjectName("Palette")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.edit = QLineEdit()
        self.edit.setObjectName("PaletteInput")
        self.edit.setPlaceholderText("Search commands, workspaces and tools…")
        self.edit.addAction(make_icon("search", 16), QLineEdit.ActionPosition.LeadingPosition)
        self.list = QListWidget()
        self.list.setObjectName("PaletteList")
        self.list.setItemDelegate(_CommandDelegate(self.list))
        self.list.setUniformItemSizes(True)
        self.hint = QLabel("↑ ↓ to move · Enter to run · Esc to close")
        self.hint.setProperty("role", "dim")
        self.hint.setContentsMargins(14, 6, 14, 8)
        lay.addWidget(self.edit)
        lay.addWidget(self.list, 1)
        lay.addWidget(self.hint)
        self.edit.textChanged.connect(self._filter)
        self.edit.installEventFilter(self)
        self.list.itemActivated.connect(self._run_item)
        self.list.itemClicked.connect(self._run_item)
        self.resize(640, 440)
        self._filter("")

    def popup(self) -> None:
        par = self.parentWidget()
        if par is not None:
            top_left = par.mapToGlobal(par.rect().topLeft())
            w = min(680, int(par.width() * 0.6))
            self.resize(max(w, 480), 440)
            self.move(top_left.x() + (par.width() - self.width()) // 2, top_left.y() + 70)
        self.edit.clear()
        self._filter("")
        self.show()
        self.edit.setFocus()

    def _filter(self, text: str) -> None:
        self.list.clear()
        scored = []
        for c in self.commands:
            s = fuzzy_score(text, f"{c.title} {c.keywords}")
            s2 = fuzzy_score(text, c.category)
            if s is None and s2 is not None and text:
                s = s2 * 0.5
            if s is not None:
                scored.append((s, c))
        scored.sort(key=lambda sc: -sc[0]) if text else None
        for _s, c in scored[:200]:
            it = QListWidgetItem()
            it.setData(Qt.ItemDataRole.UserRole, c)
            it.setData(Qt.ItemDataRole.ToolTipRole, f"{c.category}: {c.title}")
            self.list.addItem(it)
        if self.list.count():
            self.list.setCurrentRow(0)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is self.edit and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key.Key_Down, Qt.Key.Key_Up, Qt.Key.Key_PageDown, Qt.Key.Key_PageUp):
                n = self.list.count()
                if n:
                    step = {Qt.Key.Key_Down: 1, Qt.Key.Key_Up: -1, Qt.Key.Key_PageDown: 8, Qt.Key.Key_PageUp: -8}[key]
                    self.list.setCurrentRow(min(max(self.list.currentRow() + step, 0), n - 1))
                return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                it = self.list.currentItem()
                if it is not None:
                    self._run_item(it)
                return True
            if key == Qt.Key.Key_Escape:
                self.close()
                return True
        return super().eventFilter(obj, event)

    def _run_item(self, item: QListWidgetItem) -> None:
        cmd: Optional[Command] = item.data(Qt.ItemDataRole.UserRole)
        self.close()
        if cmd is not None:
            cmd.run()


def commands_from_actions(actions: Sequence[QAction], category_of: Callable[[QAction], str]) -> list[Command]:
    """Palette commands for every enabled, titled action (menus included)."""
    out, seen = [], set()
    for act in actions:
        title = _clean(act.text())
        if not title or act.isSeparator() or act.menu() is not None or title in seen:
            continue
        seen.add(title)
        sc = act.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
        out.append(Command(title, category_of(act), lambda a=act: a.trigger() if a.isEnabled() else None, sc,
                           _clean(act.toolTip() or "")))
    return out


__all__ = ["NavItem", "NavRail", "HomePage", "ClickableCard", "Command", "CommandPalette", "fuzzy_score",
           "commands_from_actions"]

