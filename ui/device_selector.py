"""Compute-device selector (Auto / GPU / CPU) shared by the Live Flow Lab and the Supersonic Lab.

Every selector reflects the one global preference in :mod:`core.compute`, so changing it in one place
(a Lab panel or the Simulation menu) updates all of them; the solvers switch at their next batch and
carry their flow state across. The GPU probe runs in the background at start-up, so the GPU entry is
greyed out (with the reason as tooltip) until it is known to work.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import QComboBox, QGridLayout, QLabel, QWidget

from core import compute
from core.compute import Device
from ui.cfd_panel import compact_combo


class _Relay(QObject):
    """Turns core.compute notifications (possibly from a worker thread) into a GUI-thread signal."""

    changed = Signal()


_relay: Optional[_Relay] = None


def compute_relay() -> _Relay:
    """Process-wide Qt signal fired when the compute preference or GPU availability changes."""
    global _relay
    if _relay is None:
        _relay = _Relay()
        compute.add_listener(lambda _dev: _relay.changed.emit())
    return _relay


class DeviceSelector(QWidget):
    """``Compute [combo]`` row plus a full-width 'Running on …' note."""

    def __init__(self, label: str = "Compute", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setVerticalSpacing(3)
        tip = ("Where the live solvers run. The GPU (CUDA on NVIDIA, OpenCL on AMD and Intel) is typically 5-25x "
               "faster than the CPU; "
               "the flow keeps running when you switch.")
        cap = QLabel(label)
        cap.setToolTip(tip)
        self.combo = QComboBox()
        for dev in Device:
            self.combo.addItem(dev.label, dev)
        self.combo.setToolTip(tip)
        compact_combo(self.combo)
        self.note = QLabel("")
        self.note.setProperty("role", "muted")
        self.note.setWordWrap(True)
        lay.addWidget(cap, 0, 0)
        lay.addWidget(self.combo, 0, 1)
        lay.addWidget(self.note, 1, 0, 1, 2)
        lay.setColumnStretch(0, 2)
        lay.setColumnStretch(1, 3)
        self.combo.currentIndexChanged.connect(self._picked)
        compute_relay().changed.connect(self.refresh)
        self._poll = QTimer(self)
        self._poll.setInterval(400)
        self._poll.timeout.connect(self._wait_probe)
        self.refresh()
        if not compute.probed():
            self._poll.start()

    def _wait_probe(self) -> None:
        if compute.probed():
            self._poll.stop()
            self.refresh()

    def _picked(self, index: int) -> None:
        dev = self.combo.itemData(index)
        if isinstance(dev, Device) and dev is not compute.preference():
            compute.set_preference(dev)

    def refresh(self) -> None:
        pref = compute.preference()
        self.combo.blockSignals(True)
        self.combo.setCurrentIndex(list(Device).index(pref))
        self.combo.blockSignals(False)
        model = self.combo.model()
        gpu_item = model.item(list(Device).index(Device.GPU)) if hasattr(model, "item") else None
        if not compute.probed():
            self.note.setText("Checking for a GPU…")
            return
        info = compute.gpu_info()
        if gpu_item is not None:
            gpu_item.setEnabled(info.available)
            gpu_item.setToolTip(info.summary())
        self.note.setText(f"Running on: {compute.describe()}")
        self.note.setToolTip(info.summary())
        self.note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
