"""Save and restore the inputs of any tool window by attribute name (for project files and undo).

Every spin box, combo box, line edit and check box held as an attribute of the window is captured as
``{attribute: value}``; restoring sets them back with signals blocked, then calls the window's refresh.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QAbstractButton, QComboBox, QDoubleSpinBox, QLineEdit, QSlider, QSpinBox, QWidget

_SKIP = {"spins"}


def widget_state(obj: Any) -> dict:
    out: dict[str, Any] = {}
    for name, w in vars(obj).items():
        if name.startswith("_") or name in _SKIP:
            continue
        if isinstance(w, (QDoubleSpinBox, QSpinBox, QSlider)):
            out[name] = w.value()
        elif isinstance(w, QComboBox):
            out[name] = w.currentText()
        elif isinstance(w, QLineEdit):
            out[name] = w.text()
        elif isinstance(w, QAbstractButton) and w.isCheckable():
            out[name] = bool(w.isChecked())
        elif isinstance(w, QWidget) and callable(getattr(w, "value", None)) and callable(getattr(w, "set_value", None)):
            try:
                out[name] = float(w.value())
            except (TypeError, ValueError):
                pass
    spins = getattr(obj, "spins", None)
    if isinstance(spins, dict):
        out["spins"] = {k: sp.value() for k, sp in spins.items()}
    return out


def apply_widget_state(obj: Any, st: dict, refresh: str = "recompute") -> None:
    if not isinstance(st, dict):
        return
    changed: list[QWidget] = []
    combos = [(n, v) for n, v in st.items() if isinstance(getattr(obj, n, None), QComboBox)]
    rest = [(n, v) for n, v in st.items() if (n, v) not in combos]
    for name, value in combos + rest:                 # combos first: they may enable or relabel other inputs
        w = getattr(obj, name, None)
        try:
            if name == "spins" and isinstance(value, dict) and isinstance(getattr(obj, "spins", None), dict):
                for k, v in value.items():
                    sp = obj.spins.get(k)
                    if sp is not None:
                        sp.blockSignals(True)
                        sp.setValue(float(v))
                        sp.blockSignals(False)
                continue
            if w is None:
                continue
            w.blockSignals(True)
            if isinstance(w, QDoubleSpinBox):
                w.setValue(float(value))
            elif isinstance(w, (QSpinBox, QSlider)):
                w.setValue(int(value))
            elif isinstance(w, QComboBox):
                k = w.findText(str(value))
                if k >= 0:
                    w.setCurrentIndex(k)
            elif isinstance(w, QLineEdit):
                w.setText(str(value))
            elif isinstance(w, QAbstractButton) and w.isCheckable():
                w.setChecked(bool(value))
            elif callable(getattr(w, "set_value", None)):
                w.set_value(float(value))
            w.blockSignals(False)
            changed.append(w)
        except (TypeError, ValueError):
            if w is not None:
                w.blockSignals(False)
    fn = getattr(obj, refresh, None)
    if callable(fn):
        fn()


__all__ = ["widget_state", "apply_widget_state"]
