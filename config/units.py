"""Display units. All computation stays in SI; only what the user sees and types is converted.

``Imperial`` follows aviation practice: speed in knots, altitude in feet, temperature differences in °F,
forces in lbf, density in slug/ft³ and pressure in lbf/ft² (psf).
"""

from __future__ import annotations

from typing import Callable

SYSTEMS = ("SI", "Imperial")

# quantity -> (scale, unit) per system: display = SI value * scale
_TABLE: dict[str, dict[str, tuple[float, str]]] = {
    "speed": {"SI": (1.0, "m/s"), "Imperial": (1.0 / 0.514444, "kt")},
    "altitude": {"SI": (1.0, "m"), "Imperial": (1.0 / 0.3048, "ft")},
    "temp_delta": {"SI": (1.0, "K"), "Imperial": (1.8, "°F")},
    "force": {"SI": (1.0, "N"), "Imperial": (1.0 / 4.448222, "lbf")},
    "density": {"SI": (1.0, "kg/m³"), "Imperial": (1.0 / 515.379, "slug/ft³")},
    "pressure": {"SI": (1.0, "Pa"), "Imperial": (1.0 / 47.880259, "psf")},
    "length": {"SI": (1.0, "m"), "Imperial": (1.0 / 0.3048, "ft")},
}

_system = "SI"
_listeners: list[Callable[[str], None]] = []


def system() -> str:
    return _system


def set_system(name: str) -> None:
    global _system
    name = name if name in SYSTEMS else "SI"
    if name == _system:
        return
    _system = name
    for fn in list(_listeners):
        fn(name)


def add_listener(fn: Callable[[str], None]) -> None:
    if fn not in _listeners:
        _listeners.append(fn)


def scale(quantity: str) -> float:
    return _TABLE[quantity][_system][0]


def unit(quantity: str) -> str:
    return _TABLE[quantity][_system][1]


def fmt(value_si: float, quantity: str, decimals: int = 1) -> str:
    """``"12.3 kt"`` - an SI value shown in the current system."""
    return f"{value_si * scale(quantity):,.{decimals}f} {unit(quantity)}"
