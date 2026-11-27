"""Hess-Smith panel method against classical airfoil results, and the aero calculator."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import panel as pm  # noqa: E402
from core.shapes2d import naca_outline  # noqa: E402


def _nodes(code: str, n: int = 160) -> np.ndarray:
    return pm.prepare(naca_outline(code, 200), n)


def test_symmetric_airfoil_at_zero_lift() -> None:
    s = pm.solve(_nodes("0012"), 0.0)
    assert abs(s.cl) < 1e-6 and abs(s.cm_quarter) < 1e-4
    assert float(s.cp.min()) == pytest.approx(-0.41, abs=0.03)          # NACA 0012 suction peak (Abbott)
    assert float(s.cp.max()) == pytest.approx(1.0, abs=0.05)            # stagnation point


def test_lift_slope_and_camber() -> None:
    s5 = pm.solve(_nodes("0012"), 5.0)
    # thick-airfoil inviscid slope: 2π (1 + 0.77 t/c) per radian
    assert s5.cl == pytest.approx(2 * math.pi * (1 + 0.77 * 0.12) * math.radians(5.0), rel=0.03)
    p = pm.polar(_nodes("2412"), np.arange(-4.0, 4.01, 1.0), 3e6)
    assert p.alpha_zero_lift == pytest.approx(-2.1, abs=0.2)           # thin-airfoil theory / Abbott: -2.1°
    assert float(np.interp(0.0, p.alpha, p.cm)) == pytest.approx(-0.053, abs=0.01)


def test_panel_convergence() -> None:
    a = pm.solve(_nodes("2412", 120), 6.0).cl
    b = pm.solve(_nodes("2412", 240), 6.0).cl
    assert a == pytest.approx(b, rel=0.01)


def test_stall_estimate_is_plausible() -> None:
    p = pm.polar(_nodes("0012"), np.arange(-20.0, 22.01, 0.5), 3e6)
    assert 12.0 < p.stall_alpha < 18.0                                  # measured ≈ 16° at Re 3×10⁶
    assert 1.3 < p.cl_max < 1.8                                         # measured ≈ 1.55
    assert p.stall_negative == pytest.approx(-p.stall_alpha, abs=0.6)
    thin = pm.polar(_nodes("0006"), np.arange(0.0, 20.01, 0.5), 3e6)
    assert thin.stall_alpha < p.stall_alpha                             # thin sections stall early (leading edge)


def test_drag_estimate() -> None:
    assert pm.profile_drag(3e6, 0.12) == pytest.approx(0.0075, rel=0.25)   # NACA 0012 minimum CD ≈ 0.006-0.008
    assert pm.profile_drag(1e5, 0.12) > pm.profile_drag(1e7, 0.12)


def test_aero_calculator_numbers() -> None:
    from ui.aero_calc_dialog import flight_numbers

    n = flight_numbers(0.0, 50.0, 1.0)
    assert n["rho"] == pytest.approx(1.225, rel=1e-3) and n["a"] == pytest.approx(340.3, rel=1e-3)
    assert n["q"] == pytest.approx(0.5 * 1.225 * 2500.0, rel=1e-3)
    assert n["re"] == pytest.approx(50.0 / 1.46e-5, rel=0.02)
    hi = flight_numbers(11000.0, 250.0, 1.0)
    assert hi["eas"] < 250.0 and hi["mach"] == pytest.approx(250.0 / 295.1, rel=2e-3)
