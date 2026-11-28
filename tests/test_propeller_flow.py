"""Propeller slipstream (momentum theory on the BEM loading) and the static-thrust BEM fix."""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import bem  # noqa: E402
from core import propeller_flow as pf  # noqa: E402

warnings.simplefilter("ignore", RuntimeWarning)


def test_static_thrust_is_the_highest() -> None:
    p = bem.Propeller(1.0, 2, 0.8)
    t = [bem.solve(p, 2000, v).thrust for v in (0.0, 5.0, 15.0, 25.0)]
    assert t[0] > t[1] > t[2] > t[3] > 0.0                # a fixed-pitch propeller pulls hardest standing still
    assert t[2] == pytest.approx(204.0, rel=0.03)          # the working point is unchanged by the rewrite


@pytest.mark.parametrize("speed", [0.0, 15.0])
def test_slipstream_carries_the_thrust(speed: float) -> None:
    p = bem.Propeller(1.0, 3, 0.9)
    res = bem.solve(p, 2400, speed)
    ss = pf.slipstream(res, p.diameter)
    # momentum: T = ∫ ρ (V + vi) (2 vi) 2π r dr
    t_m = float(np.trapz(4.0 * math.pi * ss.r * res.rho * (speed + ss.v_induced) * ss.v_induced, ss.r))
    assert t_m == pytest.approx(res.thrust, rel=0.02)
    # the far wake is twice the induced speed and the tube contracts (more so when static)
    r0 = np.array([0.4])
    far = ss.tube_radius(r0, np.array([20.0]))[0] / 0.4
    assert ss.axial_speed(r0, np.array([1e3]))[0] == pytest.approx(speed + 2 * ss.induced_at(r0)[0], rel=1e-3)
    assert far < 1.0
    if speed == 0.0:
        assert far == pytest.approx(math.sqrt(0.5), rel=0.02)   # static actuator disc: area halves
    assert np.all(ss.swirl(r0, np.array([-5.0])) < 1e-3)      # no swirl upstream
    assert ss.swirl(r0, np.array([2.0]))[0] > 0.0             # swirl in the direction of rotation


def test_streamline_crosses_the_disc_where_seeded() -> None:
    p = bem.Propeller(1.0, 2, 0.8)
    ss = pf.slipstream(bem.solve(p, 2000, 10.0), 1.0)
    pts = ss.streamline(0.3, 1.0, -1.0, 2.0, 301)
    k = int(np.argmin(np.abs(pts[:, 0])))
    assert math.hypot(pts[k, 1], pts[k, 2]) == pytest.approx(0.3, rel=0.02)
    assert math.atan2(pts[k, 2], pts[k, 1]) == pytest.approx(1.0, abs=0.03)
