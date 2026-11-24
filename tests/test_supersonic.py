"""Supersonic Lab: gas-dynamics theory against published tables, and the Euler solver against theory.

Reference values: NACA Report 1135 ("Equations, tables and charts for compressible flow", 1953).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import accel  # noqa: E402
from core import gasdynamics as gd  # noqa: E402
from core.shapes2d import Obstacle2D, builtin_outline, polygon_segments, rasterize_segments, transform_segments  # noqa: E402
from core.supersonic import Supersonic2D  # noqa: E402


# ---------------------------------------------------------------------------
# Theory (exact to the table precision)
# ---------------------------------------------------------------------------
def test_normal_shock_table_m2() -> None:
    ns = gd.normal_shock(2.0)
    assert ns.m2 == pytest.approx(0.5774, abs=1e-4)
    assert ns.p_ratio == pytest.approx(4.5000, abs=1e-4)
    assert ns.rho_ratio == pytest.approx(2.6667, abs=1e-4)
    assert ns.t_ratio == pytest.approx(1.6875, abs=1e-4)
    assert ns.p0_ratio == pytest.approx(0.7209, abs=1e-4)
    assert ns.pitot_ratio == pytest.approx(5.6404, abs=1e-4)


def test_normal_shock_limits() -> None:
    ns = gd.normal_shock(1.0)
    assert ns.m2 == pytest.approx(1.0) and ns.p_ratio == pytest.approx(1.0) and ns.p0_ratio == pytest.approx(1.0)
    assert gd.normal_shock(50.0).rho_ratio == pytest.approx(6.0, rel=2e-3)       # (g+1)/(g-1) limit
    with pytest.raises(ValueError):
        gd.normal_shock(0.8)


def test_oblique_shock_table() -> None:
    os_ = gd.oblique_shock(2.0, 10.0)
    assert os_.beta_deg == pytest.approx(39.314, abs=2e-3)
    assert os_.m2 == pytest.approx(1.6405, abs=1e-4)
    assert os_.p_ratio == pytest.approx(1.7066, abs=1e-4)
    assert gd.oblique_shock_beta(2.0, 10.0, strong=True) == pytest.approx(83.70, abs=0.01)
    assert gd.oblique_shock(2.0, 25.0) is None                                   # detached
    assert gd.max_deflection(2.0)[0] == pytest.approx(22.97, abs=0.01)
    assert gd.max_deflection(3.0)[0] == pytest.approx(34.07, abs=0.01)


def test_theta_beta_m_round_trip() -> None:
    for m in (1.5, 2.0, 3.0, 6.0):
        for th in (2.0, 8.0, 15.0):
            beta = gd.oblique_shock_beta(m, th)
            if beta is not None:
                assert gd.theta_from_beta(m, beta) == pytest.approx(th, abs=1e-6)


def test_isentropic_and_prandtl_meyer() -> None:
    assert gd.total_pressure_ratio(2.0) == pytest.approx(7.8244, abs=1e-4)
    assert gd.total_temperature_ratio(2.0) == pytest.approx(1.8, abs=1e-12)
    assert gd.area_ratio(2.0) == pytest.approx(1.6875, abs=1e-4)
    assert gd.mach_angle_deg(2.0) == pytest.approx(30.0, abs=1e-9)
    assert gd.prandtl_meyer_deg(2.0) == pytest.approx(26.3798, abs=1e-4)
    assert gd.prandtl_meyer_deg(3.0) == pytest.approx(49.7573, abs=1e-4)
    assert gd.inverse_prandtl_meyer(26.3798) == pytest.approx(2.0, abs=1e-5)
    m2, pr = gd.expansion(2.0, 10.0)
    assert m2 > 2.0 and pr < 1.0


def test_shock_angle_from_pressure_ratio_inverts_oblique_shock() -> None:
    os_ = gd.oblique_shock(3.0, 12.0)
    assert gd.shock_angle_from_pressure_ratio(3.0, os_.p_ratio) == pytest.approx(os_.beta_deg, abs=1e-6)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
def _run(s: Supersonic2D, flow_throughs: float) -> None:
    while s.time * s.mach < flow_throughs * s.nx:
        s.step(40)


def test_uniform_flow_is_preserved_exactly() -> None:
    s = Supersonic2D(80, 40, 2.5)
    s.step(150)
    r, u, v, p = s.primitives()
    assert np.abs(r - 1.0).max() < 1e-12 and np.abs(u - 2.5).max() < 1e-12 and np.abs(v).max() < 1e-12


def test_wedge_oblique_shock_matches_theory() -> None:
    m, th = 3.0, 15.0
    nx, ny = 240, 170
    length = 100.0
    x0, y0 = 40.0, ny / 2 + 0.5
    h = length * math.tan(math.radians(th))
    segs = polygon_segments(np.array([[x0, y0], [x0 + length, y0 + h], [x0 + length, y0 - h]]))
    s = Supersonic2D(nx, ny, m)
    s.set_obstacle(rasterize_segments(segs, nx, ny), length, segs)
    assert s.ib_count > 0
    _run(s, 3.0)
    theory = gd.oblique_shock(m, th)
    sp = s.surface_pressure()
    sel = (np.abs(sp["x"] - (x0 + 0.5 * length)) < 10) & (sp["ny"] > 0)
    p_up = float(np.median(sp["p"][sel]))
    assert p_up == pytest.approx(theory.p_ratio, rel=0.015)                     # wall pressure within 1.5 %
    assert gd.shock_angle_from_pressure_ratio(m, p_up) == pytest.approx(theory.beta_deg, abs=0.6)
    beta_geo = s.measured_shock_angle((x0, y0), 0.85 * length)
    assert beta_geo is not None and beta_geo == pytest.approx(theory.beta_deg, abs=1.5)


def test_cylinder_bow_shock_stagnation_pressure() -> None:
    m, d = 3.0, 56.0
    nx, ny = 300, 210
    o = builtin_outline(Obstacle2D.CYLINDER)
    segs = transform_segments(o.segments, o.pivot, d, 0.0, (130.0, ny / 2 + 0.5))
    s = Supersonic2D(nx, ny, m)
    s.set_obstacle(rasterize_segments(segs, nx, ny), d, segs)
    _run(s, 2.5)
    pitot = gd.normal_shock(m).pitot_ratio
    assert s.stagnation_pressure() == pytest.approx(pitot, rel=0.04)            # Rayleigh pitot within 4 %
    so = s.standoff_distance()
    assert so is not None and 0.4 < so / (d / 2) < 1.0                          # detached bow shock ahead


def test_diamond_airfoil_wave_drag() -> None:
    # t/c = 0.2: its sharp edges are resolved by the grid. (Knife-thin edges only a few cells thick act
    # slightly blunt and over-predict wave drag on coarse grids - a resolution limit, stated in the UI.)
    m, tc = 2.5, 0.20
    nx, ny = 300, 190
    c = 120.0
    x0, y0 = 70.0, ny / 2 + 0.5
    pts = np.array([[x0, y0], [x0 + c / 2, y0 + tc * c / 2], [x0 + c, y0], [x0 + c / 2, y0 - tc * c / 2]])
    segs = polygon_segments(pts)
    s = Supersonic2D(nx, ny, m)
    s.set_obstacle(rasterize_segments(segs, nx, ny), c, segs)
    _run(s, 3.0)
    cd, cl = s.force_coefficients()
    th = math.degrees(math.atan(tc))
    os_ = gd.oblique_shock(m, th)
    p3 = os_.p_ratio * gd.expansion(os_.m2, 2 * th)[1]
    cd_theory = (os_.p_ratio - p3) * tc / (0.7 * m * m)          # shock-expansion theory
    assert cd == pytest.approx(cd_theory, rel=0.10)
    assert abs(cl) < 0.01                                        # symmetric at zero incidence


def test_hypersonic_blunt_body_stays_physical() -> None:
    o = builtin_outline(Obstacle2D.CYLINDER)
    segs = transform_segments(o.segments, o.pivot, 30.0, 0.0, (90.0, 60.5))
    s = Supersonic2D(200, 120, 6.0)
    s.set_obstacle(rasterize_segments(segs, 200, 120), 30.0, segs)
    _run(s, 1.5)
    r, _, _, p = s.primitives()
    fluid = ~s.solid_int
    assert np.isfinite(s.U).all()
    assert r[fluid].min() > 0 and p[fluid].min() > 0
    assert s.mach_field()[fluid].max() < 6.5                     # no spurious acceleration


def test_subsonic_flow_has_no_shock() -> None:
    o = builtin_outline(Obstacle2D.CYLINDER)
    segs = transform_segments(o.segments, o.pivot, 24.0, 0.0, (80.0, 50.5))
    s = Supersonic2D(200, 100, 0.4)
    s.set_obstacle(rasterize_segments(segs, 200, 100), 24.0, segs)
    _run(s, 1.0)
    assert not s.shock_sensor().any()


def test_numpy_fallback_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accel, "HAVE_NUMBA", False)
    m = np.zeros((40, 80), bool)
    m[15:25, 25:35] = True
    s = Supersonic2D(80, 40, 2.0)
    s.set_obstacle(m, 10.0)
    s.step(60)
    assert np.isfinite(s.U).all()
    assert s.pressure_ratio()[~s.solid_int].max() > 2.0          # a shock formed ahead of the block
