"""Tests for the Live Flow Lab: 2-D shapes and the Lattice-Boltzmann solver."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import GeometryPreset  # noqa: E402
from core import accel, lbm2d  # noqa: E402
from core.geometry_manager import build_preset  # noqa: E402
from core.lbm2d import FlowLab2D  # noqa: E402
from core.shapes2d import (  # noqa: E402
    Obstacle2D, build_mask, builtin_outline, mesh_section, polygon_segments, rasterize_segments,
)


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------
def test_circle_raster_area() -> None:
    m = build_mask(builtin_outline(Obstacle2D.CYLINDER), 200, 120, 40.0, 0.0, (100.0, 60.0))
    assert m.sum() == pytest.approx(math.pi * 20 ** 2, rel=0.03)


def test_square_with_hole_even_odd() -> None:
    outer = polygon_segments(np.array([[10, 10], [50, 10], [50, 50], [10, 50]], float))
    inner = polygon_segments(np.array([[20, 20], [40, 20], [40, 40], [20, 40]], float))
    m = rasterize_segments(np.concatenate([outer, inner]), 64, 64)
    assert m[15, 15] and not m[30, 30]


def test_airfoil_angle_of_attack_pitches_nose_up() -> None:
    o = builtin_outline(Obstacle2D.AIRFOIL, "0012")
    m = build_mask(o, 300, 160, 100.0, 15.0, (100.0, 80.0))
    rows, cols = np.nonzero(m)
    le_row = rows[cols == cols.min()].mean()
    te_row = rows[cols == cols.max()].mean()
    assert le_row > te_row + 10            # z (row) grows upward: the nose is above the tail


def test_mesh_section_of_naca_wing() -> None:
    wing = build_preset(GeometryPreset.NACA_0012)
    sec = mesh_section(wing.vertices, wing.faces)
    assert sec is not None
    m = build_mask(sec, 300, 120, 200.0, 0.0, (80.0, 60.0))
    # NACA 00xx section area = 0.685 t c^2
    assert m.sum() == pytest.approx(0.685 * 0.12 * 200 ** 2, rel=0.08)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
def _cylinder_lab(nx: int = 240, ny: int = 96, d: float = 12.0, re: float = 100.0) -> FlowLab2D:
    lab = FlowLab2D(nx, ny)
    lab.set_obstacle(build_mask(builtin_outline(Obstacle2D.CYLINDER), nx, ny, d, 0.0, (0.2 * nx, ny / 2 + 0.37)), d)
    lab.set_reynolds(re)
    return lab


@pytest.mark.skipif(not accel.HAVE_NUMBA, reason="numba not installed")
def test_numba_and_numpy_steps_agree() -> None:
    lab = _cylinder_lab(120, 60, 10.0)
    lab.step(80, dye=False, particles=False)
    u = 0.08
    cu = 3.0 * lbm2d.CX * u
    ft = (lbm2d.W * (1 + cu + 0.5 * cu * cu - 1.5 * u * u)).astype(np.float32)
    out_nb, out_np = np.empty_like(lab.f), np.empty_like(lab.f)
    fnb, fnp = np.zeros(2), np.zeros(2)
    a = [x.copy() for x in (lab.ux, lab.uy, lab.rho)]
    b = [x.copy() for x in (lab.ux, lab.uy, lab.rho)]
    lbm2d._lbm_step_nb(lab.f, out_nb, lab.solid, lab.special, np.float32(u), lab.tau_x, np.float32(lab.smag),
                       *a, fnb, lbm2d.CX, lbm2d.CY, lbm2d.W, lbm2d.OPP, lbm2d.REFY, lab.sponge, ft)
    lbm2d._lbm_step_np(lab.f, out_np, lab.solid, u, lab.tau_x, lab.smag, *b, fnp, lab.sponge, ft)
    np.testing.assert_allclose(out_nb, out_np, atol=2e-6)
    np.testing.assert_allclose(fnb, fnp, rtol=1e-3, atol=1e-6)


def test_empty_tunnel_stays_uniform() -> None:
    lab = FlowLab2D(120, 48)
    while lab.steps < 2500:
        lab.step(250, dye=False, particles=False)
    np.testing.assert_allclose(lab.ux[:, 10:-10] / lab.u_lattice, 1.0, atol=0.02)
    assert np.abs(lab.uy).max() < 0.01 * lab.u_lattice
    assert abs(lab.cl) < 1e-9 and abs(lab.cd) < 1e-9


def test_cylinder_sheds_vortices_at_the_right_frequency() -> None:
    lab = _cylinder_lab()
    while lab.time < 120:
        lab.step(60, dye=False, particles=False)    # one history sample per call, like the UI frames
    st = lab.strouhal()
    _, cd = lab.mean_coefficients(40)
    assert st is not None and 0.14 < st < 0.23, st        # 0.165 unconfined, higher with 12.5 % blockage
    assert 1.2 < cd < 2.3, cd


def test_airfoil_lift_sign_follows_angle() -> None:
    cls = []
    for aoa in (-8.0, 0.0, 8.0):
        lab = FlowLab2D(240, 96)
        lab.set_obstacle(build_mask(builtin_outline(Obstacle2D.AIRFOIL, "0012"), 240, 96, 48.0, aoa,
                                    (70.0, 48.37)), 48.0)
        lab.set_reynolds(2000)
        while lab.time < 12:
            lab.step(200, dye=False, particles=False)
        cls.append(lab.mean_coefficients(4)[0])
    assert cls[0] < -0.2 and cls[2] > 0.2, cls
    assert abs(cls[1]) < 0.1, cls


def test_paint_and_reset_and_tracers() -> None:
    lab = FlowLab2D(100, 40)
    lab.paint(50, 20, 5, True)
    assert lab.solid.sum() > 60
    lab.step(50)
    assert np.isfinite(lab.dye).all() and (lab.dye[:, 0] > 0).any()
    assert np.isfinite(lab.px).all()
    lab.paint(50, 20, 6, False)
    assert lab.solid.sum() == 0
    lab.reset()
    assert lab.steps == 0 and not lab.history.t
