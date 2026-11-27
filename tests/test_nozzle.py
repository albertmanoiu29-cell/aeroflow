"""Rocket Nozzle Lab: 1-D theory, geometry, and the axisymmetric cut-cell Euler solver."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import accel, compute  # noqa: E402
from core import nozzle as nz  # noqa: E402
from core.compute import Device  # noqa: E402

AIR = nz.Gas(1.4, 28.97, 288.15)


@pytest.fixture(autouse=True)
def _cpu():
    before = compute.preference()
    compute.set_preference(Device.CPU)
    yield
    compute.set_preference(before)


def _conical(eps: float = 4.0) -> nz.NozzleGeometry:
    return nz.NozzleGeometry(throat_diameter=0.05, exit_diameter=0.05 * math.sqrt(eps), chamber_diameter=0.10,
                             chamber_length=0.03, contour=nz.Contour.CONICAL, divergent_angle=15.0,
                             wall_thickness=0.004)


def test_isentropic_relations() -> None:
    g = 1.22
    for m in (0.3, 1.0, 2.5, 4.0):
        eps = nz.area_mach(m, g)
        assert nz.mach_from_area_ratio(eps, g, supersonic=m >= 1.0) == pytest.approx(m, rel=1e-9)
        assert nz.mach_from_pressure_ratio(nz.p_over_p0(m, g), g) == pytest.approx(m, rel=1e-9)
    assert nz.area_mach(1.0, 1.4) == pytest.approx(1.0)
    assert nz.critical_pressure_ratio(1.4) == pytest.approx(0.528282, rel=1e-5)
    assert nz.normal_shock_p0_ratio(2.0, 1.4) == pytest.approx(0.72087, rel=1e-4)


def test_thrust_coefficient_matches_closed_form() -> None:
    geom = _conical(4.0)
    pc, pa = 30e5, 1e5
    th = nz.nozzle_theory(geom, AIR, pc, pa)
    g = 1.4
    pr = th.exit_pressure / pc
    cf = (math.sqrt(2 * g * g / (g - 1) * (2 / (g + 1)) ** ((g + 1) / (g - 1)) * (1 - pr ** ((g - 1) / g)))
          + (th.exit_pressure - pa) / pc * th.expansion_ratio)
    assert th.thrust_ideal / (pc * th.throat_area) == pytest.approx(cf, rel=1e-9)
    assert th.mass_flow == pytest.approx(pc * th.throat_area / nz.characteristic_velocity(AIR), rel=1e-12)
    assert th.exit_mach == pytest.approx(2.9402, abs=1e-4)
    assert th.divergence_factor == pytest.approx(0.5 * (1 + math.cos(math.radians(15.0))))
    assert th.regime == "Overexpanded"
    assert th.isp == pytest.approx(th.thrust / (th.mass_flow * nz.G0))
    # vacuum: no back pressure, more thrust; at the design back pressure the nozzle is ideally expanded
    vac = nz.nozzle_theory(geom, AIR, pc, 1.0)
    assert vac.regime == "Underexpanded" and vac.thrust > th.thrust
    ideal = nz.nozzle_theory(geom, AIR, pc, th.exit_pressure)
    assert ideal.regime == "Ideally expanded"
    assert ideal.optimum_expansion_ratio == pytest.approx(4.0, rel=1e-6)


def test_regimes_shock_and_separation() -> None:
    geom = _conical(4.0)
    # a little too little pressure for a shock-free nozzle: 1-D normal shock in the divergent section
    th = nz.nozzle_theory(geom, AIR, 3.0e5, 1.0e5)
    assert th.regime == "Shock inside the nozzle" and th.choked
    assert th.shock_x is not None and geom.throat_x() < th.shock_x < geom.length()
    assert th.exit_pressure == pytest.approx(1.0e5)
    # far too little (ε = 4 stays choked down to p_c ≈ 1.015 p_a): unchoked
    assert nz.nozzle_theory(geom, AIR, 1.01e5, 1.0e5).regime == "Unchoked (subsonic)"
    assert nz.nozzle_theory(geom, AIR, 1.02e5, 1.0e5).choked
    # Summerfield: separation when p_exit < 0.4 p_a
    sep = nz.nozzle_theory(geom, AIR, 8e5, 1.0e5)
    assert sep.separation and sep.separation_x is not None
    # sonic orifice: Ashkenas-Sherman Mach disk at 0.67 D sqrt(p0/pa)
    son = nz.NozzleGeometry(throat_diameter=0.02, chamber_diameter=0.05, contour=nz.Contour.SONIC)
    t = nz.nozzle_theory(son, AIR, 20e5, 1e5)
    assert t.expansion_ratio == pytest.approx(1.0)
    assert t.mach_disk_distance == pytest.approx(0.67 * 0.02 * math.sqrt(20))


def test_contours() -> None:
    for contour in nz.Contour:
        g = nz.NozzleGeometry(contour=contour)
        x, r = g.inner_contour()
        assert np.all(np.diff(x) > -1e-12)                         # x never goes backwards
        assert r.min() == pytest.approx(g.rt, rel=1e-6)            # the throat is the narrowest point
        assert r[-1] == pytest.approx(g.re, rel=1e-6) and r[0] == pytest.approx(g.rc, rel=1e-6)
    tn, te = nz.rao_angles(10.0, 0.8)
    assert 20.0 < tn < 35.0 and 5.0 < te < 15.0 and te < tn
    bell = nz.NozzleGeometry(contour=nz.Contour.BELL)
    cone = nz.NozzleGeometry(contour=nz.Contour.CONICAL)
    assert bell.length() < cone.length()                            # an 80 % bell is shorter than a 15° cone


def test_layout_cut_cells() -> None:
    lay = nz.layout_nozzle(_conical(4.0), 240, 60, plume_length=6.0, plume_radius=3.0)
    assert lay.n_inlet > 5 and 0 < lay.i_throat < lay.i_exit < 240
    assert 0.0 <= lay.kappa.min() and lay.kappa.max() <= 1.0
    assert ((lay.kappa > 0.0) & (lay.kappa < 1.0)).any()
    # the open area of the throat face matches the true area there (sub-cell geometry)
    f = int(round(lay.x_throat / lay.dx))
    r_face = float(lay.r_inner(f * lay.dx))
    j = np.arange(60)
    inside = j * lay.dx < r_face                                    # rows of the bore (not the air outside)
    open_area = float((lay.ax[inside, f] * 2 * math.pi * (j[inside] + 0.5)).sum()) * lay.dx ** 2
    assert open_area == pytest.approx(math.pi * r_face ** 2, rel=0.03)


@pytest.mark.parametrize("axi", [True, False])
def test_gas_at_rest_stays_at_rest(axi: bool) -> None:
    lay = nz.layout_nozzle(_conical(4.0), 160, 40, plume_length=4.0, plume_radius=3.0)
    s = nz.NozzleFlow(160, 40, gamma=1.4, axisymmetric=axi)
    s.set_geometry(lay)
    s.set_conditions(1 / 1.4, 1.0, 0.0, 1 / 1.4, 1.0)
    s.reset()
    s.step(40)
    _, u, v, _ = s.primitives()
    assert np.max(np.abs(u[~s.solid_int])) < 1e-12 and np.max(np.abs(v[~s.solid_int])) < 1e-12
    sc = nz.Scales(lay.dx, 1.0, 1.0, 1.0)
    assert abs(nz.measure(s, lay, sc, _conical(4.0))["thrust"]) < 1e-9


@pytest.mark.skipif(not accel.HAVE_NUMBA, reason="needs the compiled CPU kernels for speed")
def test_steady_nozzle_conserves_mass_and_matches_theory() -> None:
    geom = _conical(4.0)
    pc, pa, ta = 30e5, 1e5, 288.15
    lay = nz.layout_nozzle(geom, 240, 60, plume_length=4.0, plume_radius=2.6)
    s = nz.NozzleFlow(240, 60, gamma=1.4)
    s.set_geometry(lay)
    rg = AIR.gas_constant
    sc = nz.Scales(lay.dx, pa / (rg * ta), math.sqrt(1.4 * rg * ta), ta)
    s.set_conditions(pa / sc.pressure, 1.0, 0.0, pc / sc.pressure, 1.0)
    s.reset()
    s.ignite(10.0)
    s.step(2500)
    assert np.isfinite(s.U).all()
    m = nz.measure(s, lay, sc, geom)
    th = nz.nozzle_theory(geom, AIR, pc, pa)
    assert m["chamber_pressure"] == pytest.approx(pc, rel=0.01)
    assert m["mdot_exit"] == pytest.approx(m["mdot_throat"], rel=0.03)          # conservative cut cells
    assert m["mdot_exit"] == pytest.approx(th.mass_flow, rel=0.05)
    assert m["thrust"] == pytest.approx(m["thrust_exit_plane"], rel=0.03)        # wall force = momentum flux
    assert m["thrust"] == pytest.approx(th.thrust, rel=0.07)
    assert m["exit_mach"] == pytest.approx(th.exit_mach, rel=0.06)


def test_ignition_ramp_and_shutdown() -> None:
    bc = np.zeros(nz.BC_SIZE)
    bc[nz.BC_TIGN], bc[nz.BC_TAU], bc[nz.BC_TOFF] = 10.0, 4.0, nz.NEVER
    assert nz.ramp(bc, 5.0) == 0.0 and nz.ramp(bc, 12.0) == pytest.approx(0.5) and nz.ramp(bc, 20.0) == 1.0
    bc[nz.BC_TOFF], bc[nz.BC_TAUOFF], bc[nz.BC_FOFF] = 30.0, 2.0, 1.0
    assert nz.ramp(bc, 31.0) == pytest.approx(0.5) and nz.ramp(bc, 40.0) == 0.0


def test_axis_features_detect_mach_disk() -> None:
    x = np.linspace(0.0, 1.0, 501)
    mach = np.where(x < 0.1, 0.5, np.where(x < 0.3, 1.2 + 10.0 * (x - 0.1), 0.4))
    p = np.where(x < 0.3, 1.0 / (1.0 + 30.0 * (x - 0.1).clip(0)), 3.0)
    f = nz.axis_features(x, mach, p, 1.0, 0.1, 0.02)
    assert f["mach_disk_x"] == pytest.approx(0.2, abs=0.01)


def test_export_results(tmp_path) -> None:
    from ui.nozzle_window import export_results
    from visualization.nozzle_view import EngineSetup

    setup = EngineSetup(geom=_conical(4.0), gas=AIR, pc=30e5, pa=1e5)
    hist = {k: [1.0, 2.0] for k in ("t", "thrust", "pc", "mdot", "mdot_throat", "throttle")}
    out = export_results(tmp_path / "r.csv", {"setup": setup, "meas": {}, "history": hist})
    text = out.read_text(encoding="utf-8")
    assert "time_ms,thrust_N" in text and "Dt=0.05 m" in text


def test_revolved_geometry_for_the_3d_view() -> None:
    from visualization.nozzle_3d import revolve, revolve_segments

    x = np.array([0.0, 1.0, 2.0])
    r = np.array([1.0, 0.5, 1.5])
    shell = revolve(x, r, 0.0, 2.0 * math.pi, 33)
    rad = np.hypot(shell.points[:, 1], shell.points[:, 2])
    assert np.allclose(np.sort(np.unique(np.round(rad, 9))), [0.5, 1.0, 1.5])
    # two segments revolved over 270 degrees: one quad per angular step per segment, radii preserved
    surf = revolve_segments(np.array([0.0, 1.0, 1.0, 2.0]), np.array([1.0, 1.0, 1.0, 2.0]), math.pi, 2.5 * math.pi, 10)
    assert surf.n_cells == 2 * 9
    pts = surf.points
    assert np.allclose(np.hypot(pts[:, 1], pts[:, 2])[pts[:, 0] == 0.0], 1.0)
    assert revolve_segments(np.zeros(0), np.zeros(0), 0.0, 1.0, 4) is None


def test_revolve_fill_is_linear_in_radius() -> None:
    from visualization.nozzle_3d import _fill_numpy, blackbody_rgb, fill_revolved

    vd = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)       # rows = radius
    rpos = np.array([[0.0, 0.5], [1.25, 5.0]], dtype=np.float32)                # fractional rows (z, y)
    a = np.zeros((2, 2, 2), np.float32)
    b = np.zeros_like(a)
    fill_revolved(vd, rpos, a)
    _fill_numpy(vd, rpos, b)
    assert np.allclose(a, b)
    assert np.allclose(a[0, 0], [0.0, 1.0]) and np.allclose(a[0, 1], [1.0, 2.0])
    assert np.allclose(a[1, 0], [2.5, 3.5]) and np.allclose(a[1, 1], 0.0)          # beyond the last row: empty
    hot, cold = blackbody_rgb(np.array([2500.0, 800.0]))
    assert hot[1] > cold[1] and hot[2] > cold[2]                                    # hotter is whiter


def test_dual_bell_switches_mode_with_altitude() -> None:
    gas = nz.Gas(1.21, 13.4, 3420.0)
    g = nz.NozzleGeometry(throat_diameter=0.2, exit_diameter=1.6, chamber_diameter=0.4, contour=nz.Contour.DUAL_BELL)
    x, r = g.inner_contour()
    assert np.all(np.diff(x) > -1e-12) and r[-1] == pytest.approx(g.re)
    e1, x1 = g.inflection()
    assert 2.0 <= e1 < g.area_ratio() and 0.0 < x1 < x[-1]
    low = nz.nozzle_theory(g, gas, 100e5, 101325.0)
    high = nz.nozzle_theory(g, gas, 100e5, 1000.0)
    assert "low-altitude" in low.regime and "high-altitude" in high.regime
    full_at_sea = nz._nozzle_theory(g, gas, 100e5, 101325.0)
    assert low.isp > full_at_sea.isp                   # the point of a dual bell: no overexpansion loss at sea level


def test_contour_import_csv_and_dxf(tmp_path) -> None:
    ref = nz.NozzleGeometry()
    xs, rs = ref.inner_contour()
    (tmp_path / "c.csv").write_text("x_mm,r_mm\n" + "\n".join(f"{a * 1000:.4f},{b * 1000:.4f}" for a, b in zip(xs, rs)))
    dxf = ["0", "SECTION", "2", "ENTITIES", "0", "LWPOLYLINE"]
    for a, b in zip(xs, rs):
        dxf += ["10", f"{a * 1000:.4f}", "20", f"{b * 1000:.4f}"]
    (tmp_path / "c.dxf").write_text("\n".join(dxf + ["0", "ENDSEC", "0", "EOF"]))
    for name in ("c.csv", "c.dxf"):
        x, r = nz.read_contour_file(tmp_path / name)
        g = nz.NozzleGeometry(contour=nz.Contour.CUSTOM, custom_x=tuple(x), custom_r=tuple(r))
        assert g.rt == pytest.approx(ref.rt, rel=1e-3) and g.re == pytest.approx(ref.re, rel=1e-3)
        gas = nz.Gas(1.22, 22.5, 3650.0)
        assert nz.nozzle_theory(g, gas, 70e5, 1e5).thrust == pytest.approx(nz.nozzle_theory(ref, gas, 70e5, 1e5).thrust,
                                                                           rel=2e-3)
    (tmp_path / "bad.csv").write_text("1,2\n3,4\n")
    with pytest.raises(ValueError):
        nz.read_contour_file(tmp_path / "bad.csv")
