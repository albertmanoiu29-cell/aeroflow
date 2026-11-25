"""Headless regression tests for the AeroFlow Studio core (no Qt / VTK window needed).

Run from the project folder::

    python -m pytest -q tests
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import FlowConditions, GeometryPreset, GridQuality  # noqa: E402
from core import accel  # noqa: E402
from core.geometry_manager import build_preset  # noqa: E402
from core.instant_solver import InstantSolver  # noqa: E402
from core.polar_generator import export_html_report, export_polar_csv, sweep_polar  # noqa: E402
from utils.validators import validate_sweep  # noqa: E402


@pytest.fixture(scope="module")
def solver() -> InstantSolver:
    return InstantSolver(warmup=False)


@pytest.fixture(scope="module")
def wing():
    return build_preset(GeometryPreset.NACA_2412)


# ---------------------------------------------------------------------------
# Field kernels: compiled and NumPy paths must agree
# ---------------------------------------------------------------------------
def test_source_kernel_matches_numpy() -> None:
    rng = np.random.default_rng(1)
    nodes, pos = rng.normal(size=(500, 3)), rng.normal(size=(80, 3))
    a2, w = rng.uniform(0.01, 0.1, 80), rng.normal(size=80)
    ref = np.zeros((500, 3))
    accel._source_np(nodes, pos, a2, w, ref)
    got = accel.source_velocity(nodes, pos, a2, w)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


def test_horseshoe_kernel_matches_numpy() -> None:
    rng = np.random.default_rng(2)
    nodes = rng.normal(size=(400, 3))
    a = rng.normal(size=(30, 3))
    b = a + rng.normal(scale=0.3, size=(30, 3))
    g = rng.normal(size=30)
    d = np.array([math.cos(0.1), 0.0, math.sin(0.1)])
    ref = np.zeros((400, 3))
    accel._horseshoe_np(nodes, a, b, g, d, 1e-3, ref)
    got = accel.horseshoe_velocity(nodes, a, b, g, d, 1e-3)
    np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-10)


def test_horseshoe_far_field_is_small() -> None:
    """A unit horseshoe induces almost nothing very far upstream."""
    v = accel.horseshoe_velocity(np.array([[-1e4, 0.0, 0.0]]), np.array([[0.0, -0.5, 0.0]]),
                                 np.array([[0.0, 0.5, 0.0]]), np.array([1.0]), np.array([1.0, 0.0, 0.0]), 1e-6)
    assert np.linalg.norm(v) < 1e-7


# ---------------------------------------------------------------------------
# Solver physics
# ---------------------------------------------------------------------------
def test_sphere_matches_potential_flow(solver: InstantSolver) -> None:
    """Surface Cp on a sphere vs the exact 1 - 9/4 sin^2(theta)."""
    sphere = build_preset(GeometryPreset.SPHERE, tail_ratio=0.0)
    res = solver.solve(sphere, FlowConditions(alpha_deg=0.0, velocity_ms=30.0), GridQuality.MEDIUM,
                       compute_field=False)
    v = sphere.vertices - sphere.bbox_center
    cos_t = v[:, 0] / np.linalg.norm(v, axis=1)
    exact = 1.0 - 2.25 * (1.0 - cos_t ** 2)
    front = cos_t < 0.3        # windward side (the leeward side carries the separation model)
    rms = float(np.sqrt(np.mean((res.surface.cp[front] - exact[front]) ** 2)))
    assert rms < 0.15, rms
    assert abs(res.kpis.cl) < 1e-3


def test_wing_lift_and_drag_are_plausible(solver: InstantSolver, wing) -> None:
    k = solver.solve(wing, FlowConditions(alpha_deg=5.0), GridQuality.COARSE, compute_field=False).kpis
    assert 0.45 < k.cl < 0.65            # AR 6 cambered wing at 5 deg
    assert 0.01 < k.cd < 0.05
    e = k.cl ** 2 / (math.pi * k.aspect_ratio * max(k.cd_induced, 1e-9))
    assert 0.7 < e < 1.15                # Oswald span efficiency of a near-elliptic loading


def test_lift_curve_slope_and_zero_lift_angle(solver: InstantSolver, wing) -> None:
    prep = solver.prepare(wing, GridQuality.COARSE)
    cl0 = solver.coefficients(prep, FlowConditions(alpha_deg=0.0)).cl
    cl4 = solver.coefficients(prep, FlowConditions(alpha_deg=4.0)).cl
    slope = (cl4 - cl0) / math.radians(4.0)
    assert 3.5 < slope < 5.2             # finite-wing lift slope, per radian
    assert cl0 > 0.1                     # 2 % camber gives positive lift at zero incidence


def test_pitching_moment_breaks_nose_down_after_stall(solver: InstantSolver, wing) -> None:
    prep = solver.prepare(wing, GridQuality.COARSE)
    attached = solver.coefficients(prep, FlowConditions(alpha_deg=4.0)).cm
    stalled = solver.coefficients(prep, FlowConditions(alpha_deg=22.0)).cm
    assert stalled < attached - 0.02


def test_symmetric_wing_has_zero_lift_at_zero_alpha(solver: InstantSolver) -> None:
    w = build_preset(GeometryPreset.NACA_0012)
    k = solver.solve(w, FlowConditions(alpha_deg=0.0), GridQuality.COARSE, compute_field=False).kpis
    assert abs(k.cl) < 0.02


def test_volume_field_is_consistent(solver: InstantSolver, wing) -> None:
    res = solver.solve(wing, FlowConditions(alpha_deg=5.0, velocity_ms=40.0), GridQuality.COARSE)
    f = res.field
    n = f.dims[0] * f.dims[1] * f.dims[2]
    assert f.velocity is not None and f.velocity.shape == (n, 3)
    assert np.isfinite(f.velocity).all() and np.isfinite(f.cp).all()
    # The far upstream corner sees essentially the freestream.
    np.testing.assert_allclose(np.linalg.norm(f.velocity[0]), 40.0, rtol=0.05)


def test_prepare_cache_respects_quality(solver: InstantSolver, wing) -> None:
    a = solver.prepare(wing, GridQuality.COARSE)
    b = solver.prepare(wing, GridQuality.MEDIUM)
    assert a is not b and a.quality != b.quality
    assert solver.prepare(wing, GridQuality.COARSE) is a        # still cached (LRU, not one slot)
    res = solver.solve(wing, FlowConditions(), GridQuality.COARSE, prepared=b, compute_field=False)
    assert res.extra["n_surfels"] == a.n_surfels                # a MEDIUM prep is not reused for COARSE


def test_linear_basis_matches_direct_solve(solver: InstantSolver, wing) -> None:
    prep = solver.prepare(wing, GridQuality.COARSE)
    with_basis = solver.coefficients(prep, FlowConditions(alpha_deg=7.0, beta_deg=3.0))
    direct = replace(prep, sig_basis=None, vel_basis=None)
    without = solver.coefficients(direct, FlowConditions(alpha_deg=7.0, beta_deg=3.0))
    assert with_basis.cl == pytest.approx(without.cl, rel=1e-9, abs=1e-12)
    assert with_basis.cd == pytest.approx(without.cd, rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------------------
# Polars and reports
# ---------------------------------------------------------------------------
def test_polar_sweep_and_exports(solver: InstantSolver, wing, tmp_path: Path) -> None:
    polar = sweep_polar(solver, wing, FlowConditions(), -4.0, 16.0, 2.0, GridQuality.COARSE)
    assert len(polar.alpha_deg) == 11
    assert polar.cl_max > 0.8
    assert -4.0 < polar.alpha_zero_lift_deg < 0.0
    csv_path = export_polar_csv(polar, tmp_path / "polar.csv")
    assert csv_path.read_text(encoding="utf-8").count("\n") == 13
    html = export_html_report(tmp_path / "r.html", polar=polar, title="Test").read_text(encoding="utf-8")
    assert "<svg" in html and "Instant Aero Engine" in html
    # Drag-polar x ticks must not all collapse to "0" (CD spans ~0.01-0.1): they must be distinct.
    import re
    svgs = re.findall(r"<svg.*?</svg>", html)
    drag_svg = next(s for s in svgs if "drag polar" in s)
    x_ticks = re.findall(r'text-anchor="middle">([-\d.]+)</text>', drag_svg)
    assert len(x_ticks) >= 6 and len(set(x_ticks)) == len(x_ticks), x_ticks


def test_report_footer_names_openfoam(tmp_path: Path) -> None:
    out = export_html_report(tmp_path / "of.html", title="OF", engine="OpenFOAM (k-ω SST)")
    assert "OpenFOAM" in out.read_text(encoding="utf-8")


def test_validate_sweep_swaps_and_caps() -> None:
    angles, res = validate_sweep(10.0, -2.0, 1.0)
    assert angles[0] == -2.0 and angles[-1] == 10.0
    assert res.warnings


@pytest.mark.parametrize("preset", list(GeometryPreset))
def test_every_preset_solves(solver: InstantSolver, preset: GeometryPreset) -> None:
    model = build_preset(preset)
    k = solver.solve(model, FlowConditions(alpha_deg=3.0), GridQuality.COARSE, compute_field=False).kpis
    assert all(math.isfinite(x) for x in (k.cl, k.cd, k.cm))
    assert k.cd > 0.0


# ---------------------------------------------------------------------------
# Analysis extras: span loading and derived visual fields
# ---------------------------------------------------------------------------
def test_span_load_integrates_to_wing_cl(solver: InstantSolver, wing) -> None:
    res = solver.solve(wing, FlowConditions(alpha_deg=5.0), GridQuality.COARSE, compute_field=False)
    ex = res.extra
    assert {"span_eta", "span_cl", "span_chord", "span_y"} <= set(ex)
    eta = np.asarray(ex["span_eta"])
    assert -1.0 <= eta.min() < -0.8 and 0.8 < eta.max() <= 1.0
    trapz = getattr(np, "trapezoid", None) or np.trapz
    y, chord, cl = (np.asarray(ex[k]) for k in ("span_y", "span_chord", "span_cl"))
    cl_int = float(trapz(cl * chord, y)) / res.kpis.ref_area_m2
    assert cl_int == pytest.approx(res.kpis.cl, rel=0.15)
    mid = np.argmin(np.abs(eta))
    assert cl[mid] > cl[0] and cl[mid] > cl[-1]                    # loading falls toward the tips


def test_derived_visual_fields(solver: InstantSolver, wing) -> None:
    res = solver.solve(wing, FlowConditions(alpha_deg=8.0, velocity_ms=30.0), GridQuality.COARSE)
    ds = res.field.prepare_visual()
    for name in ("vorticity", "vorticity_mag", "qcriterion", "Cp0"):
        assert name in ds.point_data, name
    cp0 = np.asarray(ds.point_data["Cp0"])
    solid = np.asarray(ds.point_data["solid"]) > 0
    assert abs(float(cp0[0]) - 1.0) < 0.05                          # loss-free far upstream
    assert float(cp0[~solid].min()) < 0.99                          # the (thin) wing wake loses total pressure
    assert (np.asarray(ds.point_data["qcriterion"]) > 0).any()      # tip / trailing vortex cores exist
    assert res.field.prepare_visual() is ds                         # cached


def test_polar_stability_neutral_point() -> None:
    """Cm = Cm0 - h*CL about the CG gives a static margin h (and a neutral point h*c behind the CG)."""
    from core.results import PolarData

    a = np.arange(-6.0, 14.0, 1.0)
    cl = 0.1 * (a + 2.0)
    polar = PolarData(alpha_deg=a, cl=cl, cd=0.02 + 0.05 * cl ** 2, cm=0.05 - 0.12 * cl, ref_length_m=2.0,
                      moment_ref=(1.0, 0.0, 0.0))
    st = polar.stability()
    assert st is not None and st.stable
    assert st.static_margin == pytest.approx(0.12, abs=1e-6)
    assert st.x_neutral_point_m == pytest.approx(1.0 + 0.12 * 2.0, abs=1e-6)
    assert st.cl_alpha_per_deg == pytest.approx(0.1, abs=1e-9)
    assert st.alpha_trim_deg == pytest.approx(0.05 / 0.012 - 2.0, abs=0.05)   # Cm = 0 where CL = 0.05/0.12


def test_ground_effect_cuts_induced_drag() -> None:
    """Image-method ground effect vs the Prandtl/Wieselsberger fit phi = 1 - exp(-2.48 (2h/b)^0.768)."""
    from config.settings import FlowConditions, GeometryPreset, GridQuality
    from core.geometry_manager import build_preset
    from core.instant_solver import InstantSolver

    s = InstantSolver(warmup=False)
    m = build_preset(GeometryPreset.NACA_2412)
    prep = s.prepare(getattr(m, "model", m), GridQuality.COARSE)
    free = s.coefficients(prep, FlowConditions(alpha_deg=5.0))
    k_free = free.cd_induced / free.cl ** 2
    for h in (0.2, 0.5):
        g = s.coefficients(prep, FlowConditions(alpha_deg=5.0, ground_h_over_b=h))
        phi = (g.cd_induced / g.cl ** 2) / k_free
        ref = 1.0 - math.exp(-2.48 * (2.0 * h) ** 0.768)
        assert g.cl > free.cl
        assert phi == pytest.approx(ref, abs=0.06)


def test_flap_deflection_adds_lift_and_keeps_mesh_closed() -> None:
    from config.settings import FlowConditions, GeometryPreset, GridQuality
    from core.geometry_manager import build_preset, deflect_flap
    from core.instant_solver import InstantSolver

    s = InstantSolver(warmup=False)
    m = build_preset(GeometryPreset.NACA_2412)
    m = getattr(m, "model", m)
    flapped = deflect_flap(m, 15.0, 0.75, 0.0, 0.6)
    assert flapped.is_watertight and flapped.triangle_count == m.triangle_count
    cond = FlowConditions(alpha_deg=4.0)
    clean = s.coefficients(s.prepare(m, GridQuality.COARSE), cond)
    flap = s.coefficients(s.prepare(flapped, GridQuality.COARSE), cond)
    assert flap.cl > clean.cl + 0.2 and flap.cm < clean.cm          # more lift, nose-down moment


def test_richardson_recovers_order_and_limit() -> None:
    from core.convergence import richardson

    def f(n: float) -> float:
        return 2.0 - 0.8 * (n ** -0.5) ** 2          # second order in h = N^-1/2

    r = richardson("x", f(9000), f(2500), f(600), 9000, 2500, 600)
    assert r.observed_order == pytest.approx(2.0, abs=1e-6)
    assert r.extrapolated == pytest.approx(2.0, abs=1e-9)
    assert 0.0 < r.gci_fine < 0.01


def test_main_window_imports() -> None:
    """Guards against syntax errors in the UI modules (the other tests do not import them)."""
    import importlib

    for mod in ("ui.main_window", "ui.cfd_panel", "ui.param_study", "ui.section_cp", "ui.probe_dialog", "ui.compare",
                "ui.updates_ui", "ui.dialogs", "ui.supersonic_window", "ui.flowlab_panel"):
        importlib.import_module(mod)


def test_project_round_trip(tmp_path) -> None:
    import json
    import zipfile

    from config.settings import FlowConditions, GeometryPreset
    from core.geometry_manager import build_preset
    from core.project import ProjectError, ProjectState, load_project, save_project
    from core.results import PolarData

    m = build_preset(GeometryPreset.NACA_2412)
    m = getattr(m, "model", m)
    pol = PolarData(alpha_deg=np.array([0.0, 4.0, 8.0]), cl=np.array([0.2, 0.6, 0.9]), cd=np.array([0.01, 0.02, 0.04]),
                    cm=np.array([-0.05, -0.06, -0.07]), moment_ref=(0.25, 0.0, 0.0), geometry_name=m.name)
    st = ProjectState(conditions=FlowConditions(velocity_ms=80.0, alpha_deg=3.0, ground_h_over_b=0.2), quality="Fine (detailed)",
                      camera=[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], polar=pol)
    out = save_project(tmp_path / "wing", m, st)
    assert out.suffix == ".aeroflow"
    m2, st2 = load_project(out, tmp_path / "scratch")
    assert m2.triangle_count == m.triangle_count and m2.name == m.name
    assert st2.conditions.velocity_ms == 80.0 and st2.conditions.ground_h_over_b == 0.2
    assert st2.quality == "Fine (detailed)" and st2.camera == st.camera
    assert np.allclose(st2.polar.cl, pol.cl) and st2.polar.moment_ref == (0.25, 0.0, 0.0)
    # a project from a newer format is refused clearly
    newer = tmp_path / "newer.aeroflow"
    with zipfile.ZipFile(newer, "w") as zf:
        zf.writestr("project.json", json.dumps({"format": 99, "app_version": "9.0"}))
        zf.writestr("geometry.stl", b"")
    with pytest.raises(ProjectError, match="newer"):
        load_project(newer, tmp_path / "scratch")


def test_openfoam_compressible_case_files(tmp_path) -> None:
    from config.settings import FlowConditions, GeometryPreset, GridQuality
    from core import openfoam_bridge as ob
    from core.geometry_manager import build_preset

    m = build_preset(GeometryPreset.NACA_0012)
    m = getattr(m, "model", m)
    st = ob.CaseSettings(conditions=FlowConditions(velocity_ms=200.0, altitude_m=5000.0), quality=GridQuality.COARSE,
                         iterations=50, compressible=True)
    lay = ob.write_case(m, st, tmp_path / "case")
    d = lay.case_dir
    phys = (d / "constant/physicalProperties").read_text()
    assert "hePsiThermo" in phys and "perfectGas" in phys
    p = (d / "0/p").read_text()
    assert "freestreamPressure" in p and "[1 -1 -2 0 0 0 0]" in p          # absolute pressure in Pa
    t_inf = st.conditions.atmosphere.temperature_k
    assert f"{t_inf:.9g}" in (d / "0/T").read_text()
    assert "compressible::alphatWallFunction" in (d / "0/alphat").read_text()
    ctrl = (d / "system/controlDict").read_text()
    assert "solver          fluid;" in ctrl and "rho             rho;" in ctrl
    assert "freestreamVelocity" in (d / "0/U").read_text()
    # incompressible stays as it was
    lay2 = ob.write_case(m, ob.CaseSettings(conditions=FlowConditions()), tmp_path / "inc")
    assert "incompressibleFluid" in (lay2.case_dir / "system/controlDict").read_text()
    assert not (lay2.case_dir / "0/T").exists()
