"""Solid-motor internal ballistics, the ascent model and the nozzle geometry exports."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import nozzle as nz  # noqa: E402
from core import nozzle_export as ne  # noqa: E402
from core import solid_motor as sm  # noqa: E402
from core import trajectory as tj  # noqa: E402

APCP = sm.SOLID_PROPELLANTS["APCP (AP / Al / HTPB composite)"]
KNSB = sm.SOLID_PROPELLANTS["KNSB (KNO₃ / sorbitol)"]


def test_motor_class_letters() -> None:
    assert sm.motor_class(2.0) == "A" and sm.motor_class(2.6) == "B" and sm.motor_class(40.0) == "E"
    assert sm.motor_class(300.0) == "H" and sm.motor_class(0.5) == "1/2A"


def test_bates_grain_geometry() -> None:
    g = sm.BatesGrain(0.1, 0.03, 0.15, 2)
    a0 = 2 * (math.pi * 0.03 * 0.15 + 2 * 0.25 * math.pi * (0.1 ** 2 - 0.03 ** 2))
    assert g.burning_area(0.0) == pytest.approx(a0)
    assert g.web == pytest.approx(0.035)
    assert g.burning_area(g.web + 1e-6) == 0.0 and g.volume(g.web + 1e-6) == 0.0
    inh = sm.BatesGrain(0.1, 0.03, 0.15, 2, inhibit_ends=True)
    assert inh.burning_area(0.02) > inh.burning_area(0.0)            # core-only burning is progressive


def test_equilibrium_pressure_inverts_kn() -> None:
    cstar = 1500.0
    for pc in (2e6, 7e6):
        assert sm.equilibrium_pressure(sm.kn_for_pressure(pc, APCP, cstar), APCP, cstar) == pytest.approx(pc)


def test_knsb_38mm_motor_is_realistic() -> None:
    # a classic 38 mm, three-grain KNSB motor: published static tests give H-I class, Isp 120-135 s
    nozzle = nz.NozzleGeometry(throat_diameter=0.009, exit_diameter=0.022, chamber_diameter=0.038,
                               chamber_length=0.0, contour=nz.Contour.CONICAL)
    r = sm.simulate(sm.BatesGrain(0.032, 0.012, 0.06, 3), KNSB, nozzle)
    assert r.designation[0] in "HI"
    assert 110.0 < r.isp < 145.0
    assert 1e6 < r.peak_pressure < 8e6
    imp = float(np.sum(0.5 * (r.thrust[1:] + r.thrust[:-1]) * np.diff(r.t)))
    assert r.total_impulse == pytest.approx(imp)
    assert r.propellant_mass == pytest.approx(3 * 0.25 * math.pi * (0.032 ** 2 - 0.012 ** 2) * 0.06 * 1750.0)


def test_sizing_hits_the_chamber_pressure(tmp_path) -> None:
    geom = nz.NozzleGeometry(throat_diameter=0.15, exit_diameter=0.45, chamber_diameter=0.3,
                             contour=nz.Contour.CONICAL)
    grain = sm.BatesGrain(0.45, 0.24, 0.675, 1)
    n = sm.segments_for_pressure(grain, APCP, geom, None, 70e5)
    r = sm.simulate(sm.BatesGrain(0.45, 0.24, 0.675, n), APCP, geom)
    assert r.pc[0] == pytest.approx(70e5, rel=0.15)
    path = sm.write_eng(r, sm.BatesGrain(0.45, 0.24, 0.675, n), tmp_path / "m.eng")
    lines = [ln for ln in path.read_text().splitlines() if ln and not ln.startswith(";")]
    head = lines[0].split()
    assert head[0] == r.designation and float(head[4]) == pytest.approx(r.propellant_mass, rel=1e-3)
    pts = np.array([[float(v) for v in ln.split()] for ln in lines[1:]])
    assert np.all(np.diff(pts[:, 0]) > 0) and pts[-1, 1] == 0.0


def test_vertical_flight_matches_the_rocket_equation_in_vacuum() -> None:
    # no drag, constant thrust: burnout speed = Δv_ideal - g t_b (gravity barely changes over a few km)
    f, md = 20000.0, 20000.0 / (250.0 * tj.G0)
    v = tj.Vehicle(100.0, 400.0, 0.3, cd0=1e-9, launch_angle=90.0)
    fl = tj.fly(v, lambda t, h: (f, md), dt=0.01)
    tb = 400.0 / md
    dv = tj.ideal_delta_v(250.0, 500.0, 100.0)
    assert fl.burnout_time == pytest.approx(tb, rel=0.01)
    assert fl.dv_ideal == pytest.approx(dv, rel=0.01)
    assert fl.burnout_speed == pytest.approx(dv - tj.G0 * tb, rel=0.02)
    assert fl.apogee > fl.burnout_altitude and abs(fl.downrange_at_apogee) < 1.0


def test_flight_with_drag_and_a_pad_wait() -> None:
    heavy = tj.Vehicle(100.0, 900.0, 0.5, launch_angle=90.0)
    fl = tj.fly(heavy, lambda t, h: (0.8 * 1000.0 * tj.G0, 1000.0 / 60.0))
    assert "sat on the pad" in fl.message and fl.liftoff_twr < 1.0
    light = tj.Vehicle(100.0, 400.0, 0.3, launch_angle=85.0, turn_speed=50.0)
    fl2 = tj.fly(light, lambda t, h: (30000.0, 30000.0 / (230.0 * tj.G0)))
    assert fl2.drag_loss > 0.0 and fl2.max_q > 0.0 and fl2.downrange_at_apogee > 100.0


@pytest.mark.parametrize("contour", [nz.Contour.BELL, nz.Contour.CONICAL, nz.Contour.AEROSPIKE])
def test_exports(tmp_path, contour) -> None:
    import trimesh

    g = nz.NozzleGeometry(contour=contour)
    ne.write_csv(g, tmp_path / "c.csv")
    ne.write_dxf(g, tmp_path / "c.dxf")
    ne.write_stl(g, tmp_path / "c.stl", segments=96)
    mesh = trimesh.load(tmp_path / "c.stl")
    assert mesh.is_watertight and mesh.volume > 0.0
    assert mesh.bounds[1][0] == pytest.approx(ne.wall_profiles(g)[0][1].max() * 1e3, rel=0.05) or contour is \
        nz.Contour.AEROSPIKE
    if contour is not nz.Contour.AEROSPIKE:
        x, r = nz.read_contour_file(tmp_path / "c.dxf")                # only the INNER layer comes back
        xi, ri = g.inner_contour()
        assert float(r.min()) == pytest.approx(float(ri.min()), rel=1e-3)
        assert float(x[-1]) == pytest.approx(float(xi[-1]), rel=1e-3)
    with pytest.raises(ValueError):
        ne.write_stl(g, tmp_path / "p.stl", planar=True)


def test_orbit_from_state() -> None:
    r = tj.R_EARTH + 400e3
    v_circ = math.sqrt(tj.MU_EARTH / r)
    peri, apo = tj.orbit_from_state(400e3, v_circ, 0.0)
    assert peri == pytest.approx(400e3, abs=1.0) and apo == pytest.approx(400e3, abs=1.0)
    assert tj.orbit_from_state(400e3, 0.9 * v_circ, 0.0) is None            # perigee inside the atmosphere
    assert tj.orbit_from_state(400e3, 1.5 * v_circ, 0.0) is None            # escape


def test_two_stage_rocket_reaches_orbit() -> None:
    # a light, high-Isp two-stage vehicle with a horizontal upper-stage burn
    booster = lambda t, h: (1.2e6, 1.2e6 / (290.0 * tj.G0))                  # noqa: E731
    upper_engine = lambda t, h: (1.2e5, 1.2e5 / (340.0 * tj.G0))             # noqa: E731
    v = tj.Vehicle(6000.0, 60000.0, 2.0, engines=1, launch_angle=89.0, turn_speed=150.0)
    up = tj.UpperStage(1500.0, 14000.0, upper_engine, horizontal=True)
    fl = tj.fly(v, booster, up)
    assert fl.staging is not None and fl.staging[0] < fl.burnout_time
    assert fl.orbit is not None and fl.orbit[0] > 100e3 and "In orbit" in fl.message
    single = tj.fly(v, booster)                                              # no upper stage: suborbital
    assert single.orbit is None                                              # a high arc, not an orbit


def test_cross_section_grains() -> None:
    from core import grain_geometry as gg

    exact = sm.BatesGrain(0.1, 0.03, 0.15, 2)
    raster = gg.CrossSectionGrain("BATES (round core)", 0.1, 0.03, 0.15, 2)
    for w in (0.0, 0.01, 0.02, 0.03):
        assert raster.burning_area(w) == pytest.approx(exact.burning_area(w), rel=0.03)
        assert raster.volume(w) == pytest.approx(exact.volume(w), rel=0.02, abs=1e-6)
    assert raster.web == pytest.approx(exact.web, rel=0.02)
    star = gg.CrossSectionGrain("Star", 0.1, 0.03, 0.15, 1, inhibit_ends=True)
    bates = gg.CrossSectionGrain("BATES (round core)", 0.1, 0.03, 0.15, 1, inhibit_ends=True)
    assert star.burning_area(0.0) > 2.0 * bates.burning_area(0.0)      # a star starts with much more surface
    assert star.web < bates.web                                       # and burns out sooner
    nozzle = nz.NozzleGeometry(throat_diameter=0.015, exit_diameter=0.04, chamber_diameter=0.1,
                               contour=nz.Contour.CONICAL)
    r = sm.simulate(gg.CrossSectionGrain("Finocyl (core + fins)", 0.1, 0.03, 0.3, 1), APCP, nozzle)
    assert r.total_impulse > 0 and r.burn_time > 0
