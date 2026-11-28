"""Conceptual aircraft sizing: geometry, stability and performance against textbook relations."""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import aircraft_design as ad  # noqa: E402


def test_light_aircraft_is_cessna_like() -> None:
    d = ad.design(ad.AircraftSpec())
    assert 13.0 < d.wing_area < 17.0 and 9.5 < d.span < 11.5          # C172: 16.2 m², 11 m
    assert 23.0 < d.stall_speed < 30.0                                 # 48-55 kt
    assert 50.0 < d.v_max < 68.0                                       # ~ 110-125 kt
    assert 2.0 < d.best_climb < 5.5                                    # ~ 700 ft/min
    assert d.wing_area == pytest.approx(1100.0 / 75.0)
    assert d.span ** 2 / d.wing_area == pytest.approx(7.5)


def test_stability_responds_to_cg_and_tail() -> None:
    base = ad.design(ad.AircraftSpec())
    aft = ad.design(ad.AircraftSpec(x_cg=0.50))
    big_tail = ad.design(ad.AircraftSpec(v_h=0.9))
    assert base.static_margin > 0.05 and aft.static_margin < base.static_margin
    assert aft.static_margin == pytest.approx(base.neutral_point - 0.50)
    assert big_tail.neutral_point > base.neutral_point                 # more tail: neutral point moves aft
    assert base.cm_alpha < 0.0 and base.cm_q < 0.0


def test_polar_and_breguet() -> None:
    s = ad.AircraftSpec()
    d = ad.design(s)
    assert d.ld_max == pytest.approx(0.5 / math.sqrt(s.cd0 * d.k_induced))
    more_fuel = ad.design(replace(s, fuel_mass=300.0))
    assert more_fuel.range_km > 1.8 * d.range_km                       # ln(m0/m1) grows with the fuel
    jet = ad.design(ad.AircraftSpec(mass=9000, wing_loading=320, aspect_ratio=8.2, engine="jet",
                                    jet_thrust=31000, fuel_mass=2600, sfc=0.07, altitude=12000, cd0=0.020))
    assert 200.0 < jet.v_max < 300.0 and jet.range_km > 2000.0
