"""Rocket Nozzle Lab design tools: mixture ratio, heat transfer, separation, optimiser, noise."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import nozzle as nz  # noqa: E402
from core import nozzle_design as nd  # noqa: E402

KEROLOX = "LOX / RP-1 (kerosene)"
GEOM = nz.NozzleGeometry(throat_diameter=0.26, exit_diameter=1.04, chamber_diameter=0.42, chamber_length=0.15)


def test_mixture_ratio_tables() -> None:
    lo, hi = nd.of_range(KEROLOX)
    assert lo < nd.DEFAULT_OF[KEROLOX] < hi
    g = nd.gas_at_mixture_ratio(KEROLOX, 2.6)
    assert (g.chamber_temperature, g.molar_mass, g.gamma) == pytest.approx((3670.0, 23.3, 1.22))
    mid = nd.gas_at_mixture_ratio(KEROLOX, 2.45)                     # linear between table points
    assert mid.chamber_temperature == pytest.approx(0.5 * (3570.0 + 3670.0))
    assert nd.gas_at_mixture_ratio(KEROLOX, 99.0).chamber_temperature == pytest.approx(3600.0)   # clipped
    assert nd.of_range("APCP solid (AP / Al / HTPB)") is None
    sw = nd.mixture_sweep(KEROLOX, GEOM, 97e5, 101325.0)
    # the best Isp is fuel-rich of the hottest mixture (a lighter exhaust)
    assert sw.best_of < float(sw.of[int(np.argmax(sw.tc))])
    assert np.all(sw.isp_vac > sw.isp)


def test_bartz_heat_transfer() -> None:
    gas = nd.gas_at_mixture_ratio(KEROLOX, 2.6)
    regen = nd.bartz_heat_transfer(GEOM, gas, 97e5, "regenerative", 800.0)
    assert abs(regen.x_peak - GEOM.throat_x()) < 0.05 * GEOM.length()   # peak heating at the throat
    assert 20e6 < regen.q_peak < 150e6                                  # tens of MW/m², typical of boosters
    assert np.all(regen.t_wall <= regen.t_aw + 1e-9) and np.all(regen.q >= 0)
    rad = nd.bartz_heat_transfer(GEOM, gas, 97e5, "radiation", emissivity=0.8)
    # radiative equilibrium: convective input = emitted radiation
    assert np.allclose(rad.q, 0.8 * nd.SIGMA_SB * rad.t_wall ** 4, rtol=1e-3)
    assert np.all(rad.t_wall > regen.t_wall - 1e-9)
    mixed = nd.bartz_heat_transfer(GEOM, gas, 97e5, "mixed", 800.0, 0.8, extension_ratio=6.0)
    eps = (np.interp(mixed.x, *GEOM.inner_contour()) / GEOM.rt) ** 2
    ext = (mixed.x > GEOM.throat_x()) & (eps >= 6.0)
    assert np.allclose(mixed.t_wall[~ext], 800.0) and np.all(mixed.t_wall[ext] > 800.0)
    adi = nd.bartz_heat_transfer(GEOM, gas, 97e5, "adiabatic")
    assert np.allclose(adi.q, 0.0)


def test_separation_criteria() -> None:
    assert nd.schmucker_separation_ratio(3.0) == pytest.approx((1.88 * 3.0 - 1.0) ** -0.64)
    gas = nd.gas_at_mixture_ratio(KEROLOX, 2.6)
    assert nd.separation_point(GEOM, gas, 97e5, 101325.0) is None          # a booster nozzle stays attached
    long = nz.NozzleGeometry(throat_diameter=0.1, exit_diameter=1.0, chamber_diameter=0.2)  # ε = 100 at sea level
    x = nd.separation_point(long, gas, 50e5, 101325.0)
    assert x is not None and long.throat_x() < x < long.length()


def test_optimiser() -> None:
    gas = nd.gas_at_mixture_ratio(KEROLOX, 2.6)
    pa = 101325.0
    single = nd.optimise_expansion(GEOM, gas, 97e5, 0.0, 0.0)
    assert single.expansion_ratio == pytest.approx(nz.nozzle_theory(GEOM, gas, 97e5, pa).optimum_expansion_ratio,
                                                   rel=0.06)
    assert single.score >= single.current_score - 1e-6
    climb = nd.optimise_expansion(GEOM, gas, 97e5, 0.0, 40.0)
    assert climb.expansion_ratio > single.expansion_ratio                    # an ascent wants a longer nozzle
    capped = nd.optimise_expansion(GEOM, gas, 97e5, 0.0, 40.0, max_exit_diameter=0.8)
    assert capped.exit_diameter <= 0.8 + 1e-9


def test_jet_noise() -> None:
    gas = nd.gas_at_mixture_ratio(KEROLOX, 2.6)
    th = nz.nozzle_theory(GEOM, gas, 97e5, 101325.0)
    ns = nd.jet_noise(th, 2.0 * GEOM.re, 340.0, 100.0, 0.005)
    assert ns.acoustic_power == pytest.approx(0.005 * 0.5 * th.thrust * th.exit_velocity)
    assert ns.pwl == pytest.approx(10 * math.log10(ns.acoustic_power / 1e-12))
    assert 30.0 < ns.peak_angle < 60.0                                       # loudest ~45° off the exhaust axis
    far = nd.jet_noise(th, 2.0 * GEOM.re, 340.0, 1000.0, 0.005)
    assert float(ns.oaspl.max() - far.oaspl.max()) == pytest.approx(20.0)    # inverse-square law


def test_film_cooling_lowers_heating_at_an_isp_cost() -> None:
    import math as _m

    from core import nozzle as _nz
    from core import nozzle_design as _nd

    g = _nz.NozzleGeometry(throat_diameter=0.26, exit_diameter=1.04, chamber_diameter=0.42, chamber_length=0.15)
    gas = _nz.Gas(1.22, 23.3, 3670.0)
    base = _nd.bartz_heat_transfer(g, gas, 97e5, "regenerative")
    film = _nd.bartz_heat_transfer(g, gas, 97e5, "regenerative", film_fraction=0.02, film_temperature=400.0)
    assert 0.3 < 1.0 - film.q_peak / base.q_peak < 0.6          # 2 % film: a 30-60 % cut at the throat
    assert film.isp_loss == pytest.approx(0.02 * (1 - _m.sqrt(400 / 3670)))
    assert float(film.film_effectiveness[0]) == 1.0 and float(film.film_effectiveness[-1]) < 0.05
    assert base.isp_loss == 0.0
