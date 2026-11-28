"""Rocket-engine design tools around :mod:`core.nozzle`: propellant chemistry against mixture ratio,
convective heat transfer (Bartz) and wall cooling, flow-separation criteria, an expansion-ratio optimiser
for an altitude or a whole ascent, and an engineering estimate of the exhaust noise.

All of it is quasi-1-D engineering analysis (the same level as the nozzle theory in :mod:`core.nozzle`):
fast enough to sweep, and labelled as estimates where the underlying correlations are empirical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from core import nozzle as nz

SIGMA_SB = 5.670374419e-8          # Stefan-Boltzmann constant, W/(m² K⁴)

# ---------------------------------------------------------------------------
# Propellant chemistry against mixture ratio
# ---------------------------------------------------------------------------
#: Approximate shifting-equilibrium chamber values at ~7 MPa (typical of published thermochemistry tables;
#: within a few per cent): mixture ratio O/F → (chamber temperature K, molar mass kg/kmol, gamma).
OF_TABLES: dict[str, dict[str, tuple[float, ...]]] = {
    "LOX / RP-1 (kerosene)": {"of": (1.6, 2.0, 2.3, 2.6, 2.9, 3.2, 3.6),
                              "tc": (2950, 3400, 3570, 3670, 3700, 3680, 3600),
                              "m": (19.2, 21.0, 22.2, 23.3, 24.2, 25.0, 26.0),
                              "g": (1.25, 1.23, 1.22, 1.22, 1.21, 1.21, 1.21)},
    "LOX / LH2 (hydrogen)": {"of": (3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
                             "tc": (2650, 2980, 3250, 3420, 3510, 3550),
                             "m": (8.0, 10.0, 11.8, 13.5, 15.0, 16.3),
                             "g": (1.27, 1.25, 1.22, 1.21, 1.20, 1.20)},
    "LOX / CH4 (methane)": {"of": (2.5, 3.0, 3.4, 3.8, 4.2),
                            "tc": (3250, 3480, 3550, 3570, 3530),
                            "m": (19.0, 20.6, 21.6, 22.6, 23.4),
                            "g": (1.23, 1.21, 1.21, 1.20, 1.20)},
    "N2O4 / MMH (hypergolic)": {"of": (1.4, 1.65, 2.0, 2.4),
                                "tc": (3150, 3300, 3380, 3370),
                                "m": (20.2, 21.5, 22.4, 23.6),
                                "g": (1.24, 1.23, 1.22, 1.22)},
}
#: The mixture ratio each propellant preset in :data:`core.nozzle.PROPELLANTS` corresponds to.
DEFAULT_OF: dict[str, float] = {"LOX / RP-1 (kerosene)": 2.6, "LOX / LH2 (hydrogen)": 6.0,
                                "LOX / CH4 (methane)": 3.4, "N2O4 / MMH (hypergolic)": 1.65}


def of_range(name: str) -> Optional[tuple[float, float]]:
    """The tabulated mixture-ratio range of a bipropellant (None for solids, monopropellants, cold gas)."""
    tab = OF_TABLES.get(name)
    return None if tab is None else (tab["of"][0], tab["of"][-1])


def gas_at_mixture_ratio(name: str, of: float) -> nz.Gas:
    """Chamber gas of bipropellant ``name`` at mixture ratio ``of`` (interpolated in the table)."""
    tab = OF_TABLES[name]
    x = np.asarray(tab["of"], dtype=float)
    of = float(np.clip(of, x[0], x[-1]))
    return nz.Gas(gamma=float(np.interp(of, x, tab["g"])), molar_mass=float(np.interp(of, x, tab["m"])),
                  chamber_temperature=float(np.interp(of, x, tab["tc"])))


@dataclass(frozen=True)
class MixtureSweep:
    of: np.ndarray
    tc: np.ndarray
    cstar: np.ndarray
    isp: np.ndarray            # at the given ambient pressure
    isp_vac: np.ndarray
    best_of: float             # mixture ratio of the highest Isp at the given ambient pressure


def mixture_sweep(name: str, geom: nz.NozzleGeometry, pc: float, pa: float, planar: bool = False,
                  n: int = 41) -> MixtureSweep:
    """Isp, vacuum Isp, c* and chamber temperature across the tabulated mixture-ratio range."""
    lo, hi = of_range(name)
    ofs = np.linspace(lo, hi, n)
    tc, cs, isp, ispv = [], [], [], []
    for of in ofs:
        gas = gas_at_mixture_ratio(name, of)
        th = nz.nozzle_theory(geom, gas, pc, pa, planar)
        tc.append(gas.chamber_temperature)
        cs.append(th.cstar)
        isp.append(th.isp)
        ispv.append(th.isp_vacuum)
    isp_a = np.asarray(isp)
    return MixtureSweep(ofs, np.asarray(tc), np.asarray(cs), isp_a, np.asarray(ispv), float(ofs[int(np.argmax(isp_a))]))


# ---------------------------------------------------------------------------
# Heat transfer (Bartz) and wall cooling
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HeatTransfer:
    """Gas-side heat transfer along the nozzle wall."""

    x: np.ndarray              # m from the injector face
    mach: np.ndarray           # 1-D Mach number
    h_g: np.ndarray            # W/(m² K), Bartz
    t_aw: np.ndarray           # adiabatic-wall (recovery) temperature, K
    t_wall: np.ndarray         # gas-side wall temperature, K
    q: np.ndarray              # heat flux into the wall, W/m²
    q_peak: float
    x_peak: float
    heat_load: float           # total heat into the wall, W (per metre of width for a slot nozzle)
    cooling: str
    film_effectiveness: np.ndarray = None   # 0 (no film) .. 1 (wall fully shielded)
    isp_loss: float = 0.0                   # fraction of Isp given up to the film coolant


def gas_transport(gas: nz.Gas) -> tuple[float, float, float]:
    """Exhaust viscosity (Pa s, Bartz's correlation at the chamber temperature), cp (J/(kg K)) and Prandtl number."""
    g = gas.gamma
    mu = 1.184e-7 * gas.molar_mass ** 0.5 * gas.chamber_temperature ** 0.6
    cp = g * gas.gas_constant / (g - 1.0)
    pr = 4.0 * g / (9.0 * g - 5.0)
    return mu, cp, pr


def bartz_heat_transfer(geom: nz.NozzleGeometry, gas: nz.Gas, pc: float, cooling: str = "regenerative",
                        wall_temperature: float = 800.0, emissivity: float = 0.85, planar: bool = False,
                        n: int = 300, extension_ratio: float = 8.0, film_fraction: float = 0.0,
                        film_temperature: float = 400.0) -> HeatTransfer:
    """Convective heat transfer from the exhaust to the wall along the nozzle (Bartz 1957).

    ``cooling``: ``"regenerative"`` holds the gas-side wall at ``wall_temperature``; ``"radiation"`` finds the
    radiative-equilibrium wall temperature where the convective input equals ε σ T⁴ (a radiation-cooled
    nozzle extension); ``"mixed"`` is a regeneratively cooled chamber and nozzle up to the area ratio
    ``extension_ratio`` with a radiation-cooled extension beyond it (as on RL10, Merlin Vacuum or J-2X);
    ``"adiabatic"`` gives the recovery temperature (no cooling at all).

    Film cooling: ``film_fraction`` of the propellant flow is injected along the wall at the injector face at
    ``film_temperature``. The film lowers the temperature the wall sees, T_aw,f = T_aw − η (T_aw − T_film), with
    an effectiveness η that decays downstream as the film mixes into the core: η = exp(−x / L) with
    L ≈ 1.2 D_t × (film fraction / 2 %). The coolant leaves cooler than the core, costing roughly
    film fraction × (1 − √(T_film / T_c)) of the Isp."""
    g = gas.gamma
    x_c, r_c = geom.inner_contour()
    k_t = int(np.argmin(r_c))
    x = np.linspace(x_c[0], x_c[-1], n)
    r = np.interp(x, x_c, r_c)
    rt = geom.rt
    eps = (r / rt) if planar else (r / rt) ** 2
    x_t = float(x_c[k_t])
    mach = np.array([nz.mach_from_area_ratio(float(e), g, supersonic=xi > x_t) for e, xi in zip(eps, x)])
    mu, cp, pr = gas_transport(gas)
    cstar = nz.characteristic_velocity(gas)
    dt = 2.0 * rt
    r_curv = 0.5 * (geom.throat_upstream_radius + geom.throat_downstream_radius) * rt
    base = 0.026 / dt ** 0.2 * (mu ** 0.2 * cp / pr ** 0.6) * (pc / cstar) ** 0.8 * (dt / r_curv) ** 0.1
    area = (1.0 / np.maximum(eps, 1.0)) ** 0.9
    t0 = gas.chamber_temperature
    k = 1.0 + 0.5 * (g - 1.0) * mach * mach
    t_aw = t0 * (1.0 + pr ** (1.0 / 3.0) * 0.5 * (g - 1.0) * mach * mach) / k
    film = np.zeros_like(x)
    isp_loss = 0.0
    if film_fraction > 1e-6:
        length = 1.2 * dt * film_fraction / 0.02
        film = np.exp(-np.maximum(x - x_c[0], 0.0) / max(length, 1e-9))
        t_aw = t_aw - film * (t_aw - min(film_temperature, float(t0)))
        isp_loss = film_fraction * (1.0 - math.sqrt(min(film_temperature, t0) / t0))

    def h_of(tw: np.ndarray) -> np.ndarray:
        sigma = 1.0 / ((0.5 * tw / t0 * k + 0.5) ** 0.68 * k ** 0.12)
        return base * area * sigma

    def radiative() -> np.ndarray:
        lo, hi = np.full_like(x, 200.0), t_aw.copy()
        for _ in range(60):                                          # q_conv(T) - eps sigma T^4 = 0
            mid = 0.5 * (lo + hi)
            f = h_of(mid) * (t_aw - mid) - emissivity * SIGMA_SB * mid ** 4
            lo = np.where(f > 0.0, mid, lo)
            hi = np.where(f > 0.0, hi, mid)
        return 0.5 * (lo + hi)

    regen = np.minimum(np.full_like(x, float(wall_temperature)), t_aw)
    if cooling == "adiabatic":
        tw = t_aw.copy()
    elif cooling == "radiation":
        tw = radiative()
    elif cooling == "mixed":
        tw = np.where((x > x_t) & (eps >= extension_ratio), radiative(), regen)
    else:
        tw = regen
    h = h_of(tw)
    q = h * (t_aw - tw)
    ds = np.hypot(np.diff(x), np.diff(r))
    circ = 2.0 * np.ones_like(r) if planar else 2.0 * math.pi * r       # two walls of a slot / the bore
    load = float(np.sum(0.5 * (q[1:] * circ[1:] + q[:-1] * circ[:-1]) * ds))
    kp = int(np.argmax(q))
    return HeatTransfer(x, mach, h, t_aw, tw, q, float(q[kp]), float(x[kp]), load, cooling, film, isp_loss)


# ---------------------------------------------------------------------------
# Flow separation
# ---------------------------------------------------------------------------
def schmucker_separation_ratio(mach: float, g: float = 1.2) -> float:
    """Wall-to-ambient pressure ratio at which an overexpanded nozzle separates (Schmucker 1973):
    p_sep / p_a = (1.88 M - 1)^-0.64, M the Mach number just ahead of the separation."""
    return nz.schmucker_ratio(mach)


def separation_point(geom: nz.NozzleGeometry, gas: nz.Gas, pc: float, pa: float, criterion: str = "schmucker",
                     planar: bool = False) -> Optional[float]:
    """x (m) where the wall flow separates by the chosen criterion (None if it stays attached to the exit)."""
    x, p = nz.wall_pressure_1d(geom, gas, pc, planar)
    x_c, r_c = geom.inner_contour()
    k_t = int(np.argmin(r_c))
    g = gas.gamma
    for i in range(k_t + 1, len(x)):
        m = nz.mach_from_pressure_ratio(p[i] / pc, g)
        limit = 0.4 if criterion == "summerfield" else schmucker_separation_ratio(m, g)
        if p[i] < limit * pa:
            return float(x[i])
    return None


# ---------------------------------------------------------------------------
# Expansion-ratio optimiser
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Optimum:
    expansion_ratio: float
    exit_diameter: float
    score: float                     # average Isp over the target (s)
    current_score: float
    altitudes: np.ndarray            # km
    isp_optimum: np.ndarray
    isp_current: np.ndarray
    sea_level_separation: bool
    note: str


def optimise_expansion(geom: nz.NozzleGeometry, gas: nz.Gas, pc: float, alt_from_km: float, alt_to_km: float,
                       avoid_separation: bool = True, planar: bool = False, max_exit_diameter: Optional[float] = None,
                       n_alt: int = 25) -> Optimum:
    """The expansion ratio with the highest Isp averaged over altitudes ``alt_from_km`` to ``alt_to_km`` (equal
    weight per kilometre: a simple stand-in for an ascent). With ``avoid_separation`` no ε that separates at
    the lowest altitude (Schmucker) is accepted. A single altitude gives the ideally-expanded nozzle there."""
    from config.settings import isa_atmosphere

    alts = np.linspace(alt_from_km, max(alt_to_km, alt_from_km), n_alt)
    pas = np.array([isa_atmosphere(a * 1e3).pressure_pa for a in alts])
    if geom.contour is nz.Contour.SONIC:
        geom = replace(geom, contour=nz.Contour.BELL)

    def isp_curve(eps: float) -> np.ndarray:
        q = eps if planar else math.sqrt(eps)
        g2 = replace(geom, exit_diameter=geom.throat_diameter * q)
        return np.array([nz.nozzle_theory(g2, gas, pc, pa, planar).isp for pa in pas])

    def separates(eps: float) -> bool:
        q = eps if planar else math.sqrt(eps)
        g2 = replace(geom, exit_diameter=geom.throat_diameter * q)
        return separation_point(g2, gas, pc, float(pas[0]), "schmucker", planar) is not None

    hi = 400.0
    if max_exit_diameter:
        q = max_exit_diameter / geom.throat_diameter
        hi = min(hi, q if planar else q * q)
    grid = np.exp(np.linspace(math.log(1.5), math.log(max(hi, 1.6)), 120))
    best, best_score = None, -1e300
    for eps in grid:
        if avoid_separation and separates(float(eps)):
            continue
        sc = float(np.mean(isp_curve(float(eps))))
        if sc > best_score:
            best, best_score = float(eps), sc
    note = ""
    if best is None:
        best = float(grid[0])
        best_score = float(np.mean(isp_curve(best)))
        note = "Every expansion ratio separates at the lowest altitude: raise the chamber pressure."
    q = best if planar else math.sqrt(best)
    cur = geom.area_ratio(planar)
    return Optimum(best, geom.throat_diameter * q, best_score, float(np.mean(isp_curve(cur))), alts,
                   isp_curve(best), isp_curve(cur), separates(cur), note)


# ---------------------------------------------------------------------------
# Exhaust noise (engineering estimate)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NoiseEstimate:
    acoustic_power: float          # W
    pwl: float                     # sound power level, dB re 1 pW
    distance: float                # m
    angles: np.ndarray             # deg from the exhaust axis
    oaspl: np.ndarray              # dB re 20 µPa at ``distance``
    peak_angle: float
    mixing_peak_hz: float          # peak of the overall sound-power spectrum
    shock_peak_hz: Optional[float]  # broadband shock-associated noise, at 90° to the axis
    screech_hz: Optional[float]


def directivity_index(theta_deg: np.ndarray) -> np.ndarray:
    """Directivity of large rocket exhaust noise (dB), peaking ~5 dB about 45° off the exhaust axis and falling
    off towards the axis and forwards (a smooth fit of the shape in NASA SP-8072)."""
    th = np.asarray(theta_deg, dtype=float)
    return 5.0 - 15.0 * ((th - 45.0) / 105.0) ** 2 - 8.0 * np.exp(-((th / 12.0) ** 2))


def jet_noise(perf: nz.NozzlePerformance, exit_diameter: float, ambient_sound_speed: float = 340.0,
              distance: float = 100.0, efficiency: float = 0.005,
              shock_cell_length: Optional[float] = None) -> NoiseEstimate:
    """Overall exhaust noise by the acoustic-efficiency method: sound power = η × ½ F v_e with η ≈ 0.5 %
    (typical of large rockets, range 0.2-1 %), spread with a rocket directivity pattern. Mixing-noise peak at
    Strouhal ≈ 0.02 on the exit diameter and exit velocity (the normalised rocket-noise spectrum of NASA
    SP-8072: the dominant sources lie far down the plume); shock-associated noise and screech from the shock-cell length and a
    convection speed of 0.7 v_e. An order-of-magnitude estimate, not a prediction."""
    ve = max(perf.exit_velocity, 1.0)
    power = max(efficiency * 0.5 * max(perf.thrust, 0.0) * ve, 1e-12)
    pwl = 10.0 * math.log10(power / 1e-12)
    angles = np.linspace(0.0, 180.0, 181)
    oaspl = pwl - 10.0 * math.log10(4.0 * math.pi * distance ** 2) + directivity_index(angles)
    k = int(np.argmax(oaspl))
    uc = 0.7 * ve
    mc = uc / ambient_sound_speed
    shock = screech = None
    ls = shock_cell_length if shock_cell_length else perf.shock_cell_spacing
    if ls:
        shock = uc / ls                    # broadband shock-associated noise heard at 90° to the axis
        screech = uc / (ls * (1.0 + mc))   # Powell's screech frequency
    return NoiseEstimate(power, pwl, distance, angles, oaspl, float(angles[k]), 0.02 * ve / max(exit_diameter, 1e-6),
                         shock, screech)
