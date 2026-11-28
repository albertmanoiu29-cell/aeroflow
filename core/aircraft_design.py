"""Conceptual aircraft sizing: wing and tail sizing, static stability, drag polar and performance.

The classic first-pass method of Raymer / Roskam / Anderson:

* **Wing** from the wing loading W/S and aspect ratio; Oswald efficiency from Raymer's fit
  e = 1.78 (1 − 0.045 AR^0.68) − 0.64; parabolic polar CD = CD0 + CL² / (π e AR).
* **Tails** from volume coefficients: S_h = V_H c̄ S / l_h, S_v = V_V b S / l_v.
* **Stability**: neutral point x_np = x_ac + η V_H (a_t / a)(1 − dε/dα) c̄ (downwash dε/dα = 2a/(π AR)),
  static margin (x_np − x_cg) / c̄; derivatives Cm_α = −a·SM and Cm_q = −2 η V_H a_t l_h / c̄.
* **Performance**: thrust required against thrust available, either a propeller (blade-element momentum
  from :mod:`core.bem`, at the given rpm and power limit) or a jet of fixed thrust; top speed, best
  climb, stall speed, and Breguet range / endurance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from config.settings import isa_atmosphere

G = 9.80665


@dataclass(frozen=True)
class AircraftSpec:
    mass: float = 1100.0                 # kg, take-off
    wing_loading: float = 75.0           # kg/m²
    aspect_ratio: float = 7.5
    taper: float = 0.6
    cd0: float = 0.028
    cl_max: float = 1.6                  # with flaps as used for the stall speed
    # tails
    tail_arm_h: float = 4.6              # m, wing a.c. to horizontal tail a.c.
    tail_arm_v: float = 4.4
    v_h: float = 0.65                    # horizontal tail volume coefficient
    v_v: float = 0.045
    ar_h: float = 4.5
    x_cg: float = 0.35                   # CG position, fraction of the MAC from its leading edge
    # propulsion
    engine: str = "propeller"            # "propeller" or "jet"
    power_kw: float = 120.0              # shaft power (propeller)
    prop_diameter: float = 1.9
    prop_blades: int = 2
    prop_pitch_ratio: float = 0.75
    prop_rpm: float = 2500.0
    jet_thrust: float = 4000.0           # N (jet)
    fuel_mass: float = 150.0             # kg
    sfc: float = 0.32                    # kg/(kW h) for a piston engine, kg/(N h) for a jet (e.g. 0.08)
    altitude: float = 2000.0             # m, cruise


@dataclass
class AircraftDesign:
    spec: AircraftSpec
    wing_area: float
    span: float
    mac: float
    root_chord: float
    oswald: float
    k_induced: float
    tail_h_area: float
    tail_v_area: float
    a_wing: float                        # per rad
    a_tail: float
    downwash: float
    neutral_point: float                 # fraction of the MAC
    static_margin: float
    cm_alpha: float
    cm_q: float
    ld_max: float
    cl_ld_max: float
    stall_speed: float                   # m/s at sea level
    speeds: np.ndarray = field(repr=False)
    thrust_required: np.ndarray = field(repr=False)
    thrust_available: np.ndarray = field(repr=False)
    climb_rate: np.ndarray = field(repr=False)
    v_max: float = 0.0
    best_climb: float = 0.0
    best_climb_speed: float = 0.0
    range_km: float = 0.0
    endurance_h: float = 0.0
    notes: list[str] = field(default_factory=list)


def oswald(ar: float) -> float:
    return max(0.5, min(1.78 * (1.0 - 0.045 * ar ** 0.68) - 0.64, 0.95))


def lift_slope(ar: float, sweep_deg: float = 0.0) -> float:
    """3-D lift slope (per rad), Helmbold / DATCOM for a straight wing."""
    a0 = 2.0 * math.pi * 0.95
    return a0 / (math.sqrt(1.0 + (a0 / (math.pi * ar)) ** 2) + a0 / (math.pi * ar))


def thrust_available(spec: AircraftSpec, speeds: np.ndarray, rho: float) -> np.ndarray:
    if spec.engine == "jet":
        return np.full_like(speeds, spec.jet_thrust * rho / 1.225)
    import warnings

    from core import bem

    prop = bem.Propeller(spec.prop_diameter, spec.prop_blades, spec.prop_pitch_ratio)
    out = np.zeros_like(speeds)
    power = spec.power_kw * 1e3 * min(rho / 1.225, 1.0) ** 1.1          # a normally aspirated engine loses power
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n = spec.prop_rpm
        for k, v in enumerate(speeds):
            # a fixed-pitch propeller: the engine settles at the rpm where the propeller absorbs its power
            # (P ~ n^3 at a given advance ratio), at most 115 % of the rated rpm
            for _ in range(4):
                r = bem.solve(prop, n, float(v), rho)
                n = min(max(n * (power / max(r.power, 1e-6)) ** (1.0 / 3.0), 0.3 * spec.prop_rpm), 1.15 * spec.prop_rpm)
            r = bem.solve(prop, n, float(v), rho)
            scale = min(power / max(r.power, 1e-6), 1.0) ** (2.0 / 3.0)
            out[k] = max(r.thrust, 0.0) * scale
    return out


def design(spec: AircraftSpec) -> AircraftDesign:
    w = spec.mass * G
    s = spec.mass / spec.wing_loading
    b = math.sqrt(spec.aspect_ratio * s)
    lam = spec.taper
    cr = 2.0 * s / (b * (1.0 + lam))
    mac = 2.0 / 3.0 * cr * (1.0 + lam + lam * lam) / (1.0 + lam)
    e = oswald(spec.aspect_ratio)
    k = 1.0 / (math.pi * e * spec.aspect_ratio)
    s_h = spec.v_h * mac * s / spec.tail_arm_h
    s_v = spec.v_v * b * s / spec.tail_arm_v
    a = lift_slope(spec.aspect_ratio)
    a_t = lift_slope(spec.ar_h)
    deps = 2.0 * a / (math.pi * spec.aspect_ratio)
    eta = 0.9
    v_h_actual = s_h * spec.tail_arm_h / (s * mac)
    x_np = 0.25 + eta * v_h_actual * (a_t / a) * (1.0 - deps)
    sm = x_np - spec.x_cg
    cm_a = -a * sm
    cm_q = -2.0 * eta * v_h_actual * a_t * spec.tail_arm_h / mac
    cl_ld = math.sqrt(spec.cd0 / k)
    ld_max = 1.0 / (2.0 * math.sqrt(spec.cd0 * k))
    v_stall = math.sqrt(2.0 * w / (1.225 * s * spec.cl_max))
    atm = isa_atmosphere(spec.altitude)
    rho = atm.density
    v_lo = 0.9 * v_stall * math.sqrt(1.225 / rho)
    v_hi = 0.85 * atm.speed_of_sound if spec.engine == "jet" else min(0.85 * atm.speed_of_sound, 4.0 * v_lo)
    speeds = np.linspace(v_lo, v_hi, 60)
    q = 0.5 * rho * speeds ** 2
    cl = w / (q * s)
    t_req = q * s * (spec.cd0 + k * cl * cl)
    t_av = thrust_available(spec, speeds, rho)
    roc = (t_av - t_req) * speeds / w
    ok = t_av >= t_req
    v_max = float(speeds[ok].max()) if ok.any() else 0.0
    i_roc = int(np.argmax(roc))
    # Breguet
    m0, m1 = spec.mass, max(spec.mass - spec.fuel_mass, 1e-6)
    notes = []
    if spec.engine == "jet":
        ct = spec.sfc * G / 3600.0                                    # 1/s from kg/(N h)
        v_r = math.sqrt(2.0 * w / (rho * s * math.sqrt(spec.cd0 / (3.0 * k))))
        rng = 2.0 * math.sqrt(2.0 / (rho * s)) * math.sqrt(spec.cd0 / (3.0 * k)) ** 0.5 / ct / \
            (spec.cd0 + k * spec.cd0 / (3.0 * k)) * (math.sqrt(w) - math.sqrt(m1 * G))
        endu = ld_max / ct * math.log(m0 / m1) / 3600.0
        notes.append(f"Jet: best-range speed ≈ {v_r:.0f} m/s at {spec.altitude:.0f} m.")
    else:
        eta_p = 0.80
        c_p = spec.sfc / 3.6e6                                        # kg/J
        rng = eta_p / (c_p * G) * ld_max * math.log(m0 / m1)
        cl_e = math.sqrt(3.0 * spec.cd0 / k)
        cd_e = 4.0 * spec.cd0
        endu = eta_p / (c_p * G) * cl_e ** 1.5 / cd_e * math.sqrt(2.0 * rho * s) * (1.0 / math.sqrt(m1 * G) -
                                                                                    1.0 / math.sqrt(w)) / 3600.0
    if sm < 0.05:
        notes.append(f"Static margin {sm * 100:.0f} % of the MAC: {'UNSTABLE' if sm < 0 else 'marginal'} - move the "
                     "CG forward or enlarge the tail.")
    elif sm > 0.30:
        notes.append(f"Static margin {sm * 100:.0f} %: very stable, heavy to rotate - move the CG aft.")
    else:
        notes.append(f"Static margin {sm * 100:.0f} % of the MAC: a normal, stable aircraft.")
    if v_max <= 0.0:
        notes.append("Not enough thrust to fly level at this altitude.")
    return AircraftDesign(spec, s, b, mac, cr, e, k, s_h, s_v, a, a_t, deps, x_np, sm, cm_a, cm_q, ld_max, cl_ld,
                          v_stall, speeds, t_req, t_av, roc, v_max, float(roc[i_roc]), float(speeds[i_roc]),
                          rng / 1e3, endu, notes)


__all__ = ["AircraftSpec", "AircraftDesign", "design", "oswald", "lift_slope", "thrust_available"]

