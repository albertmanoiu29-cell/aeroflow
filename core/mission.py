"""Multi-stage launch-vehicle planning: Δv budget, payload, optimal propellant split, and an ascent check.

Each stage: an engine (thrust and mass flow from the nozzle theory, sea-level and vacuum), a number of
engines, its propellant and its structure (dry) mass. The ideal Δv of each stage is Tsiolkovsky's
Δv = g0 Isp ln(m0 / mf), using the vacuum Isp for upper stages and the average of sea-level and vacuum
for the first stage. The orbit needs about v_circ + 1.5-2 km/s of gravity, drag and steering losses.
The propellant split that maximises the payload for a given lift-off mass is found numerically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

G0 = 9.80665
MU = 3.986004418e14
R_E = 6.371e6


@dataclass
class StageSpec:
    engine: object                 # an EngineRecord (anything with .performance(pa) -> (thrust, mdot, isp))
    engines: int = 1
    propellant: float = 10000.0    # kg
    dry: float = 1000.0            # kg (tanks, structure, engines)


@dataclass
class StageResult:
    m0: float
    mf: float
    isp: float
    dv: float
    thrust_sl: float
    thrust_vac: float
    burn_time: float
    twr: float                     # at ignition (sea-level thrust for stage 1, vacuum above)


@dataclass
class MissionResult:
    stages: list[StageResult] = field(default_factory=list)
    payload: float = 0.0
    liftoff_mass: float = 0.0
    dv_total: float = 0.0
    dv_required: float = 0.0
    margin: float = 0.0            # m/s
    payload_fraction: float = 0.0


def dv_to_orbit(altitude_km: float = 200.0, losses: float = 1700.0) -> float:
    """Ideal Δv to a circular orbit at ``altitude_km`` plus typical losses (non-rotating Earth)."""
    r = R_E + altitude_km * 1e3
    return math.sqrt(MU / r) + losses


def analyse(stages: Sequence[StageSpec], payload: float, orbit_km: float = 200.0, losses: float = 1700.0
            ) -> MissionResult:
    res = MissionResult(payload=payload)
    above = payload
    masses = []
    for st in reversed(stages):
        m0 = above + st.dry + st.propellant
        masses.append((m0, above + st.dry))
        above = m0
    masses.reverse()
    res.liftoff_mass = above
    for k, (st, (m0, mf)) in enumerate(zip(stages, masses)):
        f_sl, md, isp_sl = st.engine.performance(101325.0)
        f_v, md_v, isp_v = st.engine.performance(1e-3)
        isp = 0.5 * (isp_sl + isp_v) if k == 0 else isp_v
        dv = G0 * isp * math.log(m0 / mf) if mf > 0 else 0.0
        burn = st.propellant / max(md * st.engines, 1e-9)
        thrust = (f_sl if k == 0 else f_v) * st.engines
        res.stages.append(StageResult(m0, mf, isp, dv, f_sl * st.engines, f_v * st.engines, burn,
                                      thrust / (m0 * G0)))
    res.dv_total = sum(s.dv for s in res.stages)
    res.dv_required = dv_to_orbit(orbit_km, losses)
    res.margin = res.dv_total - res.dv_required
    res.payload_fraction = payload / res.liftoff_mass if res.liftoff_mass > 0 else 0.0
    return res


def max_payload(stages: Sequence[StageSpec], orbit_km: float = 200.0, losses: float = 1700.0) -> float:
    """Largest payload that still reaches the required Δv (bisection)."""
    lo, hi = 0.0, sum(s.propellant + s.dry for s in stages)
    if analyse(stages, 0.0, orbit_km, losses).margin < 0.0:
        return 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if analyse(stages, mid, orbit_km, losses).margin >= 0.0:
            lo = mid
        else:
            hi = mid
    return lo


def optimise_split(stages: Sequence[StageSpec], orbit_km: float = 200.0, losses: float = 1700.0,
                   structure_ratios: Optional[Sequence[float]] = None) -> list[StageSpec]:
    """Redistribute the total propellant between the stages to maximise the payload. Each stage keeps its
    structural ratio dry / (dry + propellant), so tanks scale with the propellant they hold."""
    n = len(stages)
    if n < 2:
        return list(stages)
    total = sum(s.propellant for s in stages)
    eps = list(structure_ratios) if structure_ratios else [s.dry / max(s.dry + s.propellant, 1e-9) for s in stages]

    def build(fracs: np.ndarray) -> list[StageSpec]:
        out = []
        for s, f, e in zip(stages, fracs, eps):
            p = total * f
            out.append(StageSpec(s.engine, s.engines, p, max(e / max(1.0 - e, 1e-9) * p, 1.0)))
        return out

    best_f = np.array([s.propellant / total for s in stages])
    best = max_payload(build(best_f), orbit_km, losses)
    step = 0.15
    for _ in range(60):                                  # coordinate search on the simplex
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                f = best_f.copy()
                d = min(step, f[j] - 0.02)
                if d <= 0:
                    continue
                f[i] += d
                f[j] -= d
                pay = max_payload(build(f), orbit_km, losses)
                if pay > best + 1e-6:
                    best, best_f, improved = pay, f, True
        if not improved:
            step *= 0.5
            if step < 2e-3:
                break
    return build(best_f)


def fly(stages: Sequence[StageSpec], payload: float, diameter: float = 3.0, launch_angle: float = 89.0,
        turn_speed: float = 120.0):
    """Fly a two-stage vehicle with :mod:`core.trajectory` (the first two stages; a third is ignored)."""
    from core import trajectory as tj

    def table(st: StageSpec) -> Callable[[float, float], tuple[float, float]]:
        return tj.engine_thrust_table(lambda pa: st.engine.performance(pa)[:2])

    s1 = stages[0]
    upper_mass = payload + sum(s.dry + s.propellant for s in stages[2:])
    v = tj.Vehicle(s1.dry, s1.propellant, diameter, engines=s1.engines, launch_angle=launch_angle,
                   turn_speed=turn_speed)
    up = None
    if len(stages) > 1:
        s2 = stages[1]
        up = tj.UpperStage(s2.dry + upper_mass, s2.propellant, table(s2), s2.engines)
    else:
        v = tj.Vehicle(s1.dry + upper_mass, s1.propellant, diameter, engines=s1.engines, launch_angle=launch_angle,
                       turn_speed=turn_speed)
    return tj.fly(v, table(s1), up)


__all__ = ["StageSpec", "StageResult", "MissionResult", "dv_to_orbit", "analyse", "max_payload", "optimise_split",
           "fly"]
