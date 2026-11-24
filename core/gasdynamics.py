"""Analytic compressible-flow relations (perfect gas, default gamma = 1.4).

Used by the Supersonic Lab's theory panel and to validate the numerical solver. Every function is
pure Python/NumPy and covered by tests against textbook (NACA Report 1135) values.

Conventions: angles in degrees at the public API, ratios are downstream / upstream (state 2 / state
1) unless noted, and ``p0`` is total (stagnation) pressure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

GAMMA = 1.4


# ---------------------------------------------------------------------------
# Isentropic flow
# ---------------------------------------------------------------------------
def total_temperature_ratio(m: float, g: float = GAMMA) -> float:
    """T0 / T."""
    return 1.0 + 0.5 * (g - 1.0) * m * m


def total_pressure_ratio(m: float, g: float = GAMMA) -> float:
    """p0 / p."""
    return total_temperature_ratio(m, g) ** (g / (g - 1.0))


def total_density_ratio(m: float, g: float = GAMMA) -> float:
    """rho0 / rho."""
    return total_temperature_ratio(m, g) ** (1.0 / (g - 1.0))


def area_ratio(m: float, g: float = GAMMA) -> float:
    """A / A* of isentropic nozzle flow."""
    t = 2.0 / (g + 1.0) * total_temperature_ratio(m, g)
    return t ** ((g + 1.0) / (2.0 * (g - 1.0))) / m


def mach_angle_deg(m: float) -> float:
    """Mach angle mu = asin(1/M) (90° at M = 1; undefined, returned as NaN, below)."""
    return math.degrees(math.asin(1.0 / m)) if m >= 1.0 else float("nan")


def prandtl_meyer_deg(m: float, g: float = GAMMA) -> float:
    """Prandtl-Meyer function nu(M) in degrees (0 at M = 1)."""
    if m < 1.0:
        return float("nan")
    a = math.sqrt((g + 1.0) / (g - 1.0))
    b = math.sqrt(m * m - 1.0)
    return math.degrees(a * math.atan(b / a) - math.atan(b))


def inverse_prandtl_meyer(nu_deg: float, g: float = GAMMA) -> float:
    """Mach number with nu(M) = ``nu_deg`` (bisection, accurate to 1e-10)."""
    nu_max = 90.0 * (math.sqrt((g + 1.0) / (g - 1.0)) - 1.0)
    if not 0.0 <= nu_deg < nu_max:
        raise ValueError(f"Prandtl-Meyer angle must be in [0, {nu_max:.2f}) degrees")
    lo, hi = 1.0, 1.0
    while prandtl_meyer_deg(hi, g) < nu_deg:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if prandtl_meyer_deg(mid, g) < nu_deg:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Normal shock
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NormalShock:
    """Jump conditions across a normal shock."""

    m1: float
    m2: float
    p_ratio: float          # p2 / p1
    rho_ratio: float        # rho2 / rho1 (= u1 / u2)
    t_ratio: float          # T2 / T1
    p0_ratio: float         # p02 / p01 (total-pressure recovery)
    pitot_ratio: float      # p02 / p1 (Rayleigh pitot formula: what a pitot tube reads)


def normal_shock(m1: float, g: float = GAMMA) -> NormalShock:
    """Normal-shock relations for upstream Mach ``m1`` >= 1."""
    if m1 < 1.0:
        raise ValueError("A normal shock needs a supersonic upstream Mach number (M1 >= 1).")
    m1s = m1 * m1
    m2 = math.sqrt((1.0 + 0.5 * (g - 1.0) * m1s) / (g * m1s - 0.5 * (g - 1.0)))
    p_ratio = 1.0 + 2.0 * g / (g + 1.0) * (m1s - 1.0)
    rho_ratio = (g + 1.0) * m1s / ((g - 1.0) * m1s + 2.0)
    t_ratio = p_ratio / rho_ratio
    p0_ratio = p_ratio * total_pressure_ratio(m2, g) / total_pressure_ratio(m1, g)
    pitot = p_ratio * total_pressure_ratio(m2, g)
    return NormalShock(m1, m2, p_ratio, rho_ratio, t_ratio, p0_ratio, pitot)


def normal_mach_from_pressure_ratio(p_ratio: float, g: float = GAMMA) -> float:
    """Upstream normal Mach number Mn1 of a shock with static-pressure jump ``p_ratio`` (>= 1)."""
    return math.sqrt(max(1.0 + (p_ratio - 1.0) * (g + 1.0) / (2.0 * g), 1.0))


def shock_angle_from_pressure_ratio(m1: float, p_ratio: float, g: float = GAMMA) -> float:
    """Oblique-shock angle (deg) implied by a measured pressure jump at freestream Mach ``m1``."""
    mn = normal_mach_from_pressure_ratio(p_ratio, g)
    return math.degrees(math.asin(min(mn / m1, 1.0)))


# ---------------------------------------------------------------------------
# Oblique shock
# ---------------------------------------------------------------------------
def theta_from_beta(m1: float, beta_deg: float, g: float = GAMMA) -> float:
    """Flow deflection theta (deg) produced by a shock of angle beta (theta-beta-M relation)."""
    b = math.radians(beta_deg)
    num = 2.0 / math.tan(b) * (m1 * m1 * math.sin(b) ** 2 - 1.0)
    den = m1 * m1 * (g + math.cos(2.0 * b)) + 2.0
    return math.degrees(math.atan(num / den))


def max_deflection(m1: float, g: float = GAMMA) -> tuple[float, float]:
    """``(theta_max, beta_at_theta_max)`` in degrees: beyond theta_max the shock detaches."""
    mu = math.degrees(math.asin(1.0 / m1))
    lo, hi = mu, 90.0
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(120):                                  # golden-section maximisation of theta(beta)
        a = hi - phi * (hi - lo)
        b = lo + phi * (hi - lo)
        if theta_from_beta(m1, a, g) < theta_from_beta(m1, b, g):
            lo = a
        else:
            hi = b
    beta = 0.5 * (lo + hi)
    return theta_from_beta(m1, beta, g), beta


def oblique_shock_beta(m1: float, theta_deg: float, g: float = GAMMA, strong: bool = False) -> Optional[float]:
    """Shock angle beta (deg) for deflection ``theta_deg``; ``None`` when the shock is detached."""
    if m1 <= 1.0:
        return None
    if abs(theta_deg) < 1e-12:
        return mach_angle_deg(m1)
    theta = abs(theta_deg)
    t_max, b_max = max_deflection(m1, g)
    if theta > t_max:
        return None
    lo, hi = (b_max, 90.0) if strong else (mach_angle_deg(m1), b_max)
    for _ in range(200):                                  # theta(beta) is monotonic on each branch
        mid = 0.5 * (lo + hi)
        t = theta_from_beta(m1, mid, g)
        if (t < theta) != strong:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class ObliqueShock:
    """Weak-branch oblique shock solution."""

    m1: float
    theta_deg: float
    beta_deg: float
    m2: float
    p_ratio: float
    rho_ratio: float
    t_ratio: float
    p0_ratio: float


def oblique_shock(m1: float, theta_deg: float, g: float = GAMMA) -> Optional[ObliqueShock]:
    """Weak oblique shock for wedge deflection ``theta_deg`` (``None`` if detached)."""
    beta = oblique_shock_beta(m1, theta_deg, g)
    if beta is None:
        return None
    mn1 = m1 * math.sin(math.radians(beta))
    ns = normal_shock(max(mn1, 1.0), g)
    m2 = ns.m2 / math.sin(math.radians(beta - abs(theta_deg)))
    return ObliqueShock(m1, abs(theta_deg), beta, m2, ns.p_ratio, ns.rho_ratio, ns.t_ratio, ns.p0_ratio)


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------
def expansion(m1: float, turn_deg: float, g: float = GAMMA) -> tuple[float, float]:
    """Prandtl-Meyer expansion by ``turn_deg``: returns ``(M2, p2/p1)``."""
    m2 = inverse_prandtl_meyer(prandtl_meyer_deg(m1, g) + turn_deg, g)
    return m2, total_pressure_ratio(m1, g) / total_pressure_ratio(m2, g)


def billig_standoff_cylinder(m1: float) -> float:
    """Empirical bow-shock stand-off distance / radius for a circular cylinder (Billig 1967)."""
    return 0.386 * math.exp(4.67 / (m1 * m1))
