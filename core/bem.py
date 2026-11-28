"""Propeller performance by blade-element momentum theory (BEM), in milliseconds.

Each blade section is a 2-D airfoil; momentum balance on annuli gives the axial and swirl induction, with
Prandtl's tip and hub loss factors (Glauert's correction for heavily loaded annuli). The default geometry is
the program's propeller preset (:func:`core.geometry_manager.build_propeller`): constant geometric pitch,
chord peaking near 40 % radius, NACA 4-digit sections.

Airfoil model: lift slope 0.9 × 2π per radian from the zero-lift angle (thin-airfoil value for the camber),
stall at CL ≈ 1.25 with a gentle post-stall fall-off, profile drag CD = 0.010 + 0.012 (CL − 0.3)² rising
steeply beyond stall. Good to ~10 % against measured propeller data in the normal working range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Propeller:
    diameter: float = 1.0            # m
    blades: int = 2
    pitch_ratio: float = 0.8         # geometric pitch / diameter
    hub_ratio: float = 0.12          # hub radius / diameter (as in the preset)
    camber: float = 0.04             # NACA max camber (4412 -> 0.04)
    cl_max: float = 1.25
    chord_scale: float = 1.0

    def stations(self, n: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Blade radius (m), chord (m) and section pitch angle (rad) along the span."""
        d, rad = self.diameter, 0.5 * self.diameter
        r_hub = self.hub_ratio * d
        r = np.linspace(r_hub, 0.995 * rad, n)
        x = r / rad
        chord = d * np.clip(0.07 + 0.11 * np.sin(math.pi * np.clip((x - 0.1) / 1.1, 0.0, 1.0)), 0.045, None)
        chord *= self.chord_scale
        beta = np.arctan(self.pitch_ratio * d / (2.0 * math.pi * r))
        return r, chord, beta


def _airfoil(alpha: np.ndarray, camber: float, cl_max: float) -> tuple[np.ndarray, np.ndarray]:
    a0 = -math.radians(100.0 * camber)                    # zero-lift angle: NACA 4412 about -4°
    cl_lin = 0.9 * 2.0 * math.pi * (alpha - a0)
    a_stall = a0 + cl_max / (0.9 * 2.0 * math.pi)
    over = alpha - a_stall
    cl = np.where(over > 0.0, cl_max - 1.2 * np.minimum(over, math.radians(30.0)), cl_lin)
    under = a0 - cl_max / (0.9 * 2.0 * math.pi) - alpha
    cl = np.where(under > 0.0, -cl_max + 1.2 * np.minimum(under, math.radians(30.0)), cl)
    cd = 0.010 + 0.012 * (cl - 0.3) ** 2 + np.where(over > 0.0, 1.5 * np.sin(np.minimum(over, 1.2)) ** 2, 0.0)
    return cl, cd


@dataclass
class BEMResult:
    rpm: float
    speed: float
    rho: float
    thrust: float                    # N
    torque: float                    # N m
    power: float                     # W
    efficiency: float
    J: float
    CT: float
    CQ: float
    CP: float
    r: np.ndarray = field(repr=False)
    dT_dr: np.ndarray = field(repr=False)          # N/m (all blades)
    dQ_dr: np.ndarray = field(repr=False)
    alpha_deg: np.ndarray = field(repr=False)
    a_axial: np.ndarray = field(repr=False)
    stalled_fraction: float = 0.0


def solve(prop: Propeller, rpm: float, speed: float, rho: float = 1.225, n: int = 40) -> BEMResult:
    """Thrust, torque and power of ``prop`` at ``rpm`` in a flow of ``speed`` m/s (air density ``rho``)."""
    omega = max(rpm, 1e-6) * 2.0 * math.pi / 60.0
    r, chord, beta = prop.stations(n)
    rad = 0.5 * prop.diameter
    r_hub = prop.hub_ratio * prop.diameter
    b = prop.blades
    v0 = max(speed, 0.0)
    # Iterate on the induced velocities themselves (not on factors of V), so static thrust (V = 0) works:
    # momentum on each annulus, dT/dr = 4 pi r rho F (V + vi) vi and dQ/dr = 4 pi r^3 rho F (V + vi) omega a'.
    vi = np.full_like(r, 0.05 * omega * rad)
    ap = np.full_like(r, 0.01)
    for _ in range(400):
        vt = omega * r * (1.0 - ap)
        va = v0 + vi
        phi = np.arctan2(va, vt)
        alpha = beta - phi
        cl, cd = _airfoil(alpha, prop.camber, prop.cl_max)
        cn = cl * np.cos(phi) - cd * np.sin(phi)          # along the axis (thrust)
        ct = cl * np.sin(phi) + cd * np.cos(phi)          # along the rotation (torque)
        sphi = np.maximum(np.abs(np.sin(phi)), 1e-4)
        f_tip = 2.0 / math.pi * np.arccos(np.clip(np.exp(-b * (rad - r) / (2.0 * r * sphi)), 0.0, 1.0))
        f_hub = 2.0 / math.pi * np.arccos(np.clip(np.exp(-b * (r - r_hub) / (2.0 * r_hub * sphi)), 0.0, 1.0))
        f = np.maximum(f_tip * f_hub, 1e-3)
        w2 = va * va + vt * vt
        dt_blade = np.maximum(0.5 * w2 * chord * cn * b, 0.0)          # per unit rho
        dq_blade = 0.5 * w2 * chord * ct * b * r
        vi_new = 0.5 * (-v0 + np.sqrt(v0 * v0 + dt_blade / (math.pi * r * f)))
        ap_new = np.clip(dq_blade / (4.0 * math.pi * r ** 3 * f * np.maximum(v0 + vi, 1e-6) * omega), -0.5, 0.5)
        if np.max(np.abs(vi_new - vi)) < 1e-7 * (1.0 + omega * rad) and np.max(np.abs(ap_new - ap)) < 1e-8:
            vi, ap = vi_new, ap_new
            break
        vi = 0.7 * vi + 0.3 * vi_new
        ap = 0.7 * ap + 0.3 * ap_new
    a = vi / max(v0, 1e-9) if v0 > 1e-9 else np.full_like(r, np.inf)
    vt = omega * r * (1.0 - ap)
    va = v0 + vi
    w2 = va * va + vt * vt
    phi = np.arctan2(va, vt)
    alpha = beta - phi
    cl, cd = _airfoil(alpha, prop.camber, prop.cl_max)
    cn = cl * np.cos(phi) - cd * np.sin(phi)
    ct = cl * np.sin(phi) + cd * np.cos(phi)
    dT = 0.5 * rho * w2 * chord * cn * b
    dQ = 0.5 * rho * w2 * chord * ct * b * r
    thrust = float(np.trapz(dT, r))
    torque = float(np.trapz(dQ, r))
    power = torque * omega
    nrev = omega / (2.0 * math.pi)
    d = prop.diameter
    j = speed / (nrev * d)
    ct_ = thrust / (rho * nrev ** 2 * d ** 4)
    cq = torque / (rho * nrev ** 2 * d ** 5)
    cp = power / (rho * nrev ** 3 * d ** 5)
    eff = thrust * speed / power if power > 1e-9 and thrust > 0 else 0.0
    a_stall = -math.radians(100.0 * prop.camber) + prop.cl_max / (0.9 * 2.0 * math.pi)
    stalled = float(np.mean(alpha > a_stall))
    return BEMResult(rpm, speed, rho, thrust, torque, power, eff, j, ct_, cq, cp, r, dT, dQ, np.degrees(alpha), a,
                     stalled)


def sweep_advance_ratio(prop: Propeller, rpm: float, rho: float = 1.225, j_max: Optional[float] = None,
                        n: int = 40) -> dict[str, np.ndarray]:
    """CT, CP and efficiency against the advance ratio J = V / (n D), from static to windmilling."""
    nrev = rpm / 60.0
    j_max = j_max if j_max is not None else 1.3 * prop.pitch_ratio
    js = np.linspace(0.01, j_max, n)
    out = {"J": js, "CT": np.zeros(n), "CP": np.zeros(n), "eta": np.zeros(n)}
    for k, j in enumerate(js):
        res = solve(prop, rpm, j * nrev * prop.diameter, rho)
        out["CT"][k], out["CP"][k], out["eta"][k] = res.CT, res.CP, res.efficiency
    return out


def ideal_efficiency(thrust: float, speed: float, rho: float, diameter: float) -> float:
    """Froude's ideal (actuator-disc) efficiency for this thrust: the upper limit."""
    area = math.pi * 0.25 * diameter ** 2
    tc = thrust / (0.5 * rho * max(speed, 1e-6) ** 2 * area)
    return 2.0 / (1.0 + math.sqrt(1.0 + max(tc, 0.0)))
