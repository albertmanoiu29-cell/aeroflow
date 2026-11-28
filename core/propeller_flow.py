"""The air a propeller moves: its slipstream, from the blade loading of :mod:`core.bem` and momentum theory.

For each annulus of the disc, the thrust per unit radius fixes the induced axial velocity v_i through
dT/dr = 4 π r ρ (V + v_i) v_i (valid in static thrust too), and the torque fixes the swirl the air leaves
with, u_θ = (dQ/dr) / (2 π r² ρ (V + v_i)). Along the axis the classic actuator-disc result spreads the
induction smoothly: v(x) = v_i (1 + x / √(x² + R²)) (half of the far-wake value at the disc), and the
stream tubes contract by continuity. Swirl appears at the disc (half its wake value there) and is carried
downstream. The flow goes along +x; the propeller turns about +x (right-handed), as in
:func:`core.geometry_manager.build_propeller`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from core.bem import BEMResult


@dataclass
class Slipstream:
    radius: float                 # propeller radius R (m)
    speed: float                  # free-stream speed V (m/s)
    omega: float                  # rotation rate (rad/s), positive about +x
    r: np.ndarray                 # stations (m)
    v_induced: np.ndarray         # axial induced velocity at the disc (m/s)
    swirl_wake: np.ndarray        # far-wake swirl velocity (m/s), in the sense of rotation

    @property
    def wake_speed(self) -> float:
        """Peak far-wake axial speed (m/s)."""
        return float(self.speed + 2.0 * np.max(self.v_induced))

    def induced_at(self, r0: np.ndarray) -> np.ndarray:
        return np.interp(r0, self.r, self.v_induced, left=self.v_induced[0], right=0.0)

    def swirl_at(self, r0: np.ndarray) -> np.ndarray:
        return np.interp(r0, self.r, self.swirl_wake, left=self.swirl_wake[0], right=0.0)

    # -- the flow along a stream tube that crosses the disc at radius r0 ----------------------------------
    def axial_speed(self, r0: np.ndarray, x: np.ndarray) -> np.ndarray:
        vi = self.induced_at(r0)
        return self.speed + vi * (1.0 + x / np.sqrt(x * x + self.radius ** 2))

    def tube_radius(self, r0: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Continuity: r² u is constant along a thin stream tube (r0 at the disc)."""
        u_disc = self.speed + self.induced_at(r0)
        return r0 * np.sqrt(np.maximum(u_disc, 1e-6) / np.maximum(self.axial_speed(r0, x), 1e-6))

    def swirl(self, r0: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Swirl velocity (m/s); zero upstream, half the wake value at the disc."""
        return self.swirl_at(r0) * 0.5 * (1.0 + np.tanh(x / (0.04 * self.radius)))

    def streamline(self, r0: float, theta0: float, x0: float, x1: float, n: int = 240) -> np.ndarray:
        """Points (n, 3) of the steady streamline through (r0 at the disc, angle θ0), from x0 to x1."""
        xs = np.linspace(x0, x1, n)
        r0a = np.full(n, r0)
        u = self.axial_speed(r0a, xs)
        rr = self.tube_radius(r0a, xs)
        dth = self.swirl(r0a, xs) / np.maximum(rr, 1e-9) / np.maximum(u, 1e-6)
        th = theta0 + np.concatenate([[0.0], np.cumsum(0.5 * (dth[1:] + dth[:-1]) * np.diff(xs))])
        th -= float(np.interp(0.0, xs, th))                  # θ0 is the angle where it crosses the disc
        th += theta0
        return np.column_stack([xs, rr * np.cos(th), rr * np.sin(th)])


def slipstream(res: BEMResult, diameter: float) -> Slipstream:
    """The slipstream of a solved propeller (``res`` from :func:`core.bem.solve`)."""
    r = np.asarray(res.r, float)
    rho, v = res.rho, max(res.speed, 0.0)
    dtdr = np.maximum(np.asarray(res.dT_dr, float), 0.0)
    vi = 0.5 * (-v + np.sqrt(v * v + dtdr / (math.pi * r * rho)))
    dqdr = np.asarray(res.dQ_dr, float)
    swirl = dqdr / (2.0 * math.pi * r * r * rho * np.maximum(v + vi, 1e-3))
    swirl = np.clip(swirl, -3.0 * np.max(vi + v + 1e-3), 3.0 * np.max(vi + v + 1e-3))
    omega = res.rpm * 2.0 * math.pi / 60.0
    return Slipstream(0.5 * diameter, v, omega, r, vi, swirl)


__all__ = ["Slipstream", "slipstream"]
