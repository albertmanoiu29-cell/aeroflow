"""Finite wings with Prandtl's lifting-line theory (Glauert's Fourier-series solution of the monoplane equation).

A straight (unswept) wing with linear taper and linear twist (washout), its sections described by their lift
slope, zero-lift angle and maximum lift coefficient (for example from :mod:`core.panel`). Gives the lift curve,
the induced drag and span efficiency, the spanwise lift distribution and where the wing stalls first (the
first station whose local Cl reaches the section's CL,max).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Wing:
    span: float                    # m
    root_chord: float              # m
    taper: float = 0.5             # tip / root chord
    washout: float = 0.0           # degrees of twist at the tip, nose down (+ = washout)
    cl_alpha: float = 2.0 * math.pi  # section lift slope, per radian
    alpha_zero_lift: float = 0.0   # section zero-lift angle, degrees
    cl_max: float = 1.5            # section maximum lift coefficient
    cd0: float = 0.008             # section profile drag (minimum)

    @property
    def tip_chord(self) -> float:
        return self.root_chord * self.taper

    @property
    def area(self) -> float:
        return 0.5 * (self.root_chord + self.tip_chord) * self.span

    @property
    def aspect_ratio(self) -> float:
        return self.span ** 2 / self.area

    @property
    def mean_aerodynamic_chord(self) -> float:
        lam = self.taper
        return 2.0 / 3.0 * self.root_chord * (1 + lam + lam * lam) / (1 + lam)

    def chord(self, y: np.ndarray) -> np.ndarray:
        """Chord at spanwise position y (m from the centre line)."""
        return self.root_chord * (1.0 - (1.0 - self.taper) * np.abs(y) / (0.5 * self.span))

    def twist(self, y: np.ndarray) -> np.ndarray:
        """Geometric twist (degrees, nose up +) at y: linear from 0 at the root to -washout at the tip."""
        return -self.washout * np.abs(y) / (0.5 * self.span)


@dataclass
class WingSolution:
    alpha: float                   # degrees
    cl: float
    cdi: float
    e: float                       # span efficiency
    y: np.ndarray                  # stations (m)
    cl_local: np.ndarray           # section lift coefficients
    stalled: bool                  # some station is past the section CL,max


def solve(wing: Wing, alpha_deg: float, n: int = 40) -> WingSolution:
    """Lifting-line solution at angle of attack ``alpha_deg`` (symmetric loading: odd Fourier terms)."""
    th = (np.arange(1, n + 1) - 0.5) * (0.5 * math.pi) / n          # half span, root excluded, tip excluded
    y = -0.5 * wing.span * np.cos(th)                                # -b/2 .. 0
    c = wing.chord(y)
    alpha = np.radians(alpha_deg + wing.twist(y) - wing.alpha_zero_lift)
    k = 2 * np.arange(n) + 1                                         # odd harmonics
    st = np.sin(th)
    m = np.sin(np.outer(th, k)) * (4.0 * wing.span / (wing.cl_alpha * c))[:, None] \
        + np.sin(np.outer(th, k)) * (k[None, :] / st[:, None])
    a = np.linalg.solve(m, alpha)
    ar = wing.aspect_ratio
    cl = math.pi * ar * a[0]
    delta = float(np.sum(k[1:] * (a[1:] / a[0]) ** 2)) if abs(a[0]) > 1e-15 else 0.0
    e = 1.0 / (1.0 + delta)
    cdi = cl * cl / (math.pi * ar * e) if abs(a[0]) > 1e-15 else 0.0
    gamma_term = (np.sin(np.outer(th, k)) @ a)
    cl_local = 4.0 * wing.span * gamma_term / c
    ys = np.concatenate([y, -y[::-1]])
    cls = np.concatenate([cl_local, cl_local[::-1]])
    return WingSolution(alpha_deg, cl, cdi, e, ys, cls, bool(np.max(cl_local) > wing.cl_max))


@dataclass
class WingPolar:
    alpha: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cdi: np.ndarray
    e: float
    cl_alpha: float                # wing lift slope, per radian
    stall_alpha: float             # degrees (nan if not reached)
    cl_max: float
    stall_y: float                 # where it stalls first (m from the centre line, |y|)


def polar(wing: Wing, alphas: np.ndarray, n: int = 40) -> WingPolar:
    sols = [solve(wing, float(a), n) for a in alphas]
    cl = np.array([s.cl for s in sols])
    cdi = np.array([s.cdi for s in sols])
    cd = cdi + wing.cd0
    k = np.argsort(np.abs(np.asarray(alphas) - wing.alpha_zero_lift))[:2]
    slope = float((cl[k[1]] - cl[k[0]]) / math.radians(alphas[k[1]] - alphas[k[0]]))
    es = [s.e for s in sols if abs(s.cl) > 1e-6]
    st, clm, sy = math.nan, math.nan, math.nan
    for i in range(1, len(sols)):
        lo, hi = sols[i - 1], sols[i]
        if alphas[i] > wing.alpha_zero_lift and float(lo.cl_local.max()) < wing.cl_max <= float(hi.cl_local.max()):
            f = (wing.cl_max - float(lo.cl_local.max())) / (float(hi.cl_local.max()) - float(lo.cl_local.max()))
            st = float(alphas[i - 1] + f * (alphas[i] - alphas[i - 1]))
            clm = float(lo.cl + f * (hi.cl - lo.cl))
            sy = float(abs(hi.y[int(np.argmax(hi.cl_local))]))
            break
    return WingPolar(np.asarray(alphas, float), cl, cd, cdi, float(np.median(es)) if es else 1.0, slope, st, clm, sy)


__all__ = ["Wing", "WingSolution", "solve", "WingPolar", "polar"]
