"""Integral boundary layer on an airfoil, driven by the panel method's surface speed (one-way coupling).

* Laminar: Thwaites' method (θ² from the edge-velocity history, H and Cf from the pressure-gradient
  parameter λ = θ² Re due/ds); laminar separation at λ < -0.09 (it then reattaches as turbulent flow: a
  short separation bubble).
* Transition: Michel's criterion Re_θ > 1.174 (1 + 22 400 / Re_x) Re_x^0.46, or a trip at a chosen x/c.
* Turbulent: Head's entrainment method with the Ludwieg-Tillmann skin friction; separation at H ≥ 2.4.
* Drag: Squire-Young at the trailing edge, Cd = Σ 2 θ_TE (u_TE / V)^((H_TE + 5) / 2) over both surfaces.

Lengths are in chords, speeds in free-stream units (V = 1), so ν = 1 / Re. The method is the classic
"panel + integral boundary layer" of engineering codes: good drag and transition trends for attached flow;
the displacement effect on the pressures is not fed back (that is what XFOIL's full coupling adds).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

H_SEP_TURB = 2.4


@dataclass
class SurfaceBL:
    x: np.ndarray            # chordwise position of each station (x / c)
    s: np.ndarray            # arc length from the stagnation point
    ue: np.ndarray           # edge speed / V
    theta: np.ndarray        # momentum thickness / c
    H: np.ndarray            # shape factor
    cf: np.ndarray           # skin-friction coefficient, wall shear / (½ ρ V²)
    transition_x: float      # x / c (nan if laminar to the trailing edge)
    separation_x: float      # x / c of turbulent separation (1.0 if attached)
    laminar_bubble: bool     # laminar separation, reattached as turbulent


@dataclass
class AirfoilBL:
    upper: SurfaceBL
    lower: SurfaceBL
    cd: float                # Squire-Young profile drag
    cd_friction: float


def _thwaites_h_l(lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lam = np.clip(lam, -0.09, 0.25)
    h = np.where(lam >= 0.0, 2.61 - 3.75 * lam + 5.24 * lam * lam, 2.088 + 0.0731 / (lam + 0.14))
    ll = np.where(lam >= 0.0, 0.22 + 1.57 * lam - 1.8 * lam * lam, 0.22 + 1.402 * lam + 0.018 * lam / (lam + 0.107))
    return h, ll


def _h1_of_h(h: float) -> float:
    if h <= 1.6:
        return 3.3 + 0.8234 * max(h - 1.1, 1e-3) ** -1.287
    return 3.3 + 1.5501 * max(h - 0.6778, 1e-3) ** -3.064


def _h_of_h1(h1: float) -> float:
    if h1 >= 5.3:
        return 1.1 + ((h1 - 3.3) / 0.8234) ** (-1.0 / 1.287)
    return 0.6778 + (max(h1 - 3.3, 1e-6) / 1.5501) ** (-1.0 / 3.064)


def _cf_lt(h: float, re_theta: float) -> float:
    return 0.246 * 10.0 ** (-0.678 * h) * max(re_theta, 10.0) ** -0.268


def march(x: np.ndarray, s: np.ndarray, ue: np.ndarray, re: float, trip_x: Optional[float] = None) -> SurfaceBL:
    """One surface, from the stagnation point (s = 0) to the trailing edge."""
    n = len(s)
    ue = np.maximum(ue, 1e-4)
    due = np.gradient(ue, s)
    theta = np.zeros(n)
    hh = np.zeros(n)
    cf = np.zeros(n)
    nu = 1.0 / re
    # --- laminar (Thwaites) ---------------------------------------------------------------------------
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (ue[1:] ** 5 + ue[:-1] ** 5) * np.diff(s))])
    th2 = 0.45 * nu * integ / ue ** 6
    th2[0] = 0.075 * nu / max(abs(due[0]), 1e-3)                      # stagnation-point value
    th_l = np.sqrt(np.maximum(th2, 1e-14))
    lam = th2 * due / nu
    h_l, l_l = _thwaites_h_l(lam)
    k_tr = n - 1
    bubble = False
    for i in range(1, n):
        re_th = ue[i] * th_l[i] / nu
        re_x = max(ue[i] * s[i] / nu, 1.0)
        michel = re_th > 1.174 * (1.0 + 22400.0 / re_x) * re_x ** 0.46
        tripped = trip_x is not None and x[i] >= trip_x and s[i] > 0.01
        lam_sep = lam[i] < -0.09
        if michel or tripped or lam_sep:
            k_tr = i
            bubble = bool(lam_sep and not michel and not tripped)
            break
    theta[:k_tr + 1] = th_l[:k_tr + 1]
    hh[:k_tr + 1] = h_l[:k_tr + 1]
    cf[:k_tr + 1] = 2.0 * l_l[:k_tr + 1] * nu / np.maximum(ue[:k_tr + 1] * th_l[:k_tr + 1], 1e-12)
    trans_x = float(x[k_tr]) if k_tr < n - 1 else math.nan
    sep_x = 1.0
    if k_tr < n - 1:
        # --- turbulent (Head) ------------------------------------------------------------------------
        th = float(th_l[k_tr]) * (1.0 if not bubble else 1.0)
        h = 1.4
        h1 = _h1_of_h(h)
        separated = False
        for i in range(k_tr + 1, n):
            ds = s[i] - s[i - 1]
            sub = max(1, int(math.ceil(ds / 0.002)))
            for k in range(sub):
                f0 = k / sub
                u = ue[i - 1] + f0 * (ue[i] - ue[i - 1])
                du = due[i - 1] + f0 * (due[i] - due[i - 1])
                h = _h_of_h1(h1)
                if separated:
                    h = H_SEP_TURB
                    c_f = 0.0
                else:
                    c_f = _cf_lt(h, u * th / nu)
                dth = c_f / 2.0 - (h + 2.0) * th / u * du
                F = 0.0306 * max(h1 - 3.0, 1e-3) ** -0.6169
                dh1 = (u * F - h1 * (u * dth + th * du)) / (u * th)
                d = ds / sub
                th = max(th + d * dth, 1e-9)
                if not separated:
                    h1 = min(max(h1 + d * dh1, 3.31), 40.0)
            h = H_SEP_TURB if separated else _h_of_h1(h1)
            if not separated and h >= H_SEP_TURB:
                separated = True
                sep_x = float(x[i])
            theta[i] = th
            hh[i] = h
            cf[i] = 0.0 if separated else _cf_lt(h, ue[i] * th / nu)
    cf = cf * ue * ue                       # wall shear over the free-stream dynamic pressure
    cf[0] = cf[1] if n > 1 else cf[0]       # (the stagnation point carries no shear)
    return SurfaceBL(x, s, ue, theta, hh, cf, trans_x, sep_x, bubble)


def analyse(sol, re: float, trip_upper: Optional[float] = None, trip_lower: Optional[float] = None) -> AirfoilBL:
    """Boundary layers on both surfaces of a :class:`core.panel.PanelSolution` (it must carry ``vt``)."""
    vt = np.asarray(sol.vt, float)
    xc, yc = np.asarray(sol.x, float), np.asarray(sol.y, float)
    n = len(vt)
    # stagnation point: where the surface speed changes sign (nodes run clockwise: lower surface first)
    ks = None
    for k in range(n - 1):
        if vt[k] <= 0.0 < vt[k + 1]:
            ks = k
            break
    if ks is None:
        ks = int(np.argmin(np.abs(vt)))
    f = -vt[ks] / max(vt[ks + 1] - vt[ks], 1e-12)
    xs = xc[ks] + f * (xc[ks + 1] - xc[ks])
    ys = yc[ks] + f * (yc[ks + 1] - yc[ks])

    def surface(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        px = np.concatenate([[xs], xc[idx]])
        py = np.concatenate([[ys], yc[idx]])
        u = np.concatenate([[0.0], np.abs(vt[idx])])
        ss = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])
        return px, ss, u

    xu, su, uu = surface(np.arange(ks + 1, n))
    xl, sl, ul = surface(np.arange(ks, -1, -1))
    x0 = float(np.min(xc))
    chord = float(np.max(xc) - x0)
    up = march((xu - x0) / chord, su / chord, uu, re, trip_upper)
    lo = march((xl - x0) / chord, sl / chord, ul, re, trip_lower)
    cd = 0.0
    for b in (up, lo):
        h_te = min(float(b.H[-1]), 3.0)
        cd += 2.0 * float(b.theta[-1]) * float(b.ue[-1]) ** ((h_te + 5.0) / 2.0)
    trap = getattr(np, "trapezoid", None) or np.trapz
    cdf = float(trap(up.cf, up.s) + trap(lo.cf, lo.s))
    return AirfoilBL(up, lo, cd, cdf)


__all__ = ["SurfaceBL", "AirfoilBL", "march", "analyse", "H_SEP_TURB"]
