"""Multi-element airfoils (leading-edge slat, main element, slotted flap) with the Hess-Smith panel method.

Each element carries its own constant-strength source panels and its own uniform vortex strength, closed by
its own Kutta condition. All elements see each other, so the classic high-lift effects appear by
themselves: the flap's circulation raises the main element's lift and relieves its trailing-edge pressure
(the "dumping" effect), and the slat takes the leading-edge suction peak off the main element (the
"slat effect"), which delays its stall. Lift and moment are referenced to the stowed (clean) chord.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from core import panel as pm


@dataclass
class HighLift:
    """Geometry of the high-lift devices, in clean-chord units (c = 1)."""

    flap_chord: float = 0.0          # 0 = no flap (typical 0.25-0.35)
    flap_deflection: float = 20.0    # degrees, trailing edge down
    flap_gap: float = 0.015          # slot gap between the main trailing edge and the flap
    flap_overlap: float = 0.0        # flap leading edge ahead of the main trailing edge (+)
    slat_chord: float = 0.0          # 0 = no slat (typical 0.12-0.18)
    slat_deflection: float = 40.0    # degrees, leading edge down (chord line to the main chord)
    slat_gap: float = 0.02


@dataclass
class MultiSolution:
    alpha_deg: float
    cl: float                        # total, on the clean chord
    cm_quarter: float
    elements: list[pm.PanelSolution] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    cl_parts: list[float] = field(default_factory=list)


def _rotate(pts: np.ndarray, pivot: tuple[float, float], deg: float) -> np.ndarray:
    a = math.radians(-deg)                       # positive deflection = trailing edge down (clockwise)
    c, s = math.cos(a), math.sin(a)
    p = pts - np.asarray(pivot)
    return np.column_stack([p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c]) + np.asarray(pivot)


def build_elements(main_outline: np.ndarray, hl: HighLift, panels: int = 140,
                   device_code: str = "0012") -> list[tuple[str, np.ndarray]]:
    """Node arrays (clockwise, closed) of each element: main airfoil (chord shortened ahead of the flap),
    plus a slat and a flap of NACA ``device_code`` section placed with their gaps and deflections."""
    from core.shapes2d import naca_outline

    out: list[tuple[str, np.ndarray]] = []
    main = pm.prepare(main_outline, panels)
    cf = min(max(hl.flap_chord, 0.0), 0.45)
    cs = min(max(hl.slat_chord, 0.0), 0.3)
    if cf > 0.0:                                  # the main element ends where the stowed flap begins
        x_cut = 1.0 - cf + hl.flap_overlap
        main = pm.prepare(main_outline * np.array([x_cut, 1.0]), panels)
    out.append(("main", main))
    if cf > 0.0:
        fl = pm.prepare(naca_outline(device_code, 120), max(panels // 2, 60)) * cf
        te = main[0]                               # main trailing edge
        fl = fl + np.array([te[0] - hl.flap_overlap - 0.0, te[1] - hl.flap_gap - 0.02 * cf])
        fl = _rotate(fl, (te[0] - hl.flap_overlap, te[1] - hl.flap_gap), hl.flap_deflection)
        out.append(("flap", fl))
    if cs > 0.0:
        # the slat's trailing edge sits above and just behind the main nose, leaving a slot of ``slat_gap``
        sl = pm.prepare(naca_outline("9308", 120), max(panels // 2, 60)) * cs        # thin, strongly cambered
        te = sl[0].copy()
        sl = _rotate(sl - te, (0.0, 0.0), -hl.slat_deflection)               # leading edge down
        nose = main[int(np.argmin(main[:, 0]))]
        k = int(np.argmin(np.abs(main[:, 0] - (nose[0] + 0.04)) + 10.0 * (main[:, 1] < nose[1])))
        target = main[k] + np.array([0.0, hl.slat_gap])
        sl = sl + target
        out.insert(0, ("slat", sl))
    return out


def solve(elements: Sequence[tuple[str, np.ndarray]], alpha_deg: float, chord: float = 1.0) -> MultiSolution:
    """Coupled Hess-Smith solution of several closed bodies at angle ``alpha_deg``."""
    xs, ys, ths, lens, owner = [], [], [], [], []
    xn0, yn0, xn1, yn1 = [], [], [], []
    for k, (_name, nodes) in enumerate(elements):
        xn, yn = nodes[:, 0], nodes[:, 1]
        xs.append(0.5 * (xn[:-1] + xn[1:]))
        ys.append(0.5 * (yn[:-1] + yn[1:]))
        ths.append(np.arctan2(np.diff(yn), np.diff(xn)))
        lens.append(np.hypot(np.diff(xn), np.diff(yn)))
        xn0.append(xn[:-1])
        yn0.append(yn[:-1])
        xn1.append(xn[1:])
        yn1.append(yn[1:])
        owner.append(np.full(len(xn) - 1, k))
    xc, yc, th, ln = (np.concatenate(v) for v in (xs, ys, ths, lens))
    own = np.concatenate(owner)
    x0, y0, x1, y1 = (np.concatenate(v) for v in (xn0, yn0, xn1, yn1))
    n, nb = len(xc), len(elements)
    rx0, ry0 = xc[:, None] - x0[None, :], yc[:, None] - y0[None, :]
    rx1, ry1 = xc[:, None] - x1[None, :], yc[:, None] - y1[None, :]
    r0, r1 = np.hypot(rx0, ry0), np.hypot(rx1, ry1)
    beta = np.arctan2(ry1 * rx0 - rx1 * ry0, rx1 * rx0 + ry1 * ry0)
    np.fill_diagonal(beta, math.pi)
    lnr = np.log(r1 / r0)
    np.fill_diagonal(lnr, 0.0)
    d = th[:, None] - th[None, :]
    s, c = np.sin(d), np.cos(d)
    tp = 1.0 / (2.0 * math.pi)
    an_src = tp * (s * lnr + c * beta)
    at_src = tp * (s * beta - c * lnr)
    an_vor = tp * (c * lnr - s * beta)
    at_vor = tp * (s * lnr + c * beta)
    a = math.radians(alpha_deg)
    m = np.zeros((n + nb, n + nb))
    rhs = np.zeros(n + nb)
    m[:n, :n] = an_src
    for k in range(nb):
        sel = own == k
        m[:n, n + k] = an_vor[:, sel].sum(axis=1)
    rhs[:n] = -np.sin(a - th)
    starts = np.concatenate([[0], np.cumsum([len(e[1]) - 1 for e in elements])])
    for k in range(nb):
        i0, i1 = starts[k], starts[k + 1] - 1             # first and last panel of element k (its trailing edge)
        m[n + k, :n] = at_src[i0] + at_src[i1]
        for j in range(nb):
            sel = own == j
            m[n + k, n + j] = at_vor[i0, sel].sum() + at_vor[i1, sel].sum()
        rhs[n + k] = -math.cos(a - th[i0]) - math.cos(a - th[i1])
    sol = np.linalg.solve(m, rhs)
    q, gam = sol[:n], sol[n:]
    vt = at_src @ q + np.cos(a - th)
    for k in range(nb):
        vt += gam[k] * at_vor[:, own == k].sum(axis=1)
    cp = 1.0 - vt * vt
    nxn, nyn = -np.sin(th), np.cos(th)
    fx, fy = -cp * ln * nxn, -cp * ln * nyn
    # lift and moment from the pressure forces (free stream along α), clean chord, about c/4
    lift = float(np.sum(-fx * math.sin(a) + fy * math.cos(a)))
    cm = float(np.sum(-(xc - 0.25 * chord) * fy + yc * fx)) / chord ** 2
    out = MultiSolution(alpha_deg, lift / chord, cm)
    for k, (name, _nodes) in enumerate(elements):
        sel = own == k
        lek = int(np.argmin(xc[sel]))
        sol_k = pm.PanelSolution(alpha_deg, 2.0 * gam[k] * float(ln[sel].sum()) / chord, 0.0, xc[sel], yc[sel],
                                 cp[sel], np.arange(int(sel.sum())) > lek, vt[sel])
        out.elements.append(sol_k)
        out.names.append(name)
        out.cl_parts.append(float(np.sum(-fx[sel] * math.sin(a) + fy[sel] * math.cos(a))) / chord)
    return out


@dataclass
class MultiPolar:
    alpha: np.ndarray
    cl: np.ndarray
    cl_inviscid: np.ndarray
    cd: np.ndarray
    cm: np.ndarray
    stall_alpha: float
    cl_max: float
    critical_element: str            # the element whose suction peak sets the stall


def polar(elements: Sequence[tuple[str, np.ndarray]], alphas: np.ndarray, reynolds: float) -> MultiPolar:
    """Lift, drag and moment of the configuration; stall where an element's pressure recovery reaches the
    Valarezo-Chin limit (developed for exactly this: multi-element high-lift systems)."""
    from core import boundary_layer as bl

    alphas = np.asarray(alphas, float)
    lim = pm.stall_delta_cp(reynolds)
    cl_i, cd, cm, dcp_max, crit = [], [], [], [], []
    for a in alphas:
        ms = solve(elements, float(a))
        cl_i.append(ms.cl)
        cm.append(ms.cm_quarter)
        worst, who = -1.0, ""
        drag = 0.0
        for name, el, (_n, nodes) in zip(ms.names, ms.elements, elements):
            peak = float(el.cp[el.upper].min()) if el.upper.any() else float(el.cp.min())
            dc = 0.5 * (el.cp[0] + el.cp[-1]) - peak                    # upper-surface suction peak
            if dc > worst:
                worst, who = dc, name
            ce = float(np.ptp(nodes[:, 0]))
            try:
                drag += bl.analyse(el, reynolds * ce).cd * ce
            except Exception:  # noqa: BLE001 - a degenerate element: skin friction estimate
                drag += pm.profile_drag(reynolds * ce, 0.12) * ce
        dcp_max.append(worst)
        crit.append(who)
        cd.append(drag)
    cl_i = np.asarray(cl_i)
    dcp_max = np.asarray(dcp_max)
    factor = np.where(dcp_max > lim, np.clip(1.0 - (dcp_max - lim) / (0.3 * lim), 0.0, 1.0), 1.0)
    cl = pm.VISCOUS_LIFT_FACTOR * cl_i * (0.55 + 0.45 * factor)
    st, clm, who = math.nan, math.nan, ""
    over = np.nonzero(dcp_max >= lim)[0]
    if over.size and over[0] > 0:
        i = int(over[0])
        f = (lim - dcp_max[i - 1]) / max(dcp_max[i] - dcp_max[i - 1], 1e-12)
        st = float(alphas[i - 1] + f * (alphas[i] - alphas[i - 1]))
        clm = float(cl[i - 1] + f * (cl[i] - cl[i - 1]))
        who = crit[i]
    return MultiPolar(alphas, cl, cl_i, np.asarray(cd), np.asarray(cm), st, clm, who)


def clean_and_configured(main_outline: np.ndarray, hl: HighLift, reynolds: float,
                         alphas: Optional[np.ndarray] = None) -> tuple[MultiPolar, MultiPolar]:
    """Polars of the clean airfoil and of the high-lift configuration (for the ΔCL,max comparison)."""
    al = np.arange(-4.0, 26.01, 1.0) if alphas is None else alphas
    clean = polar(build_elements(main_outline, HighLift()), al, reynolds)
    conf = polar(build_elements(main_outline, hl), al, reynolds)
    return clean, conf


__all__ = ["HighLift", "MultiSolution", "build_elements", "solve", "MultiPolar", "polar", "clean_and_configured"]
