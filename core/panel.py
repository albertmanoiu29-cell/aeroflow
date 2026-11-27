"""2-D airfoil analysis with the Hess-Smith panel method (constant-strength source panels plus one uniform
vortex strength, Kutta condition at the trailing edge), and an engineering drag estimate.

Inviscid results (Cp, CL, Cm) are close to wind-tunnel data in the attached-flow range. Stall is estimated
with the Valarezo-Chin criterion (the flow separates once the pressure rise from the suction peak to the
trailing edge exceeds a Reynolds-number-dependent limit). The profile drag is a flat-plate skin-friction estimate with a thickness form factor
(laminar / turbulent by Reynolds number), with the standard induced-drag-free 2-D polar shape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def prepare(outline: np.ndarray, panels: int = 160) -> np.ndarray:
    """Nodes for the panel method: clockwise from the trailing edge along the lower surface to the leading
    edge and back along the upper surface, closed (first node = last node), cosine-spaced in x."""
    p = np.asarray(outline, dtype=float)
    if np.allclose(p[0], p[-1]):
        p = p[:-1]
    x, y = p[:, 0], p[:, 1]
    if 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) > 0.0:     # counter-clockwise: reverse
        p = p[::-1]
    p = np.roll(p, -int(np.argmax(p[:, 0])), axis=0)                          # start at the trailing edge
    le = int(np.argmin(p[:, 0]))
    lower, upper = p[:le + 1], np.vstack([p[le:], p[:1]])
    # resample each surface with cosine spacing (fine at both edges)
    n = max(panels // 2, 8)
    beta = np.linspace(0.0, math.pi, n + 1)
    xs = 0.5 * (1.0 - np.cos(beta))
    x0, x1 = float(p[le, 0]), float(p[0, 0])
    xs = x0 + (x1 - x0) * xs

    def along(surf: np.ndarray, xq: np.ndarray) -> np.ndarray:
        order = np.argsort(surf[:, 0])
        sx, sy = surf[order, 0], surf[order, 1]
        return np.interp(xq, sx, sy)

    yl = along(lower, xs)
    yu = along(upper, xs)
    low = np.column_stack([xs[::-1], yl[::-1]])            # TE -> LE
    up = np.column_stack([xs[1:], yu[1:]])                 # LE -> TE
    nodes = np.vstack([low, up])
    nodes[-1] = nodes[0] = [x1, 0.5 * (yl[-1] + yu[-1])]   # a closed (sharp) trailing edge
    return nodes


@dataclass
class PanelSolution:
    alpha_deg: float
    cl: float
    cm_quarter: float
    x: np.ndarray            # control points
    y: np.ndarray
    cp: np.ndarray
    upper: np.ndarray        # mask of upper-surface control points


def solve(nodes: np.ndarray, alpha_deg: float) -> PanelSolution:
    """Hess-Smith solution at one angle of attack (free stream speed 1, chord = the x extent)."""
    xn, yn = nodes[:, 0], nodes[:, 1]
    n = len(nodes) - 1
    xc, yc = 0.5 * (xn[:-1] + xn[1:]), 0.5 * (yn[:-1] + yn[1:])
    dx, dy = np.diff(xn), np.diff(yn)
    length = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    # geometry between control point i and the nodes of panel j
    rx0 = xc[:, None] - xn[None, :-1]
    ry0 = yc[:, None] - yn[None, :-1]
    rx1 = xc[:, None] - xn[None, 1:]
    ry1 = yc[:, None] - yn[None, 1:]
    r0 = np.hypot(rx0, ry0)
    r1 = np.hypot(rx1, ry1)
    beta = np.arctan2(ry1 * rx0 - rx1 * ry0, rx1 * rx0 + ry1 * ry0)       # angle panel j subtends at i
    np.fill_diagonal(beta, math.pi)
    lnr = np.log(r1 / r0)
    np.fill_diagonal(lnr, 0.0)
    d = th[:, None] - th[None, :]
    s, c = np.sin(d), np.cos(d)
    tp = 1.0 / (2.0 * math.pi)
    an_src = tp * (s * lnr + c * beta)          # normal velocity at i from a unit source on j
    at_src = tp * (s * beta - c * lnr)          # tangential
    an_vor = tp * (c * lnr - s * beta)          # unit vortex on j
    at_vor = tp * (s * lnr + c * beta)
    a = math.radians(alpha_deg)
    m = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)
    m[:n, :n] = an_src
    m[:n, n] = an_vor.sum(axis=1)
    rhs[:n] = -np.sin(a - th)
    # Kutta: equal tangential speed (opposite directions) on the first and last panels
    m[n, :n] = at_src[0] + at_src[-1]
    m[n, n] = at_vor[0].sum() + at_vor[-1].sum()
    rhs[n] = -math.cos(a - th[0]) - math.cos(a - th[-1])
    sol = np.linalg.solve(m, rhs)
    q, gam = sol[:n], sol[n]
    vt = at_src @ q + gam * at_vor.sum(axis=1) + np.cos(a - th)
    cp = 1.0 - vt * vt
    chord = float(xn.max() - xn.min())
    cl = 2.0 * gam * float(length.sum()) / chord
    # moment about the quarter chord from the pressure forces (nose-up positive)
    nx, ny = -np.sin(th), np.cos(th)                     # outward normals (clockwise nodes)
    fx, fy = -cp * length * nx, -cp * length * ny
    x0 = float(xn.min()) + 0.25 * chord
    cm = float(np.sum(-(xc - x0) * fy + yc * fx)) / chord ** 2
    le = int(np.argmin(xc))
    upper = np.arange(n) > le
    return PanelSolution(alpha_deg, cl, cm, xc, yc, cp, upper)


def thickness(nodes: np.ndarray) -> float:
    """Maximum thickness / chord."""
    le = int(np.argmin(nodes[:, 0]))
    low, up = nodes[:le + 1], nodes[le:]
    xs = np.linspace(nodes[:, 0].min(), nodes[:, 0].max(), 200)
    yl = np.interp(xs, low[::-1, 0], low[::-1, 1])
    yu = np.interp(xs, up[:, 0], up[:, 1])
    return float(np.max(yu - yl)) / float(np.ptp(nodes[:, 0]))


def profile_drag(reynolds: float, t_c: float, cl: float = 0.0) -> float:
    """Skin friction on both surfaces with a thickness form factor; laminar below Re 5×10⁵, mixed above
    (Prandtl-Schlichting with the laminar run subtracted). A mild lift term (∝ CL²) gives the bucket shape."""
    re = max(reynolds, 1e3)
    if re < 5e5:
        cf = 1.328 / math.sqrt(re)
    else:
        cf = 0.455 / math.log10(re) ** 2.58 - 1700.0 / re
    form = 1.0 + 2.0 * t_c + 60.0 * t_c ** 4
    return 2.0 * cf * form * (1.0 + 0.25 * cl * cl)


#: Boundary-layer "decambering": measured lift slopes are about 0.9 × the inviscid ones at chord Reynolds numbers
#: of 10⁶-10⁷ (Abbott & von Doenhoff), so the viscous estimate scales the inviscid lift by this factor.
VISCOUS_LIFT_FACTOR = 0.9


def stall_delta_cp(reynolds: float) -> float:
    """Valarezo-Chin limit of Cp_TE - Cp_min at stall (low Mach), against the chord Reynolds number."""
    lr = math.log10(min(max(reynolds, 1e5), 2e7))
    return float(np.interp(lr, [5.0, 6.0, math.log10(3e6), 7.0, math.log10(2e7)], [6.0, 10.0, 12.5, 14.0, 14.5]))


@dataclass
class Polar:
    alpha: np.ndarray
    cl: np.ndarray           # viscous estimate (VISCOUS_LIFT_FACTOR × inviscid)
    cl_inviscid: np.ndarray
    cd: np.ndarray
    cm: np.ndarray
    delta_cp: np.ndarray     # Cp_TE - Cp_min (the suction peak's pressure recovery)
    t_c: float
    cl_alpha: float          # per radian
    alpha_zero_lift: float   # degrees
    stall_alpha: float       # degrees (nan if not reached in the sweep)
    cl_max: float            # viscous estimate; nan if not reached
    stall_negative: float    # degrees, the negative-α stall (nan if not reached)


def polar(nodes: np.ndarray, alphas: np.ndarray, reynolds: float) -> Polar:
    alphas = np.asarray(alphas, float)
    tc = thickness(nodes)
    sols = [solve(nodes, float(a)) for a in alphas]
    cl_inv = np.array([s.cl for s in sols])
    cl = VISCOUS_LIFT_FACTOR * cl_inv
    cm = np.array([s.cm_quarter for s in sols])
    dcp = np.array([0.5 * (s.cp[0] + s.cp[-1]) - float(s.cp.min()) for s in sols])
    lim = stall_delta_cp(reynolds)
    # skin friction, plus pressure drag that grows as the boundary layer nears separation (thickening wake)
    cd = np.array([profile_drag(reynolds, tc, c) for c in cl]) + 0.015 * np.clip(dcp / lim, 0.0, 1.5) ** 6
    k = np.argsort(np.abs(alphas))[:2]
    slope = float((cl[k[1]] - cl[k[0]]) / math.radians(alphas[k[1]] - alphas[k[0]])) if len(alphas) > 1 else 0.0
    a0 = float(alphas[k[0]] - math.degrees(cl[k[0]] / slope)) if abs(slope) > 1e-9 else 0.0
    st, clm, neg = math.nan, math.nan, math.nan
    pos = alphas > a0
    for i in range(1, len(alphas)):
        if pos[i] and pos[i - 1] and dcp[i - 1] < lim <= dcp[i]:
            f = (lim - dcp[i - 1]) / (dcp[i] - dcp[i - 1])
            st = float(alphas[i - 1] + f * (alphas[i] - alphas[i - 1]))
            clm = float(cl[i - 1] + f * (cl[i] - cl[i - 1]))
            break
    for i in range(len(alphas) - 1, 0, -1):
        if not pos[i] and not pos[i - 1] and dcp[i] < lim <= dcp[i - 1]:
            f = (lim - dcp[i]) / (dcp[i - 1] - dcp[i])
            neg = float(alphas[i] + f * (alphas[i - 1] - alphas[i]))
            break
    return Polar(alphas, cl, cl_inv, cd, cm, dcp, tc, slope, a0, st, clm, neg)


__all__ = ["prepare", "solve", "PanelSolution", "thickness", "profile_drag", "polar", "Polar", "stall_delta_cp"]
