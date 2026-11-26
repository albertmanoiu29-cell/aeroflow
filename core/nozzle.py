"""Rocket Nozzle Lab: nozzle geometry, 1-D rocket-nozzle theory and a live axisymmetric Euler solver.

Model
-----
* **Gas**: one calorically perfect gas with the exhaust's ratio of specific heats γ and molar mass
  (frozen composition). The surrounding air is represented by the same gas at the ambient pressure and
  temperature. The jet's shock structure is set by the pressure ratio and γ, which are exact; the
  ambient density differs from real air by the molar-mass ratio.
* **Equations**: compressible Euler equations, axisymmetric (round nozzle: the r-x half plane with the
  axis on the bottom edge) or planar (a 2-D slot nozzle). Finite volumes with face areas ∝ r and the
  radial pressure source p/r, which keeps a gas at rest exactly at rest. Fluxes, reconstruction,
  immersed-boundary walls and time stepping are those of :mod:`core.supersonic` (HLLC/HLL Riemann
  solver, MUSCL-minmod, SSP-RK2).
* **Boundaries**: the injector face is a subsonic inlet at the feed total pressure and the chamber
  temperature. It acts as a closed wall before ignition, after shut-down, and whenever the chamber
  pressure exceeds the feed pressure. The axis is a symmetry line, and the far field holds the ambient
  pressure (or the flight free stream). The outflow extrapolates where supersonic and holds the ambient
  pressure where subsonic.
* **Ignition**: the feed pressure rises from ambient to the chamber pressure along a smooth ramp, so
  the start-up transient (starting shock, starting vortex, flow separation in the nozzle) plays out live.

The solver works in non-dimensional units: the reference ambient state of the last restart
(ρ_ref = ρ_a, a_ref = a_a) and the cell size Δx. Then T' = γ p'/ρ' = T / T_ref, and the reference ambient
pressure is 1/γ.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Final, Optional

import numpy as np
from numpy.typing import NDArray

from core import accel, compute
from core import supersonic as ss
from core.shapes2d import polygon_segments, rasterize_segments

log = logging.getLogger("aeroflow.core.nozzle")

F64 = NDArray[np.float64]
G0: Final[float] = 9.80665
R_UNIVERSAL: Final[float] = 8314.462618        # J / (kmol K)
G = ss.G
NEVER: Final[float] = 1e300                     # "not yet" for ignition / shut-down times

# Boundary-parameter vector (shared with the CUDA kernels, which read it from device memory so that
# throttling, altitude changes and ignition never invalidate the captured CUDA graphs).
BC_PA, BC_RHOA, BC_UF, BC_P0, BC_T0, BC_TIGN, BC_TAU, BC_TOFF, BC_TAUOFF, BC_FOFF, BC_NIN, BC_MC = range(12)
BC_SIZE: Final[int] = 16


# ---------------------------------------------------------------------------
# Propellants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Propellant:
    """Exhaust-gas properties (frozen, typical values at the usual mixture ratio)."""

    name: str
    gamma: float
    molar_mass: float              # kg / kmol
    chamber_temperature: float     # K
    note: str = ""

    @property
    def gas_constant(self) -> float:
        """Specific gas constant R = R_u / M in J / (kg K)."""
        return R_UNIVERSAL / self.molar_mass


PROPELLANTS: Final[dict[str, Propellant]] = {p.name: p for p in (
    Propellant("LOX / RP-1 (kerosene)", 1.22, 22.5, 3650.0, "O/F ≈ 2.6, like Merlin or RD-180"),
    Propellant("LOX / LH2 (hydrogen)", 1.21, 13.4, 3400.0, "O/F ≈ 6, like RS-25 or Vulcain"),
    Propellant("LOX / CH4 (methane)", 1.21, 21.5, 3550.0, "O/F ≈ 3.4, like Raptor or BE-4"),
    Propellant("N2O4 / MMH (hypergolic)", 1.23, 21.5, 3300.0, "storable, spacecraft engines"),
    Propellant("APCP solid (AP / Al / HTPB)", 1.20, 25.0, 3300.0, "composite solid rocket motor"),
    Propellant("Hydrogen peroxide 90 % (mono)", 1.26, 22.0, 1020.0, "monopropellant over a silver catalyst"),
    Propellant("Hydrazine (mono)", 1.28, 13.0, 1200.0, "monopropellant thruster"),
    Propellant("Cold gas: nitrogen", 1.40, 28.01, 300.0, "cold-gas thruster"),
    Propellant("Cold gas: air", 1.40, 28.97, 288.15, "laboratory air jet"),
)}
DEFAULT_PROPELLANT: Final[str] = "LOX / RP-1 (kerosene)"


@dataclass(frozen=True)
class Gas:
    """The working gas of a run."""

    gamma: float = 1.22
    molar_mass: float = 22.5
    chamber_temperature: float = 3650.0

    @property
    def gas_constant(self) -> float:
        return R_UNIVERSAL / self.molar_mass

    @classmethod
    def of(cls, p: Propellant) -> "Gas":
        return cls(p.gamma, p.molar_mass, p.chamber_temperature)


# ---------------------------------------------------------------------------
# Isentropic / shock relations for any gamma
# ---------------------------------------------------------------------------
def area_mach(m: float, g: float) -> float:
    """A / A* of isentropic flow at Mach ``m``."""
    m = max(m, 1e-12)
    return (2.0 / (g + 1.0) * (1.0 + 0.5 * (g - 1.0) * m * m)) ** ((g + 1.0) / (2.0 * (g - 1.0))) / m


def p_over_p0(m: float, g: float) -> float:
    return (1.0 + 0.5 * (g - 1.0) * m * m) ** (-g / (g - 1.0))


def t_over_t0(m: float, g: float) -> float:
    return 1.0 / (1.0 + 0.5 * (g - 1.0) * m * m)


def mach_from_pressure_ratio(p_p0: float, g: float) -> float:
    """Isentropic Mach number at static / total pressure ``p_p0``."""
    p_p0 = min(max(p_p0, 1e-300), 1.0)
    return math.sqrt(max(2.0 / (g - 1.0) * (p_p0 ** (-(g - 1.0) / g) - 1.0), 0.0))


def mach_from_area_ratio(eps: float, g: float, supersonic: bool = True) -> float:
    """Mach number at area ratio A/A* = ``eps`` (the supersonic or the subsonic root)."""
    if eps <= 1.0:
        return 1.0
    lo, hi = (1.0, 200.0) if supersonic else (1e-9, 1.0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        above = area_mach(mid, g) > eps
        if supersonic:
            hi, lo = (mid, lo) if above else (hi, mid)
        else:
            lo, hi = (mid, hi) if above else (lo, mid)
        if hi - lo < 1e-13 * max(hi, 1.0):
            break
    return 0.5 * (lo + hi)


def normal_shock_p_ratio(m1: float, g: float) -> float:
    return 1.0 + 2.0 * g / (g + 1.0) * (m1 * m1 - 1.0)


def normal_shock_p0_ratio(m1: float, g: float) -> float:
    """Total-pressure recovery p02 / p01 across a normal shock."""
    a = ((g + 1.0) * m1 * m1 / ((g - 1.0) * m1 * m1 + 2.0)) ** (g / (g - 1.0))
    b = ((g + 1.0) / (2.0 * g * m1 * m1 - (g - 1.0))) ** (1.0 / (g - 1.0))
    return a * b


def critical_pressure_ratio(g: float) -> float:
    """p* / p0: below this back-pressure ratio a nozzle throat is choked."""
    return (2.0 / (g + 1.0)) ** (g / (g - 1.0))


def characteristic_velocity(gas: Gas) -> float:
    """c* = p_c A_t / ṁ (m/s), set by the propellant alone."""
    g = gas.gamma
    return math.sqrt(g * gas.gas_constant * gas.chamber_temperature) / (
        g * math.sqrt((2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0))))


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
class Contour(str, Enum):
    """Shape of the divergent section."""

    BELL = "Bell (Rao thrust-optimised)"
    CONICAL = "Conical"
    SONIC = "Convergent only (sonic)"


_RAO_EPS = np.log(np.array([4.0, 10.0, 20.0, 50.0, 100.0]))
_RAO_TN = np.array([21.5, 26.5, 29.5, 33.0, 35.5])            # 80 % bell, typical (Rao, Huzel & Huang)
_RAO_TE = np.array([14.0, 10.5, 9.0, 7.5, 6.5])


def rao_angles(eps: float, fraction: float = 0.8) -> tuple[float, float]:
    """Typical start / exit wall angles (θn, θe) of a Rao parabolic bell, in degrees."""
    le = math.log(min(max(eps, 2.0), 200.0))
    tn = float(np.interp(le, _RAO_EPS, _RAO_TN))
    te = float(np.interp(le, _RAO_EPS, _RAO_TE))
    d = (0.8 - fraction) / 0.2                  # shorter bells turn the flow faster
    tn, te = tn + 5.0 * d, te + 4.0 * d
    tn = min(max(tn, 10.0), 50.0)
    return tn, min(max(te, 1.0), tn - 2.0)


@dataclass(frozen=True)
class NozzleGeometry:
    """Physical nozzle dimensions (metres, degrees)."""

    throat_diameter: float = 0.10
    exit_diameter: float = 0.35
    chamber_diameter: float = 0.20
    chamber_length: float = 0.08
    convergent_angle: float = 35.0
    contour: Contour = Contour.BELL
    divergent_angle: float = 15.0               # conical half-angle
    bell_fraction: float = 0.80                 # bell length / 15° cone length
    theta_n: float = 0.0                        # bell start angle; 0 = typical Rao value
    theta_e: float = 0.0                        # bell exit angle; 0 = typical Rao value
    throat_upstream_radius: float = 1.5         # × throat radius
    throat_downstream_radius: float = 0.382     # × throat radius
    wall_thickness: float = 0.008
    vehicle_diameter: float = 0.0               # rocket body around the engine (0 = bare engine)

    @property
    def rt(self) -> float:
        return 0.5 * self.throat_diameter

    @property
    def re(self) -> float:
        return self.rt if self.contour is Contour.SONIC else max(0.5 * self.exit_diameter, self.rt)

    @property
    def rc(self) -> float:
        return max(0.5 * self.chamber_diameter, 1.05 * self.rt)

    def area_ratio(self, planar: bool = False) -> float:
        """Exit / throat area (ε)."""
        q = self.re / self.rt
        return q if planar else q * q

    def bell_angles(self) -> tuple[float, float]:
        tn0, te0 = rao_angles(self.area_ratio(), self.bell_fraction)
        tn = self.theta_n if self.theta_n > 0 else tn0
        te = self.theta_e if self.theta_e > 0 else te0
        return tn, min(te, tn - 1.0)

    def inner_contour(self) -> tuple[F64, F64]:
        """Inner wall (x, r) from the injector face (x = 0) to the exit plane, x increasing."""
        rt, rc, re = self.rt, self.rc, self.re
        r1 = max(self.throat_upstream_radius, 0.05) * rt
        r2 = max(self.throat_downstream_radius, 0.05) * rt
        tc = math.radians(min(max(self.convergent_angle, 10.0), 75.0))
        p1r = rt + r1 * (1.0 - math.cos(tc))           # cone / upstream-arc tangent point
        if rc <= p1r:
            rc = p1r + 0.02 * rt
        lc = max(self.chamber_length, 0.0)
        cone = (rc - p1r) / math.tan(tc)
        xt = lc + cone + r1 * math.sin(tc)
        xs: list[float] = [0.0, lc]
        rs: list[float] = [rc, rc]
        for s in np.linspace(0.0, 1.0, 24)[1:]:
            xs.append(lc + s * cone)
            rs.append(rc - s * (rc - p1r))
        for phi in np.linspace(tc, 0.0, 40)[1:]:
            xs.append(xt - r1 * math.sin(phi))
            rs.append(rt + r1 * (1.0 - math.cos(phi)))
        if self.contour is Contour.SONIC:
            return np.asarray(xs), np.asarray(rs)
        if self.contour is Contour.CONICAL:
            tn = math.radians(min(max(self.divergent_angle, 2.0), 45.0))
        else:
            tn = math.radians(self.bell_angles()[0])
        # downstream throat arc, stopped early if it already reaches the exit radius
        if rt + r2 * (1.0 - math.cos(tn)) > re:
            tn = math.acos(max(1.0 - (re - rt) / r2, -1.0))
        for phi in np.linspace(0.0, tn, 40)[1:]:
            xs.append(xt + r2 * math.sin(phi))
            rs.append(rt + r2 * (1.0 - math.cos(phi)))
        nx_, nr_ = xs[-1], rs[-1]
        if re - nr_ <= 1e-9 * rt:
            return np.asarray(xs), np.asarray(rs)
        if self.contour is Contour.CONICAL:
            ex = nx_ + (re - nr_) / math.tan(tn)
            for s in np.linspace(0.0, 1.0, 120)[1:]:
                xs.append(nx_ + s * (ex - nx_))
                rs.append(nr_ + s * (re - nr_))
            return np.asarray(xs), np.asarray(rs)
        # Rao parabola: quadratic Bezier N -> E, tangent to θn at N and to θe at E
        eps = self.area_ratio()
        ln = max(self.bell_fraction, 0.3) * (math.sqrt(eps) - 1.0) * rt / math.tan(math.radians(15.0))
        ex, er = xt + max(ln, (nx_ - xt) + 0.5 * rt), re
        te = math.radians(self.bell_angles()[1])
        m1, m2 = math.tan(tn), math.tan(te)
        c1, c2 = nr_ - m1 * nx_, er - m2 * ex
        qx = (c2 - c1) / (m1 - m2) if abs(m1 - m2) > 1e-9 else 0.5 * (nx_ + ex)
        qr = m1 * qx + c1
        if not (nx_ < qx < ex and nr_ <= qr <= er + 1e-12):
            qx, qr = 0.5 * (nx_ + ex), 0.5 * (nr_ + er)  # inconsistent angles: fall back to a straight line
        for s in np.linspace(0.0, 1.0, 160)[1:]:
            xs.append((1 - s) ** 2 * nx_ + 2 * (1 - s) * s * qx + s * s * ex)
            rs.append((1 - s) ** 2 * nr_ + 2 * (1 - s) * s * qr + s * s * er)
        return np.asarray(xs), np.asarray(rs)

    def throat_x(self) -> float:
        x, r = self.inner_contour()
        return float(x[int(np.argmin(r))])

    def length(self) -> float:
        """Injector face to exit plane."""
        return float(self.inner_contour()[0][-1])

    def x_at_area_ratio(self, eps: float, planar: bool = False) -> Optional[float]:
        """Position in the divergent section where A/A_t = ``eps`` (None outside the nozzle)."""
        x, r = self.inner_contour()
        k = int(np.argmin(r))
        xd, rd = x[k:], r[k:] / self.rt
        a = rd if planar else rd * rd
        if eps < a[0] or eps > a[-1]:
            return None
        return float(np.interp(eps, a, xd))


# ---------------------------------------------------------------------------
# 1-D rocket nozzle theory
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NozzlePerformance:
    """Quasi-1-D isentropic nozzle performance with the classical corrections."""

    regime: str
    regime_note: str
    choked: bool
    chamber_pressure: float          # Pa
    ambient_pressure: float          # Pa
    mass_flow: float                 # kg/s (per metre of width for a planar nozzle)
    cstar: float                     # m/s
    exit_mach: float
    exit_pressure: float             # Pa
    exit_temperature: float          # K
    exit_velocity: float             # m/s
    thrust: float                    # N, with the divergence loss
    thrust_ideal: float              # N, 1-D (λ = 1)
    thrust_vacuum: float             # N
    isp: float                       # s
    isp_vacuum: float                # s
    cf: float
    divergence_factor: float         # λ
    expansion_ratio: float
    optimum_expansion_ratio: float
    jet_mach: float                  # fully expanded jet Mach number
    jet_diameter: float              # m (full height for planar)
    shock_cell_spacing: Optional[float]    # m (Prandtl vortex-sheet model)
    mach_disk_distance: Optional[float]    # m from the exit (Ashkenas-Sherman; sonic nozzles)
    separation: bool
    separation_x: Optional[float]          # m from the injector face (Summerfield)
    shock_x: Optional[float]               # m: 1-D normal shock inside the nozzle
    throat_area: float
    exit_area: float


def nozzle_theory(geom: NozzleGeometry, gas: Gas, pc: float, pa: float, planar: bool = False) -> NozzlePerformance:
    """Performance of ``geom`` fed at chamber pressure ``pc`` into ambient pressure ``pa`` (both Pa)."""
    g, rgas, tc = gas.gamma, gas.gas_constant, gas.chamber_temperature
    pa = max(pa, 1e-6)
    pc = max(pc, 1e-6)
    rt, re = geom.rt, geom.re
    at = 2.0 * rt if planar else math.pi * rt * rt
    ae = 2.0 * re if planar else math.pi * re * re
    eps = ae / at
    cstar = characteristic_velocity(gas)
    m_sub = mach_from_area_ratio(eps, g, False) if eps > 1.0 + 1e-9 else 1.0
    m_sup = mach_from_area_ratio(eps, g, True) if eps > 1.0 + 1e-9 else 1.0
    pe_sub, pe_sup = pc * p_over_p0(m_sub, g), pc * p_over_p0(m_sup, g)
    p_ns = pe_sup * normal_shock_p_ratio(m_sup, g) if m_sup > 1.0 else pe_sup
    shock_x = None
    if pa >= pe_sub:
        choked = False
        me = mach_from_pressure_ratio(pa / pc, g) if pc > pa else 0.0
        te = tc * t_over_t0(me, g)
        ve = me * math.sqrt(g * rgas * te)
        pe = pa
        mdot = pa / (rgas * te) * ve * ae
        regime = "Unchoked (subsonic)"
        note = ("The feed pressure is too low to choke the throat: the flow stays subsonic and leaves at the "
                "ambient pressure. Raise the chamber pressure.")
    else:
        choked = True
        mdot = pc * at / cstar
        if eps > 1.0 + 1e-9 and pa > p_ns:
            # normal shock in the divergent section: pe = pa, total pressure lost across the shock
            target = pa / pc * eps                     # = (pe/p02) (Ae/A2*)
            lo, hi = 1e-9, 1.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if p_over_p0(mid, g) * area_mach(mid, g) > target:
                    lo = mid
                else:
                    hi = mid
            me = 0.5 * (lo + hi)
            p0_ratio = (pa / pc) / p_over_p0(me, g)
            lo, hi = 1.0, m_sup
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if normal_shock_p0_ratio(mid, g) > p0_ratio:
                    lo = mid
                else:
                    hi = mid
            shock_x = geom.x_at_area_ratio(area_mach(0.5 * (lo + hi), g), planar)
            pe = pa
            regime = "Shock inside the nozzle"
            note = ("1-D theory puts a normal shock in the divergent section: the nozzle is heavily over-"
                    "expanded. A real nozzle separates from the wall first (see the separation estimate).")
        else:
            me, pe = m_sup, pe_sup
            if abs(pe / pa - 1.0) < 0.03:
                regime = "Ideally expanded"
                note = "The exit pressure matches the ambient: the most thrust this nozzle can give here."
            elif pe < pa:
                regime = "Overexpanded"
                note = ("The exit pressure is below the ambient: oblique shocks from the lip squeeze the jet, "
                        "forming the shock diamonds.")
            else:
                regime = "Underexpanded"
                note = ("The exit pressure is above the ambient: expansion fans at the lip let the jet bulge "
                        "out, then reflect as shocks (shock diamonds, Mach disks when strong).")
        te = tc * t_over_t0(me, g)
        ve = me * math.sqrt(g * rgas * te)
    if geom.contour is Contour.SONIC:
        lam = 1.0
    else:
        alpha = math.radians(geom.divergent_angle if geom.contour is Contour.CONICAL else geom.bell_angles()[1])
        lam = math.sin(alpha) / alpha if planar else 0.5 * (1.0 + math.cos(alpha))
    thrust_ideal = mdot * ve + (pe - pa) * ae
    thrust = mdot * ve * lam + (pe - pa) * ae
    if choked:
        te_v = tc * t_over_t0(m_sup, g)
        thrust_vac = mdot * m_sup * math.sqrt(g * rgas * te_v) * lam + pe_sup * ae
    else:
        thrust_vac = float("nan")
    mj = mach_from_pressure_ratio(pa / pc, g) if pc > pa else 0.0
    eps_opt = area_mach(mj, g) if mj > 1.0 else 1.0
    dj = 2.0 * rt * (eps_opt if planar else math.sqrt(eps_opt))
    cell = None
    if choked and mj > 1.0 and regime in ("Overexpanded", "Underexpanded") or (choked and eps <= 1.0 + 1e-9 and mj > 1.0):
        beta = math.sqrt(mj * mj - 1.0)
        cell = (2.0 if planar else 1.306) * beta * dj      # Prandtl vortex-sheet model (Tam & Tanna)
    disk = None
    if choked and eps <= 1.0 + 1e-9 and pc / pa > 4.0 and not planar:
        disk = 0.67 * 2.0 * re * math.sqrt(pc / pa)        # Ashkenas & Sherman (sonic orifice)
    sep, sep_x = False, None
    if choked and eps > 1.0 + 1e-9 and pe_sup < 0.4 * pa:
        sep = True                                          # Summerfield: separation near p_wall ≈ 0.4 p_a
        m_sep = mach_from_pressure_ratio(0.4 * pa / pc, g)
        sep_x = geom.x_at_area_ratio(area_mach(m_sep, g), planar) if m_sep > 1.0 else None
    return NozzlePerformance(
        regime=regime, regime_note=note, choked=choked, chamber_pressure=pc, ambient_pressure=pa,
        mass_flow=mdot, cstar=cstar, exit_mach=me, exit_pressure=pe, exit_temperature=te, exit_velocity=ve,
        thrust=thrust, thrust_ideal=thrust_ideal, thrust_vacuum=thrust_vac, isp=thrust / (mdot * G0),
        isp_vacuum=thrust_vac / (mdot * G0) if choked else float("nan"), cf=thrust / (pc * at),
        divergence_factor=lam, expansion_ratio=eps, optimum_expansion_ratio=eps_opt, jet_mach=mj, jet_diameter=dj,
        shock_cell_spacing=cell, mach_disk_distance=disk, separation=sep, separation_x=sep_x, shock_x=shock_x,
        throat_area=at, exit_area=ae)


def chamber_pressure_for_mass_flow(mdot: float, geom: NozzleGeometry, gas: Gas, planar: bool = False) -> float:
    """Chamber pressure (Pa) that passes ``mdot`` through a choked throat: p_c = ṁ c* / A_t."""
    at = 2.0 * geom.rt if planar else math.pi * geom.rt ** 2
    return mdot * characteristic_velocity(gas) / at


def wall_pressure_1d(geom: NozzleGeometry, gas: Gas, pc: float, planar: bool = False) -> tuple[F64, F64]:
    """Isentropic wall pressure (x, p in Pa) along the nozzle: subsonic before the throat, supersonic after."""
    x, r = geom.inner_contour()
    k = int(np.argmin(r))
    a = r / geom.rt
    a = a if planar else a * a
    p = np.empty_like(x)
    for i, ai in enumerate(a):
        m = mach_from_area_ratio(float(ai), gas.gamma, supersonic=i > k)
        p[i] = pc * p_over_p0(m, gas.gamma)
    return x, p


# ---------------------------------------------------------------------------
# Grid layout
# ---------------------------------------------------------------------------
@dataclass
class NozzleLayout:
    """The nozzle placed on the solver grid."""

    nx: int
    ny: int
    dx: float                        # cell size (m)
    mask: NDArray[np.bool_]          # solid cells (ny, nx)
    segments: F64                    # wall outline in cell coordinates (M, 2, 2)
    inside: NDArray[np.bool_]        # fluid cells inside the engine
    n_inlet: int                     # rows of the injector face (inlet)
    i_exit: int                      # last column inside the nozzle
    i_throat: int
    x_exit: float                    # m
    x_throat: float
    wall_x: F64                      # closed wall polygon (m) for drawing
    wall_r: F64
    contour_x: F64
    contour_r: F64
    wall_thickness: float            # m, as built (at least ~2.5 cells)
    kappa: F64                       # fluid volume fraction of each cell (ny, nx)
    ax: F64                          # open fraction of each x face (ny, nx + 1)
    ay: F64                          # open fraction of each y face (ny + 1, nx)

    @property
    def throat_cells(self) -> float:
        return float(np.min(self.contour_r)) / self.dx

    def r_inner(self, x: F64 | float) -> F64:
        return np.interp(x, self.contour_x, self.contour_r)

    def chamber_mach(self, gamma: float, planar: bool = False) -> float:
        """1-D Mach number in the chamber at the injector face (subsonic root of A_c / A_t)."""
        q = float(self.contour_r[0] / np.min(self.contour_r))
        return mach_from_area_ratio(q if planar else q * q, gamma, supersonic=False)


def cut_cells(segs: F64, nx: int, ny: int, ncol: int, k: int = 8) -> tuple[F64, F64, F64]:
    """Fluid volume fractions and face apertures of the first ``ncol`` columns by ``k``× supersampling of
    the wall polygon (the rest of the grid is open). Returns ``(kappa, ax, ay)``."""
    kappa = np.ones((ny, nx))
    ax = np.ones((ny, nx + 1))
    ay = np.ones((ny + 1, nx))
    ncol = int(min(max(ncol, 1), nx))
    fine = ~rasterize_segments((segs + 0.5) * k - 0.5, ncol * k, ny * k)          # True = fluid
    f = fine.astype(np.float64)
    kappa[:, :ncol] = f.reshape(ny, k, ncol, k).mean(axis=(1, 3))
    rows = f.reshape(ny, k, ncol * k).mean(axis=1)                                 # (ny, fine columns)
    ax[:, 0] = rows[:, 0]
    if ncol > 1:
        ax[:, 1:ncol] = 0.5 * (rows[:, k - 1:-1:k][:, :ncol - 1] + rows[:, k::k][:, :ncol - 1])
    ax[:, ncol] = rows[:, -1] if ncol == nx else 0.5 * (rows[:, -1] + 1.0)
    cols = f.reshape(ny * k, ncol, k).mean(axis=2)                                 # (fine rows, columns)
    ay[0, :ncol] = cols[0]
    ay[1:ny, :ncol] = 0.5 * (cols[k - 1:-1:k][:ny - 1] + cols[k::k][:ny - 1])
    ay[ny, :ncol] = cols[-1]
    return kappa, ax, ay


def layout_nozzle(geom: NozzleGeometry, nx: int, ny: int, plume_length: float = 10.0,
                  plume_radius: float = 3.5) -> NozzleLayout:
    """Scale ``geom`` onto an ``nx`` × ``ny`` grid: the injector face on the left edge, the axis on the
    bottom edge, ``plume_length`` exit diameters of plume downstream and ``plume_radius`` exit radii above."""
    cx, cr = geom.inner_contour()
    x_e = float(cx[-1])
    re = geom.re
    r_out = float(cr.max()) + geom.wall_thickness
    rv = 0.5 * geom.vehicle_diameter
    l_need = x_e + max(plume_length, 1.0) * 2.0 * re
    r_need = max(max(plume_radius, 1.3) * re, 1.25 * r_out, 1.15 * rv)
    dx = max(l_need / nx, r_need / ny)
    t = max(geom.wall_thickness, 2.6 * dx)
    x0 = -3.0 * dx
    ro = cr + t
    inner_x = np.concatenate([[x0], cx])
    inner_r = np.concatenate([[cr[0]], cr])
    lc = max(geom.chamber_length, 0.0)
    if rv > float(np.interp(lc, cx, ro)) + dx:
        sel = cx >= lc
        ox = np.concatenate([cx[sel][::-1], [lc, lc, x0]])
        orr = np.concatenate([ro[sel][::-1], [float(np.interp(lc, cx, ro)), rv, rv]])
    else:
        ox = np.concatenate([cx[::-1], [x0]])
        orr = np.concatenate([ro[::-1], [ro[0]]])
    wall_x = np.concatenate([inner_x, [x_e], ox])
    wall_r = np.concatenate([inner_r, [ro[-1]], orr])
    pts = np.column_stack([wall_x / dx - 0.5, wall_r / dx - 0.5])
    segs = polygon_segments(pts)
    i_exit = int(min(max(math.floor(x_e / dx - 0.5), 0), nx - 1))
    kappa, ax, ay = cut_cells(segs, nx, ny, i_exit + 6)
    mask = kappa < 0.02                         # (nearly) all-solid cells are walls; the rest are cut cells
    ax[:, 1:-1][mask[:, 1:] | mask[:, :-1]] = 0.0
    ax[:, 0][mask[:, 0]] = 0.0
    ax[:, -1][mask[:, -1]] = 0.0
    ay[1:-1][mask[1:] | mask[:-1]] = 0.0
    ay[0][mask[0]] = 0.0
    ay[-1][mask[-1]] = 0.0
    xc = (np.arange(nx) + 0.5) * dx
    rc_ = (np.arange(ny) + 0.5) * dx
    inside = (xc[None, :] < x_e) & (rc_[:, None] < np.interp(xc, cx, cr)[None, :]) & ~mask
    n_inlet = int(np.sum((rc_ < cr[0] + 0.5 * dx) & ~mask[:, 0]))
    x_t = float(cx[int(np.argmin(cr))])
    i_throat = int(min(max(round(x_t / dx - 0.5), 0), nx - 1))
    return NozzleLayout(nx, ny, dx, mask, segs, inside, n_inlet, i_exit, i_throat, x_e, x_t, wall_x, wall_r,
                        cx, cr, t, kappa, ax, ay)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
if accel.HAVE_NUMBA:
    from numba import njit, prange

    @njit(inline="always", cache=True)
    def _wall_pressure(r, u, v, p, nx_, ny_, g):  # pragma: no cover - compiled
        """Pressure on a wall with unit normal (nx_, ny_) pointing from the fluid into the wall: the exact
        piston solution (a reflected shock when the gas moves into the wall, an expansion when away)."""
        un = u * nx_ + v * ny_
        c = math.sqrt(g * p / r)
        if un > 0.0:
            k = 0.25 * (g + 1.0) * un
            return p + r * un * (k + math.sqrt(k * k + c * c))
        x = 1.0 + 0.5 * (g - 1.0) * un / c
        if x <= 0.0:
            return 0.0
        return p * x ** (2.0 * g / (g - 1.0))

    @njit(parallel=True, cache=True, nogil=True)
    def _rk_cut_nb(Ubase, Ucur, F, Gf, W, solid, ax, ay, wx, wr, nwx, nwr, ks, dt, a, axi, g,
                   out):  # pragma: no cover - compiled
        """Fused SSP-RK stage of the cut-cell (embedded-boundary) scheme: face fluxes times the open
        fraction of each face, axisymmetric face areas r_face / r_cell, the radial pressure source κ p / r,
        and the wall pressure on the cut part of each cell (projected areas ``wx``, ``wr``). ``axi = False``
        gives the planar update. ``out`` may alias ``Ubase``."""
        _, ny, nx = out.shape
        b = 1.0 - a
        for jj in prange(ny):
            rm = 1.0
            rp = 1.0
            if axi:
                rc = jj + 0.5
                rm = jj / rc
                rp = (jj + 1.0) / rc
            for ii in range(nx):
                if solid[jj + 2, ii + 2]:
                    continue
                aL = ax[jj, ii]
                aR = ax[jj, ii + 1]
                aB = ay[jj, ii] * rm
                aT = ay[jj + 1, ii] * rp
                pw = 0.0
                if wx[jj, ii] != 0.0 or wr[jj, ii] != 0.0:
                    pw = _wall_pressure(W[0, jj + 2, ii + 2], W[1, jj + 2, ii + 2], W[2, jj + 2, ii + 2],
                                        W[3, jj + 2, ii + 2], nwx[jj, ii], nwr[jj, ii], g)
                for k in range(4):
                    r = aL * F[k, jj, ii] - aR * F[k, jj, ii + 1] + aB * Gf[k, jj, ii] - aT * Gf[k, jj + 1, ii]
                    if k == 1:
                        r += pw * wx[jj, ii]
                    elif k == 2:
                        r += ks[jj, ii] * W[3, jj + 2, ii + 2] + pw * wr[jj, ii]
                    out[k, jj, ii] = a * Ubase[k, jj, ii] + b * (Ucur[k, jj, ii] + dt * r)


def _fluxes_np(W: F64, wall: NDArray[np.bool_], g: float) -> tuple[F64, F64]:
    """First-order HLL face fluxes (NumPy fallback): x faces (4, ny, nx+1), y faces (4, ny+1, nx)."""
    out = []
    for axis in (0, 1):
        if axis == 0:
            Lw, Rw = W[:, 2:-2, 1:-2], W[:, 2:-2, 2:-1]
            Ls, Rs = wall[2:-2, 1:-2], wall[2:-2, 2:-1]
            nl, tl, nr, tr = Lw[1], Lw[2], Rw[1], Rw[2]
        else:
            Lw, Rw = W[:, 1:-2, 2:-2], W[:, 2:-1, 2:-2]
            Ls, Rs = wall[1:-2, 2:-2], wall[2:-1, 2:-2]
            nl, tl, nr, tr = Lw[2], Lw[1], Rw[2], Rw[1]
        rl, pl, rr, pr = Lw[0].copy(), Lw[3].copy(), Rw[0].copy(), Rw[3].copy()
        nl, tl, nr, tr = nl.copy(), tl.copy(), nr.copy(), tr.copy()
        m = Ls & ~Rs
        rl[m], nl[m], tl[m], pl[m] = rr[m], -nr[m], tr[m], pr[m]
        m = Rs & ~Ls
        rr[m], nr[m], tr[m], pr[m] = rl[m], -nl[m], tl[m], pl[m]
        f = ss._hll_np(rl, nl, tl, pl, rr, nr, tr, pr, g)
        f[:, Ls & Rs] = 0.0
        fx, fy = (f[1], f[2]) if axis == 0 else (f[2], f[1])
        out.append(np.stack([f[0], fx, fy, f[3]]))
    return out[0], out[1]


def ramp(bc: F64, t: float) -> float:
    """Feed-pressure fraction at simulated time ``t``: smooth rise after ignition, smooth fall after
    shut-down (the same formula runs in the CUDA ghost-cell kernel)."""
    ti, tau, toff, tauoff, foff = bc[BC_TIGN], bc[BC_TAU], bc[BC_TOFF], bc[BC_TAUOFF], bc[BC_FOFF]
    if t < ti:
        return 0.0
    s = min((t - ti) / max(tau, 1e-12), 1.0)
    f = s * s * (3.0 - 2.0 * s)
    if t >= toff:
        s2 = min((t - toff) / max(tauoff, 1e-12), 1.0)
        f = foff * (1.0 - s2 * s2 * (3.0 - 2.0 * s2))
    return f


class NozzleFlow(ss.Supersonic2D):
    """Live Euler solution of a rocket nozzle and its exhaust plume (axisymmetric or planar)."""

    def __init__(self, nx: int = 640, ny: int = 160, gamma: float = 1.22, axisymmetric: bool = True) -> None:
        self.axi = bool(axisymmetric)
        self.bc = np.zeros(BC_SIZE)
        g = float(gamma)
        self.bc[BC_PA] = 1.0 / g
        self.bc[BC_RHOA] = 1.0
        self.bc[BC_P0] = 1.0 / g
        self.bc[BC_T0] = 1.0
        self.bc[BC_TIGN] = NEVER
        self.bc[BC_TAU] = 1.0
        self.bc[BC_TOFF] = NEVER
        self.bc[BC_TAUOFF] = 1.0
        self._inside: Optional[NDArray[np.bool_]] = None
        self._t_bc = 0.0
        self.segments = None
        self.cut_count = 0
        self.bc[BC_MC] = 0.15
        super().__init__(nx, ny, mach=0.0, gamma=g)
        self._set_cut_cells(np.ones((ny, nx)), np.ones((ny, nx + 1)), np.ones((ny + 1, nx)))

    # -- state -------------------------------------------------------------------
    @property
    def freestream(self) -> tuple[float, float, float, float]:
        return self.bc[BC_RHOA], self.bc[BC_UF], 0.0, self.bc[BC_PA]

    def reset(self) -> None:
        """Quiescent start: ambient air everywhere (at rest inside the engine), engine off."""
        self._drain()
        self._gpu_dirty = True
        self._U_stale = False
        self.bc[BC_TIGN] = NEVER
        self.bc[BC_TOFF] = NEVER
        ra, uf, _, pa = self.freestream
        c = self._cons(ra, uf, 0.0, pa)
        for k in range(4):
            self.U[k] = c[k]
        if self._inside is not None and self._inside.shape == self.solid_int.shape:
            rest = self._cons(ra, 0.0, 0.0, pa)
            for k in range(4):
                self.U[k][self._inside] = rest[k]
        rest = self._cons(ra, 0.0, 0.0, pa)
        for k in range(4):
            self.U[k][self.solid_int] = rest[k]
        self.time = 0.0
        self.steps = 0
        self._t_bc = 0.0
        self._update_primitives()

    def set_geometry(self, layout: NozzleLayout) -> None:
        """Place the engine walls (the field is kept; call :meth:`reset` for a fresh start)."""
        if layout.mask.shape != (self.ny, self.nx):
            raise ValueError(f"layout {layout.mask.shape} != grid {(self.ny, self.nx)}")
        self._drain()
        self._gpu_dirty = True
        self.solid_int = layout.mask.copy()
        self._solid_pad[:] = False
        self._solid_pad[G:-G, G:-G] = self.solid_int
        self.segments = np.asarray(layout.segments, dtype=float)
        self._build_ib(None)                  # walls are cut cells, not immersed-boundary ghost cells
        self._probe = None
        self._set_cut_cells(layout.kappa, layout.ax, layout.ay)
        self._inside = layout.inside.copy()
        self.bc[BC_NIN] = float(layout.n_inlet)
        self.bc[BC_MC] = layout.chamber_mach(self.gamma, not self.axi)
        self.ref_length = max(float(layout.r_inner(layout.x_exit)) / layout.dx, 1.0)
        self._update_primitives()

    def _set_cut_cells(self, kappa: F64, ax: F64, ay: F64) -> None:
        """Per-cell geometry of the embedded-boundary scheme: face apertures, the radial source weight κ/r,
        and the wall's projected areas and unit normal in each cut cell (the closure of the cell's faces,
        so a gas at rest stays exactly at rest)."""
        ny = self.ny
        self._ax = np.ascontiguousarray(ax, dtype=np.float64)
        self._ay = np.ascontiguousarray(ay, dtype=np.float64)
        self._kappa = np.ascontiguousarray(kappa, dtype=np.float64)
        if self.axi:
            rc = np.arange(ny, dtype=float)[:, None] + 0.5
            rm, rp, sc = (rc - 0.5) / rc, (rc + 0.5) / rc, 1.0 / rc
        else:
            rm = rp = np.ones((ny, 1))
            sc = np.zeros((ny, 1))
        self._ks = np.ascontiguousarray(kappa * sc)
        wx = ax[:, 1:] - ax[:, :-1]
        wr = ay[1:] * rp - ay[:-1] * rm - self._ks
        wx[np.abs(wx) < 1e-12] = 0.0
        wr[np.abs(wr) < 1e-12] = 0.0
        wx[self.solid_int] = 0.0
        wr[self.solid_int] = 0.0
        gx, gr = ax[:, :-1] - ax[:, 1:], ay[:-1] - ay[1:]          # direction into the wall
        norm = np.hypot(gx, gr)
        ok = norm > 1e-9
        self._wx, self._wr = np.ascontiguousarray(wx), np.ascontiguousarray(wr)
        self._nwx = np.ascontiguousarray(np.where(ok, gx / np.where(ok, norm, 1.0), 0.0))
        self._nwr = np.ascontiguousarray(np.where(ok, gr / np.where(ok, norm, 1.0), 0.0))
        self.cut_count = int(((wx != 0.0) | (wr != 0.0)).sum())

    def _cut_args(self) -> tuple:
        return self._ax, self._ay, self._wx, self._wr, self._nwx, self._nwr, self._ks

    def wall_pressure_cells(self, Wi: Optional[F64] = None) -> tuple[NDArray[np.int64], NDArray[np.int64], F64]:
        """Cut cells with a wall: ``(j, i, wall pressure p_w)``, of the published state by default."""
        Wi = self._Wpub[:, G:-G, G:-G] if Wi is None else Wi
        js, iis = np.nonzero((self._wx != 0.0) | (self._wr != 0.0))
        r, u, v, p = Wi[0, js, iis], Wi[1, js, iis], Wi[2, js, iis], Wi[3, js, iis]
        g = self.gamma
        un = u * self._nwx[js, iis] + v * self._nwr[js, iis]
        c = np.sqrt(g * p / r)
        k = 0.25 * (g + 1.0) * un
        comp = p + r * un * (k + np.sqrt(k * k + c * c))
        expn = p * np.maximum(1.0 + 0.5 * (g - 1.0) * un / c, 0.0) ** (2.0 * g / (g - 1.0))
        return js, iis, np.where(un > 0.0, comp, expn)

    # -- engine control (all times are solver time) ---------------------------------
    @property
    def ignited(self) -> bool:
        return self.bc[BC_TIGN] < NEVER and self.bc[BC_TOFF] >= NEVER

    def throttle_fraction(self, t: Optional[float] = None) -> float:
        return ramp(self.bc, self.time if t is None else t)

    def ignite(self, ramp_time: float) -> None:
        self.bc[BC_TIGN] = self.time
        self.bc[BC_TAU] = max(ramp_time, 1e-9)
        self.bc[BC_TOFF] = NEVER

    def shutdown(self, ramp_time: float) -> None:
        if self.bc[BC_TIGN] >= NEVER or self.bc[BC_TOFF] < NEVER:
            return
        self.bc[BC_FOFF] = self.throttle_fraction()
        self.bc[BC_TOFF] = self.time
        self.bc[BC_TAUOFF] = max(ramp_time, 1e-9)

    def set_conditions(self, p_amb: float, rho_amb: float, u_flight: float, p0: float, t0: float) -> None:
        """Non-dimensional ambient and chamber values; take effect from the next step (live)."""
        self.bc[BC_PA] = p_amb
        self.bc[BC_RHOA] = rho_amb
        self.bc[BC_UF] = u_flight
        self.bc[BC_P0] = p0
        self.bc[BC_T0] = t0

    # -- boundaries --------------------------------------------------------------------
    def _fill_ghosts(self, W: F64) -> None:
        g, gam, bc = G, self.gamma, self.bc
        pa, ra, uf = bc[BC_PA], bc[BC_RHOA], bc[BC_UF]
        # left: injector face (inlet rows) and far field above the engine
        r, u, v, p = W[0, g:-g, g], W[1, g:-g, g], W[2, g:-g, g], W[3, g:-g, g]
        n = r.shape[0]
        inlet = np.arange(n) < int(bc[BC_NIN])
        f = ramp(bc, self._t_bc)
        p0 = pa + f * (bc[BC_P0] - pa)
        t0 = bc[BC_T0]
        gl = self._far(r, u, v, p, -u)
        if inlet.any():
            # Non-reflecting characteristic injector (linearised, Thompson / Poinsot-Lele): the outgoing
            # acoustic wave J- = p - Z u leaves the chamber freely, the incoming one J+ = p_t + Z u_t comes from
            # the feed state (total pressure p0 and temperature T0 at the chamber Mach number BC_MC set by the
            # contraction ratio). No undamped chamber resonance; the steady state sits at p0, T0. Backflow
            # (u < 0) closes the injector.
            mc = bc[BC_MC]
            tt = t0 / (1.0 + 0.5 * (gam - 1.0) * mc * mc)
            ut = mc * math.sqrt(tt)
            pt = p0 * (tt / t0) ** (gam / (gam - 1.0))
            z = np.sqrt(gam * p * r)
            jp, jm = pt + z * ut, p - z * u
            pg = 0.5 * (jp + jm)
            ug = (jp - jm) / (2.0 * z)
            open_ = inlet & (f > 0.0) & (ug > 0.0) & (pg > 1e-7)
            closed = inlet & ~open_
            ug = np.minimum(ug, 0.98 * math.sqrt(2.0 * t0 / (gam + 1.0)))
            tg = t0 - 0.5 * (gam - 1.0) * ug * ug
            gl[0] = np.where(open_, gam * pg / tg, np.where(closed, r, gl[0]))
            gl[1] = np.where(open_, ug, np.where(closed, -u, gl[1]))
            gl[2] = np.where(open_, 0.0, np.where(closed, v, gl[2]))
            gl[3] = np.where(open_, pg, np.where(closed, p, gl[3]))
        W[:, g:-g, :g] = gl[:, :, None]
        # right: supersonic extrapolation, or the ambient pressure where the outflow is subsonic
        r, u, v, p = W[0, g:-g, -g - 1], W[1, g:-g, -g - 1], W[2, g:-g, -g - 1], W[3, g:-g, -g - 1]
        sup = u >= np.sqrt(gam * p / r)
        fr = self._far(r, u, v, p, u, outlet=True)
        gr = np.where(sup[None], np.stack([r, u, v, p]), fr)
        W[:, g:-g, -g:] = gr[:, :, None]
        # top: far field
        r, u, v, p = W[0, -g - 1, g:-g], W[1, -g - 1, g:-g], W[2, -g - 1, g:-g], W[3, -g - 1, g:-g]
        W[:, -g:, g:-g] = self._far(r, u, v, p, v)[:, None, :]
        # bottom: the axis (or the symmetry plane of a slot nozzle)
        for m in range(g):
            W[:, g - 1 - m, g:-g] = W[:, g + m, g:-g]
            W[2, g - 1 - m, g:-g] *= -1.0
        for sj in (slice(0, g), slice(-g, None)):          # corners (read by no stencil)
            for si in (slice(0, g), slice(-g, None)):
                W[0, sj, si], W[1, sj, si], W[2, sj, si], W[3, sj, si] = ra, uf, 0.0, pa

    def _far(self, r: F64, u: F64, v: F64, p: F64, un: F64, outlet: bool = False) -> F64:
        """Far-field ghost states. With a flight speed: the free stream (unless ``outlet``). In still air:
        outflow (``un`` >= 0, velocity out of the domain) leaves at the ambient static pressure; inflow is
        drawn from air at rest, i.e. at the ambient *total* pressure and temperature (entrainment), which
        cannot feed energy into the domain."""
        gam, bc = self.gamma, self.bc
        pa, ra, uf = bc[BC_PA], bc[BC_RHOA], bc[BC_UF]
        if uf > 0.0 and not outlet:
            return np.stack([np.full_like(r, ra), np.full_like(r, uf), np.zeros_like(r), np.full_like(r, pa)])
        ta = gam * pa / ra
        q2 = u * u + v * v
        q2c = np.minimum(q2, 0.9 * 2.0 * ta / (gam + 1.0))
        scale = np.sqrt(q2c / np.maximum(q2, 1e-300))
        tt = ta - 0.5 * (gam - 1.0) * q2c
        pin = pa * (tt / ta) ** (gam / (gam - 1.0))
        out = un >= 0.0
        return np.stack([np.where(out, r, gam * pin / tt), np.where(out, u, u * scale),
                         np.where(out, v, v * scale), np.where(out, pa, pin)])

    # -- stepping ------------------------------------------------------------------------
    def _step_cpu(self, n: int) -> int:
        use_nb = accel.HAVE_NUMBA
        locked = False
        if use_nb:
            locked = accel._KERNEL_LOCK.acquire(blocking=False)
            if not locked:
                return 0
        t0 = time.perf_counter()
        try:
            for _ in range(n):
                self._t_bc = self.time
                self._primitives(self.U, use_nb)
                dt = self._dt(use_nb)
                if use_nb:
                    wall, solid = self._wall_pad, self._solid_pad
                    ss._flux_x_nb(self.W, wall, self.gamma, self._F)
                    ss._flux_y_nb(self.W, wall, self.gamma, self._Gf)
                    self._U1[:, self.solid_int] = self.U[:, self.solid_int]
                    _rk_cut_nb(self.U, self.U, self._F, self._Gf, self.W, solid, *self._cut_args(), dt, 0.0,
                               self.axi, self.gamma, self._U1)
                    self._t_bc = self.time + dt
                    self._primitives(self._U1, use_nb)
                    ss._flux_x_nb(self.W, wall, self.gamma, self._F)
                    ss._flux_y_nb(self.W, wall, self.gamma, self._Gf)
                    _rk_cut_nb(self.U, self._U1, self._F, self._Gf, self.W, solid, *self._cut_args(), dt, 0.5,
                               self.axi, self.gamma, self.U)
                else:
                    self._U1[:] = self.U + dt * self._residual_axi_np()
                    self._t_bc = self.time + dt
                    self._primitives(self._U1, False)
                    self.U[:] = 0.5 * self.U + 0.5 * (self._U1 + dt * self._residual_axi_np())
                self.time += dt
                self.steps += 1
        except Exception as exc:  # noqa: BLE001
            if use_nb:
                log.warning("Nozzle kernel failed (%s); switching to the NumPy fallback.", exc)
                accel._disable_numba(exc)
            raise
        finally:
            if locked:
                accel._KERNEL_LOCK.release()
        self.last_step_ms = (time.perf_counter() - t0) * 1000.0 / max(n, 1)
        self._t_bc = self.time
        self._primitives(self.U, False)
        return n

    def _residual_axi_np(self) -> F64:
        F, Gf = _fluxes_np(self.W, self._wall_pad, self.gamma)
        ny = self.ny
        if self.axi:
            rc = np.arange(ny, dtype=float)[:, None] + 0.5
            rm, rp = (rc - 0.5) / rc, (rc + 0.5) / rc
        else:
            rm = rp = np.ones((ny, 1))
        ax, ay = self._ax, self._ay
        R = (ax[None, :, :-1] * F[:, :, :-1] - ax[None, :, 1:] * F[:, :, 1:]
             + (ay[:-1] * rm)[None] * Gf[:, :-1, :] - (ay[1:] * rp)[None] * Gf[:, 1:, :])
        Wi = self.W[:, G:-G, G:-G]
        js, iis, pw = self.wall_pressure_cells(Wi)
        R[2] += self._ks * Wi[3]
        R[1, js, iis] += pw * self._wx[js, iis]
        R[2, js, iis] += pw * self._wr[js, iis]
        R[:, self.solid_int] = 0.0
        return R

    def _begin_gpu(self, n: int) -> bool:
        try:
            if self._gpu is None or (self._gpu.nx, self._gpu.ny) != (self.nx, self.ny):
                from core.nozzle_gpu import NozzleGPU

                self._gpu = NozzleGPU(self.nx, self.ny, self.axi)
                self._gpu_dirty = True
            if self._gpu_dirty:
                self._gpu.upload(self.U, self._solid_pad, self._wall_pad, self._ib_arrays())
                self._gpu.upload_cut(*self._cut_args())
                self._gpu_dirty = False
            self._gpu.set_params(self.bc)
            self._gpu.enqueue(n, self.gamma, (0.0, 0.0, 0.0, 0.0), self.time)
        except Exception as exc:  # noqa: BLE001 - fall back to the CPU, never crash the Lab
            compute.gpu_failed(exc)
            self._gpu = None
            self._gpu_dirty = True
            return False
        self._pending_gpu = n
        self.device = "GPU"
        return True

    def temperature_ratio(self) -> F64:
        r, _, _, p = self.primitives()
        out = self.gamma * p / r
        out = out.copy()
        out[self.solid_int] = np.nan
        return out


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Scales:
    """Reference values that turn solver units into SI."""

    dx: float              # m per cell
    rho: float             # kg/m³
    a: float               # m/s
    temperature: float     # K

    @property
    def pressure(self) -> float:
        return self.rho * self.a * self.a

    @property
    def time(self) -> float:
        return self.dx / self.a


def _peaks(s: F64, min_prom: float, min_dist: int) -> list[int]:
    """Indices of local maxima of ``s`` with at least ``min_prom`` prominence on both sides."""
    out: list[int] = []
    n = len(s)
    for i in range(1, n - 1):
        if not (s[i] >= s[i - 1] and s[i] > s[i + 1]):
            continue
        left = s[max(0, i - 4 * min_dist):i].min() if i > 0 else s[i]
        right = s[i + 1:min(n, i + 1 + 4 * min_dist)].min()
        if s[i] - max(left, right) < min_prom:
            continue
        if out and i - out[-1] < min_dist:
            if s[i] > s[out[-1]]:
                out[-1] = i
            continue
        out.append(i)
    return out


def axis_features(x: F64, mach: F64, p: F64, p_amb: float, x_exit: float, r_exit: float) -> dict:
    """Shock structure along the jet axis (positions in metres downstream of the exit plane).

    * ``mach_disks``: normal shocks on the axis (Mach number falls from > 1.15 to < 0.95 within a few cells
      while the pressure jumps).
    * ``diamond_x``: shock cells, i.e. the compressions where the shock pattern crosses the axis (pressure
      peaks of at least 10 % prominence), and their median spacing.
    * ``core_length``: how far the jet stays supersonic on the axis (gaps behind Mach disks included).

    A convergent nozzle is still subsonic on the axis at its exit (the sonic line bulges downstream), so
    the search starts where the axis first turns supersonic."""
    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
    n = len(x)
    empty = {"mach_disks": [], "mach_disk_x": None, "diamond_x": [], "diamonds": 0, "diamond_spacing": None,
             "core_length": None, "supersonic_start": None}
    i0 = int(np.searchsorted(x, x_exit))
    if i0 >= n - 4:
        return empty
    look = int(max(6.0 * r_exit / dx, 8))
    sup = np.nonzero(mach[i0:min(n, i0 + look)] > 1.05)[0]
    if sup.size == 0:
        return empty
    s0 = i0 + int(sup[0])
    # supersonic extent: allow subsonic gaps (behind Mach disks) up to two exit diameters long
    gap = int(max(4.0 * r_exit / dx, 4))
    last = s0
    for i in range(s0, n):
        if mach[i] > 1.0:
            last = i
        elif i - last > gap:
            break
    disks: list[int] = []
    i = s0 + 1
    step = int(max(0.6 * r_exit / dx, 4))
    while i < min(last + 2, n - 3):
        if mach[i - 1] > 1.15 and float(mach[i:i + 3].min()) < 0.95 and p[min(i + 2, n - 1)] > 1.5 * p[i - 1]:
            disks.append(i)
            i += step
        else:
            i += 1
    end = min(n - 1, last + 2)
    peaks: list[int] = []
    spacing = None
    if end - s0 > 16:
        # Shock-cell length = dominant wavelength of the axial pressure pattern (robust to small ripples);
        # the cells are then the strongest compressions at least ~0.6 of that length apart.
        seg = np.log(np.maximum(p[s0:end], 1e-300) / max(p_amb, 1e-300))
        sm = np.convolve(seg, np.ones(3) / 3.0, mode="same")
        y = sm - np.polyval(np.polyfit(np.arange(sm.size), sm, 1), np.arange(sm.size))
        npad = 1 << int(math.ceil(math.log2(max(sm.size * 8, 256))))
        spec = np.abs(np.fft.rfft(y * np.hanning(sm.size), npad))
        freq = np.fft.rfftfreq(npad, d=dx)
        lam_lo, lam_hi = 1.0 * r_exit, max((end - s0) * dx / 1.2, 1.2 * r_exit)
        ok = (freq > 1.0 / lam_hi) & (freq < 1.0 / lam_lo)
        if ok.any() and float(np.std(y)) > 0.03:
            k = int(np.argmax(np.where(ok, spec, 0.0)))
            spacing = 1.0 / float(freq[k])
            peaks = [s0 + q for q in _peaks(sm, 0.05, int(max(0.6 * spacing / dx, 3)))]
    out = dict(empty)
    out["supersonic_start"] = (s0 - i0) * dx
    out["core_length"] = float(x[last] - x_exit)
    out["mach_disks"] = [float(x[k] - x_exit) for k in disks]
    out["mach_disk_x"] = out["mach_disks"][0] if disks else None
    out["diamond_x"] = [float(x[k] - x_exit) for k in peaks]
    out["diamonds"] = len(peaks)
    out["diamond_spacing"] = spacing
    return out


def measure(solver: NozzleFlow, lay: NozzleLayout, sc: Scales, geom: NozzleGeometry, features: bool = True) -> dict:
    """Simulated performance and flow features, in SI units.

    Thrust is the axial momentum and pressure flux through the exit plane,
    F = ∫ (ρu² + p − p_a) dA, which equals ṁ v_e + (p_e − p_a) A_e in steady flow."""
    r, u, v, p = solver.primitives()
    g = solver.gamma
    solid = solver.solid_int
    ny = solver.ny
    j = np.arange(ny, dtype=float)
    lo, hi = j * sc.dx, (j + 1.0) * sc.dx                         # radial extent of each row
    pa = solver.bc[BC_PA]
    out: dict = {"time_s": solver.time * sc.time, "steps": solver.steps,
                 "throttle": solver.throttle_fraction(), "ignited": solver.ignited}

    def ring(rmax: float) -> F64:
        """Area of each row's part of the disc (slot) of radius ``rmax``."""
        top = np.clip(rmax, lo, hi)
        if solver.axi:
            return math.pi * (top * top - lo * lo)
        return 2.0 * (top - lo)                                     # both halves of the slot, per metre

    def plane(ri: F64, ui: F64, pi: F64, area: F64) -> dict:
        mflux = ri * ui * area
        return {"mdot": float(mflux.sum()) * sc.rho * sc.a,
                "momentum": float(((ri * ui * ui + pi - pa) * area).sum()) * sc.pressure,
                "p_avg": float((pi * area).sum() / max(area.sum(), 1e-30)) * sc.pressure, "mflux": mflux}

    # exit plane: the cells of the last nozzle column, weighted by the exact exit area of each row
    ie = lay.i_exit
    re = float(lay.r_inner(lay.x_exit))
    area_e = ring(re)
    rows = np.nonzero((area_e > 0.0) & ~solid[:, ie])[0]
    ex = plane(r[rows, ie], u[rows, ie], p[rows, ie], area_e[rows])
    # throat: the x face nearest the throat, weighted by its open fraction (states averaged across it)
    f = int(min(max(round(lay.x_throat / sc.dx), 1), solver.nx - 1))
    if solver.axi:
        full = math.pi * (hi * hi - lo * lo)
    else:
        full = 2.0 * (hi - lo)
    area_t = full * solver._ax[:, f]
    tr = np.nonzero((area_t > 0.0) & (lo < float(lay.r_inner(f * sc.dx))))[0]        # inside the wall only
    th = plane(0.5 * (r[tr, f - 1] + r[tr, f]), 0.5 * (u[tr, f - 1] + u[tr, f]),
               0.5 * (p[tr, f - 1] + p[tr, f]), area_t[tr])
    out["thrust_exit_plane"] = ex["momentum"]
    # Thrust = the force of the gas on the engine's internal surfaces: wall pressure (relative to ambient) on
    # every cut-cell wall face of the chamber and nozzle, plus the reaction at the injector face. Unlike the
    # exit-plane momentum flux it is a real force during start-up too; in steady flow the two agree. The
    # pressure on the engine's outer surfaces (and a vehicle base) is reported separately as the base force.
    if solver.axi:
        a_row = 2.0 * math.pi * (j + 0.5) * sc.dx * sc.dx
    else:
        a_row = np.full(ny, 2.0 * sc.dx)
    js, iis, pw = solver.wall_pressure_cells()
    fw = (pw - pa) * solver._wx[js, iis] * a_row[js]
    xw, rw = (iis + 0.5) * sc.dx, (js + 0.5) * sc.dx
    inner = (xw < lay.x_exit + sc.dx) & (rw < lay.r_inner(xw) + 0.5 * lay.wall_thickness)
    nin = max(int(solver.bc[BC_NIN]), 1)
    ri, ui, pi = r[:nin, 0], u[:nin, 0], p[:nin, 0]
    f_inj = float(((pi + ri * ui * ui - pa) * solver._ax[:nin, 0] * a_row[:nin]).sum())
    out["thrust"] = (float(fw[inner].sum()) + f_inj) * sc.pressure
    out["base_force"] = float(fw[~inner].sum()) * sc.pressure
    out["thrust_installed"] = out["thrust"] + out["base_force"]
    out["mdot_exit"] = ex["mdot"]
    out["mdot_throat"] = th["mdot"]
    out["exit_pressure"] = ex["p_avg"]
    mf = ex["mflux"]
    mtot = float(mf.sum())
    c_e = np.sqrt(g * p[rows, ie] / r[rows, ie])
    if mtot > 1e-12:
        w = mf / mtot
        out["exit_velocity"] = float((w * u[rows, ie]).sum()) * sc.a
        out["exit_mach"] = float((w * np.hypot(u[rows, ie], v[rows, ie]) / c_e).sum())
        out["exit_temperature"] = float((w * g * p[rows, ie] / r[rows, ie]).sum()) * sc.temperature
    else:
        out["exit_velocity"] = out["exit_mach"] = 0.0
        out["exit_temperature"] = float(np.mean(g * p[rows, ie] / r[rows, ie])) * sc.temperature
    md = out["mdot_exit"]
    out["isp"] = out["thrust"] / (md * G0) if md > 1e-9 else 0.0
    # chamber: static and total pressure at the injector face, averaged over the chamber rows
    nin = max(int(solver.bc[BC_NIN]), 1)
    rch, uch, pch = r[:nin, 0], u[:nin, 0], p[:nin, 0]
    mch = np.abs(uch) / np.sqrt(g * pch / rch)
    p0ch = pch * (1.0 + 0.5 * (g - 1.0) * mch * mch) ** (g / (g - 1.0))
    out["chamber_pressure"] = float(np.mean(p0ch)) * sc.pressure
    at = (2.0 * geom.rt) if not solver.axi else math.pi * geom.rt ** 2
    out["cf"] = out["thrust"] / max(out["chamber_pressure"] * at, 1e-9)
    out["cstar"] = out["chamber_pressure"] * at / md if md > 1e-9 else 0.0
    # exit-plane profile
    out["exit_r"] = (rows + 0.5) * sc.dx
    out["exit_profile"] = {"mach": np.hypot(u[rows, ie], v[rows, ie]) / c_e, "u": u[rows, ie] * sc.a,
                           "p": p[rows, ie] * sc.pressure, "T": g * p[rows, ie] / r[rows, ie] * sc.temperature}
    # centre line (axis row)
    ra, ua, pa_line = r[0], u[0], p[0]
    ma = np.hypot(ua, v[0]) / np.sqrt(g * pa_line / ra)
    xs = (np.arange(solver.nx) + 0.5) * sc.dx
    out["axis_x"] = xs
    out["axis_mach"] = ma
    out["axis_p"] = pa_line * sc.pressure
    out["axis_T"] = g * pa_line / ra * sc.temperature
    out["axis_u"] = ua * sc.a
    out["p_amb"] = pa * sc.pressure
    if features:
        out.update(axis_features(xs, ma, pa_line * sc.pressure, pa * sc.pressure, lay.x_exit,
                                 float(lay.r_inner(lay.x_exit))))
    # pressure on the inner wall (cut cells whose wall lies above them, inside the engine)
    js, iis, pw = solver.wall_pressure_cells()
    xw, rw = (iis + 0.5) * sc.dx, (js + 0.5) * sc.dx
    inner = ((xw < lay.x_exit) & (rw < lay.r_inner(xw) + 0.5 * lay.wall_thickness) & (solver._nwr[js, iis] > 0.2)
             & (solver._kappa[js, iis] > 0.3))                    # sliver cells carry little information
    xi, pi_ = xw[inner], pw[inner] * sc.pressure
    if xi.size:
        k = np.round(xi / sc.dx).astype(np.int64)                  # one sample per column: the median
        cols, inv = np.unique(k, return_inverse=True)
        med = np.array([np.median(pi_[inv == c]) for c in range(cols.size)])
        out["wall_x"], out["wall_p"] = cols * sc.dx, med
    else:
        out["wall_x"], out["wall_p"] = xi, pi_
    return out


def warmup() -> None:
    """Compile the nozzle kernel (or load it from the numba cache)."""
    if not accel.HAVE_NUMBA:
        return
    s = NozzleFlow(24, 12, 1.4)
    accel.precompile(_rk_cut_nb, s.U, s.U, s._F, s._Gf, s.W, s._solid_pad, *s._cut_args(), 0.1, 0.5, True, 1.4,
                     s._U1)
