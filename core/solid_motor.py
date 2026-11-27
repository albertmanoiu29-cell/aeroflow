"""Solid rocket motor internal ballistics: BATES grain burn-back with a Saint-Robert (de Vieille) burn-rate law.

The grain is ``segments`` hollow cylinders (outer diameter D, core d, length L), burning on the core and,
unless inhibited, on both ends. At each web distance w the burning area A_b(w) fixes the Klemmung
Kn = A_b / A_t, and the quasi-steady chamber pressure balances the gas made with the gas leaving through
the choked throat:

    ρ_p a p^n A_b = p A_t / c*    =>    p = (Kn a ρ_p c*)^(1 / (1 - n))

The thrust at each instant comes from the nozzle theory of :mod:`core.nozzle` at that chamber pressure
(over- / under-expansion and separation included). Erosive burning, throat erosion and the ignition and
tail-off transients are neglected: the curve is the classic quasi-steady estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from core import nozzle as nz

G0 = 9.80665


@dataclass(frozen=True)
class SolidPropellant:
    name: str
    a_mm: float            # burn rate coefficient, mm/s at 1 MPa (r = a · p[MPa]^n)
    n: float               # pressure exponent
    density: float         # kg/m³ (as cast)
    gamma: float = 1.2     # exhaust (frozen, effective)
    molar_mass: float = 25.0
    chamber_temperature: float = 3000.0
    note: str = ""

    @property
    def gas(self) -> nz.Gas:
        return nz.Gas(self.gamma, self.molar_mass, self.chamber_temperature)

    def burn_rate(self, p_pa: float) -> float:
        """Burn rate (m/s) at chamber pressure ``p_pa``."""
        return self.a_mm * 1e-3 * (max(p_pa, 1.0) / 1e6) ** self.n


#: Typical values (amateur and professional literature; real batches vary by ±20 %).
SOLID_PROPELLANTS: dict[str, SolidPropellant] = {p.name: p for p in (
    SolidPropellant("APCP (AP / Al / HTPB composite)", 3.5, 0.35, 1770.0, 1.20, 25.0, 3300.0,
                    "professional / high-power composite"),
    SolidPropellant("KNSB (KNO₃ / sorbitol)", 8.26, 0.319, 1750.0, 1.137, 39.9, 1600.0,
                    "'rocket candy' (Nakka, 0.8-2.6 MPa)"),
    SolidPropellant("KNDX (KNO₃ / dextrose)", 8.88, 0.619, 1785.0, 1.131, 42.4, 1710.0,
                    "'rocket candy' (Nakka, 0.8-2.6 MPa)"),
    SolidPropellant("Double-base (NC / NG)", 7.0, 0.30, 1600.0, 1.22, 24.0, 2800.0, "smokeless, tactical motors"),
)}


@dataclass(frozen=True)
class BatesGrain:
    outer_diameter: float          # m
    core_diameter: float           # m
    segment_length: float          # m
    segments: int = 4
    inhibit_ends: bool = False     # True: only the core burns (a progressive grain)

    @property
    def web(self) -> float:
        """Burnable web: to the case wall, or half the segment length if the ends burn and reach first."""
        w = 0.5 * (self.outer_diameter - self.core_diameter)
        return w if self.inhibit_ends else min(w, 0.5 * self.segment_length)

    def burning_area(self, w: float) -> float:
        """Total burning surface (m²) after a regression of ``w``."""
        d = self.core_diameter + 2.0 * w
        length = self.segment_length - (0.0 if self.inhibit_ends else 2.0 * w)
        if d >= self.outer_diameter or length <= 0.0:
            return 0.0
        a = math.pi * d * length
        if not self.inhibit_ends:
            a += 2.0 * 0.25 * math.pi * (self.outer_diameter ** 2 - d * d)
        return self.segments * a

    def volume(self, w: float = 0.0) -> float:
        """Remaining propellant volume (m³)."""
        d = self.core_diameter + 2.0 * w
        length = self.segment_length - (0.0 if self.inhibit_ends else 2.0 * w)
        if d >= self.outer_diameter or length <= 0.0:
            return 0.0
        return self.segments * 0.25 * math.pi * (self.outer_diameter ** 2 - d * d) * length

    @property
    def port_area(self) -> float:
        return 0.25 * math.pi * self.core_diameter ** 2


@dataclass
class MotorResult:
    t: np.ndarray                  # s
    pc: np.ndarray                 # Pa
    thrust: np.ndarray             # N
    kn: np.ndarray
    mdot: np.ndarray               # kg/s
    web: np.ndarray                # m
    total_impulse: float           # N s
    burn_time: float               # s
    average_thrust: float          # N
    peak_thrust: float
    peak_pressure: float
    propellant_mass: float         # kg
    isp: float                     # delivered, s
    designation: str               # e.g. "K1050"
    port_to_throat: float
    volumetric_loading: float
    warnings: list[str] = field(default_factory=list)


def motor_class(total_impulse: float) -> str:
    """NAR / TRA impulse class: A up to 2.5 N s, each further letter doubles."""
    if total_impulse <= 0.0:
        return "-"
    if total_impulse <= 0.3125:
        return "1/4A"
    if total_impulse <= 0.625:
        return "1/2A"
    k = max(0, math.ceil(math.log2(total_impulse / 2.5)))
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return letters[k] if k < len(letters) else "Z+"


def equilibrium_pressure(kn: float, prop: SolidPropellant, cstar: float) -> float:
    """Chamber pressure (Pa) at Klemmung ``kn``."""
    if kn <= 0.0:
        return 0.0
    a_si = prop.a_mm * 1e-3 / (1e6 ** prop.n)          # m/s per Pa^n
    return (kn * a_si * prop.density * cstar) ** (1.0 / (1.0 - prop.n))


def kn_for_pressure(pc: float, prop: SolidPropellant, cstar: float) -> float:
    a_si = prop.a_mm * 1e-3 / (1e6 ** prop.n)
    return pc ** (1.0 - prop.n) / (a_si * prop.density * cstar)


def simulate(grain: BatesGrain, prop: SolidPropellant, geom: nz.NozzleGeometry, gas: Optional[nz.Gas] = None,
             pa: float = 101325.0, cstar_efficiency: float = 0.95, steps: int = 240) -> MotorResult:
    """Burn the grain through the nozzle ``geom`` into ambient pressure ``pa`` (exhaust: the propellant's own gas
    unless ``gas`` is given)."""
    gas = gas or prop.gas
    at = math.pi * geom.rt ** 2
    cstar = nz.characteristic_velocity(gas) * cstar_efficiency
    web = grain.web
    ws = np.linspace(0.0, web, steps + 1)
    t = [0.0]
    pcs, fs, kns, mds = [], [], [], []
    for k, w in enumerate(ws):
        kn = grain.burning_area(w) / at
        pc = equilibrium_pressure(kn, prop, cstar)
        pcs.append(pc)
        kns.append(kn)
        mds.append(pc * at / cstar)
        if pc > 1.2 * pa:
            th = nz.nozzle_theory(geom, gas, pc, pa)
            fs.append(max(th.thrust, 0.0))                 # C_F p_c A_t: C_F does not depend on c*
        else:
            fs.append(0.0)
        if k:
            r_mid = prop.burn_rate(0.5 * (pcs[-1] + pcs[-2]))
            t.append(t[-1] + (ws[k] - ws[k - 1]) / max(r_mid, 1e-9))
    t_a, pc_a, f_a = np.asarray(t), np.asarray(pcs), np.asarray(fs)
    trap = getattr(np, "trapezoid", None) or np.trapz
    imp = float(trap(f_a, t_a)) if len(t_a) > 1 else 0.0
    m_prop = grain.volume(0.0) * prop.density
    burn = float(t_a[-1])
    avg = imp / burn if burn > 0 else 0.0
    cls = motor_class(imp)
    warn = []
    ptt = grain.port_area / at
    if ptt < 2.0:
        warn.append(f"Port-to-throat area ratio {ptt:.2f} (< 2): expect erosive burning and a pressure spike at "
                    "start-up.")
    if pc_a.max() > 10e6:
        warn.append(f"Peak pressure {pc_a.max() / 1e6:.1f} MPa: beyond most amateur motor cases.")
    if pc_a.max() < 1.5 * pa:
        warn.append("The pressure never rises much above ambient: the throat is too large for this grain "
                    "(raise Kn: more or longer segments, a smaller throat).")
    if prop.n >= 0.6:
        warn.append(f"Pressure exponent n = {prop.n:.2f}: pressure reacts strongly to Kn changes (less stable).")
    case_vol = 0.25 * math.pi * grain.outer_diameter ** 2 * grain.segment_length * grain.segments
    return MotorResult(t_a, pc_a, f_a, np.asarray(kns), np.asarray(mds), ws, imp, burn, avg, float(f_a.max()),
                       float(pc_a.max()), m_prop, imp / (m_prop * G0) if m_prop > 0 else 0.0,
                       f"{cls}{avg:.0f}", ptt, grain.volume(0.0) / case_vol if case_vol > 0 else 0.0, warn)


def segments_for_pressure(grain: BatesGrain, prop: SolidPropellant, geom: nz.NozzleGeometry,
                          gas: Optional[nz.Gas], pc: float, cstar_efficiency: float = 0.95) -> int:
    """Segments needed for an initial chamber pressure of about ``pc``."""
    gas = gas or prop.gas
    kn = kn_for_pressure(pc, prop, nz.characteristic_velocity(gas) * cstar_efficiency)
    one = BatesGrain(grain.outer_diameter, grain.core_diameter, grain.segment_length, 1, grain.inhibit_ends)
    a1 = one.burning_area(0.0)
    return max(1, round(kn * math.pi * geom.rt ** 2 / max(a1, 1e-12)))


def write_eng(result: MotorResult, grain: BatesGrain, path: str | Path, name: str = "AeroFlow",
              case_mass: Optional[float] = None, delays: str = "0") -> Path:
    """The thrust curve as a RASP ``.eng`` file (OpenRocket, RockSim and most simulators read it)."""
    path = Path(path)
    m_prop = result.propellant_mass
    case = case_mass if case_mass is not None else 0.6 * m_prop
    dia = grain.outer_diameter * 1e3 + 6.0                                   # the case around the grain
    length = grain.segment_length * grain.segments * 1e3 + 0.5 * dia
    code = result.designation.replace("/", "_")
    lines = [f"; {name} - {result.designation} from AeroFlow Studio (BATES, {grain.segments} segments)",
             f"{code} {dia:.0f} {length:.0f} {delays} {m_prop:.4f} {m_prop + case:.4f} AeroFlow"]
    t, f = result.t, result.thrust
    k = np.linspace(0, len(t) - 1, min(len(t), 30)).astype(int)            # RASP readers expect a few dozen points
    for tt, ff in zip(t[k], f[k]):
        if tt > 0.0:
            lines.append(f"   {tt:.4f} {max(ff, 0.0):.2f}")
    lines.append(f"   {t[-1] + 0.02 * max(t[-1], 0.05):.4f} 0.00")              # the tail-off to zero thrust
    lines.append(";")
    path.write_text("\n".join(lines) + "\n", encoding="ascii", errors="replace")
    return path


__all__ = ["write_eng", "SolidPropellant", "SOLID_PROPELLANTS", "BatesGrain", "MotorResult", "motor_class",
           "equilibrium_pressure", "kn_for_pressure", "simulate", "segments_for_pressure"]
