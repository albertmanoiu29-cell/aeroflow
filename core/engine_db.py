"""The engine library: Nozzle Lab engines saved by name (JSON in the user data folder), with their
performance at any ambient pressure from :func:`core.nozzle.nozzle_theory`."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Optional

from core import nozzle as nz


@dataclass
class EngineRecord:
    name: str
    geom: dict
    gamma: float
    molar_mass: float
    chamber_temperature: float
    pc: float                      # Pa
    propellant: str = ""
    notes: str = ""

    def geometry(self) -> nz.NozzleGeometry:
        g = dict(self.geom)
        g["contour"] = nz.Contour[g["contour"]] if g.get("contour") in nz.Contour.__members__ else nz.Contour.BELL
        for k in ("custom_x", "custom_r"):
            g[k] = tuple(g.get(k, ()))
        known = {f.name for f in fields(nz.NozzleGeometry)}
        return replace(nz.NozzleGeometry(), **{k: v for k, v in g.items() if k in known})

    def gas(self) -> nz.Gas:
        return nz.Gas(self.gamma, self.molar_mass, self.chamber_temperature)

    def performance(self, pa: float) -> tuple[float, float, float]:
        """(thrust N, mass flow kg/s, Isp s) at ambient pressure ``pa``."""
        th = nz.nozzle_theory(self.geometry(), self.gas(), self.pc, max(pa, 1e-3))
        return max(th.thrust, 0.0), th.mass_flow, th.isp

    def summary(self) -> str:
        f0, md, isp0 = self.performance(101325.0)
        fv, _, ispv = self.performance(1e-3)
        return (f"{self.name}: {f0 / 1e3:,.1f} kN SL / {fv / 1e3:,.1f} kN vac, Isp {isp0:.0f} / {ispv:.0f} s, "
                f"{md:,.1f} kg/s, p_c {self.pc / 1e5:.0f} bar")


def record_from(name: str, geom: nz.NozzleGeometry, gas: nz.Gas, pc: float, propellant: str = "",
                notes: str = "") -> EngineRecord:
    g = asdict(geom)
    g["contour"] = geom.contour.name
    g["custom_x"] = list(geom.custom_x)
    g["custom_r"] = list(geom.custom_r)
    return EngineRecord(name, g, gas.gamma, gas.molar_mass, gas.chamber_temperature, pc, propellant, notes)


def library_path() -> Path:
    from config.settings import get_paths

    return get_paths().root / "engines.json"


def load(path: Optional[Path] = None) -> list[EngineRecord]:
    p = path or library_path()
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for d in data.get("engines", []):
        try:
            out.append(EngineRecord(**d))
        except TypeError:
            continue
    return out


def save(engines: list[EngineRecord], path: Optional[Path] = None) -> Path:
    p = path or library_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"format": "aeroflow-engines", "version": 1, "engines": [asdict(e) for e in engines]},
                            indent=2), encoding="utf-8")
    return p


def add(rec: EngineRecord, path: Optional[Path] = None) -> list[EngineRecord]:
    """Add (or replace, by name) an engine and save the library."""
    lib = [e for e in load(path) if e.name != rec.name]
    lib.append(rec)
    save(lib, path)
    return lib


def remove(name: str, path: Optional[Path] = None) -> list[EngineRecord]:
    lib = [e for e in load(path) if e.name != name]
    save(lib, path)
    return lib


def builtin() -> list[EngineRecord]:
    """The Nozzle Lab presets as engine records (rocket engines only)."""
    from ui.nozzle_window import PRESETS

    out = []
    for name, p in PRESETS.items():
        prop = nz.PROPELLANTS.get(p["prop"])
        if prop is None or "Cold gas" in p["prop"]:
            continue
        g = replace(nz.NozzleGeometry(), **p["geom"])
        out.append(record_from(name, g, nz.Gas.of(prop), p["pc"] * 1e5, p["prop"], "built-in preset"))
    return out


__all__ = ["EngineRecord", "record_from", "library_path", "load", "save", "add", "remove", "builtin"]
