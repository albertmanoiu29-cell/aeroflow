"""Project files (``.aeroflow``): one file that reopens a whole session.

A project is a zip archive with

* ``project.json`` - format version, app version, geometry metadata, flow conditions, grid quality,
  solver engine, reference-area mode, camera and the last polar (arrays as lists);
* ``geometry.stl`` - the exact (possibly edited) geometry as a binary STL.

Results are not stored: the Instant engine recomputes them in about a second, and OpenFOAM results stay
in their case folders (the path is remembered).
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config.settings import APP_VERSION, FlowConditions
from core.geometry_manager import GeometryModel, load_geometry_file
from core.results import PolarData

FORMAT = 1
SUFFIX = ".aeroflow"


class ProjectError(RuntimeError):
    """The file is not a readable AeroFlow project."""


@dataclass
class ProjectState:
    conditions: FlowConditions = field(default_factory=FlowConditions)
    quality: str = "Medium"
    engine: str = "instant"
    reference_mode: str = ""
    camera: Optional[list[list[float]]] = None
    polar: Optional[PolarData] = None
    openfoam_case: str = ""
    notes: str = ""
    workspaces: dict = field(default_factory=dict)   # labs and tools: {"nozzle": ..., "tools": {...}}


def _polar_to_json(p: PolarData) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(p):
        v = getattr(p, f.name)
        out[f.name] = v.tolist() if isinstance(v, np.ndarray) else (list(v) if isinstance(v, tuple) else v)
    return out


def _polar_from_json(d: dict[str, Any]) -> PolarData:
    names = {f.name for f in fields(PolarData)}
    kw = {k: v for k, v in d.items() if k in names}
    for k in ("alpha_deg", "cl", "cd", "cm"):
        kw[k] = np.asarray(kw.get(k, []), dtype=float)
    if "moment_ref" in kw:
        kw["moment_ref"] = tuple(kw["moment_ref"])
    return PolarData(**kw)


def save_project(path: str | Path, model: GeometryModel, state: ProjectState) -> Path:
    """Write ``model`` and ``state`` to ``path`` (``.aeroflow`` is appended if missing)."""
    out = Path(path)
    if out.suffix.lower() != SUFFIX:
        out = out.with_suffix(SUFFIX)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".aeroflow.tmp")
    stl = tmp.with_suffix(".stl")
    try:
        model.export_stl(stl, binary=True)
        doc = {
            "format": FORMAT, "app_version": APP_VERSION,
            "geometry": {"name": model.name, "history": list(model.history),
                         "reference_mode": state.reference_mode or getattr(model.reference_mode, "value", "")},
            "conditions": asdict(state.conditions), "quality": state.quality, "engine": state.engine,
            "camera": state.camera, "openfoam_case": state.openfoam_case, "notes": state.notes,
            "polar": _polar_to_json(state.polar) if state.polar is not None else None,
            "workspaces": state.workspaces or {},
        }
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("project.json", json.dumps(doc, indent=2))
            zf.write(stl, "geometry.stl")
        tmp.replace(out)                         # atomic: a crash never leaves a half-written project
    finally:
        stl.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
    return out


def load_project(path: str | Path, scratch_dir: Path) -> tuple[GeometryModel, ProjectState]:
    """Read a project. The geometry is extracted to ``scratch_dir`` and loaded with the normal importer."""
    p = Path(path)
    try:
        with zipfile.ZipFile(p) as zf:
            doc = json.loads(zf.read("project.json").decode("utf-8"))
            stl_bytes = zf.read("geometry.stl")
    except (OSError, KeyError, zipfile.BadZipFile, ValueError) as exc:
        raise ProjectError(f"{p.name} is not a readable AeroFlow project: {exc}") from exc
    if int(doc.get("format", 0)) > FORMAT:
        raise ProjectError(f"{p.name} was saved by a newer AeroFlow Studio ({doc.get('app_version')}); "
                           "please update the app.")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    stl = scratch_dir / "project_geometry.stl"
    stl.write_bytes(stl_bytes)
    model = load_geometry_file(stl).model
    geo = doc.get("geometry") or {}
    model = model.renamed(str(geo.get("name") or p.stem))
    cond_fields = {f.name for f in fields(FlowConditions)}
    cond = FlowConditions(**{k: float(v) for k, v in (doc.get("conditions") or {}).items() if k in cond_fields})
    polar = _polar_from_json(doc["polar"]) if doc.get("polar") else None
    state = ProjectState(conditions=cond, quality=str(doc.get("quality", "Medium")), engine=str(doc.get("engine", "instant")),
                         reference_mode=str(geo.get("reference_mode", "")), camera=doc.get("camera"), polar=polar,
                         openfoam_case=str(doc.get("openfoam_case", "")), notes=str(doc.get("notes", "")),
                         workspaces=dict(doc.get("workspaces") or {}))
    return model, state


def peek(path: str | Path) -> dict[str, Any]:
    """The project's JSON (for previews) without loading the geometry."""
    with zipfile.ZipFile(Path(path)) as zf:
        return json.loads(io.TextIOWrapper(zf.open("project.json"), encoding="utf-8").read())
