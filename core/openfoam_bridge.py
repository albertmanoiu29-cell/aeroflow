"""Engine B - automated OpenFOAM wrapper.

Responsibilities:

* **Detection** of an OpenFOAM installation: native (on ``PATH`` or under a standard install
  prefix), Windows WSL2, or Docker (official ``openfoam/*`` images). The result is a
  :class:`OpenFOAMEnvironment` that never raises. Use ``mode == UNAVAILABLE`` to show the
  "Not Detected" badge.
* **Case generation**: a complete, valid case (``0/``, ``constant/``, ``system/``) sized
  automatically from the geometry. It covers blockMesh, surface features, snappyHexMesh with
  boundary layers, k-omega SST RANS, and a forceCoeffs function object. The dictionaries are
  written so they work with both OpenFOAM Foundation (v8-v12: ``surfaceFeatures``,
  ``momentumTransport``, ``physicalProperties``, ``foamRun -solver incompressibleFluid``) and
  ESI/openfoam.com (v2xxx: ``surfaceFeatureExtract``, ``turbulenceProperties``,
  ``transportProperties``, ``simpleFoam``).
* **Execution** of blockMesh -> surface features -> snappyHexMesh -overwrite -> simpleFoam ->
  foamToVTK as subprocesses. Output is parsed line by line for residuals, iteration numbers and
  force coefficients, and reported through plain callbacks, so this module has no Qt dependency.
  :meth:`OpenFOAMRunner.abort` terminates the whole process tree (including WSL and Docker
  children), so no orphaned solvers remain.
* **Loading** the foamToVTK output into PyVista and converting it to a
  :class:`~core.results.SolverResult`. Reading files is safe in a worker thread; rendering
  happens later on the GUI thread.
"""

from __future__ import annotations

import logging
import math
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Final, Optional, Sequence

import numpy as np

from config.settings import (
    OPENFOAM, OPENFOAM_MESH_PRESETS, ExecutionMode, FlowConditions, GridQuality, get_paths,
)
from core.geometry_manager import GeometryInfo, GeometryModel
from core.results import AeroKPIs, FlowField, SolverResult, SurfaceField
from utils.validators import reynolds_number, sanitize_filename, validate_flow_conditions

log = logging.getLogger("aeroflow.core.openfoam")

LogFn = Callable[[str], None]
ProgressFn = Callable[[float, str], None]
ResidualFn = Callable[[int, dict[str, float]], None]
CoeffFn = Callable[[int, dict[str, float]], None]

IS_WINDOWS: Final[bool] = platform.system() == "Windows"
CREATE_NO_WINDOW: Final[int] = 0x08000000 if IS_WINDOWS else 0
CREATE_NEW_PROCESS_GROUP: Final[int] = 0x00000200 if IS_WINDOWS else 0

_RE_TIME = re.compile(r"^\s*Time\s*=\s*([0-9.eE+-]+)\s*[a-zA-Z]*\s*$")   # OpenFOAM 12+ appends a unit: "1s"
_RE_RESIDUAL = re.compile(r"Solving for\s+(\w+),\s*Initial residual\s*=\s*([0-9.eE+-]+)")
_RE_COEFF = re.compile(r"\b(Cd|Cl|Cm|CmPitch|Cs)\s*[:=]\s*([-+0-9.eE]+)")
_RE_SNAPPY_PHASE = re.compile(r"^(Castellated mesh|Snapping|Layer addition|Refinement phase|Morph iteration)")

# Shell fragment that finds and sources an OpenFOAM environment when one is not already on PATH.
_SOURCE_SNIPPET: Final[str] = (
    'AF_B=""; '
    'for f in /opt/openfoam*/etc/bashrc /usr/lib/openfoam/openfoam*/etc/bashrc '
    '"$HOME"/OpenFOAM/OpenFOAM-*/etc/bashrc /opt/OpenFOAM-*/etc/bashrc; do '
    '[ -f "$f" ] && AF_B="$f"; done; '
    'if ! command -v simpleFoam >/dev/null 2>&1 && ! command -v foamRun >/dev/null 2>&1; then '
    '[ -n "$AF_B" ] && . "$AF_B" >/dev/null 2>&1; fi; '
)
_PROBE_SCRIPT: Final[str] = _SOURCE_SNIPPET + (
    'echo "AF_BASHRC=$AF_B"; echo "AF_PROJECT=$WM_PROJECT"; echo "AF_VERSION=$WM_PROJECT_VERSION"; '
    'for c in simpleFoam foamRun blockMesh snappyHexMesh surfaceFeatures surfaceFeatureExtract '
    'foamToVTK decomposePar reconstructPar mpirun; do '
    'command -v "$c" >/dev/null 2>&1 && echo "AF_HAS=$c"; done; true'
)


class OpenFOAMError(RuntimeError):
    """Raised when a case cannot be generated or a stage fails."""


class OpenFOAMAborted(RuntimeError):
    """Raised when the user aborts a run."""


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OpenFOAMEnvironment:
    """Result of OpenFOAM detection."""

    mode: ExecutionMode = ExecutionMode.UNAVAILABLE
    version: str = ""
    project: str = ""
    bashrc: str = ""
    docker_image: str = ""
    wsl_distro: str = ""
    executables: tuple[str, ...] = ()
    detail: str = "OpenFOAM was not found."
    docker_available: bool = False
    wsl_available: bool = False

    @property
    def available(self) -> bool:
        """True when cases can be run."""
        return self.mode != ExecutionMode.UNAVAILABLE

    @property
    def is_esi(self) -> bool:
        """ESI/openfoam.com versions are named ``v2312`` etc."""
        return self.version.lower().startswith("v") or "com" in self.project.lower()

    @property
    def uses_foam_run(self) -> bool:
        """Foundation v11+ replaced simpleFoam with ``foamRun -solver incompressibleFluid``."""
        return "foamRun" in self.executables and "simpleFoam" not in self.executables

    @property
    def feature_command(self) -> str:
        """Feature-extraction utility name for this flavour."""
        if "surfaceFeatures" in self.executables:
            return "surfaceFeatures"
        return "surfaceFeatureExtract"

    @property
    def badge_text(self) -> str:
        """Short human-readable status for the top-bar pill."""
        if not self.available:
            return "OpenFOAM: Not Detected — Using Instant Aero Engine"
        where = {ExecutionMode.NATIVE: "native", ExecutionMode.WSL: "WSL2",
                 ExecutionMode.DOCKER: "Docker"}.get(self.mode, "")
        return f"OpenFOAM {self.version or ''} ({where})".replace("  ", " ")


def _run_quiet(argv: Sequence[str], timeout: float) -> tuple[int, str]:
    """Run a short command, returning ``(returncode, combined output)``. Never raises."""
    try:
        cp = subprocess.run(list(argv), capture_output=True, timeout=timeout, creationflags=CREATE_NO_WINDOW,
                            stdin=subprocess.DEVNULL)
        out = (cp.stdout or b"") + b"\n" + (cp.stderr or b"")
        # wsl.exe writes UTF-16LE for its own messages.
        text = out.decode("utf-8", errors="replace")
        if "\x00" in text:
            text = out.decode("utf-16-le", errors="replace")
        return cp.returncode, text
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return -1, str(exc)


def _parse_probe(text: str) -> tuple[str, str, str, tuple[str, ...]]:
    """Parse the ``AF_*`` lines printed by :data:`_PROBE_SCRIPT`."""
    bashrc = project = version = ""
    exes: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("AF_BASHRC="):
            bashrc = line.split("=", 1)[1]
        elif line.startswith("AF_PROJECT="):
            project = line.split("=", 1)[1]
        elif line.startswith("AF_VERSION="):
            version = line.split("=", 1)[1]
        elif line.startswith("AF_HAS="):
            exes.append(line.split("=", 1)[1])
    return bashrc, project, version, tuple(dict.fromkeys(exes))


def _usable(exes: tuple[str, ...]) -> bool:
    """True when the minimum tool chain is present."""
    solver = "simpleFoam" in exes or "foamRun" in exes
    return solver and "blockMesh" in exes and "snappyHexMesh" in exes and "foamToVTK" in exes


def _detect_native(timeout: float) -> Optional[OpenFOAMEnvironment]:
    if IS_WINDOWS or shutil.which("bash") is None:
        return None
    code, out = _run_quiet(["bash", "-c", _PROBE_SCRIPT], timeout)
    bashrc, project, version, exes = _parse_probe(out)
    if code == 0 and _usable(exes):
        return OpenFOAMEnvironment(ExecutionMode.NATIVE, version, project, bashrc, executables=exes,
                                   detail=f"Native installation ({bashrc or 'on PATH'}).")
    return None


def _detect_wsl(timeout: float) -> tuple[Optional[OpenFOAMEnvironment], bool]:
    if not IS_WINDOWS or shutil.which("wsl") is None:
        return None, False
    code, out = _run_quiet(["wsl", "-e", "bash", "-c", _PROBE_SCRIPT], timeout)
    if code != 0 and "AF_" not in out:
        return None, False
    bashrc, project, version, exes = _parse_probe(out)
    if _usable(exes):
        return OpenFOAMEnvironment(ExecutionMode.WSL, version, project, bashrc, executables=exes,
                                   detail="OpenFOAM inside the default WSL2 distribution.",
                                   wsl_available=True), True
    return None, True


def _detect_docker(timeout: float) -> tuple[Optional[OpenFOAMEnvironment], bool]:
    if shutil.which("docker") is None:
        return None, False
    code, _ = _run_quiet(["docker", "info", "--format", "{{.ServerVersion}}"], timeout)
    if code != 0:
        return None, False
    code, out = _run_quiet(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], timeout)
    images = [ln.strip() for ln in out.splitlines() if "openfoam" in ln.lower() and ":" in ln]
    preferred = [img for want in OPENFOAM.docker_images for img in images if img.startswith(want)]
    for image in preferred + [i for i in images if i not in preferred]:
        c2, o2 = _run_quiet(["docker", "run", "--rm", "--entrypoint", "bash", image, "-c", _PROBE_SCRIPT],
                            max(timeout, 30.0))
        bashrc, project, version, exes = _parse_probe(o2)
        if c2 == 0 and _usable(exes):
            return OpenFOAMEnvironment(ExecutionMode.DOCKER, version, project, bashrc, docker_image=image,
                                       executables=exes, detail=f"Docker image {image}.",
                                       docker_available=True), True
    return None, True


def detect_openfoam(timeout: float = OPENFOAM.detect_timeout_s) -> OpenFOAMEnvironment:
    """Detect OpenFOAM: native first, then WSL2, then Docker. Never raises."""
    try:
        env = _detect_native(timeout)
        if env:
            log.info("OpenFOAM detected: %s", env.detail)
            return env
        env_w, wsl_ok = _detect_wsl(timeout)
        if env_w:
            log.info("OpenFOAM detected: %s", env_w.detail)
            return env_w
        env_d, docker_ok = _detect_docker(timeout)
        if env_d:
            log.info("OpenFOAM detected: %s", env_d.detail)
            return replace_env(env_d, wsl_available=wsl_ok)
        hints = []
        if docker_ok:
            hints.append(f"Docker is running: pull an image with 'docker pull {OPENFOAM.docker_images[0]}'.")
        if wsl_ok:
            hints.append("WSL2 is available but OpenFOAM is not installed inside it.")
        detail = "OpenFOAM was not found. " + " ".join(hints) if hints else (
            "OpenFOAM was not found on PATH, in WSL2 or as a Docker image.")
        log.info(detail)
        return OpenFOAMEnvironment(detail=detail, docker_available=docker_ok, wsl_available=wsl_ok)
    except Exception as exc:  # detection must never break start-up
        log.warning("OpenFOAM detection failed: %s", exc)
        return OpenFOAMEnvironment(detail=f"Detection failed: {exc}")


def replace_env(env: OpenFOAMEnvironment, **changes: Any) -> OpenFOAMEnvironment:
    """Return a copy of ``env`` with some fields replaced."""
    from dataclasses import replace
    return replace(env, **changes)


def setup_instructions(env: Optional[OpenFOAMEnvironment] = None) -> list[tuple[str, str]]:
    """Platform-specific ``(title, shell commands)`` steps for the Setup Guide dialog."""
    system = platform.system()
    image = OPENFOAM.docker_images[0]
    steps: list[tuple[str, str]] = []
    if system == "Linux":
        steps.append(("Ubuntu / Debian (OpenFOAM Foundation 11)",
                      "sudo sh -c \"wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc\"\n"
                      "sudo add-apt-repository http://dl.openfoam.org/ubuntu\n"
                      "sudo apt-get update && sudo apt-get install -y openfoam11\n"
                      "echo 'source /opt/openfoam11/etc/bashrc' >> ~/.bashrc"))
        steps.append(("Alternative: ESI openfoam.com (v2312)",
                      "curl -s https://dl.openfoam.com/add-debian-repo.sh | sudo bash\n"
                      "sudo apt-get install -y openfoam2312-default"))
    elif system == "Windows":
        steps.append(("1 · Enable WSL2 (PowerShell as Administrator)", "wsl --install -d Ubuntu-22.04"))
        steps.append(("2 · Install OpenFOAM inside Ubuntu (WSL terminal)",
                      "sudo sh -c \"wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc\"\n"
                      "sudo add-apt-repository http://dl.openfoam.org/ubuntu\n"
                      "sudo apt-get update && sudo apt-get install -y openfoam11"))
    steps.append(("Docker (any OS, requires Docker Desktop / Engine)", f"docker pull {image}"))
    steps.append(("Then", "Restart AeroFlow Studio, or click 'Re-detect' in this dialog."))
    return steps


# ---------------------------------------------------------------------------
# Case generation
# ---------------------------------------------------------------------------
HEADER_TEMPLATE: Final[str] = """/*--------------------------------*- C++ -*----------------------------------*\\
  Generated by AeroFlow Studio on {date}
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       {cls};
    {location}object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""


def _header(cls: str, obj: str, location: str = "") -> str:
    loc = f'location    "{location}";\n    ' if location else ""
    return HEADER_TEMPLATE.format(date=datetime.now().strftime("%Y-%m-%d %H:%M"), cls=cls, obj=obj, location=loc)


def _vec(v: Sequence[float]) -> str:
    return "(" + " ".join(f"{float(x):.9g}" for x in v) + ")"


@dataclass(frozen=True)
class CaseSettings:
    """Inputs for :func:`write_case`."""

    conditions: FlowConditions
    quality: GridQuality = GridQuality.MEDIUM
    iterations: int = 600
    n_procs: int = 1
    write_interval: int = OPENFOAM.write_interval
    boundary_layers: Optional[int] = None      # prism layers on the body (None: the quality preset's value)
    compressible: bool = False                 # perfect-gas air (fluid / rhoSimpleFoam) instead of incompressible
    transient: bool = False                    # time-accurate PIMPLE run (unsteady wakes, shedding)
    convective_times: float = 12.0             # transient duration in t* = t V / L (flow passes over the body)
    frames: int = 24                           # transient: saved, evenly spaced snapshots


@dataclass(frozen=True)
class CaseLayout:
    """Geometry of the generated domain (also used for progress reporting and validation)."""

    case_dir: Path
    domain_min: tuple[float, float, float]
    domain_max: tuple[float, float, float]
    cells: tuple[int, int, int]
    base_cell: float
    location_in_mesh: tuple[float, float, float]
    lateral_type: str
    freestream: tuple[float, float, float]
    nu: float
    rho: float
    k_inlet: float
    omega_inlet: float


def _write_ascii_stl(model: GeometryModel, path: Path) -> None:
    """ASCII STL with a single solid named ``body`` (patch names become ``body*``)."""
    tri = model.vertices[model.faces]
    nrm = model.face_normals
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["solid body"]
    for n, t in zip(nrm, tri):
        lines.append(f"facet normal {n[0]:.7e} {n[1]:.7e} {n[2]:.7e}\n outer loop")
        lines.extend(f"  vertex {q[0]:.9e} {q[1]:.9e} {q[2]:.9e}" for q in t)
        lines.append(" endloop\nendfacet")
    lines.append("endsolid body\n")
    path.write_text("\n".join(lines), encoding="ascii")


def write_case(model: GeometryModel, settings: CaseSettings, case_dir: Optional[Path] = None) -> CaseLayout:
    """Generate a complete OpenFOAM case directory for ``model``.

    Raises:
        OpenFOAMError: if the directory cannot be written.
    """
    cond, _ = validate_flow_conditions(settings.conditions)
    info: GeometryInfo = model.info
    if case_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        case_dir = get_paths().cases / f"{sanitize_filename(model.name)}_{stamp}"
    case_dir = Path(case_dir)
    try:
        for sub in ("0", "constant/triSurface", "constant/geometry", "system", "logs"):
            (case_dir / sub).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OpenFOAMError(f"Cannot create case directory {case_dir}: {exc}") from exc

    preset = OPENFOAM_MESH_PRESETS[settings.quality]
    if settings.boundary_layers is not None:
        preset = replace(preset, boundary_layers=int(max(0, min(settings.boundary_layers, 20))))
    lo = np.asarray(info.bounds_min, dtype=float)
    hi = np.asarray(info.bounds_max, dtype=float)
    dim = max(info.chord_m, info.span_m, info.height_m, 1e-6)
    l_c = max(info.chord_m, 0.2 * dim)                 # length scale that sets the cell size
    lateral = max(1.0 * dim, 4.0 * l_c)
    margin_lo = np.array([max(1.5 * dim, 3.0 * l_c), lateral, lateral])
    margin_hi = np.array([max(3.0 * dim, 6.0 * l_c), lateral, lateral])
    d_min, d_max = lo - margin_lo, hi + margin_hi
    base = 6.0 * l_c / preset.base_cells_per_length
    cells = np.maximum(np.ceil((d_max - d_min) / base).astype(int), 4)
    loc = d_min + base * np.array([0.5123, 0.4871, 0.5317])

    atm = cond.atmosphere
    v_inf = np.asarray(cond.freestream_vector())
    speed = float(np.linalg.norm(v_inf))
    nu = atm.dynamic_viscosity / atm.density
    k_in = 1.5 * (OPENFOAM.turbulence_intensity * speed) ** 2
    omega_in = k_in / (nu * OPENFOAM.viscosity_ratio)
    aligned = abs(cond.alpha_deg) < 0.5 and abs(cond.beta_deg) < 0.5
    lateral_bc = "slip" if aligned else "freestream"

    layout = CaseLayout(case_dir, tuple(d_min), tuple(d_max), tuple(int(c) for c in cells), float(base),
                        tuple(float(x) for x in loc), lateral_bc, tuple(float(x) for x in v_inf), float(nu),
                        float(atm.density), float(k_in), float(omega_in))
    try:
        _write_ascii_stl(model, case_dir / "constant/triSurface/geometry.stl")
        shutil.copyfile(case_dir / "constant/triSurface/geometry.stl", case_dir / "constant/geometry/geometry.stl")
        files = {
            "system/blockMeshDict": _block_mesh_dict(layout),
            "system/surfaceFeaturesDict": _surface_features_dict(),
            "system/surfaceFeatureExtractDict": _surface_feature_extract_dict(),
            "system/snappyHexMeshDict": _snappy_dict(layout, info, preset, settings.quality),
            "system/meshQualityDict": _mesh_quality_dict(),
            "system/controlDict": _control_dict(layout, info, cond, settings),
            "system/fvSchemes": _fv_schemes(settings.transient and not settings.compressible),
            "system/fvSolution": _fv_solution(settings.transient and not settings.compressible),
            "system/decomposeParDict": _decompose_dict(max(1, settings.n_procs)),
            "constant/turbulenceProperties": _turbulence(esi=True),
            "constant/momentumTransport": _turbulence(esi=False),
            "constant/transportProperties": _transport(nu, "transportProperties"),
            "constant/physicalProperties": _transport(nu, "physicalProperties"),
            "0/U": _field_u(layout),
            "0/p": _field_p(layout),
            "0/k": _field_turb(layout, "k", "[0 2 -2 0 0 0 0]", k_in, "kqRWallFunction"),
            "0/omega": _field_turb(layout, "omega", "[0 0 -1 0 0 0 0]", omega_in, "omegaWallFunction"),
            "0/nut": _field_nut(layout),
        }
        if settings.compressible:
            phys = _cmp_physical(atm.dynamic_viscosity)
            files.update({"constant/physicalProperties": phys, "constant/thermophysicalProperties": phys,
                          "0/U": _cmp_u(layout), "system/fvSchemes": _cmp_fv_schemes(),
                          "system/fvSolution": _cmp_fv_solution(), "system/fvConstraints": _cmp_fv_constraints()})
            files.update(_cmp_fields(atm.pressure_pa, atm.temperature_k))
            files.pop("constant/transportProperties", None)
        for rel, text in files.items():
            (case_dir / rel).write_text(text, encoding="utf-8")
        (case_dir / "case.foam").write_text("", encoding="utf-8")
    except OSError as exc:
        raise OpenFOAMError(f"Cannot write case files: {exc}") from exc
    log.info("OpenFOAM case written to %s (%d×%d×%d base cells)", case_dir, *layout.cells)
    return layout


def _block_mesh_dict(c: CaseLayout) -> str:
    x0, y0, z0 = c.domain_min
    x1, y1, z1 = c.domain_max
    nx, ny, nz = c.cells
    return _header("dictionary", "blockMeshDict") + f"""convertToMeters 1;

vertices
(
    ({x0:.9g} {y0:.9g} {z0:.9g})
    ({x1:.9g} {y0:.9g} {z0:.9g})
    ({x1:.9g} {y1:.9g} {z0:.9g})
    ({x0:.9g} {y1:.9g} {z0:.9g})
    ({x0:.9g} {y0:.9g} {z1:.9g})
    ({x1:.9g} {y0:.9g} {z1:.9g})
    ({x1:.9g} {y1:.9g} {z1:.9g})
    ({x0:.9g} {y1:.9g} {z1:.9g})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

boundary
(
    inlet
    {{
        type patch;
        faces ((0 4 7 3));
    }}
    outlet
    {{
        type patch;
        faces ((1 2 6 5));
    }}
    tunnelWalls
    {{
        type patch;
        faces
        (
            (0 1 5 4)
            (3 7 6 2)
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);

mergePatchPairs ();
"""


def _surface_features_dict() -> str:
    return _header("dictionary", "surfaceFeaturesDict") + """surfaces ("geometry.stl");

includedAngle   150;

subsetFeatures
{
    nonManifoldEdges yes;
    openEdges       yes;
}

writeObj        no;
"""


def _surface_feature_extract_dict() -> str:
    return _header("dictionary", "surfaceFeatureExtractDict") + """geometry.stl
{
    extractionMethod    extractFromSurface;
    includedAngle       150;
    subsetFeatures
    {
        nonManifoldEdges yes;
        openEdges       yes;
    }
    writeObj            no;
}
"""


def _snappy_dict(c: CaseLayout, info: GeometryInfo, preset: Any, quality: GridQuality) -> str:
    lo = np.asarray(info.bounds_min)
    hi = np.asarray(info.bounds_max)
    l_c = max(info.chord_m, 0.2 * max(info.chord_m, info.span_m, info.height_m))
    box_lo = lo - np.array([0.5, 0.5, 0.5]) * l_c
    box_hi = hi + np.array([2.5, 0.5, 0.5]) * l_c
    box_level = 1 if quality == GridQuality.COARSE else 2
    lmin, lmax = preset.surface_min_level, preset.surface_max_level
    return _header("dictionary", "snappyHexMeshDict") + f"""castellatedMesh true;
snap            true;
addLayers       {"true" if preset.boundary_layers > 0 else "false"};

geometry
{{
    body
    {{
        type triSurfaceMesh;
        file "geometry.stl";      // explicit file: required by OpenFOAM 11+ and accepted by ESI
    }}
    refinementBox
    {{
        type searchableBox;
        min {_vec(box_lo)};
        max {_vec(box_hi)};
    }}
}}

castellatedMeshControls
{{
    maxLocalCells       2000000;
    maxGlobalCells      8000000;
    minRefinementCells  10;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 3;

    features
    (
        {{
            file "geometry.eMesh";
            level {lmax};
        }}
    );

    refinementSurfaces
    {{
        body
        {{
            level ({lmin} {lmax});
            patchInfo
            {{
                type wall;
            }}
        }}
    }}

    resolveFeatureAngle 30;

    refinementRegions
    {{
        refinementBox
        {{
            mode inside;
            level  {box_level};                 // OpenFOAM 11+ (openfoam.org)
            levels ((1E15 {box_level}));        // ESI / openfoam.com and older openfoam.org
        }}
    }}

    // Seed point inside the fluid region; each OpenFOAM flavour reads one of these names.
    insidePoint    {_vec(c.location_in_mesh)};
    locationInMesh {_vec(c.location_in_mesh)};
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch    3;
    tolerance       2.0;
    nSolveIter      50;
    nRelaxIter      5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes   true;
    layers
    {{
        "body.*"
        {{
            nSurfaceLayers {max(preset.boundary_layers, 1)};
        }}
    }}
    expansionRatio      1.2;
    finalLayerThickness 0.4;
    minThickness        0.1;
    nGrow               0;
    featureAngle        60;
    slipFeatureAngle    30;
    nRelaxIter          3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals      3;
    nSmoothThickness    10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle  90;
    minMedianAxisAngle  90;
    nBufferCellsNoExtrude 0;
    nLayerIter          50;
}}

meshQualityControls
{{
    #include "meshQualityDict"
}}

mergeTolerance 1e-6;
"""


def _mesh_quality_dict() -> str:
    return _header("dictionary", "meshQualityDict") + """maxNonOrtho         65;
maxBoundarySkewness 20;
maxInternalSkewness 4;
maxConcave          80;
minVol              1e-13;
minTetQuality       1e-15;
minArea             -1;
minTwist            0.02;
minDeterminant      0.001;
minFaceWeight       0.05;
minVolRatio         0.01;
minTriangleTwist    -1;
nSmoothScale        4;
errorReduction      0.75;
relaxed
{
    maxNonOrtho     75;
}
"""


def _control_dict(c: CaseLayout, info: GeometryInfo, cond: FlowConditions, s: CaseSettings) -> str:
    a, b = math.radians(cond.alpha_deg), math.radians(cond.beta_deg)
    drag_dir = (math.cos(a) * math.cos(b), math.sin(b), math.sin(a) * math.cos(b))
    lift_dir = (-math.sin(a), 0.0, math.cos(a))
    iters = int(max(10, s.iterations))
    write = int(max(1, min(s.write_interval, iters)))
    application = "rhoSimpleFoam" if s.compressible else ("pimpleFoam" if s.transient else "simpleFoam")
    solver = "fluid" if s.compressible else "incompressibleFluid"
    rho_line = "rho             rho;" if s.compressible else "rho             rhoInf;"
    if s.transient and not s.compressible:
        return _transient_control(c, info, cond, s, application, solver, rho_line, drag_dir, lift_dir)
    return _header("dictionary", "controlDict") + f"""application     {application};
solver          {solver};

startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {iters};
deltaT          1;
writeControl    timeStep;
writeInterval   {write};
purgeWrite      2;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{{
    forceCoeffs
    {{
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        log             yes;
        patches         ("body.*");
        {rho_line}
        rhoInf          {c.rho:.9g};
        CofR            {_vec(info.moment_ref_point)};
        dragDir         {_vec(drag_dir)};
        liftDir         {_vec(lift_dir)};
        pitchAxis       (0 1 0);
        magUInf         {cond.velocity_ms:.9g};
        lRef            {info.chord_m:.9g};
        Aref            {info.reference_area_m2:.9g};
    }}
    wallShearStress
    {{
        type            wallShearStress;
        libs            ("libfieldFunctionObjects.so");
        patches         ("body.*");
        writeControl    writeTime;
        log             false;
    }}
    yPlus
    {{
        type            yPlus;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        log             false;
    }}
}}
"""


def transient_end_time(info: GeometryInfo, cond: FlowConditions, s: CaseSettings) -> float:
    """Physical end time of a transient run: ``convective_times`` passes of the flow over the body."""
    length = max(info.chord_m, 1e-6)
    return float(s.convective_times) * length / max(cond.velocity_ms, 1e-6)


def _transient_control(c: CaseLayout, info: GeometryInfo, cond: FlowConditions, s: CaseSettings, application: str,
                       solver: str, rho_line: str, drag_dir: tuple, lift_dir: tuple) -> str:
    end = transient_end_time(info, cond, s)
    every = end / max(int(s.frames), 1)
    dt0 = 0.2 * c.base_cell / max(cond.velocity_ms, 1e-6)      # Courant ~0.2 on the background mesh to start
    return _header("dictionary", "controlDict") + f"""application     {application};
solver          {solver};

startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {end:.9g};
deltaT          {dt0:.6g};
adjustTimeStep  yes;
maxCo           2;
maxDeltaT       {every:.6g};
writeControl    adjustableRunTime;
writeInterval   {every:.9g};
purgeWrite      0;
writeFormat     binary;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   8;
runTimeModifiable true;

functions
{{
    forceCoeffs
    {{
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        log             yes;
        patches         ("body.*");
        {rho_line}
        rhoInf          {c.rho:.9g};
        CofR            {_vec(info.moment_ref_point)};
        dragDir         {_vec(drag_dir)};
        liftDir         {_vec(lift_dir)};
        pitchAxis       (0 1 0);
        magUInf         {cond.velocity_ms:.9g};
        lRef            {info.chord_m:.9g};
        Aref            {info.reference_area_m2:.9g};
    }}
}}
"""


def _fv_schemes(transient: bool = False) -> str:
    return _header("dictionary", "fvSchemes") + """ddtSchemes
{
    default         %s;
}""" % ("Euler" if transient else "steadyState") + """

gradSchemes
{
    default         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
    grad(k)         cellLimited Gauss linear 1;
    grad(omega)     cellLimited Gauss linear 1;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
    div((nuEff*dev(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear limited corrected 0.33;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         limited corrected 0.33;
}

wallDist
{
    method          meshWave;
}
"""


def _fv_solution(transient: bool = False) -> str:
    algorithm = """{
    nNonOrthogonalCorrectors 1;
    consistent      no;
    residualControl
    {
        p               %g;
        U               %g;
        "(k|omega)"     %g;
    }
}""" % (OPENFOAM.residual_target, OPENFOAM.residual_target * 0.1, OPENFOAM.residual_target * 0.1)
    return _header("dictionary", "fvSolution") + f"""solvers
{{
    p
    {{
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-7;
        relTol          0.05;
    }}
    "(U|k|omega)"
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
    }}
    "(U|k|omega)Final"
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0;
    }}
    pFinal
    {{
        $p;
        relTol          0;
    }}
}}

SIMPLE
{algorithm}

PIMPLE
{{
    nOuterCorrectors 1;
    nCorrectors      {2 if transient else 1};
    nNonOrthogonalCorrectors 1;
    consistent      no;
}}

{_TRANSIENT_RELAX if transient else _STEADY_RELAX}
"""


_STEADY_RELAX = """relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
        k               0.7;
        omega           0.7;
        ".*"            0.7;
    }
}"""
# Time-accurate PIMPLE: no under-relaxation (it would slow the physical time evolution).
_TRANSIENT_RELAX = """relaxationFactors
{
    equations
    {
        ".*"            1;
    }
}"""


def _decompose_dict(n: int) -> str:
    return _header("dictionary", "decomposeParDict") + f"""numberOfSubdomains {n};
method          scotch;
"""


def _turbulence(esi: bool) -> str:
    name = "turbulenceProperties" if esi else "momentumTransport"
    # ESI reads RASModel (older releases) or model (v2206+); Foundation reads only model.
    model_lines = "RASModel        kOmegaSST;\n    model           kOmegaSST;" if esi else "model           kOmegaSST;"
    return _header("dictionary", name) + f"""simulationType  RAS;

RAS
{{
    {model_lines}
    turbulence      on;
    printCoeffs     on;
}}
"""


def _transport(nu: float, obj: str) -> str:
    return _header("dictionary", obj) + f"""transportModel  Newtonian;
viscosityModel  constant;
nu              [0 2 -1 0 0 0 0] {nu:.9g};
"""


def _lateral_u(c: CaseLayout) -> str:
    if c.lateral_type == "slip":
        return "        type            slip;"
    return f"""        type            freestreamVelocity;
        freestreamValue uniform {_vec(c.freestream)};
        value           uniform {_vec(c.freestream)};"""


def _field_u(c: CaseLayout) -> str:
    u = _vec(c.freestream)
    return _header("volVectorField", "U", "0") + f"""dimensions      [0 1 -1 0 0 0 0];

internalField   uniform {u};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {u};
    }}
    outlet
    {{
        type            inletOutlet;
        inletValue      uniform (0 0 0);
        value           uniform {u};
    }}
    tunnelWalls
    {{
{_lateral_u(c)}
    }}
    "body.*"
    {{
        type            noSlip;
    }}
}}
"""


def _field_p(c: CaseLayout) -> str:
    lateral = ("        type            slip;" if c.lateral_type == "slip" else
               "        type            freestreamPressure;\n        freestreamValue uniform 0;\n"
               "        value           uniform 0;")
    return _header("volScalarField", "p", "0") + f"""dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
    inlet
    {{
        type            zeroGradient;
    }}
    outlet
    {{
        type            fixedValue;
        value           uniform 0;
    }}
    tunnelWalls
    {{
{lateral}
    }}
    "body.*"
    {{
        type            zeroGradient;
    }}
}}
"""


def _field_turb(c: CaseLayout, name: str, dims: str, value: float, wall: str) -> str:
    lateral = ("        type            slip;" if c.lateral_type == "slip" else
               f"        type            inletOutlet;\n        inletValue      uniform {value:.9g};\n"
               f"        value           uniform {value:.9g};")
    return _header("volScalarField", name, "0") + f"""dimensions      {dims};

internalField   uniform {value:.9g};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {value:.9g};
    }}
    outlet
    {{
        type            inletOutlet;
        inletValue      uniform {value:.9g};
        value           uniform {value:.9g};
    }}
    tunnelWalls
    {{
{lateral}
    }}
    "body.*"
    {{
        type            {wall};
        value           uniform {value:.9g};
    }}
}}
"""


# ---------------------------------------------------------------------------
# Compressible (perfect-gas air) case files - modelled on OpenFOAM's aerofoilNACA0012Steady tutorial
# ---------------------------------------------------------------------------
_FAR = '"(inlet|outlet|tunnelWalls)"'


def _cmp_physical(mu: float) -> str:
    return _header("dictionary", "physicalProperties") + f"""thermoType
{{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleEnthalpy;
}}

mixture
{{
    specie          {{ molWeight 28.9; }}
    thermodynamics  {{ Cp 1005; Hf 0; hf 0; }}
    transport       {{ mu {mu:.6g}; Pr 0.71; }}
}}
"""


def _cmp_u(c: CaseLayout) -> str:
    u = _vec(c.freestream)
    return _header("volVectorField", "U", "0") + f"""dimensions      [0 1 -1 0 0 0 0];

internalField   uniform {u};

boundaryField
{{
    {_FAR}
    {{
        type            freestreamVelocity;
        freestreamValue uniform {u};
        value           uniform {u};
    }}
    "body.*"
    {{
        type            noSlip;
    }}
}}
"""


def _cmp_scalar(name: str, dims: str, value: float, far: str, wall: str) -> str:
    return _header("volScalarField", name, "0") + f"""dimensions      {dims};

internalField   uniform {value:.9g};

boundaryField
{{
    {_FAR}
    {{
{far}
    }}
    "body.*"
    {{
{wall}
    }}
}}
"""


def _cmp_fields(p_inf: float, t_inf: float) -> dict[str, str]:
    return {
        "0/p": _cmp_scalar("p", "[1 -1 -2 0 0 0 0]", p_inf,
                           f"        type            freestreamPressure;\n        freestreamValue uniform {p_inf:.9g};\n"
                           f"        value           uniform {p_inf:.9g};",
                           "        type            zeroGradient;"),
        "0/T": _cmp_scalar("T", "[0 0 0 1 0 0 0]", t_inf,
                           f"        type            inletOutlet;\n        inletValue      uniform {t_inf:.9g};\n"
                           f"        value           uniform {t_inf:.9g};",
                           "        type            zeroGradient;"),
        "0/alphat": _cmp_scalar("alphat", "[1 -1 -1 0 0 0 0]", 0.0,
                                "        type            calculated;\n        value           uniform 0;",
                                "        type            compressible::alphatWallFunction;\n"
                                "        value           uniform 0;"),
    }


def _cmp_fv_schemes() -> str:
    return _header("dictionary", "fvSchemes") + """ddtSchemes      { default steadyState; }

gradSchemes
{
    default         Gauss linear;
    limited         cellLimited Gauss linear 1;
    grad(U)         $limited;
    grad(k)         $limited;
    grad(omega)     $limited;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind limited;
    energy          bounded Gauss linearUpwind limited;
    div(phi,h)      $energy;
    div(phi,e)      $energy;
    div(phi,K)      $energy;
    div(phi,Ekp)    $energy;
    turbulence      bounded Gauss upwind;
    div(phi,k)      $turbulence;
    div(phi,omega)  $turbulence;
    div(((rho*nuEff)*dev2(T(grad(U)))))    Gauss linear;
}

laplacianSchemes      { default Gauss linear corrected; }
interpolationSchemes  { default linear; }
snGradSchemes         { default corrected; }
wallDist              { method meshWave; }
"""


def _cmp_fv_solution() -> str:
    # PIMPLE (OpenFOAM.org modular 'fluid' solver, steady via steadyState ddt) and SIMPLE (ESI rhoSimpleFoam)
    return _header("dictionary", "fvSolution") + """solvers
{
    p
    {
        solver          GAMG;
        smoother        DIC;
        tolerance       1e-8;
        relTol          0.01;
    }
    "(U|k|omega|h|e)"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-10;
        relTol          0.1;
    }
}

PIMPLE
{
    nNonOrthogonalCorrectors 0;
    residualControl { p 1e-6; U 1e-5; "(k|omega|h)" 1e-5; }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    transonic       no;
    consistent      no;
    pMinFactor      0.1;
    pMaxFactor      2;
    residualControl { p 1e-6; U 1e-5; "(k|omega|h|e)" 1e-5; }
}

relaxationFactors
{
    fields          { p 0.3; rho 0.01; }
    equations       { U 0.7; h 0.7; e 0.7; "(k|omega)" 0.7; }
}
"""


def _cmp_fv_constraints() -> str:
    return _header("dictionary", "fvConstraints") + """#includeConstraint limitPressure(minFactor=0.1, maxFactor=2)
#includeConstraint limitTemperature(min=101, max=1000)
"""


def _field_nut(c: CaseLayout) -> str:
    lateral = "        type            slip;" if c.lateral_type == "slip" else \
        "        type            calculated;\n        value           uniform 0;"
    return _header("volScalarField", "nut", "0") + f"""dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
    inlet
    {{
        type            calculated;
        value           uniform 0;
    }}
    outlet
    {{
        type            calculated;
        value           uniform 0;
    }}
    tunnelWalls
    {{
{lateral}
    }}
    "body.*"
    {{
        type            nutkWallFunction;
        value           uniform 0;
    }}
}}
"""


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def windows_to_wsl_path(path: Path) -> str:
    """Convert ``C:\\Users\\me\\case`` to ``/mnt/c/Users/me/case`` (UNC WSL paths are passed through)."""
    p = PureWindowsPath(str(path))
    s = str(p)
    if s.startswith("\\\\wsl"):
        parts = p.parts[1:] if len(p.parts) > 1 else ()
        return "/" + "/".join(parts[1:]) if parts else "/"
    drive = p.drive.rstrip(":").lower()
    rest = "/".join(p.parts[1:])
    return f"/mnt/{drive}/{rest}" if drive else s.replace("\\", "/")


@dataclass
class RunReport:
    """Summary of a finished (or failed) run."""

    case_dir: Path
    success: bool
    iterations: int = 0
    residuals: dict[str, list[float]] = field(default_factory=dict)
    coefficients: dict[str, float] = field(default_factory=dict)
    elapsed_s: float = 0.0
    message: str = ""
    mesh_quality: str = ""                  # checkMesh summary (cells, non-orthogonality, skewness, verdict)


@dataclass(frozen=True)
class _Stage:
    name: str
    command: str
    weight: float
    log_name: str


class OpenFOAMRunner:
    """Runs the full pipeline for one case. Not reusable: create one runner per run.

    All callbacks are invoked from the calling (worker) thread.
    """

    def __init__(self, env: OpenFOAMEnvironment, on_log: Optional[LogFn] = None,
                 on_progress: Optional[ProgressFn] = None, on_residual: Optional[ResidualFn] = None,
                 on_coefficients: Optional[CoeffFn] = None) -> None:
        if not env.available:
            raise OpenFOAMError("OpenFOAM is not available on this machine.")
        self.env = env
        self._on_log = on_log or (lambda s: None)
        self._on_progress = on_progress or (lambda f, m: None)
        self._on_residual = on_residual or (lambda i, r: None)
        self._on_coeff = on_coefficients or (lambda i, c: None)
        self._proc: Optional[subprocess.Popen[str]] = None
        self._proc_lock = threading.Lock()
        self._aborted = threading.Event()
        self._container = f"aeroflow_{uuid.uuid4().hex[:10]}"
        self._case_dir: Optional[Path] = None

    # -- control ---------------------------------------------------------
    @property
    def aborted(self) -> bool:
        """True once :meth:`abort` has been called."""
        return self._aborted.is_set()

    def abort(self) -> None:
        """Terminate the running stage and its whole process tree. Safe from any thread."""
        self._aborted.set()
        with self._proc_lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        log.warning("Aborting OpenFOAM run")
        try:
            if self.env.mode == ExecutionMode.DOCKER:
                _run_quiet(["docker", "kill", self._container], 15.0)
            if self.env.mode == ExecutionMode.WSL and self._case_dir is not None:
                marker = windows_to_wsl_path(self._case_dir)
                _run_quiet(["wsl", "-e", "bash", "-c", f"pkill -f {shlex.quote(marker)} || true"], 10.0)
            if IS_WINDOWS:
                _run_quiet(["taskkill", "/PID", str(proc.pid), "/T", "/F"], 10.0)
            else:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    proc.terminate()
            try:
                proc.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                if not IS_WINDOWS:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                proc.kill()
        except Exception as exc:  # abort must never raise
            log.error("Error while aborting: %s", exc)

    # -- command construction --------------------------------------------
    def _argv(self, case_dir: Path, command: str) -> list[str]:
        """Build the full argv to run ``command`` (an OpenFOAM utility line) for ``case_dir``."""
        if self.env.mode == ExecutionMode.NATIVE:
            case = shlex.quote(str(case_dir))
            return ["bash", "-c", f"{_SOURCE_SNIPPET} cd {case} && {command} -case {case}"]
        if self.env.mode == ExecutionMode.WSL:
            case = shlex.quote(windows_to_wsl_path(case_dir))
            return ["wsl", "-e", "bash", "-c", f"{_SOURCE_SNIPPET} cd {case} && {command} -case {case}"]
        if self.env.mode == ExecutionMode.DOCKER:
            inner = OPENFOAM.docker_case_dir
            argv = ["docker", "run", "--rm", "--name", self._container, "-v", f"{case_dir}:{inner}",
                    "-w", inner, "-e", "HOME=/tmp"]
            if hasattr(os, "getuid") and not IS_WINDOWS:
                argv += ["-u", f"{os.getuid()}:{os.getgid()}"]
            argv += ["--entrypoint", "bash", self.env.docker_image, "-c",
                     f"{_SOURCE_SNIPPET} cd {inner} && {command} -case {inner}"]
            return argv
        raise OpenFOAMError("OpenFOAM is not available.")

    def _stages(self, n_procs: int, compressible: bool = False, transient: bool = False) -> list[_Stage]:
        """Pipeline for the detected flavour."""
        if compressible:
            solver = "foamRun -solver fluid" if "foamRun" in self.env.executables else "rhoSimpleFoam"
        elif transient:
            solver = "foamRun -solver incompressibleFluid" if "foamRun" in self.env.executables else "pimpleFoam"
        else:
            solver = "foamRun -solver incompressibleFluid" if self.env.uses_foam_run else "simpleFoam"
        stages = [
            _Stage("Generating background mesh (blockMesh)", "blockMesh", 0.03, "blockMesh"),
            _Stage("Extracting surface features", self.env.feature_command, 0.02, "features"),
            _Stage("Meshing around geometry (snappyHexMesh)", "snappyHexMesh -overwrite", 0.24, "snappyHexMesh"),
            _Stage("Checking mesh quality (checkMesh)", "checkMesh", 0.01, "checkMesh"),
        ]
        has_mpi = "mpirun" in self.env.executables and "decomposePar" in self.env.executables
        if n_procs > 1 and has_mpi:
            stages.append(_Stage("Decomposing case", "decomposePar -force", 0.01, "decomposePar"))
            stages.append(_Stage("Solving RANS k-ω SST", f"mpirun -np {n_procs} {solver} -parallel", 0.62, "solver"))
            stages.append(_Stage("Reconstructing", "reconstructPar -noZero" if transient else "reconstructPar -latestTime", 0.02, "reconstructPar"))
        else:
            stages.append(_Stage("Solving RANS k-ω SST", solver, 0.64, "solver"))
        stages.append(_Stage("Converting results (foamToVTK)", "foamToVTK -noZero" if transient else
                             "foamToVTK -latestTime", 0.06, "foamToVTK"))
        return stages

    # -- execution -------------------------------------------------------
    def run(self, layout: CaseLayout, settings: CaseSettings) -> RunReport:
        """Run the whole pipeline. Raises :class:`OpenFOAMAborted` or :class:`OpenFOAMError`."""
        return self.run_stages(layout, settings)

    def run_stages(self, layout: CaseLayout, settings: CaseSettings, mesh_only: bool = False,
                   solve_only: bool = False) -> RunReport:
        """Run the pipeline, or only its meshing stages (``mesh_only``), or only the solver on an existing
        mesh without the VTK export (``solve_only``; used by polar sweeps)."""
        case_dir = layout.case_dir
        self._case_dir = case_dir
        report = RunReport(case_dir=case_dir, success=False)
        t0 = time.perf_counter()
        transient = settings.transient and not settings.compressible
        stages = self._stages(settings.n_procs, settings.compressible, transient)
        self._end_time = None
        if transient:          # progress = simulated time / end time (the time step adapts)
            m = re.search(r"^endTime\s+([0-9.eE+-]+);", (case_dir / "system" / "controlDict").read_text(), re.M)
            self._end_time = float(m.group(1)) if m else None
        mesh_names = {"blockMesh", "features", "snappyHexMesh", "checkMesh"}
        if mesh_only:
            stages = [s for s in stages if s.log_name in mesh_names]
        elif solve_only:
            stages = [s for s in stages if s.log_name not in mesh_names and s.log_name != "foamToVTK"]
        total = sum(s.weight for s in stages) or 1.0
        stages = [replace(s, weight=s.weight / total) for s in stages]
        done = 0.0
        for stage in stages:
            if self.aborted:
                raise OpenFOAMAborted("Simulation aborted by the user.")
            self._on_progress(done, stage.name)
            self._on_log(f"▶ {stage.name}")
            is_solver = stage.log_name == "solver"
            code, tail = self._run_stage(case_dir, stage, done, settings.iterations if is_solver else 0, report)
            if self.aborted:
                raise OpenFOAMAborted("Simulation aborted by the user.")
            if stage.log_name == "checkMesh":
                report.mesh_quality = summarise_check_mesh(case_dir / "logs" / "log.checkMesh")
                self._on_log(f"  Mesh: {report.mesh_quality}")
                done += stage.weight
                continue                       # informative only: a few failed checks do not stop the run
            if code != 0:
                hint = _failure_hint(stage.log_name, tail)
                raise OpenFOAMError(f"{stage.name} failed (exit code {code}). {hint}\n--- last output ---\n{tail}")
            done += stage.weight
        self._on_progress(1.0, "OpenFOAM run complete")
        report.coefficients.update(read_force_coefficients(case_dir) or report.coefficients)
        report.success = True
        report.elapsed_s = time.perf_counter() - t0
        report.message = f"Completed {report.iterations} iterations in {report.elapsed_s:.0f} s."
        return report

    def _run_stage(self, case_dir: Path, stage: _Stage, base_progress: float, iterations: int,
                   report: RunReport) -> tuple[int, str]:
        """Run one stage, streaming its output. Returns ``(exit code, last lines)``."""
        argv = self._argv(case_dir, stage.command)
        log.debug("Running: %s", " ".join(argv))
        kwargs: dict[str, Any] = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                      text=True, encoding="utf-8", errors="replace", bufsize=1)
        if IS_WINDOWS:
            kwargs["creationflags"] = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        try:
            proc = subprocess.Popen(argv, **kwargs)
        except OSError as exc:
            raise OpenFOAMError(f"Could not start {stage.command}: {exc}") from exc
        with self._proc_lock:
            self._proc = proc
        tail: list[str] = []
        iteration = 0
        current: dict[str, float] = {}
        last_emit = 0.0
        log_path = case_dir / "logs" / f"log.{stage.log_name}"
        try:
            with log_path.open("w", encoding="utf-8") as fh:
                assert proc.stdout is not None
                for raw in proc.stdout:
                    line = raw.rstrip("\n")
                    fh.write(raw)
                    tail.append(line)
                    if len(tail) > 40:
                        tail.pop(0)
                    if iterations:
                        m = _RE_TIME.match(line)
                        if m:
                            if current:
                                self._emit_residuals(iteration, current, report)
                                current = {}
                            end_time = getattr(self, "_end_time", None)
                            if end_time:                 # transient: count time steps, progress by time
                                iteration += 1
                                try:
                                    frac = min(float(m.group(1)) / end_time, 1.0)
                                except ValueError:
                                    frac = 0.0
                            else:
                                try:
                                    iteration = int(float(m.group(1)))
                                except ValueError:
                                    iteration += 1
                                frac = min(iteration / max(iterations, 1), 1.0)
                            report.iterations = iteration
                            now = time.monotonic()
                            if now - last_emit > 0.2:
                                self._on_progress(base_progress + stage.weight * frac,
                                                  f"Time {100 * frac:.0f} % ({iteration} steps)" if end_time
                                                  else f"Iteration {iteration} / {iterations}")
                                last_emit = now
                            continue
                        r = _RE_RESIDUAL.search(line)
                        if r:
                            name = r.group(1)
                            if name not in current:          # first (initial) residual per iteration
                                try:
                                    current[name] = float(r.group(2))
                                except ValueError:
                                    pass
                            continue
                        coeffs = dict((k, float(v)) for k, v in _RE_COEFF.findall(line) if _is_float(v))
                        if coeffs:
                            if "CmPitch" in coeffs:
                                coeffs["Cm"] = coeffs.pop("CmPitch")
                            report.coefficients.update(coeffs)
                            self._on_coeff(iteration, dict(report.coefficients))
                    else:
                        if stage.log_name == "snappyHexMesh":
                            ph = _RE_SNAPPY_PHASE.match(line)
                            if ph:
                                self._on_log(f"  snappyHexMesh: {ph.group(1)}")
                        if "FOAM FATAL" in line or "FOAM FATAL IO ERROR" in line:
                            self._on_log(line)
            if current:
                self._emit_residuals(iteration, current, report)
            code = proc.wait()
        finally:
            with self._proc_lock:
                self._proc = None
        if any("FOAM FATAL" in t for t in tail) and code == 0:
            code = 1
        return code, "\n".join(tail[-25:])

    def _emit_residuals(self, iteration: int, values: dict[str, float], report: RunReport) -> None:
        for k, v in values.items():
            report.residuals.setdefault(k, []).append(v)
        self._on_residual(iteration, dict(values))


def summarise_check_mesh(log_path: Path) -> str:
    """One line from a checkMesh log: cell count, worst non-orthogonality and skewness, and the verdict."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "checkMesh did not run"
    parts = []
    m = re.search(r"^\s*cells:\s+(\d+)", text, re.M)
    if m:
        parts.append(f"{int(m.group(1)):,} cells")
    m = re.search(r"Mesh non-orthogonality Max:\s*([\d.eE+-]+)", text)
    if m:
        parts.append(f"max non-orthogonality {float(m.group(1)):.1f}°")
    m = re.search(r"Max skewness = ([\d.eE+-]+)", text)
    if m:
        parts.append(f"max skewness {float(m.group(1)):.2f}")
    m = re.search(r"Failed (\d+) mesh checks", text)
    parts.append(f"{m.group(1)} quality check(s) failed" if m else ("mesh OK" if "Mesh OK" in text else "verdict n/a"))
    return ", ".join(parts)


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _failure_hint(stage: str, tail: str) -> str:
    """Plain-English hint for common failures."""
    t = tail.lower()
    if "fatal io error" in t or ("keyword" in t and "undefined" in t):
        return ("OpenFOAM rejected one of the generated case files (a dictionary format problem, not your "
                "geometry). Please report the log above so the case generator can be fixed.")
    if stage == "snappyHexMesh":
        if "locationinmesh" in t or "outside" in t:
            return "The mesh seed point ended up inside the geometry; check that the model is closed and centred."
        if "leak" in t or "not closed" in t or "open edges" in t:
            return "Meshing failed — the geometry may not be watertight. Try 'Repair mesh' first."
        return "Meshing failed. See the snappyHexMesh log in the case's logs folder for details."
    if stage == "solver":
        if "floating point exception" in t or "nan" in t:
            return "The solver diverged. Try a coarser mesh, lower speed, or a smaller angle of attack."
        return "The solver stopped early. See the Activity Log for details."
    if stage == "features":
        return "Feature extraction failed; the STL may contain invalid triangles."
    if "permission denied" in t:
        return "Permission denied while writing the case directory."
    return ""


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def read_force_coefficients(case_dir: Path) -> dict[str, float]:
    """Read the last row of the forceCoeffs output (Foundation or ESI layout). Empty dict if absent."""
    root = Path(case_dir) / "postProcessing"
    if not root.is_dir():
        return {}
    candidates = sorted(list(root.rglob("forceCoeffs*.dat")) + list(root.rglob("coefficient*.dat")),
                        key=lambda p: p.stat().st_mtime)
    for path in reversed(candidates):
        try:
            header: list[str] = []
            last: list[str] = []
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("#"):
                    cols = line.lstrip("#").split()
                    if cols and cols[0].lower() == "time":
                        header = cols
                elif line.strip():
                    last = line.split()
            if not header or not last:
                continue
            out: dict[str, float] = {}
            for name, value in zip(header, last):
                key = {"CmPitch": "Cm", "Cm": "Cm", "Cd": "Cd", "Cl": "Cl", "Cs": "Cs"}.get(name)
                if key and _is_float(value):
                    out[key] = float(value)
            if out:
                return out
        except OSError:
            continue
    return {}


def find_vtk_output(case_dir: Path) -> Optional[Path]:
    """Return the newest foamToVTK output file (``.vtm`` or ``.vtk``) of the internal field."""
    vtk_dir = Path(case_dir) / "VTK"
    if not vtk_dir.is_dir():
        return None
    vtm = sorted(vtk_dir.glob("*.vtm"), key=lambda p: p.stat().st_mtime)
    if vtm:
        return vtm[-1]
    vtks = [p for p in vtk_dir.glob("*.vtk")]
    return max(vtks, key=lambda p: p.stat().st_mtime) if vtks else None


def _surface_from_block(block: Any) -> Optional[SurfaceField]:
    """Convert a PyVista boundary patch to a :class:`SurfaceField` (Cp filled later)."""
    try:
        try:
            surf = block.extract_surface(algorithm="dataset_surface").triangulate()
        except TypeError:                         # PyVista < 0.47 has no `algorithm` keyword
            surf = block.extract_surface().triangulate()
        if surf.n_points == 0:
            return None
        if "p" not in surf.point_data and "p" in surf.cell_data:
            surf = surf.cell_data_to_point_data()
        faces = np.asarray(surf.faces).reshape(-1, 4)[:, 1:]
        p = np.asarray(surf.point_data["p"], dtype=np.float32) if "p" in surf.point_data else \
            np.zeros(surf.n_points, np.float32)
        extra: dict[str, np.ndarray] = {}
        if "wallShearStress" not in surf.point_data and "wallShearStress" in surf.cell_data:
            surf = surf.cell_data_to_point_data()
        if "wallShearStress" in surf.point_data:      # kinematic (m^2/s^2) in incompressible solvers
            extra["wss"] = np.linalg.norm(np.asarray(surf.point_data["wallShearStress"], dtype=np.float32), axis=1)
        if "yPlus" in surf.point_data:
            extra["yPlus"] = np.asarray(surf.point_data["yPlus"], dtype=np.float32)
        return SurfaceField(np.asarray(surf.points, np.float32), faces.astype(np.int32), p, extra)
    except Exception as exc:
        log.warning("Could not extract the body surface from OpenFOAM output: %s", exc)
        return None


def transient_frames(case_dir: Path) -> list[tuple[float, Path]]:
    """Internal-field VTK files of a transient run, ``[(time, path), ...]`` sorted by time.

    foamToVTK names its files by time-step *index*, so the physical times come from the case's time
    folders (same order); without them the index is used as the label."""
    case_dir = Path(case_dir)
    vtk_dir = case_dir / "VTK"
    files = []
    for f in (vtk_dir.glob("*.vtk") if vtk_dir.is_dir() else []):
        try:
            files.append((int(f.stem.rsplit("_", 1)[-1]), f))
        except ValueError:
            continue
    files.sort()
    times = []
    for d in case_dir.iterdir():
        if d.is_dir():
            try:
                t = float(d.name)
            except ValueError:
                continue
            if t > 0.0:
                times.append(t)
    times.sort()
    if len(times) == len(files):
        return [(t, f) for t, (_i, f) in zip(times, files)]
    return [(float(i), f) for i, f in files]


def force_history(case_dir: Path) -> dict[str, np.ndarray]:
    """Whole forceCoeffs time series: ``{"t", "Cl", "Cd", "Cm"}`` (empty dict if absent)."""
    root = Path(case_dir) / "postProcessing"
    files = sorted(list(root.rglob("forceCoeffs*.dat")) + list(root.rglob("coefficient*.dat")),
                   key=lambda p: p.stat().st_mtime) if root.is_dir() else []
    for path in reversed(files):
        header: list[str] = []
        rows: list[list[float]] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("#"):
                cols = line.lstrip("#").split()
                if cols and cols[0].lower() == "time":
                    header = cols
            elif line.strip() and header:
                try:
                    rows.append([float(x) for x in line.split()[:len(header)]])
                except ValueError:
                    continue
        if header and rows:
            a = np.asarray(rows)
            idx = {name: k for k, name in enumerate(header)}
            out = {"t": a[:, 0]}
            for key, names in (("Cl", ("Cl",)), ("Cd", ("Cd",)), ("Cm", ("CmPitch", "Cm"))):
                for n in names:
                    if n in idx and idx[n] < a.shape[1]:
                        out[key] = a[:, idx[n]]
                        break
            return out
    return {}


def load_openfoam_result(model: GeometryModel, layout: CaseLayout, settings: CaseSettings,
                         report: RunReport, vtk_path: Optional[Path] = None) -> SolverResult:
    """Read foamToVTK output and assemble a :class:`SolverResult`.

    Pressure in incompressible OpenFOAM is kinematic (p/rho), so ``Cp = p / (0.5 V^2)``.

    Raises:
        OpenFOAMError: if no readable output exists.
    """
    import pyvista as pv   # local import keeps the module importable without VTK for detection

    path = vtk_path or find_vtk_output(layout.case_dir)
    if path is None:
        raise OpenFOAMError("foamToVTK produced no output files.")
    frame_time = path.stem.rsplit("_", 1)[-1]          # body patch files of the same write time
    try:
        data = pv.read(str(path))
    except Exception as exc:
        raise OpenFOAMError(f"Could not read {path.name}: {exc}") from exc

    internal = None
    body_blocks: list[Any] = []
    if isinstance(data, pv.MultiBlock):
        def walk(mb: Any, name: str = "") -> None:
            nonlocal internal
            for i in range(mb.n_blocks):
                blk = mb[i]
                key = (mb.get_block_name(i) or "") if hasattr(mb, "get_block_name") else ""
                full = f"{name}/{key}"
                if blk is None:
                    continue
                if isinstance(blk, pv.MultiBlock):
                    walk(blk, full)
                elif "internal" in key.lower() and internal is None:
                    internal = blk
                elif "body" in key.lower():
                    body_blocks.append(blk)
        walk(data)
        if internal is None:
            for i in range(data.n_blocks):
                if data[i] is not None and not isinstance(data[i], pv.MultiBlock):
                    internal = data[i]
                    break
    else:
        internal = data
    if internal is None or internal.n_points == 0:
        raise OpenFOAMError("The OpenFOAM result contains no volume mesh.")
    if body_blocks == []:
        # Legacy-format foamToVTK writes one folder per patch next to the internal file
        # (VTK/body/body_<time>.vtk); some versions nest them under VTK/<case>_<time>/ instead.
        for patch_dir in (path.parent / path.stem, path.parent / "body"):
            files = sorted(patch_dir.rglob("body*.vt*"), key=lambda p: p.stat().st_mtime) if patch_dir.is_dir() else []
            if files:
                times = {f.stem.rsplit("_", 1)[-1] for f in files}
                newest_time = frame_time if frame_time in times else files[-1].stem.rsplit("_", 1)[-1]
                for f in files:
                    if f.stem.rsplit("_", 1)[-1] != newest_time:
                        continue                      # an older write time of the same patch
                    try:
                        body_blocks.append(pv.read(str(f)))
                    except Exception:
                        pass
            if body_blocks:
                break

    cond, _ = validate_flow_conditions(settings.conditions)
    v_inf = max(cond.velocity_ms, 1e-9)
    q_kin = 0.5 * v_inf * v_inf
    # Incompressible: p is kinematic gauge pressure (p/rho). Compressible: absolute pressure in Pa.
    p_ref = cond.atmosphere.pressure_pa if settings.compressible else 0.0
    q_p = 0.5 * cond.atmosphere.density * v_inf * v_inf if settings.compressible else q_kin
    if "U" not in internal.point_data and "U" in internal.cell_data:
        internal = internal.cell_data_to_point_data()
    if "U" not in internal.point_data:
        raise OpenFOAMError("The OpenFOAM result has no velocity field 'U'.")
    vel = np.asarray(internal.point_data["U"], dtype=np.float32)
    internal.point_data["velocity"] = vel
    internal.point_data["speed"] = np.linalg.norm(vel, axis=1).astype(np.float32)
    p = np.asarray(internal.point_data["p"], dtype=np.float32) if "p" in internal.point_data else \
        np.zeros(internal.n_points, np.float32)
    internal.point_data["Cp"] = ((p - p_ref) / q_p).astype(np.float32)
    if "T" not in internal.point_data and "T" in internal.cell_data:
        internal = internal.cell_data_to_point_data()
    if "T" in internal.point_data:                    # local Mach number (perfect-gas air)
        t = np.maximum(np.asarray(internal.point_data["T"], dtype=np.float32), 1.0)
        internal.point_data["Mach"] = (internal.point_data["speed"] / np.sqrt(1.4 * 287.05 * t)).astype(np.float32)
    internal.set_active_vectors("velocity")

    surface: Optional[SurfaceField] = None
    if body_blocks:
        merged = body_blocks[0] if len(body_blocks) == 1 else pv.merge(body_blocks)
        surface = _surface_from_block(merged)
        if surface is not None:
            extra = {"yPlus": surface.extra["yPlus"]} if "yPlus" in surface.extra else {}
            if "wss" in surface.extra:                 # kinematic (incompressible) or Pa (compressible)
                extra["Cf"] = (surface.extra["wss"] / q_p).astype(np.float32)        # skin-friction coefficient
            surface = SurfaceField(surface.vertices, surface.faces, ((surface.cp - p_ref) / q_p).astype(np.float32),
                                   extra)

    info = model.info
    atm = cond.atmosphere
    q = 0.5 * atm.density * v_inf * v_inf
    co = report.coefficients
    cl, cd, cm = co.get("Cl", 0.0), co.get("Cd", 0.0), co.get("Cm", 0.0)
    re_num = reynolds_number(atm.density, v_inf, info.chord_m, atm.dynamic_viscosity)
    cp_all = internal.point_data["Cp"]
    kpis = AeroKPIs(
        alpha_deg=cond.alpha_deg, beta_deg=cond.beta_deg, velocity_ms=v_inf, cl=cl, cd=cd, cm=cm,
        cd_pressure=cd, lift_n=cl * q * info.reference_area_m2, drag_n=cd * q * info.reference_area_m2,
        moment_nm=cm * q * info.reference_area_m2 * info.chord_m, dynamic_pressure_pa=q,
        stagnation_pressure_pa=atm.pressure_pa + q, static_pressure_pa=atm.pressure_pa,
        max_velocity_ms=float(internal.point_data["speed"].max()),
        cp_min=float(surface.cp.min()) if surface is not None and len(surface.cp) else float(cp_all.min()),
        cp_max=float(surface.cp.max()) if surface is not None and len(surface.cp) else float(cp_all.max()),
        reynolds=re_num, mach=cond.mach, ref_area_m2=info.reference_area_m2, ref_length_m=info.chord_m,
        aspect_ratio=info.aspect_ratio)
    warnings: list[str] = []
    if not co:
        warnings.append("No force coefficients were found in postProcessing; CL/CD are unavailable.")
    if surface is not None and "yPlus" in surface.extra and len(surface.extra["yPlus"]):
        yp = float(np.median(surface.extra["yPlus"]))
        if not 30.0 <= yp <= 300.0:
            warnings.append(f"Median y+ on the body is {yp:.0f}; the wall functions used here want 30-300 - "
                            f"{'refine' if yp > 300 else 'coarsen'} the near-wall cells (boundary layers / "
                            "resolution) for reliable friction and drag.")
    if report.mesh_quality:
        log.info("Mesh quality: %s", report.mesh_quality)
    final = {k: v[-1] for k, v in report.residuals.items() if v}
    if final and max(final.values()) > 1e-2:
        warnings.append("Residuals are still high — the solution may not be converged.")
    return SolverResult(engine="OpenFOAM", conditions=cond, kpis=kpis,
                        field=FlowField(dataset=internal, freestream_ms=v_inf), surface=surface,
                        geometry_name=model.name, elapsed_s=report.elapsed_s, warnings=warnings,
                        residuals=report.residuals,
                        extra={"case_dir": str(layout.case_dir), "cells": layout.cells,
                               "mesh_quality": report.mesh_quality,
                               "transient": bool(settings.transient and not settings.compressible),
                               "iterations": report.iterations, "vtk_file": str(path)})


def run_openfoam_polar(model: GeometryModel, settings: CaseSettings, env: OpenFOAMEnvironment,
                       runner: "OpenFOAMRunner", alphas: Sequence[float],
                       on_point: Optional[Callable[[int, float, dict[str, float]], None]] = None,
                       cancelled: Optional[Callable[[], bool]] = None,
                       progress: Optional[Callable[[float, str], None]] = None) -> tuple[Path, list[tuple[float, dict]]]:
    """OpenFOAM polar: mesh once, then one solver run per angle of attack on a copy of that mesh.

    Only the inflow direction (0/U and the far-field patches) and the force directions change with α, so
    the snappyHexMesh mesh is reused: each extra point costs one solver run (no meshing, no VTK export).
    Returns the base case directory and ``[(alpha, coefficients), ...]``."""
    base_cond = settings.conditions
    first = replace(settings, conditions=replace(base_cond, alpha_deg=float(alphas[0])))
    base = write_case(model, first)
    n = len(alphas)
    if progress is not None:          # one meshing run + n solver runs on one overall progress bar
        runner._on_progress = lambda f, m: progress(f / (n + 1), f"Mesh: {m}")
    runner.run_stages(base, first, mesh_only=True)
    points: list[tuple[float, dict]] = []
    for k, a in enumerate(alphas):
        if runner.aborted or (cancelled is not None and cancelled()):
            raise OpenFOAMAborted("Polar sweep aborted by the user.")
        st = replace(settings, conditions=replace(base_cond, alpha_deg=float(a)))
        lay = write_case(model, st, base.case_dir.parent / f"{base.case_dir.name}_a{a:+06.2f}".replace("+", "p")
                         .replace("-", "m"))
        shutil.copytree(base.case_dir / "constant" / "polyMesh", lay.case_dir / "constant" / "polyMesh",
                        dirs_exist_ok=True)
        if progress is not None:
            runner._on_progress = lambda f, m, k=k, a=a: progress((k + 1 + f) / (n + 1), f"α {a:g}° ({k + 1}/{n}): {m}")
        rep = runner.run_stages(lay, st, mesh_only=False, solve_only=True)
        coeffs = read_force_coefficients(lay.case_dir) or dict(rep.coefficients)
        points.append((float(a), coeffs))
        if on_point is not None:
            on_point(k, float(a), coeffs)
    return base.case_dir, points


def run_openfoam_case(model: GeometryModel, settings: CaseSettings, env: OpenFOAMEnvironment,
                      runner: OpenFOAMRunner, case_dir: Optional[Path] = None) -> SolverResult:
    """Convenience: write the case, run it with ``runner``, and load the result."""
    layout = write_case(model, settings, case_dir)
    report = runner.run(layout, settings)
    return load_openfoam_result(model, layout, settings, report)
