"""Input sanitisation, mesh-integrity checks and safe numerical helpers.

Every solver entry point funnels user input through this module so that
physically meaningless values (zero airspeed, NaN, inverted sweep ranges,
degenerate triangles ...) are clamped or rejected *before* they reach the
numerical core. Contains no Qt dependency.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from config.settings import (
    MAX_MESH_FILE_MB, MAX_POLAR_POINTS, MAX_TRIANGLES_HARD, MAX_TRIANGLES_WARN,
    MM_DETECTION_LENGTH, PARAM_LIMITS, SUPPORTED_MESH_EXTENSIONS, FlowConditions,
)

if TYPE_CHECKING:  # pragma: no cover
    import trimesh

EPS: Final[float] = 1e-12
Severity = Literal["info", "warning", "error"]


# ---------------------------------------------------------------------------
# Safe math
# ---------------------------------------------------------------------------
def clamp(value: float, lo: float, hi: float, default: Optional[float] = None) -> float:
    """Clamp ``value`` to ``[lo, hi]``; non-finite or non-numeric input yields ``default`` (or ``lo``)."""
    if lo > hi:
        lo, hi = hi, lo
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        v = lo if default is None else float(default)
    return min(max(v, lo), hi)


def safe_div(num: float, den: float, default: float = 0.0, eps: float = EPS) -> float:
    """Divide, returning ``default`` when the denominator is ~0 or any input is non-finite."""
    if not (math.isfinite(num) and math.isfinite(den)) or abs(den) < eps:
        return default
    return num / den


def safe_sqrt(x: float) -> float:
    """Square root that maps negatives / NaN to 0."""
    return math.sqrt(x) if math.isfinite(x) and x > 0.0 else 0.0


def lift_to_drag(cl: float, cd: float) -> float:
    """L/D ratio that never returns inf/NaN (0.0 when drag is ~0)."""
    return safe_div(cl, cd, default=0.0, eps=1e-9)


def reynolds_number(rho: float, velocity: float, length: float, mu: float) -> float:
    """Reynolds number ``rho V L / mu`` (0.0 when ``mu`` or inputs are invalid)."""
    return max(safe_div(rho * abs(velocity) * abs(length), mu, default=0.0), 0.0)


def dynamic_pressure(rho: float, velocity: float) -> float:
    """Dynamic pressure ``0.5 rho V^2`` in Pa."""
    return 0.5 * rho * velocity * velocity if math.isfinite(rho * velocity) else 0.0


def safe_normalize(v: ArrayLike, fallback: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> NDArray[np.float64]:
    """Normalise one vector; returns ``fallback`` for zero-length or non-finite input."""
    arr = np.asarray(v, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return np.asarray(fallback, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < EPS:
        return np.asarray(fallback, dtype=np.float64)
    return arr / norm


def safe_normalize_rows(a: ArrayLike, eps: float = EPS) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Normalise each row of an ``(N, k)`` array.

    Returns:
        ``(normalised, valid)`` where ``valid`` marks rows with finite, non-zero length;
        invalid rows are returned as zeros.
    """
    arr = np.asarray(a, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    valid = (np.isfinite(norms) & (norms > eps)).reshape(-1)
    out = np.zeros_like(arr)
    out[valid] = arr[valid] / norms[valid]
    return out, valid


def sanitize_array(a: ArrayLike, nan: float = 0.0, clip: Optional[tuple[float, float]] = None) -> NDArray[np.float64]:
    """Return a float64 copy with NaN/inf replaced (and optionally clipped)."""
    out = np.nan_to_num(np.array(a, dtype=np.float64, copy=True), nan=nan, posinf=np.finfo(np.float64).max / 1e6,
                        neginf=-np.finfo(np.float64).max / 1e6)
    if clip is not None:
        np.clip(out, clip[0], clip[1], out=out)
    return out


def parse_float(text: str, default: float, lo: float = -math.inf, hi: float = math.inf) -> float:
    """Parse user text (accepts a decimal comma) into a clamped float, falling back to ``default``."""
    try:
        value = float(str(text).strip().replace(",", "."))
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    return min(max(value, lo), hi)


# ---------------------------------------------------------------------------
# Validation result container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ValidationIssue:
    """One finding produced by a validator."""

    severity: Severity
    message: str
    field_name: Optional[str] = None


@dataclass
class ValidationResult:
    """Accumulates validation findings. ``ok`` is True when there are no errors."""

    issues: list[ValidationIssue] = field(default_factory=list)

    def info(self, message: str, field_name: Optional[str] = None) -> None:
        self.issues.append(ValidationIssue("info", message, field_name))

    def warn(self, message: str, field_name: Optional[str] = None) -> None:
        self.issues.append(ValidationIssue("warning", message, field_name))

    def error(self, message: str, field_name: Optional[str] = None) -> None:
        self.issues.append(ValidationIssue("error", message, field_name))

    def extend(self, other: "ValidationResult") -> None:
        """Append another result's issues."""
        self.issues.extend(other.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        """True when no errors were recorded."""
        return not self.errors

    def summary(self, max_items: int = 5) -> str:
        """Human-readable multi-line summary (errors first)."""
        ordered = self.errors + self.warnings + [i for i in self.issues if i.severity == "info"]
        lines = [f"- {i.message}" for i in ordered[:max_items]]
        if len(ordered) > max_items:
            lines.append(f"- ...and {len(ordered) - max_items} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Flow conditions & polar sweeps
# ---------------------------------------------------------------------------
_FLOW_FIELDS: Final[tuple[tuple[str, str], ...]] = (
    ("velocity_ms", "velocity"), ("alpha_deg", "alpha"), ("beta_deg", "beta"),
    ("altitude_m", "altitude"), ("temperature_offset_k", "temp_offset"),
)


def validate_flow_conditions(conditions: FlowConditions) -> tuple[FlowConditions, ValidationResult]:
    """Clamp flow conditions to physical ranges and report what changed.

    Returns:
        ``(safe_conditions, result)``. ``safe_conditions`` is always usable.
    """
    result = ValidationResult()
    values: dict[str, float] = {}
    for attr, key in _FLOW_FIELDS:
        raw = getattr(conditions, attr)
        limit = PARAM_LIMITS[key]
        safe = limit.clamp(raw)
        try:
            finite = math.isfinite(float(raw))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            result.warn(f"{limit.label} was not a valid number; using {safe:g} {limit.unit}.", key)
        elif safe != raw:
            result.warn(f"{limit.label} {raw:g} {limit.unit} is out of range; clamped to {safe:g} {limit.unit}.", key)
        values[attr] = safe
    safe_conditions = replace(conditions, **values)
    mach = safe_conditions.mach
    if mach > 0.85:
        result.warn(f"Mach {mach:.2f}: transonic/supersonic effects are not modelled by the Instant Engine.", "velocity")
    elif mach > 0.3:
        result.warn(f"Mach {mach:.2f}: compressibility corrections are applied (Prandtl-Glauert lift, "
                    "Kármán-Tsien surface pressures); they hold until the flow turns locally supersonic.",
                    "velocity")
    return safe_conditions, result


def validate_sweep(start: float, stop: float, step: float) -> tuple[list[float], ValidationResult]:
    """Build a safe list of angles of attack for a polar sweep.

    Swaps inverted ranges, enforces the minimum step, and caps the number of
    points at :data:`config.settings.MAX_POLAR_POINTS`.
    """
    result = ValidationResult()
    alpha = PARAM_LIMITS["alpha"]
    step_lim = PARAM_LIMITS["polar_step"]
    a0, a1 = alpha.clamp(start), alpha.clamp(stop)
    if a0 > a1:
        a0, a1 = a1, a0
        result.warn("Sweep start was greater than end; the range was swapped.", "polar_start")
    st = step_lim.clamp(step)
    if st != step:
        result.warn(f"Sweep step adjusted to {st:g} deg.", "polar_step")
    if a1 - a0 < 1e-9:
        return [round(a0, 6)], result
    n = int(math.floor((a1 - a0) / st + 1e-9)) + 1
    if n > MAX_POLAR_POINTS:
        st = (a1 - a0) / (MAX_POLAR_POINTS - 1)
        n = MAX_POLAR_POINTS
        result.warn(f"Too many sweep points; step increased to {st:.3f} deg.", "polar_step")
    angles = [round(a0 + i * st, 6) for i in range(n)]
    if angles[-1] < a1 - 1e-6 and len(angles) < MAX_POLAR_POINTS:
        angles.append(round(a1, 6))
    return angles, result


# ---------------------------------------------------------------------------
# Files & mesh integrity
# ---------------------------------------------------------------------------
def validate_mesh_path(path: str | Path) -> ValidationResult:
    """Check that ``path`` exists, is a supported, non-empty, reasonably sized file."""
    result = ValidationResult()
    try:
        p = Path(path)
        if not p.exists():
            result.error(f"File not found: {p.name}")
            return result
        if not p.is_file():
            result.error(f"Not a file: {p.name}")
            return result
        if p.suffix.lower() not in SUPPORTED_MESH_EXTENSIONS:
            result.error(f"Unsupported format '{p.suffix}'. Supported: {', '.join(SUPPORTED_MESH_EXTENSIONS)}")
        size_mb = p.stat().st_size / (1024 * 1024)
        if p.stat().st_size == 0:
            result.error("The file is empty.")
        elif size_mb > MAX_MESH_FILE_MB:
            result.error(f"File is {size_mb:.0f} MB; the limit is {MAX_MESH_FILE_MB:.0f} MB.")
    except OSError as exc:
        result.error(f"Cannot read file: {exc}")
    return result


@dataclass
class MeshReport:
    """Outcome of :func:`inspect_mesh`."""

    vertex_count: int = 0
    face_count: int = 0
    degenerate_faces: int = 0
    nonfinite_vertices: int = 0
    is_watertight: bool = False
    is_winding_consistent: bool = False
    volume: float = 0.0
    extents: tuple[float, float, float] = (0.0, 0.0, 0.0)
    result: ValidationResult = field(default_factory=ValidationResult)

    @property
    def is_usable(self) -> bool:
        """True when the mesh can be handed to a solver."""
        return self.result.ok


def inspect_mesh(mesh: "trimesh.Trimesh | None") -> MeshReport:
    """Run integrity checks on a mesh (never raises).

    Detects empty meshes, non-finite vertices, degenerate (zero-area) faces,
    open / non-manifold surfaces, extreme sizes and likely millimetre units.
    """
    report = MeshReport()
    res = report.result
    if mesh is None:
        res.error("No geometry loaded.")
        return report
    try:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
    except Exception as exc:  # noqa: BLE001
        res.error(f"Mesh data could not be read: {exc}")
        return report

    report.vertex_count, report.face_count = int(len(vertices)), int(len(faces))
    if report.vertex_count == 0 or report.face_count == 0:
        res.error("The mesh is empty (no triangles).")
        return report
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        res.error("The mesh is not a triangle mesh.")
        return report
    if faces.min() < 0 or faces.max() >= report.vertex_count:
        res.error("The mesh references vertices that do not exist (corrupt file).")
        return report

    finite = np.isfinite(vertices).all(axis=1)
    report.nonfinite_vertices = int((~finite).sum())
    if report.nonfinite_vertices:
        res.error(f"{report.nonfinite_vertices} vertices contain NaN/inf coordinates.")
        return report

    extents_arr = vertices.max(axis=0) - vertices.min(axis=0)
    report.extents = (float(extents_arr[0]), float(extents_arr[1]), float(extents_arr[2]))
    diag = float(np.linalg.norm(extents_arr))
    if diag < 1e-9:
        res.error("The geometry has zero size.")
        return report

    tri = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    report.degenerate_faces = int((areas <= 1e-10 * diag * diag).sum())
    if report.degenerate_faces:
        res.warn(f"{report.degenerate_faces} degenerate (zero-area) triangles found; they can be removed by Repair.")

    try:
        report.is_watertight = bool(mesh.is_watertight)
        report.is_winding_consistent = bool(mesh.is_winding_consistent)
        if report.is_watertight:
            report.volume = abs(float(mesh.volume))
    except Exception:  # noqa: BLE001 - trimesh can raise on pathological topology
        report.is_watertight = report.is_winding_consistent = False

    if not report.is_watertight:
        res.warn("The surface is not watertight: volume is unavailable and OpenFOAM meshing may fail. "
                 "Instant Engine results remain valid.")
    elif not report.is_winding_consistent:
        res.warn("Triangle winding is inconsistent; normals may be flipped. Use Repair.")

    if report.face_count > MAX_TRIANGLES_HARD:
        res.error(f"{report.face_count:,} triangles exceeds the hard limit of {MAX_TRIANGLES_HARD:,}; decimate the model.")
    elif report.face_count > MAX_TRIANGLES_WARN:
        res.warn(f"{report.face_count:,} triangles is very dense; the Instant Engine subsamples the surface automatically.")

    if max(report.extents) > MM_DETECTION_LENGTH:
        res.info(f"Longest dimension is {max(report.extents):.1f} units: this may be millimetres (use Convert mm to m).")
    if min(report.extents) < 1e-3 * max(report.extents):
        res.info("The geometry is extremely thin along one axis (treated as a flat plate).")
    return report


# ---------------------------------------------------------------------------
# NACA 4-digit codes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NACA4:
    """Parsed NACA 4-digit airfoil parameters (all as fractions of chord)."""

    code: str
    max_camber: float
    camber_position: float
    thickness: float


_NACA4_RE: Final[re.Pattern[str]] = re.compile(r"^\s*(?:NACA\s*)?(\d{4})\s*$", re.IGNORECASE)


def parse_naca4(text: str) -> NACA4:
    """Parse a NACA 4-digit code such as ``"2412"`` or ``"NACA 4415"``.

    Raises:
        ValueError: if the code is malformed, has zero thickness, or specifies
            camber with no camber position.
    """
    match = _NACA4_RE.match(text or "")
    if not match:
        raise ValueError("Enter exactly four digits, e.g. 2412.")
    code = match.group(1)
    m, p, t = int(code[0]) / 100.0, int(code[1]) / 10.0, int(code[2:]) / 100.0
    if t <= 0.0:
        raise ValueError("The last two digits (thickness) must not be 00.")
    if m > 0.0 and p <= 0.0:
        raise ValueError("A cambered airfoil needs a non-zero camber position (second digit).")
    return NACA4(code, m, p, t)


def validate_naca_code(text: str) -> tuple[Optional[NACA4], ValidationResult]:
    """Non-raising wrapper around :func:`parse_naca4`."""
    result = ValidationResult()
    try:
        return parse_naca4(text), result
    except ValueError as exc:
        result.error(str(exc), "naca_code")
        return None, result


# ---------------------------------------------------------------------------
# Misc sanitisation
# ---------------------------------------------------------------------------
_RESERVED_NAMES: Final[frozenset[str]] = frozenset(
    {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)})


def sanitize_filename(name: str, default: str = "untitled", max_len: int = 80) -> str:
    """Make ``name`` safe as a file/folder name on Windows, macOS and Linux."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name)).strip(" .")
    cleaned = re.sub(r"\s+", "_", cleaned)[:max_len].strip(" ._")
    if not cleaned or cleaned.split(".")[0].upper() in _RESERVED_NAMES:
        return default
    return cleaned


def clamp_dimension(value: float, lo: float = 0.01, hi: float = 1000.0, default: float = 1.0) -> float:
    """Clamp a physical length (chord, span) to a sane positive range."""
    return clamp(value, lo, hi, default)
