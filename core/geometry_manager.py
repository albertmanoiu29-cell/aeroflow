"""Geometry ingestion, conditioning, repair and procedural presets.

Design notes
------------
* :class:`GeometryModel` is **immutable** (read-only NumPy arrays). Every
  operation returns a *new* model, so worker threads can safely read a model
  while the GUI thread swaps in a new one. :class:`GeometryManager` holds the
  current model behind a lock, with undo.
* Importing never needs Qt. Loaders try ``trimesh`` first (and ``numpy-stl`` for
  STL), then fall back to the built-in NumPy parsers for STL / OBJ / OFF, so
  common formats load even when an optional library misbehaves. ``.vtk`` files
  use PyVista (imported lazily).
* Topology checks (watertight, winding, volume) are computed with NumPy edge
  counting, so they work identically with or without ``trimesh``.
* Presets are generated analytically (NACA 4-digit sections, bodies of
  revolution, tapered/swept wings) as closed, outward-oriented triangle meshes.

Axes: freestream along +X (nose at -X), span along Y, up is +Z.
"""

from __future__ import annotations

import logging
import math
import re
import struct
import threading
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Final, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from config.settings import (
    MAX_TRIANGLES_HARD, MM_DETECTION_LENGTH, GeometryPreset, ReferenceAreaMode,
)
from utils.validators import (
    MeshReport, NACA4, clamp_dimension, inspect_mesh, parse_naca4,
    validate_mesh_path,
)

log = logging.getLogger("aeroflow.core.geometry")
SUCCESS_LEVEL: Final[int] = 25  # custom level registered by utils.logger

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ProgressCallback = Callable[[float, str], None]


class GeometryError(Exception):
    """A user-facing geometry problem; the message is safe to show in the UI."""


# ---------------------------------------------------------------------------
# Low-level mesh math
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TopologyStats:
    """Edge-based topology summary of a triangle mesh."""

    boundary_edges: int
    nonmanifold_edges: int
    inconsistent_edges: int
    components: int

    @property
    def watertight(self) -> bool:
        """True when every edge is shared by exactly two triangles."""
        return self.boundary_edges == 0 and self.nonmanifold_edges == 0

    @property
    def winding_consistent(self) -> bool:
        """True when all shared edges are traversed in opposite directions."""
        return self.inconsistent_edges == 0 and self.nonmanifold_edges == 0


def compute_topology(faces: IntArray, n_vertices: int) -> TopologyStats:
    """Count boundary / non-manifold / inconsistently wound edges and components."""
    if len(faces) == 0:
        return TopologyStats(0, 0, 0, 0)
    a = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    b = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo.astype(np.int64) * np.int64(n_vertices) + hi.astype(np.int64)
    direction = np.where(a < b, 1.0, -1.0)
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    signed = np.bincount(inverse, weights=direction)
    boundary = int((counts == 1).sum())
    nonmanifold = int((counts > 2).sum())
    inconsistent = int(((counts == 2) & (signed != 0.0)).sum())
    graph = coo_matrix((np.ones(len(lo)), (lo, hi)), shape=(n_vertices, n_vertices))
    used = np.zeros(n_vertices, dtype=bool)
    used[np.unique(faces)] = True
    n_comp, labels = connected_components(graph, directed=False)
    components = int(len(np.unique(labels[used])))
    return TopologyStats(boundary, nonmanifold, inconsistent, components)


def face_vectors(vertices: FloatArray, faces: IntArray) -> FloatArray:
    """Return ``cross(e1, e2)`` per face (length = 2 x area, direction = normal)."""
    tri = vertices[faces]
    return np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])


def signed_volume(vertices: FloatArray, faces: IntArray) -> float:
    """Signed volume by the divergence theorem (positive for outward-wound closed meshes)."""
    if len(faces) == 0:
        return 0.0
    origin = vertices.mean(axis=0)
    tri = vertices[faces] - origin
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def sample_surface(vertices: FloatArray, faces: IntArray, n: int,
                   seed: int = 7) -> tuple[FloatArray, FloatArray, IntArray]:
    """Area-weighted, stratified random samples on a triangle surface.

    Returns:
        ``(points, unit_normals, face_ids)``; deterministic for a given ``seed``.

    Raises:
        GeometryError: if the surface has no area.
    """
    tri = vertices[faces]
    vec = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(vec, axis=1)
    total = float(area.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise GeometryError("The surface has zero area.")
    cdf = np.cumsum(area)
    cdf /= cdf[-1]
    rng = np.random.default_rng(seed)
    u = (np.arange(n) + rng.random(n)) / n
    fid = np.minimum(np.searchsorted(cdf, u), len(faces) - 1)
    s = np.sqrt(rng.random(n))
    r2 = rng.random(n)
    wa, wb, wc = 1.0 - s, s * (1.0 - r2), s * r2
    pts = wa[:, None] * tri[fid, 0] + wb[:, None] * tri[fid, 1] + wc[:, None] * tri[fid, 2]
    denom = np.maximum(2.0 * area[fid], 1e-300)[:, None]
    nrm = vec[fid] / denom
    return pts, nrm, fid


def _rotation_matrix(axis: Sequence[float], angle_rad: float) -> FloatArray:
    """Rodrigues rotation matrix about ``axis`` (need not be normalised)."""
    a = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(a)
    if n < 1e-15:
        return np.eye(3)
    x, y, z = a / n
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    t = 1.0 - c
    return np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                     [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                     [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])


_AXES: Final[dict[str, FloatArray]] = {
    "+X": np.array([1.0, 0.0, 0.0]), "-X": np.array([-1.0, 0.0, 0.0]),
    "+Y": np.array([0.0, 1.0, 0.0]), "-Y": np.array([0.0, -1.0, 0.0]),
    "+Z": np.array([0.0, 0.0, 1.0]), "-Z": np.array([0.0, 0.0, -1.0]),
}


# ---------------------------------------------------------------------------
# Immutable geometry model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProjectedAreas:
    """Silhouette areas of a body (m^2)."""

    planform: float   # projected onto the XY plane (viewed from +Z)
    frontal: float    # projected onto the YZ plane (viewed from +X)
    side: float       # projected onto the XZ plane (viewed from +Y)


class GeometryModel:
    """An immutable triangle surface with cached derived quantities.

    The vertex / face arrays are read-only. Use the module-level operations
    (``center_model``, ``scale_model`` ...) to derive new models.
    """

    def __init__(self, vertices: Any, faces: Any, name: str = "geometry",
                 source: Optional[Path] = None, history: Sequence[str] = (),
                 reference_mode: ReferenceAreaMode = ReferenceAreaMode.PLANFORM) -> None:
        try:
            v = np.array(vertices, dtype=np.float64, copy=True)
            f = np.array(faces, dtype=np.int64, copy=True)
        except (TypeError, ValueError) as exc:
            raise GeometryError(f"Mesh data is not numeric: {exc}") from exc
        if v.ndim != 2 or v.shape[1] != 3 or len(v) == 0:
            raise GeometryError("The mesh has no valid vertices.")
        if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0:
            raise GeometryError("The mesh has no triangles.")
        if f.min() < 0 or f.max() >= len(v):
            raise GeometryError("The mesh references vertices that do not exist (corrupt file).")
        if not np.isfinite(v).all():
            raise GeometryError("The mesh contains NaN or infinite coordinates.")
        v.setflags(write=False)
        f.setflags(write=False)
        self.vertices: FloatArray = v
        self.faces: IntArray = f
        self.name = name
        self.source = source
        self.history: tuple[str, ...] = tuple(history)
        self.reference_mode = reference_mode

    # -- basic derived quantities ---------------------------------------
    @property
    def triangle_count(self) -> int:
        """Number of triangles."""
        return int(len(self.faces))

    @property
    def vertex_count(self) -> int:
        """Number of vertices."""
        return int(len(self.vertices))

    @cached_property
    def bounds_min(self) -> FloatArray:
        """Lower corner of the axis-aligned bounding box."""
        return self.vertices.min(axis=0)

    @cached_property
    def bounds_max(self) -> FloatArray:
        """Upper corner of the axis-aligned bounding box."""
        return self.vertices.max(axis=0)

    @cached_property
    def extents(self) -> FloatArray:
        """Bounding-box size ``(Lx, Ly, Lz)``."""
        return self.bounds_max - self.bounds_min

    @cached_property
    def bbox_center(self) -> FloatArray:
        """Bounding-box centre."""
        return 0.5 * (self.bounds_min + self.bounds_max)

    @cached_property
    def face_vecs(self) -> FloatArray:
        """``cross(e1, e2)`` per face."""
        return face_vectors(self.vertices, self.faces)

    @cached_property
    def face_areas(self) -> FloatArray:
        """Triangle areas."""
        return 0.5 * np.linalg.norm(self.face_vecs, axis=1)

    @cached_property
    def face_normals(self) -> FloatArray:
        """Unit triangle normals (zero for degenerate triangles)."""
        denom = 2.0 * self.face_areas
        out = np.zeros_like(self.face_vecs)
        ok = denom > 1e-300
        out[ok] = self.face_vecs[ok] / denom[ok, None]
        return out

    @cached_property
    def surface_area(self) -> float:
        """Total (wetted) surface area."""
        return float(self.face_areas.sum())

    @cached_property
    def vertex_normals(self) -> FloatArray:
        """Area-weighted unit vertex normals."""
        acc = np.zeros_like(self.vertices)
        for k in range(3):
            np.add.at(acc, self.faces[:, k], self.face_vecs)
        norm = np.linalg.norm(acc, axis=1, keepdims=True)
        return np.divide(acc, norm, out=np.zeros_like(acc), where=norm > 1e-300)

    @cached_property
    def topology(self) -> TopologyStats:
        """Edge-based topology statistics."""
        return compute_topology(self.faces, self.vertex_count)

    @property
    def is_watertight(self) -> bool:
        """True when the surface is closed."""
        return self.topology.watertight

    @property
    def is_winding_consistent(self) -> bool:
        """True when triangle winding is consistent."""
        return self.topology.winding_consistent

    @cached_property
    def signed_volume_value(self) -> float:
        """Signed enclosed volume (negative => normals point inward)."""
        return signed_volume(self.vertices, self.faces)

    @property
    def volume(self) -> float:
        """Enclosed volume in m^3 (0.0 unless the mesh is watertight)."""
        return abs(self.signed_volume_value) if self.is_watertight else 0.0

    @cached_property
    def volume_centroid(self) -> FloatArray:
        """Centroid of the enclosed volume (falls back to the bounding-box centre)."""
        if not self.is_watertight or abs(self.signed_volume_value) < 1e-12:
            return self.bbox_center.copy()
        tri = self.vertices[self.faces]
        origin = self.vertices.mean(axis=0)
        t = tri - origin
        vol6 = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2]))
        cen = (t[:, 0] + t[:, 1] + t[:, 2]) / 4.0
        total = vol6.sum()
        if abs(total) < 1e-300:
            return self.bbox_center.copy()
        return origin + (vol6[:, None] * cen).sum(axis=0) / total

    @cached_property
    def projected_areas(self) -> ProjectedAreas:
        """Silhouette areas, estimated by rasterising dense surface samples."""
        return _projected_areas(self)

    @cached_property
    def info(self) -> "GeometryInfo":
        """Aerodynamic reference quantities (cached)."""
        return compute_info(self)

    # -- derivation helpers ---------------------------------------------
    def derive(self, vertices: Any, op: str, faces: Optional[Any] = None) -> "GeometryModel":
        """Create a new model sharing metadata, with an extra history entry."""
        return GeometryModel(vertices, self.faces if faces is None else faces, self.name,
                             self.source, (*self.history, op), self.reference_mode)

    def with_reference_mode(self, mode: ReferenceAreaMode) -> "GeometryModel":
        """Return a copy using another reference-area convention (shares cached silhouette areas)."""
        new = GeometryModel(self.vertices, self.faces, self.name, self.source, self.history, mode)
        for key in ("projected_areas", "topology", "signed_volume_value"):
            if key in self.__dict__:
                new.__dict__[key] = self.__dict__[key]
        return new

    def renamed(self, name: str) -> "GeometryModel":
        """Return a copy with a new display name."""
        return GeometryModel(self.vertices, self.faces, name, self.source, self.history,
                             self.reference_mode)

    # -- conversions -----------------------------------------------------
    def to_pyvista(self) -> Any:
        """Convert to ``pyvista.PolyData`` (call on the GUI thread)."""
        import pyvista as pv

        n = self.triangle_count
        flat = np.column_stack([np.full(n, 3, dtype=np.int64), self.faces]).ravel()
        return pv.PolyData(self.vertices.astype(np.float32), flat)

    def to_trimesh(self) -> Any:
        """Convert to ``trimesh.Trimesh`` (raises ``GeometryError`` if trimesh is missing)."""
        try:
            import trimesh
        except ImportError as exc:
            raise GeometryError("The 'trimesh' package is not installed.") from exc
        return trimesh.Trimesh(vertices=self.vertices.copy(), faces=self.faces.copy(), process=False)

    def export_stl(self, path: str | Path, binary: bool = True) -> Path:
        """Write the surface as an STL file (used by the OpenFOAM bridge)."""
        return write_stl(self, path, binary)


# ---------------------------------------------------------------------------
# Aerodynamic reference extraction
# ---------------------------------------------------------------------------
def _projected_areas(model: GeometryModel) -> ProjectedAreas:
    """Rasterise dense surface samples onto the three coordinate planes."""
    total = model.surface_area
    n = int(min(600_000, max(200_000, 60 * model.triangle_count)))
    pts, _, _ = sample_surface(model.vertices, model.faces, n, seed=11)
    cell_target = math.sqrt(12.0 * total / n)  # >= ~12 samples per cell on average

    def raster(a: int, b: int) -> float:
        """Occupied-cell area at cell size h and 2h, combined by Richardson extrapolation.

        Counting occupied cells overestimates the area by an amount proportional to the cell
        size (partially covered boundary cells), so ``2 A(h) - A(2h)`` cancels the first-order term.
        """
        p0, p1 = pts[:, a], pts[:, b]
        lo0, lo1 = float(p0.min()), float(p1.min())
        e0, e1 = float(p0.max()) - lo0, float(p1.max()) - lo1
        cell = max(cell_target, max(e0, e1, 1e-12) / 600.0)
        i = np.minimum(((p0 - lo0) / cell).astype(np.int64), int(e0 / cell))
        j = np.minimum(((p1 - lo1) / cell).astype(np.int64), int(e1 / cell))

        def covered(ii: IntArray, jj: IntArray, size: float) -> float:
            """Area of occupied cells, weighting bounding-box edge cells by the fraction inside the box."""
            width = int(jj.max()) + 1
            ids = np.unique(ii * width + jj)
            iu, ju = ids // width, ids % width
            wi = np.minimum(1.0, (e0 - iu * size) / size)
            wj = np.minimum(1.0, (e1 - ju * size) / size)
            return float((np.clip(wi, 0.0, 1.0) * np.clip(wj, 0.0, 1.0)).sum() * size * size)

        fine = covered(i, j, cell)
        coarse = covered(i // 2, j // 2, 2.0 * cell)
        return float(min(max(2.0 * fine - coarse, 0.8 * fine), fine))

    return ProjectedAreas(planform=raster(0, 1), frontal=raster(1, 2), side=raster(0, 2))


@dataclass(frozen=True)
class GeometryInfo:
    """Aerodynamic reference quantities of a model (all SI units)."""

    name: str
    triangle_count: int
    vertex_count: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    chord_m: float
    span_m: float
    height_m: float
    planform_area_m2: float
    frontal_area_m2: float
    reference_area_m2: float
    reference_mode: ReferenceAreaMode
    aspect_ratio: float
    wetted_area_m2: float
    volume_m3: float
    is_watertight: bool
    is_winding_consistent: bool
    moment_ref_point: tuple[float, float, float]
    possibly_millimetres: bool
    report: MeshReport

    @property
    def reference_length_m(self) -> float:
        """Characteristic length (the X extent, i.e. along the freestream)."""
        return self.chord_m

    def rows(self) -> list[tuple[str, str]]:
        """``(label, value)`` rows for the Geometry Info tab."""
        return [
            ("Name", self.name),
            ("Length / chord  (X)", f"{self.chord_m:.4g} m"),
            ("Span  (Y)", f"{self.span_m:.4g} m"),
            ("Height  (Z)", f"{self.height_m:.4g} m"),
            ("Planform area", f"{self.planform_area_m2:.4g} m²"),
            ("Frontal area", f"{self.frontal_area_m2:.4g} m²"),
            ("Reference area in use", f"{self.reference_area_m2:.4g} m²"),
            ("Aspect ratio", f"{self.aspect_ratio:.3g}"),
            ("Wetted area", f"{self.wetted_area_m2:.4g} m²"),
            ("Volume", f"{self.volume_m3:.4g} m³" if self.is_watertight else "n/a (not watertight)"),
            ("Triangles", f"{self.triangle_count:,}"),
            ("Vertices", f"{self.vertex_count:,}"),
            ("Watertight", "Yes" if self.is_watertight else "No"),
        ]


def compute_info(model: GeometryModel) -> GeometryInfo:
    """Compute :class:`GeometryInfo` for ``model``."""
    ext = model.extents
    areas = model.projected_areas
    chord, span, height = (max(float(x), 1e-9) for x in ext)
    planform = max(areas.planform, 1e-12)
    frontal = max(areas.frontal, 1e-12)
    ref = planform if model.reference_mode is ReferenceAreaMode.PLANFORM else frontal
    ref_point = (float(model.bounds_min[0] + 0.25 * chord), float(model.bbox_center[1]),
                 float(model.bbox_center[2]))
    return GeometryInfo(
        name=model.name, triangle_count=model.triangle_count, vertex_count=model.vertex_count,
        bounds_min=tuple(float(x) for x in model.bounds_min),
        bounds_max=tuple(float(x) for x in model.bounds_max),
        chord_m=chord, span_m=span, height_m=height, planform_area_m2=planform,
        frontal_area_m2=frontal, reference_area_m2=ref, reference_mode=model.reference_mode,
        aspect_ratio=span * span / planform, wetted_area_m2=model.surface_area,
        volume_m3=model.volume, is_watertight=model.is_watertight,
        is_winding_consistent=model.is_winding_consistent, moment_ref_point=ref_point,
        possibly_millimetres=float(ext.max()) > MM_DETECTION_LENGTH, report=inspect_mesh(model),
    )


# ---------------------------------------------------------------------------
# File readers (NumPy fallbacks)
# ---------------------------------------------------------------------------
def _merge_vertices(vertices: FloatArray, faces: IntArray) -> tuple[FloatArray, IntArray]:
    """Merge coincident vertices (relative tolerance 1e-7 of the bounding diagonal)."""
    diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    tol = max(diag * 1e-7, 1e-12)
    keys = np.round(vertices / tol).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    return vertices[first], inverse[faces]


def _read_stl_native(path: Path) -> tuple[FloatArray, IntArray]:
    """Parse binary or ASCII STL with NumPy."""
    data = path.read_bytes()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data) and n > 0:
            dt = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])
            arr = np.frombuffer(data, dtype=dt, count=n, offset=84)
            tri = arr["v"].astype(np.float64).reshape(-1, 3)
            return _merge_vertices(tri, np.arange(len(tri)).reshape(-1, 3))
    found = re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", data, flags=re.IGNORECASE)
    if not found or len(found) % 3:
        raise GeometryError("The STL file is corrupt or contains no triangles.")
    try:
        tri = np.asarray(found).astype(np.float64)
    except ValueError as exc:
        raise GeometryError("The STL file contains invalid numbers.") from exc
    return _merge_vertices(tri, np.arange(len(tri)).reshape(-1, 3))


def _read_stl_numpy_stl(path: Path) -> tuple[FloatArray, IntArray]:
    """Parse STL with ``numpy-stl`` (secondary reader)."""
    from stl import mesh as stl_mesh  # type: ignore[import-not-found]

    m = stl_mesh.Mesh.from_file(str(path))
    tri = np.asarray(m.vectors, dtype=np.float64).reshape(-1, 3)
    return _merge_vertices(tri, np.arange(len(tri)).reshape(-1, 3))


def _read_obj_native(path: Path) -> tuple[FloatArray, IntArray]:
    """Parse Wavefront OBJ (vertices and faces only; polygons are fan-triangulated)."""
    verts: list[tuple[float, float, float]] = []
    tris: list[tuple[int, int, int]] = []
    with path.open("r", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                if len(p) >= 4:
                    verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    i = int(tok.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                for k in range(1, len(idx) - 1):
                    tris.append((idx[0], idx[k], idx[k + 1]))
    if not verts or not tris:
        raise GeometryError("The OBJ file contains no triangles.")
    return np.asarray(verts, dtype=np.float64), np.asarray(tris, dtype=np.int64)


def _read_off_native(path: Path) -> tuple[FloatArray, IntArray]:
    """Parse OFF (vertices and polygon faces; polygons are fan-triangulated)."""
    lines = [ln.strip() for ln in path.read_text(errors="replace").splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not lines or not lines[0].upper().startswith("OFF"):
        raise GeometryError("Not a valid OFF file.")
    head = lines[0][3:].split()
    pos = 1
    if not head:
        head = lines[pos].split()
        pos += 1
    nv, nf = int(head[0]), int(head[1])
    verts = np.array([[float(x) for x in lines[pos + i].split()[:3]] for i in range(nv)])
    tris: list[tuple[int, int, int]] = []
    for i in range(nf):
        p = lines[pos + nv + i].split()
        k = int(p[0])
        idx = [int(x) for x in p[1:1 + k]]
        for j in range(1, k - 1):
            tris.append((idx[0], idx[j], idx[j + 1]))
    if not tris:
        raise GeometryError("The OFF file contains no triangles.")
    return verts, np.asarray(tris, dtype=np.int64)


def _read_trimesh(path: Path) -> tuple[FloatArray, IntArray]:
    """Load any trimesh-supported format; scenes are concatenated."""
    import trimesh  # type: ignore[import-not-found]

    try:
        obj = trimesh.load(str(path), force="mesh")
    except TypeError:
        obj = trimesh.load(str(path))
    if hasattr(obj, "geometry"):  # a Scene
        parts = [g for g in obj.geometry.values() if hasattr(g, "faces") and len(g.faces)]
        if not parts:
            raise GeometryError("The file contains no triangle meshes.")
        obj = trimesh.util.concatenate(parts)
    if not hasattr(obj, "faces") or len(obj.faces) == 0:
        raise GeometryError("The file contains no triangle surface.")
    return np.asarray(obj.vertices, dtype=np.float64), np.asarray(obj.faces, dtype=np.int64)


def _read_vtk(path: Path) -> tuple[FloatArray, IntArray]:
    """Load a legacy/XML VTK file through PyVista and extract its triangulated surface."""
    import pyvista as pv

    data = pv.read(str(path))
    surf = data.extract_surface().triangulate().clean()
    if surf.n_cells == 0:
        raise GeometryError("The VTK file contains no surface cells.")
    faces = np.asarray(surf.faces).reshape(-1, 4)[:, 1:]
    return np.asarray(surf.points, dtype=np.float64), faces.astype(np.int64)


@dataclass
class LoadResult:
    """Outcome of :func:`load_geometry_file`."""

    model: GeometryModel
    report: MeshReport
    loader: str
    suggestions: list[str] = field(default_factory=list)


def load_geometry_file(path: str | Path, progress: Optional[ProgressCallback] = None) -> LoadResult:
    """Load a geometry file (safe to call from a worker thread).

    Args:
        path: ``.stl``, ``.obj``, ``.ply``, ``.off`` or ``.vtk``.
        progress: Optional ``(fraction, message)`` callback.

    Raises:
        GeometryError: with a user-presentable message for any failure.
    """
    def tick(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)

    p = Path(path)
    check = validate_mesh_path(p)
    if not check.ok:
        raise GeometryError(check.errors[0].message)
    ext = p.suffix.lower()
    tick(0.05, f"Reading {p.name}")

    attempts: list[tuple[str, Callable[[Path], tuple[FloatArray, IntArray]]]] = []
    if ext == ".vtk":
        attempts = [("pyvista", _read_vtk)]
    else:
        attempts = [("trimesh", _read_trimesh)]
        if ext == ".stl":
            attempts += [("numpy-stl", _read_stl_numpy_stl), ("built-in STL", _read_stl_native)]
        elif ext == ".obj":
            attempts += [("built-in OBJ", _read_obj_native)]
        elif ext == ".off":
            attempts += [("built-in OFF", _read_off_native)]

    errors: list[str] = []
    result: Optional[tuple[FloatArray, IntArray]] = None
    used = ""
    for label, reader in attempts:
        try:
            result = reader(p)
            used = label
            break
        except GeometryError as exc:
            errors.append(str(exc))
        except ImportError:
            errors.append(f"{label} is not installed")
        except Exception as exc:  # noqa: BLE001 - third-party loaders raise many types
            errors.append(f"{label}: {exc}")
            log.debug("Loader %s failed for %s: %s", label, p.name, exc)
    if result is None:
        detail = next((e for e in errors if "not installed" not in e), errors[0] if errors else "unknown error")
        raise GeometryError(f"Could not read '{p.name}': {detail}")

    vertices, faces = result
    tick(0.55, "Building mesh")
    if len(faces) > MAX_TRIANGLES_HARD:
        raise GeometryError(f"{len(faces):,} triangles exceeds the limit of {MAX_TRIANGLES_HARD:,}; decimate the model first.")
    if used in ("trimesh", "pyvista"):
        vertices, faces = _merge_vertices(vertices, faces)
    model = GeometryModel(vertices, faces, name=p.stem, source=p, history=(f"Loaded {p.name}",))
    tick(0.8, "Checking mesh")
    report = inspect_mesh(model)
    suggestions: list[str] = []
    if report.degenerate_faces or not model.is_watertight or not model.is_winding_consistent:
        suggestions.append("Run Repair to clean up the mesh.")
    sug = suggest_scale(model)
    if sug is not None:
        suggestions.append(sug.reason)
    tick(1.0, "Done")
    log.log(SUCCESS_LEVEL, "Loaded %s: %s triangles (%s)", p.name, f"{model.triangle_count:,}", used)
    return LoadResult(model, report, used, suggestions)


def write_stl(model: GeometryModel, path: str | Path, binary: bool = True) -> Path:
    """Write ``model`` as STL (binary by default). Returns the written path."""
    target = Path(path)
    tri = model.vertices[model.faces]
    normals = model.face_normals
    target.parent.mkdir(parents=True, exist_ok=True)
    if binary:
        dt = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])
        arr = np.zeros(len(tri), dtype=dt)
        arr["n"], arr["v"] = normals, tri
        with target.open("wb") as fh:
            fh.write(b"AeroFlow Studio".ljust(80, b" "))
            fh.write(struct.pack("<I", len(tri)))
            fh.write(arr.tobytes())
    else:
        with target.open("w") as fh:
            fh.write("solid geometry\n")
            for n, t in zip(normals, tri):
                fh.write(f"facet normal {n[0]:.9e} {n[1]:.9e} {n[2]:.9e}\n outer loop\n")
                for q in t:
                    fh.write(f"  vertex {q[0]:.9e} {q[1]:.9e} {q[2]:.9e}\n")
                fh.write(" endloop\nendfacet\n")
            fh.write("endsolid geometry\n")
    return target


# ---------------------------------------------------------------------------
# Conditioning operations (each returns a new model)
# ---------------------------------------------------------------------------
def center_model(model: GeometryModel, mode: str = "bbox") -> GeometryModel:
    """Translate so the bounding-box centre (``"bbox"``) or volume centroid (``"mass"``) is at the origin."""
    c = model.volume_centroid if mode == "mass" else model.bbox_center
    return model.derive(model.vertices - c, "Centered")


def transform_model(model: GeometryModel, matrix: FloatArray, op: str, recenter: bool = True) -> GeometryModel:
    """Apply a 3x3 matrix about the origin (flipping faces if it mirrors the mesh)."""
    m = np.asarray(matrix, dtype=np.float64)
    verts = model.vertices @ m.T
    faces = model.faces
    if np.linalg.det(m) < 0:
        faces = faces[:, [0, 2, 1]]
    new = model.derive(verts, op, faces)
    return center_model(new) if recenter else new


def rotate_model(model: GeometryModel, axis: str, degrees: float) -> GeometryModel:
    """Rotate about the X, Y or Z axis through the model centre by ``degrees``."""
    ax = _AXES.get("+" + axis.strip().upper().lstrip("+-"))
    if ax is None:
        raise GeometryError(f"Unknown rotation axis '{axis}'.")
    centered = center_model(model)
    return transform_model(centered, _rotation_matrix(ax, math.radians(degrees)),
                           f"Rotated {degrees:g}° about {axis.upper()}")


def rotate_90(model: GeometryModel, axis: str, turns: int = 1) -> GeometryModel:
    """Rotate in 90° increments (``turns`` may be negative)."""
    return rotate_model(model, axis, 90.0 * int(turns))


def align_nose(model: GeometryModel, nose_direction: str) -> GeometryModel:
    """Rotate the model so the direction its nose currently points becomes -X.

    Args:
        nose_direction: One of ``"+X" "-X" "+Y" "-Y" "+Z" "-Z"``: the axis the nose points along now.
    """
    key = nose_direction.strip().upper()
    if key not in _AXES:
        raise GeometryError(f"Unknown nose direction '{nose_direction}'.")
    u, target = _AXES[key], _AXES["-X"]
    if np.allclose(u, target):
        return model
    if np.allclose(u, -target):
        rot = _rotation_matrix(_AXES["+Z"], math.pi)
    else:
        axis = np.cross(u, target)
        rot = _rotation_matrix(axis, math.acos(float(np.clip(u @ target, -1.0, 1.0))))
    return transform_model(center_model(model), rot, f"Nose aligned from {key} to -X")


def auto_orient(model: GeometryModel) -> GeometryModel:
    """Heuristic: longest axis -> X, then second longest -> Y (a proper rotation).

    Suitable for fuselages, rockets and cars. Do not use on isolated wings,
    whose longest dimension is the span.
    """
    order = np.argsort(-model.extents)       # source axes sorted by size
    perm = np.zeros((3, 3))
    for new_axis, src in enumerate(order):
        perm[new_axis, src] = 1.0
    if np.linalg.det(perm) < 0:
        perm[2] *= -1.0                       # keep a proper rotation
    return transform_model(center_model(model), perm, "Auto-oriented (longest axis -> X)")


def scale_model(model: GeometryModel, factor: float, recenter: bool = True) -> GeometryModel:
    """Uniformly scale about the origin by ``factor``."""
    if not math.isfinite(factor) or factor <= 0.0:
        raise GeometryError("Scale factor must be a positive number.")
    new = model.derive(model.vertices * factor, f"Scaled x{factor:g}")
    return center_model(new) if recenter else new


def convert_mm_to_m(model: GeometryModel) -> GeometryModel:
    """Convert a millimetre model to metres."""
    return scale_model(model, 1e-3)


def scale_to_length(model: GeometryModel, target_length_m: float) -> GeometryModel:
    """Scale so the X extent (length / chord) equals ``target_length_m``."""
    return scale_model(model, clamp_dimension(target_length_m, 1e-4, 1e5) / max(model.extents[0], 1e-12))


def scale_to_span(model: GeometryModel, target_span_m: float) -> GeometryModel:
    """Scale so the Y extent (span) equals ``target_span_m``."""
    return scale_model(model, clamp_dimension(target_span_m, 1e-4, 1e5) / max(model.extents[1], 1e-12))


@dataclass(frozen=True)
class ScaleSuggestion:
    """A suggested unit conversion."""

    factor: float
    reason: str


def suggest_scale(model: GeometryModel) -> Optional[ScaleSuggestion]:
    """Suggest millimetre -> metre conversion when the longest dimension exceeds the threshold."""
    longest = float(model.extents.max())
    if longest > MM_DETECTION_LENGTH:
        return ScaleSuggestion(1e-3, f"Longest dimension is {longest:.0f} units: this looks like millimetres. "
                                     f"Use 'Convert mm → m'.")
    return None


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------
@dataclass
class RepairReport:
    """What :func:`repair_model` changed."""

    merged_vertices: int = 0
    removed_degenerate: int = 0
    removed_duplicate: int = 0
    removed_unreferenced: int = 0
    flipped_components: int = 0
    holes_filled: int = 0
    watertight_before: bool = False
    watertight_after: bool = False
    messages: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """One-line summary."""
        parts = [(self.merged_vertices, "vertices merged"), (self.removed_degenerate, "degenerate faces removed"),
                 (self.removed_duplicate, "duplicate faces removed"), (self.flipped_components, "parts re-oriented"),
                 (self.holes_filled, "faces added to fill holes")]
        done = [f"{n} {t}" for n, t in parts if n]
        state = "watertight" if self.watertight_after else "still not watertight"
        return ("; ".join(done) if done else "No changes were needed") + f" - mesh is {state}."


def repair_model(model: GeometryModel) -> tuple[GeometryModel, RepairReport]:
    """Merge duplicate vertices, drop degenerate / duplicate faces, orient normals outward.

    Optionally uses ``trimesh`` to fix inconsistent winding and fill small holes
    (skipped silently when unavailable). Never raises for ordinary meshes.
    """
    rep = RepairReport(watertight_before=model.is_watertight)
    v = model.vertices.copy()
    f = model.faces.copy()
    diag = float(np.linalg.norm(model.extents))

    n0 = len(v)
    v, f = _merge_vertices(v, f)
    rep.merged_vertices = n0 - len(v)

    area = 0.5 * np.linalg.norm(face_vectors(v, f), axis=1)
    distinct = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    keep = distinct & (area > 1e-12 * diag * diag)
    rep.removed_degenerate = int((~keep).sum())
    f = f[keep]

    if len(f):
        _, first = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
        if len(first) != len(f):
            rep.removed_duplicate = len(f) - len(first)
            f = f[np.sort(first)]
    if len(f) == 0:
        raise GeometryError("Repair removed every triangle; the mesh is unusable.")

    try:
        import trimesh  # type: ignore[import-not-found]
        from trimesh import repair as tm_repair  # type: ignore[import-not-found]

        tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
        before = len(tm.faces)
        try:
            tm_repair.fix_normals(tm)
        except Exception as exc:  # noqa: BLE001
            rep.messages.append(f"trimesh normal fix skipped: {exc}")
        if not tm.is_watertight:
            try:
                tm_repair.fill_holes(tm)
            except Exception as exc:  # noqa: BLE001
                rep.messages.append(f"trimesh hole filling skipped: {exc}")
        v, f = np.asarray(tm.vertices, dtype=np.float64), np.asarray(tm.faces, dtype=np.int64)
        rep.holes_filled = max(len(f) - before, 0)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        rep.messages.append(f"trimesh repair skipped: {exc}")

    used = np.unique(f)
    if len(used) != len(v):
        remap = -np.ones(len(v), dtype=np.int64)
        remap[used] = np.arange(len(used))
        rep.removed_unreferenced = len(v) - len(used)
        v, f = v[used], remap[f]

    # Orient each connected component outward (divergence-theorem volume sign).
    graph = coo_matrix((np.ones(3 * len(f)), (np.concatenate([f[:, 0], f[:, 1], f[:, 2]]),
                                              np.concatenate([f[:, 1], f[:, 2], f[:, 0]]))),
                       shape=(len(v), len(v)))
    _, labels = connected_components(graph, directed=False)
    face_label = labels[f[:, 0]]
    for lab in np.unique(face_label):
        idx = np.where(face_label == lab)[0]
        if signed_volume(v, f[idx]) < 0.0:
            f[idx] = f[idx][:, [0, 2, 1]]
            rep.flipped_components += 1

    out = GeometryModel(v, f, model.name, model.source, (*model.history, "Repaired"), model.reference_mode)
    rep.watertight_after = out.is_watertight
    log.info("Repair: %s", rep.summary())
    return out, rep


# ---------------------------------------------------------------------------
# Procedural geometry
# ---------------------------------------------------------------------------
Part = tuple[FloatArray, IntArray]


def _orient_outward(v: FloatArray, f: IntArray) -> Part:
    """Flip all triangles if the closed surface is wound inward."""
    return (v, f[:, [0, 2, 1]]) if signed_volume(v, f) < 0.0 else (v, f)


def _merge_parts(parts: Sequence[Part]) -> Part:
    """Concatenate closed parts into one mesh (parts may overlap)."""
    verts, faces, offset = [], [], 0
    for v, f in parts:
        verts.append(v)
        faces.append(f + offset)
        offset += len(v)
    return np.vstack(verts), np.vstack(faces)


def naca4_loop(code: NACA4, points: int = 36) -> FloatArray:
    """Closed NACA 4-digit outline in unit-chord coordinates.

    Returns ``(2*points-2, 2)`` array of ``(x, z)`` starting at the trailing edge,
    running over the upper surface to the leading edge and back under the lower
    surface. Uses the closed-trailing-edge thickness polynomial.
    """
    beta = np.linspace(0.0, math.pi, points)
    x = 0.5 * (1.0 - np.cos(beta))
    t, m, p = code.thickness, code.max_camber, code.camber_position
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    if m > 0.0 and p > 0.0:
        front = x < p
        yc = np.where(front, m / p ** 2 * (2 * p * x - x ** 2),
                      m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
        dyc = np.where(front, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    else:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
    th = np.arctan(dyc)
    xu, zu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, zl = x + yt * np.sin(th), yc - yt * np.cos(th)
    upper = np.column_stack([xu[::-1], zu[::-1]])
    lower = np.column_stack([xl[1:-1], zl[1:-1]])
    return np.vstack([upper, lower])


def _polygon_centroid(x: FloatArray, z: FloatArray) -> tuple[float, float]:
    """Area centroid of a closed polygon (falls back to the vertex mean)."""
    x2, z2 = np.roll(x, -1), np.roll(z, -1)
    cross = x * z2 - x2 * z
    a = 0.5 * cross.sum()
    if abs(a) < 1e-14:
        return float(x.mean()), float(z.mean())
    return float(((x + x2) * cross).sum() / (6 * a)), float(((z + z2) * cross).sum() / (6 * a))


def _wing_surface(loop: FloatArray, s: FloatArray, chord: FloatArray, x_le: FloatArray,
                  z_le: FloatArray, twist_deg: Optional[FloatArray] = None) -> Part:
    """Closed wing surface lofted through scaled sections along the span coordinate ``s`` (-> Y)."""
    ns, n = len(s), len(loop)
    chord = np.maximum(chord, 1e-3)
    X = x_le[:, None] + chord[:, None] * loop[None, :, 0]
    Z = z_le[:, None] + chord[:, None] * loop[None, :, 1]
    if twist_deg is not None:
        phi = np.radians(twist_deg)[:, None]
        xq = (x_le + 0.25 * chord)[:, None]
        dx, dz = X - xq, Z - z_le[:, None]
        X = xq + dx * np.cos(phi) + dz * np.sin(phi)
        Z = z_le[:, None] + (-dx * np.sin(phi) + dz * np.cos(phi))
    Y = np.repeat(s[:, None], n, axis=1)
    verts = np.stack([X, Y, Z], axis=2).reshape(-1, 3)
    c0 = _polygon_centroid(X[0], Z[0])
    c1 = _polygon_centroid(X[-1], Z[-1])
    verts = np.vstack([verts, [c0[0], s[0], c0[1]], [c1[0], s[-1], c1[1]]])
    k = np.arange(n)
    k1 = (k + 1) % n
    tris = []
    for j in range(ns - 1):
        a, b = j * n + k, j * n + k1
        c, d = (j + 1) * n + k1, (j + 1) * n + k
        tris.append(np.column_stack([a, b, c]))
        tris.append(np.column_stack([a, c, d]))
    cap0, cap1 = ns * n, ns * n + 1
    tris.append(np.column_stack([np.full(n, cap0), k1, k]))
    last = (ns - 1) * n
    tris.append(np.column_stack([np.full(n, cap1), last + k, last + k1]))
    return _orient_outward(verts, np.vstack(tris).astype(np.int64))


def _revolve(x: FloatArray, r: FloatArray, n_theta: int = 40, width_y: float = 1.0,
             width_z: float = 1.0, z_center: float = 0.0) -> Part:
    """Body of revolution about the X axis. ``r[0]`` and ``r[-1]`` must be 0 (pole vertices)."""
    m = len(x) - 1
    if m < 2:
        raise GeometryError("A body of revolution needs at least three profile points.")
    th = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    rings = []
    for i in range(1, m):
        rings.append(np.column_stack([np.full(n_theta, x[i]), r[i] * width_y * np.cos(th),
                                      r[i] * width_z * np.sin(th) + z_center]))
    verts = np.vstack([[x[0], 0.0, z_center], *rings, [x[-1], 0.0, z_center]])
    nr = len(rings)
    k = np.arange(n_theta)
    k1 = (k + 1) % n_theta

    def ring(i: int, kk: IntArray) -> IntArray:
        return 1 + i * n_theta + kk

    tris = [np.column_stack([np.zeros(n_theta, dtype=np.int64), ring(0, k1), ring(0, k)])]
    for i in range(nr - 1):
        a, b, c, d = ring(i, k), ring(i, k1), ring(i + 1, k1), ring(i + 1, k)
        tris.append(np.column_stack([a, b, c]))
        tris.append(np.column_stack([a, c, d]))
    tris.append(np.column_stack([ring(nr - 1, k), ring(nr - 1, k1), np.full(n_theta, 1 + nr * n_theta)]))
    return _orient_outward(verts, np.vstack(tris).astype(np.int64))


def _fuselage_profile(length: float, radius: float, n: int = 44, nose_frac: float = 0.22,
                      tail_start: float = 0.7, tail_power: float = 1.6) -> tuple[FloatArray, FloatArray]:
    """Streamlined fuselage radius profile with an ogive nose and a tapered tail."""
    xi = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, n)))
    r = np.empty_like(xi)
    for i, q in enumerate(xi):
        if q < nose_frac:
            r[i] = radius * math.sin(0.5 * math.pi * q / nose_frac) ** 0.8
        elif q <= tail_start:
            r[i] = radius
        else:
            v = (q - tail_start) / (1.0 - tail_start)
            r[i] = radius * max(1.0 - v ** tail_power, 0.0) ** 1.0
    r[0] = r[-1] = 0.0
    return xi * length, r


def _symmetric_stations(half_span: float, n_half: int) -> FloatArray:
    """Station positions from -half_span to +half_span (includes 0), clustered toward the tips."""
    ns = 2 * n_half
    return half_span * np.sin(math.pi * (np.arange(ns + 1) / ns - 0.5))


def _rot_x90(v: FloatArray) -> FloatArray:
    """Proper rotation by +90° about X: (x, y, z) -> (x, -z, y)."""
    return np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])


def build_naca_wing(code: str = "2412", span: float = 6.0, chord: float = 1.0, taper: float = 1.0,
                    sweep_deg: float = 0.0, twist_deg: float = 0.0, dihedral_deg: float = 0.0,
                    name: Optional[str] = None) -> GeometryModel:
    """Build a closed, finite 3-D wing from a NACA 4-digit section.

    Args:
        code: Four digits, e.g. ``"2412"`` or ``"NACA 0012"``.
        span: Wingspan in metres (clamped to 0.05 .. 1000).
        chord: Root chord in metres (clamped to 0.01 .. 1000).
        taper: Tip chord / root chord (0.1 .. 1).
        sweep_deg: Leading-edge sweep.
        twist_deg: Linear washout (tip relative to root, negative = nose down).
        dihedral_deg: Dihedral angle.

    Raises:
        GeometryError: for an invalid NACA code.
    """
    try:
        naca = parse_naca4(code)
    except ValueError as exc:
        raise GeometryError(str(exc)) from exc
    span = clamp_dimension(span, 0.05, 1000.0, 6.0)
    chord = clamp_dimension(chord, 0.01, 1000.0, 1.0)
    taper = float(np.clip(taper, 0.1, 1.0))
    half = 0.5 * span
    s = _symmetric_stations(half, 14)
    frac = np.abs(s) / half
    ch = chord * (1.0 - (1.0 - taper) * frac)
    tan_le = math.tan(math.radians(float(np.clip(sweep_deg, -60.0, 70.0))))
    x_le = np.abs(s) * tan_le + 0.25 * (chord - ch) * (1.0 if sweep_deg == 0 else 0.0)
    z_le = np.abs(s) * math.tan(math.radians(float(np.clip(dihedral_deg, -30.0, 30.0))))
    twist = twist_deg * frac
    v, f = _wing_surface(naca4_loop(naca, 36), s, ch, x_le, z_le, twist if twist_deg else None)
    model = GeometryModel(v, f, name or f"NACA {naca.code} wing", None, (f"Generated NACA {naca.code} wing",))
    return center_model(model)


def _build_delta() -> GeometryModel:
    """Swept delta wing / fighter concept: fuselage, delta wing, vertical stabiliser."""
    length, radius = 14.0, 0.95
    fx, fr = _fuselage_profile(length, radius, 48, nose_frac=0.30, tail_start=0.78, tail_power=1.4)
    fuselage = _revolve(fx, fr, 40, 1.0, 0.85)
    root_c, tip_c, half = 7.2, 0.7, 4.2
    s = _symmetric_stations(half, 14)
    a = np.abs(s) / half
    chord = root_c - (root_c - tip_c) * a
    x_le = 5.0 + (root_c - chord)                  # straight trailing edge
    wing = _wing_surface(naca4_loop(parse_naca4("0005"), 30), s, chord, x_le, np.full_like(s, -0.12))
    h = 2.7
    sf = h * np.sin(0.5 * math.pi * np.arange(0, 11) / 10.0) + 0.45
    sf[0] = 0.45
    af = (sf - 0.45) / (h)
    chord_f = 3.4 - 2.2 * af
    xle_f = 9.3 + 2.6 * af * 1.0
    fv, ff = _wing_surface(naca4_loop(parse_naca4("0006"), 24), sf, chord_f, xle_f, np.zeros_like(sf))
    fin = (_rot_x90(fv), ff)
    v, f = _merge_parts([fuselage, wing, fin])
    return center_model(GeometryModel(v, f, "Delta fighter concept", None, ("Generated delta fighter",)))


def _build_glider() -> GeometryModel:
    """High-aspect-ratio sailplane with slender fuselage and T-tail."""
    length, radius = 7.4, 0.34
    fx, fr = _fuselage_profile(length, radius, 48, nose_frac=0.12, tail_start=0.20, tail_power=1.0)
    fr = np.where(fr > 0, np.maximum(fr, 0.05), 0.0)
    fuselage = _revolve(fx, fr, 28, 0.85, 1.15)
    half = 7.5
    s = _symmetric_stations(half, 16)
    a = np.abs(s) / half
    chord = 1.15 - 0.70 * a
    x_le = 1.9 + 0.25 * (1.15 - chord)             # straight quarter-chord line
    z_le = 0.12 + np.abs(s) * math.tan(math.radians(3.0))
    wing = _wing_surface(naca4_loop(parse_naca4("4412"), 34), s, chord, x_le, z_le,
                         -2.0 * a)
    h = 1.35
    sf = h * np.sin(0.5 * math.pi * np.arange(0, 9) / 8.0)
    af = sf / h
    fin_v, fin_f = _wing_surface(naca4_loop(parse_naca4("0009"), 24), sf, 1.05 - 0.45 * af,
                                 6.15 + 0.35 * af, np.zeros_like(sf))
    fin = (_rot_x90(fin_v) + np.array([0.0, 0.0, 0.15]), fin_f)
    hs = 1.6
    ss = _symmetric_stations(hs, 8)
    ah = np.abs(ss) / hs
    stab = _wing_surface(naca4_loop(parse_naca4("0009"), 24), ss, 0.62 - 0.22 * ah,
                         6.55 + 0.1 * ah, np.full_like(ss, 0.15 + h))
    v, f = _merge_parts([fuselage, wing, fin, stab])
    return center_model(GeometryModel(v, f, "Glider / sailplane", None, ("Generated glider",)))


def _build_sphere(radius: float = 0.5, tail_ratio: float = 0.0) -> GeometryModel:
    """Sphere (``tail_ratio == 0``) or teardrop with a tapered tail of length ``tail_ratio`` x diameter."""
    tail_ratio = float(np.clip(tail_ratio, 0.0, 6.0))
    phi_f = np.linspace(0.0, 0.5 * math.pi, 18, endpoint=False)
    x_front = -radius * np.cos(phi_f)
    r_front = radius * np.sin(phi_f)
    if tail_ratio <= 1e-6:
        phi_b = np.linspace(0.5 * math.pi, math.pi, 19)
        x_back, r_back = -radius * np.cos(phi_b), radius * np.sin(phi_b)
    else:
        lt = 2.0 * radius * tail_ratio
        w = np.linspace(0.0, 1.0, 26)
        x_back = lt * w
        r_back = radius * np.maximum(1.0 - w ** 1.6, 0.0) ** 1.2
    x = np.concatenate([x_front, x_back])
    r = np.concatenate([r_front, r_back])
    r[0] = r[-1] = 0.0
    v, f = _revolve(x, r, 48)
    nm = "Aerodynamic sphere" if tail_ratio <= 1e-6 else "Teardrop body"
    return center_model(GeometryModel(v, f, nm, None, (f"Generated {nm.lower()}",)))


def build_preset(preset: GeometryPreset, naca_code: str = "2412", span: float = 6.0,
                 chord: float = 1.0, tail_ratio: float = 0.0) -> GeometryModel:
    """Generate one of the built-in geometries (no external files needed).

    Args:
        preset: Which preset to build.
        naca_code / span / chord: Used by ``CUSTOM_NACA``.
        tail_ratio: Teardrop tail length / diameter for ``SPHERE`` (0 = pure sphere).
    """
    try:
        preset = GeometryPreset(preset)      # also accepts the plain-string value
    except ValueError:
        raise GeometryError(f"Unknown preset: {preset}") from None
    if preset is GeometryPreset.NACA_0012:
        return build_naca_wing("0012", 6.0, 1.0, name="NACA 0012 wing")
    if preset is GeometryPreset.NACA_2412:
        return build_naca_wing("2412", 6.0, 1.0, name="NACA 2412 wing")
    if preset is GeometryPreset.DELTA:
        return _build_delta()
    if preset is GeometryPreset.GLIDER:
        return _build_glider()
    if preset is GeometryPreset.SPHERE:
        return _build_sphere(0.5, tail_ratio)
    if preset is GeometryPreset.CUSTOM_NACA:
        return build_naca_wing(naca_code, span, chord)
    raise GeometryError(f"Unknown preset: {preset}")


# ---------------------------------------------------------------------------
# Thread-safe holder with undo
# ---------------------------------------------------------------------------
class GeometryManager:
    """Holds the current :class:`GeometryModel` (thread-safe) and applies edits with undo.

    Heavy operations (loading, repair of large meshes) should run in a worker
    using the module-level functions; the GUI thread then calls :meth:`set_model`.
    """

    MAX_UNDO: Final[int] = 12

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model: Optional[GeometryModel] = None
        self._undo: list[GeometryModel] = []

    @property
    def model(self) -> Optional[GeometryModel]:
        """The current model (``None`` if nothing is loaded)."""
        with self._lock:
            return self._model

    @property
    def has_model(self) -> bool:
        """True when a model is loaded."""
        return self.model is not None

    @property
    def can_undo(self) -> bool:
        """True when an earlier state is available."""
        with self._lock:
            return bool(self._undo)

    def require(self) -> GeometryModel:
        """Return the current model or raise :class:`GeometryError`."""
        model = self.model
        if model is None:
            raise GeometryError("No geometry loaded. Drop an STL/OBJ file or choose a preset.")
        return model

    def set_model(self, model: GeometryModel, keep_undo: bool = True) -> GeometryModel:
        """Make ``model`` current (pushing the previous model onto the undo stack)."""
        with self._lock:
            if keep_undo and self._model is not None:
                self._undo.append(self._model)
                del self._undo[:-self.MAX_UNDO]
            elif not keep_undo:
                self._undo.clear()
            self._model = model
            return model

    def apply(self, operation: Callable[..., GeometryModel], *args: Any, **kwargs: Any) -> GeometryModel:
        """Apply ``operation(current_model, *args)`` and make the result current."""
        return self.set_model(operation(self.require(), *args, **kwargs))

    def undo(self) -> Optional[GeometryModel]:
        """Restore the previous model, if any."""
        with self._lock:
            if not self._undo:
                return self._model
            self._model = self._undo.pop()
            return self._model

    def set_reference_mode(self, mode: ReferenceAreaMode) -> GeometryModel:
        """Switch between planform and frontal reference area."""
        with self._lock:
            self._model = self.require().with_reference_mode(mode)   # not an undoable geometry edit
            return self._model

    def clear(self) -> None:
        """Remove the current model and the undo history."""
        with self._lock:
            self._model = None
            self._undo.clear()
