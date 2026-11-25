"""2-D obstacle shapes for the Live Flow Lab.

Every shape is described as a *segment soup* in unit coordinates: an ``(M, 2, 2)`` array of line
segments ``[[x0, z0], [x1, z1]]`` whose closed loops bound the solid. Polygons convert trivially;
a cross-section of an arbitrary 3-D triangle mesh (which is naturally a set of unordered segments)
is handled the same way, so imported STL models work without any loop reconstruction.

:func:`rasterize_segments` turns a segment soup into a boolean cell mask with an even-odd scanline
fill, which is robust to unordered segments, multiple loops and holes.

Coordinates: ``x`` is streamwise (flow along +x), ``z`` is up. Reference length ``1`` is the chord
for streamlined shapes and the diameter / height for bluff bodies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from utils.validators import parse_naca4

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class Obstacle2D(str, Enum):
    """Built-in obstacles for the Live Flow Lab (values double as UI labels)."""

    AIRFOIL = "NACA airfoil"
    AIRFOIL_FILE = "Airfoil from file (.dat)"
    CYLINDER = "Circular cylinder"
    SQUARE = "Square cylinder"
    PLATE = "Flat plate"
    ELLIPSE = "Ellipse (3:1)"
    WEDGE = "Wedge / triangle"
    MODEL_SECTION = "Section of the 3-D model"
    EMPTY = "Empty tunnel (draw your own)"


#: Shapes whose reference length is the chord (streamlined) rather than the diameter (bluff).
STREAMLINED: frozenset[Obstacle2D] = frozenset({Obstacle2D.AIRFOIL, Obstacle2D.AIRFOIL_FILE, Obstacle2D.PLATE, Obstacle2D.ELLIPSE,
                                                Obstacle2D.MODEL_SECTION})


@dataclass(frozen=True)
class ShapeOutline:
    """A unit-size outline plus the point it rotates about (quarter chord or centre)."""

    segments: FloatArray            # (M, 2, 2)
    pivot: tuple[float, float]      # rotation centre in unit coordinates
    name: str


def polygon_segments(points: FloatArray) -> FloatArray:
    """Closed polygon (N, 2) -> segments (N, 2, 2)."""
    p = np.asarray(points, dtype=float)
    return np.stack([p, np.roll(p, -1, axis=0)], axis=1)


def _circle(n: int = 96, rx: float = 0.5, rz: float = 0.5) -> FloatArray:
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([rx * np.cos(t), rz * np.sin(t)])


def naca_outline(code: str = "0012", points: int = 120) -> FloatArray:
    """Closed NACA 4-digit outline with unit chord, leading edge at x = 0."""
    from core.geometry_manager import naca4_loop

    return naca4_loop(parse_naca4(code), points)


def read_airfoil_dat(path: str | Path) -> tuple[str, FloatArray]:
    """Read an airfoil coordinate file (UIUC / XFOIL ``.dat``) -> ``(name, closed outline)``.

    Both layouts are accepted: **Selig** (one loop, trailing edge -> upper surface -> leading edge ->
    lower surface -> trailing edge) and **Lednicer** (a line with the point counts, then the upper and
    the lower surface, each from the leading edge). The outline is normalised to unit chord with the
    leading edge at the origin and the chord line along +x.

    Raises:
        ValueError: when the file holds no usable coordinates.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValueError(f"Cannot read {p.name}: {exc}") from exc
    name, rows = "", []
    for line in text.splitlines():
        parts = line.replace(",", " ").split()
        try:
            if len(parts) >= 2:
                rows.append((float(parts[0]), float(parts[1])))
                continue
        except ValueError:
            pass
        if not rows and line.strip() and not name:
            name = line.strip()[:60]
    if len(rows) < 8:
        raise ValueError(f"{p.name} does not contain airfoil coordinates (x y per line).")
    pts = np.asarray(rows, dtype=float)
    c0, c1 = pts[0]
    lednicer = (c0 > 1.5 and c1 > 1.5 and float(c0).is_integer() and float(c1).is_integer()
                and int(c0) + int(c1) <= len(pts) - 1)
    if lednicer:                                                  # counts, then upper and lower (LE -> TE)
        nu, nl = int(c0), int(c1)
        upper, lower = pts[1:1 + nu], pts[1 + nu:1 + nu + nl]
        if len(upper) < 3 or len(lower) < 3:
            raise ValueError(f"{p.name}: the Lednicer point counts do not match the data.")
        loop = np.vstack([upper[::-1], lower[1:]])
    else:
        loop = pts
    loop = loop[np.isfinite(loop).all(axis=1)]
    keep = np.concatenate([[True], np.linalg.norm(np.diff(loop, axis=0), axis=1) > 1e-9])
    loop = loop[keep]
    if len(loop) < 8:
        raise ValueError(f"{p.name} has too few distinct points.")
    le = loop[int(np.argmin(loop[:, 0]))]
    te = 0.5 * (loop[0] + loop[-1])
    chord = te - le
    c = float(np.hypot(*chord))
    if c < 1e-9:
        raise ValueError(f"{p.name}: could not find the chord (leading and trailing edge coincide).")
    ang = -math.atan2(chord[1], chord[0])
    rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    out = ((loop - le) @ rot.T) / c
    return (name or p.stem), out


def airfoil_file_outline(path: str | Path) -> ShapeOutline:
    """Outline (pivot at the quarter chord) from an airfoil ``.dat`` file."""
    name, pts = read_airfoil_dat(path)
    return ShapeOutline(polygon_segments(pts), (0.25, 0.0), name)


def builtin_outline(kind: Obstacle2D, naca_code: str = "2412") -> ShapeOutline:
    """Unit outline of a built-in obstacle (not for :attr:`Obstacle2D.MODEL_SECTION` / ``EMPTY``)."""
    if kind is Obstacle2D.AIRFOIL:
        return ShapeOutline(polygon_segments(naca_outline(naca_code)), (0.25, 0.0), f"NACA {naca_code}")
    if kind is Obstacle2D.CYLINDER:
        return ShapeOutline(polygon_segments(_circle()), (0.0, 0.0), "Cylinder")
    if kind is Obstacle2D.SQUARE:
        sq = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
        return ShapeOutline(polygon_segments(sq), (0.0, 0.0), "Square")
    if kind is Obstacle2D.PLATE:
        t = 0.03          # 3 % thick plate, rounded only by the grid
        pl = np.array([[0.0, -t / 2], [1.0, -t / 2], [1.0, t / 2], [0.0, t / 2]])
        return ShapeOutline(polygon_segments(pl), (0.5, 0.0), "Flat plate")
    if kind is Obstacle2D.ELLIPSE:
        e = _circle(rx=0.5, rz=1.0 / 6.0) + np.array([0.5, 0.0])
        return ShapeOutline(polygon_segments(e), (0.5, 0.0), "Ellipse")
    if kind is Obstacle2D.WEDGE:
        w = np.array([[0.0, 0.0], [1.0, -0.5], [1.0, 0.5]])
        return ShapeOutline(polygon_segments(w), (0.5, 0.0), "Wedge")
    raise ValueError(f"{kind.value} has no built-in outline")


def mesh_section(vertices: FloatArray, faces: NDArray[np.integer], y0: Optional[float] = None
                 ) -> Optional[ShapeOutline]:
    """Cut a triangle mesh with the plane ``y = y0`` (default: mid-span) -> unit-chord segment soup.

    The section is scaled so its streamwise (x) extent is 1 with the leading edge at x = 0, and
    its z coordinate is centred. Returns ``None`` when the plane misses the mesh.
    """
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    if y0 is None:
        y0 = 0.5 * float(v[:, 1].min() + v[:, 1].max())
    # Nudge the plane off exact vertex heights so no edge lies in it.
    span = float(np.ptp(v[:, 1])) or 1.0
    d = v[:, 1] - (y0 + 1e-7 * span)
    tri_d = d[f]                                         # (F, 3)
    pos = tri_d > 0.0
    n_pos = pos.sum(axis=1)
    cut = (n_pos == 1) | (n_pos == 2)
    if not cut.any():
        return None
    f, tri_d, pos = f[cut], tri_d[cut], pos[cut]
    tri = v[f]                                           # (C, 3, 3)
    # For each triangle find the two edges whose endpoints straddle the plane.
    edges = ((0, 1), (1, 2), (2, 0))
    crosses = np.stack([pos[:, a] != pos[:, b] for a, b in edges], axis=1)   # (C, 3), exactly two True
    pts = []
    for a, b in edges:
        da, db = tri_d[:, a], tri_d[:, b]
        t = da / np.where(np.abs(da - db) < 1e-300, 1e-300, da - db)
        p = tri[:, a] + t[:, None] * (tri[:, b] - tri[:, a])
        pts.append(p[:, [0, 2]])
    pts_arr = np.stack(pts, axis=1)                      # (C, 3 edges, 2)
    order = np.argsort(~crosses, axis=1, kind="stable")[:, :2]
    segs = np.take_along_axis(pts_arr, order[:, :, None], axis=1)
    x_min, x_max = float(segs[..., 0].min()), float(segs[..., 0].max())
    chord = x_max - x_min
    if chord <= 1e-12:
        return None
    z_mid = 0.5 * float(segs[..., 1].min() + segs[..., 1].max())
    segs = (segs - np.array([x_min, z_mid])) / chord
    return ShapeOutline(segs, (0.25, 0.0), "Model section")


def transform_segments(segs: FloatArray, pivot: tuple[float, float], scale: float, angle_deg: float,
                       origin: tuple[float, float]) -> FloatArray:
    """Rotate unit segments about ``pivot`` by the angle of attack, scale to cells, move to ``origin``.

    A positive angle of attack pitches the nose *up* (the trailing edge moves down), with the
    flow travelling in +x.
    """
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    p = segs - np.asarray(pivot)
    x, z = p[..., 0], p[..., 1]
    xr = x * c + z * s
    zr = -x * s + z * c
    return np.stack([xr * scale + origin[0], zr * scale + origin[1]], axis=-1)


def rasterize_segments(segs: FloatArray, nx: int, ny: int) -> BoolArray:
    """Even-odd scanline fill of a segment soup on an ``(ny, nx)`` grid of unit cells.

    Cell ``(j, i)`` has its centre at ``(i, j)`` in segment coordinates. A ray along each row
    counts segment crossings, so loops may be unordered, open-ended pieces of a closed outline,
    or contain holes.
    """
    mask = np.zeros((ny, nx), dtype=bool)
    if segs is None or len(segs) == 0:
        return mask
    x0, z0 = segs[:, 0, 0], segs[:, 0, 1]
    x1, z1 = segs[:, 1, 0], segs[:, 1, 1]
    z_lo, z_hi = np.minimum(z0, z1), np.maximum(z0, z1)
    j_min = max(0, int(math.floor(z_lo.min())))
    j_max = min(ny - 1, int(math.ceil(z_hi.max())))
    dz = z1 - z0
    safe_dz = np.where(np.abs(dz) < 1e-300, 1e-300, dz)
    cols = np.arange(nx, dtype=float)
    for j in range(j_min, j_max + 1):
        y = float(j)
        hit = (z_lo <= y) & (z_hi > y) & (np.abs(dz) > 1e-12)    # half-open: shared vertices count once
        if not hit.any():
            continue
        xs = x0[hit] + (y - z0[hit]) * (x1[hit] - x0[hit]) / safe_dz[hit]
        xs.sort()
        n = len(xs) - (len(xs) % 2)
        for k in range(0, n, 2):
            mask[j] |= (cols >= xs[k]) & (cols <= xs[k + 1])
    return mask


def build_mask(outline: Optional[ShapeOutline], nx: int, ny: int, length_cells: float, angle_deg: float,
               center: tuple[float, float]) -> BoolArray:
    """Mask for an outline placed with its reference length = ``length_cells`` at ``center``.

    For streamlined (chord-referenced) outlines ``center`` is where the pivot (quarter chord) goes.
    """
    if outline is None:
        return np.zeros((ny, nx), dtype=bool)
    segs = transform_segments(outline.segments, outline.pivot, length_cells, angle_deg, center)
    return rasterize_segments(segs, nx, ny)
