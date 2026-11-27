"""Export a Nozzle Lab engine's geometry: contour tables (CSV), drawings (DXF R12) and a 3-D-printable solid
(STL, millimetres). An aerospike exports its cowl and its plug as two parts."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from core import nozzle as nz


def wall_profiles(geom: nz.NozzleGeometry, planar: bool = False) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Closed wall cross-sections (name, x, r) in metres, counter-clockwise in the (x, r) plane."""
    t = max(geom.wall_thickness, 1e-5)
    out = []
    if geom.is_aerospike:
        sh = geom.aerospike(planar)
        lc = float(sh.cowl_x[1])
        cx = np.array([0.0, lc, sh.x_lip, sh.x_lip, lc, 0.0])
        cr = np.array([sh.r_outer, sh.r_outer, sh.r_lip, sh.r_lip + t, sh.r_outer + t, sh.r_outer + t])
        out.append(("cowl", cx, cr))
        px = np.concatenate([[0.0], sh.plug_x, [sh.x_base]])
        pr = np.concatenate([[0.0], sh.plug_r, [0.0]])
        out.append(("plug", px, pr))
    else:
        x, r = geom.inner_contour()
        out.append(("nozzle", np.concatenate([x, x[::-1]]), np.concatenate([r, (r + t)[::-1]])))
    fixed = []
    for name, x, r in out:
        keep = np.concatenate([[True], np.hypot(np.diff(x), np.diff(r)) > 1e-12])     # no repeated points
        x, r = x[keep], r[keep]
        area = 0.5 * float(np.sum(x * np.roll(r, -1) - np.roll(x, -1) * r))
        if area < 0.0:
            x, r = x[::-1], r[::-1]
        fixed.append((name, x, r))
    return fixed


def write_csv(geom: nz.NozzleGeometry, path: str | Path, planar: bool = False) -> Path:
    """The inner wall (x, r in mm). A bell / cone / imported nozzle also lists the outer wall radius."""
    path = Path(path)
    lines = []
    if geom.is_aerospike:
        sh = geom.aerospike(planar)
        lines.append("part,x_mm,r_mm")
        lines += [f"cowl,{x * 1e3:.4f},{r * 1e3:.4f}" for x, r in zip(sh.cowl_x, sh.cowl_r)]
        lines += [f"plug,{x * 1e3:.4f},{r * 1e3:.4f}" for x, r in zip(sh.plug_x, sh.plug_r)]
        lines.append(f"plug,{sh.x_base * 1e3:.4f},0.0000")
    else:
        x, r = geom.inner_contour()
        lines.append("x_mm,r_mm,r_outer_mm")
        lines += [f"{a * 1e3:.4f},{b * 1e3:.4f},{(b + geom.wall_thickness) * 1e3:.4f}" for a, b in zip(x, r)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_dxf(geom: nz.NozzleGeometry, path: str | Path, planar: bool = False) -> Path:
    """A 2-D drawing in millimetres (DXF R12, LINE entities; the axis on y = 0). Layers: INNER (the flow
    surface, which :func:`core.nozzle.read_contour_file` reads back), OUTER (the rest of the wall), AXIS."""
    path = Path(path)
    ents: list[str] = []

    def line(layer: str, a: tuple[float, float], b: tuple[float, float]) -> None:
        ents.extend(["0", "LINE", "8", layer, "10", f"{a[0] * 1e3:.4f}", "20", f"{a[1] * 1e3:.4f}", "30", "0.0",
                     "11", f"{b[0] * 1e3:.4f}", "21", f"{b[1] * 1e3:.4f}", "31", "0.0"])

    def poly(layer: str, x: np.ndarray, r: np.ndarray) -> None:
        for k in range(len(x) - 1):
            line(layer, (float(x[k]), float(r[k])), (float(x[k + 1]), float(r[k + 1])))

    if geom.is_aerospike:
        sh = geom.aerospike(planar)
        t = geom.wall_thickness
        poly("INNER", sh.cowl_x, sh.cowl_r)
        poly("OUTER", np.array([sh.x_lip, sh.x_lip, float(sh.cowl_x[1]), 0.0]),
             np.array([sh.r_lip, sh.r_lip + t, sh.r_outer + t, sh.r_outer + t]))
        poly("PLUG", np.append(sh.plug_x, sh.x_base), np.append(sh.plug_r, 0.0))
        x_end = sh.x_base
    else:
        x, r = geom.inner_contour()
        poly("INNER", x, r)
        poly("OUTER", x, r + geom.wall_thickness)
        x_end = float(x[-1])
    line("AXIS", (0.0, 0.0), (x_end, 0.0))
    text = "\n".join(["0", "SECTION", "2", "HEADER", "9", "$INSUNITS", "70", "4", "0", "ENDSEC",
                      "0", "SECTION", "2", "ENTITIES", *ents, "0", "ENDSEC", "0", "EOF"]) + "\n"
    path.write_text(text, encoding="ascii")
    return path


def revolve_profile(x: np.ndarray, r: np.ndarray, segments: int = 180) -> tuple[np.ndarray, np.ndarray]:
    """Triangles of the solid of revolution of a closed (x, r) polygon about the x axis. Points on the axis
    become single vertices, so the surface is closed (watertight)."""
    verts: list[np.ndarray] = []
    index: list[np.ndarray] = []
    th = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    n = 0
    for xi, ri in zip(x, r):
        if ri <= 1e-12:
            verts.append(np.array([[xi, 0.0, 0.0]]))
            index.append(np.full(segments, n))
            n += 1
        else:
            verts.append(np.column_stack([np.full(segments, xi), ri * np.cos(th), ri * np.sin(th)]))
            index.append(np.arange(n, n + segments))
            n += segments
    faces = []
    m = len(x)
    for k in range(m):
        a, b = index[k], index[(k + 1) % m]
        a1, b1 = np.roll(a, -1), np.roll(b, -1)
        for tri in (np.column_stack([a, b, b1]), np.column_stack([a, b1, a1])):
            ok = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
            faces.append(tri[ok])
    return np.concatenate(verts), np.concatenate(faces)


def write_stl(geom: nz.NozzleGeometry, path: str | Path, planar: bool = False, segments: int = 180) -> Path:
    """A watertight solid of the nozzle wall (and an aerospike's plug) in millimetres, ready for a slicer."""
    import trimesh

    if planar:
        raise ValueError("STL export builds a round (axisymmetric) nozzle; switch the shape to Round first.")
    path = Path(path)
    meshes = []
    for _name, x, r in wall_profiles(geom, planar):
        v, f = revolve_profile(x * 1e3, r * 1e3, segments)
        mesh = trimesh.Trimesh(v, f, process=True)
        trimesh.repair.fix_normals(mesh)
        meshes.append(mesh)
    combined = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    combined.export(str(path))
    return path


__all__ = ["wall_profiles", "write_csv", "write_dxf", "write_stl", "revolve_profile"]
