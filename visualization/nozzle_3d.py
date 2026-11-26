"""Rocket Nozzle Lab: a 3-D view of the nozzle and its exhaust plume.

The axisymmetric solution of :mod:`core.nozzle` is revolved about the nozzle axis, live:

* **Volume**: GPU volume rendering of the selected field inside the exhaust (the surrounding air is
  transparent), so the plume, its shock diamonds and Mach disks are seen as a 3-D glowing body. The nozzle
  is drawn as a 270° cut-away shell so the chamber and throat stay visible.
* **Cut-away**: a 90° notch is cut out of the engine and the plume; the field is painted on the two exposed
  faces, the plume boundary (50 % exhaust gas) is a translucent revolved surface, and the nozzle a shell.

A planar (slot) nozzle is extruded across its width instead of revolved.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from config.theme import CFD_BACKGROUND_BOTTOM, CFD_BACKGROUND_TOP, PALETTE
from visualization.nozzle_view import NField, NozzleView
from visualization.supersonic_view import iso_segments

MODES = ("Volume (glowing plume)", "Cut-away (field on the cut faces)")
MAX_VOXELS = 6_000_000


def _cmap(lut: np.ndarray):
    from matplotlib.colors import ListedColormap

    return ListedColormap(np.asarray(lut, dtype=float) / 255.0)


def _block_mean(a: np.ndarray, f: int) -> np.ndarray:
    """Average ``f`` × ``f`` blocks (crops the remainder)."""
    if f <= 1:
        return a
    ny, nx = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    return a[:ny, :nx].reshape(ny // f, f, nx // f, f).mean(axis=(1, 3))


def revolve(x: np.ndarray, r: np.ndarray, theta0: float, theta1: float, n: int):
    """Surface of revolution of the polyline (x, r) about the x axis, from ``theta0`` to ``theta1`` (rad)."""
    import pyvista as pv

    th = np.linspace(theta0, theta1, n)
    X = np.repeat(x[:, None], n, axis=1)
    Y = r[:, None] * np.cos(th)[None, :]
    Z = r[:, None] * np.sin(th)[None, :]
    grid = pv.StructuredGrid(X, Y, Z)
    return grid.extract_surface(algorithm="dataset_surface")


def revolve_segments(xs: np.ndarray, rs: np.ndarray, theta0: float, theta1: float, n: int):
    """Revolve line segments given as consecutive point pairs (marching-squares output) into quads."""
    import pyvista as pv

    m = len(xs) // 2
    if m == 0:
        return None
    th = np.linspace(theta0, theta1, n)
    x = xs.reshape(m, 2)
    r = rs.reshape(m, 2)
    pts = np.empty((m, 2, n, 3))
    pts[..., 0] = x[:, :, None]
    pts[..., 1] = r[:, :, None] * np.cos(th)[None, None, :]
    pts[..., 2] = r[:, :, None] * np.sin(th)[None, None, :]
    pts = pts.reshape(-1, 3)
    base = (np.arange(m) * 2 * n)[:, None] + np.arange(n - 1)[None, :]          # (m, n-1)
    a, b = base, base + 1                                                       # first endpoint, k and k+1
    c, d = base + n + 1, base + n                                               # second endpoint
    quads = np.stack([np.full_like(a, 4), a, b, c, d], axis=-1).reshape(-1)
    return pv.PolyData(pts, faces=quads)


class Nozzle3DView(QWidget):
    """Live 3-D view of the Rocket Nozzle Lab (follows the 2-D view's field and solver)."""

    def __init__(self, source: NozzleView, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        from pyvistaqt import QtInteractor

        self.source = source
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        bar = QWidget()
        bar.setObjectName("Root")
        h = QHBoxLayout(bar)
        h.setContentsMargins(8, 4, 8, 4)
        self.mode = QComboBox()
        for m in MODES:
            self.mode.addItem(m)
        self.mode.currentIndexChanged.connect(lambda _i: self._rebuild())
        h.addWidget(QLabel("3-D:"))
        h.addWidget(self.mode)
        for text, key in (("Side", "side"), ("Three-quarter", "iso"), ("From behind", "rear"), ("Nozzle close-up", "close")):
            b = QPushButton(text)
            b.setProperty("role", "ghost")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _=False, k=key: self.view(k))
            h.addWidget(b)
        h.addStretch(1)
        self.info = QLabel("")
        self.info.setProperty("role", "muted")
        h.addWidget(self.info)
        root.addWidget(bar)
        self.plotter = QtInteractor(self, multi_samples=4, auto_update=False)
        root.addWidget(self.plotter.interactor if hasattr(self.plotter, "interactor") else self.plotter, 1)
        self.plotter.set_background(CFD_BACKGROUND_BOTTOM, top=CFD_BACKGROUND_TOP)
        self.plotter.enable_anti_aliasing("fxaa")
        self._key: tuple = ()
        self._vol = None
        self._vol_index = None
        self._last_steps = -1
        self._last_field: Optional[NField] = None
        self._volume_ok = True
        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.refresh)
        self._camera_set = False
        self._closed = False

    # -- life cycle --------------------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.refresh(force=True)
        self.timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802
        self.timer.stop()
        super().hideEvent(event)

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.timer.stop()
        try:
            self.plotter.close()
        except Exception:  # noqa: BLE001
            pass

    # -- scene ---------------------------------------------------------------------------
    def _geometry_key(self) -> tuple:
        s = self.source
        lay = s.layout_
        return (id(lay), s.setup.planar, self.mode.currentIndex(), s.field)

    def refresh(self, force: bool = False) -> None:
        s = self.source
        if self._closed or s.solver is None or s.layout_ is None or not self.isVisible():
            return
        key = self._geometry_key()
        if key != self._key or force:
            self._key = key
            self._rebuild()
            return
        if s.solver.steps == self._last_steps and s.field is self._last_field:
            return
        self._update_fields()
        self.plotter.render()

    def _rebuild(self) -> None:
        """Rebuild the whole scene (new grid, geometry, mode or field)."""
        s = self.source
        if s.solver is None or s.layout_ is None:
            return
        self._key = self._geometry_key()
        p = self.plotter
        p.clear()
        p.remove_all_lights()
        p.enable_lightkit()
        self._vol = None
        lay = s.layout_
        planar = s.setup.planar
        # the engine: a 270° cut-away shell (a planar nozzle is extruded across its width instead)
        x = np.maximum(lay.wall_x, 0.0)
        if planar:
            self._add_planar_walls(x, lay.wall_r)
        else:
            shell = revolve(x, lay.wall_r, math.pi, 2.5 * math.pi, 64)
            p.add_mesh(shell, color="#A9B4C2", smooth_shading=True, specular=0.6, specular_power=30,
                       ambient=0.15, name="walls", reset_camera=False)
        if self.mode.currentIndex() == 0:
            self._build_volume()
        else:
            self._build_cutaway()
        self._update_fields()
        if not self._camera_set:
            self.view("iso")
            self._camera_set = True
        p.render()

    def _add_planar_walls(self, x: np.ndarray, r: np.ndarray) -> None:
        import pyvista as pv

        width = 2.0 * float(np.max(r))
        for sign in (1.0, -1.0):
            n = len(x)
            pts = np.concatenate([np.column_stack([x, sign * r, np.full(n, -width / 2)]),
                                  np.column_stack([x, sign * r, np.full(n, width / 2)])])
            faces = []
            for k in range(n - 1):
                faces += [4, k, k + 1, n + k + 1, n + k]
            self.plotter.add_mesh(pv.PolyData(pts, faces=np.asarray(faces)), color="#A9B4C2", smooth_shading=True,
                                  specular=0.5, name=f"walls{sign}", reset_camera=False)

    # -- volume mode ---------------------------------------------------------------------
    def _build_volume(self) -> None:
        import pyvista as pv

        s = self.source
        solver, lay = s.solver, s.layout_
        nx = solver.nx
        ny = self._rows = self._needed_rows()                # only the radial band the exhaust occupies
        f = 1
        while (nx // f) * (2 * (ny // f)) ** 2 > MAX_VOXELS:
            f += 1
        nxd, nyd = nx // f, ny // f
        h = lay.dx * f
        n2 = 2 * nyd
        planar = s.setup.planar
        coords = (np.arange(n2) + 0.5) * h - nyd * h
        yy, zz = np.meshgrid(coords, coords, indexing="ij")             # (y, z)
        rr = np.abs(yy) if planar else np.hypot(yy, zz)
        ridx = np.floor(rr / h).astype(np.int64)
        outside = ridx >= nyd
        self._vol_index = (np.where(outside, 0, ridx).T, outside.T, f, nxd, nyd)   # (z, y) order
        vol = pv.ImageData(dimensions=(nxd, n2, n2), spacing=(h, h, h), origin=(0.5 * h, -nyd * h + 0.5 * h,
                                                                                  -nyd * h + 0.5 * h))
        vol.point_data["v"] = np.zeros(nxd * n2 * n2, dtype=np.float32)
        self._vol = vol
        _, _, _, lut_name, _, _ = s.field_values(s.field)
        opacity = [0.0, 0.04, 0.12, 0.25, 0.42, 0.6, 0.78, 0.92]
        try:
            self.plotter.add_volume(vol, scalars="v", clim=(0.0, 1.0), cmap=_cmap(s._lut_for(lut_name)),
                                    opacity=opacity, shade=False, show_scalar_bar=False, mapper="smart",
                                    name="plume", reset_camera=False, opacity_unit_distance=1.5 * h)
            self._volume_ok = True
            try:                                        # finer ray steps than the default: no banding
                mapper = self.plotter.actors["plume"].GetMapper()
                mapper.SetAutoAdjustSampleDistances(False)
                mapper.SetSampleDistance(0.4 * h)
            except Exception:  # noqa: BLE001 - mapper-specific; the default still works
                pass
        except Exception as exc:  # noqa: BLE001 - very old GPUs lack 3-D textures
            self._volume_ok = False
            self.info.setText(f"Volume rendering is not available here ({exc}); use the cut-away mode.")

    def _needed_rows(self) -> int:
        """Rows (from the axis) that contain exhaust gas, with a margin; at least ~1.6 nozzle radii."""
        s = self.source
        solver, lay = s.solver, s.layout_
        r, _, _, p = (a.astype(np.float32) for a in solver.primitives())
        y = s._exhaust_fraction(r, p)
        rows = np.nonzero((y > 0.15).any(axis=1))[0]
        base = int(1.6 * float(np.max(lay.wall_r)) / lay.dx)
        need = int(rows.max() * 1.15) + 6 if rows.size else base
        return int(min(max(need, base, 8), solver.ny))

    def _plume_values(self) -> np.ndarray:
        """The selected field on the half plane, scaled to 0..1 inside the exhaust and 0 in the surrounding air
        (so the volume shows the plume itself)."""
        s = self.source
        val, lo, hi, _, _, _ = s.field_values(s.field)
        v = np.clip((np.nan_to_num(val, nan=lo) - lo) / max(hi - lo, 1e-12), 0.0, 1.0).astype(np.float32)
        if s.field is not NField.GLOW:
            r, _, _, p = (a.astype(np.float32) for a in s.solver.primitives())
            y = s._exhaust_fraction(r, p)
            v = np.where(y > 0.3, np.maximum(v, 0.03), 0.0).astype(np.float32)
        v[s.solver.solid_int] = 0.0
        return v

    # -- cut-away mode ---------------------------------------------------------------------
    def _build_cutaway(self) -> None:
        import pyvista as pv

        s = self.source
        solver, lay = s.solver, s.layout_
        self._rows = self._needed_rows()                    # crop the faces to the plume's radial band
        length, rad = solver.nx * lay.dx, self._rows * lay.dx
        tc = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        faces = np.array([4, 0, 1, 2, 3])
        a = pv.PolyData(np.array([[0, 0, 0], [length, 0, 0], [length, -rad, 0], [0, -rad, 0]], float), faces=faces)
        b = pv.PolyData(np.array([[0, 0, 0], [length, 0, 0], [length, 0, rad], [0, 0, rad]], float), faces=faces)
        for m in (a, b):
            m.active_texture_coordinates = tc
        self._faces = (a, b)

    def _update_fields(self) -> None:
        s = self.source
        if s.solver is None:
            return
        self._last_steps = s.solver.steps
        self._last_field = s.field
        if self.mode.currentIndex() == 0:
            if self._vol is not None and self._volume_ok:
                v = self._plume_values()
                if self._needed_rows() > self._rows:          # the plume has grown past the voxel block
                    self._rebuild()
                    return
                ridx, outside, f, nxd, nyd = self._vol_index
                vd = _block_mean(v[:self._rows], f)[:nyd, :nxd]
                arr = vd[ridx]                                   # (z, y, x)
                arr[outside] = 0.0
                self._vol.point_data["v"][:] = arr.ravel()
                self._vol.GetPointData().Modified()
            self.info.setText(f"{s.field.value} · exhaust only · {s.engine_state().lower()}")
        else:
            self._update_cutaway()
            self.info.setText(f"{s.field.value} on the cut faces · translucent: plume boundary (50 % exhaust)")

    def _update_cutaway(self) -> None:
        import pyvista as pv

        s = self.source
        solver, lay = s.solver, s.layout_
        val, lo, hi, lut_name, _, _ = s.field_values(s.field)
        idx = np.clip((np.nan_to_num(val, nan=lo) - lo) / max(hi - lo, 1e-12) * 255.0, 0, 255).astype(np.uint8)
        rgb = s._lut_for(lut_name)[idx]
        rgb[solver.solid_int] = (60, 66, 78)
        r, _, _, p = (q.astype(np.float32) for q in solver.primitives())
        y = s._exhaust_fraction(r, p)
        # still air is see-through on the cut faces (fully for the glow, partly for the other fields)
        if s.field is NField.GLOW:
            alpha = np.clip(np.nan_to_num(val) * 2.5, 0.0, 1.0)
        else:
            alpha = np.where(y > 0.3, 1.0, 0.35)
        alpha[solver.solid_int] = 1.0
        rgba = np.dstack([rgb, (alpha * 255).astype(np.uint8)])[:self._rows]
        tex = pv.Texture(np.ascontiguousarray(rgba[::-1]))      # image row 0 is the top (outer) edge
        a, b = self._faces
        self.plotter.add_mesh(a, texture=tex, name="face_a", ambient=0.6, diffuse=0.5, reset_camera=False)
        self.plotter.add_mesh(b, texture=tex, name="face_b", ambient=0.6, diffuse=0.5, reset_camera=False)
        if self._needed_rows() > self._rows:                  # the plume outgrew the faces
            self._rebuild()
            return
        y[solver.solid_int] = 0.0
        xs, ys = iso_segments(y, 0.5)
        surf = revolve_segments((xs + 0.5) * lay.dx, (ys + 0.5) * lay.dx, math.pi, 2.5 * math.pi, 28)
        if surf is not None:
            self.plotter.add_mesh(surf, color=PALETTE.cyan, opacity=0.22, smooth_shading=False, name="boundary",
                                  specular=0.3, reset_camera=False)
        else:
            self.plotter.remove_actor("boundary")

    # -- camera ---------------------------------------------------------------------------------
    def view(self, which: str) -> None:
        """Camera presets (z up, x to the right; the cut-away notch, y < 0 and z > 0, faces the viewer)."""
        s = self.source
        lay = s.layout_
        if lay is None or s.solver is None:
            return
        length = s.solver.nx * lay.dx
        rows = getattr(self, "_rows", s.solver.ny)
        rad = max(rows * lay.dx, float(np.max(lay.wall_r)))
        if which == "close":
            xe = lay.x_exit
            d = max(2.2 * xe, 6.0 * float(np.max(lay.wall_r)))
            focal = (0.55 * xe, 0.0, 0.0)
            pos = (0.55 * xe - 0.35 * d, -0.8 * d, 0.55 * d)
        else:
            span = max(length, 3.0 * rad)
            focal = (0.5 * length, 0.0, 0.0)
            d = 0.95 * span
            pos = {"side": (0.5 * length, -1.05 * d, 0.12 * d),
                   "iso": (0.5 * length - 0.45 * d, -0.75 * d, 0.55 * d),
                   "rear": (length + 0.45 * d, -0.55 * d, 0.45 * d)}[which]
        self.plotter.camera_position = [pos, focal, (0.0, 0.0, 1.0)]
        if which != "close":
            self.plotter.reset_camera()                 # keep the direction, fit the whole engine and plume
            self.plotter.camera.zoom(1.15)
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def save_png(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.plotter.screenshot(str(out))
        return out
