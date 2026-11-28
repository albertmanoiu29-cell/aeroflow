"""3-D view of a spinning propeller and the air it moves (slipstream from :mod:`core.propeller_flow`).

The blades turn and the air particles move on the same slowed-down clock, so what you see is the real
motion in slow motion: particles drawn in from upstream, sped up through the disc, twisted into a helix
by the swirl, and the slipstream contracting behind the propeller.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from config.theme import CFD_BACKGROUND_BOTTOM, CFD_BACKGROUND_TOP, PALETTE
from core.propeller_flow import Slipstream

N_PARTICLES = 2600
CROSS_SECONDS = 2.6            # at "1x" the fastest air crosses the view in about this long


class Propeller3DView(QWidget):
    """Rotating blades, animated particles, steady helical streamlines and the slipstream boundary."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        from pyvistaqt import QtInteractor

        self.plotter = QtInteractor(self, multi_samples=8, auto_update=False)
        self.plotter.set_background(CFD_BACKGROUND_BOTTOM, top=CFD_BACKGROUND_TOP)
        self.plotter.enable_lightkit()
        self.t_particles = QCheckBox("Air particles")
        self.t_lines = QCheckBox("Streamlines")
        self.t_tube = QCheckBox("Slipstream boundary")
        for t, on in ((self.t_particles, True), (self.t_lines, True), (self.t_tube, True)):
            t.setProperty("toggle", "true")
            t.setChecked(on)
            t.toggled.connect(lambda _v: self._rebuild_static())
        self.speed = QSlider(Qt.Orientation.Horizontal)
        self.speed.setRange(10, 300)
        self.speed.setValue(100)
        self.speed.setFixedWidth(140)
        self.speed.setToolTip("Animation speed")
        self.info = QLabel("")
        self.info.setProperty("role", "muted")
        bar = QHBoxLayout()
        bar.setContentsMargins(8, 6, 8, 6)
        bar.setSpacing(14)
        for w in (self.t_particles, self.t_lines, self.t_tube):
            bar.addWidget(w)
        bar.addSpacing(10)
        bar.addWidget(QLabel("Speed"))
        bar.addWidget(self.speed)
        bar.addStretch(1)
        bar.addWidget(self.info)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addLayout(bar)
        lay.addWidget(self.plotter.interactor, 1)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self._rng = np.random.default_rng(7)
        self.ss: Optional[Slipstream] = None
        self._blades = None
        self._blade_actor = None
        self._points = None
        self._angle = 0.0
        self._parts: Optional[np.ndarray] = None
        self._camera_set = False
        self.time_scale = 1.0

    # -- set-up ------------------------------------------------------------------------------------
    def set_propeller(self, diameter: float, blades: int, pitch_ratio: float, ss: Slipstream) -> None:
        import pyvista as pv

        from core.geometry_manager import build_propeller

        key = (round(diameter, 6), blades, round(pitch_ratio, 6))
        if getattr(self, "_geo_key", None) != key:
            self._geo_key = key
            m = build_propeller(diameter, blades, pitch_ratio)
            faces = np.column_stack([np.full(len(m.faces), 3), m.faces]).ravel()
            self._blades = pv.PolyData(np.asarray(m.vertices, float), faces).compute_normals(
                split_vertices=True, feature_angle=40.0)
        self.ss = ss
        r = ss.radius
        u_max = max(ss.wake_speed, 1e-3)
        # slow motion: the fastest air crosses the 6 R view in CROSS_SECONDS (at the 1x speed setting)
        self.time_scale = (6.0 * r / u_max) / CROSS_SECONDS
        rev = ss.omega / (2.0 * math.pi)
        self.info.setText(f"{rev * 60:,.0f} rpm shown {1.0 / max(self.time_scale, 1e-9):,.0f}× slower · wake "
                          f"{ss.wake_speed:.1f} m/s")
        self._parts = None
        self._rebuild_static()
        if not self._camera_set:
            self._camera_set = True
            self.plotter.camera_position = [(-1.4 * r, -7.6 * r, 3.2 * r), (1.7 * r, 0.0, -0.2 * r), (0.0, 0.0, 1.0)]
        self.plotter.render()

    def _rebuild_static(self) -> None:
        import pyvista as pv

        p, ss = self.plotter, self.ss
        if ss is None or self._blades is None:
            return
        r = ss.radius
        for name in ("blades", "lines", "tube", "particles"):
            p.remove_actor(name, render=False)
        self._blade_actor = p.add_mesh(self._blades, color="#C9D1DC", smooth_shading=True, specular=0.6,
                                       specular_power=30, name="blades", reset_camera=False)
        self._blade_actor.SetOrientation(math.degrees(self._angle), 0.0, 0.0)
        x0, x1 = -2.0 * r, 5.0 * r
        cmap = "turbo"
        clim = (min(ss.speed, ss.speed + 0.2 * float(np.max(ss.v_induced))), ss.wake_speed)
        if self.t_lines.isChecked():
            lines = []
            speeds = []
            for k, r0 in enumerate(np.linspace(0.3, 0.92, 4) * r):
                for j in range(3):
                    th0 = 2.0 * math.pi * (j / 3.0 + 0.13 * k)
                    pts = ss.streamline(float(r0), th0, x0, x1, 260)
                    lines.append(pts)
                    speeds.append(ss.axial_speed(np.full(len(pts), r0), pts[:, 0]))
            poly = pv.PolyData()
            allp = np.concatenate(lines)
            cells, off = [], 0
            for pts in lines:
                cells += [len(pts)] + list(range(off, off + len(pts)))
                off += len(pts)
            poly.points = allp
            poly.lines = np.asarray(cells)
            poly.point_data["speed"] = np.concatenate(speeds)
            tube = poly.tube(radius=0.006 * r * 2.0, n_sides=10)
            p.add_mesh(tube, scalars="speed", cmap=cmap, clim=clim, smooth_shading=True, name="lines",
                       reset_camera=False, show_scalar_bar=False, opacity=0.9)
        if self.t_tube.isChecked():
            xs = np.linspace(-0.05 * r, x1, 80)
            rr = ss.tube_radius(np.full(len(xs), 0.985 * r), xs)
            th = np.linspace(0.0, 2.0 * math.pi, 64)
            X = np.repeat(xs[:, None], len(th), axis=1)
            Y = rr[:, None] * np.cos(th)[None, :]
            Z = rr[:, None] * np.sin(th)[None, :]
            surf = pv.StructuredGrid(X, Y, Z).extract_surface(algorithm="dataset_surface")
            p.add_mesh(surf, color=PALETTE.cyan, opacity=0.10, smooth_shading=True, name="tube",
                       reset_camera=False)
        if self.t_particles.isChecked():
            if self._parts is None:
                self._parts = self._spawn(N_PARTICLES, initial=True)
            pts = self._positions()
            self._points = pv.PolyData(pts)
            self._points.point_data["speed"] = ss.axial_speed(self._parts[:, 0], self._parts[:, 2])
            p.add_mesh(self._points, scalars="speed", cmap=cmap, clim=clim, render_points_as_spheres=True,
                       point_size=5, name="particles", reset_camera=False, show_scalar_bar=False)
        else:
            self._points = None
        bars = getattr(p, "scalar_bars", None)
        for title in (list(bars.keys()) if bars is not None else []):
            p.remove_scalar_bar(title, render=False)
        if self.t_lines.isChecked() or self.t_particles.isChecked():
            bar = p.add_scalar_bar("air speed (m/s)", vertical=True, position_x=0.90, position_y=0.08, height=0.32,
                                   width=0.05, color=PALETTE.text_muted, title_font_size=12, label_font_size=11,
                                   n_labels=4, fmt="%.0f")
            bar.SetTextPositionToPrecedeScalarBar()
            if hasattr(bar, "SetVerticalTitleSeparation"):
                bar.SetVerticalTitleSeparation(8)
        p.render()

    # -- particles: (r0 at the disc, angle, x) ----------------------------------------------------------
    def _spawn(self, n: int, initial: bool = False) -> np.ndarray:
        ss = self.ss
        r = ss.radius
        out = np.empty((n, 3))
        inside = self._rng.uniform(0.0, 1.0, n) < 0.82
        out[:, 0] = np.where(inside, r * np.sqrt(self._rng.uniform(0.02, 0.97, n)),
                             r * self._rng.uniform(1.05, 1.7, n))
        out[:, 1] = self._rng.uniform(0.0, 2.0 * math.pi, n)
        out[:, 2] = self._rng.uniform(-2.0 * r, 5.0 * r, n) if initial else -2.0 * r - self._rng.uniform(0, 0.3 * r, n)
        return out

    def _positions(self) -> np.ndarray:
        ss, pr = self.ss, self._parts
        r0, th, x = pr[:, 0], pr[:, 1], pr[:, 2]
        inside = r0 < ss.radius
        rr = np.where(inside, ss.tube_radius(r0, x), r0)
        return np.column_stack([x, rr * np.cos(th), rr * np.sin(th)])

    # -- animation -------------------------------------------------------------------------------------
    def start(self) -> None:
        if self.ss is not None:
            self.timer.start()

    def stop(self) -> None:
        self.timer.stop()

    def _tick(self) -> None:
        ss = self.ss
        if ss is None or not self.isVisible():
            return
        dt = self.timer.interval() / 1000.0 * self.time_scale * self.speed.value() / 100.0
        self._angle = (self._angle + ss.omega * dt) % (2.0 * math.pi)
        if self._blade_actor is not None:
            self._blade_actor.SetOrientation(math.degrees(self._angle), 0.0, 0.0)
        if self._points is not None and self._parts is not None:
            pr = self._parts
            r0, x = pr[:, 0], pr[:, 2]
            inside = r0 < ss.radius
            u = np.where(inside, ss.axial_speed(r0, x), ss.speed + 0.25 * ss.induced_at(np.full_like(r0, 0.9 *
                                                                                                    ss.radius))
                         * (1.0 + x / np.sqrt(x * x + ss.radius ** 2)) * np.exp(-(r0 - ss.radius) / ss.radius))
            rr = np.where(inside, ss.tube_radius(r0, x), r0)
            pr[:, 1] += np.where(inside, ss.swirl(r0, x) / np.maximum(rr, 1e-9), 0.0) * dt
            pr[:, 2] += u * dt
            gone = pr[:, 2] > 5.0 * ss.radius
            if gone.any():
                pr[gone] = self._spawn(int(gone.sum()))
            self._points.points = self._positions()
            self._points.point_data["speed"] = np.where(inside, ss.axial_speed(r0, pr[:, 2]), u)
        self.plotter.render()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.start()

    def hideEvent(self, event) -> None:  # noqa: N802
        super().hideEvent(event)
        self.stop()

    def shutdown(self) -> None:
        self.stop()
        try:
            self.plotter.close()
        except Exception:  # noqa: BLE001
            pass


__all__ = ["Propeller3DView"]
