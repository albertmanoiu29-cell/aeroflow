"""Burn-back of arbitrary solid-propellant grain cross-sections (star, finocyl, moon burner, C-slot ...).

The surface regresses normal to itself at the same rate everywhere, so the burning surface after a web
distance w is the level set d = w of the distance d from the initial port (a fast-marching / distance-
transform construction, as in openMotor). On a fine raster of the cross-section the distance map gives, for
every w, the burning perimeter (length of the d = w contour inside the case) and the port area. A grain of
``segments`` pieces of length ``segment_length`` then burns on that perimeter along its length, and, unless
inhibited, on both ends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SHAPES = ("BATES (round core)", "Star", "Finocyl (core + fins)", "Moon burner (offset core)", "C-slot")


def port_mask(shape: str, n: int, outer_d: float, core_d: float, points: int = 6, point_depth: float = 0.5,
              fin_width: float = 0.08, offset: float = 0.25) -> np.ndarray:
    """Boolean (n, n) raster of the initial port (True = empty) on the square around the grain."""
    r_out = 0.5 * outer_d
    xs = (np.arange(n) + 0.5) / n * outer_d - r_out
    X, Y = np.meshgrid(xs, xs)
    R = np.hypot(X, Y)
    TH = np.arctan2(Y, X)
    rc = 0.5 * core_d
    if shape.startswith("BATES"):
        return R <= rc
    if shape.startswith("Star"):
        # star port: radius oscillating between the core and core + depth between the points
        ro = rc + point_depth * (r_out - rc)
        k = points
        frac = np.abs(((TH * k / (2 * math.pi)) % 1.0) - 0.5) * 2.0       # 1 at a point, 0 between points
        return R <= rc + (ro - rc) * frac
    if shape.startswith("Finocyl"):
        fin_len = rc + point_depth * (r_out - rc)
        port = R <= rc
        for j in range(points):
            a = 2 * math.pi * j / points
            u = X * math.cos(a) + Y * math.sin(a)
            v = -X * math.sin(a) + Y * math.cos(a)
            port |= (u >= 0.0) & (u <= fin_len) & (np.abs(v) <= 0.5 * fin_width * outer_d)
        return port
    if shape.startswith("Moon"):
        return np.hypot(X - offset * (r_out - rc), Y) <= rc
    if shape.startswith("C-slot"):
        slot = (X >= 0.0) & (X <= r_out) & (np.abs(Y) <= 0.5 * fin_width * outer_d)
        return (R <= rc) | slot
    raise ValueError(f"Unknown grain shape {shape!r}")


def _contour_length(field: np.ndarray, level: float, inside: np.ndarray, h: float) -> float:
    """Length of the iso-line field = level within ``inside`` (marching squares, cell size h)."""
    a = field - level
    s = a > 0.0
    s00, s01, s10, s11 = s[:-1, :-1], s[:-1, 1:], s[1:, :-1], s[1:, 1:]
    ok = inside[:-1, :-1] & inside[:-1, 1:] & inside[1:, :-1] & inside[1:, 1:]
    mixed = ~((s00 == s01) & (s00 == s10) & (s00 == s11)) & ok
    j, i = np.nonzero(mixed)
    if j.size == 0:
        return 0.0
    a00, a01, a10, a11 = a[j, i], a[j, i + 1], a[j + 1, i], a[j + 1, i + 1]

    def frac(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.clip(np.nan_to_num(p / (p - q), nan=0.5), 0.0, 1.0)

    ex = np.stack([frac(a00, a01), np.ones_like(a00), frac(a10, a11), np.zeros_like(a00)], axis=1)
    ey = np.stack([np.zeros_like(a00), frac(a01, a11), np.ones_like(a00), frac(a00, a10)], axis=1)
    e = np.stack([s00[j, i] != s01[j, i], s01[j, i] != s11[j, i], s10[j, i] != s11[j, i], s00[j, i] != s10[j, i]], 1)
    total = 0.0
    two = e.sum(axis=1) == 2
    if two.any():
        idx = np.argsort(~e[two], axis=1, kind="stable")[:, :2]
        rows = np.arange(int(two.sum()))
        x1, y1 = ex[two][rows, idx[:, 0]], ey[two][rows, idx[:, 0]]
        x2, y2 = ex[two][rows, idx[:, 1]], ey[two][rows, idx[:, 1]]
        total += float(np.hypot(x2 - x1, y2 - y1).sum())
    four = e.sum(axis=1) == 4
    total += float(four.sum()) * 2.0 * 0.7071
    return total * h


@dataclass
class CrossSectionGrain:
    """A cylindrical grain of any port shape (same interface as :class:`core.solid_motor.BatesGrain`)."""

    shape: str
    outer_diameter: float
    core_diameter: float
    segment_length: float
    segments: int = 1
    inhibit_ends: bool = False
    points: int = 6
    point_depth: float = 0.5
    fin_width: float = 0.08
    offset: float = 0.25
    resolution: int = 360
    _w: np.ndarray = field(default=None, repr=False)
    _perim: np.ndarray = field(default=None, repr=False)
    _port: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        from scipy import ndimage

        n = self.resolution
        h = self.outer_diameter / n
        port = port_mask(self.shape, n, self.outer_diameter, self.core_diameter, self.points, self.point_depth,
                         self.fin_width, self.offset)
        xs = (np.arange(n) + 0.5) * h - 0.5 * self.outer_diameter
        X, Y = np.meshgrid(xs, xs)
        case = np.hypot(X, Y) <= 0.5 * self.outer_diameter
        port &= case
        d = ndimage.distance_transform_edt(~port) * h                     # regression distance from the port
        prop = case & ~port
        web = float(d[prop].max()) if prop.any() else 0.0
        ws = np.linspace(0.0, web, 120)
        area_case = math.pi * 0.25 * self.outer_diameter ** 2
        cell = h * h
        self._w = ws
        self._perim = np.array([_contour_length(d, w + 1e-9, case, h) for w in ws])
        self._perim[-1] = 0.0
        self._port = np.array([min(float(((d <= w) & case).sum()) * cell, area_case) for w in ws])
        self._area_case = area_case

    # -- the BatesGrain interface -------------------------------------------------------------------------
    @property
    def web(self) -> float:
        w = float(self._w[-1])
        return w if self.inhibit_ends else min(w, 0.5 * self.segment_length)

    def _at(self, arr: np.ndarray, w: float) -> float:
        return float(np.interp(w, self._w, arr))

    def burning_area(self, w: float) -> float:
        if w >= self._w[-1]:
            return 0.0
        length = self.segment_length - (0.0 if self.inhibit_ends else 2.0 * w)
        if length <= 0.0:
            return 0.0
        a = self._at(self._perim, w) * length
        if not self.inhibit_ends:
            a += 2.0 * (self._area_case - self._at(self._port, w))
        return self.segments * a

    def volume(self, w: float = 0.0) -> float:
        if w >= self._w[-1]:
            return 0.0
        length = self.segment_length - (0.0 if self.inhibit_ends else 2.0 * w)
        if length <= 0.0:
            return 0.0
        return self.segments * (self._area_case - self._at(self._port, w)) * length

    @property
    def port_area(self) -> float:
        return float(self._port[0])

    def perimeter_curve(self) -> tuple[np.ndarray, np.ndarray]:
        return self._w.copy(), self._perim.copy()

    def section_image(self, n: int = 160) -> np.ndarray:
        """(n, n) image for drawing: 0 outside, 1 propellant, 2 port."""
        h = self.outer_diameter / n
        port = port_mask(self.shape, n, self.outer_diameter, self.core_diameter, self.points, self.point_depth,
                         self.fin_width, self.offset)
        xs = (np.arange(n) + 0.5) * h - 0.5 * self.outer_diameter
        X, Y = np.meshgrid(xs, xs)
        case = np.hypot(X, Y) <= 0.5 * self.outer_diameter
        img = np.zeros((n, n))
        img[case] = 1.0
        img[case & port] = 2.0
        return img


__all__ = ["SHAPES", "port_mask", "CrossSectionGrain"]
