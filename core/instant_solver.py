"""Engine A - the Instant Aero Engine (pure NumPy / SciPy, no external CFD binaries).

Method (all vectorised; the volume-field superposition runs in the compiled kernels of
:mod:`core.accel` when numba is installed, typically well under a second):

1. **Surface discretisation.** The triangle mesh is sampled (curvature-adaptive) and
   clustered into *surfels*: small oriented discs with position, outward normal and area.
   This works for any triangle soup (no connectivity needed).
2. **Potential flow.** Each surfel carries a source strength. The Neumann (no-penetration)
   condition gives a dense second-kind system, LU-factored once per geometry and solved for any
   freestream direction. This gives the thickness flow (stagnation, shoulder suction) and the
   surface pressure coefficient ``Cp = 1 - |V|^2 / V_inf^2``. Verified against the exact sphere
   solution ``Cp = 1 - 9/4 sin^2(theta)``.
3. **Lift and induced drag.** Spanwise sections (chord, camber, thickness) are extracted from the
   mesh and fed to a Weissinger-type vortex lattice (horseshoe vortices on the quarter-chord line,
   control points at three quarters, trailing legs along the freestream). Section zero-lift angle
   and pitching moment come from thin-airfoil theory applied to the extracted camber line.
   Post-stall behaviour blends to a flat-plate model.
4. **Viscous corrections.** Prandtl-Schlichting skin friction with a laminar/turbulent split,
   a Hoerner form factor, induced drag from the lattice, and a leeward-surface separation model
   that raises drag for bluff shapes and for wings approaching stall.
5. **Volume field.** On a structured Cartesian grid the velocity is the superposition of the
   freestream, the source-surfel field, and the vortex-lattice field, followed by a smooth
   surface-normal tangential projection near the body, zero velocity inside it, and a
   momentum-conserving viscous wake deficit.

The result is a :class:`~core.results.SolverResult` with the same structure as the OpenFOAM
engine. This is an *engineering preview*: Mach is limited to the Prandtl-Glauert range and
massively separated or transonic flows are only approximated.

Thread-safety: :class:`InstantSolver` holds no GUI objects and may run in any worker thread; the
only shared state is a small LRU preparation cache guarded by a lock.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Final, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lu_factor, lu_solve
from scipy.spatial import cKDTree

from config.settings import (
    DOMAIN_DOWNSTREAM, DOMAIN_LATERAL, DOMAIN_UPSTREAM, GRID_RESOLUTIONS,
    STALL_ALPHA_DEG, TRANSITION_REYNOLDS, FlowConditions, GridQuality,
)
from core import accel
from core.geometry_manager import GeometryError, GeometryInfo, GeometryModel
from core.results import AeroKPIs, ChordCp, FlowField, SolverResult, SurfaceField
from utils.validators import reynolds_number, validate_flow_conditions

log = logging.getLogger("aeroflow.core.instant")

_trapz = getattr(np, "trapezoid", None) or np.trapz   # NumPy 2.x name, 1.x fallback

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ProgressFn = Callable[[float, str], None]
CancelFn = Callable[[], bool]

FOUR_PI: Final[float] = 4.0 * math.pi
SURFEL_COUNTS: Final[dict[GridQuality, int]] = {
    GridQuality.COARSE: 700, GridQuality.MEDIUM: 1100, GridQuality.FINE: 1700,
}
NORMAL_FEATURE_WEIGHT: Final[float] = 1.5   # normal-vs-position weight when clustering surfels
CHORD_SHAPE_EPS: Final[float] = 0.015        # leading-edge softening of the lifting pressure shape
LIFT_FILAMENT_EDGES: Final[tuple[float, ...]] = (0.0, 0.08, 0.22, 0.42, 0.65, 1.0)
SEPARATION_ANGLE_START_DEG: Final[float] = 16.0
SEPARATION_ANGLE_FULL_DEG: Final[float] = 30.0
#: Normalisation of the chordwise lifting-pressure shape sqrt((1-x)/(x+eps)) (was recomputed per call).
_shape_grid = np.linspace(0.0, 1.0, 2001)
_CHORD_SHAPE_NORM: Final[float] = float(_trapz(np.sqrt((1.0 - _shape_grid) / (_shape_grid + CHORD_SHAPE_EPS)),
                                               _shape_grid))
del _shape_grid


class SolverCancelled(Exception):
    """Raised inside a solve when the cancel callback returns True."""


def karman_tsien(cp0: FloatArray, mach: float) -> FloatArray:
    """Kármán-Tsien compressibility correction of incompressible pressure coefficients.

    ``Cp = Cp0 / (β + M²/(1+β) · Cp0/2)`` with ``β = sqrt(1 - M²)``. More accurate than Prandtl-Glauert
    (``Cp0/β``) at high suction; valid below the critical Mach number. M < 0.05 returns ``cp0``."""
    m = float(min(max(mach, 0.0), 0.95))
    if m < 0.05:
        return cp0
    b = math.sqrt(1.0 - m * m)
    den = b + (m * m / (1.0 + b)) * 0.5 * cp0
    return cp0 / np.maximum(den, 0.25 * b)          # guard: the rule breaks down near Cp -> -2β(1+β)/M²


def critical_cp(mach: float, gamma: float = 1.4) -> float:
    """Pressure coefficient at which the local flow reaches Mach 1 (isentropic). -inf for M = 0."""
    m = float(mach)
    if m <= 1e-6:
        return float("-inf")
    g = gamma
    return 2.0 / (g * m * m) * (((2.0 + (g - 1.0) * m * m) / (g + 1.0)) ** (g / (g - 1.0)) - 1.0)


def _smoothstep(x: FloatArray | float) -> FloatArray | float:
    """Smooth 0 -> 1 ramp on [0, 1]."""
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# Wing / section analysis
# ---------------------------------------------------------------------------
@dataclass
class _Sections:
    """Spanwise section data extracted from the surface samples."""

    y_edges: FloatArray
    y_c: FloatArray
    dy: FloatArray
    chord: FloatArray
    x_le: FloatArray
    z_ref: FloatArray
    alpha0: FloatArray        # zero-lift angle, rad
    cm0: FloatArray           # section pitching moment about c/4 (thin-airfoil theory)
    tc: FloatArray            # thickness / chord
    valid: NDArray[np.bool_]
    x_qc_edge: FloatArray = field(default_factory=lambda: np.zeros(0))
    z_edge: FloatArray = field(default_factory=lambda: np.zeros(0))
    x_le_edge: FloatArray = field(default_factory=lambda: np.zeros(0))
    chord_edge: FloatArray = field(default_factory=lambda: np.zeros(0))

    @property
    def n_valid(self) -> int:
        """Number of usable stations."""
        return int(self.valid.sum())

    @property
    def planform_area(self) -> float:
        """Sum of chord x width over valid stations."""
        return float((self.chord * self.dy)[self.valid].sum())

    @property
    def mean_alpha0(self) -> float:
        """Chord-weighted mean zero-lift angle (rad)."""
        w = (self.chord * self.dy)[self.valid]
        return float((self.alpha0[self.valid] * w).sum() / max(w.sum(), 1e-12))

    @property
    def mean_tc(self) -> float:
        """Chord-weighted mean thickness ratio."""
        w = (self.chord * self.dy)[self.valid]
        return float((self.tc[self.valid] * w).sum() / max(w.sum(), 1e-12))


def _analyse_sections(pts: FloatArray, chord_ref: float, n_bins: int = 24,
                      n_chord: int = 16) -> Optional[_Sections]:
    """Estimate chord, camber and thickness per spanwise bin from dense surface samples."""
    y = pts[:, 1]
    y0, y1 = float(y.min()), float(y.max())
    span = y1 - y0
    if span < 1e-6 * max(chord_ref, 1e-9):
        return None
    k = np.arange(n_bins + 1)
    edges = y0 + 0.5 * span * (1.0 - np.cos(math.pi * k / n_bins))
    which = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, n_bins - 1)
    order = np.argsort(which, kind="stable")
    counts = np.bincount(which, minlength=n_bins)
    starts = np.concatenate([[0], np.cumsum(counts)])
    xi_edges = 0.5 * (1.0 - np.cos(math.pi * np.arange(n_chord + 1) / n_chord))

    chord = np.zeros(n_bins)
    x_le = np.zeros(n_bins)
    z_ref = np.zeros(n_bins)
    alpha0 = np.zeros(n_bins)
    cm0 = np.zeros(n_bins)
    tc = np.zeros(n_bins)
    valid = np.zeros(n_bins, dtype=bool)
    theta = np.linspace(0.0, math.pi, 97)
    xi_theta = 0.5 * (1.0 - np.cos(theta))

    for j in range(n_bins):
        sel = order[starts[j]:starts[j + 1]]
        if len(sel) < 40:
            continue
        x, z = pts[sel, 0], pts[sel, 2]
        xa, xb = float(x.min()), float(x.max())
        c = xb - xa
        if c < 1e-3 * chord_ref:
            continue
        xi = (x - xa) / c
        b = np.clip(np.searchsorted(xi_edges, xi, side="right") - 1, 0, n_chord - 1)
        cnt = np.bincount(b, minlength=n_chord)
        zu = np.full(n_chord, -np.inf)
        zl = np.full(n_chord, np.inf)
        np.maximum.at(zu, b, z)
        np.minimum.at(zl, b, z)
        xi_mean = np.bincount(b, weights=xi, minlength=n_chord) / np.maximum(cnt, 1)
        ok = cnt >= 3
        chord[j], x_le[j], valid[j] = c, xa, True
        zu_f = np.where(np.isfinite(zu), zu, 0.0)
        zl_f = np.where(np.isfinite(zl), zl, 0.0)
        z_mid = np.where(cnt >= 1, 0.5 * (zu_f + zl_f), 0.0)
        tc[j] = float(np.max((zu - zl)[ok]) / c) if ok.any() else 0.1
        z_ref[j] = float(z_mid[ok].mean()) if ok.any() else float(z.mean())
        if ok.sum() >= 8:
            coef = np.polyfit(xi_mean[ok], z_mid[ok], 4)
            z_ref[j] = float(np.polyval(coef, 0.25))
            dzdx = np.polyval(np.polyder(coef), xi_theta) / c
            a0 = -_trapz(dzdx * (np.cos(theta) - 1.0), theta) / math.pi
            a1 = 2.0 * _trapz(dzdx * np.cos(theta), theta) / math.pi
            a2 = 2.0 * _trapz(dzdx * np.cos(2.0 * theta), theta) / math.pi
            alpha0[j] = float(np.clip(a0, -0.25, 0.25))
            cm0[j] = float(np.clip(0.25 * math.pi * (a2 - a1), -0.3, 0.3))
    if valid.sum() < 3:
        return None
    y_c = 0.5 * (edges[:-1] + edges[1:])
    dy = np.diff(edges)
    sec = _Sections(edges, y_c, dy, chord, x_le, z_ref, alpha0, cm0, tc, valid)
    yv = y_c[valid]
    sec.x_qc_edge = np.interp(edges, yv, (x_le + 0.25 * chord)[valid])
    sec.x_le_edge = np.interp(edges, yv, x_le[valid])
    sec.chord_edge = np.interp(edges, yv, chord[valid])
    sec.z_edge = np.interp(edges, yv, z_ref[valid])
    return sec


# ---------------------------------------------------------------------------
# Vortex lattice kernels
# ---------------------------------------------------------------------------
def _horseshoe_terms(X: FloatArray, A: FloatArray, B: FloatArray, dhat: FloatArray,
                     eps2: float) -> tuple[FloatArray, FloatArray]:
    """Velocity at points ``X`` (P,3) due to K unit-strength horseshoes A->B with legs along ``dhat``.

    Returns ``(bound, legs)``, each (P, K, 3). The core parameter ``eps2`` (length^2) regularises
    the Biot-Savart singularity on the filaments. (Implemented in :mod:`core.accel` with the cross
    products written per component.)
    """
    return accel.horseshoe_terms(X, A, B, dhat, eps2)


@dataclass
class _VortexSystem:
    """Result of the lifting solution."""

    gamma: FloatArray               # (n_valid,) circulation / V (m), stall-scaled
    stations: IntArray              # indices into the sections arrays
    a: FloatArray                   # (n_valid, 3) bound-segment starts (quarter chord)
    b: FloatArray                   # (n_valid, 3) bound-segment ends
    cl_linear: float
    cl: float
    cd_induced: float
    cm_lift: float
    cm_camber: float
    cy: float
    scale: float                    # stall scale applied to gamma
    s_stall: float                  # 0 attached ... 1 fully stalled
    alpha_eff_rad: float
    alpha_stall_rad: float


def _cf_flat_plate(re: float) -> float:
    """Prandtl-Schlichting mean skin-friction coefficient with a transition correction."""
    re = max(re, 1.0e3)
    lam = 1.328 / math.sqrt(re)
    if re <= TRANSITION_REYNOLDS:
        return lam
    turb = 0.455 / (math.log10(re) ** 2.58) - 1700.0 / re
    return max(turb, 0.5 * lam)


def _form_factor(info: GeometryInfo, tc_mean: float) -> float:
    """Hoerner form factor: thin-wing formula for wing-like shapes, slender-body formula otherwise."""
    c_mean = info.planform_area_m2 / max(info.span_m, 1e-9)
    wing_like = info.aspect_ratio >= 1.2 and info.height_m < 0.35 * max(c_mean, 1e-9)
    if wing_like:
        t = float(np.clip(tc_mean, 0.02, 0.35))
        return 1.0 + 2.0 * t + 60.0 * t ** 4
    d_eq = math.sqrt(4.0 * max(info.frontal_area_m2, 1e-12) / math.pi)
    f = max(info.chord_m / max(d_eq, 1e-9), 3.0)
    return 1.0 + 1.5 / f ** 1.5 + 7.0 / f ** 3


# ---------------------------------------------------------------------------
# Prepared geometry
# ---------------------------------------------------------------------------
@dataclass
class PreparedGeometry:
    """Everything that depends on the geometry but not on the flow condition."""

    model: GeometryModel
    quality: GridQuality
    info: GeometryInfo
    # dense surface samples (SDF queries, section analysis)
    sample_pts: FloatArray
    sample_nrm: FloatArray
    # surfels
    pos: FloatArray
    nrm: FloatArray
    area: FloatArray
    a2: FloatArray
    gx: FloatArray
    gy: FloatArray
    gz: FloatArray
    lu: tuple[FloatArray, IntArray]
    # vertex -> surfel interpolation
    v_idx: IntArray
    v_w: FloatArray
    vertex_normals: FloatArray
    # sections and viscous constants
    sections: Optional[_Sections]
    tc_mean: float
    form_factor: float
    outward_sign: float
    nbr: IntArray
    # Linear bases: the source strengths are exactly linear in the freestream direction, so
    # sigma(dhat) = sig_basis @ dhat and the induced surface velocity = vel_basis @ dhat. This
    # replaces an LU solve plus three m x m mat-vecs per angle of attack with two tiny products.
    sig_basis: Optional[FloatArray] = None          # (m, 3)
    vel_basis: Optional[FloatArray] = None          # (m, 3, 3): [surfel, velocity component, dhat component]
    prep_seconds: float = 0.0
    _tree: Optional[cKDTree] = None
    _tree_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def n_surfels(self) -> int:
        """Number of surface elements."""
        return int(len(self.pos))

    @property
    def tree(self) -> cKDTree:
        """KD-tree of the dense surface samples (built lazily, thread-safe)."""
        with self._tree_lock:
            if self._tree is None:
                self._tree = cKDTree(self.sample_pts)
            return self._tree


def _curvature_density(model: GeometryModel) -> FloatArray:
    """Per-face sampling density multiplier (>= 1) that concentrates surfels on curved regions."""
    vn = model.vertex_normals
    fn = model.face_normals
    dev = 1.0 - np.einsum("fkj,fj->fk", vn[model.faces], fn)
    kappa = np.clip(dev.mean(axis=1), 0.0, 1.0)
    return 1.0 + 3.0 * np.clip(kappa / 0.05, 0.0, 4.0)


def _weighted_samples(model: GeometryModel, n: int, seed: int
                      ) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Curvature-weighted stratified surface samples with the area each sample represents."""
    tri = model.vertices[model.faces]
    area = model.face_areas
    g = _curvature_density(model)
    w = area * g
    total_w = float(w.sum())
    if not np.isfinite(total_w) or total_w <= 0.0:
        raise GeometryError("The surface has zero area.")
    cdf = np.cumsum(w) / total_w
    rng = np.random.default_rng(seed)
    u = (np.arange(n) + rng.random(n)) / n
    fid = np.minimum(np.searchsorted(cdf, u), len(area) - 1)
    s = np.sqrt(rng.random(n))
    r2 = rng.random(n)
    pts = ((1.0 - s)[:, None] * tri[fid, 0] + (s * (1.0 - r2))[:, None] * tri[fid, 1]
           + (s * r2)[:, None] * tri[fid, 2])
    return pts, model.face_normals[fid], total_w / (n * g[fid])


def _make_surfels(pts: FloatArray, nrm: FloatArray, sample_area: FloatArray, m_target: int,
                  seed: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Cluster surface samples into surfels (position on the surface, unit normal, area)."""
    total = float(sample_area.sum())
    m = int(min(m_target, max(len(pts) // 4, 8)))
    ell = math.sqrt(total / m)
    feat = np.hstack([pts / ell, NORMAL_FEATURE_WEIGHT * nrm])
    rng = np.random.default_rng(seed)
    centers = feat[rng.choice(len(feat), size=m, replace=False)]
    labels = np.zeros(len(feat), dtype=np.int64)
    for _ in range(8):
        _, labels = cKDTree(centers).query(feat, workers=-1)
        cnt = np.bincount(labels, minlength=m)
        new = np.column_stack([np.bincount(labels, weights=feat[:, k], minlength=m)
                               for k in range(feat.shape[1])])
        empty = cnt == 0
        new[~empty] /= cnt[~empty, None]
        if empty.any():
            new[empty] = feat[rng.choice(len(feat), size=int(empty.sum()), replace=False)]
        centers = new
    _, labels = cKDTree(centers).query(feat, workers=-1)
    cnt = np.bincount(labels, minlength=m)
    keep = cnt > 0
    mean_pos = np.column_stack([np.bincount(labels, weights=pts[:, k], minlength=m) for k in range(3)])
    mean_pos[keep] /= cnt[keep, None]
    d2 = ((pts - mean_pos[labels]) ** 2).sum(1)
    order = np.lexsort((d2, labels))
    first = np.searchsorted(labels[order], np.arange(m))
    first = np.minimum(first, len(order) - 1)
    snap = order[first]
    pos = pts[snap]
    nsum = np.column_stack([np.bincount(labels, weights=nrm[:, k] * sample_area, minlength=m)
                            for k in range(3)])
    nlen = np.linalg.norm(nsum, axis=1, keepdims=True)
    normals = np.where(nlen > 1e-9, nsum / np.maximum(nlen, 1e-300), nrm[snap])
    area = np.bincount(labels, weights=sample_area, minlength=m)
    return pos[keep], normals[keep], area[keep]


def _influence_matrices(pos: FloatArray, nrm: FloatArray, area: FloatArray, a2: FloatArray
                        ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Velocity-influence matrices ``G{x,y,z}`` (per unit source strength) and the normal system matrix."""
    m = len(pos)
    gx, gy, gz = (np.empty((m, m)) for _ in range(3))
    weight = area / FOUR_PI
    block = 256
    for i0 in range(0, m, block):
        i1 = min(i0 + block, m)
        dx = pos[i0:i1, 0, None] - pos[None, :, 0]
        dy = pos[i0:i1, 1, None] - pos[None, :, 1]
        dz = pos[i0:i1, 2, None] - pos[None, :, 2]
        r2 = dx * dx + dy * dy + dz * dz + a2[None, :]
        inv = weight[None, :] / (r2 * np.sqrt(r2))
        inv[np.arange(i1 - i0), np.arange(i0, i1)] = 0.0
        gx[i0:i1], gy[i0:i1], gz[i0:i1] = inv * dx, inv * dy, inv * dz
    k = nrm[:, 0, None] * gx + nrm[:, 1, None] * gy + nrm[:, 2, None] * gz
    k[np.arange(m), np.arange(m)] += 0.5
    return gx, gy, gz, k


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GridSpec:
    """Structured solution grid."""

    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    dims: tuple[int, int, int]
    domain_length: float

    @property
    def point_count(self) -> int:
        """Total number of grid points."""
        return self.dims[0] * self.dims[1] * self.dims[2]

    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """``(xmin, xmax, ymin, ymax, zmin, zmax)``."""
        o, s, n = self.origin, self.spacing, self.dims
        return (o[0], o[0] + s[0] * (n[0] - 1), o[1], o[1] + s[1] * (n[1] - 1),
                o[2], o[2] + s[2] * (n[2] - 1))


def grid_spec(info: GeometryInfo, quality: GridQuality) -> GridSpec:
    """Grid extending ``2.5 L`` upstream, ``5 L`` downstream and ``2.5 L`` laterally / vertically.

    ``L`` is the mean chord, but never less than 20 % of the largest body dimension so that
    long slender vehicles keep a usable resolution. Cells are close to isotropic.
    """
    lo = np.asarray(info.bounds_min)
    hi = np.asarray(info.bounds_max)
    c_mean = float(np.clip(info.planform_area_m2 / max(info.span_m, 1e-9), 0.3 * info.chord_m, info.chord_m))
    big = max(info.chord_m, info.span_m, info.height_m)
    length = max(c_mean, 0.2 * big)
    p0 = lo - np.array([DOMAIN_UPSTREAM, DOMAIN_LATERAL, DOMAIN_LATERAL]) * length
    p1 = hi + np.array([DOMAIN_DOWNSTREAM, DOMAIN_LATERAL, DOMAIN_LATERAL]) * length
    ext = p1 - p0
    budget = float(np.prod(GRID_RESOLUTIONS[quality]))
    h = (float(np.prod(ext)) / budget) ** (1.0 / 3.0)
    dims = np.clip(np.round(ext / h).astype(int) + 1, 12, 220)
    spacing = ext / (dims - 1)
    return GridSpec(tuple(float(v) for v in p0), tuple(float(v) for v in spacing),
                    tuple(int(v) for v in dims), float(length))


# ---------------------------------------------------------------------------
# Internal aero state
# ---------------------------------------------------------------------------
@dataclass
class _Aero:
    """Per-condition aerodynamic solution (no volume grid)."""

    kpis: AeroKPIs
    dhat: FloatArray
    lhat: FloatArray
    sigma: FloatArray
    cp_surfel: FloatArray
    cp_potential: FloatArray
    vortex: Optional[_VortexSystem]
    cd_profile: float
    warnings: list[str]


PREP_CACHE_SIZE: Final[int] = 4    # geometry preparations kept (model x quality)


class InstantSolver:
    """The Instant Aero Engine. Create one instance and reuse it between runs (it is thread-safe)."""

    def __init__(self, warmup: bool = True) -> None:
        self._lock = threading.Lock()
        # Most-recently-used last. Keyed by model identity *and* quality, so a polar sweep at one
        # quality and a field solve at another no longer evict each other's preparation.
        self._cache: list[PreparedGeometry] = []
        if warmup:
            accel.warmup_async()

    def _cached(self, model: GeometryModel, quality: GridQuality) -> Optional[PreparedGeometry]:
        with self._lock:
            for i, p in enumerate(self._cache):
                if p.model is model and p.quality == quality:
                    self._cache.append(self._cache.pop(i))
                    return p
        return None

    def _remember(self, prep: PreparedGeometry) -> None:
        with self._lock:
            self._cache = [p for p in self._cache if not (p.model is prep.model and p.quality == prep.quality)]
            self._cache.append(prep)
            del self._cache[:-PREP_CACHE_SIZE]

    def clear_cache(self) -> None:
        """Forget all geometry preparations (frees memory)."""
        with self._lock:
            self._cache.clear()

    # -- preparation -----------------------------------------------------
    def prepare(self, model: GeometryModel, quality: GridQuality = GridQuality.MEDIUM,
                progress: Optional[ProgressFn] = None,
                cancel: Optional[CancelFn] = None) -> PreparedGeometry:
        """Run all geometry-dependent preprocessing (cached per model and quality).

        Raises:
            GeometryError: if the geometry cannot be discretised.
            SolverCancelled: if ``cancel()`` returns True.
        """
        hit = self._cached(model, quality)
        if hit is not None:
            return hit
        t0 = time.perf_counter()

        def tick(frac: float, msg: str) -> None:
            if cancel and cancel():
                raise SolverCancelled()
            if progress:
                progress(frac, msg)

        info = model.info
        tick(0.02, "Sampling surface")
        m_target = SURFEL_COUNTS[quality]
        n_samples = int(np.clip(25 * m_target, 20_000, 80_000))
        pts, nrm, s_area = _weighted_samples(model, n_samples, seed=17)
        centre = model.bbox_center
        outward = float(np.sum(s_area * np.einsum("ij,ij->i", nrm, pts - centre)))
        sign = -1.0 if outward < 0.0 else 1.0
        if sign < 0.0:
            nrm = -nrm
            log.warning("Surface normals point inward on average; flipped for the solver.")
        tick(0.12, "Building surface elements")
        pos, snrm, sarea = _make_surfels(pts, nrm, s_area, m_target, seed=23)
        a2 = sarea / math.pi
        tick(0.22, "Assembling influence matrices")
        gx, gy, gz, k = _influence_matrices(pos, snrm, sarea, a2)
        tick(0.34, "Factorising system")
        lu = lu_factor(k, check_finite=False)
        del k
        sig_basis = lu_solve(lu, -snrm, check_finite=False)                       # (m, 3)
        vel_basis = np.stack([gx @ sig_basis, gy @ sig_basis, gz @ sig_basis], axis=1)   # (m, 3, 3)
        tick(0.45, "Analysing wing sections")
        sections = _analyse_sections(pts, info.chord_m)
        tc_mean = sections.mean_tc if sections is not None else 0.12
        ff = _form_factor(info, tc_mean)
        tick(0.52, "Mapping surface to mesh")
        ell = math.sqrt(float(sarea.sum()) / len(pos))
        vnorm = model.vertex_normals * sign
        surf_feat = np.hstack([pos / ell, NORMAL_FEATURE_WEIGHT * snrm])
        vert_feat = np.hstack([model.vertices / ell, NORMAL_FEATURE_WEIGHT * vnorm])
        kk = min(4, len(pos))
        surf_tree = cKDTree(surf_feat)                     # built once, used for both queries
        dist, idx = surf_tree.query(vert_feat, k=kk, workers=-1)
        dist = dist.reshape(len(vert_feat), kk)
        idx = idx.reshape(len(vert_feat), kk)
        w = 1.0 / (dist * dist + 0.3 ** 2)
        w /= w.sum(axis=1, keepdims=True)
        nbr = surf_tree.query(surf_feat, k=min(7, len(pos)))[1]
        prep = PreparedGeometry(
            model=model, quality=quality, info=info, sample_pts=pts, sample_nrm=nrm, pos=pos, nrm=snrm,
            area=sarea, a2=a2, gx=gx, gy=gy, gz=gz, lu=lu, v_idx=idx.astype(np.int64), v_w=w,
            vertex_normals=vnorm, sections=sections, tc_mean=tc_mean, form_factor=ff, outward_sign=sign,
            nbr=np.asarray(nbr, dtype=np.int64).reshape(len(pos), -1),
            sig_basis=sig_basis, vel_basis=vel_basis, prep_seconds=time.perf_counter() - t0)
        self._remember(prep)
        tick(0.6, "Geometry ready")
        log.info("Prepared %s: %d surfels, %d sections, %.2fs", model.name, prep.n_surfels,
                 sections.n_valid if sections else 0, prep.prep_seconds)
        return prep

    # -- lifting solution ------------------------------------------------
    def _vortex_lattice(self, prep: PreparedGeometry, dhat: FloatArray, alpha: float, beta: float,
                        mach: float, s_ref: float, l_ref: float, ground_h_over_b: float = 0.0
                        ) -> Optional[_VortexSystem]:
        """Weissinger-type lifting solution including stall blending.

        ``ground_h_over_b`` > 0 adds a ground plane that distance (times the span) below the lifting line:
        every horseshoe gets a mirror image of opposite strength (image method), which is exactly the
        flat-ground boundary condition. It raises the lift slope and cuts the induced drag."""
        sec = prep.sections
        if sec is None or sec.n_valid < 3:
            return None
        idx = np.where(sec.valid)[0]
        n = len(idx)
        a = np.column_stack([sec.x_qc_edge[idx], sec.y_edges[idx], sec.z_edge[idx]])
        b = np.column_stack([sec.x_qc_edge[idx + 1], sec.y_edges[idx + 1], sec.z_edge[idx + 1]])
        ctrl = np.column_stack([sec.x_le[idx] + 0.75 * sec.chord[idx], sec.y_c[idx], sec.z_ref[idx]])
        c_mean = float(np.average(sec.chord[idx], weights=sec.dy[idx]))
        zhat = np.array([0.0, 0.0, 1.0])
        xhat = np.array([1.0, 0.0, 0.0])   # planar (linearised) trailing legs; keeps the lattice well conditioned

        bound, legs = _horseshoe_terms(ctrl, a, b, xhat, eps2=(0.002 * c_mean) ** 2)
        w_mat = ((bound + legs) @ zhat)                       # (n, n): control i, horseshoe j
        a_img = b_img = None
        if ground_h_over_b > 1e-6:
            span = float(np.max(b[:, 1]) - np.min(a[:, 1]))
            z_ground = float(np.mean(0.5 * (a[:, 2] + b[:, 2]))) - ground_h_over_b * max(span, 1e-9)
            a_img, b_img = a.copy(), b.copy()
            a_img[:, 2] = 2.0 * z_ground - a[:, 2]
            b_img[:, 2] = 2.0 * z_ground - b[:, 2]
            bi, li = _horseshoe_terms(ctrl, a_img, b_img, xhat, eps2=(0.002 * c_mean) ** 2)
            w_mat = w_mat - ((bi + li) @ zhat)               # images carry -gamma
        # Thick, bluff sections do not generate planform lift (a sphere must have CL = 0).
        lift_eff = 1.0 - _smoothstep((sec.tc[idx] - 0.45) / 0.45)
        a_eff = alpha - sec.alpha0[idx]
        v_n = lift_eff * math.cos(beta) * np.sin(a_eff)
        try:
            gamma = np.linalg.solve(w_mat + 1e-9 * np.eye(n), -v_n)
        except np.linalg.LinAlgError:
            gamma = np.linalg.lstsq(w_mat, -v_n, rcond=None)[0]
        pg = 1.0 / math.sqrt(max(1.0 - min(mach, 0.8) ** 2, 0.2))
        gamma = gamma * pg

        mid = 0.5 * (a + b)
        bnd_m, leg_m = _horseshoe_terms(mid, a, b, xhat, eps2=(0.002 * c_mean) ** 2)
        bnd_m[np.arange(n), np.arange(n)] = 0.0               # no self-induction of a bound segment
        v_ind = np.einsum("pkc,k->pc", bnd_m + leg_m, gamma)
        if a_img is not None:                                 # the ground's images also induce at the wing
            bnd_i, leg_i = _horseshoe_terms(mid, a_img, b_img, xhat, eps2=(0.002 * c_mean) ** 2)
            v_ind = v_ind - np.einsum("pkc,k->pc", bnd_i + leg_i, gamma)
        v_loc = dhat[None, :] + v_ind
        force = gamma[:, None] * np.cross(v_loc, b - a)
        f_tot = force.sum(axis=0)
        lhat = zhat - (zhat @ dhat) * dhat
        lhat /= max(np.linalg.norm(lhat), 1e-12)
        shat = np.cross(lhat, dhat)
        k_c = 2.0 / max(s_ref, 1e-12)
        cl_lin = float(k_c * (f_tot @ lhat))
        cd_i = float(k_c * (f_tot @ dhat))
        cy = float(k_c * (f_tot @ shat))
        ref = np.asarray(prep.info.moment_ref_point)
        r = mid - ref
        my = float((r[:, 2] * force[:, 0] - r[:, 0] * force[:, 2]).sum())
        cm_lift = 2.0 * my / max(s_ref * l_ref, 1e-12)
        cm_camber = float((sec.cm0[idx] * sec.chord[idx] ** 2 * sec.dy[idx]).sum()
                          / max(s_ref * l_ref, 1e-12)) * pg

        # Stall: smooth blend of the linear lift toward a flat-plate normal-force model.
        alpha_eff = alpha - sec.mean_alpha0
        a_stall = math.radians(float(np.clip(STALL_ALPHA_DEG - 5.0 + 40.0 * (prep.tc_mean - 0.05), 10.0, 18.0)))
        s_stall = float(_smoothstep((abs(alpha_eff) - a_stall) / math.radians(6.0)))
        cl_flat = 1.1 * math.sin(2.0 * alpha_eff)
        cl_final = (1.0 - s_stall) * cl_lin + s_stall * cl_flat
        scale = cl_final / cl_lin if abs(cl_lin) > 1e-5 else 1.0
        scale = float(np.clip(scale, -3.0, 3.0))
        # Post-stall pitch break: once the flow separates the centre of pressure moves from the
        # quarter chord toward ~0.42 c (flat-plate value), giving the classic nose-down moment.
        # Attached-flow CM about the aerodynamic centre stays (correctly) nearly constant.
        cn = cl_final * math.cos(alpha_eff)
        cm_stall = -s_stall * (0.42 - 0.25) * cn * c_mean / max(l_ref, 1e-12)
        return _VortexSystem(
            gamma=gamma * scale, stations=idx, a=a, b=b, cl_linear=cl_lin, cl=cl_final,
            # scale == 1 while attached; post-stall the induced drag can only fall (it used to grow
            # with scale^2 up to 9x when the linear CL was small or negative).
            cd_induced=cd_i * min(scale * scale, 1.0),
            cm_lift=cm_lift * scale + cm_stall, cm_camber=cm_camber * (1.0 - s_stall), cy=cy * scale,
            scale=scale, s_stall=s_stall, alpha_eff_rad=alpha_eff, alpha_stall_rad=a_stall)

    # -- per-condition aerodynamics -------------------------------------
    def _aero(self, prep: PreparedGeometry, cond: FlowConditions) -> _Aero:
        """Compute coefficients, surfel pressures and the lifting solution (no grid)."""
        cond, check = validate_flow_conditions(cond)
        warnings = [i.message for i in check.warnings]
        info, atm = prep.info, cond.atmosphere
        v_inf = cond.velocity_ms
        q = 0.5 * atm.density * v_inf * v_inf
        dhat = np.asarray(cond.freestream_direction())
        alpha, beta = math.radians(cond.alpha_deg), math.radians(cond.beta_deg)
        s_ref, l_ref = info.reference_area_m2, info.chord_m
        re = reynolds_number(atm.density, v_inf, l_ref, atm.dynamic_viscosity)

        if prep.sig_basis is not None and prep.vel_basis is not None:
            sigma = prep.sig_basis @ dhat
            vel = dhat[None, :] + prep.vel_basis @ dhat
        else:
            sigma = lu_solve(prep.lu, -(prep.nrm @ dhat), check_finite=False)
            vel = dhat[None, :] + np.column_stack([prep.gx @ sigma, prep.gy @ sigma, prep.gz @ sigma])
        vel -= ((vel * prep.nrm).sum(1))[:, None] * prep.nrm          # tangential surface velocity
        cp_pot = np.clip(1.0 - (vel * vel).sum(1), -8.0, 1.0)
        cp_pot = 0.5 * cp_pot + 0.5 * cp_pot[prep.nbr].mean(axis=1)   # damp discretisation noise
        cp_pot = karman_tsien(cp_pot, cond.mach)          # thickness pressures at the flight Mach number

        vortex = self._vortex_lattice(prep, dhat, alpha, beta, cond.mach, s_ref, l_ref,
                                      float(getattr(cond, "ground_h_over_b", 0.0)))
        zhat = np.array([0.0, 0.0, 1.0])
        lhat = zhat - (zhat @ dhat) * dhat
        lhat /= max(np.linalg.norm(lhat), 1e-12)

        cp = cp_pot.copy()
        sec = prep.sections
        if vortex is not None and sec is not None and abs(vortex.cl) > 1e-9:
            cp -= self._lifting_increment(prep, vortex, lhat, sec)

        # Leeward-surface separation: replace pressure on strongly diverging surfaces by base pressure.
        n_dot_d = prep.nrm @ dhat
        psi = np.degrees(np.arcsin(np.clip(n_dot_d, -1.0, 1.0)))
        sep = _smoothstep((psi - SEPARATION_ANGLE_START_DEG)
                          / (SEPARATION_ANGLE_FULL_DEG - SEPARATION_ANGLE_START_DEG)) * (n_dot_d > 0.0)
        log_re = math.log10(max(re, 1.0))
        cp_base = -0.45 + 0.30 * float(_smoothstep((log_re - 5.3) / 0.5))
        cd_sep = float(np.sum(sep * (cp - cp_base) * n_dot_d * prep.area) / max(s_ref, 1e-12))
        cd_sep = float(np.clip(cd_sep, 0.0, 3.0))
        cp_final = np.clip((1.0 - sep) * cp + sep * cp_base, -8.0, 1.0)

        cd_f = _cf_flat_plate(re) * prep.form_factor * info.wetted_area_m2 / max(s_ref, 1e-12)
        cl = cd_i = cm = 0.0
        cd_stall = 0.0
        stalled, a_stall_deg = False, STALL_ALPHA_DEG
        if vortex is not None:
            cl, cd_i = vortex.cl, vortex.cd_induced
            cm = vortex.cm_lift + vortex.cm_camber
            cd_stall = vortex.s_stall * 1.1 * math.sin(vortex.alpha_eff_rad) ** 2
            stalled = vortex.s_stall > 0.05
            a_stall_deg = math.degrees(vortex.alpha_stall_rad) + math.degrees(sec.mean_alpha0 if sec else 0.0)
        cd = cd_f + cd_i + cd_sep + cd_stall

        v_max = v_inf * math.sqrt(max(1.0 - float(cp_final.min()), 0.0))
        kpis = AeroKPIs(
            alpha_deg=cond.alpha_deg, beta_deg=cond.beta_deg, velocity_ms=v_inf, cl=float(cl), cd=float(cd),
            cm=float(cm), cd_friction=float(cd_f), cd_induced=float(cd_i), cd_pressure=float(cd_sep + cd_stall),
            lift_n=float(cl * q * s_ref), drag_n=float(cd * q * s_ref), moment_nm=float(cm * q * s_ref * l_ref),
            dynamic_pressure_pa=float(q), stagnation_pressure_pa=float(atm.pressure_pa + q),
            static_pressure_pa=float(atm.pressure_pa), max_velocity_ms=float(v_max),
            cp_min=float(cp_final.min()), cp_max=float(cp_final.max()), reynolds=float(re),
            mach=float(cond.mach), ref_area_m2=float(s_ref), ref_length_m=float(l_ref),
            aspect_ratio=float(info.aspect_ratio), stalled=bool(stalled), stall_alpha_deg=float(a_stall_deg))
        if re < 5.0e4:
            warnings.append(f"Reynolds number {re:.1e} is very low; viscous estimates are rough.")
        h_b = float(getattr(cond, "ground_h_over_b", 0.0))
        if 0.0 < h_b < 0.08:
            warnings.append(f"Very close to the ground (h/b {h_b:.2f}): the lifting-line ground-effect model "
                            "overestimates the effect below about h/b = 0.08.")
        cp_star = critical_cp(cond.mach)
        if cond.mach > 0.3 and float(cp_final.min()) < cp_star:
            warnings.append(f"Local flow is supersonic (Cp min {float(cp_final.min()):.2f} < critical Cp "
                            f"{cp_star:.2f} at M {cond.mach:.2f}): shocks form that this engine does not "
                            "model - use the Supersonic Lab or a compressible solver.")
        return _Aero(kpis, dhat, lhat, sigma, cp_final, cp_pot, vortex, float(cd_f + cd_sep + cd_stall), warnings)

    @staticmethod
    def _lifting_increment(prep: PreparedGeometry, vortex: _VortexSystem, lhat: FloatArray,
                           sec: _Sections) -> FloatArray:
        """Chordwise lifting pressure difference mapped onto the surfels (upper suction, lower pressure)."""
        idx = vortex.stations
        y_c = sec.y_c[idx]
        chord = sec.chord[idx]
        cl_sec = 2.0 * vortex.gamma / np.maximum(chord, 1e-9)
        y = prep.pos[:, 1]
        in_span = (y >= sec.y_edges[idx[0]]) & (y <= sec.y_edges[idx[-1] + 1])
        cl_y = np.interp(y, y_c, cl_sec)
        c_y = np.maximum(np.interp(y, y_c, chord), 1e-9)
        x_le = np.interp(y, y_c, sec.x_le[idx])
        z_ref = np.interp(y, y_c, sec.z_ref[idx])
        raw = (prep.pos[:, 0] - x_le) / c_y
        # Only surfels that actually lie on the local section get lifting pressure. Previously points
        # ahead of the leading edge (e.g. a fuselage nose) were clipped to xi = 0 and received the
        # maximum leading-edge suction.
        on_section = (raw >= -0.02) & (raw <= 1.02) & (np.abs(prep.pos[:, 2] - z_ref) <= 0.3 * c_y)
        xi = np.clip(raw, 0.0, 1.0)
        shape = np.sqrt((1.0 - xi) / (xi + CHORD_SHAPE_EPS)) / _CHORD_SHAPE_NORM
        side = np.tanh((prep.nrm @ lhat) / 0.35)
        return 0.5 * side * cl_y * shape * in_span * on_section

    # -- public API ------------------------------------------------------
    def coefficients(self, prepared: PreparedGeometry, conditions: FlowConditions) -> AeroKPIs:
        """Fast scalar results for one condition (no volume field). Used by polar sweeps."""
        return self._aero(prepared, conditions).kpis

    def cached_preparation(self, model: GeometryModel, quality: GridQuality) -> Optional[PreparedGeometry]:
        """The preparation for ``model`` at ``quality`` if it is already cached (never computes one)."""
        return self._cached(model, quality)

    def surface_preview(self, prepared: PreparedGeometry, conditions: FlowConditions
                        ) -> tuple[AeroKPIs, FloatArray]:
        """KPIs plus per-vertex surface Cp in ~5-20 ms (no volume field) — for live slider previews."""
        aero = self._aero(prepared, conditions)
        cp_vertex = np.clip((aero.cp_surfel[prepared.v_idx] * prepared.v_w).sum(axis=1), -8.0, 1.0)
        return aero.kpis, cp_vertex

    def solve(self, model: GeometryModel, conditions: FlowConditions,
              quality: GridQuality = GridQuality.MEDIUM, progress: Optional[ProgressFn] = None,
              cancel: Optional[CancelFn] = None, prepared: Optional[PreparedGeometry] = None,
              compute_field: bool = True) -> SolverResult:
        """Run the full solve: coefficients, surface pressure, chordwise Cp and the volume field.

        Args:
            model: Geometry to analyse.
            conditions: Freestream definition (validated and clamped internally).
            quality: Coarse / Medium / Fine resolution.
            progress: ``(fraction 0-1, message)`` callback.
            cancel: Callback returning True to abort (raises :class:`SolverCancelled`).
            prepared: Reuse a prior :meth:`prepare` result for this model and quality.
            compute_field: When False the volume grid is skipped (coefficients and surface only).
        """
        t0 = time.perf_counter()

        def tick(frac: float, msg: str) -> None:
            if cancel and cancel():
                raise SolverCancelled()
            if progress:
                progress(frac, msg)

        # A supplied preparation is only reused when it matches both the model and the quality.
        prep = (prepared if prepared is not None and prepared.model is model and prepared.quality == quality
                else None)
        if prep is None:
            prep = self.prepare(model, quality, lambda f, m: tick(0.6 * f, m), cancel)
        cond, _ = validate_flow_conditions(conditions)
        tick(0.62, "Solving surface flow")
        aero = self._aero(prep, cond)
        tick(0.68, "Mapping pressure to the mesh")
        cp_vertex = np.clip((aero.cp_surfel[prep.v_idx] * prep.v_w).sum(axis=1), -8.0, 1.0)
        surface = SurfaceField(model.vertices.astype(np.float32), model.faces.astype(np.int32),
                               cp_vertex.astype(np.float32))
        chord_cp = self._chord_cp(prep, cp_vertex, aero.lhat)
        warnings = list(aero.warnings)
        field_out = FlowField(freestream_ms=cond.velocity_ms)
        kpis = aero.kpis
        if compute_field:
            field_out, vmax_grid = self._volume_field(prep, aero, cond, quality, tick)
            if vmax_grid > kpis.max_velocity_ms:
                kpis = _replace_kpi(kpis, max_velocity_ms=float(vmax_grid))
        tick(1.0, "Done")
        elapsed = time.perf_counter() - t0
        log.info("Instant solve finished in %.2fs: CL=%.3f CD=%.4f", elapsed, kpis.cl, kpis.cd)
        extra = {"n_surfels": prep.n_surfels, "prep_seconds": prep.prep_seconds,
                 "cp_potential_min": float(aero.cp_potential.min()),
                 "form_factor": prep.form_factor, "mean_tc": prep.tc_mean,
                 "gamma": None if aero.vortex is None else aero.vortex.gamma.copy(),
                 "stall_blend": 0.0 if aero.vortex is None else aero.vortex.s_stall,
                 "cl_pressure_integral": self._cl_pressure_integral(prep, aero, cond)}
        extra.update(self._span_load(prep, aero))
        return SolverResult(engine="Instant Aero Engine", conditions=cond, kpis=kpis, field=field_out,
                            surface=surface, chord_cp=chord_cp, geometry_name=model.name,
                            elapsed_s=elapsed, warnings=warnings, extra=extra)

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _span_load(prep: PreparedGeometry, aero: _Aero) -> dict[str, object]:
        """Spanwise loading for the analysis chart: station ``y``, local section ``cl``, chord and the
        normalised load ``cl·c / (CL·c_mean)`` (1 everywhere for an ideal elliptic-equivalent mean)."""
        vs, sec = aero.vortex, prep.sections
        if vs is None or sec is None or len(vs.stations) < 3:
            return {}
        idx = vs.stations
        y = np.asarray(sec.y_c[idx], dtype=float)
        chord = np.asarray(sec.chord[idx], dtype=float)
        gamma = np.asarray(vs.gamma, dtype=float)              # Gamma / V (m)
        cl_local = 2.0 * gamma / np.maximum(chord, 1e-9)
        span = float(sec.y_edges[idx[-1] + 1] - sec.y_edges[idx[0]])
        y_mid = 0.5 * float(sec.y_edges[idx[0]] + sec.y_edges[idx[-1] + 1])
        return {"span_y": y, "span_eta": 2.0 * (y - y_mid) / max(span, 1e-9), "span_cl": cl_local,
                "span_chord": chord, "span_gamma": gamma, "span_b": span}

    @staticmethod
    def _cl_pressure_integral(prep: PreparedGeometry, aero: _Aero, cond: FlowConditions) -> float:
        """Lift coefficient from integrating surfel pressures (diagnostic cross-check)."""
        s_ref = prep.info.reference_area_m2
        return float(np.sum(-aero.cp_surfel * (prep.nrm @ aero.lhat) * prep.area) / max(s_ref, 1e-12))

    @staticmethod
    def _chord_cp(prep: PreparedGeometry, cp_vertex: FloatArray, lhat: FloatArray) -> ChordCp:
        """Cp versus x/c on the slice nearest the symmetry plane."""
        v = prep.model.vertices
        info = prep.info
        y_mid = 0.5 * (info.bounds_min[1] + info.bounds_max[1])
        span = max(info.span_m, 1e-9)
        chosen = np.zeros(len(v), dtype=bool)
        for frac in (0.004, 0.01, 0.02, 0.04, 0.08, 0.16):
            chosen = np.abs(v[:, 1] - y_mid) <= frac * span
            if chosen.sum() >= 40:
                break
        if chosen.sum() < 4:
            return ChordCp(y_mid, np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
        x = (v[chosen, 0] - info.bounds_min[0]) / max(info.chord_m, 1e-9)
        cp = cp_vertex[chosen]
        upper = (prep.vertex_normals[chosen] @ lhat) >= 0.0
        ou, ol = np.argsort(x[upper]), np.argsort(x[~upper])
        return ChordCp(float(y_mid), x[upper][ou], cp[upper][ou], x[~upper][ol], cp[~upper][ol])

    def _volume_field(self, prep: PreparedGeometry, aero: _Aero, cond: FlowConditions,
                      quality: GridQuality, tick: Callable[[float, str], None]
                      ) -> tuple[FlowField, float]:
        """Evaluate the nondimensional velocity on the Cartesian grid and assemble the field."""
        spec = grid_spec(prep.info, quality)
        nx, ny, nz = spec.dims
        xs = spec.origin[0] + spec.spacing[0] * np.arange(nx)
        ys = spec.origin[1] + spec.spacing[1] * np.arange(ny)
        zs = spec.origin[2] + spec.spacing[2] * np.arange(nz)
        zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")       # x fastest after ravel
        nodes = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        n_nodes = len(nodes)
        h_cell = float(np.mean(spec.spacing))
        dhat = aero.dhat

        tick(0.72, "Computing body distance field")
        tree = prep.tree
        k = 4
        dist, idx = tree.query(nodes, k=k, workers=-1)
        nn = prep.sample_nrm[idx]                                  # (N, k, 3)
        wgt = 1.0 / (dist + 0.25 * h_cell)
        signed = np.einsum("nkc,nkc->nk", nodes[:, None, :] - prep.sample_pts[idx], nn)
        inside = ((signed * wgt).sum(axis=1) < 0.0) & (dist[:, 0] < 2.0 * h_cell + 0.5 * prep.info.height_m)
        n_hat = (nn * wgt[..., None]).sum(axis=1)
        n_hat /= np.maximum(np.linalg.norm(n_hat, axis=1, keepdims=True), 1e-12)
        d0 = dist[:, 0]

        # Source-surfel and vortex-lattice contributions (compiled kernels in core.accel). The work
        # is split into a few blocks only so progress and cancellation stay responsive.
        wsrc = aero.sigma * prep.area / FOUR_PI
        vel = np.empty((n_nodes, 3))
        vel[:] = dhat
        filaments = self._filament_set(prep, aero)
        c_mean = prep.info.planform_area_m2 / max(prep.info.span_m, 1e-9)
        eps2 = max(0.06 * c_mean, 0.4 * h_cell) ** 2
        n_blocks = 8 if accel.HAVE_NUMBA else 24
        block = max(1, -(-n_nodes // n_blocks))
        for start in range(0, n_nodes, block):
            stop = min(start + block, n_nodes)
            out = vel[start:stop]
            accel.source_velocity(nodes[start:stop], prep.pos, prep.a2, wsrc, out=out)
            if filaments is not None:
                fa, fb, fg = filaments
                accel.horseshoe_velocity(nodes[start:stop], fa, fb, fg, dhat, eps2, out=out)
            vel[start:stop] = out
            tick(0.74 + 0.16 * stop / n_nodes, "Evaluating flow field")

        tick(0.91, "Applying boundary conditions and wake")
        vel[inside] = 0.0
        delta = max(1.5 * h_cell, 0.02 * prep.info.chord_m)
        w_proj = np.exp(-((d0 / delta) ** 2)) * (~inside)
        vel -= (w_proj * (vel * n_hat).sum(1))[:, None] * n_hat
        sp = np.linalg.norm(vel, axis=1)
        over = sp > 2.5
        if over.any():
            vel[over] *= (2.5 / sp[over])[:, None]
        cp = 1.0 - (vel * vel).sum(1)                              # before the wake: static pressure ~ inviscid
        cp[inside] = 1.0
        vel -= self._wake_deficit(prep, aero, nodes, dhat, inside)[:, None] * dhat
        speed = np.linalg.norm(vel, axis=1)
        v_inf = cond.velocity_ms
        field_out = FlowField(
            origin=spec.origin, spacing=spec.spacing, dims=spec.dims,
            velocity=(vel * v_inf).astype(np.float32), cp=np.clip(cp, -8.0, 1.0).astype(np.float32),
            solid=inside.astype(np.uint8), freestream_ms=v_inf)
        return field_out, float(speed.max() * v_inf)

    @staticmethod
    def _filament_set(prep: PreparedGeometry, aero: _Aero) -> Optional[tuple[FloatArray, FloatArray, FloatArray]]:
        """Chordwise-distributed horseshoes carrying the stall-scaled circulation."""
        vs, sec = aero.vortex, prep.sections
        if vs is None or sec is None or np.max(np.abs(vs.gamma)) < 1e-12:
            return None
        edges = np.asarray(LIFT_FILAMENT_EDGES)
        phi = np.arcsin(np.sqrt(edges))
        cum = phi + np.sin(phi) * np.cos(phi)                       # integral of the flat-plate loading
        weights = np.diff(cum) / (0.5 * math.pi)
        xi = 0.5 * (edges[:-1] + edges[1:])
        idx = vs.stations
        a_list, b_list, g_list = [], [], []
        for j, st in enumerate(idx):
            for w, x in zip(weights, xi):
                a_list.append([sec.x_le_edge[st] + x * sec.chord_edge[st], sec.y_edges[st], sec.z_edge[st]])
                b_list.append([sec.x_le_edge[st + 1] + x * sec.chord_edge[st + 1], sec.y_edges[st + 1],
                               sec.z_edge[st + 1]])
                g_list.append(vs.gamma[j] * w)
        return np.asarray(a_list), np.asarray(b_list), np.asarray(g_list)

    @staticmethod
    def _wake_deficit(prep: PreparedGeometry, aero: _Aero, nodes: FloatArray, dhat: FloatArray,
                      inside: NDArray[np.bool_]) -> FloatArray:
        """Momentum-conserving Gaussian wake deficit (fraction of V_inf) behind the body."""
        info = prep.info
        ref = prep.model.bbox_center
        e2 = np.array([0.0, 1.0, 0.0]) - dhat[1] * dhat
        e2 /= max(np.linalg.norm(e2), 1e-12)
        e3 = np.cross(dhat, e2)
        sp = prep.sample_pts - ref
        s_b, y_b, z_b = sp @ dhat, sp @ e2, sp @ e3
        s_trail = float(s_b.max())
        a0 = 0.5 * float(y_b.max() - y_b.min())
        b0 = 0.5 * float(z_b.max() - z_b.min())
        yc, zc = 0.5 * float(y_b.max() + y_b.min()), 0.5 * float(z_b.max() + z_b.min())
        rel = nodes - ref
        s, y, z = rel @ dhat - s_trail, rel @ e2 - yc, rel @ e3 - zc
        length = max(info.chord_m, 1e-6)
        s_pos = np.maximum(s, 0.0)
        a = 0.9 * a0 + 0.04 * s_pos + 1e-3
        b = 0.75 * b0 + 0.08 * np.sqrt(s_pos * length) + 1e-3
        start = _smoothstep((s + 0.02 * length) / (0.12 * length))
        amp = aero.cd_profile * info.reference_area_m2 / (2.0 * math.pi * a * b)
        amp = np.minimum(amp, 0.85)
        prof = np.exp(-((y / a) ** 2 + (z / b) ** 2))
        out = amp * prof * start
        out[inside] = 0.0
        return out


def _replace_kpi(k: AeroKPIs, **changes: float) -> AeroKPIs:
    """Return a copy of an :class:`AeroKPIs` with some fields replaced."""
    from dataclasses import replace
    return replace(k, **changes)
