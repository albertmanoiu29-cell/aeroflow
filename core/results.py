"""Shared result containers exchanged between solvers, viewports and charts.

Both CFD engines (Instant Aero Engine and OpenFOAM) produce the same
:class:`SolverResult`, so the 3-D viewport, charts and report exporter never need
to know which engine ran.

Only NumPy is required at import time; PyVista is imported lazily inside the
``to_pyvista`` helpers, which must be called on the GUI thread.

Coordinate convention: +X is the freestream direction (nose at -X), +Y is
lateral, +Z is up. Flat grid arrays use VTK point ordering (x fastest):
``index = i + nx * (j + ny * k)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from config.settings import FlowConditions


def lambda2_criterion(gradient: NDArray[Any], chunk: int = 200_000) -> NDArray[np.float32]:
    """λ2 vortex criterion (Jeong & Hussain 1995) from the velocity gradient (N, 9), row-major du_i/dx_j.

    λ2 is the middle eigenvalue of S² + Ω² (S, Ω: symmetric / antisymmetric parts of the gradient); a
    vortex core is where λ2 < 0. Unlike Q it does not mark regions of pure shear near walls as vortices."""
    g = np.asarray(gradient, dtype=np.float64).reshape(-1, 3, 3)
    out = np.empty(len(g), dtype=np.float32)
    for k in range(0, len(g), chunk):                 # bounded memory on big OpenFOAM meshes
        j = g[k:k + chunk]
        s = 0.5 * (j + j.transpose(0, 2, 1))
        w = 0.5 * (j - j.transpose(0, 2, 1))
        m = s @ s + w @ w
        out[k:k + chunk] = np.linalg.eigvalsh(m)[:, 1]
    out[~np.isfinite(out)] = 0.0
    return out

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Scalar KPIs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AeroKPIs:
    """Scalar aerodynamic results for one flow condition."""

    alpha_deg: float
    beta_deg: float
    velocity_ms: float
    cl: float
    cd: float
    cm: float
    cd_friction: float = 0.0
    cd_induced: float = 0.0
    cd_pressure: float = 0.0
    lift_n: float = 0.0
    drag_n: float = 0.0
    moment_nm: float = 0.0
    dynamic_pressure_pa: float = 0.0
    stagnation_pressure_pa: float = 0.0
    static_pressure_pa: float = 0.0
    max_velocity_ms: float = 0.0
    cp_min: float = 0.0
    cp_max: float = 1.0
    reynolds: float = 0.0
    mach: float = 0.0
    ref_area_m2: float = 1.0
    ref_length_m: float = 1.0
    aspect_ratio: float = 1.0
    stalled: bool = False
    stall_alpha_deg: float = 15.0

    @property
    def l_over_d(self) -> float:
        """Lift-to-drag ratio (0.0 when drag is ~0)."""
        return self.cl / self.cd if abs(self.cd) > 1e-9 else 0.0

    def table_rows(self) -> list[tuple[str, str, str]]:
        """Return ``(label, value, unit)`` rows for tables and the HTML report."""
        return [
            ("Lift coefficient CL", f"{self.cl:+.4f}", ""),
            ("Drag coefficient CD", f"{self.cd:.5f}", ""),
            ("Pitching moment CM", f"{self.cm:+.4f}", ""),
            ("Lift-to-drag L/D", f"{self.l_over_d:.2f}", ""),
            ("Lift force", f"{self.lift_n:,.1f}", "N"),
            ("Drag force", f"{self.drag_n:,.1f}", "N"),
            ("Pitching moment", f"{self.moment_nm:,.1f}", "N·m"),
            ("Friction drag CD,f", f"{self.cd_friction:.5f}", ""),
            ("Induced drag CD,i", f"{self.cd_induced:.5f}", ""),
            ("Pressure / separation drag", f"{self.cd_pressure:.5f}", ""),
            ("Stagnation pressure", f"{self.stagnation_pressure_pa:,.0f}", "Pa"),
            ("Maximum velocity", f"{self.max_velocity_ms:.1f}", "m/s"),
            ("Minimum Cp", f"{self.cp_min:.3f}", ""),
            ("Reynolds number", f"{self.reynolds:.3e}", ""),
            ("Mach number", f"{self.mach:.3f}", ""),
            ("Reference area", f"{self.ref_area_m2:.4g}", "m²"),
            ("Reference length", f"{self.ref_length_m:.4g}", "m"),
            ("Aspect ratio", f"{self.aspect_ratio:.2f}", ""),
        ]


# ---------------------------------------------------------------------------
# Volume field
# ---------------------------------------------------------------------------
@dataclass
class FlowField:
    """A volumetric flow solution.

    Either a structured Cartesian grid (Instant Engine; flat arrays) or an
    arbitrary PyVista dataset (OpenFOAM, stored in :attr:`dataset`).
    """

    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    dims: tuple[int, int, int] = (0, 0, 0)
    velocity: Optional[NDArray[np.float32]] = None   # (N, 3)
    cp: Optional[NDArray[np.float32]] = None         # (N,)
    solid: Optional[NDArray[np.uint8]] = None        # (N,) 1 inside the body
    dataset: Any = None                              # pre-built pv.DataSet (OpenFOAM)
    freestream_ms: float = 1.0
    _cached: Any = field(default=None, init=False, repr=False)

    @property
    def point_count(self) -> int:
        """Number of grid points."""
        nx, ny, nz = self.dims
        return nx * ny * nz

    @property
    def speed(self) -> Optional[NDArray[np.float32]]:
        """Velocity magnitude per point."""
        if self.velocity is None:
            return None
        return np.linalg.norm(self.velocity, axis=1).astype(np.float32)

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """``(xmin, xmax, ymin, ymax, zmin, zmax)`` of the grid."""
        (ox, oy, oz), (dx, dy, dz), (nx, ny, nz) = self.origin, self.spacing, self.dims
        return (ox, ox + dx * (nx - 1), oy, oy + dy * (ny - 1), oz, oz + dz * (nz - 1))

    def to_pyvista(self) -> Any:
        """Return (and cache) a ``pyvista.DataSet`` with ``velocity``, ``speed``, ``Cp`` and ``solid`` arrays.

        Building data objects is thread-safe (no rendering), so workers may call this ahead of time.
        """
        if self._cached is not None:
            return self._cached
        if self.dataset is not None:
            self._cached = self.dataset
            return self._cached
        import pyvista as pv  # lazy: keeps the numerical core importable headlessly

        if self.velocity is None:
            raise ValueError("FlowField has no velocity data.")
        grid = pv.ImageData(dimensions=self.dims, spacing=self.spacing, origin=self.origin)
        grid.point_data["velocity"] = np.ascontiguousarray(self.velocity, dtype=np.float32)
        grid.point_data["speed"] = self.speed
        if self.cp is not None:
            grid.point_data["Cp"] = np.ascontiguousarray(self.cp, dtype=np.float32)
        if self.solid is not None:
            grid.point_data["solid"] = np.ascontiguousarray(self.solid, dtype=np.uint8)
        grid.set_active_vectors("velocity")
        self._cached = grid
        return grid

    def prepare_visual(self) -> Any:
        """Build the dataset plus the derived visualisation fields and cache it.

        Adds ``vorticity`` (vector), ``vorticity_mag`` (1/s), ``qcriterion`` (vortex-core indicator,
        positive where rotation dominates strain) and ``Cp0`` (total-pressure coefficient
        ``Cp + |V|²/V∞²``: 1 in loss-free flow, lower inside the viscous wake).

        Pure data processing (no rendering), so it may run in a worker thread; workers call it so the
        ~0.1 s cost never stalls the GUI. Safe to call repeatedly.
        """
        ds = self.to_pyvista()
        if "qcriterion" in ds.point_data:
            return ds
        try:
            if "velocity" in ds.point_data:
                d = ds.compute_derivative(scalars="velocity", gradient=True, vorticity=True, qcriterion=True)
                for name in ("vorticity", "qcriterion"):
                    if name in d.point_data:
                        ds.point_data[name] = np.asarray(d.point_data[name], dtype=np.float32)
                if "gradient" in d.point_data:
                    ds.point_data["lambda2"] = lambda2_criterion(np.asarray(d.point_data["gradient"]))
                if "vorticity" in ds.point_data:
                    ds.point_data["vorticity_mag"] = np.linalg.norm(
                        np.asarray(ds.point_data["vorticity"]), axis=1).astype(np.float32)
            if "Cp" in ds.point_data and "speed" in ds.point_data:
                v_inf = max(float(self.freestream_ms), 1e-9)
                sp = np.asarray(ds.point_data["speed"], dtype=np.float32)
                ds.point_data["Cp0"] = (np.asarray(ds.point_data["Cp"], dtype=np.float32)
                                        + (sp / v_inf) ** 2).astype(np.float32)
            if "velocity" in ds.point_data:
                ds.set_active_vectors("velocity")
        except Exception:  # noqa: BLE001 - derived fields are optional extras
            import logging
            logging.getLogger("aeroflow.core.results").debug("Derived flow fields unavailable", exc_info=True)
        return ds


# ---------------------------------------------------------------------------
# Surface field
# ---------------------------------------------------------------------------
@dataclass
class SurfaceField:
    """Triangle surface with a per-vertex pressure coefficient."""

    vertices: NDArray[np.float32]
    faces: NDArray[np.int32]
    cp: NDArray[np.float32]
    _cached: Any = field(default=None, init=False, repr=False)

    def to_pyvista(self) -> Any:
        """Return (and cache) a ``pyvista.PolyData`` with a ``Cp`` point array (GUI thread only)."""
        if self._cached is not None:
            return self._cached
        import pyvista as pv

        n = len(self.faces)
        flat = np.column_stack([np.full(n, 3, dtype=np.int64), self.faces.astype(np.int64)]).ravel()
        poly = pv.PolyData(np.asarray(self.vertices, dtype=np.float32), flat)
        poly.point_data["Cp"] = np.asarray(self.cp, dtype=np.float32)
        self._cached = poly
        return poly


@dataclass(frozen=True)
class ChordCp:
    """Chordwise pressure distribution at one spanwise station (for the Cp-vs-x/c chart)."""

    y_position: float
    x_over_c_upper: FloatArray
    cp_upper: FloatArray
    x_over_c_lower: FloatArray
    cp_lower: FloatArray

    @property
    def is_empty(self) -> bool:
        """True when no points were found."""
        return len(self.x_over_c_upper) + len(self.x_over_c_lower) == 0


# ---------------------------------------------------------------------------
# Complete solver result
# ---------------------------------------------------------------------------
@dataclass
class SolverResult:
    """Everything one CFD run produces."""

    engine: str
    conditions: FlowConditions
    kpis: AeroKPIs
    field: FlowField
    surface: Optional[SurfaceField] = None
    chord_cp: Optional[ChordCp] = None
    geometry_name: str = ""
    elapsed_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    residuals: dict[str, list[float]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Polar
# ---------------------------------------------------------------------------
@dataclass
class PolarData:
    """Aerodynamic polar: coefficients versus angle of attack.

    Moments are about :attr:`moment_ref` (25 % of the reference chord by default).
    """

    alpha_deg: FloatArray
    cl: FloatArray
    cd: FloatArray
    cm: FloatArray
    reynolds: float = 0.0
    mach: float = 0.0
    velocity_ms: float = 0.0
    ref_area_m2: float = 1.0
    ref_length_m: float = 1.0
    span_m: float = 1.0
    aspect_ratio: float = 1.0
    moment_ref: tuple[float, float, float] = (0.0, 0.0, 0.0)
    geometry_name: str = ""
    engine: str = "Instant Aero Engine"

    def __post_init__(self) -> None:
        order = np.argsort(self.alpha_deg)
        self.alpha_deg = np.asarray(self.alpha_deg, dtype=np.float64)[order]
        self.cl = np.asarray(self.cl, dtype=np.float64)[order]
        self.cd = np.asarray(self.cd, dtype=np.float64)[order]
        self.cm = np.asarray(self.cm, dtype=np.float64)[order]

    @property
    def l_over_d(self) -> FloatArray:
        """L/D at every sweep point (0 where drag is ~0)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            ld = np.where(np.abs(self.cd) > 1e-9, self.cl / self.cd, 0.0)
        return np.nan_to_num(ld)

    @property
    def cl_max(self) -> float:
        """Maximum lift coefficient in the sweep."""
        return float(self.cl.max()) if len(self.cl) else 0.0

    @property
    def alpha_stall_deg(self) -> float:
        """Angle of attack at CL,max."""
        return float(self.alpha_deg[int(np.argmax(self.cl))]) if len(self.cl) else 0.0

    @property
    def ld_max(self) -> float:
        """Best lift-to-drag ratio."""
        ld = self.l_over_d
        return float(ld.max()) if len(ld) else 0.0

    @property
    def alpha_best_ld_deg(self) -> float:
        """Angle of attack of the best L/D (optimal cruise angle)."""
        ld = self.l_over_d
        return float(self.alpha_deg[int(np.argmax(ld))]) if len(ld) else 0.0

    @property
    def alpha_zero_lift_deg(self) -> float:
        """Zero-lift angle of attack by linear interpolation (NaN if CL never crosses zero)."""
        cl, a = self.cl, self.alpha_deg
        for i in range(len(cl) - 1):
            if cl[i] == 0.0:
                return float(a[i])
            if cl[i] * cl[i + 1] < 0.0:
                t = cl[i] / (cl[i] - cl[i + 1])
                return float(a[i] + t * (a[i + 1] - a[i]))
        return math.nan

    def cl_at(self, alpha_deg: float) -> float:
        """Linearly interpolated CL (clamped at the sweep ends)."""
        return float(np.interp(alpha_deg, self.alpha_deg, self.cl))

    def cd_at(self, alpha_deg: float) -> float:
        """Linearly interpolated CD (clamped at the sweep ends)."""
        return float(np.interp(alpha_deg, self.alpha_deg, self.cd))

    def cm_at(self, alpha_deg: float) -> float:
        """Linearly interpolated CM (clamped at the sweep ends)."""
        return float(np.interp(alpha_deg, self.alpha_deg, self.cm))

    def rows(self) -> list[tuple[float, float, float, float, float]]:
        """``(alpha, CL, CD, CM, L/D)`` rows for CSV export."""
        ld = self.l_over_d
        return [(float(a), float(l), float(d), float(m), float(r))
                for a, l, d, m, r in zip(self.alpha_deg, self.cl, self.cd, self.cm, ld)]
