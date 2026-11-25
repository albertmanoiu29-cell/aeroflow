"""Central application constants for AeroFlow Studio.

Holds everything that more than one module needs: application metadata,
writable paths, enumerations, UI geometry constants, the International Standard
Atmosphere (ISA) model, the validated parameter ranges that drive every slider
and spin box, and the default flow conditions.

This module deliberately has **no Qt dependency** so that the numerical core
(``core/``) can be imported and unit-tested headlessly.

Coordinate convention used throughout the application
-----------------------------------------------------
* +X : freestream direction (the wind blows toward +X, the nose points to -X)
* +Y : lateral / spanwise
* +Z : up
* Freestream vector: ``V * (cos a cos b, sin b, sin a cos b)`` where ``a`` is
  the angle of attack and ``b`` is the sideslip angle.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------
APP_NAME: Final[str] = "AeroFlow Studio"
APP_ID: Final[str] = "aeroflow_studio"
APP_VERSION: Final[str] = "1.4.1"
ORG_NAME: Final[str] = "AeroFlow"
#: Public project page (releases, issue tracker) and the GitHub "owner/repo" used by the update check.
PROJECT_URL: Final[str] = "https://github.com/albertmanoiu29-cell/aeroflow"
GITHUB_REPO: Final[str] = "albertmanoiu29-cell/aeroflow"
ORG_DOMAIN: Final[str] = "aeroflow.studio"

# ---------------------------------------------------------------------------
# UI geometry constants (pixels)
# ---------------------------------------------------------------------------
TOP_BAR_HEIGHT: Final[int] = 56
LEFT_DOCK_WIDTH: Final[int] = 390      # the control panel needs ~360 px plus the scroll bar
RIGHT_DOCK_WIDTH: Final[int] = 370
STATUS_BAR_HEIGHT: Final[int] = 28
MIN_WINDOW_SIZE: Final[tuple[int, int]] = (1200, 760)
DEFAULT_WINDOW_SIZE: Final[tuple[int, int]] = (1680, 980)

# ---------------------------------------------------------------------------
# Geometry import limits
# ---------------------------------------------------------------------------
SUPPORTED_MESH_EXTENSIONS: Final[tuple[str, ...]] = (".stl", ".obj", ".ply", ".off", ".vtk")
MESH_FILE_FILTER: Final[str] = (
    "3D Geometry (*.stl *.obj *.ply *.off *.vtk);;"
    "STL (*.stl);;OBJ (*.obj);;PLY (*.ply);;OFF (*.off);;VTK (*.vtk);;All files (*)"
)
MAX_MESH_FILE_MB: Final[float] = 500.0
MAX_TRIANGLES_WARN: Final[int] = 500_000
MAX_TRIANGLES_HARD: Final[int] = 5_000_000
MM_DETECTION_LENGTH: Final[float] = 100.0  # longest extent above this => probably millimetres
MAX_POLAR_POINTS: Final[int] = 61

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
G0: Final[float] = 9.80665                 # m/s^2, standard gravity
R_AIR: Final[float] = 287.05287            # J/(kg K), specific gas constant of dry air
GAMMA_AIR: Final[float] = 1.4
SEA_LEVEL_TEMPERATURE: Final[float] = 288.15   # K
SEA_LEVEL_PRESSURE: Final[float] = 101_325.0   # Pa
SUTHERLAND_MU_REF: Final[float] = 1.458e-6     # kg/(m s K^0.5)
SUTHERLAND_S: Final[float] = 110.4             # K

# ---------------------------------------------------------------------------
# Instant-engine tuning constants
# ---------------------------------------------------------------------------
STALL_ALPHA_DEG: Final[float] = 15.0
TRANSITION_REYNOLDS: Final[float] = 5.0e5
DEFAULT_COLORMAPS: Final[tuple[str, ...]] = ("turbo", "plasma", "viridis", "coolwarm", "inferno", "magma")


# ---------------------------------------------------------------------------
# Enumerations (str values double as human-readable UI labels)
# ---------------------------------------------------------------------------
class AppMode(str, Enum):
    """Top-level application modes shown in the segmented switcher."""

    CFD = "Wind Tunnel CFD"
    POLAR = "Polar Analysis"
    LIVE = "Live Flow Lab"


class SolverEngine(str, Enum):
    """Available CFD back-ends."""

    INSTANT = "Instant Aero Engine"
    OPENFOAM = "OpenFOAM"


class ExecutionMode(str, Enum):
    """How OpenFOAM can be executed on this machine."""

    NATIVE = "native"
    WSL = "wsl"
    DOCKER = "docker"
    UNAVAILABLE = "unavailable"


class GridQuality(str, Enum):
    """Resolution presets shared by both engines."""

    COARSE = "Coarse (fast)"
    MEDIUM = "Medium"
    FINE = "Fine (detailed)"


class ReferenceAreaMode(str, Enum):
    """Which projected area is used as the aerodynamic reference area."""

    PLANFORM = "Planform (wings / aircraft)"
    FRONTAL = "Frontal (cars / bluff bodies)"


class SeedShape(str, Enum):
    """Streamline seed-rake shapes."""

    LINE = "Line"
    PLANE = "Plane"
    SPHERE = "Sphere"


class SliceAxis(str, Enum):
    """Normal axis of the interactive slice plane."""

    X = "X"
    Y = "Y"
    Z = "Z"


class ScalarField(str, Enum):
    """Scalar field shown on slices / streamlines."""

    VELOCITY = "Velocity magnitude"
    PRESSURE = "Pressure coefficient"
    VORTICITY = "Vorticity (swirl)"
    TOTAL_PRESSURE = "Total pressure (wake)"


class IsoField(str, Enum):
    """Quantity drawn as a 3-D isosurface in the wind-tunnel view."""

    QCRITERION = "Vortex cores (Q-criterion)"
    LAMBDA2 = "Vortex cores (λ2 criterion)"
    SPEED = "Speed shell"
    WAKE = "Wake (total-pressure loss)"


class GeometryPreset(str, Enum):
    """Procedural built-in geometries."""

    NACA_0012 = "NACA 0012 Symmetric Wing"
    NACA_2412 = "NACA 2412 Cambered Wing"
    DELTA = "Swept Delta / Fighter Concept"
    GLIDER = "Glider / Sailplane"
    SPHERE = "Aerodynamic Sphere / Teardrop"
    CUSTOM_NACA = "Custom NACA 4-Digit Wing"


# ---------------------------------------------------------------------------
# Grid / mesh resolution presets
# ---------------------------------------------------------------------------
# Point budgets of the Instant engine's volume grid (the product sets the cell count; the domain shape
# sets the actual dims). Raised ~3.4x in v1.1 now that the field kernels are compiled.
GRID_RESOLUTIONS: Final[dict[GridQuality, tuple[int, int, int]]] = {
    GridQuality.COARSE: (54, 36, 36),
    GridQuality.MEDIUM: (72, 48, 48),
    GridQuality.FINE: (108, 72, 72),
}

# Wind-tunnel domain extents in units of the characteristic length L.
DOMAIN_UPSTREAM: Final[float] = 2.5
DOMAIN_DOWNSTREAM: Final[float] = 5.0
DOMAIN_LATERAL: Final[float] = 2.5


@dataclass(frozen=True)
class MeshPreset:
    """snappyHexMesh resolution preset."""

    base_cells_per_length: int
    surface_min_level: int
    surface_max_level: int
    boundary_layers: int


OPENFOAM_MESH_PRESETS: Final[dict[GridQuality, MeshPreset]] = {
    GridQuality.COARSE: MeshPreset(8, 2, 3, 2),
    GridQuality.MEDIUM: MeshPreset(12, 3, 4, 3),
    GridQuality.FINE: MeshPreset(16, 3, 5, 4),
}


@dataclass(frozen=True)
class OpenFOAMConfig:
    """Static configuration for the OpenFOAM bridge."""

    required_executables: tuple[str, ...] = (
        "blockMesh", "surfaceFeatureExtract", "snappyHexMesh", "simpleFoam", "foamToVTK",
    )
    probe_executable: str = "simpleFoam"
    docker_images: tuple[str, ...] = (
        "openfoam/openfoam11-paraview510",
        "openfoam/openfoam10-paraview510",
        "openfoam/openfoam2312-paraview510",
    )
    docker_case_dir: str = "/home/openfoam/case"
    detect_timeout_s: float = 10.0
    write_interval: int = 100
    residual_target: float = 1.0e-4
    turbulence_intensity: float = 0.01
    viscosity_ratio: float = 10.0


OPENFOAM: Final[OpenFOAMConfig] = OpenFOAMConfig()

OPENFOAM_SETUP_URLS: Final[dict[str, str]] = {
    "OpenFOAM Foundation (Linux packages)": "https://openfoam.org/download/",
    "OpenFOAM ESI / openfoam.com": "https://www.openfoam.com/download",
    "Windows WSL2 install guide": "https://learn.microsoft.com/windows/wsl/install",
    "Docker Desktop": "https://www.docker.com/products/docker-desktop/",
}


# ---------------------------------------------------------------------------
# Writable application paths
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AppPaths:
    """Writable directories used by the application."""

    root: Path
    logs: Path
    cases: Path
    exports: Path
    cache: Path

    @classmethod
    def build(cls, root: Path) -> "AppPaths":
        """Create an :class:`AppPaths` rooted at ``root`` (directories not yet created)."""
        return cls(root=root, logs=root / "logs", cases=root / "cases",
                   exports=root / "exports", cache=root / "cache")

    def all_dirs(self) -> tuple[Path, ...]:
        """Return every directory that must exist."""
        return (self.root, self.logs, self.cases, self.exports, self.cache)


def _user_data_root() -> Path:
    """Return the platform-specific per-user data directory."""
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return base / APP_ID


@lru_cache(maxsize=1)
def get_paths() -> AppPaths:
    """Return (and create) the application's writable directories.

    Falls back to the system temp directory if the per-user location is not
    writable, so the application can always start.

    Raises:
        RuntimeError: if no writable location can be found at all.
    """
    for candidate in (_user_data_root(), Path(tempfile.gettempdir()) / APP_ID):
        paths = AppPaths.build(candidate)
        try:
            for directory in paths.all_dirs():
                directory.mkdir(parents=True, exist_ok=True)
            return paths
        except OSError:
            continue
    raise RuntimeError("AeroFlow Studio could not find a writable data directory.")


# ---------------------------------------------------------------------------
# International Standard Atmosphere
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AtmosphereState:
    """Thermodynamic state of air at one altitude."""

    altitude_m: float
    temperature_k: float
    pressure_pa: float
    density: float
    speed_of_sound: float
    dynamic_viscosity: float

    @property
    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity nu = mu / rho in m^2/s."""
        return self.dynamic_viscosity / max(self.density, 1e-12)


# (base altitude m, base temperature K, lapse rate K/m) for each ISA layer.
_ISA_LAYERS: Final[tuple[tuple[float, float, float], ...]] = (
    (0.0, 288.15, -0.0065),
    (11_000.0, 216.65, 0.0),
    (20_000.0, 216.65, 0.0010),
    (32_000.0, 228.65, 0.0028),
    (47_000.0, 270.65, 0.0),
    (51_000.0, 270.65, -0.0028),
    (71_000.0, 214.65, -0.0020),
)
ISA_MIN_ALTITUDE: Final[float] = -500.0
ISA_MAX_ALTITUDE: Final[float] = 80_000.0


def _build_isa_base_pressures() -> tuple[float, ...]:
    """Pre-compute the pressure at the base of every ISA layer."""
    pressures = [SEA_LEVEL_PRESSURE]
    for i in range(len(_ISA_LAYERS) - 1):
        h0, t0, lapse = _ISA_LAYERS[i]
        dh = _ISA_LAYERS[i + 1][0] - h0
        p0 = pressures[-1]
        if abs(lapse) < 1e-12:
            p1 = p0 * math.exp(-G0 * dh / (R_AIR * t0))
        else:
            t1 = t0 + lapse * dh
            p1 = p0 * (t1 / t0) ** (-G0 / (R_AIR * lapse))
        pressures.append(p1)
    return tuple(pressures)


_ISA_BASE_PRESSURE: Final[tuple[float, ...]] = _build_isa_base_pressures()


def isa_atmosphere(altitude_m: float, temperature_offset_k: float = 0.0) -> AtmosphereState:
    """Evaluate the ISA model.

    Args:
        altitude_m: Geometric altitude above sea level in metres. Non-finite
            values fall back to sea level; values are clamped to the model range.
        temperature_offset_k: Deviation from the standard day temperature (ISA+dT).
            Applied to temperature only; pressure follows the standard profile.

    Returns:
        The :class:`AtmosphereState` at the requested altitude.
    """
    h = float(altitude_m) if math.isfinite(altitude_m) else 0.0
    h = min(max(h, ISA_MIN_ALTITUDE), ISA_MAX_ALTITUDE)
    idx = 0
    for i, (base_h, _, _) in enumerate(_ISA_LAYERS):
        if h >= base_h:
            idx = i
    base_h, base_t, lapse = _ISA_LAYERS[idx]
    base_p = _ISA_BASE_PRESSURE[idx]
    dh = h - base_h
    t_std = base_t + lapse * dh
    if abs(lapse) < 1e-12:
        p = base_p * math.exp(-G0 * dh / (R_AIR * base_t))
    else:
        p = base_p * (t_std / base_t) ** (-G0 / (R_AIR * lapse))
    t = max(t_std + (temperature_offset_k if math.isfinite(temperature_offset_k) else 0.0), 150.0)
    rho = p / (R_AIR * t)
    a = math.sqrt(GAMMA_AIR * R_AIR * t)
    mu = SUTHERLAND_MU_REF * t ** 1.5 / (t + SUTHERLAND_S)
    return AtmosphereState(h, t, p, rho, a, mu)


# ---------------------------------------------------------------------------
# Validated parameter ranges (drive every slider + spin box in the UI)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParamLimit:
    """Range, default and UI metadata for one numeric parameter."""

    label: str
    unit: str
    minimum: float
    maximum: float
    default: float
    decimals: int = 2
    step: float = 0.1
    tooltip: str = ""

    def clamp(self, value: float) -> float:
        """Clamp ``value`` into range; non-finite input yields the default."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return self.default
        if not math.isfinite(v):
            return self.default
        return min(max(v, self.minimum), self.maximum)


PARAM_LIMITS: Final[dict[str, ParamLimit]] = {
    # --- Flow conditions ----------------------------------------------------
    "velocity": ParamLimit("Wind speed", "m/s", 1.0, 340.0, 50.0, 1, 1.0,
                           "How fast the air moves past your model (freestream velocity)."),
    "alpha": ParamLimit("Angle of attack", "deg", -20.0, 30.0, 5.0, 1, 0.5,
                        "Tilt of the model relative to the oncoming air. Lift grows with angle until the wing stalls."),
    "beta": ParamLimit("Sideslip", "deg", -20.0, 20.0, 0.0, 1, 0.5,
                       "Sideways angle of the wind. 0 means the wind hits the nose head-on."),
    "altitude": ParamLimit("Altitude", "m", 0.0, 15_000.0, 0.0, 0, 100.0,
                           "Height above sea level. Higher means thinner, less dense air."),
    "temp_offset": ParamLimit("Temperature offset", "K", -40.0, 40.0, 0.0, 1, 1.0,
                              "How much hotter or colder than a standard day. 0 is standard."),
    "ground_h": ParamLimit("Ground height h/b", "", 0.0, 1.0, 0.0, 2, 0.01,
                           "Height of the wing above the ground as a fraction of the span (0 = free flight, no "
                           "ground). Close to the ground (h/b < 0.3) the wing gets more lift and much less induced "
                           "drag - ground effect. Instant engine only."),
    # --- Streamline / visual controls ----------------------------------------
    "streamline_count": ParamLimit("Streamline count", "", 10.0, 300.0, 80.0, 0, 5.0,
                                   "How many flow lines are released upstream of the model."),
    "tube_radius": ParamLimit("Tube thickness", "L", 0.001, 0.030, 0.006, 3, 0.001,
                              "Thickness of the streamline tubes, relative to model length."),
    "seed_width": ParamLimit("Seed width", "L", 0.1, 2.5, 1.0, 2, 0.05,
                             "Side-to-side size of the area where streamlines start."),
    "seed_height": ParamLimit("Seed height", "L", 0.1, 2.5, 0.6, 2, 0.05,
                              "Up-and-down size of the area where streamlines start."),
    "seed_distance": ParamLimit("Seed distance upstream", "L", 0.2, 2.0, 0.8, 2, 0.05,
                                "How far in front of the model the streamlines start."),
    "slice_position": ParamLimit("Slice position", "L", -1.5, 1.5, 0.0, 2, 0.02,
                                 "Where the 2D cross-section plane cuts through the model."),
    "particle_count": ParamLimit("Tracer particles", "", 50.0, 2000.0, 400.0, 0, 50.0,
                                 "Number of glowing smoke particles in the animation."),
    "particle_speed": ParamLimit("Animation speed", "x", 0.1, 4.0, 1.0, 2, 0.1,
                                 "Playback speed of the animated flow tracers."),
    "vector_density": ParamLimit("Arrow density", "", 5.0, 60.0, 20.0, 0, 1.0,
                                 "Number of arrows across the slice plane."),
    "iso_level": ParamLimit("Iso level", "", 0.0, 1.0, 0.5, 2, 0.05,
                            "Where the 3-D surface is drawn. Vortex cores: higher = only the strongest "
                            "vortices. Speed shell: 0 = 0.5×V∞ (slow air) … 1 = 1.5×V∞ (fast air). "
                            "Wake: relative to the strongest loss in the result; higher = only the core."),
    # --- Solver / polar -----------------------------------------------------
    "openfoam_iterations": ParamLimit("OpenFOAM iterations", "", 50.0, 5000.0, 600.0, 0, 50.0,
                                      "Maximum number of solver iterations before stopping."),
    "polar_start": ParamLimit("Sweep start", "deg", -20.0, 30.0, -6.0, 1, 1.0,
                              "First angle of attack in the sweep."),
    "polar_stop": ParamLimit("Sweep end", "deg", -20.0, 30.0, 18.0, 1, 1.0,
                             "Last angle of attack in the sweep."),
    "polar_step": ParamLimit("Sweep step", "deg", 0.25, 5.0, 2.0, 2, 0.25,
                             "Angle increment between sweep points. Smaller is smoother but slower."),
    # --- Live Flow Lab (2-D real-time tunnel) -------------------------------
    "lab_aoa": ParamLimit("Angle of attack", "deg", -30.0, 30.0, 8.0, 1, 0.5,
                          "Tilt of the body. Drag it while the flow runs and watch the wake react."),
    "lab_size": ParamLimit("Body size", "x", 0.4, 1.6, 1.0, 2, 0.05,
                           "Size of the body relative to the tunnel. Bigger bodies show more detail but feel "
                           "the tunnel walls more (blockage)."),
    "lab_speed": ParamLimit("Simulation speed", "x", 0.25, 4.0, 1.0, 2, 0.25,
                            "How fast simulated time runs. Higher speeds use more CPU per frame."),
    "lab_brush": ParamLimit("Brush size", "cells", 1.0, 12.0, 3.0, 0, 1.0,
                            "Radius of the wall brush when drawing obstacles with the mouse."),
}

#: Reynolds-number range of the Live Flow Lab (logarithmic slider).
LAB_RE_RANGE: Final[tuple[float, float]] = (40.0, 100_000.0)
LAB_RE_DEFAULT: Final[float] = 2000.0


# ---------------------------------------------------------------------------
# Flow conditions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FlowConditions:
    """Freestream definition shared by both solver engines."""

    velocity_ms: float = PARAM_LIMITS["velocity"].default
    alpha_deg: float = PARAM_LIMITS["alpha"].default
    beta_deg: float = PARAM_LIMITS["beta"].default
    altitude_m: float = PARAM_LIMITS["altitude"].default
    temperature_offset_k: float = PARAM_LIMITS["temp_offset"].default
    ground_h_over_b: float = 0.0          # 0: free flight; > 0: ground plane at h = this * span below the wing

    def clamped(self) -> "FlowConditions":
        """Return a copy with every field clamped to its physically meaningful range."""
        return replace(
            self,
            velocity_ms=PARAM_LIMITS["velocity"].clamp(self.velocity_ms),
            alpha_deg=PARAM_LIMITS["alpha"].clamp(self.alpha_deg),
            beta_deg=PARAM_LIMITS["beta"].clamp(self.beta_deg),
            altitude_m=PARAM_LIMITS["altitude"].clamp(self.altitude_m),
            temperature_offset_k=PARAM_LIMITS["temp_offset"].clamp(self.temperature_offset_k),
            ground_h_over_b=PARAM_LIMITS["ground_h"].clamp(self.ground_h_over_b),
        )

    @property
    def atmosphere(self) -> AtmosphereState:
        """ISA state at this altitude and temperature offset."""
        return isa_atmosphere(self.altitude_m, self.temperature_offset_k)

    @property
    def mach(self) -> float:
        """Freestream Mach number."""
        return self.velocity_ms / max(self.atmosphere.speed_of_sound, 1e-9)

    def freestream_direction(self) -> tuple[float, float, float]:
        """Unit freestream vector ``(cos a cos b, sin b, sin a cos b)``."""
        a = math.radians(self.alpha_deg)
        b = math.radians(self.beta_deg)
        return (math.cos(a) * math.cos(b), math.sin(b), math.sin(a) * math.cos(b))

    def freestream_vector(self) -> tuple[float, float, float]:
        """Freestream velocity vector in m/s."""
        dx, dy, dz = self.freestream_direction()
        v = self.velocity_ms
        return (v * dx, v * dy, v * dz)
