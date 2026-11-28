"""System report for the About dialog and bug reports.

Everything is gathered locally; nothing is sent anywhere. The user's home folder is replaced by ``~`` so a
pasted report does not reveal the account name.
"""

from __future__ import annotations

import os
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Optional


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "not installed"
    except Exception:  # noqa: BLE001 - a broken install must not break the report
        return "unknown"


def redact(text: str) -> str:
    """Replace the home folder (in both slash styles) with ``~``."""
    home = str(Path.home())
    for h in {home, home.replace("\\", "/")}:
        if h and len(h) > 3:
            text = text.replace(h, "~")
    return text


def os_name() -> str:
    """Readable OS name; Windows 11 reports itself as "10" to older Pythons (build 22000+ is 11)."""
    if sys.platform == "win32":
        build = sys.getwindowsversion().build
        return f"Windows {'11' if build >= 22000 else platform.release()} (build {build})"
    return platform.platform()


def system_report(openfoam: Optional[str] = None) -> str:
    """Plain-text report: app, OS, Python, key libraries, CPU/GPU compute back-ends."""
    from config.settings import APP_NAME, APP_VERSION, get_paths
    from core import accel, compute

    lines = [
        f"{APP_NAME} {APP_VERSION}",
        f"OS: {os_name()} ({platform.machine()})",
        f"Python: {sys.version.split()[0]} ({sys.executable})",
        f"CPU threads: {os.cpu_count()}",
    ]
    try:
        from PySide6 import QtCore

        lines.append(f"Qt: {QtCore.qVersion()} (PySide6 {_version('PySide6')})")
    except Exception:  # noqa: BLE001
        lines.append("Qt: unavailable")
    for label, dist in (("VTK", "vtk"), ("PyVista", "pyvista"), ("NumPy", "numpy"), ("SciPy", "scipy"),
                        ("pyqtgraph", "pyqtgraph"), ("CuPy", "cupy-cuda12x"), ("pyopencl", "pyopencl")):
        lines.append(f"{label}: {_version(dist)}")
    lines.append(f"CPU kernels: {accel.backend()}")
    if compute.probed():
        lines.append(f"GPU: {compute.gpu_info().summary()}")
        lines.append(f"Live / Supersonic Labs run on: {compute.describe()}")
    else:
        lines.append("GPU: still being detected")
    lines.append(f"Compute preference: {compute.preference().value}")
    if openfoam:
        lines.append(f"OpenFOAM: {openfoam}")
    lines.append(f"Install folder: {Path(__file__).resolve().parents[1]}")
    lines.append(f"Logs: {get_paths().logs}")
    return redact("\n".join(lines))
