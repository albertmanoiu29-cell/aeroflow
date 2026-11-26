"""GPU (CUDA) backends of the Labs: same results as the CPU kernels, safe device switching.

Skipped automatically on machines without a working CUDA GPU / CuPy.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import accel, compute  # noqa: E402
from core.compute import Device  # noqa: E402
from core.lbm2d import FlowLab2D  # noqa: E402
from core.shapes2d import polygon_segments, rasterize_segments, transform_segments  # noqa: E402
from core.supersonic import Supersonic2D  # noqa: E402

_forced = os.environ.get("AEROFLOW_DEVICE", "").strip().lower()
gpu = pytest.mark.skipif(not compute.gpu_info().available or _forced in ("cpu", "gpu"),
                         reason=compute.gpu_info().reason or "AEROFLOW_DEVICE pins the compute device")
cpu_compiled = pytest.mark.skipif(not accel.HAVE_NUMBA, reason="the reference is the numba CPU kernel")


@pytest.fixture(autouse=True)
def _restore_preference():
    before = compute.preference()
    yield
    compute.set_preference(before)


def _wedge(nx: int = 140, ny: int = 84, mach: float = 2.5) -> Supersonic2D:
    s = Supersonic2D(nx, ny, mach)
    t = np.tan(np.radians(12.0))
    segs = transform_segments(polygon_segments(np.array([[0.0, 0.0], [1.0, t], [1.0, -t]])), (0.0, 0.0),
                              0.35 * nx, 4.0, (0.25 * nx, 0.5 * ny + 0.37))
    s.set_obstacle(rasterize_segments(segs, nx, ny), 0.35 * nx, segs)
    return s


def _cylinder_lab(nx: int = 160, ny: int = 64) -> FlowLab2D:
    lab = FlowLab2D(nx, ny)
    jj, ii = np.ogrid[0:ny, 0:nx]
    lab.set_obstacle((ii - 0.25 * nx) ** 2 + (jj - 0.5 * ny - 0.37) ** 2 <= (0.09 * ny) ** 2, 0.18 * ny)
    lab.set_reynolds(150.0)
    return lab


@gpu
@cpu_compiled
def test_supersonic_gpu_matches_cpu() -> None:
    compute.set_preference(Device.CPU)
    a = _wedge()
    a.step(150)
    compute.set_preference(Device.GPU)
    b = _wedge()
    b.step(150)
    assert (a.device, b.device) == ("CPU", "GPU")
    for x, y in zip(a.primitives(), b.primitives()):
        assert np.max(np.abs(x - y)) < 1e-9          # same algorithm in double precision: round-off only
    assert abs(a.time - b.time) < 1e-9
    assert a.force_coefficients() == pytest.approx(b.force_coefficients(), abs=1e-9)
    # the GPU computes the displayed Mach field itself (float32)
    assert np.max(np.abs(a.mach_field() - b.mach_field())) < 1e-5
    assert np.isfinite(b.U).all() and np.max(np.abs(a.U - b.U)) < 1e-9       # U is fetched on demand


@gpu
def test_supersonic_switch_keeps_state() -> None:
    compute.set_preference(Device.GPU)
    s = _wedge()
    s.step(60)
    t, steps = s.time, s.steps
    p_gpu = s.pressure_ratio().copy()
    compute.set_preference(Device.CPU)
    s.step(1)                                         # continues from the GPU state, on the CPU
    assert s.device == "CPU" and s.steps == steps + 1 and s.time > t
    assert np.nanmax(np.abs(s.pressure_ratio() - p_gpu)) < 0.2 * np.nanmax(p_gpu)
    compute.set_preference(Device.GPU)
    s.step(5)
    assert s.device == "GPU" and np.isfinite(s.U).all()


@gpu
def test_supersonic_async_batch_publishes_on_finish() -> None:
    compute.set_preference(Device.GPU)
    s = _wedge()
    before = s.mach_field().copy()
    s.begin_step(40, background=True)
    s.finish_step()
    after = s.mach_field()
    assert s.steps == 40 and not np.allclose(before, after)
    s.reset()                                         # a reset while idle is clean
    assert s.steps == 0 and np.isfinite(s.mach_field()).all()


@gpu
@cpu_compiled
def test_flowlab_gpu_matches_cpu() -> None:
    compute.set_preference(Device.CPU)
    a = _cylinder_lab()
    for _ in range(6):
        a.step(40)
    compute.set_preference(Device.GPU)
    b = _cylinder_lab()
    for _ in range(6):
        b.step(40)
    assert (a.device, b.device) == ("CPU", "GPU")
    for name in ("ux", "uy", "rho"):
        assert np.max(np.abs(getattr(a, name) - getattr(b, name))) < 1e-4, name     # float32 round-off
    assert np.max(np.abs(a.dye - b.dye)) < 1e-3
    assert a.cd == pytest.approx(b.cd, abs=1e-3)


@gpu
@cpu_compiled
def test_flowlab_paint_and_switch_on_gpu() -> None:
    def run(first: Device) -> FlowLab2D:
        compute.set_preference(first)
        lab = _cylinder_lab()
        lab.step(200)
        lab.paint(100.0, 20.0, 4.0, True)             # edit the obstacle while the state lives on `first`
        lab.step(50)
        compute.set_preference(Device.CPU)
        lab.step(50)
        return lab

    gpu_run, cpu_run = run(Device.GPU), run(Device.CPU)
    assert gpu_run.solid[20, 100] and abs(gpu_run.ux[20, 100]) < 1e-6
    assert gpu_run.device == "CPU" and gpu_run.steps == 300 and np.isfinite(gpu_run.f).all()
    # GPU -> paint -> GPU -> CPU must give the same flow as doing everything on the CPU.
    assert np.max(np.abs(gpu_run.ux - cpu_run.ux)) < 1e-4
    assert np.max(np.abs(gpu_run.f - cpu_run.f)) < 1e-4


def test_cpu_preference_never_uses_gpu() -> None:
    compute.set_preference(Device.CPU)
    assert not compute.use_gpu()
    s = Supersonic2D(40, 24, 2.0)
    s.step(3)
    assert s.device == "CPU"


@gpu
@cpu_compiled
def test_nozzle_gpu_matches_cpu() -> None:
    # cut-cell walls, the axisymmetric update, the characteristic injector, ignition ramp and shut-down
    from core import nozzle as nz

    def run():
        geom = nz.NozzleGeometry(throat_diameter=0.05, exit_diameter=0.10, chamber_diameter=0.10,
                                 chamber_length=0.03, wall_thickness=0.004)
        lay = nz.layout_nozzle(geom, 200, 50, plume_length=5.0, plume_radius=3.0)
        s = nz.NozzleFlow(200, 50, gamma=1.3)
        s.set_geometry(lay)
        s.set_conditions(1 / 1.3, 1.0, 0.0, 20 / 1.3, 3.0)
        s.reset()
        s.ignite(30.0)
        for _ in range(4):
            s.step(40)
        s.shutdown(10.0)
        s.step(60)
        return s

    compute.set_preference(Device.CPU)
    a = run()
    compute.set_preference(Device.GPU)
    b = run()
    assert (a.device, b.device) == ("CPU", "GPU")
    assert abs(a.time - b.time) < 1e-9
    assert np.max(np.abs(a.U - b.U)) / np.max(np.abs(a.U)) < 1e-10
