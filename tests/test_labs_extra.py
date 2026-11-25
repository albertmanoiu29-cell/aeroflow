"""Airfoil coordinate files and the Live Flow Lab time averages."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.geometry_manager import naca4_loop  # noqa: E402
from core.lbm2d import FlowLab2D  # noqa: E402
from core.shapes2d import airfoil_file_outline, read_airfoil_dat  # noqa: E402
from utils.validators import parse_naca4  # noqa: E402

LEDNICER = """NACA 4412 test
       5.       5.

 0.0  0.0
 0.25 0.12
 0.5  0.11
 0.75 0.06
 1.0  0.001

 0.0  0.0
 0.25 -0.02
 0.5  -0.015
 0.75 -0.006
 1.0  -0.001
"""


def test_selig_file_is_normalised(tmp_path) -> None:
    loop = naca4_loop(parse_naca4("2412"), 80)
    a = np.radians(5.0)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    f = tmp_path / "foil.dat"
    np.savetxt(f, (loop @ rot.T) * 2.5 + [3.0, 4.0], header="My foil", comments="")
    name, pts = read_airfoil_dat(f)
    assert name == "My foil"
    assert pts[:, 0].min() == pytest.approx(0.0, abs=0.01) and pts[:, 0].max() == pytest.approx(1.0, abs=0.01)
    assert np.abs(pts - loop).max() < 0.01
    assert airfoil_file_outline(f).segments.shape[1:] == (2, 2)


def test_lednicer_file(tmp_path) -> None:
    f = tmp_path / "l.dat"
    f.write_text(LEDNICER)
    _, pts = read_airfoil_dat(f)
    assert len(pts) == 9 and pts[:, 1].max() > 0.1 and pts[:, 1].min() < -0.01


def test_bad_file_raises(tmp_path) -> None:
    f = tmp_path / "bad.dat"
    f.write_text("hello\nworld\n")
    with pytest.raises(ValueError):
        read_airfoil_dat(f)


def test_time_averages_show_the_wake() -> None:
    lab = FlowLab2D(160, 64)
    jj, ii = np.ogrid[0:64, 0:160]
    lab.set_obstacle((ii - 40) ** 2 + (jj - 32.4) ** 2 <= 36, 12)
    lab.set_reynolds(150)
    for _ in range(12):
        lab.step(100)
    assert lab.average_time > 0
    mean = lab.mean_speed()
    assert mean.shape == lab.ux.shape and np.isfinite(mean).all()
    assert mean[32, 60] < mean[5, 60]                       # slower air in the wake than beside it
    rms = lab.rms_fluctuation()
    assert rms.min() >= 0.0 and rms[lab.solid].max() == 0.0
    lab.reset()
    assert lab.average_time == 0.0
