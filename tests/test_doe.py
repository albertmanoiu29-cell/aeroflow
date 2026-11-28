"""Design of experiments: run matrices, batch evaluation, main effects and the response surface."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import FlowConditions, GeometryPreset, GridQuality  # noqa: E402
from core import doe  # noqa: E402

A = doe.Factor("alpha_deg", "α", "deg", -4.0, 12.0, 5)
V = doe.Factor("velocity_ms", "V", "m/s", 20.0, 80.0, 3)


def test_designs_have_the_right_size_and_bounds() -> None:
    ff = doe.design([A, V], "Full factorial")
    assert ff.shape == (15, 2)
    lhs = doe.design([A, V], "Latin hypercube", samples=30)
    assert lhs.shape == (30, 2)
    for j, f in enumerate((A, V)):              # one sample per stratum in every factor
        strata = np.floor((lhs[:, j] - f.lo) / (f.hi - f.lo) * 30).astype(int)
        assert sorted(strata) == list(range(30))
    cc = doe.design([A, V, doe.Factor("beta_deg", "β", "deg", -5, 5)], "Central composite")
    assert cc.shape == (8 + 6 + 1, 3)
    assert cc.min(axis=0) == pytest.approx([-4.0, 20.0, -5.0]) and cc.max(axis=0) == pytest.approx([12.0, 80.0, 5.0])


def test_quadratic_surface_recovers_a_known_function() -> None:
    X = doe.design([A, V], "Full factorial")
    res = doe.DOEResult([A, V], X)
    y = 3.0 - (X[:, 0] - 6.0) ** 2 / 10.0 + 0.01 * X[:, 1]
    res.Y = {"L/D": y}
    s = doe.fit_quadratic(res, "L/D")
    assert s.r2 == pytest.approx(1.0, abs=1e-9)
    best, val = doe.optimum(s, maximise=True)
    assert best[0] == pytest.approx(6.0, abs=0.05) and best[1] == pytest.approx(80.0)
    assert val == pytest.approx(3.8, abs=1e-3)
    eff = doe.main_effects(res, "L/D")
    assert len(eff[0][0]) == 5 and np.argmax(eff[0][1]) == 2      # α = 4 is the closest level to 6


def test_batch_on_a_wing() -> None:
    from core.geometry_manager import build_preset
    from core.instant_solver import InstantSolver

    s = InstantSolver(warmup=False)
    m = build_preset(GeometryPreset.NACA_2412)
    m = getattr(m, "model", m)
    X = doe.design([A, V], "Central composite")
    res = doe.run(s, m, FlowConditions(), GridQuality.COARSE, [A, V], X)
    assert res.runs == len(X)
    cl = res.Y["CL"]
    assert np.all(np.diff(cl[np.isclose(res.X[:, 1], 50.0)][np.argsort(res.X[np.isclose(res.X[:, 1], 50.0), 0])]) > 0)
    surf = doe.fit_quadratic(res, "CL")
    assert surf.r2 > 0.98                       # CL is nearly linear in α below the stall


def test_combined_report_html(tmp_path) -> None:
    from core import report_book as rb

    png = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                        "1f15c4890000000d49444154789c6360000002000001e221bc330000000049454e44ae426082")
    logo = tmp_path / "logo.png"
    logo.write_bytes(png)
    meta = rb.ReportMeta("Wing <study>", "sub", "Me", "Org", notes="line 1\nline 2", logo=logo)
    secs = [rb.Section("A", rows=[(f"k{i}", str(i)) for i in range(14)], images=[("pic", png)]),
            rb.Section("B", text="hello")]
    out = rb.write(tmp_path / "r.html", meta, secs)
    h = out.read_text(encoding="utf-8")
    assert "Wing &lt;study&gt;" in h and "<h2>Contents</h2>" in h and "1. A" in h and "2. B" in h
    assert h.count("data:image/png;base64,") == 2                 # the logo and the figure, embedded
    assert "kv two" in h and "k13" in h
