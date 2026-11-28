"""Design of experiments: several flight conditions varied together, run as one batch.

* :func:`design` builds the run matrix: a full-factorial grid, a Latin hypercube (space-filling, any number
  of runs) or a face-centred central composite design (the classic response-surface design).
* :func:`run` evaluates every run with the Instant engine (the geometry preparation is shared, so a few
  hundred runs take seconds) and returns the responses.
* :func:`main_effects` averages each response over the runs at each level of a factor.
* :func:`fit_quadratic` fits a full quadratic response surface (in coded -1..1 factors) by least squares,
  with R²; :func:`optimum` searches the fitted surface for the best point inside the design space.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Sequence

import numpy as np

#: Responses computed for every run: name -> function of AeroKPIs.
RESPONSES: dict[str, Callable] = {
    "CL": lambda a: a.cl,
    "CD": lambda a: a.cd,
    "CM": lambda a: a.cm,
    "L/D": lambda a: a.cl / a.cd if abs(a.cd) > 1e-12 else float("nan"),
    "Lift (N)": lambda a: a.lift_n,
    "Drag (N)": lambda a: a.drag_n,
    "Reynolds": lambda a: a.reynolds,
    "Mach": lambda a: a.mach,
}


@dataclass(frozen=True)
class Factor:
    key: str            # FlowConditions field
    label: str
    unit: str
    lo: float
    hi: float
    levels: int = 5     # for the full factorial


DESIGNS = ("Full factorial", "Latin hypercube", "Central composite")


@dataclass
class DOEResult:
    factors: list[Factor]
    X: np.ndarray                                    # (runs, factors) in physical units
    Y: dict[str, np.ndarray] = field(default_factory=dict)
    design: str = DESIGNS[0]

    @property
    def runs(self) -> int:
        return int(self.X.shape[0])


def design(factors: Sequence[Factor], kind: str = DESIGNS[0], samples: int = 40, seed: int = 1) -> np.ndarray:
    """The run matrix (runs, factors) in physical units."""
    k = len(factors)
    if k == 0:
        return np.zeros((0, 0))
    lo = np.array([f.lo for f in factors])
    hi = np.array([f.hi for f in factors])
    if kind == "Full factorial":
        axes = [np.linspace(f.lo, f.hi, max(int(f.levels), 2)) for f in factors]
        return np.array(list(itertools.product(*axes)), dtype=float)
    if kind == "Latin hypercube":
        n = max(int(samples), 2)
        rng = np.random.default_rng(seed)
        u = np.empty((n, k))
        for j in range(k):                           # one sample in each of n strata per factor, shuffled
            u[:, j] = (rng.permutation(n) + rng.random(n)) / n
        return lo + u * (hi - lo)
    if kind == "Central composite":                  # face-centred: corners, axial points on the faces, centre
        pts = [list(c) for c in itertools.product((-1.0, 1.0), repeat=k)]
        for j in range(k):
            for s in (-1.0, 1.0):
                p = [0.0] * k
                p[j] = s
                pts.append(p)
        pts.append([0.0] * k)
        c = np.array(pts)
        return lo + (c + 1.0) * 0.5 * (hi - lo)
    raise ValueError(f"unknown design {kind!r}")


def run(solver, model, base, quality, factors: Sequence[Factor], X: np.ndarray,
        progress: Optional[Callable[[int, int], None]] = None,
        cancelled: Optional[Callable[[], bool]] = None, kind: str = DESIGNS[0]) -> DOEResult:
    """Evaluate every run (rows of ``X``) with the Instant engine."""
    prep = solver.cached_preparation(model, quality) or solver.prepare(model, quality)
    res = DOEResult(list(factors), np.zeros((0, len(factors))), design=kind)
    rows, ys = [], {k: [] for k in RESPONSES}
    for i, x in enumerate(X):
        if cancelled is not None and cancelled():
            break
        cond = replace(base, **{f.key: float(v) for f, v in zip(factors, x)})
        a = solver.coefficients(prep, cond)
        rows.append(x)
        for k, fn in RESPONSES.items():
            ys[k].append(float(fn(a)))
        if progress is not None:
            progress(i + 1, len(X))
    res.X = np.array(rows, dtype=float).reshape(-1, len(factors))
    res.Y = {k: np.array(v) for k, v in ys.items()}
    return res


def main_effects(res: DOEResult, response: str, bins: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """For each factor: (level centres, mean response). Continuous designs are binned into ``bins``."""
    y = res.Y[response]
    out = []
    for j, f in enumerate(res.factors):
        x = res.X[:, j]
        levels = np.unique(np.round(x, 9))
        if len(levels) > bins:
            edges = np.linspace(f.lo, f.hi, bins + 1)
            idx = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
            centres = 0.5 * (edges[:-1] + edges[1:])
            means = np.array([np.nanmean(y[idx == b]) if np.any(idx == b) else np.nan for b in range(bins)])
        else:
            centres = levels
            means = np.array([np.nanmean(y[np.isclose(x, v)]) for v in levels])
        out.append((centres, means))
    return out


def _coded(res_factors: Sequence[Factor], X: np.ndarray) -> np.ndarray:
    lo = np.array([f.lo for f in res_factors])
    hi = np.array([f.hi for f in res_factors])
    return 2.0 * (X - lo) / np.where(hi > lo, hi - lo, 1.0) - 1.0


def _terms(Z: np.ndarray) -> np.ndarray:
    """Columns 1, z_i, z_i z_j (i<j), z_i² of a full quadratic model."""
    n, k = Z.shape
    cols = [np.ones(n)]
    cols += [Z[:, i] for i in range(k)]
    cols += [Z[:, i] * Z[:, j] for i in range(k) for j in range(i + 1, k)]
    cols += [Z[:, i] ** 2 for i in range(k)]
    return np.column_stack(cols)


@dataclass
class Surface:
    factors: list[Factor]
    coef: np.ndarray
    r2: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return _terms(_coded(self.factors, X)) @ self.coef


def fit_quadratic(res: DOEResult, response: str) -> Optional[Surface]:
    """Least-squares quadratic response surface; None when there are fewer runs than terms."""
    y = res.Y[response]
    ok = np.isfinite(y)
    Z = _coded(res.factors, res.X[ok])
    A = _terms(Z)
    if A.shape[0] < A.shape[1]:
        return None
    coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y[ok] - pred) ** 2))
    ss_tot = float(np.sum((y[ok] - y[ok].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return Surface(list(res.factors), coef, r2)


def optimum(surf: Surface, maximise: bool = True, n: int = 21) -> tuple[np.ndarray, float]:
    """The best point of the fitted surface on a grid over the design space (then refined locally)."""
    k = len(surf.factors)
    per = max(int(round(20000 ** (1.0 / k))), 3) if k else 1
    per = min(per, n * 3)
    axes = [np.linspace(f.lo, f.hi, per) for f in surf.factors]
    G = np.array(list(itertools.product(*axes)), dtype=float)
    v = surf.predict(G)
    j = int(np.nanargmax(v) if maximise else np.nanargmin(v))
    best, val = G[j].copy(), float(v[j])
    lo = np.array([f.lo for f in surf.factors])
    hi = np.array([f.hi for f in surf.factors])
    step = (hi - lo) / per
    for _ in range(40):                              # pattern search inside the bounds
        improved = False
        for i in range(k):
            for s in (-1.0, 1.0):
                c = best.copy()
                c[i] = np.clip(c[i] + s * step[i], lo[i], hi[i])
                cv = float(surf.predict(c)[0])
                if (cv > val) if maximise else (cv < val):
                    best, val, improved = c, cv, True
        if not improved:
            step *= 0.5
            if np.all(step < 1e-6 * np.maximum(hi - lo, 1e-9)):
                break
    return best, val


__all__ = ["RESPONSES", "Factor", "DESIGNS", "DOEResult", "design", "run", "main_effects", "Surface",
           "fit_quadratic", "optimum"]
