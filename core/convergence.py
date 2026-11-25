"""Grid convergence check: the same case on three resolutions, Richardson extrapolation and the GCI.

For three grids with element counts N1 > N2 > N3 (fine, medium, coarse) in 3-D, the representative
spacing is h ~ N^(-1/3) for volume meshes; the Instant engine's resolution is set by the surface
elements (surfels), so h ~ N^(-1/2). With refinement ratios r21 = h2/h1, r32 = h3/h2 the observed order
p and the extrapolated value follow Celik et al. (2008, J. Fluids Eng. 130, 078001); the fine-grid
convergence index is GCI = 1.25 |e21| / (r21^p - 1), with e21 the relative fine-medium difference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ConvergenceResult:
    quantity: str
    values: tuple[float, float, float]          # fine, medium, coarse
    observed_order: float                       # NaN when oscillatory / not in the asymptotic range
    extrapolated: float
    gci_fine: float                             # relative uncertainty of the fine-grid value (fraction)
    monotonic: bool

    def summary(self) -> str:
        f, m, c = self.values
        spread = 100 * (max(self.values) - min(self.values)) / max(abs(f), 1e-12)
        if not math.isfinite(self.observed_order):
            if spread < 1.0:
                verdict = "grid-independent (the grids agree within round-off of the method)"
            else:
                verdict = "oscillates between grids" if not self.monotonic else "not yet in the asymptotic range"
            return (f"{self.quantity}: fine {f:.5f} · medium {m:.5f} · coarse {c:.5f} — spread {spread:.2f} %, "
                    f"{verdict}")
        return (f"{self.quantity}: fine {f:.5f} · medium {m:.5f} · coarse {c:.5f} → extrapolated "
                f"{self.extrapolated:.5f} (order {self.observed_order:.2f}, GCI {100 * self.gci_fine:.2f} %)")


def richardson(quantity: str, fine: float, medium: float, coarse: float, n_fine: float, n_medium: float,
               n_coarse: float, dim: int = 2) -> ConvergenceResult:
    """Celik et al. (2008) procedure with fixed-point iteration for unequal refinement ratios."""
    h1, h2, h3 = (n_fine ** (-1.0 / dim), n_medium ** (-1.0 / dim), n_coarse ** (-1.0 / dim))
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = medium - fine, coarse - medium
    monotonic = e21 * e32 > 0
    p: float = float("nan")
    if monotonic and abs(e21) > 1e-14 and r21 > 1.0 + 1e-6 and r32 > 1.0 + 1e-6:
        s = 1.0 if e32 / e21 > 0 else -1.0
        p_est = 1.0
        for _ in range(50):
            q = math.log((r21 ** p_est - s) / (r32 ** p_est - s))
            p_new = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
            if abs(p_new - p_est) < 1e-8:
                break
            p_est = p_new
        p = p_est if 0.2 < p_est < 8.0 else float("nan")
    if math.isfinite(p):
        extrap = (r21 ** p * fine - medium) / (r21 ** p - 1.0)
        ea21 = abs((fine - medium) / fine) if abs(fine) > 1e-12 else float("nan")
        gci = 1.25 * ea21 / (r21 ** p - 1.0)
    else:
        extrap, gci = float("nan"), float("nan")
    return ConvergenceResult(quantity, (fine, medium, coarse), p, extrap, gci, monotonic)


def grid_check(solver, model, conditions, progress=None) -> tuple[list[ConvergenceResult], list[int]]:
    """Run the Instant engine on Fine, Medium and Coarse and analyse CL, CD and CM."""
    from config.settings import GridQuality

    kpis, sizes = [], []
    for k, q in enumerate((GridQuality.FINE, GridQuality.MEDIUM, GridQuality.COARSE)):
        if progress is not None:
            progress(k, q.value)
        prep = solver.prepare(model, q)
        kpis.append(solver.coefficients(prep, conditions))
        sizes.append(int(prep.n_surfels))
    out = []
    for name, attr in (("CL", "cl"), ("CD", "cd"), ("CM", "cm")):
        vals = [float(getattr(k, attr)) for k in kpis]
        out.append(richardson(name, *vals, *sizes))
    return out, sizes


def describe(results: list[ConvergenceResult], sizes: Optional[list[int]] = None) -> str:
    head = f"Surface elements (fine / medium / coarse): {' / '.join(map(str, sizes))}\n" if sizes else ""
    return head + "\n".join(r.summary() for r in results)
