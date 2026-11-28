"""Comparing two polars: key figures (CLmax, lift slope, zero-lift angle, minimum drag, best L/D, moment) and
the differences B − A on a common angle-of-attack range."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.results import PolarData


def _ld(p: PolarData) -> np.ndarray:
    cd = np.asarray(p.cd, dtype=float)
    return np.where(np.abs(cd) > 1e-12, np.asarray(p.cl, dtype=float) / np.where(np.abs(cd) > 1e-12, cd, 1.0),
                    np.nan)


def metrics(p: PolarData) -> list[tuple[str, float, int]]:
    """(name, value, decimals) of the figures engineers compare."""
    a = np.asarray(p.alpha_deg, dtype=float)
    cl, cd, cm = (np.asarray(x, dtype=float) for x in (p.cl, p.cd, p.cm))
    out: list[tuple[str, float, int]] = []
    i = int(np.nanargmax(cl))
    out += [("CL max", float(cl[i]), 3), ("α at CL max (°)", float(a[i]), 1)]
    lin = (a >= -4.0) & (a <= 6.0)
    if lin.sum() < 3:
        lin = np.argsort(np.abs(a))[: min(5, len(a))]
    k, c0 = np.polyfit(a[lin], cl[lin], 1)
    out += [("Lift slope CLα (/°)", float(k), 4), ("Zero-lift angle α0 (°)", float(-c0 / k) if k else np.nan, 2)]
    j = int(np.nanargmin(cd))
    out += [("CD min", float(cd[j]), 5), ("CL at CD min", float(cl[j]), 3)]
    ld = _ld(p)
    m = int(np.nanargmax(ld))
    out += [("Best L/D", float(ld[m]), 2), ("α at best L/D (°)", float(a[m]), 1), ("CL at best L/D", float(cl[m]), 3)]
    km, m0 = np.polyfit(a[lin], cm[lin], 1)
    out += [("CM at α = 0", float(m0), 4), ("Moment slope CMα (/°)", float(km), 5)]
    return out


@dataclass
class PolarDiff:
    alpha: np.ndarray
    d_cl: np.ndarray
    d_cd: np.ndarray
    d_cm: np.ndarray
    d_ld: np.ndarray


def difference(a: PolarData, b: PolarData, n: int = 121) -> PolarDiff:
    """B − A on the overlap of their angle ranges (linear interpolation)."""
    aa, ab = np.asarray(a.alpha_deg, dtype=float), np.asarray(b.alpha_deg, dtype=float)
    lo, hi = max(aa.min(), ab.min()), min(aa.max(), ab.max())
    if hi <= lo:
        empty = np.array([])
        return PolarDiff(empty, empty, empty, empty, empty)
    x = np.linspace(lo, hi, n)
    oa, ob = np.argsort(aa), np.argsort(ab)

    def f(p, o, al, y):
        return np.interp(x, al[o], np.asarray(y, dtype=float)[o])

    return PolarDiff(x, f(b, ob, ab, b.cl) - f(a, oa, aa, a.cl), f(b, ob, ab, b.cd) - f(a, oa, aa, a.cd),
                     f(b, ob, ab, b.cm) - f(a, oa, aa, a.cm), f(b, ob, ab, _ld(b)) - f(a, oa, aa, _ld(a)))


__all__ = ["metrics", "PolarDiff", "difference"]
