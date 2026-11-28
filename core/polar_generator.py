"""Polar sweeps and report export.

A sweep prepares the geometry once (surfel factorisation, section analysis) and then evaluates
the coefficients-only fast path of the Instant Aero Engine for each angle of attack, so a 31-point
sweep typically completes in well under a second after the one-off preparation.

Also provides the CSV and self-contained HTML report writers used by the Polar Analysis mode.
"""

from __future__ import annotations

import base64
import csv
import html
import logging
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from config.settings import APP_NAME, APP_VERSION, FlowConditions, GridQuality
from core.geometry_manager import GeometryModel
from core.instant_solver import InstantSolver, SolverCancelled
from core.results import AeroKPIs, PolarData
from utils.validators import sanitize_filename, validate_sweep

log = logging.getLogger("aeroflow.core.polar")

FloatArray = NDArray[np.float64]
ProgressFn = Callable[[float, str], None]
CancelFn = Callable[[], bool]


def sweep_polar(solver: InstantSolver, model: GeometryModel, conditions: FlowConditions,
                alpha_start: float, alpha_end: float, alpha_step: float,
                quality: GridQuality = GridQuality.MEDIUM, progress: Optional[ProgressFn] = None,
                cancel: Optional[CancelFn] = None) -> PolarData:
    """Run an angle-of-attack sweep with the Instant engine.

    Raises:
        SolverCancelled: if ``cancel()`` returns True.
        ValueError: if the sweep produces no points.
    """
    angles, check = validate_sweep(alpha_start, alpha_end, alpha_step)
    for issue in check.warnings:
        log.warning("Sweep: %s", issue.message)
    if not angles:
        raise ValueError("The polar sweep contains no angles of attack.")
    prep = solver.prepare(model, quality, lambda f, m: progress and progress(0.3 * f, m), cancel)
    cl, cd, cm = [], [], []
    first: Optional[AeroKPIs] = None
    for i, a in enumerate(angles):
        if cancel and cancel():
            raise SolverCancelled()
        k = solver.coefficients(prep, replace(conditions, alpha_deg=a))
        first = first or k
        cl.append(k.cl)
        cd.append(k.cd)
        cm.append(k.cm)
        if progress:
            progress(0.3 + 0.7 * (i + 1) / len(angles), f"α = {a:+.1f}°")
    assert first is not None
    info = prep.info
    return PolarData(
        alpha_deg=np.asarray(angles), cl=np.asarray(cl), cd=np.asarray(cd), cm=np.asarray(cm),
        reynolds=first.reynolds, mach=first.mach, velocity_ms=first.velocity_ms,
        ref_area_m2=info.reference_area_m2, ref_length_m=info.chord_m, span_m=info.span_m,
        aspect_ratio=info.aspect_ratio, moment_ref=info.moment_ref_point, geometry_name=model.name)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_polar_csv(polar: PolarData, path: str | Path) -> Path:
    """Write the polar to CSV and return the final path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["# " + APP_NAME + " polar", polar.geometry_name, f"Re={polar.reynolds:.3e}",
                    f"Mach={polar.mach:.3f}", f"V={polar.velocity_ms:.2f} m/s"])
        w.writerow(["alpha_deg", "CL", "CD", "CM", "L_over_D"])
        for row in polar.rows():
            w.writerow([f"{v:.6f}" for v in row])
    return out


def _svg_chart(x: FloatArray, series: list[tuple[str, FloatArray, str]], xlabel: str, ylabel: str,
               width: int = 460, height: int = 300) -> str:
    """Tiny dependency-free SVG line chart for the HTML report."""
    ml, mr, mt, mb = 56, 14, 14, 42
    x0, x1 = float(np.min(x)), float(np.max(x))
    ys = np.concatenate([s[1] for s in series])
    y0, y1 = float(np.min(ys)), float(np.max(ys))
    if y1 - y0 < 1e-9:
        y0, y1 = y0 - 1.0, y1 + 1.0
    pad = 0.06 * (y1 - y0)
    y0, y1 = y0 - pad, y1 + pad
    if x1 - x0 < 1e-9:
        x1 = x0 + 1.0
    def fmt(v: float, span: float) -> str:
        """Tick label with enough decimals for the axis range (CD axes span ~0.01)."""
        decimals = 0 if span >= 20 else 1 if span >= 2 else 2 if span >= 0.2 else 3 if span >= 0.02 else 4
        return f"{v:.{decimals}f}"

    sx = lambda v: ml + (v - x0) / (x1 - x0) * (width - ml - mr)       # noqa: E731
    sy = lambda v: mt + (1.0 - (v - y0) / (y1 - y0)) * (height - mt - mb)  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
           f'font-family="Segoe UI,Arial" font-size="11">',
           f'<rect width="{width}" height="{height}" fill="#16191E" rx="8"/>']
    for i in range(6):
        gy = y0 + (y1 - y0) * i / 5
        gx = x0 + (x1 - x0) * i / 5
        out.append(f'<line x1="{ml}" x2="{width - mr}" y1="{sy(gy):.1f}" y2="{sy(gy):.1f}" stroke="#252A31"/>')
        out.append(f'<text x="{ml - 6}" y="{sy(gy) + 4:.1f}" fill="#A3AAB5" text-anchor="end">{fmt(gy, y1 - y0)}</text>')
        out.append(f'<text x="{sx(gx):.1f}" y="{height - mb + 16}" fill="#A3AAB5" text-anchor="middle">'
                   f'{fmt(gx, x1 - x0)}</text>')
    if y0 < 0.0 < y1:
        out.append(f'<line x1="{ml}" x2="{width - mr}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" stroke="#475569"/>')
    for _name, y, colour in series:
        pts = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(x, y))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2"/>')
    out.append(f'<text x="{(ml + width - mr) / 2:.0f}" y="{height - 6}" fill="#E8EAED" text-anchor="middle">'
               f'{html.escape(xlabel)}</text>')
    out.append(f'<text x="14" y="{height / 2:.0f}" fill="#E8EAED" text-anchor="middle" '
               f'transform="rotate(-90 14 {height / 2:.0f})">{html.escape(ylabel)}</text>')
    lx = ml + 8
    for name, _, colour in series:
        out.append(f'<rect x="{lx}" y="{mt + 4}" width="10" height="10" fill="{colour}"/>'
                   f'<text x="{lx + 14}" y="{mt + 13}" fill="#E8EAED">{html.escape(name)}</text>')
        lx += 18 + 7 * len(name)
    out.append("</svg>")
    return "".join(out)


def export_html_report(path: str | Path, polar: Optional[PolarData] = None,
                       kpis: Optional[AeroKPIs] = None, geometry_rows: Optional[list[tuple[str, str]]] = None,
                       title: str = "Aerodynamic Report", notes: str = "",
                       images: Optional[list[tuple[str, str | Path]]] = None,
                       condition_rows: Optional[list[tuple[str, str]]] = None,
                       engine: str = "") -> Path:
    """Write a self-contained HTML report.

    Charts are inline SVG; ``images`` (caption, PNG path) such as 3-D streamline snapshots are
    embedded as base64 data URIs, so the single file can be e-mailed as-is.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    esc = html.escape
    parts: list[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{esc(title)}</title><style>"
        "body{background:#101216;color:#E8EAED;font-family:'Segoe UI Variable Text','Segoe UI',Inter,Arial,"
        "sans-serif;margin:0;padding:40px 48px;line-height:1.45}"
        ".wrap{max-width:1180px;margin:0 auto}"
        "header{display:flex;align-items:center;gap:14px;padding-bottom:18px;margin-bottom:8px;"
        "border-bottom:1px solid #252A31}"
        ".mark{width:40px;height:40px;border-radius:11px;background:linear-gradient(135deg,#6FA4FF,#1D3F8F)}"
        "h1{color:#FFFFFF;font-weight:600;font-size:26px;margin:0}"
        "h2{color:#A3AAB5;font-size:12px;letter-spacing:.12em;text-transform:uppercase;font-weight:700;"
        "margin:34px 0 10px}"
        "table{border-collapse:separate;border-spacing:0;min-width:380px;background:#16191E;border:1px solid #252A31;"
        "border-radius:10px;overflow:hidden}td,th{padding:7px 16px;border-bottom:1px solid #252A31;text-align:left}"
        "tr:last-child td{border-bottom:none}th{color:#A3AAB5;font-weight:600}td:nth-child(2){font-variant-numeric:"
        "tabular-nums}.grid{display:flex;flex-wrap:wrap;gap:16px}"
        ".muted{color:#A3AAB5}svg{max-width:100%}"
        "footer{margin-top:40px;padding-top:14px;border-top:1px solid #252A31;color:#6F7682;font-size:12px}"
        # Printing / PDF: dark text on white paper (the SVG presentation colours are overridden by CSS).
        "@media print{body{background:#fff;color:#1B1F24;padding:0}h1{color:#1B1F24}h2{color:#555E6B;"
        "break-after:avoid}table{background:#fff;border-color:#E1E4EA}td,th{border-bottom:1px solid #E1E4EA}"
        "th,.muted{color:#555E6B}header,footer{border-color:#E1E4EA}"
        "svg{break-inside:avoid}svg>rect:first-child{fill:#FFFFFF;stroke:#E1E4EA}svg text{fill:#3A404A}"
        "svg line{stroke:#E1E4EA}img{break-inside:avoid;max-width:100%}table{break-inside:avoid}}"
        "</style></head><body>",
        "<div class='wrap'><header><div class='mark'></div><div>"
        f"<h1>{esc(title)}</h1><div class='muted'>{esc(APP_NAME)} {esc(APP_VERSION)} · "
        f"{datetime.now():%Y-%m-%d %H:%M}</div></div></header>"]
    if polar is not None:
        parts.append(f"<div class='muted'>{esc(polar.geometry_name)} · {esc(polar.engine)} · "
                     f"V = {polar.velocity_ms:.1f} m/s · Re = {polar.reynolds:.3e} · M = {polar.mach:.3f}</div>")
    if notes:
        parts.append(f"<p>{esc(notes)}</p>")
    if geometry_rows:
        parts.append("<h2>Geometry</h2><table>" + "".join(
            f"<tr><td>{esc(a)}</td><td>{esc(b)}</td></tr>" for a, b in geometry_rows) + "</table>")
    if condition_rows:
        parts.append("<h2>Flow conditions</h2><table>" + "".join(
            f"<tr><td>{esc(a)}</td><td>{esc(b)}</td></tr>" for a, b in condition_rows) + "</table>")
    if images:
        parts.append("<h2>Snapshots</h2><div class='grid'>")
        for caption, img_path in images:
            try:
                data = base64.b64encode(Path(img_path).read_bytes()).decode("ascii")
            except OSError as exc:
                log.warning("Report image %s skipped: %s", img_path, exc)
                continue
            parts.append(f"<figure style='margin:0'><img src='data:image/png;base64,{data}' "
                         f"style='max-width:640px;width:100%;border-radius:10px;border:1px solid #252A31'>"
                         f"<figcaption class='muted'>{esc(caption)}</figcaption></figure>")
        parts.append("</div>")
    if kpis is not None:
        parts.append("<h2>Flow solution</h2><table>" + "".join(
            f"<tr><td>{esc(a)}</td><td>{esc(b)}</td><td class='muted'>{esc(c)}</td></tr>"
            for a, b, c in kpis.table_rows()) + "</table>")
    if polar is not None and len(polar.alpha_deg) > 1:
        a = polar.alpha_deg
        parts.append("<h2>Polar</h2><div class='grid'>")
        parts.append(_svg_chart(a, [("CL", polar.cl, "#4D8EFF")], "α (deg)", "CL"))
        parts.append(_svg_chart(polar.cd, [("CL", polar.cl, "#A08BFA")], "CD", "CL (drag polar)"))
        parts.append(_svg_chart(a, [("L/D", polar.l_over_d, "#22C55E")], "α (deg)", "L/D"))
        parts.append(_svg_chart(a, [("CM", polar.cm, "#F59E0B")], "α (deg)", "CM"))
        parts.append("</div><h2>Summary</h2><table>")
        za = polar.alpha_zero_lift_deg
        for k, v in (("CL,max", f"{polar.cl_max:.3f} at {polar.alpha_stall_deg:.1f}°"),
                     ("Best L/D", f"{polar.ld_max:.1f} at {polar.alpha_best_ld_deg:.1f}°"),
                     ("Zero-lift α", "n/a" if math.isnan(za) else f"{za:.2f}°")):
            parts.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
        parts.append("</table><h2>Data</h2><table><tr><th>α (deg)</th><th>CL</th><th>CD</th><th>CM</th>"
                     "<th>L/D</th></tr>")
        for r in polar.rows():
            parts.append("<tr>" + "".join(f"<td>{v:.4f}</td>" for v in r) + "</tr>")
        parts.append("</table>")
    engine_name = engine or (polar.engine if polar is not None else "") or "Instant Aero Engine"
    if "openfoam" in engine_name.lower():
        footer = (f"Generated with {esc(engine_name)} (RANS finite-volume CFD). Verify mesh convergence "
                  "before relying on the numbers.")
    else:
        footer = (f"Generated by the {esc(engine_name)} (potential-flow + empirical models). Results are "
                  "engineering estimates, not certified CFD.")
    parts.append(f"<footer>{footer}</footer></div></body></html>")
    out.write_text("".join(parts), encoding="utf-8")
    return out


def default_export_name(polar: PolarData, suffix: str) -> str:
    """Safe default file name for exports."""
    return sanitize_filename(f"{polar.geometry_name or 'polar'}_polar") + suffix
