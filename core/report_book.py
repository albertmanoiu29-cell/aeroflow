"""Combined reports: a title page (logo, title, author, organisation, date, notes), a table of contents and
one section per workspace, lab or tool, each with tables and images. Written as one self-contained HTML file
(images embedded), styled for print, so it converts to a clean PDF."""

from __future__ import annotations

import base64
import datetime as _dt
import html
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config.settings import APP_NAME, APP_VERSION


@dataclass
class Section:
    title: str
    text: str = ""
    rows: list[tuple[str, str]] = field(default_factory=list)
    images: list[tuple[str, bytes]] = field(default_factory=list)      # (caption, PNG bytes)


@dataclass
class ReportMeta:
    title: str = "Engineering report"
    subtitle: str = ""
    author: str = ""
    organisation: str = ""
    date: str = ""
    notes: str = ""
    logo: Optional[Path] = None


def _data_uri(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


_CSS = """
@page{size:A4;margin:14mm 14mm 16mm 14mm}
*{box-sizing:border-box}
body{font-family:'Segoe UI Variable Text','Segoe UI',Inter,Arial,sans-serif;color:#1B1F24;background:#fff;margin:0;
line-height:1.45;font-size:11pt}
.title{height:262mm;display:flex;flex-direction:column;justify-content:space-between;page-break-after:always}
.title .logo{max-height:34mm;max-width:80mm;object-fit:contain}
.title h1{font-size:30pt;font-weight:600;margin:0 0 6mm;color:#11151A;line-height:1.15}
.title .sub{font-size:14pt;color:#555E6B;margin-bottom:12mm}
.title .rule{height:3px;width:42mm;background:var(--accent);margin:10mm 0 8mm}
.meta td{padding:2px 14px 2px 0;border:none;font-size:11pt}.meta td:first-child{color:#6F7682}
.notes{white-space:pre-wrap;border-left:3px solid var(--accent);padding:4mm 6mm;background:#F5F7FA;margin-top:8mm}
.footer{color:#8A919C;font-size:9pt}
.toc{page-break-after:always}.toc h2{margin-top:0}.toc ol{padding-left:6mm}.toc li{margin:2mm 0;font-size:12pt}
h2{font-size:17pt;font-weight:600;margin:0 0 4mm;color:#11151A;border-bottom:1px solid #E1E4EA;padding-bottom:2mm}
section{page-break-before:always}
section:first-of-type{page-break-before:auto}
p.text{white-space:pre-wrap;color:#3A404A}
table.kv{border-collapse:collapse;margin:3mm 0 6mm;min-width:110mm;break-inside:avoid}
table.kv td{padding:1.6mm 4mm;border-bottom:1px solid #E1E4EA;font-variant-numeric:tabular-nums}
table.kv td:first-child{color:#555E6B;width:55mm}
table.kv.two{width:100%;font-size:9.5pt}table.kv.two td{padding:1.2mm 3mm}
table.kv.two td:nth-child(1),table.kv.two td:nth-child(3){color:#555E6B;width:24%}
table.kv.two td:nth-child(3){border-left:1px solid #E1E4EA}
figure{margin:4mm 0 7mm;break-inside:avoid}figure img{max-width:100%;max-height:118mm;border:1px solid #E1E4EA;
border-radius:2mm}figcaption{color:#6F7682;font-size:9.5pt;margin-top:1.5mm}
"""


def build_html(meta: ReportMeta, sections: list[Section], accent: str = "#4C8DFF") -> str:
    esc = html.escape
    date = meta.date or _dt.date.today().strftime("%d %B %Y")
    out = ["<!doctype html><html><head><meta charset='utf-8'>", f"<title>{esc(meta.title)}</title>",
           f"<style>:root{{--accent:{esc(accent)}}}{_CSS}</style></head><body>"]
    logo = ""
    if meta.logo is not None and Path(meta.logo).is_file():
        mime = mimetypes.guess_type(str(meta.logo))[0] or "image/png"
        logo = f"<img class='logo' src='{_data_uri(Path(meta.logo).read_bytes(), mime)}' alt=''>"
    rows = [(k, v) for k, v in (("Author", meta.author), ("Organisation", meta.organisation), ("Date", date),
                                ("Sections", str(len(sections)))) if v]
    out.append("<div class='title'><div>" + logo + "</div><div>")
    out.append(f"<h1>{esc(meta.title)}</h1>")
    if meta.subtitle:
        out.append(f"<div class='sub'>{esc(meta.subtitle)}</div>")
    out.append("<div class='rule'></div><table class='meta'>")
    out += [f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in rows]
    out.append("</table>")
    if meta.notes.strip():
        out.append(f"<div class='notes'>{esc(meta.notes.strip())}</div>")
    out.append(f"</div><div class='footer'>Prepared with {esc(APP_NAME)} {esc(APP_VERSION)}</div></div>")
    if len(sections) > 1:
        out.append("<div class='toc'><h2>Contents</h2><ol>")
        out += [f"<li>{esc(s.title)}</li>" for s in sections]
        out.append("</ol></div>")
    for k, s in enumerate(sections, 1):
        out.append(f"<section><h2>{k}. {esc(s.title)}</h2>")
        if s.text.strip():
            out.append(f"<p class='text'>{esc(s.text.strip())}</p>")
        for cap, data in s.images:
            out.append(f"<figure><img src='{_data_uri(data)}' alt=''><figcaption>{esc(cap)}</figcaption></figure>")
        if s.rows:
            if len(s.rows) > 10:                     # long tables in two columns
                half = (len(s.rows) + 1) // 2
                left, right = s.rows[:half], s.rows[half:] + [("", "")] * (2 * half - len(s.rows))
                out.append("<table class='kv two'>")
                out += [f"<tr><td>{esc(a)}</td><td>{esc(b)}</td><td>{esc(c)}</td><td>{esc(d)}</td></tr>"
                        for (a, b), (c, d) in zip(left, right)]
            else:
                out.append("<table class='kv'>")
                out += [f"<tr><td>{esc(a)}</td><td>{esc(b)}</td></tr>" for a, b in s.rows]
            out.append("</table>")
        out.append("</section>")
    out.append("</body></html>")
    return "".join(out)


def write(path: str | Path, meta: ReportMeta, sections: list[Section], accent: str = "#4C8DFF") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build_html(meta, sections, accent), encoding="utf-8")
    return p


__all__ = ["Section", "ReportMeta", "build_html", "write"]
