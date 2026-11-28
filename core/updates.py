"""Update check and download (GitHub Releases).

* :func:`latest_release` asks the GitHub API for the newest published release of the project. The only
  data sent is a generic ``User-Agent: AeroFlow-Studio/<version>``; nothing about the user or the PC.
* :func:`is_newer` compares dotted versions (``v1.10.0`` > ``1.9.3``).
* :func:`download` fetches the release's Windows installer with progress and cancellation, and checks the
  size against the one GitHub reports.

No Qt here: the GUI runs these on a worker thread.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from config.settings import APP_VERSION, GITHUB_REPO

log = logging.getLogger("aeroflow.core.updates")

API_LATEST = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
_HEADERS = {"User-Agent": f"AeroFlow-Studio/{APP_VERSION}", "Accept": "application/vnd.github+json"}


class UpdateError(RuntimeError):
    """The check or the download failed (offline, rate-limited, unexpected response)."""


@dataclass(frozen=True)
class ReleaseInfo:
    version: str              # "1.5.0" (without the leading v)
    name: str
    page_url: str             # the release page on GitHub
    notes: str                # release notes (Markdown)
    installer_url: str = ""   # browser_download_url of the Windows setup, if attached
    installer_name: str = ""
    installer_size: int = 0
    installer_sha256: str = ""  # from GitHub's asset digest ("sha256:…"), when GitHub provides it


def parse_version(text: str) -> tuple[int, ...]:
    """``"v1.4.0"`` -> ``(1, 4, 0)``; non-numeric suffixes (``-beta``) are ignored."""
    nums = re.findall(r"\d+", text.split("-")[0].split("+")[0])
    return tuple(int(n) for n in nums[:4]) or (0,)


def is_newer(candidate: str, current: str = APP_VERSION) -> bool:
    a, b = parse_version(candidate), parse_version(current)
    width = max(len(a), len(b))
    return a + (0,) * (width - len(a)) > b + (0,) * (width - len(b))


def latest_release(timeout: float = 8.0) -> ReleaseInfo:
    """The newest published (non-draft, non-prerelease) release. Raises :class:`UpdateError`."""
    req = urllib.request.Request(API_LATEST, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:      # noqa: S310 - fixed https URL
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise UpdateError("No published release was found.") from exc
        if exc.code == 403:
            raise UpdateError("GitHub refused the request (rate limit); try again later.") from exc
        raise UpdateError(f"GitHub answered HTTP {exc.code}.") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise UpdateError("Could not reach GitHub — check the internet connection.") from exc
    except ValueError as exc:
        raise UpdateError("GitHub sent an unreadable answer.") from exc
    tag = str(data.get("tag_name") or "")
    if not tag:
        raise UpdateError("The latest release has no version tag.")
    setup = next((a for a in data.get("assets") or []
                  if str(a.get("name", "")).lower().endswith(".exe") and "setup" in str(a.get("name", "")).lower()),
                 None)
    return ReleaseInfo(
        version=tag.lstrip("vV"),
        name=str(data.get("name") or tag),
        page_url=str(data.get("html_url") or ""),
        notes=str(data.get("body") or ""),
        installer_url=str(setup.get("browser_download_url", "")) if setup else "",
        installer_name=str(setup.get("name", "")) if setup else "",
        installer_size=int(setup.get("size", 0)) if setup else 0,
        installer_sha256=_sha256_of(setup),
    )


def _sha256_of(asset: Optional[dict]) -> str:
    digest = str((asset or {}).get("digest") or "")
    algo, _, value = digest.partition(":")
    return value.lower() if algo.lower() == "sha256" and re.fullmatch(r"[0-9a-fA-F]{64}", value) else ""


def download(info: ReleaseInfo, dest_dir: Path, progress: Optional[Callable[[int, int], None]] = None,
             cancelled: Optional[Callable[[], bool]] = None, timeout: float = 30.0) -> Path:
    """Download the installer of ``info`` into ``dest_dir``. ``progress(done, total)``; returns the file.

    Raises :class:`UpdateError` (also when cancelled; the partial file is removed)."""
    if not info.installer_url.startswith("https://"):
        raise UpdateError("This release has no Windows installer attached.")
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = Path(info.installer_name or "AeroFlowStudio-Setup.exe").name
    out = dest_dir / name
    part = out.with_suffix(out.suffix + ".part")
    req = urllib.request.Request(info.installer_url, headers={"User-Agent": _HEADERS["User-Agent"]})
    done = 0
    sha = hashlib.sha256()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, part.open("wb") as fh:  # noqa: S310
            total = int(resp.headers.get("Content-Length") or info.installer_size or 0)
            while True:
                if cancelled is not None and cancelled():
                    raise UpdateError("Download cancelled.")
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
                sha.update(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)
    except UpdateError:
        part.unlink(missing_ok=True)
        raise
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        part.unlink(missing_ok=True)
        raise UpdateError(f"The download failed: {exc}") from exc
    if info.installer_size and done != info.installer_size:
        part.unlink(missing_ok=True)
        raise UpdateError(f"The download is incomplete ({done:,} of {info.installer_size:,} bytes).")
    if info.installer_sha256 and sha.hexdigest() != info.installer_sha256:
        part.unlink(missing_ok=True)
        raise UpdateError("The download does not match the checksum GitHub published for it; it was deleted. "
                          "Try again, or download the installer from the release page.")
    part.replace(out)
    log.info("Downloaded %s (%d bytes)", out, done)
    return out
