"""Update check / download and the system report (network fully mocked)."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import updates  # noqa: E402
from core.updates import ReleaseInfo, UpdateError, is_newer, parse_version  # noqa: E402


class _Resp(io.BytesIO):
    def __init__(self, data: bytes, length: bool = True) -> None:
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))} if length else {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_version_compare() -> None:
    assert parse_version("v1.4.0") == (1, 4, 0)
    assert parse_version("2.0.0-beta.1") == (2, 0, 0)
    assert is_newer("1.10.0", "1.9.9")
    assert is_newer("v1.4.1", "1.4.0")
    assert not is_newer("1.4", "1.4.0")
    assert not is_newer("1.3.9", "1.4.0")


def test_latest_release_parses_github_answer(monkeypatch) -> None:
    answer = {"tag_name": "v9.9.0", "name": "AeroFlow Studio 9.9.0", "html_url": "https://example/rel",
              "body": "notes", "assets": [{"name": "AeroFlow.Studio.exe", "browser_download_url": "https://x/a",
                                           "size": 1},
                                          {"name": "AeroFlowStudio-Setup-9.9.0.exe",
                                           "browser_download_url": "https://x/setup.exe", "size": 123}]}
    seen = {}

    def fake_urlopen(req, timeout):
        seen["ua"] = req.get_header("User-agent")
        return _Resp(json.dumps(answer).encode())

    monkeypatch.setattr(updates.urllib.request, "urlopen", fake_urlopen)
    info = updates.latest_release()
    assert info.version == "9.9.0" and is_newer(info.version)
    assert info.installer_name == "AeroFlowStudio-Setup-9.9.0.exe" and info.installer_size == 123
    assert seen["ua"].startswith("AeroFlow-Studio/")          # generic: nothing about the user


def test_latest_release_offline(monkeypatch) -> None:
    def fail(req, timeout):
        raise updates.urllib.error.URLError("no network")

    monkeypatch.setattr(updates.urllib.request, "urlopen", fail)
    with pytest.raises(UpdateError, match="internet"):
        updates.latest_release()


def test_download_checks_size_and_cancels(monkeypatch, tmp_path) -> None:
    payload = b"x" * 200_000
    info = ReleaseInfo("9.9.0", "n", "https://p", "", "https://x/setup.exe", "Setup-9.9.0.exe", len(payload))
    monkeypatch.setattr(updates.urllib.request, "urlopen", lambda req, timeout: _Resp(payload))
    got = []
    out = updates.download(info, tmp_path, progress=lambda d, t: got.append((d, t)))
    assert out.read_bytes() == payload and got[-1] == (len(payload), len(payload))
    # wrong size -> rejected, no file left behind
    bad = ReleaseInfo("9.9.0", "n", "https://p", "", "https://x/setup.exe", "Bad.exe", len(payload) + 1)
    with pytest.raises(UpdateError, match="incomplete"):
        updates.download(bad, tmp_path)
    assert not (tmp_path / "Bad.exe").exists() and not (tmp_path / "Bad.exe.part").exists()
    with pytest.raises(UpdateError, match="cancelled"):
        updates.download(info, tmp_path / "c", cancelled=lambda: True)
    # not https -> refused
    with pytest.raises(UpdateError):
        updates.download(ReleaseInfo("1", "n", "", "", "http://insecure/x.exe"), tmp_path)


def test_system_report_hides_home_folder() -> None:
    from utils.sysinfo import system_report

    rep = system_report("OpenFOAM test")
    assert "AeroFlow Studio" in rep and "OpenFOAM test" in rep
    assert str(Path.home()) not in rep
