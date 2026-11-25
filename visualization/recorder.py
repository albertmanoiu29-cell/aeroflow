"""Streams a widget's frames into an H.264 MP4 while a Lab runs (constant memory, any length)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QWidget

log = logging.getLogger("aeroflow.viz.recorder")


class FrameRecorder:
    """``capture()`` once per displayed frame; every ``skip``-th frame is encoded (60 FPS display -> 30 FPS
    video by default). The video keeps the size of the first frame (later frames are cropped / padded)."""

    def __init__(self, widget: QWidget, path: str | Path, fps: int = 30, skip: int = 2) -> None:
        try:
            import imageio.v2 as iio
        except ImportError as exc:
            raise RuntimeError("Video recording needs imageio and imageio-ffmpeg.") from exc
        self.widget = widget
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.skip = max(1, int(skip))
        self._n = 0
        self.frames = 0
        first = self._grab()
        self.h, self.w = first.shape[0] - first.shape[0] % 2, first.shape[1] - first.shape[1] % 2
        try:
            self._writer = iio.get_writer(str(self.path), fps=fps, codec="libx264", quality=8,
                                          pixelformat="yuv420p", macro_block_size=1)
        except Exception as exc:  # noqa: BLE001 - imageio-ffmpeg missing
            raise RuntimeError("Video recording needs the imageio-ffmpeg package (run setup_windows.bat "
                               "again, or: pip install imageio-ffmpeg).") from exc
        self._write(first)

    def _grab(self) -> np.ndarray:
        img = self.widget.grab().toImage().convertToFormat(QImage.Format.Format_RGB888)
        w, h, bpl = img.width(), img.height(), img.bytesPerLine()
        buf = np.frombuffer(img.constBits(), dtype=np.uint8, count=bpl * h).reshape(h, bpl)
        return buf[:, :w * 3].reshape(h, w, 3).copy()

    def _write(self, frame: np.ndarray) -> None:
        out = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        hh, ww = min(self.h, frame.shape[0]), min(self.w, frame.shape[1])
        out[:hh, :ww] = frame[:hh, :ww]
        self._writer.append_data(out)
        self.frames += 1

    def capture(self) -> None:
        self._n += 1
        if self._n % self.skip:
            return
        try:
            self._write(self._grab())
        except Exception as exc:  # noqa: BLE001 - never break the simulation over a frame
            log.debug("Frame skipped: %s", exc)

    def close(self) -> Optional[Path]:
        try:
            self._writer.close()
        except Exception as exc:  # noqa: BLE001
            log.warning("Closing the video failed: %s", exc)
            return None
        log.info("Video saved: %s (%d frames)", self.path, self.frames)
        return self.path
