"""Structured, thread-safe logging for AeroFlow Studio.

Provides:

* :func:`setup_logging` - rotating file log + stderr + a Qt bridge handler.
* :func:`get_logger` - namespaced loggers (``aeroflow.<name>``); every logger
  gains a ``.success()`` method (custom level 25) for "converged"-type events.
* :class:`LogEmitter` / :class:`QtLogHandler` - forward records from *any*
  thread to the GUI thread via a queued Qt signal.
* :class:`ConsoleLogWidget` - colour-coded, timestamped, filterable in-app
  console with batched, non-blocking rendering (safe under OpenFOAM's output
  volume).
* :func:`install_exception_hooks` - routes uncaught exceptions (main and worker
  threads) into the log instead of silently killing the app.
"""

from __future__ import annotations

import html
import logging
import logging.handlers
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Deque, Final, Optional

from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtWidgets import QFileDialog, QTextEdit, QWidget

from config.settings import APP_ID, get_paths
from config.theme import LOG_LEVEL_COLORS, PALETTE, mono_font

# ---------------------------------------------------------------------------
# Custom SUCCESS level
# ---------------------------------------------------------------------------
SUCCESS_LEVEL: Final[int] = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def _success(self: logging.Logger, message: object, *args: Any, **kwargs: Any) -> None:
    """``logger.success(...)``: log at the custom SUCCESS level."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)  # noqa: SLF001


logging.Logger.success = _success

ROOT_LOGGER_NAME: Final[str] = "aeroflow"
_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
_setup_lock = threading.Lock()
_setup_done = False


# ---------------------------------------------------------------------------
# Qt bridge
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LogPayload:
    """One log record in a Qt-signal-friendly, immutable form."""

    timestamp: float
    level_name: str
    level_no: int
    source: str
    message: str


class LogEmitter(QObject):
    """Qt object that re-emits log records as a signal.

    Create it on the GUI thread (``setup_logging`` does). Emitting from worker
    threads is safe: Qt delivers the signal to GUI-thread slots through a queued
    connection automatically.
    """

    record_emitted = Signal(object)  # LogPayload


class QtLogHandler(logging.Handler):
    """Logging handler that forwards records to a :class:`LogEmitter`."""

    def __init__(self, emitter: LogEmitter, level: int = logging.DEBUG) -> None:
        super().__init__(level)
        self._emitter = emitter
        self._exc_formatter = logging.Formatter()

    def emit(self, record: logging.LogRecord) -> None:
        """Convert ``record`` to a :class:`LogPayload` and emit it."""
        try:
            message = record.getMessage()
            if record.exc_info:
                message += "\n" + self._exc_formatter.formatException(record.exc_info)
            source = record.name[len(ROOT_LOGGER_NAME) + 1:] if record.name.startswith(
                ROOT_LOGGER_NAME + ".") else record.name
            payload = LogPayload(record.created, record.levelname, record.levelno, source, message)
            self._emitter.record_emitted.emit(payload)
        except RuntimeError:
            # Emitter already destroyed during interpreter / Qt shutdown.
            return
        except Exception:  # noqa: BLE001 - logging must never raise
            self.handleError(record)


_emitter: Optional[LogEmitter] = None


def get_emitter() -> LogEmitter:
    """Return the process-wide :class:`LogEmitter` (create it on the GUI thread)."""
    global _emitter
    if _emitter is None:
        _emitter = LogEmitter()
    return _emitter


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def setup_logging(level: int = logging.INFO, log_dir: Optional[Path] = None,
                  console: bool = True) -> logging.Logger:
    """Configure application logging (idempotent).

    Args:
        level: Minimum level for the file / stderr handlers.
        log_dir: Directory for ``aeroflow.log`` (default: per-user data dir).
        console: Also write to stderr when a stream is available.

    Returns:
        The root ``aeroflow`` logger.
    """
    global _setup_done
    root = logging.getLogger(ROOT_LOGGER_NAME)
    with _setup_lock:
        if _setup_done:
            return root
        root.setLevel(logging.DEBUG)
        root.propagate = False
        formatter = logging.Formatter(_LOG_FORMAT, _DATE_FORMAT)

        try:
            directory = log_dir or get_paths().logs
            directory.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                directory / f"{APP_ID}.log", maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as exc:
            sys.stderr.write(f"[aeroflow] File logging disabled: {exc}\n")

        if console and sys.stderr is not None:
            stream_handler = logging.StreamHandler(sys.stderr)
            stream_handler.setLevel(level)
            stream_handler.setFormatter(formatter)
            root.addHandler(stream_handler)

        root.addHandler(QtLogHandler(get_emitter(), logging.DEBUG))
        _setup_done = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Return the logger ``aeroflow.<name>`` (``name`` may already be prefixed)."""
    if name == ROOT_LOGGER_NAME or name.startswith(ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def install_exception_hooks() -> None:
    """Log uncaught exceptions from the main thread and from ``threading`` threads."""
    log = get_logger("crash")

    def _excepthook(exc_type: type[BaseException], exc: BaseException,
                    tb: Optional[TracebackType]) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        log.critical("Uncaught exception", exc_info=(exc_type, exc, tb))

    def _thread_hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is SystemExit or args.exc_value is None:
            return
        log.critical("Uncaught exception in thread %s", getattr(args.thread, "name", "?"),
                     exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = _excepthook
    threading.excepthook = _thread_hook


# ---------------------------------------------------------------------------
# Console widget
# ---------------------------------------------------------------------------
class ConsoleLogWidget(QTextEdit):
    """Read-only, colour-coded, timestamped log console.

    Records arrive through :meth:`on_record` (connected to the emitter by
    :meth:`attach`), are buffered, and flushed to the document in batches every
    ~50 ms so that bursts of solver output never stall the GUI.
    """

    MAX_BLOCKS: Final[int] = 6000
    HISTORY: Final[int] = 20000
    FLUSH_MS: Final[int] = 50

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Console")
        self.setReadOnly(True)
        self.setFont(mono_font(11))
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.document().setMaximumBlockCount(self.MAX_BLOCKS)
        self._history: Deque[LogPayload] = deque(maxlen=self.HISTORY)
        self._pending: list[LogPayload] = []
        self._min_level = logging.DEBUG
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(self.FLUSH_MS)
        self._timer.timeout.connect(self._flush)

    # -- wiring ---------------------------------------------------------
    def attach(self, emitter: Optional[LogEmitter] = None) -> None:
        """Subscribe to log records (defaults to the global emitter)."""
        (emitter or get_emitter()).record_emitted.connect(self.on_record)

    # -- public API -----------------------------------------------------
    @Slot(object)
    def on_record(self, payload: LogPayload) -> None:
        """Receive one record (GUI thread)."""
        self._history.append(payload)
        if payload.level_no < self._min_level:
            return
        self._pending.append(payload)
        if not self._timer.isActive():
            self._timer.start()

    def set_min_level(self, level: int) -> None:
        """Filter by minimum level and re-render the retained history."""
        self._min_level = int(level)
        self.clear()
        self._pending = [p for p in self._history if p.level_no >= self._min_level][-self.MAX_BLOCKS:]
        self._flush()

    def clear_log(self) -> None:
        """Clear the display and the retained history."""
        self._history.clear()
        self._pending.clear()
        self.clear()

    def save_to_file(self, path: Optional[str] = None) -> Optional[Path]:
        """Save the retained history as plain text.

        Args:
            path: Destination; if omitted a file dialog is shown.

        Returns:
            The written path, or ``None`` if cancelled or the write failed.
        """
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Save log", "aeroflow_log.txt", "Text (*.txt)")
        if not path:
            return None
        try:
            target = Path(path)
            lines = (f"{self._format_time(p.timestamp)} {p.level_name:<8} [{p.source}] {p.message}"
                     for p in self._history)
            target.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return target
        except OSError as exc:
            get_logger("logger").error("Could not save log: %s", exc)
            return None

    # -- internals ------------------------------------------------------
    @staticmethod
    def _format_time(ts: float) -> str:
        return time.strftime("%H:%M:%S", time.localtime(ts)) + f".{int((ts % 1) * 1000):03d}"

    def _render(self, p: LogPayload) -> str:
        color = LOG_LEVEL_COLORS.get(p.level_name, PALETTE.text)
        message = html.escape(p.message)
        return (
            f'<div style="margin:0;white-space:pre-wrap;">'
            f'<span style="color:{PALETTE.text_dim};">{self._format_time(p.timestamp)}</span> '
            f'<span style="color:{color};font-weight:700;">{p.level_name[:4]:<4}</span> '
            f'<span style="color:{PALETTE.indigo};">{html.escape(p.source)}</span> '
            f'<span style="color:{color};">{message}</span></div>'
        )

    @Slot()
    def _flush(self) -> None:
        """Append all buffered records in one batch, keeping auto-scroll behaviour."""
        if not self._pending:
            return
        batch, self._pending = self._pending, []
        bar = self.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 4
        self.setUpdatesEnabled(False)
        try:
            for payload in batch:
                self.append(self._render(payload))
        finally:
            self.setUpdatesEnabled(True)
        if at_bottom:
            bar.setValue(bar.maximum())
