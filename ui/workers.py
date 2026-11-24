"""Background workers (QRunnable + QThreadPool) for every heavy operation.

Rules enforced here:

* No worker touches a widget or VTK object. Results travel back only through Qt signals, which
  are queued onto the GUI thread.
* Every worker can be cancelled. Cancellation is cooperative for the NumPy engines (checked
  between steps) and pre-emptive for OpenFOAM, where the subprocess tree is killed.
* :class:`TaskManager` keeps Python references to running workers, so their signal objects are
  never garbage-collected mid-run. It can cancel everything on shutdown.
"""

from __future__ import annotations

import logging
import threading
import traceback
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from config.settings import FlowConditions, GridQuality
from core.geometry_manager import GeometryError, GeometryModel, load_geometry_file, repair_model
from core.instant_solver import InstantSolver, SolverCancelled
from core.openfoam_bridge import (
    CaseSettings, OpenFOAMAborted, OpenFOAMEnvironment, OpenFOAMError, OpenFOAMRunner, detect_openfoam,
    load_openfoam_result, write_case,
)
from core.polar_generator import sweep_polar

log = logging.getLogger("aeroflow.ui.workers")


class WorkerCancelled(Exception):
    """Raised inside a worker when cancellation was requested."""


class WorkerSignals(QObject):
    """Signals shared by all workers (emitted from the pool thread, delivered queued)."""

    started = Signal(str)                    # task id
    progress = Signal(float, str)            # fraction 0..1, message
    message = Signal(str)                    # free-form log line
    result = Signal(object)                  # payload
    error = Signal(str, str)                 # user message, traceback
    cancelled = Signal()
    finished = Signal(str)                   # task id (always emitted last)
    # OpenFOAM extras
    residuals = Signal(int, object)          # iteration, {field: residual}
    coefficients = Signal(int, object)       # iteration, {Cl, Cd, Cm}


class BaseWorker(QRunnable):
    """A cancellable task. Subclasses implement :meth:`work`, which returns the result payload."""

    title: str = "Task"

    def __init__(self) -> None:
        super().__init__()
        self.setAutoDelete(False)          # TaskManager owns the lifetime
        self.signals = WorkerSignals()
        self.task_id = uuid.uuid4().hex[:8]
        self._cancel = threading.Event()

    # -- cancellation ------------------------------------------------------
    def cancel(self) -> None:
        """Request cancellation (thread-safe)."""
        self._cancel.set()

    @property
    def is_cancelled(self) -> bool:
        """True after :meth:`cancel`."""
        return self._cancel.is_set()

    def check_cancel(self) -> bool:
        """Callback form for the solvers (returns True to abort)."""
        return self._cancel.is_set()

    def report(self, fraction: float, message: str) -> None:
        """Emit progress (also raises if cancelled, so long loops stop promptly)."""
        if self._cancel.is_set():
            raise WorkerCancelled()
        self.signals.progress.emit(float(max(0.0, min(1.0, fraction))), message)

    # -- execution ---------------------------------------------------------
    def work(self) -> Any:
        """Do the work (pool thread). Override in subclasses."""
        raise NotImplementedError

    @Slot()
    def run(self) -> None:
        """QRunnable entry point: wraps :meth:`work` with uniform error handling."""
        self.signals.started.emit(self.task_id)
        try:
            if self.is_cancelled:                # cancelled while still queued in the pool
                raise WorkerCancelled()
            payload = self.work()
            if self.is_cancelled:
                self.signals.cancelled.emit()
            else:
                self.signals.result.emit(payload)
        except (WorkerCancelled, SolverCancelled, OpenFOAMAborted):
            self.signals.cancelled.emit()
        except (GeometryError, OpenFOAMError, ValueError) as exc:
            log.error("%s failed: %s", self.title, exc)
            self.signals.error.emit(str(exc), traceback.format_exc())
        except MemoryError:
            msg = "Out of memory — try a coarser grid or a smaller mesh."
            log.error("%s: %s", self.title, msg)
            self.signals.error.emit(msg, traceback.format_exc())
        except Exception as exc:  # noqa: BLE001 - any failure must reach the UI, never kill the thread
            log.exception("%s crashed", self.title)
            self.signals.error.emit(f"Unexpected error in {self.title}: {exc}", traceback.format_exc())
        finally:
            self.signals.finished.emit(self.task_id)


def _prepare_visuals(result: Any, report: Callable[[float, str], None]) -> None:
    """Build the viewport dataset and derived fields (vorticity, Q-criterion, total pressure) in the
    worker thread, so the GUI thread only has to draw them."""
    field = getattr(result, "field", None)
    if field is None or (field.velocity is None and field.dataset is None):
        return
    report(0.97, "Preparing visualisation fields")
    try:
        field.prepare_visual()
        if result.surface is not None:
            result.surface.to_pyvista()
    except Exception as exc:  # noqa: BLE001 - visuals are rebuilt on the GUI thread if this fails
        log.debug("Visual pre-processing skipped: %s", exc)


# ---------------------------------------------------------------------------
# Concrete workers
# ---------------------------------------------------------------------------
class LoadGeometryWorker(BaseWorker):
    """Load (and inspect) a mesh file. Result: :class:`core.geometry_manager.LoadResult`."""

    title = "Geometry import"

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    def work(self) -> Any:
        return load_geometry_file(self.path, progress=self.report)


class RepairWorker(BaseWorker):
    """Repair a mesh. Result: ``(GeometryModel, RepairReport)``."""

    title = "Mesh repair"

    def __init__(self, model: GeometryModel) -> None:
        super().__init__()
        self.model = model

    def work(self) -> Any:
        self.report(0.1, "Repairing mesh")
        out = repair_model(self.model)
        self.report(1.0, "Repair complete")
        return out


class InstantSolveWorker(BaseWorker):
    """Run Engine A. Result: :class:`core.results.SolverResult`."""

    title = "Instant Aero Engine"

    def __init__(self, solver: InstantSolver, model: GeometryModel, conditions: FlowConditions,
                 quality: GridQuality) -> None:
        super().__init__()
        self.solver, self.model, self.conditions, self.quality = solver, model, conditions, quality

    def work(self) -> Any:
        result = self.solver.solve(self.model, self.conditions, self.quality, progress=self.report,
                                   cancel=self.check_cancel)
        _prepare_visuals(result, self.report)
        return result


@dataclass(frozen=True)
class SweepRequest:
    """Parameters of a polar sweep."""

    conditions: FlowConditions
    alpha_start: float
    alpha_end: float
    alpha_step: float
    quality: GridQuality


class PolarSweepWorker(BaseWorker):
    """Run an angle-of-attack sweep. Result: :class:`core.results.PolarData`."""

    title = "Polar sweep"

    def __init__(self, solver: InstantSolver, model: GeometryModel, request: SweepRequest) -> None:
        super().__init__()
        self.solver, self.model, self.request = solver, model, request

    def work(self) -> Any:
        r = self.request
        return sweep_polar(self.solver, self.model, r.conditions, r.alpha_start, r.alpha_end, r.alpha_step,
                           r.quality, progress=self.report, cancel=self.check_cancel)


class DetectOpenFOAMWorker(BaseWorker):
    """Probe for OpenFOAM (can take several seconds). Result: :class:`OpenFOAMEnvironment`."""

    title = "OpenFOAM detection"

    def work(self) -> Any:
        self.report(0.1, "Looking for OpenFOAM")
        return detect_openfoam()


class OpenFOAMWorker(BaseWorker):
    """Generate, mesh, solve and load an OpenFOAM case. Result: :class:`core.results.SolverResult`."""

    title = "OpenFOAM"

    def __init__(self, env: OpenFOAMEnvironment, model: GeometryModel, settings: CaseSettings) -> None:
        super().__init__()
        self.env, self.model, self.settings = env, model, settings
        self._runner: Optional[OpenFOAMRunner] = None
        self._runner_lock = threading.Lock()

    def cancel(self) -> None:
        """Abort immediately: kills the running OpenFOAM process tree."""
        super().cancel()
        with self._runner_lock:
            runner = self._runner
        if runner is not None:
            runner.abort()

    def work(self) -> Any:
        self.report(0.0, "Writing OpenFOAM case")
        layout = write_case(self.model, self.settings)
        self.signals.message.emit(f"Case directory: {layout.case_dir}")
        runner = OpenFOAMRunner(
            self.env,
            on_log=self.signals.message.emit,
            on_progress=lambda f, m: self.signals.progress.emit(float(f), m),
            on_residual=lambda i, r: self.signals.residuals.emit(int(i), r),
            on_coefficients=lambda i, c: self.signals.coefficients.emit(int(i), c))
        with self._runner_lock:
            self._runner = runner
        if self.is_cancelled:
            raise OpenFOAMAborted("Simulation aborted by the user.")
        report = runner.run(layout, self.settings)
        self.report(0.99, "Loading OpenFOAM results")
        result = load_openfoam_result(self.model, layout, self.settings, report)
        _prepare_visuals(result, self.report)
        return result


# ---------------------------------------------------------------------------
# Task manager
# ---------------------------------------------------------------------------
Callback = Callable[..., None]


class _Relay(QObject):
    """GUI-thread receiver: worker signals connect to these *slots* (so delivery is queued onto the
    GUI thread), and the slots then invoke the plain Python callbacks there."""

    def __init__(self, parent: QObject, on_result: Optional[Callback], on_error: Optional[Callback],
                 on_progress: Optional[Callback], on_cancelled: Optional[Callback],
                 on_finished: Optional[Callback]) -> None:
        super().__init__(parent)
        self._cb = (on_result, on_error, on_progress, on_cancelled, on_finished)

    @Slot(object)
    def result(self, payload: object) -> None:
        if self._cb[0]:
            self._cb[0](payload)

    @Slot(str, str)
    def error(self, message: str, tb: str) -> None:
        if self._cb[1]:
            self._cb[1](message, tb)

    @Slot(float, str)
    def progress(self, fraction: float, message: str) -> None:
        if self._cb[2]:
            self._cb[2](fraction, message)

    @Slot()
    def cancelled(self) -> None:
        if self._cb[3]:
            self._cb[3]()

    @Slot(str)
    def finished(self, _task_id: str) -> None:
        try:
            if self._cb[4]:
                self._cb[4]()
        finally:
            self.deleteLater()


class TaskManager(QObject):
    """Starts workers on a thread pool and tracks them until they finish."""

    busy_changed = Signal(bool)

    def __init__(self, parent: Optional[QObject] = None, max_threads: int = 4) -> None:
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(max(2, max_threads))
        self._active: dict[str, BaseWorker] = {}
        self._lock = threading.Lock()

    def start(self, worker: BaseWorker, on_result: Optional[Callback] = None,
              on_error: Optional[Callback] = None, on_progress: Optional[Callback] = None,
              on_cancelled: Optional[Callback] = None, on_finished: Optional[Callback] = None) -> BaseWorker:
        """Connect callbacks (always executed on the GUI thread) and start ``worker``.

        Must be called from the GUI thread.
        """
        s = worker.signals
        relay = _Relay(self, on_result, on_error, on_progress, on_cancelled, on_finished)
        s.result.connect(relay.result)
        s.error.connect(relay.error)
        s.progress.connect(relay.progress)
        s.cancelled.connect(relay.cancelled)
        s.finished.connect(self._on_finished)
        s.finished.connect(relay.finished)
        with self._lock:
            was_idle = not self._active
            self._active[worker.task_id] = worker
        if was_idle:
            self.busy_changed.emit(True)
        self.pool.start(worker)
        return worker

    @Slot(str)
    def _on_finished(self, task_id: str) -> None:
        with self._lock:
            self._active.pop(task_id, None)
            idle = not self._active
        if idle:
            self.busy_changed.emit(False)

    def is_running(self, worker_type: type) -> bool:
        """True when a worker of the given class is active."""
        with self._lock:
            return any(isinstance(w, worker_type) for w in self._active.values())

    def cancel(self, worker_type: Optional[type | tuple[type, ...]] = None) -> int:
        """Cancel active workers (all, or only those of ``worker_type`` — a class or a tuple of classes).

        Returns how many were signalled."""
        with self._lock:
            targets = [w for w in self._active.values() if worker_type is None or isinstance(w, worker_type)]
        for w in targets:
            w.cancel()
        return len(targets)

    @property
    def active_count(self) -> int:
        """Number of running workers."""
        with self._lock:
            return len(self._active)

    def shutdown(self, timeout_ms: int = 4000) -> bool:
        """Cancel everything and wait for the pool. Returns True when all threads finished."""
        self.cancel()
        self.pool.clear()
        done = self.pool.waitForDone(timeout_ms)
        if not done:
            log.warning("Some background tasks did not stop within %d ms.", timeout_ms)
        return done
