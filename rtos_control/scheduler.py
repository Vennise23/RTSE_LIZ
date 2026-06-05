"""Periodic-task scheduling framework modelled on uC/OS-II.

Design notes:

* Each task is a ``threading.Thread`` released on a fixed period using
  **absolute** wake-up times (``next_release += period``), not the naive
  ``sleep(period)`` pattern. This eliminates the slow drift that would
  otherwise accumulate from execution-time jitter.

* uC/OS-II convention is followed for priorities: **the smaller the
  number, the higher the priority**. We pass that number through to the
  OS thread priority (best-effort) so the host scheduler tries to give
  Watchdog/Perception preference over Decision, although CPython's GIL
  caps the benefit.

* GIL caveat: CPython only runs one Python bytecode at a time, so true
  preemption between tasks does not exist; cooperative yielding happens
  at the bytecode-tick boundary and on I/O calls. ``threading.Lock``
  has no priority inheritance, so a low-priority task can briefly delay
  a higher-priority one. Both are deliberate gaps vs real uC/OS-II;
  README.md discusses how the analysis still remains meaningful.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from .instrumentation import TaskLogger


@dataclass
class TaskSpec:
    name: str
    period: float
    priority: int            # uC/OS-II style: 1 = highest
    body: Callable[[], Optional[str]]
    deadline: Optional[float] = None  # defaults to period

    def deadline_value(self) -> float:
        return self.deadline if self.deadline is not None else self.period


# Best-effort OS priority mapping. The exact numbers below are Windows
# THREAD_PRIORITY_* constants; on POSIX we just fall back gracefully.
def _apply_os_priority(priority: int) -> None:
    """Map our 1..N priority to the best the OS will let us have."""
    try:
        if sys.platform.startswith("win"):
            import ctypes  # type: ignore
            handle = ctypes.windll.kernel32.GetCurrentThread()
            # 1->2 (HIGHEST), 2->1 (ABOVE_NORMAL), 3->0 (NORMAL), >=4 -> -1 (BELOW_NORMAL)
            mapping = {1: 2, 2: 1, 3: 0}
            ctypes.windll.kernel32.SetThreadPriority(handle, mapping.get(priority, -1))
        else:
            # POSIX: nice is process-wide so we don't touch it from a
            # thread. SCHED_FIFO would need root + careful CAP_SYS_NICE.
            # Document this in README; do not raise.
            pass
    except Exception:
        # Never let a missing OS API kill the task; we still get a
        # logically-priority-tagged Python thread.
        pass


class PeriodicTask(threading.Thread):
    """One periodic real-time task.

    The ``body`` callable is invoked once per release. Its return value
    (an optional string) is written into the CSV's ``note`` column,
    which gives the task an audit trail for transient events (e.g.,
    "lane_change_LEFT", "perception_stale").
    """

    def __init__(
        self,
        spec: TaskSpec,
        stop_event: threading.Event,
        logger: TaskLogger,
    ) -> None:
        super().__init__(name=spec.name, daemon=True)
        self.spec = spec
        self._stop_event = stop_event
        self._logger = logger
        self._release_idx = 0

    @property
    def logger(self) -> TaskLogger:
        return self._logger

    def run(self) -> None:
        _apply_os_priority(self.spec.priority)
        period = self.spec.period
        # Absolute-time release schedule: next_release advances by period
        # regardless of how long the body took. If the body overruns we
        # log the deadline miss but do not let the slip propagate
        # forward (catch up by skipping idle sleep).
        next_release = time.perf_counter()
        while not self._stop_event.is_set():
            release_time = time.perf_counter()
            note = ""
            try:
                result = self.spec.body()
                if isinstance(result, str):
                    note = result
            except Exception as exc:  # pragma: no cover - defensive
                note = f"exc:{type(exc).__name__}:{exc}"
            exec_time = time.perf_counter() - release_time
            self._logger.log(
                release_idx=self._release_idx,
                release_time=release_time,
                exec_time=exec_time,
                note=note,
            )
            self._release_idx += 1

            next_release += period
            sleep_for = next_release - time.perf_counter()
            if sleep_for > 0:
                # ``Event.wait`` lets us exit promptly on shutdown.
                self._stop_event.wait(sleep_for)
            else:
                # We are late. Realign next_release to "now" so we do
                # not enter a tight catch-up burst that would starve
                # other tasks. This matches uC/OS-II's behavior when a
                # task overruns by more than one period.
                next_release = time.perf_counter()


class TaskRuntime:
    """Holds a set of periodic tasks and starts/stops them together."""

    def __init__(self) -> None:
        self._tasks: List[PeriodicTask] = []
        self._loggers: List[TaskLogger] = []
        self._stop_event = threading.Event()

    def add(self, spec: TaskSpec) -> PeriodicTask:
        logger = TaskLogger(
            task_name=spec.name,
            period=spec.period,
            deadline=spec.deadline_value(),
        )
        task = PeriodicTask(spec, self._stop_event, logger)
        self._tasks.append(task)
        self._loggers.append(logger)
        return task

    @property
    def loggers(self) -> List[TaskLogger]:
        return list(self._loggers)

    @property
    def tasks(self) -> List[PeriodicTask]:
        return list(self._tasks)

    def start_all(self) -> None:
        # Higher priority first, so on platforms where thread creation
        # order weakly biases scheduling order, the watchdog wins ties.
        for task in sorted(self._tasks, key=lambda t: t.spec.priority):
            task.start()

    def stop_all(self, join_timeout: float = 1.0) -> None:
        self._stop_event.set()
        for task in self._tasks:
            task.join(timeout=join_timeout)
        for logger in self._loggers:
            logger.close()

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event


__all__ = ["TaskSpec", "PeriodicTask", "TaskRuntime"]
