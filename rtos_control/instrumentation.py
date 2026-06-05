"""Per-cycle CSV logging and timing helpers.

Two collaborators:

* ``TaskLogger`` — one instance per periodic task. Owns its own CSV
  file, accumulates running statistics (max/avg C, deadline misses),
  and writes one row per release with raw measurements.
* ``CycleTimer`` — a tiny context manager used inside the task body so
  the per-cycle execution time C is sampled identically for every task.

The CSV schema is the input to ``schedulability_analysis.py``. Keep it
stable; columns are documented in the header line itself.
"""

from __future__ import annotations

import csv
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

from . import config


CSV_HEADER = [
    "release_idx",           # 0-based release counter for this task
    "release_time",          # absolute perf_counter time when the body started
    "exec_time_s",           # C: measured body execution time (seconds)
    "period_s",              # T: nominal period
    "deadline_s",            # D: deadline (== T for implicit-deadline tasks)
    "finish_time",           # release_time + exec_time_s
    "deadline_abs",          # release_time + deadline_s
    "missed_deadline",       # 1 if finish_time > deadline_abs else 0
    "note",                  # free-form (e.g., "perception_unhealthy")
]


@dataclass
class TaskStats:
    """Running aggregates kept in memory for the end-of-run summary."""
    task_name: str
    period: float
    deadline: float
    releases: int = 0
    deadline_misses: int = 0
    sum_exec_time: float = 0.0
    max_exec_time: float = 0.0   # observed WCET
    min_exec_time: float = float("inf")

    @property
    def avg_exec_time(self) -> float:
        return self.sum_exec_time / self.releases if self.releases else 0.0

    @property
    def cpu_utilization(self) -> float:
        """U = WCET / T (observed)."""
        return self.max_exec_time / self.period if self.period > 0 else 0.0

    def record(self, exec_time: float, missed: bool) -> None:
        self.releases += 1
        self.sum_exec_time += exec_time
        if exec_time > self.max_exec_time:
            self.max_exec_time = exec_time
        if exec_time < self.min_exec_time:
            self.min_exec_time = exec_time
        if missed:
            self.deadline_misses += 1


class TaskLogger:
    """Per-task CSV writer + stats accumulator.

    Thread-safe enough to allow the watchdog to read ``stats`` while the
    owning task is writing rows: stats updates are protected by a lock
    and rows are appended on the owning thread only.
    """

    def __init__(
        self,
        task_name: str,
        period: float,
        deadline: Optional[float] = None,
        log_dir: str = config.LOG_DIR,
    ) -> None:
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"{task_name}.csv")
        self._path = path
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(CSV_HEADER)
        self._unflushed = 0
        self._stats_lock = threading.Lock()
        self.stats = TaskStats(
            task_name=task_name,
            period=period,
            deadline=deadline if deadline is not None else period,
        )

    @property
    def path(self) -> str:
        return self._path

    def log(
        self,
        release_idx: int,
        release_time: float,
        exec_time: float,
        note: str = "",
    ) -> bool:
        finish = release_time + exec_time
        deadline_abs = release_time + self.stats.deadline
        missed = finish > deadline_abs
        self._writer.writerow([
            release_idx,
            f"{release_time:.9f}",
            f"{exec_time:.9f}",
            f"{self.stats.period:.9f}",
            f"{self.stats.deadline:.9f}",
            f"{finish:.9f}",
            f"{deadline_abs:.9f}",
            1 if missed else 0,
            note,
        ])
        with self._stats_lock:
            self.stats.record(exec_time, missed)
        self._unflushed += 1
        if self._unflushed >= config.LOG_FLUSH_EVERY:
            self._fh.flush()
            self._unflushed = 0
        return missed

    def snapshot_stats(self) -> TaskStats:
        """Return a *copy* of the running stats so watchdog can read safely."""
        with self._stats_lock:
            s = self.stats
            return TaskStats(
                task_name=s.task_name,
                period=s.period,
                deadline=s.deadline,
                releases=s.releases,
                deadline_misses=s.deadline_misses,
                sum_exec_time=s.sum_exec_time,
                max_exec_time=s.max_exec_time,
                min_exec_time=s.min_exec_time,
            )

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except OSError:
            pass


@contextmanager
def measure(out: list):
    """Context manager that appends the elapsed seconds to ``out``."""
    start = time.perf_counter()
    try:
        yield
    finally:
        out.append(time.perf_counter() - start)


def format_stats_table(loggers) -> str:
    """Render an end-of-run summary across all task loggers."""
    rows = ["task                T(ms)   D(ms)    runs   miss   avgC(ms)  WCET(ms)   U=C/T"]
    rows.append("-" * len(rows[0]))
    total_u = 0.0
    for lg in loggers:
        s = lg.snapshot_stats()
        u = s.cpu_utilization
        total_u += u
        rows.append(
            f"{s.task_name:<18} "
            f"{s.period*1000:>6.1f} "
            f"{s.deadline*1000:>6.1f} "
            f"{s.releases:>7d} "
            f"{s.deadline_misses:>5d} "
            f"{s.avg_exec_time*1000:>9.3f} "
            f"{s.max_exec_time*1000:>9.3f} "
            f"{u:>7.3f}"
        )
    rows.append("-" * len(rows[0]))
    rows.append(f"Total observed utilization (sum WCET/T): {total_u:.3f}")
    return "\n".join(rows)


__all__ = ["TaskLogger", "TaskStats", "measure", "format_stats_table", "CSV_HEADER"]
