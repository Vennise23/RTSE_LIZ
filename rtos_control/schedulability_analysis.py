"""Offline schedulability analysis from the per-task CSV logs.

Reads every ``<TaskName>.csv`` written by ``instrumentation.TaskLogger``
under the configured ``LOG_DIR`` and prints, for each task:

    * Observed WCET (max exec_time_s) and average C.
    * Per-task CPU utilization U_i = WCET_i / T_i.
    * Deadline-miss count from the CSV (sanity-check against the
      ``missed_deadline`` column).

Then for the task set as a whole:

    * Total utilization Sum(U_i).
    * Liu & Layland RMS sufficient bound  U <= n * (2^(1/n) - 1).
    * EDF necessary-and-sufficient bound  U <= 1 (implicit-deadline).
    * Response-Time Analysis (RTA) for fixed-priority preemptive
      scheduling, treating the uC/OS-II priority number as the rate-
      monotonic priority (smaller = higher). Iterates R until fixed
      point and compares against D.

Implementation is pure standard library: no pandas, no numpy.
"""

from __future__ import annotations

import csv
import glob
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Try to read config.LOG_DIR; fall back to "logs" if invoked stand-alone.
try:
    from . import config  # type: ignore
    DEFAULT_LOG_DIR = config.LOG_DIR
    DEFAULT_PRIORITIES = {
        "Watchdog":   config.TASK_WATCHDOG_PRIORITY,
        "Perception": config.TASK_PERCEPTION_PRIORITY,
        "Actuation":  config.TASK_ACTUATION_PRIORITY,
        "Decision":   config.TASK_DECISION_PRIORITY,
    }
except ImportError:
    DEFAULT_LOG_DIR = "logs"
    DEFAULT_PRIORITIES = {
        "Watchdog": 1, "Perception": 2, "Actuation": 3, "Decision": 4,
    }


# ----------------------------------------------------------------------
# Data ingestion
# ----------------------------------------------------------------------
@dataclass
class TaskMetrics:
    name: str
    period: float
    deadline: float
    wcet: float
    avg_c: float
    releases: int
    missed: int
    priority: int               # 1 = highest

    @property
    def utilization(self) -> float:
        return self.wcet / self.period if self.period > 0 else 0.0


def _load_one(csv_path: str, priority: int) -> Optional[TaskMetrics]:
    name = os.path.splitext(os.path.basename(csv_path))[0]
    period = 0.0
    deadline = 0.0
    wcet = 0.0
    sum_c = 0.0
    releases = 0
    missed = 0
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                c = float(row["exec_time_s"])
                t = float(row["period_s"])
                d = float(row["deadline_s"])
                m = int(row["missed_deadline"])
            except (KeyError, ValueError):
                continue
            period = t
            deadline = d
            sum_c += c
            if c > wcet:
                wcet = c
            if m:
                missed += 1
            releases += 1
    if releases == 0:
        return None
    return TaskMetrics(
        name=name,
        period=period,
        deadline=deadline,
        wcet=wcet,
        avg_c=sum_c / releases,
        releases=releases,
        missed=missed,
        priority=priority,
    )


def load_metrics(
    log_dir: str = DEFAULT_LOG_DIR,
    priorities: Dict[str, int] = DEFAULT_PRIORITIES,
) -> List[TaskMetrics]:
    out: List[TaskMetrics] = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        prio = priorities.get(name, 99)
        m = _load_one(path, prio)
        if m is not None:
            out.append(m)
    out.sort(key=lambda m: m.priority)  # high priority first
    return out


# ----------------------------------------------------------------------
# Schedulability tests
# ----------------------------------------------------------------------
def rms_bound(n: int) -> float:
    """Liu & Layland sufficient bound: n * (2^(1/n) - 1)."""
    if n <= 0:
        return 0.0
    return n * (2.0 ** (1.0 / n) - 1.0)


def edf_bound() -> float:
    """EDF necessary-and-sufficient bound for implicit deadlines."""
    return 1.0


def rta_response_time(
    target: TaskMetrics,
    higher_priority: List[TaskMetrics],
    max_iter: int = 1000,
) -> Tuple[float, bool]:
    """Worst-case response time R_i via fixed-point iteration.

    R^{k+1}_i = C_i + sum_{j in hp(i)} ceil(R^k_i / T_j) * C_j

    Returns (R, converged). Converged means R stopped growing before
    exceeding the task's deadline; if R > D the task is infeasible.
    """
    c = target.wcet
    d = target.deadline
    r = c
    for _ in range(max_iter):
        interference = sum(
            math.ceil(r / hp.period) * hp.wcet for hp in higher_priority
        )
        r_next = c + interference
        if r_next > d:
            return (r_next, False)  # exceeds deadline; not schedulable
        if math.isclose(r_next, r, rel_tol=0.0, abs_tol=1e-9):
            return (r_next, True)
        r = r_next
    return (r, False)


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def render_report(metrics: List[TaskMetrics]) -> str:
    if not metrics:
        return "No CSV logs found. Run 'python -m rtos_control.main' first."

    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("SPEEDTRIALS2D -- Schedulability analysis report")
    lines.append("=" * 78)

    # ---- Per-task table ------------------------------------------
    lines.append("")
    lines.append("Per-task measurements (sorted by priority; smaller number = higher)")
    lines.append("-" * 78)
    header = (f"{'task':<14} {'prio':>4} {'T(ms)':>8} {'D(ms)':>8} "
              f"{'runs':>6} {'miss':>6} {'avgC(ms)':>10} {'WCET(ms)':>10} {'U=C/T':>8}")
    lines.append(header)
    lines.append("-" * 78)
    total_u = 0.0
    for m in metrics:
        total_u += m.utilization
        lines.append(
            f"{m.name:<14} {m.priority:>4d} "
            f"{m.period*1000:>8.2f} {m.deadline*1000:>8.2f} "
            f"{m.releases:>6d} {m.missed:>6d} "
            f"{m.avg_c*1000:>10.3f} {m.wcet*1000:>10.3f} "
            f"{m.utilization:>8.3f}"
        )
    lines.append("-" * 78)
    lines.append(f"Total observed utilization  U = {total_u:.4f}")

    # ---- RMS / EDF -----------------------------------------------
    n = len(metrics)
    rms = rms_bound(n)
    edf = edf_bound()
    lines.append("")
    lines.append("Global utilization bounds")
    lines.append("-" * 78)
    lines.append(f"  RMS (Liu & Layland)   bound for n={n}: U_lub = {rms:.4f}")
    lines.append(f"     -> U ({total_u:.4f}) {'<=' if total_u <= rms else '> '} "
                 f"U_lub  => {'PASS (sufficient)' if total_u <= rms else 'INCONCLUSIVE (need RTA)'}")
    lines.append(f"  EDF                   bound: U <= {edf:.4f}")
    lines.append(f"     -> U ({total_u:.4f}) {'<=' if total_u <= edf else '> '} "
                 f"1.0     => {'PASS' if total_u <= edf else 'FAIL'}")

    # ---- RTA -----------------------------------------------------
    lines.append("")
    lines.append("Response-Time Analysis (fixed-priority preemptive)")
    lines.append("-" * 78)
    lines.append(f"{'task':<14} {'prio':>4} {'R(ms)':>10} {'D(ms)':>10} {'verdict':>14}")
    lines.append("-" * 78)
    all_pass = True
    for i, m in enumerate(metrics):
        higher = metrics[:i]  # already sorted by priority ascending == hp first
        r, ok = rta_response_time(m, higher)
        verdict = "SCHEDULABLE" if ok and r <= m.deadline else "NOT SCHEDULABLE"
        if not ok or r > m.deadline:
            all_pass = False
        lines.append(
            f"{m.name:<14} {m.priority:>4d} "
            f"{r*1000:>10.3f} {m.deadline*1000:>10.3f} "
            f"{verdict:>14}"
        )
    lines.append("-" * 78)
    lines.append(f"Overall RTA verdict: {'SCHEDULABLE' if all_pass else 'NOT SCHEDULABLE'}")

    # ---- Observed deadline misses (sanity) -----------------------
    lines.append("")
    lines.append("Observed deadline misses during the run")
    lines.append("-" * 78)
    any_miss = False
    for m in metrics:
        if m.missed > 0:
            any_miss = True
            rate = 100.0 * m.missed / m.releases
            lines.append(f"  {m.name:<14}  {m.missed} miss(es) / "
                         f"{m.releases} releases  ({rate:.2f}%)")
    if not any_miss:
        lines.append("  None.")

    lines.append("")
    lines.append("Note on Python/GIL:")
    lines.append("  The numbers above use observed WCET on CPython. Because of")
    lines.append("  the GIL and the absence of priority inheritance on stdlib")
    lines.append("  Lock, real worst-case execution time on this platform can")
    lines.append("  exceed the observed maximum under heavy contention. See")
    lines.append("  README.md, section 'Known platform limits'.")
    lines.append("=" * 78)
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    log_dir = DEFAULT_LOG_DIR
    if argv and len(argv) >= 1:
        log_dir = argv[0]
    metrics = load_metrics(log_dir)
    print(render_report(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
