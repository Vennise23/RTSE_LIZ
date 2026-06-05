"""Entry point: assemble the four periodic tasks and run.

Usage:
    python -m rtos_control.main                      # mock mode, 20s
    python -m rtos_control.main --mode mock --seconds 30
    python -m rtos_control.main --mode real          # talks to SpeedTrials2D.exe
    python -m rtos_control.main --mode real --seconds 60
"""

from __future__ import annotations

import argparse
import queue
import signal
import sys
import time
from typing import Optional

from . import config
from .game_interface import GameInterface, MockGameInterface, RealGameInterface
from .instrumentation import format_stats_table
from .scheduler import TaskRuntime, TaskSpec
from .state import Command, SharedState
from .tasks import (
    build_actuation_task,
    build_decision_task,
    build_perception_task,
    build_watchdog_task,
)


def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPEEDTRIALS2D rule-based RT control system")
    p.add_argument("--mode", choices=["mock", "real"], default="mock",
                   help="Which GameInterface implementation to use.")
    p.add_argument("--seconds", type=float, default=None,
                   help="Run duration. Default: mock=20s, real=until Ctrl+C.")
    p.add_argument("--seed", type=int, default=42,
                   help="Mock world RNG seed (mock mode only).")
    p.add_argument("--no-overlay", action="store_true",
                   help="Disable the live perception overlay window (real mode only).")
    return p.parse_args(argv)


def _build_runtime(game: GameInterface, shared: SharedState, command_q: "queue.Queue[Command]"):
    runtime = TaskRuntime()

    # We need to give Watchdog references to the loggers of the other
    # tasks. Build them first, then build watchdog last with the list.
    perception_task = runtime.add(TaskSpec(
        name="Perception",
        period=config.TASK_PERCEPTION_PERIOD,
        priority=config.TASK_PERCEPTION_PRIORITY,
        body=build_perception_task(game, shared),
    ))
    actuation_task = runtime.add(TaskSpec(
        name="Actuation",
        period=config.TASK_ACTUATION_PERIOD,
        priority=config.TASK_ACTUATION_PRIORITY,
        body=build_actuation_task(game, shared, command_q),
    ))
    decision_task = runtime.add(TaskSpec(
        name="Decision",
        period=config.TASK_DECISION_PERIOD,
        priority=config.TASK_DECISION_PRIORITY,
        body=build_decision_task(shared, command_q),
    ))
    watchdog_task = runtime.add(TaskSpec(
        name="Watchdog",
        period=config.TASK_WATCHDOG_PERIOD,
        priority=config.TASK_WATCHDOG_PRIORITY,
        body=build_watchdog_task(
            shared,
            command_q,
            monitored_loggers=[perception_task.logger,
                               decision_task.logger,
                               actuation_task.logger],
        ),
    ))
    return runtime


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if args.mode == "mock":
        game: GameInterface = MockGameInterface(seed=args.seed)
        default_seconds = config.MOCK_RUN_SECONDS_DEFAULT
    else:
        game = RealGameInterface(show_overlay=not args.no_overlay)
        default_seconds = None

    run_seconds = args.seconds if args.seconds is not None else default_seconds

    shared = SharedState()
    command_q: "queue.Queue[Command]" = queue.Queue(maxsize=config.COMMAND_QUEUE_MAX)

    runtime = _build_runtime(game, shared, command_q)

    print(f"[main] Starting SPEEDTRIALS2D RT control system "
          f"(mode={args.mode}, run={'until Ctrl+C' if run_seconds is None else f'{run_seconds:.1f}s'})")
    game.start()
    runtime.start_all()

    # Install a Ctrl+C handler that flips the stop event.
    def _sigint(_signo, _frame):
        print("\n[main] SIGINT received, stopping...")
        runtime.stop_event.set()
    signal.signal(signal.SIGINT, _sigint)

    try:
        if run_seconds is None:
            # Real mode: run forever; wake periodically to check stop event.
            while not runtime.stop_event.is_set():
                runtime.stop_event.wait(0.5)
        else:
            deadline = time.perf_counter() + run_seconds
            while not runtime.stop_event.is_set() and time.perf_counter() < deadline:
                runtime.stop_event.wait(min(0.5, deadline - time.perf_counter()))
    finally:
        runtime.stop_all()
        game.stop()

    print("\n[main] Run complete. Per-task summary:")
    print(format_stats_table(runtime.loggers))
    print("\n[main] CSV logs written to:")
    for lg in runtime.loggers:
        print(f"  - {lg.path}")
    print("\n[main] Next step: python -m rtos_control.schedulability_analysis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
