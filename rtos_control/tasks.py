"""Concrete periodic task bodies.

Each ``build_*_task`` factory closes over the shared collaborators
(game interface, shared state, command queue, sibling task loggers)
and returns a stateless callable suitable for ``PeriodicTask``.

Why closures instead of classes? The decision policy itself is pure
(see decision.py); only the wiring lives here. Keeping the task body
as a single short callable also makes it trivial to time the body with
``perf_counter`` from inside the scheduler — no method-call overhead
to subtract from the WCET measurement.
"""

from __future__ import annotations

import queue
import time
from typing import List, Optional

from . import config
from .decision import DecisionMemory, decide
from .game_interface import GameInterface
from .instrumentation import TaskLogger
from .state import Command, CommandKind, SharedState


# ----------------------------------------------------------------------
# Perception
# ----------------------------------------------------------------------
def build_perception_task(game: GameInterface, shared: SharedState):
    """20 ms task. Pull a fresh snapshot and store it under the mutex."""
    def body() -> Optional[str]:
        state = game.read_state()
        shared.update_state(state)
        if not state.perception_healthy:
            return "perception_unhealthy"
        return ""
    return body


# ----------------------------------------------------------------------
# Decision
# ----------------------------------------------------------------------
def build_decision_task(
    shared: SharedState,
    command_q: "queue.Queue[Command]",
):
    """50 ms task. Read shared state, run policy, push command."""
    memory: List[DecisionMemory] = [DecisionMemory()]  # mutable via closure

    def body() -> Optional[str]:
        snap = shared.snapshot()
        # If perception is stale we should not act on it. Push a HOLD
        # so actuation just keeps the wheel centred and let watchdog
        # decide whether to escalate.
        if snap.state_age() > config.WATCHDOG_STALE_STATE_SEC:
            cmd = Command(
                kind=CommandKind.HOLD,
                issued_at=time.perf_counter(),
                reason="state_stale",
            )
            _enqueue_latest(command_q, cmd)
            return "state_stale"

        # If watchdog has declared the system degraded, force a HOLD
        # too — actuation will apply the degraded throttle.
        if snap.degraded:
            cmd = Command(
                kind=CommandKind.HOLD,
                issued_at=time.perf_counter(),
                reason="degraded",
                source="decision",
            )
            _enqueue_latest(command_q, cmd)
            return "degraded_hold"

        result = decide(snap.state, memory[0])
        memory[0] = result.memory
        _enqueue_latest(command_q, result.command)
        return result.command.kind.value
    return body


def _enqueue_latest(q: "queue.Queue[Command]", cmd: Command) -> None:
    """Push ``cmd`` onto the queue, evicting an older command if full.

    Rationale: stale commands are worse than no command at all — by
    the time the actuator picks them up the world has moved on. We
    keep the queue shallow (config.COMMAND_QUEUE_MAX) and overwrite.
    """
    try:
        q.put_nowait(cmd)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(cmd)
        except queue.Full:
            pass


# ----------------------------------------------------------------------
# Actuation
# ----------------------------------------------------------------------
def build_actuation_task(
    game: GameInterface,
    shared: SharedState,
    command_q: "queue.Queue[Command]",
):
    """30 ms task. Drain the most recent command and push it to the game."""
    # Internal state: how long to keep the wheel pulsed and which way.
    pulse_state = {
        "until": 0.0,
        "steering": 0.0,
        "acceleration": config.ACCEL_CRUISE,
    }

    def body() -> Optional[str]:
        now = time.perf_counter()
        # Drain queue to keep only the freshest command. We discard
        # any backlog rather than execute stale instructions.
        latest: Optional[Command] = None
        try:
            while True:
                latest = command_q.get_nowait()
        except queue.Empty:
            pass

        if latest is not None:
            steering, acceleration, hold_until = _command_to_floats(latest, now)
            pulse_state["steering"] = steering
            pulse_state["acceleration"] = acceleration
            pulse_state["until"] = hold_until
            if latest.kind in (CommandKind.MOVE_LEFT, CommandKind.MOVE_RIGHT):
                new_lane = shared.snapshot().state.own_lane + (
                    -1 if latest.kind is CommandKind.MOVE_LEFT else +1
                )
                new_lane = max(0, min(config.NUM_LANES - 1, new_lane))
                shared.set_target_lane(new_lane)

        # If the steering pulse has expired, recentre the wheel.
        if now >= pulse_state["until"]:
            pulse_state["steering"] = 0.0

        # Watchdog override: if the system is degraded, dampen everything.
        if shared.snapshot().degraded:
            pulse_state["acceleration"] = config.ACCEL_BRAKE
            pulse_state["steering"] = 0.0

        game.send_command(pulse_state["steering"], pulse_state["acceleration"])

        if latest is not None:
            return f"sent:{latest.kind.value}"
        return ""
    return body


def _command_to_floats(cmd: Command, now: float):
    """Map a discrete Command to (steering, acceleration, hold_until)."""
    if cmd.kind is CommandKind.MOVE_LEFT:
        return (-config.STEER_MAGNITUDE, config.ACCEL_CRUISE, now + config.LANE_HOLD_TIME)
    if cmd.kind is CommandKind.MOVE_RIGHT:
        return (+config.STEER_MAGNITUDE, config.ACCEL_CRUISE, now + config.LANE_HOLD_TIME)
    if cmd.kind is CommandKind.SLOW_DOWN:
        return (0.0, config.ACCEL_SLOWDOWN, now)
    if cmd.kind is CommandKind.SPEED_UP:
        return (0.0, config.ACCEL_CRUISE, now)
    if cmd.kind is CommandKind.DEGRADE:
        return (0.0, config.ACCEL_BRAKE, now)
    # HOLD
    return (0.0, config.ACCEL_CRUISE, now)


# ----------------------------------------------------------------------
# Watchdog
# ----------------------------------------------------------------------
def build_watchdog_task(
    shared: SharedState,
    command_q: "queue.Queue[Command]",
    monitored_loggers: List[TaskLogger],
):
    """20 ms task. Highest priority. Detects:
       - perception staleness
       - persistent deadline misses on Perception or Decision
       and triggers a degrade command (centre + slow down).
    """
    # Per-task baselines so we only react to *new* misses since last tick.
    last_miss_counts = {lg.stats.task_name: 0 for lg in monitored_loggers}

    def body() -> Optional[str]:
        now = time.perf_counter()
        snap = shared.snapshot()

        notes = []
        degrade = False

        # Staleness check
        age = snap.state_age(now)
        if age > config.WATCHDOG_STALE_STATE_SEC:
            degrade = True
            notes.append(f"stale_state_{age*1000:.1f}ms")

        # Misses since last watchdog tick
        for lg in monitored_loggers:
            s = lg.snapshot_stats()
            delta = s.deadline_misses - last_miss_counts.get(s.task_name, 0)
            last_miss_counts[s.task_name] = s.deadline_misses
            if delta > 0:
                notes.append(f"{s.task_name}_miss+{delta}")
                if s.task_name in ("Perception", "Decision") and delta >= 1:
                    degrade = True

        # Consecutive perception unhealthy snapshots
        if snap.perception_misses >= config.WATCHDOG_PERCEPTION_MISS_LIMIT:
            degrade = True
            notes.append(f"perception_unhealthy_x{snap.perception_misses}")

        if degrade:
            shared.set_degraded(True)
            # Highest-priority command bypasses the queue's age — we
            # still go through the queue so the actuator's drain logic
            # always sees the freshest entry.
            cmd = Command(
                kind=CommandKind.DEGRADE,
                issued_at=now,
                source="watchdog",
                reason=";".join(notes) or "degrade",
            )
            _enqueue_latest(command_q, cmd)
        else:
            # Auto-recover when conditions return to normal.
            if snap.degraded and snap.perception_misses == 0 and age <= config.WATCHDOG_STALE_STATE_SEC:
                shared.set_degraded(False)
                notes.append("recovered")

        return ";".join(notes)
    return body


__all__ = [
    "build_perception_task",
    "build_decision_task",
    "build_actuation_task",
    "build_watchdog_task",
]
