"""Data model and the mutex-protected shared snapshot.

The shared snapshot is the single rendezvous point between
Perception (writer), Decision (reader), and Watchdog (reader).
It plays the role uC/OS-II would give to a mailbox protected by a
``OSMutex``: at any instant at most one task holds the lock, so the
snapshot a reader sees is internally consistent.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import List, Optional


# ----------------------------------------------------------------------
# Value objects (immutable on the wire — copy on read, never mutate in place)
# ----------------------------------------------------------------------
class TokenColor(str, Enum):
    GREEN = "green"
    RED = "red"
    YELLOW = "yellow"   # treated as a wildcard / unknown for safety


@dataclass(frozen=True)
class Token:
    """A perception-side observation of a colored token on the road."""
    lane: int           # 0 .. NUM_LANES-1
    distance: float     # normalized [0, 1]; 0 = at the car, 1 = horizon
    color: TokenColor


@dataclass(frozen=True)
class Obstacle:
    """A solid obstacle (other car, debris, etc.). Always treated as danger."""
    lane: int
    distance: float


@dataclass(frozen=True)
class GameState:
    """A single immutable snapshot of what perception sees this cycle."""
    timestamp: float                 # perf_counter() at the moment of capture
    own_lane: int                    # which lane the car is currently in
    speed_norm: float                # normalized 0..1 (1 = top speed)
    tokens: tuple = ()               # tuple[Token, ...]
    obstacles: tuple = ()            # tuple[Obstacle, ...]
    perception_healthy: bool = True  # set False if the underlying sensor failed

    @staticmethod
    def empty() -> "GameState":
        return GameState(
            timestamp=time.perf_counter(),
            own_lane=1,
            speed_norm=0.0,
            tokens=(),
            obstacles=(),
            perception_healthy=False,
        )


# ----------------------------------------------------------------------
# Commands flowing from Decision -> Actuation
# ----------------------------------------------------------------------
class CommandKind(str, Enum):
    HOLD = "HOLD"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    SPEED_UP = "SPEED_UP"
    SLOW_DOWN = "SLOW_DOWN"
    DEGRADE = "DEGRADE"       # watchdog-issued: centre + slow down


@dataclass(frozen=True)
class Command:
    kind: CommandKind
    issued_at: float                  # perf_counter() at decision time
    source: str = "decision"          # "decision" | "watchdog"
    reason: str = ""                  # short human-readable justification (for logs)


# ----------------------------------------------------------------------
# Mutex-protected shared snapshot
# ----------------------------------------------------------------------
class SharedState:
    """
    Thread-safe container for the latest perception snapshot plus a few
    health flags. Modeled on uC/OS-II's mailbox-behind-mutex pattern.

    Concurrency notes:
      * Writers (Perception) call ``update_state`` and may also update
        the ``perception_consecutive_misses`` counter via dedicated
        helpers.
      * Readers (Decision, Watchdog) call ``snapshot`` which returns a
        copy of the immutable GameState plus auxiliary fields, so the
        caller can release the lock immediately and reason on a stable
        view.
      * Python's standard ``threading.Lock`` does **not** implement
        priority inheritance; under heavy contention a low-priority
        writer could in principle delay a high-priority reader. We
        document this as a known gap vs real uC/OS-II in README.md.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: GameState = GameState.empty()
        self._perception_misses: int = 0
        # Actuation reports the last commanded lane so Decision knows the
        # target it asked for (vs perception's measurement of where we are).
        self._last_target_lane: int = 1
        self._last_target_set_at: float = 0.0
        # Watchdog raises this flag when the system is degraded so Decision
        # can short-circuit risky moves until perception recovers.
        self._degraded: bool = False

    # ---- writers --------------------------------------------------
    def update_state(self, state: GameState) -> None:
        with self._lock:
            self._state = state
            if state.perception_healthy:
                self._perception_misses = 0
            else:
                self._perception_misses += 1

    def record_perception_miss(self) -> None:
        with self._lock:
            self._perception_misses += 1

    def set_target_lane(self, lane: int) -> None:
        with self._lock:
            self._last_target_lane = lane
            self._last_target_set_at = time.perf_counter()

    def set_degraded(self, degraded: bool) -> None:
        with self._lock:
            self._degraded = degraded

    # ---- readers --------------------------------------------------
    def snapshot(self) -> "StateSnapshot":
        with self._lock:
            return StateSnapshot(
                state=self._state,
                perception_misses=self._perception_misses,
                last_target_lane=self._last_target_lane,
                last_target_set_at=self._last_target_set_at,
                degraded=self._degraded,
            )


@dataclass(frozen=True)
class StateSnapshot:
    """Read-only view returned by SharedState.snapshot()."""
    state: GameState
    perception_misses: int
    last_target_lane: int
    last_target_set_at: float
    degraded: bool

    def state_age(self, now: Optional[float] = None) -> float:
        return (now if now is not None else time.perf_counter()) - self.state.timestamp


__all__ = [
    "TokenColor",
    "Token",
    "Obstacle",
    "GameState",
    "CommandKind",
    "Command",
    "SharedState",
    "StateSnapshot",
]
