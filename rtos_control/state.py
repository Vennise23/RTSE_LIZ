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

from . import config

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
    lane: int           # 0 .. NUM_LANES-1 (kept for compatibility)
    distance: float     # normalized [0, 1]; 0 = at the car, 1 = horizon
    color: TokenColor
    x_pos: float = 0.5  # normalized horizontal position in [0, 1]


@dataclass(frozen=True)
class Obstacle:
    """A solid obstacle (other car, debris, etc.). Always treated as danger."""
    lane: int
    distance: float
    x_pos: float = 0.5


@dataclass(frozen=True)
class GameState:
    """A single immutable snapshot of what perception sees this cycle."""
    timestamp: float                 # perf_counter() at the moment of capture
    own_lane: int                    # which lane the car is currently in
    speed_norm: float                # normalized 0..1 (1 = top speed)
    brightness: float = 1.0         # normalized 0..1, 1 = fully lit
    low_light_active: bool = False   # challenge 1 active flag
    rear_pressure: float = 0.0       # normalized 0..1, higher means the chase car is closer
    rear_chase_active: bool = False  # challenge 2 active flag
    rear_chase_lane: int = -1        # lane of the chasing car, -1 when inactive
    rear_time_left: float = 0.0      # seconds left before the chase expires
    police_alert: bool = False       # challenge 3 active flag
    police_lane: int = -1           # lane of the police car, -1 when inactive
    police_time_left: float = 0.0   # seconds left before the police challenge expires
    golden_lane_active: bool = False # golden lane event flag
    golden_lane_index: int = -1      # lane index that is golden, -1 when inactive
    golden_time_left: float = 0.0    # seconds left before golden lane expires
    golden_lane_passed: bool = False # whether player was in the golden lane at expiry
    gold_tokens_collected: int = 0    # cumulative green tokens collected
    red_tokens_collected: int = 0     # cumulative red tokens collected
    event_pass_count: int = 0         # how many required events were passed
    tactical_win: bool = False       # tactical victory flag
    game_over: bool = False          # set True when the police car collides with the player
    game_over_reason: str = ""
    tokens: tuple = ()               # tuple[Token, ...]
    obstacles: tuple = ()            # tuple[Obstacle, ...]
    perception_healthy: bool = True  # set False if the underlying sensor failed

    @staticmethod
    def empty() -> "GameState":
        return GameState(
            timestamp=time.perf_counter(),
            own_lane=config.LANE_CENTER_INDEX,
            speed_norm=0.0,
            brightness=1.0,
            low_light_active=False,
            rear_pressure=0.0,
            rear_chase_active=False,
            rear_chase_lane=-1,
            rear_time_left=0.0,
            police_alert=False,
            police_lane=-1,
            police_time_left=0.0,
            golden_lane_active=False,
            golden_lane_index=-1,
            golden_time_left=0.0,
            golden_lane_passed=False,
            gold_tokens_collected=0,
            red_tokens_collected=0,
            event_pass_count=0,
            tactical_win=False,
            game_over=False,
            game_over_reason="",
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
    RECOVER_LIGHT = "RECOVER_LIGHT"
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
        self._last_target_lane: int = config.LANE_CENTER_INDEX
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
