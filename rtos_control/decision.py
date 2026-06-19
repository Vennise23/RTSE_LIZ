"""
Decision policy (OVERRIDE + COST HYBRID)

Priority order:
1. LOW LIGHT (HARD OVERRIDE)
2. POLICE EMERGENCY (HARD OVERRIDE)
3. CHASE PRESSURE (HIGH PRIORITY)
4. NORMAL COST-BASED DRIVING
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from . import config
from .state import Command, CommandKind, GameState, TokenColor


# ----------------------------------------------------------------------
# COST WEIGHTS (only used in NORMAL mode)
# ----------------------------------------------------------------------
LANE_CHANGE_COST = 0.8
STABILITY_LANE_COST = 0.3
CENTER_LANE_BONUS = 0.1
INVALID_ACTION_COST = 1e9


def _side_has_any_bad_color(state: GameState, own_x: float, lookahead: float, side: str) -> bool:
    """Reject lanes that contain yellow/red unless they are clearly the only option."""
    for tok in state.tokens:
        if tok.distance <= 0.0 or tok.distance > lookahead:
            continue
        dx = _token_x_pos(tok) - own_x
        if side == "left" and dx >= 0.0:
            continue
        if side == "right" and dx <= 0.0:
            continue
        if tok.color in (TokenColor.YELLOW, TokenColor.RED):
            return True
    return False


def _green_target_score(state: GameState, own_x: float, lookahead: float, side: str) -> float:
    """Score only clean green opportunities on one side."""
    if _side_has_any_bad_color(state, own_x, lookahead, side):
        return 0.0
    return _side_green_score(state, own_x, lookahead, side)


def _side_is_clean_green_target(state: GameState, own_x: float, lookahead: float, brake_dist: float, side: str) -> bool:
    """Hard gate: only allow a lane change if the side is clean and green."""
    if _side_has_any_bad_color(state, own_x, lookahead, side):
        return False
    if _side_has_near_red(state, own_x, brake_dist, side):
        return False
    return _side_green_score(state, own_x, lookahead, side) > 0.0


# ----------------------------------------------------------------------
# MEMORY
# ----------------------------------------------------------------------
@dataclass
class DecisionMemory:
    last_switch_time: float = -1e9
    last_command_kind: CommandKind = CommandKind.HOLD
    low_light_count: int = 0


@dataclass
class DecisionResult:
    command: Command
    memory: DecisionMemory


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def _target_lane(action: CommandKind, lane: int) -> int:
    if action == CommandKind.MOVE_LEFT:
        return max(0, lane - 1)
    if action == CommandKind.MOVE_RIGHT:
        return min(config.NUM_LANES - 1, lane + 1)
    return lane


def _token_proximity(d: float, lookahead: float) -> float:
    return max(0.0, 1.0 - d / lookahead)


def _effective_lookahead(speed_norm: float) -> float:
    return (
        config.LOOKAHEAD_BASE
        + config.LOOKAHEAD_SPEED_GAIN * max(0.0, min(1.0, speed_norm))
    )


# ----------------------------------------------------------------------
# HARD SAFETY: POLICE
# ----------------------------------------------------------------------
def _imminent_police_collision(state: GameState, lane: int) -> bool:
    if not getattr(state, "police_alert", False):
        return False

    pl = int(getattr(state, "police_lane", -1))
    if pl < 0:
        return False

    t = float(getattr(state, "police_time_left", 999))
    dist = abs(lane - pl)

    return (t <= 2.5 and dist <= 2) or (t <= 1.5 and dist <= 3)


# ----------------------------------------------------------------------
# COST MODEL (NORMAL ONLY)
# ----------------------------------------------------------------------
def _color_cost(lane, red_min, green, yellow, lookahead):
    cost = 0.0

    if red_min[lane] <= lookahead:
        cost += config.RED_PENALTY * _token_proximity(red_min[lane], lookahead)

    for d in green[lane]:
        if d <= lookahead:
            cost -= config.GREEN_REWARD * _token_proximity(d, lookahead)

    for d in yellow[lane]:
        if d <= lookahead:
            cost += config.YELLOW_PENALTY * 2.0 * _token_proximity(d, lookahead)

    return cost


def _stability_cost(target, current):
    center = (config.NUM_LANES - 1) / 2
    return (
        abs(target - current) * STABILITY_LANE_COST
        + abs(target - center) * CENTER_LANE_BONUS
    )


def _obstacle_cost(lane, obs, lookahead):
    d = obs[lane]
    if d <= 0:
        return 0.0
    if d <= lookahead:
        return config.RED_PENALTY * (1.0 - d / lookahead)
    return 0.0


def _action_cost(action, own, red, green, yellow, obs, lookahead, state):
    target = _target_lane(action, own)

    if _imminent_police_collision(state, target):
        return INVALID_ACTION_COST

    cost = 0.0

    if action in (CommandKind.MOVE_LEFT, CommandKind.MOVE_RIGHT):
        cost += LANE_CHANGE_COST

    cost += _color_cost(target, red, green, yellow, lookahead)
    cost += _obstacle_cost(target, obs, lookahead)
    cost += _stability_cost(target, own)

    return cost


# ----------------------------------------------------------------------
# MAIN DECISION FUNCTION
# ----------------------------------------------------------------------
def decide(
    state: GameState,
    memory: DecisionMemory,
    now: Optional[float] = None,
) -> DecisionResult:

    now = now or time.perf_counter()

    own = state.own_lane
    if own < 0:
        return DecisionResult(Command(CommandKind.HOLD, now, "invalid_lane"), memory)

    if getattr(state, "game_over", False):
        return DecisionResult(Command(CommandKind.HOLD, now, "game_over"), memory)

    # ============================================================
    # 1. LOW LIGHT — HARD OVERRIDE (FIXED)
    # ============================================================
    low_light = (
        getattr(state, "low_light_active", False)
        or state.brightness == config.LOW_LIGHT_THRESHOLD
    )

    if low_light:
        memory.low_light_count += 1

        # FORCE ACTION: IGNORE ALL COSTS
        return DecisionResult(
            Command(
                CommandKind.RECOVER_LIGHT,
                now,
                "LOW_LIGHT_OVERRIDE_ACCEL_-1.0"
            ),
            memory,
        )

    # ============================================================
    # 2. POLICE — HARD OVERRIDE
    # ============================================================
    if getattr(state, "police_alert", False):
        pl = getattr(state, "police_lane", -1)

        # force move away from police lane if possible
        if pl >= 0:
            if own <= pl:
                action = CommandKind.MOVE_LEFT
            else:
                action = CommandKind.MOVE_RIGHT

            return DecisionResult(
                Command(action, now, "POLICE_ESCAPE_OVERRIDE"),
                memory,
            )

    # ============================================================
    # 3. CHASE PRESSURE — HIGH PRIORITY OVERRIDE
    # ============================================================
    if getattr(state, "rear_chase_active", False):
        pressure = getattr(state, "rear_pressure", 0.0)

        if pressure > 0.6:
            # escape strategy: change lane aggressively
            if own < config.NUM_LANES - 1:
                action = CommandKind.MOVE_RIGHT
            else:
                action = CommandKind.MOVE_LEFT

            return DecisionResult(
                Command(action, now, "CHASE_ESCAPE"),
                memory,
            )

    # ============================================================
    # 4. NORMAL COST POLICY
    # ============================================================
    lookahead = _effective_lookahead(state.speed_norm)

    red, green, yellow, obs = _lane_metrics(state)

    actions = [
        CommandKind.HOLD,
        CommandKind.MOVE_LEFT,
        CommandKind.MOVE_RIGHT,
        CommandKind.SLOW_DOWN,
    ]

    best_action = CommandKind.HOLD
    best_cost = float("inf")

    for a in actions:
        target = _target_lane(a, own)

        cost = _action_cost(
            a, own, red, green, yellow, obs, lookahead, state
        )

        if cost < best_cost:
            best_cost = cost
            best_action = a

    return DecisionResult(
        Command(best_action, now, "COST_POLICY_V3"),
        memory,
    )


# (reuse your existing lane metrics function if already defined elsewhere)
def _lane_metrics(state: GameState):
    n = config.NUM_LANES
    red_min = [1.2] * n
    green = [[] for _ in range(n)]
    yellow = [[] for _ in range(n)]
    obstacle_min = [1.2] * n

    for t in state.tokens:
        if 0 <= t.lane < n:
            if t.color == TokenColor.RED:
                red_min[t.lane] = min(red_min[t.lane], t.distance)
            elif t.color == TokenColor.GREEN:
                green[t.lane].append(t.distance)
            elif t.color == TokenColor.YELLOW:
                yellow[t.lane].append(t.distance)

    for o in state.obstacles:
        if 0 <= o.lane < n:
            obstacle_min[o.lane] = min(obstacle_min[o.lane], o.distance)

    return red_min, green, yellow, obstacle_min


__all__ = ["decide", "DecisionMemory", "DecisionResult"]