"""Color-first decision policy.

The function ``decide`` takes a perception snapshot (plus a small slice
of decision-task memory) and returns a ``Command``. It is deliberately
pure: no globals, no I/O, no random state. That is what makes it both
unit-testable and time-predictable for response-time analysis.

The policy is intentionally simple:

    1. If a red token is directly ahead, move toward the safer side.
    2. Otherwise prefer the side with visible green tokens.
    3. Avoid yellow if we can, and never choose a side with a near red.
    4. If nothing looks better, hold lane and cruise.

This version does not try to reason in terms of lane 0/1/2 selection.
It looks at the token colors and their horizontal placement relative to
the car, which matches the "human-like" rule the user described.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from . import config
from .state import Command, CommandKind, GameState, Token, TokenColor, Obstacle


# ----------------------------------------------------------------------
# Tunables for the color-first policy
# ----------------------------------------------------------------------
CENTER_BAND = 0.11
SIDE_CLEAR_BAND = 0.04
GREEN_GAIN = 1.0
YELLOW_COST = 0.45
RED_COST = 3.0


def _effective_lookahead(speed_norm: float) -> float:
    return config.LOOKAHEAD_BASE + config.LOOKAHEAD_SPEED_GAIN * max(0.0, min(1.0, speed_norm))


def _own_x_pos(state: GameState) -> float:
    """Normalize the car position to the same 0..1 scale as token x positions."""
    if hasattr(state, "own_lane_pos"):
        raw = float(getattr(state, "own_lane_pos"))
    else:
        raw = float(state.own_lane)
    denom = max(1, config.NUM_LANES - 1)
    if raw > 1.0:
        raw = raw / denom
    return max(0.0, min(1.0, raw))


def _token_x_pos(tok: Token) -> float:
    return max(0.0, min(1.0, float(getattr(tok, "x_pos", 0.5))))


def _token_proximity(distance: float, lookahead: float) -> float:
    return max(0.0, 1.0 - distance / lookahead)


def _side_score(state: GameState, own_x: float, lookahead: float, side: str) -> float:
    """Score colors on one side of the car using horizontal placement."""
    score = 0.0
    for tok in state.tokens:
        if tok.distance <= 0.0 or tok.distance > lookahead:
            continue
        dx = _token_x_pos(tok) - own_x
        if side == "left" and dx >= 0.0:
            continue
        if side == "right" and dx <= 0.0:
            continue
        proximity = _token_proximity(tok.distance, lookahead)
        lateral = max(0.0, 1.0 - min(1.0, abs(dx) / 0.5))

        if tok.color is TokenColor.GREEN:
            score += GREEN_GAIN * proximity * (0.7 + 0.3 * lateral)
        elif tok.color is TokenColor.YELLOW:
            score -= YELLOW_COST * proximity * (0.8 + 0.2 * lateral)
        elif tok.color is TokenColor.RED:
            score -= RED_COST * proximity * (1.0 + 0.4 * lateral)

    for obs in state.obstacles:
        if obs.distance <= 0.0 or obs.distance > lookahead:
            continue
        dx = float(getattr(obs, "x_pos", own_x)) - own_x
        if side == "left" and dx >= 0.0:
            continue
        if side == "right" and dx <= 0.0:
            continue
        proximity = _token_proximity(obs.distance, lookahead)
        score -= RED_COST * proximity
    return score


def _side_green_score(state: GameState, own_x: float, lookahead: float, side: str) -> float:
    """Positive-only version used to trigger more aggressive lane changes."""
    score = 0.0
    for tok in state.tokens:
        if tok.color is not TokenColor.GREEN:
            continue
        if tok.distance <= 0.0 or tok.distance > lookahead:
            continue
        dx = _token_x_pos(tok) - own_x
        if side == "left" and dx >= 0.0:
            continue
        if side == "right" and dx <= 0.0:
            continue
        proximity = _token_proximity(tok.distance, lookahead)
        lateral = max(0.0, 1.0 - min(1.0, abs(dx) / 0.5))
        score += GREEN_GAIN * proximity * (0.7 + 0.3 * lateral)
    return score


def _front_red_ahead(state: GameState, own_x: float, brake_dist: float) -> bool:
    for tok in state.tokens:
        if tok.color is not TokenColor.RED:
            continue
        if tok.distance <= 0.0 or tok.distance > brake_dist:
            continue
        if abs(_token_x_pos(tok) - own_x) <= CENTER_BAND:
            return True
    for obs in state.obstacles:
        if obs.distance <= 0.0 or obs.distance > brake_dist:
            continue
        if abs(float(getattr(obs, "x_pos", own_x)) - own_x) <= CENTER_BAND:
            return True
    return False


def _side_has_near_red(state: GameState, own_x: float, brake_dist: float, side: str) -> bool:
    for tok in state.tokens:
        if tok.color is not TokenColor.RED:
            continue
        if tok.distance <= 0.0 or tok.distance > brake_dist:
            continue
        dx = _token_x_pos(tok) - own_x
        if side == "left" and dx < -SIDE_CLEAR_BAND:
            return True
        if side == "right" and dx > SIDE_CLEAR_BAND:
            return True
    for obs in state.obstacles:
        if obs.distance <= 0.0 or obs.distance > brake_dist:
            continue
        dx = float(getattr(obs, "x_pos", own_x)) - own_x
        if side == "left" and dx < -SIDE_CLEAR_BAND:
            return True
        if side == "right" and dx > SIDE_CLEAR_BAND:
            return True
    return False


# ----------------------------------------------------------------------
# Public decision entry point
# ----------------------------------------------------------------------
@dataclass
class DecisionMemory:
    last_switch_time: float = -1e9
    last_command_kind: CommandKind = CommandKind.HOLD
    low_light_count: int = 0
    low_light_active: bool = False
    low_light_recovery_sent: bool = False


@dataclass
class DecisionResult:
    command: Command
    memory: DecisionMemory


def decide(
    state: GameState,
    memory: DecisionMemory,
    now: Optional[float] = None,
) -> DecisionResult:
    now = now if now is not None else time.perf_counter()
    own = state.own_lane
    if own < 0:
        cmd = Command(kind=CommandKind.HOLD, issued_at=now, reason="invalid_own_lane")
        return DecisionResult(cmd, memory)

    own_x = _own_x_pos(state)
    lookahead = _effective_lookahead(state.speed_norm)
    brake_dist = config.BRAKE_DIST
    low_light_count = (
        memory.low_light_count + 1
        if state.brightness < config.LOW_LIGHT_THRESHOLD
        else 0
    )
    low_light_active = low_light_count >= config.LOW_LIGHT_CONFIRM_FRAMES

    if low_light_active:
        memory = DecisionMemory(
            last_switch_time=memory.last_switch_time,
            last_command_kind=CommandKind.RECOVER_LIGHT,
            low_light_count=low_light_count,
            low_light_active=low_light_active,
            low_light_recovery_sent=True,
        )
        cmd = Command(
            kind=CommandKind.RECOVER_LIGHT,
            issued_at=now,
            reason="low_light_recovery",
        )
        return DecisionResult(cmd, memory)

    left_score = _side_score(state, own_x, lookahead, "left")
    right_score = _side_score(state, own_x, lookahead, "right")
    center_score = _side_score(state, own_x, lookahead, "center")
    left_green = _side_green_score(state, own_x, lookahead, "left")
    right_green = _side_green_score(state, own_x, lookahead, "right")

    # 1. Safety: if a red is straight ahead, move to the safer side.
    if _front_red_ahead(state, own_x, brake_dist):
        candidates = []
        if not _side_has_near_red(state, own_x, brake_dist, "left"):
            candidates.append(("left", left_score, CommandKind.MOVE_LEFT))
        if not _side_has_near_red(state, own_x, brake_dist, "right"):
            candidates.append(("right", right_score, CommandKind.MOVE_RIGHT))

        if candidates:
            candidates.sort(key=lambda item: item[1], reverse=True)
            side, score, kind = candidates[0]
            if score > -RED_COST:
                memory = DecisionMemory(
                    last_switch_time=now,
                    last_command_kind=kind,
                    low_light_count=low_light_count,
                    low_light_active=low_light_active,
                )
                cmd = Command(
                    kind=kind,
                    issued_at=now,
                    reason=f"avoid_red_to_{side}_score={score:.2f}",
                )
                return DecisionResult(cmd, memory)

        memory = DecisionMemory(
            last_switch_time=memory.last_switch_time,
            last_command_kind=CommandKind.SLOW_DOWN,
            low_light_count=low_light_count,
            low_light_active=low_light_active,
        )
        cmd = Command(
            kind=CommandKind.SLOW_DOWN,
            issued_at=now,
            reason="front_red_no_clear_side",
        )
        return DecisionResult(cmd, memory)

    # 2. Reward: if one side has a clear green advantage, go there.
    best_side = "left" if left_green >= right_green else "right"
    best_green = max(left_green, right_green)
    side_score = left_score if best_side == "left" else right_score

    if best_green > 0.0 and side_score > -RED_COST:
        kind = CommandKind.MOVE_LEFT if best_side == "left" else CommandKind.MOVE_RIGHT
        memory = DecisionMemory(
            last_switch_time=now,
            last_command_kind=kind,
            low_light_count=low_light_count,
            low_light_active=low_light_active,
        )
        cmd = Command(
            kind=kind,
            issued_at=now,
            reason=f"chase_green_to_{best_side}_green={best_green:.2f}",
        )
        return DecisionResult(cmd, memory)

    # 3. Stability: keep cruising unless the center is saturated with bad colors.
    # Low-light is only observed/logged for now; it does not change actuation.
    if center_score < -config.GREEN_REWARD:
        memory = DecisionMemory(
            last_switch_time=memory.last_switch_time,
            last_command_kind=CommandKind.SLOW_DOWN,
            low_light_count=low_light_count,
            low_light_active=low_light_active,
        )
        cmd = Command(
            kind=CommandKind.SLOW_DOWN,
            issued_at=now,
            reason="center_heavy_bad_colors",
        )
        return DecisionResult(cmd, memory)

    memory = DecisionMemory(
        last_switch_time=memory.last_switch_time,
        last_command_kind=CommandKind.HOLD,
        low_light_count=low_light_count,
        low_light_active=low_light_active,
    )
    cmd = Command(
        kind=CommandKind.HOLD,
        issued_at=now,
        reason="cruise",
    )
    return DecisionResult(cmd, memory)


__all__ = ["decide", "DecisionMemory", "DecisionResult"]
