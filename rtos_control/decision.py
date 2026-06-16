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
YELLOW_COST = 0.85
RED_COST = 3.0
# Favor green collection aggressively, but avoid lane-flapping.
GREEN_CHASE_MIN = 0.10
GREEN_SWITCH_COOLDOWN_SEC = 0.25


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
    red_seen = False
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
            if red_seen:
                continue
            score += GREEN_GAIN * proximity * (0.7 + 0.3 * lateral)
        elif tok.color is TokenColor.YELLOW:
            score -= YELLOW_COST * proximity * (0.9 + 0.3 * lateral)
        elif tok.color is TokenColor.RED:
            red_seen = True
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
    red_block = False
    for tok in state.tokens:
        if tok.color is not TokenColor.GREEN:
            if tok.color is TokenColor.RED:
                red_block = True
            continue
        if tok.distance <= 0.0 or tok.distance > lookahead:
            continue
        dx = _token_x_pos(tok) - own_x
        if side == "left" and dx >= 0.0:
            continue
        if side == "right" and dx <= 0.0:
            continue
        if red_block:
            continue
        proximity = _token_proximity(tok.distance, lookahead)
        lateral = max(0.0, 1.0 - min(1.0, abs(dx) / 0.5))
        score += GREEN_GAIN * proximity * (0.8 + 0.2 * lateral)
    return score


def _front_red_ahead(state: GameState, own_x: float, brake_dist: float) -> bool:
    for tok in state.tokens:
        if tok.color is not TokenColor.RED:
            continue
        if tok.distance <= 0.0 or tok.distance > brake_dist:
            continue
        # Lane-based check is more stable than pure x-position matching.
        if int(getattr(tok, "lane", -1)) == int(getattr(state, "own_lane", -1)):
            return True
    for obs in state.obstacles:
        if obs.distance <= 0.0 or obs.distance > brake_dist:
            continue
        if int(getattr(obs, "lane", -1)) == int(getattr(state, "own_lane", -1)):
            return True
    return False


def _front_red_best_side(state: GameState, own_x: float, brake_dist: float) -> Optional[str]:
    """Pick the safer side when a red is directly ahead."""
    left_blocked = _side_has_near_red(state, own_x, brake_dist, "left")
    right_blocked = _side_has_near_red(state, own_x, brake_dist, "right")
    if left_blocked and right_blocked:
        return None
    if not left_blocked and right_blocked:
        return "left"
    if left_blocked and not right_blocked:
        return "right"

    left_score = _side_score(state, own_x, brake_dist, "left")
    right_score = _side_score(state, own_x, brake_dist, "right")
    return "left" if left_score >= right_score else "right"


def _front_red_debug(state: GameState, own_x: float, brake_dist: float) -> str:
    """Compact debug string for front-red decisions."""
    left_blocked = _side_has_near_red(state, own_x, brake_dist, "left")
    right_blocked = _side_has_near_red(state, own_x, brake_dist, "right")
    red_tokens = [
        tok for tok in state.tokens
        if tok.color is TokenColor.RED
        and 0.0 < tok.distance <= brake_dist
        and int(getattr(tok, "lane", -1)) == int(getattr(state, "own_lane", -1))
    ]
    red_count = len(red_tokens)
    sample = "none"
    if red_tokens:
        tok = red_tokens[0]
        sample = f"d={tok.distance:.2f} x={_token_x_pos(tok):.2f}"
    return (
        f"own_x={own_x:.2f} red_count={red_count} red={sample} "
        f"left_blocked={int(left_blocked)} right_blocked={int(right_blocked)}"
    )


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
# Public decision entry point
# ----------------------------------------------------------------------
@dataclass
class DecisionMemory:
    last_switch_time: float = -1e9
    last_command_kind: CommandKind = CommandKind.HOLD
    low_light_count: int = 0
    low_light_active: bool = False
    low_light_recovery_sent: bool = False
    police_alert_seen: bool = False
    game_over_seen: bool = False


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
    left_score = _side_score(state, own_x, lookahead, "left")
    right_score = _side_score(state, own_x, lookahead, "right")
    center_score = _side_score(state, own_x, lookahead, "center")
    left_green = _green_target_score(state, own_x, lookahead, "left")
    right_green = _green_target_score(state, own_x, lookahead, "right")
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

    if getattr(state, "game_over", False):
        memory = DecisionMemory(
            last_switch_time=memory.last_switch_time,
            last_command_kind=CommandKind.HOLD,
            low_light_count=low_light_count,
            low_light_active=low_light_active,
            low_light_recovery_sent=memory.low_light_recovery_sent,
            police_alert_seen=getattr(state, "police_alert", False) or memory.police_alert_seen,
            game_over_seen=True,
        )
        cmd = Command(
            kind=CommandKind.HOLD,
            issued_at=now,
            reason=f"game_over:{getattr(state, 'game_over_reason', 'unknown')}",
        )
        return DecisionResult(cmd, memory)

    # 0. Hard safety: if red is directly ahead, force an immediate lane change
    # to the safer side before any green-chasing logic can run.
    if _front_red_ahead(state, own_x, brake_dist):
        best_side = _front_red_best_side(state, own_x, brake_dist)
        debug = _front_red_debug(state, own_x, brake_dist)
        if best_side is not None:
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
                reason=f"hard_avoid_front_red_to_{best_side};{debug}",
            )
            return DecisionResult(cmd, memory)

        cmd = Command(
            kind=CommandKind.SLOW_DOWN,
            issued_at=now,
            reason=f"front_red_no_clear_side;{_front_red_debug(state, own_x, brake_dist)}",
        )
        return DecisionResult(cmd, memory)

    if getattr(state, "police_alert", False):
        chase_lane = int(getattr(state, "rear_chase_lane", -1))
        time_left = float(getattr(state, "rear_time_left", 0.0))
        chase_reason = f"rear_chase_lane={chase_lane} time_left={time_left:.1f}"
        memory = DecisionMemory(
            last_switch_time=memory.last_switch_time,
            last_command_kind=memory.last_command_kind,
            low_light_count=low_light_count,
            low_light_active=low_light_active,
            low_light_recovery_sent=memory.low_light_recovery_sent,
            police_alert_seen=True,
            game_over_seen=False,
        )

        left_safe = not _side_has_near_red(state, own_x, brake_dist, "left")
        right_safe = not _side_has_near_red(state, own_x, brake_dist, "right")
        if left_safe or right_safe:
            if left_safe and right_safe:
                best_side = "left" if left_score >= right_score else "right"
            else:
                best_side = "left" if left_safe else "right"
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
                reason=f"avoid_chasing_car_to_{best_side};{chase_reason}",
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
            reason=f"chasing_car_no_clear_side;{chase_reason}",
        )
        return DecisionResult(cmd, memory)

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

    if getattr(state, "low_light_active", False):
        memory = DecisionMemory(
            last_switch_time=memory.last_switch_time,
            last_command_kind=CommandKind.RECOVER_LIGHT,
            low_light_count=low_light_count,
            low_light_active=low_light_active,
            low_light_recovery_sent=True,
            police_alert_seen=getattr(state, "police_alert", False),
            game_over_seen=memory.game_over_seen,
        )
        cmd = Command(
            kind=CommandKind.RECOVER_LIGHT,
            issued_at=now,
            reason="low_light_recovery",
        )
        return DecisionResult(cmd, memory)

    # 2. Reward: if one side has a green target and it is safe, go there.
    best_side = "left" if left_green >= right_green else "right"
    best_green = max(left_green, right_green)
    other_green = min(left_green, right_green)
    side_score = left_score if best_side == "left" else right_score
    green_edge = best_green - other_green
    best_side_safe = _side_is_clean_green_target(state, own_x, lookahead, brake_dist, best_side)
    green_recently_switched = (now - memory.last_switch_time) < GREEN_SWITCH_COOLDOWN_SEC

    if best_green > 0.0 and best_side_safe:
        if green_edge >= config.SWITCH_MARGIN or best_green >= GREEN_CHASE_MIN or not green_recently_switched:
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

    # If both sides look dirty, do not force a green chase.
    if _side_has_any_bad_color(state, own_x, lookahead, best_side):
        best_side_safe = False

    # 3. Stability: keep cruising unless the center is saturated with bad colors.
    # Low-light is only observed/logged for now; it does not change actuation.
    if center_score < -(config.GREEN_REWARD * 1.75):
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
