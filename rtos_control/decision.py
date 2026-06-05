"""Rule-based decision policy.

The function ``decide`` takes a perception snapshot (plus a small slice
of decision-task memory) and returns a ``Command``. It is deliberately
pure: no globals, no I/O, no random state. That is what makes it both
unit-testable and time-predictable for response-time analysis.

Priority order is fixed and never tunable at runtime:

    1. SAFETY  — avoid imminent red tokens / obstacles in our lane.
    2. REWARD  — prefer adjacent lanes with significantly more greens.
    3. STABILITY — otherwise hold the current lane (and respect the
       lateral cool-down so we don't oscillate).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional

from . import config
from .state import Command, CommandKind, GameState, Token, TokenColor, Obstacle


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
def _effective_lookahead(speed_norm: float) -> float:
    """Look further ahead at higher speeds: smaller reaction window."""
    return config.LOOKAHEAD_BASE + config.LOOKAHEAD_SPEED_GAIN * max(0.0, min(1.0, speed_norm))


def _lane_is_valid(lane: int) -> bool:
    return 0 <= lane < config.NUM_LANES


def _hazards_in_lane(state: GameState, lane: int, brake_dist: float) -> bool:
    """True if there's an imminent RED token or any obstacle in ``lane``.

    Yellow is intentionally NOT a SAFETY hazard: triggering the emergency
    lane-change branch on yellow would make us flee lanes that also
    contain greens, which is a worse outcome than just taking the yellow
    debuff. Yellow is still discouraged via the lane-reward penalty in
    ``_lane_reward`` (see ``YELLOW_PENALTY`` in config).
    """
    for tok in state.tokens:
        if tok.lane != lane:
            continue
        if tok.color is TokenColor.RED and 0.0 < tok.distance <= brake_dist:
            return True
    for obs in state.obstacles:
        if obs.lane == lane and 0.0 < obs.distance <= brake_dist:
            return True
    return False


def _lane_reward(state: GameState, lane: int, lookahead: float) -> float:
    """Net reward of a lane within the look-ahead window.

    Positive contributions: greens (closer ones weighted more).
    Negative contributions: reds, obstacles, yellows (all amplified).
    """
    reward = 0.0
    for tok in state.tokens:
        if tok.lane != lane:
            continue
        if tok.distance <= 0.0 or tok.distance > lookahead:
            continue
        proximity = max(0.0, 1.0 - tok.distance / lookahead)
        proximity = proximity ** config.REWARD_DECAY_NEAR_BIAS
        if tok.color is TokenColor.GREEN:
            reward += config.GREEN_REWARD * proximity
        elif tok.color is TokenColor.RED:
            reward -= config.RED_PENALTY * proximity
        elif tok.color is TokenColor.YELLOW:
            # Yellow = unpredictable debuff (random in Phase 1); weight
            # equal to red so any lane with a yellow is strictly worse
            # than an empty one and we never pick it over an alternative.
            reward -= config.YELLOW_PENALTY * proximity
    for obs in state.obstacles:
        if obs.lane != lane:
            continue
        if obs.distance <= 0.0 or obs.distance > lookahead:
            continue
        proximity = max(0.0, 1.0 - obs.distance / lookahead)
        reward -= config.RED_PENALTY * proximity
    return reward


# ----------------------------------------------------------------------
# Public decision entry point
# ----------------------------------------------------------------------
@dataclass
class DecisionMemory:
    """Cross-cycle memory owned by the Decision task.

    The decision function itself stays pure: it receives this and
    returns the updated copy alongside the command.
    """
    last_switch_time: float = -1e9   # perf_counter() of last lane change
    last_command_kind: CommandKind = CommandKind.HOLD


@dataclass
class DecisionResult:
    command: Command
    memory: DecisionMemory


def decide(
    state: GameState,
    memory: DecisionMemory,
    now: Optional[float] = None,
) -> DecisionResult:
    """Pure decision function. See module docstring for the policy."""
    now = now if now is not None else time.perf_counter()
    own = state.own_lane
    if not _lane_is_valid(own):
        # Defensive fallback: aim for center.
        cmd = Command(kind=CommandKind.HOLD, issued_at=now,
                      reason="invalid_own_lane")
        return DecisionResult(cmd, memory)

    lookahead = _effective_lookahead(state.speed_norm)
    brake_dist = config.BRAKE_DIST

    # ---- 1. SAFETY -------------------------------------------------
    if _hazards_in_lane(state, own, brake_dist):
        # Try to escape sideways. Prefer the safer of the two adjacent
        # lanes; if both are unsafe, slow down and stay.
        left  = own - 1 if _lane_is_valid(own - 1) else None
        right = own + 1 if _lane_is_valid(own + 1) else None
        candidates = []
        if left is not None and not _hazards_in_lane(state, left, brake_dist):
            candidates.append((left, CommandKind.MOVE_LEFT))
        if right is not None and not _hazards_in_lane(state, right, brake_dist):
            candidates.append((right, CommandKind.MOVE_RIGHT))
        if candidates:
            # Among safe escapes pick the higher-reward one.
            candidates.sort(
                key=lambda c: _lane_reward(state, c[0], lookahead), reverse=True,
            )
            target_lane, kind = candidates[0]
            memory = DecisionMemory(
                last_switch_time=now,
                last_command_kind=kind,
            )
            cmd = Command(
                kind=kind,
                issued_at=now,
                reason=f"avoid_red_to_lane_{target_lane}",
            )
            return DecisionResult(cmd, memory)

        # Boxed in: every reachable lane has a red inside brake_dist.
        # Pick the lane whose CLOSEST red is the FURTHEST away — that
        # buys us the most time before the next collision, and may let
        # the lane go clear before we hit it. Also brake hard.
        def _closest_red_dist(lane: int) -> float:
            best = float("inf")
            for tok in state.tokens:
                if tok.lane != lane or tok.color is not TokenColor.RED:
                    continue
                if 0.0 < tok.distance < best:
                    best = tok.distance
            for obs in state.obstacles:
                if obs.lane != lane:
                    continue
                if 0.0 < obs.distance < best:
                    best = obs.distance
            return best

        options = [own]
        if left is not None:
            options.append(left)
        if right is not None:
            options.append(right)
        options.sort(key=_closest_red_dist, reverse=True)  # furthest first
        best_lane = options[0]
        if best_lane == own:
            chosen_kind = CommandKind.SLOW_DOWN
            reason = "boxed_in_stay_and_brake"
            switch_time = memory.last_switch_time
        else:
            chosen_kind = (CommandKind.MOVE_LEFT if best_lane < own
                           else CommandKind.MOVE_RIGHT)
            reason = (f"boxed_in_dive_to_lane_{best_lane}_"
                      f"(red@{_closest_red_dist(best_lane):.2f})")
            switch_time = now
        memory = DecisionMemory(
            last_switch_time=switch_time,
            last_command_kind=chosen_kind,
        )
        cmd = Command(
            kind=chosen_kind,
            issued_at=now,
            reason=reason,
        )
        return DecisionResult(cmd, memory)

    # ---- 2. REWARD -------------------------------------------------
    # Respect lateral cool-down to suppress oscillation around equal-reward lanes.
    in_cooldown = (now - memory.last_switch_time) < config.SWITCH_COOLDOWN
    own_reward = _lane_reward(state, own, lookahead)
    best_lane = own
    best_reward = own_reward
    best_kind: Optional[CommandKind] = None
    for cand_lane, kind in (
        (own - 1, CommandKind.MOVE_LEFT),
        (own + 1, CommandKind.MOVE_RIGHT),
    ):
        if not _lane_is_valid(cand_lane):
            continue
        # Only consider a lane change if the adjacent lane is also safe.
        if _hazards_in_lane(state, cand_lane, brake_dist):
            continue
        cand_reward = _lane_reward(state, cand_lane, lookahead)
        if cand_reward > best_reward:
            best_reward = cand_reward
            best_lane = cand_lane
            best_kind = kind

    if (
        not in_cooldown
        and best_kind is not None
        and (best_reward - own_reward) >= config.SWITCH_MARGIN
    ):
        memory = DecisionMemory(
            last_switch_time=now,
            last_command_kind=best_kind,
        )
        cmd = Command(
            kind=best_kind,
            issued_at=now,
            reason=f"chase_green_to_lane_{best_lane}_dR={best_reward - own_reward:.2f}",
        )
        return DecisionResult(cmd, memory)

    # ---- 3. STABILITY ---------------------------------------------
    # Hold lane; cruise at full throttle unless reward is markedly
    # negative in our own lane (means there are far-but-real reds ahead
    # that may close in by next cycle).
    if own_reward < -config.GREEN_REWARD:
        memory = DecisionMemory(
            last_switch_time=memory.last_switch_time,
            last_command_kind=CommandKind.SLOW_DOWN,
        )
        cmd = Command(
            kind=CommandKind.SLOW_DOWN,
            issued_at=now,
            reason="far_reds_ahead",
        )
        return DecisionResult(cmd, memory)

    memory = DecisionMemory(
        last_switch_time=memory.last_switch_time,
        last_command_kind=CommandKind.HOLD,
    )
    cmd = Command(
        kind=CommandKind.HOLD,
        issued_at=now,
        reason="cruise" if not in_cooldown else "cooldown",
    )
    return DecisionResult(cmd, memory)


__all__ = ["decide", "DecisionMemory", "DecisionResult"]
