"""
Decision policy (OVERRIDE + COST HYBRID)

Priority order:
1. LOW LIGHT (HARD OVERRIDE)
2. POLICE EMERGENCY (HARD OVERRIDE)
3. GOLDEN LANE (HIGH PRIORITY)
4. CHASE PRESSURE (HIGH PRIORITY)
5. NORMAL COST-BASED DRIVING
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
LANE_CHANGE_COST = config.COST_LANE_CHANGE
STABILITY_LANE_COST = config.COST_STABILITY_LANE
CENTER_LANE_BONUS = config.COST_CENTER_LANE
INVALID_ACTION_COST = 1e9
ACTIONS = (
    CommandKind.HOLD,
    CommandKind.MOVE_LEFT,
    CommandKind.MOVE_RIGHT,
    CommandKind.SLOW_DOWN,
)


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
    police_event_active: bool = False
    police_event_started_at: float = -1e9
    police_red_baseline: int = 0
    police_red_done: bool = False
    post_police_avoid_red_until: float = -1e9
    last_golden_lane: int = -1


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


def _normalize_lane_index(lane_like) -> int:
    """Best-effort lane normalization from noisy perception/OCR fields."""
    try:
        lane = int(round(float(lane_like)))
    except (TypeError, ValueError):
        return -1
    if 0 <= lane < config.NUM_LANES:
        return lane
    return -1


def _token_x_pos(tok) -> float:
    """Return normalized token x position; default to center when missing."""
    return float(getattr(tok, "x_pos", 0.5))


def _side_green_score(state: GameState, own_x: float, lookahead: float, side: str) -> float:
    """Sum proximity-weighted green reward on one side of the car."""
    score = 0.0
    for tok in state.tokens:
        if tok.color != TokenColor.GREEN:
            continue
        if tok.distance <= 0.0 or tok.distance > lookahead:
            continue
        dx = _token_x_pos(tok) - own_x
        if side == "left" and dx >= 0.0:
            continue
        if side == "right" and dx <= 0.0:
            continue
        score += _token_proximity(tok.distance, lookahead)
    return score


def _side_has_near_red(state: GameState, own_x: float, brake_dist: float, side: str) -> bool:
    """Detect an imminent red token on one side of the car."""
    for tok in state.tokens:
        if tok.color != TokenColor.RED:
            continue
        if tok.distance <= 0.0 or tok.distance > brake_dist:
            continue
        dx = _token_x_pos(tok) - own_x
        if side == "left" and dx >= 0.0:
            continue
        if side == "right" and dx <= 0.0:
            continue
        return True
    return False


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


def _best_police_red_action(state: GameState, own: int) -> CommandKind:
    """During police event, steer toward the safest lane that has a near red token.

    Constraint: never choose the police lane if avoidable, because entering it
    can end the run immediately.
    """
    police_lane = int(getattr(state, "police_lane", -1))
    n = config.NUM_LANES

    # Immediate escape if we're currently aligned with police lane.
    if 0 <= police_lane < n and own == police_lane:
        left_lane = max(0, own - 1)
        right_lane = min(n - 1, own + 1)
        red_min, _, _, obs_min = _lane_metrics(state)

        # Score both possible exits and choose safer one.
        left_score = red_min[left_lane] + obs_min[left_lane] + 0.05
        right_score = red_min[right_lane] + obs_min[right_lane] + 0.05

        if own == 0:
            return CommandKind.MOVE_RIGHT
        if own == n - 1:
            return CommandKind.MOVE_LEFT
        return CommandKind.MOVE_LEFT if left_score <= right_score else CommandKind.MOVE_RIGHT

    # Candidate actions in priority order: minimal maneuver first.
    candidates = (CommandKind.HOLD, CommandKind.MOVE_LEFT, CommandKind.MOVE_RIGHT)
    red_min, _, _, obs_min = _lane_metrics(state)

    best_action = CommandKind.HOLD
    best_score = float("inf")
    for action in candidates:
        target = _target_lane(action, own)

        # Hard safety: do not move into police lane.
        if 0 <= police_lane < n and target == police_lane:
            continue

        # Lower is better:
        # - prioritize nearer red tokens (negative component)
        # - avoid lanes with closer obstacles
        # - keep lane changes modest for stability
        has_red = red_min[target] < 1.2
        red_term = red_min[target] if has_red else 9.0
        obstacle_term = obs_min[target]
        maneuver_term = abs(target - own) * 0.15
        score = red_term + obstacle_term + maneuver_term

        if score < best_score:
            best_score = score
            best_action = action

    return best_action


def _police_escape_only_action(state: GameState, own: int) -> CommandKind:
    """After one red is collected, keep only collision-avoidance behavior."""
    police_lane = int(getattr(state, "police_lane", -1))
    return _best_avoid_red_action(state, own, disallow_lane=police_lane)


def _best_avoid_red_action(state: GameState, own: int, disallow_lane: int = -1) -> CommandKind:
    """Choose HOLD/LEFT/RIGHT that minimizes red-hit risk in the near term."""
    red_min, _, _, obs_min = _lane_metrics(state)
    candidates = (CommandKind.HOLD, CommandKind.MOVE_LEFT, CommandKind.MOVE_RIGHT)

    best_action = CommandKind.HOLD
    best_score = float("inf")
    for action in candidates:
        target = _target_lane(action, own)
        if 0 <= disallow_lane < config.NUM_LANES and target == disallow_lane:
            continue

        # Lower score is better.
        # Strongly penalize nearby red/obstacle, then prefer less maneuver.
        red_penalty = max(0.0, 1.0 - min(1.2, red_min[target])) * 6.0
        obs_penalty = max(0.0, 1.0 - min(1.2, obs_min[target])) * 3.0
        maneuver_penalty = abs(target - own) * 0.1
        score = red_penalty + obs_penalty + maneuver_penalty

        if score < best_score:
            best_score = score
            best_action = action

    return best_action


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
            cost += config.YELLOW_PENALTY * _token_proximity(d, lookahead)

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

    # Red-veto: in normal mode, heavily discourage lanes that contain red
    # in the near lookahead corridor.
    red_veto_guard = max(config.BRAKE_DIST, lookahead * 0.85)
    if red[target] <= red_veto_guard:
        cost += config.RED_PENALTY * 6.0

    cost += _color_cost(target, red, green, yellow, lookahead)
    cost += _obstacle_cost(target, obs, lookahead)
    cost += _stability_cost(target, own)

    return cost


def _best_far_green_action(
    own: int,
    red_min,
    green,
    yellow,
    obs_min,
    lookahead: float,
) -> Optional[CommandKind]:
    """Prefer the farthest safe green token during normal driving.

    Safety gates are strict: reject lanes with imminent red/obstacle.
    """
    best_action: Optional[CommandKind] = None
    best_score = -1.0

    for action in (CommandKind.HOLD, CommandKind.MOVE_LEFT, CommandKind.MOVE_RIGHT):
        target = _target_lane(action, own)

        # Hard safety first: reject any lane with red too close, not only
        # immediate BRAKE range.
        red_guard = max(config.BRAKE_DIST + 0.10, lookahead * 1.05)
        if red_min[target] <= red_guard:
            continue
        if obs_min[target] <= max(config.BRAKE_DIST, lookahead * 0.90):
            continue

        if not green[target]:
            continue

        far_green = max(green[target])

        # Green-seeking with explicit red-risk aversion.
        # Lower score impact from yellow than red; red remains dominant.
        red_risk_penalty = max(0.0, (lookahead * 1.35) - min(red_min[target], 1.2)) * 2.6
        yellow_near_penalty = 0.04 if any(d <= lookahead for d in yellow[target]) else 0.0
        maneuver_penalty = abs(target - own) * 0.03
        hold_penalty = 0.06 if action is CommandKind.HOLD else 0.0
        lane_change_bonus = 0.05 if action in (CommandKind.MOVE_LEFT, CommandKind.MOVE_RIGHT) else 0.0
        score = far_green - red_risk_penalty - yellow_near_penalty - maneuver_penalty - hold_penalty + lane_change_bonus

        if score > best_score:
            best_score = score
            best_action = action

    return best_action


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
        if not memory.police_event_active:
            memory.police_event_active = True
            memory.police_event_started_at = now
            memory.police_red_baseline = int(getattr(state, "red_tokens_collected", 0))
            memory.police_red_done = False

        # Stop chasing red once exactly one (or more) red has been collected
        # during the current police event window.
        current_red = int(getattr(state, "red_tokens_collected", 0))
        if current_red > memory.police_red_baseline:
            memory.police_red_done = True
            # Strict mode: latch red-avoidance for the remainder of the
            # police event. It is cleared only when police_alert is off.
            memory.post_police_avoid_red_until = max(
                memory.post_police_avoid_red_until,
                float("inf"),
            )

        # In real mode, red collection feedback may be delayed/noisy.
        # Stop aggressive red chasing after a short window and keep only
        # collision-avoidance to reduce police-car failures.
        if (now - memory.police_event_started_at) >= 2.5:
            memory.police_red_done = True

        if memory.police_red_done:
            action = _police_escape_only_action(state, own)
            reason = "POLICE_ESCAPE_AFTER_RED"
        else:
            action = _best_police_red_action(state, own)
            reason = "POLICE_RED_TOKEN_OVERRIDE"
        return DecisionResult(
            Command(action, now, reason),
            memory,
        )
    else:
        # Reset event tracking once police event is over.
        memory.police_event_active = False
        memory.police_event_started_at = -1e9
        memory.police_red_done = False
        memory.police_red_baseline = int(getattr(state, "red_tokens_collected", 0))
        memory.post_police_avoid_red_until = -1e9

    # ============================================================
    # 3. GOLDEN LANE — HIGH PRIORITY TARGET
    # ============================================================
    if getattr(state, "golden_lane_active", False):
        target = _normalize_lane_index(getattr(state, "golden_lane_index", -1))
        time_left = float(getattr(state, "golden_time_left", 0.0))
        if target >= 0:
            memory.last_golden_lane = target
        elif memory.last_golden_lane >= 0:
            # Use last known valid lane when perception briefly glitches.
            target = memory.last_golden_lane
        if 0 <= target < config.NUM_LANES:
            if time_left <= 1.0:
                # Final 1-second hard lock: do nothing except align to
                # the golden lane and hold there at expiry.
                if own < target:
                    return DecisionResult(
                        Command(CommandKind.MOVE_RIGHT, now, f"GOLDEN_HARD_LOCK_L{target + 1}"),
                        memory,
                    )
                if own > target:
                    return DecisionResult(
                        Command(CommandKind.MOVE_LEFT, now, f"GOLDEN_HARD_LOCK_L{target + 1}"),
                        memory,
                    )
                return DecisionResult(
                    Command(CommandKind.HOLD, now, f"GOLDEN_HARD_LOCK_L{target + 1}_ON_TARGET"),
                    memory,
                )
            if own < target:
                return DecisionResult(
                    Command(CommandKind.MOVE_RIGHT, now, f"GOLDEN_L{target + 1}"),
                    memory,
                )
            if own > target:
                return DecisionResult(
                    Command(CommandKind.MOVE_LEFT, now, f"GOLDEN_L{target + 1}"),
                    memory,
                )
            return DecisionResult(
                Command(CommandKind.HOLD, now, f"GOLDEN_L{target + 1}_ON_TARGET"),
                memory,
            )
        # Keep event override semantics even if lane OCR/index is temporarily invalid.
        return DecisionResult(
            Command(CommandKind.HOLD, now, "GOLDEN_ACTIVE_NO_VALID_LANE"),
            memory,
        )

    # ============================================================
    # 4. CHASE PRESSURE — HIGH PRIORITY OVERRIDE
    # ============================================================
    if getattr(state, "rear_chase_active", False):
        # Escape immediately whenever chase is active.
        if own < config.NUM_LANES - 1:
            action = CommandKind.MOVE_RIGHT
        else:
            action = CommandKind.MOVE_LEFT

        return DecisionResult(
            Command(action, now, "CHASE_ESCAPE"),
            memory,
        )

    # ============================================================
    # 4b. POST-POLICE RED AVOIDANCE (SHORT COOLDOWN)
    # ============================================================
    if now < memory.post_police_avoid_red_until:
        action = _best_avoid_red_action(state, own)
        return DecisionResult(
            Command(action, now, "POST_POLICE_AVOID_RED"),
            memory,
        )

    # ============================================================
    # 5. NORMAL COST POLICY
    # ============================================================
    lookahead = _effective_lookahead(state.speed_norm)

    red, green, yellow, obs = _lane_metrics(state)

    # Red emergency override in normal mode: if current lane is risky,
    # prioritize immediate avoidance over green pursuit.
    red_emergency_guard = max(config.BRAKE_DIST + 0.08, lookahead * 0.95)
    obs_emergency_guard = max(config.BRAKE_DIST, lookahead * 0.90)
    if red[own] <= red_emergency_guard or obs[own] <= obs_emergency_guard:
        action = _best_avoid_red_action(state, own)
        if _target_lane(action, own) == own and red[own] <= config.BRAKE_DIST:
            action = CommandKind.SLOW_DOWN
        return DecisionResult(
            Command(action, now, "RED_EMERGENCY_AVOID"),
            memory,
        )

    # If no challenge/event is active, explicitly pursue the farthest safe
    # green opportunity first; fall back to weighted cost if none is viable.
    far_green_action = _best_far_green_action(own, red, green, yellow, obs, lookahead)
    if far_green_action is not None:
        return DecisionResult(
            Command(far_green_action, now, "FAR_GREEN_POLICY"),
            memory,
        )

    best_action = CommandKind.HOLD
    best_cost = float("inf")

    for a in ACTIONS:
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
