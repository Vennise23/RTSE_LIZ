"""
Member 3 — Driving AI (V10 FIXED)
CORE GOAL: MAX DISTANCE (NOT GREEN CHASING)

FIXES:
- RRV / RRVVV deadlock escape
- NO red entry EVER
- green only local reward
- void escape when trapped
- fast reaction (no goal delay)
- lane lock stability without lag
"""

import struct
import numpy as np
import comms
from rtos import shared_data, data_lock


# =========================
# STATE MEMORY
# =========================
_lane_lock = -1
_lock_counter = 0

_speed_mem = 0.85
_last_reason = ""
_counter = 0

PRINT_EVERY = 20


# =========================
# HELPERS
# =========================
def lane_of(x, centers):
    return int(np.argmin([abs(x - c) for c in centers]))


def corridor_width(i, red, yellow, n):
    w = 1
    j = i - 1
    while j >= 0 and not red[j] and not yellow[j]:
        w += 1
        j -= 1

    j = i + 1
    while j < n and not red[j] and not yellow[j]:
        w += 1
        j += 1
    return w


def is_trapped(i, red, n):
    """RRV / RRVVV fix detection"""
    left_block = (i <= 0) or red[i - 1]
    right_block = (i >= n - 1) or red[i + 1]
    return left_block and right_block


def nearest_safe_lane(current, red, n):
    candidates = [i for i in range(n) if not red[i]]
    if not candidates:
        return current
    return min(candidates, key=lambda i: abs(i - current))


def lane_red_distance(i, red, n):
    """Return the distance in lanes to the nearest red token lane."""
    if not any(red.values()):
        return n

    distances = [abs(i - j) for j in range(n) if red[j]]
    return min(distances) if distances else n


def is_red_danger_zone(i, red, n):
    """Hard block lanes that are red or directly adjacent to red."""
    if red[i]:
        return True
    if i > 0 and red[i - 1]:
        return True
    if i < n - 1 and red[i + 1]:
        return True
    return False


# =========================
# GREEN-FIRST TARGETING
# =========================
def lane_token_score(tokens, lane_centers, frame_h, color, current_lane=None):
    score = {i: 0.0 for i in range(len(lane_centers))}

    for t in tokens:
        if t['color'] != color:
            continue

        li = lane_of(t['x'], lane_centers)
        dist = max(frame_h - t['y'], 20)
        proximity = 1.0 / dist

        # Prefer tokens that are lower on screen and closer to our current lane.
        lateral_bias = 1.0
        if current_lane is not None:
            lateral_bias = 1.0 / (1.0 + abs(li - current_lane) * 0.65)

        if color == 'green':
            score[li] += proximity * 5.0 * lateral_bias
        elif color == 'yellow':
            score[li] += proximity * 1.6 * lateral_bias
        elif color == 'red':
            score[li] += proximity * 0.15 * lateral_bias

    return score


# =========================
# DECISION ENGINE (FAST SAFETY FIRST)
# =========================
def lane_risk_score(i, red, yellow, n):

    # HARD BLOCK
    if is_red_danger_zone(i, red, n):
        return 1e9

    risk = 0

    # local danger
    red_dist = lane_red_distance(i, red, n)
    risk += max(0, 4 - red_dist) * 250

    # trap pattern detection (RRV / RRVVV)
    if i > 0 and i < n - 1:
        if red[i - 1] and red[i + 1]:
            risk += 1200

    # yellow weak penalty
    if yellow[i]:
        risk += 120

    return risk


def safe_corridor_score(i, red, yellow, n):

    # how long can we move forward without hitting red
    width = 1

    j = i
    # look LEFT corridor
    k = i - 1
    while k >= 0 and not red[k]:
        width += 1
        k -= 1

    # look RIGHT corridor
    k = i + 1
    while k < n and not red[k]:
        width += 1
        k += 1

    return width

def decide(tokens, lane_centers, current_lane, frame_h):

    n = len(lane_centers)
    if n == 0:
        return -1, "no_lane"

    red = {i: False for i in range(n)}
    yellow = {i: False for i in range(n)}

    forward_cut = int(frame_h * 0.85)
    forward = [t for t in tokens if t['y'] < forward_cut]

    for t in forward:
        i = lane_of(t['x'], lane_centers)
        if t['color'] == "red":
            red[i] = True
        elif t['color'] == "yellow":
            yellow[i] = True

    green_scores = lane_token_score(forward, lane_centers, frame_h, 'green', current_lane)
    yellow_scores = lane_token_score(forward, lane_centers, frame_h, 'yellow', current_lane)

    # =========================
    # EMERGENCY (NO DISCUSSION)
    # =========================
    if red[current_lane]:
        return nearest_safe_lane(current_lane, red, n), "EMERGENCY_RED"

    # =========================
    # TRAP ESCAPE (RRV FIX)
    # =========================
    if is_trapped(current_lane, red, n):
        escape = nearest_safe_lane(current_lane, red, n)
        return escape, "TRAP_ESCAPE"

    # =========================
    # GREEN FIRST
    # =========================
    best_green_lane = -1
    best_green_score = 0.0
    for i in range(n):
        if is_red_danger_zone(i, red, n):
            continue
        score = green_scores[i]
        score += lane_red_distance(i, red, n) * 0.25
        if score > best_green_score:
            best_green_score = score
            best_green_lane = i

    if best_green_lane != -1 and best_green_score >= 0.012:
        return best_green_lane, f"CHASE_GREEN_{best_green_lane}"

    # =========================
    # YELLOW SECOND
    # =========================
    best_yellow_lane = -1
    best_yellow_score = 0.0
    for i in range(n):
        if is_red_danger_zone(i, red, n):
            continue
        score = yellow_scores[i]
        score += lane_red_distance(i, red, n) * 0.15
        if score > best_yellow_score:
            best_yellow_score = score
            best_yellow_lane = i

    if best_yellow_lane != -1 and best_yellow_score >= 0.011:
        # Only take yellow if there is no strong green option.
        return best_yellow_lane, f"CHASE_YELLOW_{best_yellow_lane}"

    # =========================
    # FALLBACK SAFE SEARCH
    # =========================
    best_lane = current_lane
    best_score = -1e9

    for i in range(n):
        risk = lane_risk_score(i, red, yellow, n)
        if risk > 0:
            continue  # avoid dangerous lanes

        corridor = safe_corridor_score(i, red, yellow, n)

        score = 0
        score += corridor * 120
        score += lane_red_distance(i, red, n) * 140

        # If we already saw some green elsewhere, gently bias toward it.
        score += green_scores[i] * 900

        # stability (avoid zigzag)
        score -= abs(i - current_lane) * 35

        # slight center preference
        score -= abs(i - n//2) * 10

        # yellow penalty
        if yellow[i]:
            score -= 260

        # stay far from any red cluster even when falling back
        score += lane_red_distance(i, red, n) * 60

        if score > best_score:
            best_score = score
            best_lane = i

    return best_lane, "SAFE_CORRIDOR_NAV"


# =========================
# CONTROL (FAST RESPONSE + MOMENTUM)
# =========================
def control(target_lane, lane_centers, frame_w, rear_close):

    global _speed_mem

    if target_lane < 0:
        return 0.0, 0.6

    car_x = frame_w / 2
    target_x = lane_centers[target_lane]

    steering = (target_x - car_x) / (frame_w / 2)
    steering = float(np.clip(steering * 0.75, -1, 1))

    # =========================
    # SPEED MODEL (FIXED)
    # =========================
    turn = abs(steering)

    # base speed
    desired = 0.95

    # turn slows down
    desired -= 0.5 * turn

    # straight boost (IMPORTANT)
    if turn < 0.12:
        desired += 0.03

    # rear pressure
    if rear_close:
        desired += 0.10

    # smooth memory (prevents drop to 0.6 spikes)
    _speed_mem = _speed_mem * 0.80 + desired * 0.20
    _speed_mem = float(np.clip(_speed_mem, 0.4, 1.0))

    return steering, _speed_mem


# =========================
# MAIN LOOP
# =========================
def driving_logic_task():

    global _lane_lock, _lock_counter
    global _counter, _last_reason

    with data_lock:
        frame = shared_data['latest_front_frame']
        tokens = list(shared_data['detected_tokens'])
        lane_centers = list(shared_data['lane_centers'])
        current_lane = shared_data['current_lane']
        rear_close = shared_data['rear_vehicle_close']

    if frame is None:
        return

    h, w, _ = frame.shape

    target, reason = decide(tokens, lane_centers, current_lane, h)

    # =========================
    # FAST LANE LOCK (GREEN CAN PREEMPT)
    # =========================
    if _lane_lock == -1:
        _lane_lock = target
    elif target == _lane_lock:
        _lock_counter = 0
    elif target >= 0 and (
        "CHASE_GREEN" in reason
        or "EMERGENCY_RED" in reason
        or "TRAP_ESCAPE" in reason
    ):
        _lane_lock = target
        _lock_counter = 0
    else:
        _lock_counter += 1
        if _lock_counter > 1:  # faster reaction, less stickiness
            _lane_lock = target
            _lock_counter = 0

    steer, accel = control(_lane_lock, lane_centers, w, rear_close)

    with data_lock:
        shared_data['target_lane'] = _lane_lock
        shared_data['decision_reason'] = reason
        shared_data['steering_input'] = steer
        shared_data['acceleration_input'] = accel

    # =========================
    # DEBUG
    # =========================
    _counter += 1
    if _counter % PRINT_EVERY == 0 or reason != _last_reason:
        print(f"[V10] {reason:<14} lane={current_lane}->{_lane_lock} "
              f"steer={steer:+.2f} accel={accel:.2f}")
        _last_reason = reason


# =========================
# SEND
# =========================
def send_controls_task():
    if comms.control_conn is None:
        return

    with data_lock:
        s = shared_data['steering_input']
        a = shared_data['acceleration_input']

    try:
        comms.control_conn.sendall(struct.pack('ff', s, a))
    except Exception as e:
        print("send error:", e)
        comms.control_conn = None
