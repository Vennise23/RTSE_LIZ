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


# =========================
# LOCAL GREEN ONLY (NO CHASE)
# =========================
def local_green_score(tokens, lane_centers, frame_h, current_lane, n):
    score = {i: 0 for i in range(n)}

    for t in tokens:
        if t['color'] != 'green':
            continue

        li = lane_of(t['x'], lane_centers)
        dist = frame_h - t['y']

        # ONLY local influence (prevents chasing far green)
        if abs(li - current_lane) <= 1:
            score[li] += 1.0 / max(dist, 20)

    best_lane = max(score, key=score.get)
    return best_lane if score[best_lane] > 0.02 else -1


# =========================
# DECISION ENGINE (FAST SAFETY FIRST)
# =========================
def lane_risk_score(i, red, yellow, n):

    # HARD BLOCK
    if red[i]:
        return 1e9

    risk = 0

    # local danger
    if i > 0 and red[i - 1]:
        risk += 500
    if i < n - 1 and red[i + 1]:
        risk += 500

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
    # GLOBAL SAFE SEARCH (IMPORTANT FIX)
    # =========================
    best_lane = current_lane
    best_score = -1e9

    for i in range(n):

        risk = lane_risk_score(i, red, yellow, n)
        if risk > 800:
            continue  # avoid dangerous lanes

        corridor = safe_corridor_score(i, red, yellow, n)

        score = 0
        score += corridor * 120

        # stability (avoid zigzag)
        score -= abs(i - current_lane) * 35

        # slight center preference
        score -= abs(i - n//2) * 10

        # yellow penalty
        if yellow[i]:
            score -= 150

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
    # FAST LANE LOCK (NO LAG)
    # =========================
    if _lane_lock == -1:
        _lane_lock = target

    elif target == _lane_lock:
        _lock_counter = 0

    else:
        _lock_counter += 1
        if _lock_counter > 2:  # FAST reaction
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