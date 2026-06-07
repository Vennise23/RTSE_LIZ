"""
Member 3 — Driving AI (Stable Goal Planner v5)

FIXES:
- removes distance instability
- prevents green flicker chasing
- strong red avoidance
- goal commitment system
- temporal smoothing + lane lock
"""

import struct
import numpy as np

import comms
from rtos import shared_data, data_lock


PRINT_EVERY = 25

# ---------------------------------------------------------
# STATE MEMORY (IMPORTANT FIX)
# ---------------------------------------------------------
_lane_lock = -1
_lock_counter = 0

_goal_lane = -1
_goal_timer = 0

_decision_counter = 0
_last_reason = None


# =========================================================
# HELPERS
# =========================================================
def lane_of(x, lane_centers):
    return int(np.argmin([abs(x - c) for c in lane_centers]))


def corridor_width(i, red, yellow, n):
    width = 1

    j = i - 1
    while j >= 0 and not red[j] and not yellow[j]:
        width += 1
        j -= 1

    j = i + 1
    while j < n and not red[j] and not yellow[j]:
        width += 1
        j += 1

    return width


# =========================================================
# STAGE 1 — FILTER (hard safety)
# =========================================================
def filter_safe(current_lane, red, n):
    safe = [i for i in range(n) if not red[i]]
    if not safe:
        return [current_lane]
    return safe


# =========================================================
# STAGE 2 — GREEN GOAL SELECTION (STABLE)
# =========================================================
def select_green_goal(tokens, lane_centers, red, yellow, current_lane, frame_h, n):

    forward_cutoff = int(frame_h * 0.85)
    tokens = [t for t in tokens if t['y'] < forward_cutoff]

    green_score = {}

    for i in range(n):
        green_score[i] = 0

    for t in tokens:
        li = lane_of(t['x'], lane_centers)
        dist = frame_h - t['y']

        if t['color'] == 'green':
            green_score[li] += 1000 / max(dist, 10)

        elif t['color'] == 'yellow':
            green_score[li] -= 200 / max(dist, 10)

        elif t['color'] == 'red':
            green_score[li] -= 5000 / max(dist, 10)

    # pick best green lane (stable)
    best_lane = max(green_score.items(), key=lambda x: x[1])[0]

    if green_score[best_lane] > 1:
        return best_lane

    return -1


# =========================================================
# STAGE 3 — COST PLANNER (stable navigation)
# =========================================================
def score_lane(i, tokens, lane_centers, current_lane, red, yellow, green, frame_h, n):

    cost = 0.0

    # -------------------------
    # red/yellow hard penalty
    # -------------------------
    if red[i]:
        cost += 10000
    if yellow[i]:
        cost += 800

    # neighbor danger
    if i > 0:
        if red[i - 1]:
            cost += 1500
        elif yellow[i - 1]:
            cost += 400

    if i < n - 1:
        if red[i + 1]:
            cost += 1500
        elif yellow[i + 1]:
            cost += 400

    # corridor bonus
    cost -= corridor_width(i, red, yellow, n) * 60

    # switching cost
    cost += abs(i - current_lane) * 50

    # -------------------------
    # green influence (weak)
    # -------------------------
    for t in tokens:
        li = lane_of(t['x'], lane_centers)
        if li != i:
            continue

        dist = frame_h - t['y']
        w = np.exp(-dist / (frame_h * 0.4))

        if t['color'] == 'green':
            cost -= 400 * w

    return cost


# =========================================================
# DECISION ENGINE (GOAL + FALLBACK)
# =========================================================
def decide_target_lane(tokens, lane_centers, current_lane, frame_h):

    global _goal_lane, _goal_timer

    n = len(lane_centers)
    if n == 0:
        return -1, "no_lanes"

    if current_lane < 0:
        return n // 2, "init_center"

    red = {i: False for i in range(n)}
    yellow = {i: False for i in range(n)}
    green = {i: False for i in range(n)}

    forward_cutoff = int(frame_h * 0.85)
    forward = [t for t in tokens if t['y'] < forward_cutoff]

    for t in forward:
        i = lane_of(t['x'], lane_centers)

        if t['color'] == 'red':
            red[i] = True
        elif t['color'] == 'yellow':
            yellow[i] = True
        elif t['color'] == 'green':
            green[i] = True

    # =====================================================
    # EMERGENCY
    # =====================================================
    if red[current_lane]:
        safe = filter_safe(current_lane, red, n)
        best = min(safe, key=lambda i: abs(i - current_lane))
        _goal_timer = 0
        _goal_lane = -1
        return best, "emergency"

    # =====================================================
    # GOAL LOCK SYSTEM (IMPORTANT FIX)
    # =====================================================
    if _goal_timer > 0 and _goal_lane != -1:
        _goal_timer -= 1
        return _goal_lane, "goal_locked"

    # =====================================================
    # NEW GOAL FROM GREEN
    # =====================================================
    new_goal = select_green_goal(tokens, lane_centers, red, yellow, current_lane, frame_h, n)

    if new_goal != -1:
        _goal_lane = new_goal
        _goal_timer = 25   # HOLD TARGET (VERY IMPORTANT)
        return _goal_lane, "new_green_goal"

    # =====================================================
    # FALLBACK MODE (NO GREEN)
    # =====================================================
    candidates = filter_safe(current_lane, red, n)

    best = min(
        candidates,
        key=lambda i: score_lane(
            i, tokens, lane_centers,
            current_lane, red, yellow, green,
            frame_h, n
        )
    )
    return best, "safe_navigation"


# =========================================================
# CONTROL
# =========================================================
def compute_controls(target_lane, lane_centers, frame_w, rear_close):

    if target_lane < 0:
        return 0.0, 0.6

    target_x = lane_centers[target_lane]
    car_x = frame_w / 2

    steering = (target_x - car_x) / (frame_w / 2)
    steering = max(-1.0, min(1.0, steering))

    accel = 0.95 * (1.0 - 0.75 * abs(steering))

    if rear_close:
        accel = min(1.0, accel + 0.1)

    accel = max(0.4, min(1.0, accel))

    return float(steering), float(accel)


# =========================================================
# MAIN LOOP
# =========================================================
def driving_logic_task():

    global _lane_lock, _lock_counter
    global _decision_counter, _last_reason

    with data_lock:
        frame = shared_data['latest_front_frame']
        tokens = list(shared_data['detected_tokens'])
        lane_centers = list(shared_data['lane_centers'])
        current_lane = shared_data['current_lane']
        rear_close = shared_data['rear_vehicle_close']

    if frame is None:
        return

    h, w, _ = frame.shape

    candidate, reason = decide_target_lane(
        tokens, lane_centers, current_lane, h
    )

    # ============================
    # LANE SMOOTHING
    # ============================
    if _lane_lock == -1:
        _lane_lock = candidate

    elif candidate == _lane_lock:
        _lock_counter = 0

    else:
        _lock_counter += 1
        if _lock_counter > 3:
            _lane_lock = candidate
            _lock_counter = 0

    target_lane = _lane_lock

    steering, accel = compute_controls(
        target_lane, lane_centers, w, rear_close
    )

    with data_lock:
        shared_data['target_lane'] = target_lane
        shared_data['decision_reason'] = reason
        shared_data['steering_input'] = steering
        shared_data['acceleration_input'] = accel

    _decision_counter += 1

    if reason != _last_reason or _decision_counter % PRINT_EVERY == 0:
        print(
            f"[STABLE-AI] {reason:<18} "
            f"{current_lane}->{target_lane} "
            f"steer={steering:+.2f} accel={accel:.2f}"
        )
        _last_reason = reason


# =========================================================
# SEND CONTROL
# =========================================================
def send_controls_task():

    if comms.control_conn is None:
        return

    with data_lock:
        steering = shared_data['steering_input']
        accel = shared_data['acceleration_input']

    try:
        comms.control_conn.sendall(struct.pack('ff', steering, accel))
    except Exception as e:
        print("Control send error:", e)
        comms.control_conn = None