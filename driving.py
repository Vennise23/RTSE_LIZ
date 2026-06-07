"""
Member 3 —  Driving AI (Cost-Based Planner)

Key upgrades:
- cost-based lane scoring (NOT rule-based)
- lookahead safety
- switching penalty
- temporal stability (hysteresis)
- emergency override
"""

import struct
import numpy as np

import comms
from rtos import shared_data, data_lock


PRINT_EVERY = 25

_decision_counter = 0
_last_reason = None

_lane_lock = -1
_lock_counter = 0


# ---------------------------------------------------------
# COST-BASED PLANNING CORE
# ---------------------------------------------------------
def corridor_width(lane_idx, red, yellow, n):

    width = 1

    i = lane_idx - 1

    while i >= 0:

        if red[i] or yellow[i]:
            break

        width += 1
        i -= 1

    i = lane_idx + 1

    while i < n:

        if red[i] or yellow[i]:
            break

        width += 1
        i += 1

    return width

def compute_lane_cost(
        i,
        lane_centers,
        current_lane,
        red,
        yellow,
        green,
        n):

    cost = 0.0

    # ==================================================
    # 1. HARD SAFETY
    # ==================================================

    if red[i]:
        cost += 5000

    if yellow[i]:
        cost += 1200

    # ==================================================
    # 2. NEIGHBOUR DANGER
    # ==================================================

    if i > 0:

        if red[i - 1]:
            cost += 800

        elif yellow[i - 1]:
            cost += 300

    if i < n - 1:

        if red[i + 1]:
            cost += 800

        elif yellow[i + 1]:
            cost += 300

    # ==================================================
    # 3. GREEN REWARD
    # ==================================================

    if green[i]:

        reward = 120

        reward -= abs(i - current_lane) * 20

        cost -= max(reward, 0)

    # ==================================================
    # 4. SAFE CORRIDOR BONUS
    # ==================================================

    width = corridor_width(i, red, n)

    cost -= width * 40

    # ==================================================
    # 5. LANE CHANGE COST
    # ==================================================

    cost += abs(i - current_lane) * 35

    # ==================================================
    # 6. EDGE PENALTY
    # ==================================================

    if i == 0 or i == n - 1:
        cost += 20

    return cost

def best_escape_lane(
        current_lane,
        red,
        yellow,
        n):

    best_lane = current_lane
    best_score = -99999

    for i in range(n):

        if red[i]:
            continue

        if yellow[i]:
            continue

        width = corridor_width(
            i,
            red,
            yellow,
            n
        )

        score = (
            width * 100
            - abs(i - current_lane) * 10
        )

        if score > best_score:

            best_score = score
            best_lane = i

    return best_lane

# ---------------------------------------------------------
# DECISION
# ---------------------------------------------------------
def decide_target_lane(tokens, lane_centers, current_lane, rear_close, frame_h):

    n = len(lane_centers)
    if n == 0:
        return -1, "no_lanes"

    if current_lane < 0:
        return n // 2, "init_center"

    def lane_of(x):
        return int(np.argmin([abs(x - c) for c in lane_centers]))

    # -------------------------------------------------
    # detection range (early reaction)
    # -------------------------------------------------
    forward_cutoff = int(frame_h * 0.85)
    forward = [t for t in tokens if t['y'] < forward_cutoff]

    red = {i: False for i in range(n)}
    yellow = {i: False for i in range(n)}
    green = {i: False for i in range(n)}

    for t in forward:
        idx = lane_of(t['x'])

        if t['color'] == 'red':
            red[idx] = True
        elif t['color'] == 'yellow':
            yellow[idx] = True
        elif t['color'] == 'green':
            green[idx] = True

    # -------------------------------------------------
    # EMERGENCY OVERRIDE (hard safety)
    # -------------------------------------------------
    if red[current_lane]:
        # escape immediately
        candidates = list(range(n))
        best =best_escape_lane(
            current_lane,
            red,
            n
        )
        return best, "emergency_red"

    # -------------------------------------------------
    # COST SEARCH (GLOBAL DECISION)
    # -------------------------------------------------
    candidates = list(range(n))

    costs = []
    for i in candidates:
        c = compute_lane_cost(
                i,
                lane_centers,
                current_lane,
                red,
                yellow,
                green,
                n
            )
        costs.append((c, i))

    best_lane = min(costs)[1]

    # -------------------------------------------------
    # REAR VEHICLE OVERRIDE
    # -------------------------------------------------
    if rear_close:
        # prefer left/right safe lane
        adj = [i for i in (current_lane - 1, current_lane + 1) if 0 <= i < n]
        safe_adj = [i for i in adj if not red[i]]
        if safe_adj:
            best_lane = safe_adj[0]
            return best_lane, "rear_escape"

    return best_lane, "cost_planning"


# ---------------------------------------------------------
# CONTROL
# ---------------------------------------------------------
def compute_controls(target_lane, lane_centers, frame_w, rear_close):

    if target_lane < 0 or not lane_centers:
        return 0.0, 0.7

    target_x = lane_centers[target_lane]
    car_x = frame_w / 2.0

    steering = (target_x - car_x) / (frame_w / 2.0)
    steering = max(-1.0, min(1.0, steering))

    # speed control
    accel = 0.95

    # slow down when turning
    accel *= (1.0 - 0.6 * abs(steering))

    # rear car boost
    if rear_close:
        accel = min(1.0, accel + 0.1)

    accel = max(0.5, min(1.0, accel))

    return float(steering), float(accel)


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
def driving_logic_task():

    with data_lock:
        front_frame = shared_data['latest_front_frame']
        tokens = list(shared_data['detected_tokens'])
        lane_centers = list(shared_data['lane_centers'])
        current_lane = shared_data['current_lane']
        rear_close = shared_data['rear_vehicle_close']

    if front_frame is None:
        return

    frame_h, frame_w, _ = front_frame.shape

    global _lane_lock, _lock_counter

    candidate_lane, reason = decide_target_lane(
        tokens, lane_centers, current_lane, rear_close, frame_h
    )

    # -------------------------------------------------
    # TEMPORAL STABILITY (HYSTERESIS)
    # -------------------------------------------------
    LOCK_FRAMES = 2

    if reason == "emergency_red":

        _lane_lock = candidate_lane
        _lock_counter = 0

    elif _lane_lock == -1 and candidate_lane != -1:

        _lane_lock = candidate_lane

    elif candidate_lane == _lane_lock:

        _lock_counter = 0

    else:

        _lock_counter += 1

        if _lock_counter > LOCK_FRAMES:

            _lane_lock = candidate_lane
            _lock_counter = 0

    # clamp
    if _lane_lock >= len(lane_centers):
        _lane_lock = max(0, len(lane_centers) // 2)

    target_lane = _lane_lock

    steering, accel = compute_controls(
        target_lane, lane_centers, frame_w, rear_close
    )

    with data_lock:
        shared_data['target_lane'] = target_lane
        shared_data['decision_reason'] = reason
        shared_data['steering_input'] = steering
        shared_data['acceleration_input'] = accel

    # -------------------------------------------------
    # DEBUG
    # -------------------------------------------------
    global _decision_counter, _last_reason
    _decision_counter += 1

    if reason != _last_reason or _decision_counter % PRINT_EVERY == 0:
        print(
            f"[AI-PROD] reason={reason:<15} "
            f"lane={current_lane}->{target_lane} "
            f"steer={steering:+.2f} accel={accel:.2f} "
            f"(tokens={len(tokens)})"
        )
        _last_reason = reason


# ---------------------------------------------------------
# SEND CONTROL
# ---------------------------------------------------------
def send_controls_task():

    if comms.control_conn is None:
        return

    with data_lock:
        steering = shared_data['steering_input']
        accel = shared_data['acceleration_input']

    try:
        data = struct.pack('ff', steering, accel)
        comms.control_conn.sendall(data)
    except Exception as e:
        print(f"Control send error: {e}")
        comms.control_conn = None