import struct
import numpy as np

import comms
from rtos import shared_data, data_lock


# =========================================================
# CONFIG
# =========================================================
PRINT_EVERY = 25
TTC_HORIZON = 12.0          # how far ahead we care
DANGER_THRESHOLD = 0.65     # risk trigger
ESCAPE_MARGIN = 0.10        # bias to move away from danger


# =========================================================
# TTC ESTIMATION (CORE PREDICTION MODEL)
# =========================================================
def estimate_ttc(y, frame_h):
    """
    Simple but effective proxy:
    - higher y = closer to car
    - assume constant approach speed
    """

    norm_dist = 1.0 - (y / frame_h)
    norm_dist = max(0.01, norm_dist)

    assumed_speed = 0.06  # tuned constant (higher = faster approach)

    return norm_dist / assumed_speed


# =========================================================
# LANE MAPPING
# =========================================================
def lane_of(x, lane_centers):
    return int(np.argmin([abs(x - c) for c in lane_centers]))


# =========================================================
# TTC RISK FIELD (PREDICTIVE OCCUPANCY)
# =========================================================
def compute_risk_field(tokens, lane_centers, frame_h):
    n = len(lane_centers)
    risk = np.zeros(n, dtype=np.float32)

    for t in tokens:
        if t['color'] != 'red':
            continue

        lane = lane_of(t['x'], lane_centers)
        ttc = estimate_ttc(t['y'], frame_h)

        if ttc > TTC_HORIZON:
            continue

        # convert TTC → risk (closer = higher risk)
        r = (TTC_HORIZON - ttc) / TTC_HORIZON

        risk[lane] += r

        # spread risk to neighbors (important for realism)
        if lane - 1 >= 0:
            risk[lane - 1] += r * 0.6
        if lane + 1 < n:
            risk[lane + 1] += r * 0.6

    return risk


# =========================================================
# GREEN ATTRACTION FIELD
# =========================================================
def green_field(tokens, lane_centers, frame_h):
    n = len(lane_centers)
    reward = np.zeros(n, dtype=np.float32)

    for t in tokens:
        if t['color'] != 'green':
            continue

        lane = lane_of(t['x'], lane_centers)
        ttc = estimate_ttc(t['y'], frame_h)

        if ttc > TTC_HORIZON:
            continue

        r = (TTC_HORIZON - ttc) / TTC_HORIZON
        reward[lane] += r

    return reward


# =========================================================
# SAFETY ESCAPE (HARD KERNEL)
# =========================================================
def emergency_escape(current_lane, risk, n):
    """
    Always guarantee escape if possible.
    """

    # 1-hop escape
    candidates = [i for i in range(n) if risk[i] < DANGER_THRESHOLD]

    if candidates:
        return min(candidates, key=lambda x: abs(x - current_lane)), "TTC_ESCAPE"

    # 2-hop escape
    for d in [-2, 2]:
        i = current_lane + d
        if 0 <= i < n:
            return i, "TTC_2HOP_ESCAPE"

    return current_lane, "TTC_TRAPPED"


# =========================================================
# MAIN DECISION ENGINE
# =========================================================
def decide_target_lane(tokens, lane_centers, current_lane, frame_h):

    n = len(lane_centers)
    if n == 0:
        return -1, "no_lanes"

    if current_lane < 0:
        return n // 2, "init_center"

    def lane_of(x):
        return int(np.argmin([abs(x - c) for c in lane_centers]))

    # -------------------------------
    # BUILD LANE STATE
    # -------------------------------
    red = {i: False for i in range(n)}
    yellow = {i: False for i in range(n)}
    green = {i: False for i in range(n)}

    forward_cutoff = int(frame_h * 0.6)
    forward = [t for t in tokens if t['y'] < forward_cutoff]

    for t in forward:
        i = lane_of(t['x'])

        if t['color'] == 'red':
            red[i] = True
        elif t['color'] == 'yellow':
            yellow[i] = True
        elif t['color'] == 'green':
            green[i] = True

    # =====================================================
    # 🥇 RULE 1: GREEN ALWAYS WIN
    # =====================================================
    green_lanes = [i for i in range(n) if green[i] and not red[i]]

    if green_lanes:
        best = min(green_lanes, key=lambda x: abs(x - current_lane))
        return best, "GREEN_OVERRIDE"

    # =====================================================
    # 🥈 RULE 2: YELLOW + RED → VOID ONLY (SAFE LANES)
    # =====================================================
    safe_void = [
        i for i in range(n)
        if not red[i] and not yellow[i]
    ]

    if safe_void:
        best = min(safe_void, key=lambda x: abs(x - current_lane))
        return best, "SAFE_VOID"

    # =====================================================
    # 🥉 RULE 3: ONLY RED → ESCAPE
    # =====================================================
    if any(red.values()):
        safe = [i for i in range(n) if not red[i]]

        if safe:
            best = min(safe, key=lambda x: abs(x - current_lane))
            return best, "ESCAPE_RED"

        # 2-hop escape
        for d in [-2, 2]:
            i = current_lane + d
            if 0 <= i < n:
                return i, "ESCAPE_2HOP"

    # =====================================================
    # FALLBACK
    # =====================================================
    return current_lane, "HOLD"


# =========================================================
# CONTROL MAPPING
# =========================================================
def compute_controls(target_lane, lane_centers, frame_w, rear_close):

    if target_lane < 0 or not lane_centers:
        return 0.0, 0.6

    target_x = lane_centers[target_lane]
    car_x = frame_w / 2.0

    steering = (target_x - car_x) / (frame_w / 2.0)
    steering = float(np.clip(steering, -1.0, 1.0))

    accel = 0.88 * (1.0 - 0.55 * abs(steering))

    if rear_close:
        accel += 0.12

    return steering, float(np.clip(accel, 0.4, 1.0))


# =========================================================
# RTOS TASK
# =========================================================
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

    target_lane, reason = decide_target_lane(
        tokens, lane_centers, current_lane, frame_h
    )

    steering, accel = compute_controls(
        target_lane, lane_centers, frame_w, rear_close
    )

    with data_lock:
        shared_data['target_lane'] = target_lane
        shared_data['decision_reason'] = reason
        shared_data['steering_input'] = steering
        shared_data['acceleration_input'] = accel


# =========================================================
# SEND TASK
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
        print(f"Control send error: {e}")
        comms.control_conn = None