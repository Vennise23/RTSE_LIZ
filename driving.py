"""
Member 3 — AI Driving Logic.

Reads perception state from ``rtos.shared_data``, picks a target lane
according to a priority cascade, converts it to (steering, accel), and
publishes those back to ``shared_data`` for ``send_controls_task`` to send.
"""

import struct

import numpy as np

import comms
from rtos import shared_data, data_lock


# Print the latest decision to the terminal every PRINT_EVERY runs so the
# driving logic is observable without an OpenCV debug window.
PRINT_EVERY = 25  # at 40 ms period -> roughly once per second
EMERGENCY_REASONS = ("escape_mode", "stuck", "no_escape")
_decision_counter = 0
_last_reason = None
_lane_lock = -1
_lock_counter = 0

# ---------------------------------------------------------
# Decision logic
# ---------------------------------------------------------
def decide_target_lane(tokens, lane_centers, current_lane, rear_close, frame_h):

    n = len(lane_centers)
    if n == 0:
        return -1, "no_lanes"

    if current_lane < 0:
        return n // 2, "no_current_lane"

    def lane_of(x):
        return int(np.argmin([abs(x - c) for c in lane_centers]))

    forward_cutoff = int(frame_h * 0.65)
    forward = [t for t in tokens if t['y'] < forward_cutoff]

    danger = {i: False for i in range(n)}

    for t in forward:
        idx = lane_of(t['x'])
        if t['color'] in ('red', 'yellow'):
            danger[idx] = True

    left = current_lane - 1
    right = current_lane + 1

    # -------------------------------------------------
    # ⭐ 1. BOUNDARY SAFE ADJACENT
    # -------------------------------------------------
    candidates = []
    if left >= 0:
        candidates.append(left)
    if right < n:
        candidates.append(right)

    # -------------------------------------------------
    # ⭐ 2. IF CURRENT SAFE → STAY
    # -------------------------------------------------
    if not danger[current_lane]:
        return current_lane, "safe_hold"

    # -------------------------------------------------
    # ⭐ 3. FIND SAFE NEIGHBOR
    # -------------------------------------------------
    safe = [i for i in candidates if not danger[i]]
    if safe:
        return safe[0], "avoid_danger"

    # -------------------------------------------------
    # ⭐ 4. ESCAPE MODE (IMPORTANT FIX)
    # -------------------------------------------------
    # all lanes dangerous → pick least bad direction
    risk_score = []
    for i in candidates:
        score = 1 if danger[i] else 0
        risk_score.append((score, i))

    if risk_score:
        best = min(risk_score)[1]
        return best, "escape_mode"

    # -------------------------------------------------
    # ⭐ 5. NO MOVE POSSIBLE
    # -------------------------------------------------
    return current_lane, "stuck"


def compute_controls(target_lane, lane_centers, frame_w, rear_close):
    """Convert a target lane into (steering, acceleration)."""
    if target_lane < 0 or not lane_centers:
        return 0.0, 0.7

    target_x = lane_centers[target_lane]
    car_x = frame_w / 2.0
    steering = (target_x - car_x) / (frame_w / 2.0)
    steering = max(-1.0, min(1.0, steering))

    base = 1.0 if rear_close else 0.85
    accel = base * (1.0 - 0.4 * abs(steering))
    accel = max(0.3, min(1.0, accel))

    if rear_close:
        base = 1.0
    else:
        base = 0.9

    # ⭐ ADD THIS
    if target_lane < 0:
        return 0.0, 0.8

    # force forward if no steering change for long time
    if abs(steering) < 0.05:
        accel = max(accel, 0.75)
    return float(steering), float(accel)


# ---------------------------------------------------------
# Periodic tasks
# ---------------------------------------------------------
def driving_logic_task():
    """Read perception state, decide controls, publish to shared_data."""
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

    # -----------------------------
    # ⭐ LOCK LOGIC
    # -----------------------------
    LOCK_FRAMES = 8  # ~0.3s stable (depends on 25Hz task)

    if _lane_lock == -1 and candidate_lane != -1:
        _lane_lock = candidate_lane
        _lock_counter = 0

    elif reason in EMERGENCY_REASONS:
        # ❗ force override lock faster
        _lock_counter += 2
        if _lock_counter > 3:
            _lane_lock = candidate_lane
            _lock_counter = 0

    elif candidate_lane == _lane_lock:
        _lock_counter = 0

    else:
        _lock_counter += 1
        if _lock_counter > LOCK_FRAMES:
            _lane_lock = candidate_lane
            _lock_counter = 0

    if _lane_lock >= len(lane_centers):
        _lane_lock = len(lane_centers) // 2
    
    target_lane = _lane_lock
    steering, accel = compute_controls(target_lane, lane_centers, frame_w, rear_close)

    with data_lock:
        shared_data['target_lane'] = target_lane
        shared_data['decision_reason'] = reason
        shared_data['steering_input'] = steering
        shared_data['acceleration_input'] = accel

    # Visible heartbeat for Member 3 — print on cadence OR whenever the
    # reason changes (so you can see "hold -> chase_green" transitions live).
    global _decision_counter, _last_reason
    _decision_counter += 1
    if reason != _last_reason or _decision_counter % PRINT_EVERY == 0:
        print(
            f"[DrivingLogic] reason={reason:<18} "
            f"current_lane={current_lane} target_lane={target_lane} "
            f"steer={steering:+.2f} accel={accel:.2f} "
            f"(tokens={len(tokens)})"
        )
        _last_reason = reason


def send_controls_task():
    """Push the latest decision to the simulator."""
    if comms.control_conn is None:
        return

    with data_lock:
        steering_input = shared_data['steering_input']
        acceleration_input = shared_data['acceleration_input']

    try:
        data = struct.pack('ff', steering_input, acceleration_input)
        comms.control_conn.sendall(data)
    except Exception as e:
        print(f"Control send error: {e}")
        comms.control_conn = None
