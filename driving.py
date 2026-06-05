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
_decision_counter = 0
_last_reason = None


# ---------------------------------------------------------
# Decision logic
# ---------------------------------------------------------
def decide_target_lane(tokens, lane_centers, current_lane, rear_close, frame_h):
    """
    Pick the best lane to be in.

    Priority (high -> low):
      1. Rear vehicle close       -> sidestep to an adjacent lane
      2. Red/Yellow in current    -> move to a safer adjacent lane
      3. Green in adjacent lane   -> grab it
      4. Otherwise                -> hold current lane (or center if unknown)

    Returns (target_lane_index, reason_str).
    """
    n_lanes = len(lane_centers)
    if n_lanes == 0:
        return -1, "no_lanes"

    if current_lane < 0:
        return n_lanes // 2, "no_current_lane"

    def lane_of(token_x):
        return int(np.argmin([abs(token_x - c) for c in lane_centers]))

    # Only consider tokens that are in front of us (upper portion of frame).
    forward_cutoff = int(frame_h * 0.65)
    forward_tokens = [t for t in tokens if t['y'] < forward_cutoff]

    danger_in_lane = {i: False for i in range(n_lanes)}
    green_in_lane = {i: False for i in range(n_lanes)}
    for t in forward_tokens:
        idx = lane_of(t['x'])
        if t['color'] in ('red', 'yellow'):
            danger_in_lane[idx] = True
        elif t['color'] == 'green':
            green_in_lane[idx] = True

    adjacent = [i for i in (current_lane - 1, current_lane + 1) if 0 <= i < n_lanes]

    if rear_close and adjacent:
        safe = [i for i in adjacent if not danger_in_lane[i]]
        if safe:
            return safe[0], "rear_close_swerve"
        return adjacent[0], "rear_close_forced"

    if danger_in_lane[current_lane]:
        safe = [i for i in adjacent if not danger_in_lane[i]]
        if safe:
            return safe[0], "avoid_danger"
        return current_lane, "no_safe_neighbor"

    if not green_in_lane[current_lane]:
        green_adj = [i for i in adjacent if green_in_lane[i] and not danger_in_lane[i]]
        if green_adj:
            return green_adj[0], "chase_green"

    return current_lane, "hold"


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
    target_lane, reason = decide_target_lane(
        tokens, lane_centers, current_lane, rear_close, frame_h
    )
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
