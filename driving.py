"""
Member 3 — Driving AI (V3.0 TACTICAL — Game-Day build)

GAME-DAY PRIORITY: **Tactical win > distance**.
  Tactical win = net +60 green tokens (green - red) AND pass every event once.
  Distance at 180s is only the fallback if the tactical win is out of reach.

So this version REVERSES the old "max distance, never chase green" policy:
we now actively farm green and deliberately satisfy each event's pass rule.

Event handling (rules confirmed by decompiling the V3.0 build + operator brief):
  EV1 Darkness  -> brake FULLY: steering=0, accel=-1.0. Any input while the
                   overlay is up is penalised (-10% speed), and the pass rule
                   is "decelerate fully", so we freeze + brake.
  EV2 Police    -> a cop spawns AHEAD in one lane (front cam, blue+red flash).
                   We MUST collect a red token within 5s, but COLLIDING with
                   the cop is an instant GAME OVER -> its lane is a hard-avoid.
  EV3/EV4 Chase -> a faster car closes from BEHIND (rear cam). If it is in our
                   lane, switch lanes to avoid the collision (-50% speed).
  EV5 Golden    -> one lane turns all-green for 5s; be in that lane when the
                   timer expires. We lock onto it and hold.

Priority order each tick:  Darkness > Police > Chasing > Golden > Tactical-green.
(Police outranks Golden because a missed police = speed halved and a possible
collision game-over; golden can be re-passed on the next rotation.)
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
# HELPERS (shared with old safe-corridor nav)
# =========================
def lane_of(x, centers):
    return int(np.argmin([abs(x - c) for c in centers]))


def is_trapped(i, red, n):
    """RRV / RRVVV deadlock detection."""
    left_block = (i <= 0) or red[i - 1]
    right_block = (i >= n - 1) or red[i + 1]
    return left_block and right_block


def nearest_safe_lane(current, red, n, avoid=-1):
    """Closest lane that is not red and not the hard-avoid lane (e.g. cop)."""
    candidates = [i for i in range(n) if not red[i] and i != avoid]
    if not candidates:
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
    k = i - 1
    while k >= 0 and not red[k]:
        width += 1
        k -= 1
    k = i + 1
    while k < n and not red[k]:
        width += 1
        k += 1
    return width


# =========================
# PERCEPTION -> PER-LANE MAPS
# =========================
def build_lane_maps(tokens, lane_centers, frame_h):
    """Return (red, yellow, green_score) dicts keyed by lane index.

    ``green_score[i]`` rewards closer greens more (proximity weighting), so the
    tactical scorer prefers lanes whose greens we can actually reach soon.
    """
    n = len(lane_centers)
    red = {i: False for i in range(n)}
    yellow = {i: False for i in range(n)}
    green_score = {i: 0.0 for i in range(n)}

    forward_cut = max(int(frame_h * 0.85), 1)
    for t in tokens:
        if t['y'] >= forward_cut:
            continue  # too low in frame -> already at the car, can't steer to it
        i = lane_of(t['x'], lane_centers)
        if t['color'] == 'red':
            red[i] = True
        elif t['color'] == 'yellow':
            yellow[i] = True
        elif t['color'] == 'green':
            # Linear proximity weight in [0,1]: a green near the top of the
            # reachable window is worth ~1, one about to leave it ~0. Count
            # matters as much as nearness so we genuinely farm green (net +60).
            w = (forward_cut - t['y']) / forward_cut
            green_score[i] += max(0.0, w)
    return red, yellow, green_score


def nearest_red_lane(tokens, lane_centers, current_lane, frame_h, avoid_lane):
    """EV2 helper: lane of the most reachable red token (never the cop lane).

    Picks the red that is closest to us laterally (fewest lane changes),
    tie-broken by vertical closeness so we commit to one we can actually hit
    inside the 5s window.
    """
    n = len(lane_centers)
    best_lane, best_key = -1, None
    for t in tokens:
        if t['color'] != 'red':
            continue
        li = lane_of(t['x'], lane_centers)
        if li == avoid_lane or li >= n:
            continue
        closeness = frame_h - t['y']            # bigger = nearer
        key = (abs(li - current_lane), -closeness)
        if best_key is None or key < best_key:
            best_key = key
            best_lane = li
    return best_lane


    green_scores = lane_token_score(forward, lane_centers, frame_h, 'green', current_lane)
    yellow_scores = lane_token_score(forward, lane_centers, frame_h, 'yellow', current_lane)

    # =========================
    # EMERGENCY (NO DISCUSSION)
    # =========================
    if red[current_lane]:
        return nearest_safe_lane(current_lane, red, n, avoid=cop_lane), "EMERGENCY_RED"
    if is_trapped(current_lane, red, n):
        return nearest_safe_lane(current_lane, red, n, avoid=cop_lane), "TRAP_ESCAPE"

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
            best_score, best_lane = score, i
    return best_lane, "TACTICAL_GREEN"


# =========================
# CONTROL (fast response + momentum)
# =========================
def control(target_lane, lane_centers, frame_w, rear_close):
    global _speed_mem

    if target_lane < 0 or not lane_centers:
        return 0.0, 0.6

    # Lane count can shrink between ticks (detect_lanes is noisy) while the
    # lane-lock still holds an older, larger index -> clamp to stay in range.
    target_lane = max(0, min(target_lane, len(lane_centers) - 1))
    car_x = frame_w / 2
    target_x = lane_centers[target_lane]

    steering = (target_x - car_x) / (frame_w / 2)
    steering = float(np.clip(steering * 0.75, -1, 1))

    turn = abs(steering)
    desired = 0.95
    desired -= 0.5 * turn
    if turn < 0.12:
        desired += 0.03
    if rear_close:
        desired += 0.10

    _speed_mem = _speed_mem * 0.80 + desired * 0.20
    _speed_mem = float(np.clip(_speed_mem, 0.4, 1.0))
    return steering, _speed_mem


# =========================
# MAIN LOOP
# =========================
def driving_logic_task():
    global _lane_lock, _lock_counter
    global _counter, _last_reason, _speed_mem

    with data_lock:
        frame = shared_data['latest_front_frame']
        tokens = list(shared_data['detected_tokens'])
        lane_centers = list(shared_data['lane_centers'])
        current_lane = shared_data['current_lane']
        rear_close = shared_data['rear_vehicle_close']
        ev = {
            'darkness': shared_data['event_darkness'],
            'police': shared_data['event_police'],
            'police_lane': shared_data['police_lane'],
            'golden_lane': shared_data['event_golden_lane'],
            'rear_lane': shared_data['rear_threat_lane'],
        }

    if frame is None:
        return

    h, w, _ = frame.shape

    # ===============================================================
    # EV1 DARKNESS OVERRIDE — brake fully, no steering. Any input while
    # the overlay is up is penalised, and the pass rule is "fully brake".
    # ===============================================================
    if ev['darkness']:
        _lane_lock = -1            # drop the lock so we re-acquire cleanly after
        _speed_mem = 0.0
        with data_lock:
            shared_data['target_lane'] = current_lane
            shared_data['decision_reason'] = "EV1_DARKNESS_BRAKE"
            shared_data['steering_input'] = 0.0
            shared_data['acceleration_input'] = -1.0
        _counter += 1
        if _counter % PRINT_EVERY == 0 or _last_reason != "EV1_DARKNESS_BRAKE":
            print("[V3] EV1_DARKNESS_BRAKE  steer=+0.00 accel=-1.00 (freeze+brake)")
            _last_reason = "EV1_DARKNESS_BRAKE"
        return

    target, reason = decide(tokens, lane_centers, current_lane, h, ev)

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

    _counter += 1
    if _counter % PRINT_EVERY == 0 or reason != _last_reason:
        print(f"[V3] {reason:<20} lane={current_lane}->{_lane_lock} "
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
