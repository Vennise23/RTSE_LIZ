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


def safe_corridor_score(i, red, n):
    """How many lanes wide the red-free corridor around lane i is."""
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
    close_red = {i: False for i in range(n)}
    yellow = {i: False for i in range(n)}
    green_score = {i: 0.0 for i in range(n)}
    red_score = {i: 0.0 for i in range(n)}

    forward_cut = max(int(frame_h * 0.85), 1)
    close_cut = frame_h * 0.55           # below this y a red is imminent
    for t in tokens:
        if t['y'] >= forward_cut:
            continue  # too low in frame -> already at the car, can't steer to it
        i = lane_of(t['x'], lane_centers)
        if t['color'] == 'red':
            red[i] = True
            if t['y'] > close_cut:
                close_red[i] = True
            # CLOSER red = bigger penalty (about to be collected). Reds repel
            # harder than greens attract, so we never take a red for a green.
            red_score[i] += t['y'] / forward_cut
        elif t['color'] == 'yellow':
            yellow[i] = True
        elif t['color'] == 'green':
            # Linear proximity weight in [0,1]: a green near the top of the
            # reachable window is worth ~1, one about to leave it ~0. Count
            # matters as much as nearness so we genuinely farm green (net +60).
            w = (forward_cut - t['y']) / forward_cut
            green_score[i] += max(0.0, w)
    return red, close_red, yellow, green_score, red_score


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


# =========================
# TACTICAL DECISION ENGINE
# =========================
def decide(tokens, lane_centers, current_lane, frame_h, ev):
    """Return (target_lane, reason). Darkness is handled by the caller."""
    n = len(lane_centers)
    if n == 0:
        return -1, "no_lane"

    red, close_red, yellow, green_score, red_score = build_lane_maps(
        tokens, lane_centers, frame_h)

    cop_lane = ev['police_lane'] if ev['police'] else -1
    golden = ev['golden_lane']
    rear = ev['rear_lane']

    # ---------------------------------------------------------------
    # EV2 POLICE — grab a red within 5s, NEVER enter the cop's lane.
    # ---------------------------------------------------------------
    if ev['police']:
        # If the cop is sitting in our lane, getting out is non-negotiable.
        target = nearest_red_lane(tokens, lane_centers, current_lane, frame_h, cop_lane)
        if target < 0:
            # No red visible yet: hold a safe lane (not the cop's) and wait for
            # one to scroll in, keeping out of the kill lane.
            target = nearest_safe_lane(current_lane, red, n, avoid=cop_lane)
            return target, "EV2_POLICE_WAIT_RED"
        return target, "EV2_POLICE_TAKE_RED"

    # ---------------------------------------------------------------
    # EV3/EV4 CHASING — car behind in our lane -> dodge to a DIFFERENT lane.
    # ---------------------------------------------------------------
    if rear == current_lane and rear >= 0:
        cands = [i for i in range(n)
                 if not red[i] and i != current_lane and i != cop_lane]
        if not cands:
            cands = [i for i in range(n) if not red[i] and i != current_lane]
        if cands:
            return min(cands, key=lambda i: abs(i - current_lane)), "EV34_CHASING_DODGE"
        # boxed in by reds on both sides: hold and let speed model cope
        return current_lane, "EV34_CHASING_BOXED"

    # ---------------------------------------------------------------
    # EV5 GOLDEN — lock onto the golden lane and hold until it expires.
    # ---------------------------------------------------------------
    if golden >= 0 and golden < n and not close_red[golden]:
        return golden, "EV5_GOLDEN_HOLD"

    # ---------------------------------------------------------------
    # SAFETY — only a CLOSE red in our lane is an emergency; far reds are
    # handled smoothly by the scoring penalty below (earlier, less jerky).
    # ---------------------------------------------------------------
    if close_red[current_lane]:
        return nearest_safe_lane(current_lane, close_red, n, avoid=cop_lane), "EMERGENCY_RED"
    if is_trapped(current_lane, close_red, n):
        return nearest_safe_lane(current_lane, close_red, n, avoid=cop_lane), "TRAP_ESCAPE"

    # ---------------------------------------------------------------
    # TACTICAL — minimise red FIRST, then farm green. Reds repel by proximity
    # (a near red is a big penalty, a far one small); greens attract; an
    # imminent red lane is never entered. Red weight (300) > green weight (170)
    # so we never trade a red for a green, but we still cross for clean green.
    # ---------------------------------------------------------------
    best_lane, best_score = current_lane, -1e9
    for i in range(n):
        if close_red[i]:
            continue                              # never enter an imminent red
        score = 0.0
        score += green_score[i] * 170.0           # farm green (net +60)
        score -= red_score[i] * 300.0             # PRIMARY: avoid red (repels harder)
        score += safe_corridor_score(i, red, n) * 15.0
        score -= abs(i - current_lane) * 18.0      # light anti-zigzag only
        if yellow[i]:
            score -= 100.0
        if i == cop_lane:
            score -= 1e6                            # hard-avoid the cop lane
        if score > best_score:
            best_score, best_lane = score, i
    # Hold current lane unless another lane is clearly better — kills 1-tick
    # flip-flop when two lanes score within a hair of each other.
    cur_score = (green_score[current_lane] * 170.0
                 - red_score[current_lane] * 300.0
                 + safe_corridor_score(current_lane, red, n) * 15.0) \
        if not close_red[current_lane] else -1e9
    if best_lane != current_lane and best_score - cur_score < 25.0:
        return current_lane, "TACTICAL_HOLD"
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

    # ===============================================================
    # FAST LANE LOCK (anti-jitter, but reacts in <=3 ticks)
    # ===============================================================
    # Event lanes (police/golden/dodge) must take effect immediately — only the
    # ordinary green-farming target gets the hysteresis lock.
    instant = reason in ("EV2_POLICE_TAKE_RED", "EV2_POLICE_WAIT_RED",
                         "EV34_CHASING_DODGE", "EV5_GOLDEN_HOLD",
                         "EMERGENCY_RED", "TRAP_ESCAPE")
    if instant or _lane_lock == -1:
        _lane_lock = target
        _lock_counter = 0
    elif target == _lane_lock:
        _lock_counter = 0
    else:
        # decide() already applies a switch-margin against flip-flop, so the
        # lock only needs a 1-tick confirm to commit quickly to a green.
        _lock_counter += 1
        if _lock_counter > 1:
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
