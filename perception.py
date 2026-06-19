"""
Member 1 — Perception / Computer Vision (OPTIMIZED VERSION)

Key improvements:
  - Frame ID filtering (no stale frame processing)
  - Detection throttling (reduces CPU load)
  - Lane caching (no per-frame lane recomputation)
  - Always process latest frame only
"""

import os
import select
import time

import cv2
import numpy as np

import comms
import rtos
from rtos import shared_data, data_lock


# Set env RTSE_NO_WINDOW=1 to suppress all OpenCV windows. IMPORTANT for frame
# capture: the Unity game PAUSES/freezes when our cv2 windows steal focus (this
# is why the rear camera went static mid-capture), so capture runs must use it.
SHOW_CAMERA = os.environ.get("RTSE_NO_WINDOW") != "1"
# Set env RTSE_DUMP=<dir> to save a handful of raw front frames for offline
# inspection / CV tuning. Default off — zero impact on normal runs.
_DUMP_DIR = os.environ.get("RTSE_DUMP")
_DUMP_N = int(os.environ.get("RTSE_DUMP_N", "6"))
_DUMP_GAP = float(os.environ.get("RTSE_DUMP_GAP", "1.0"))


# ---------------------------------------------------------
# Camera reading (UNCHANGED but safe)
# ---------------------------------------------------------
def _read_single_camera(sock, window_name, data_key):
    if sock is None:
        return

    try:
        latest_frame_data = None

        sock.settimeout(None)
        length_bytes = sock.recv(4)
        if not length_bytes:
            return

        image_length = int.from_bytes(length_bytes, 'little')
        received_bytes = b''

        while len(received_bytes) < image_length and rtos.is_running:
            packet = sock.recv(image_length - len(received_bytes))
            if not packet:
                break
            received_bytes += packet

        if len(received_bytes) == image_length:
            latest_frame_data = received_bytes

        # Drain queue → always keep freshest frame
        while rtos.is_running:
            readable, _, _ = select.select([sock], [], [], 0.0)
            if not readable:
                break

            sock.settimeout(1.0)
            length_bytes = sock.recv(4)
            if not length_bytes:
                return

            image_length = int.from_bytes(length_bytes, 'little')
            received_bytes = b''

            while len(received_bytes) < image_length and rtos.is_running:
                packet = sock.recv(image_length - len(received_bytes))
                if not packet:
                    break
                received_bytes += packet

            if len(received_bytes) == image_length:
                latest_frame_data = received_bytes

        if latest_frame_data is not None:
            np_arr = np.frombuffer(latest_frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                with data_lock:
                    shared_data[data_key] = frame

                    # ✅ FRAME ID INCREMENT (IMPORTANT)
                    if data_key == 'latest_front_frame':
                        shared_data['front_frame_id'] += 1
                    elif data_key == 'latest_back_frame':
                        shared_data['back_frame_id'] += 1

                if SHOW_CAMERA:
                    cv2.imshow(window_name, cv2.resize(frame, (640, 480)))
                    cv2.waitKey(1)

    except Exception:
        pass


def read_front_camera_task():
    _read_single_camera(comms.front_camera_sock, "Front Camera", 'latest_front_frame')


def read_back_camera_task():
    _read_single_camera(comms.back_camera_sock, "Back Camera", 'latest_back_frame')


# ---------------------------------------------------------
# Token detection (UNCHANGED CORE LOGIC)
# ---------------------------------------------------------
def detect_tokens(frame):
    detected_tokens = []
    height, width, _ = frame.shape

    roi_x1 = int(width * 0.15)
    roi_x2 = int(width * 0.85)
    roi_y1 = int(height * 0.05)
    roi_y2 = int(height * 0.70)

    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Ranges widened from live token samples: the game's tokens are vivid but
    # pastel — pink "red" tokens sit at H~11 (just past the old H<=10 cut) and
    # gold "yellow" tokens at H~18 (just below the old H>=20 cut), so both were
    # being missed. The background is blue (H~120) and the road is grey (low
    # S), so a lower S/V floor is safe and only adds real tokens.
    color_ranges = {
        'green': [(np.array([38, 70, 70]), np.array([90, 255, 255]))],
        'yellow': [(np.array([16, 80, 80]), np.array([34, 255, 255]))],
        'red': [
            (np.array([0, 70, 70]), np.array([15, 255, 255])),
            (np.array([168, 70, 70]), np.array([180, 255, 255]))
        ],
    }

    kernel = np.ones((5, 5), np.uint8)

    for color, ranges in color_ranges.items():
        mask = None

        for lower, upper in ranges:
            current_mask = cv2.inRange(hsv, lower, upper)
            mask = current_mask if mask is None else cv2.bitwise_or(mask, current_mask)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 180 or area > 8000:   # lower floor -> see far tokens sooner
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            if aspect_ratio < 0.6 or aspect_ratio > 1.6:
                continue

            x += roi_x1
            y += roi_y1

            detected_tokens.append({
                'color': color,
                'x': x + w // 2,
                'y': y + h // 2,
                'area': area,
                'box': (x, y, w, h),
            })

    return detected_tokens


# ---------------------------------------------------------
# Lane detection (same logic, unchanged)
# ---------------------------------------------------------
def detect_lanes(frame):
    height, width, _ = frame.shape
    roi = frame[int(height * 0.55):int(height * 0.95), :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 60, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    col = np.sum(mask, axis=0).astype(np.float32)

    if col.max() < 1:
        return [], -1

    col /= col.max()

    idx = np.where(col > 0.35)[0]
    if len(idx) == 0:
        return [], -1

    separators = []
    start = idx[0]
    prev = idx[0]

    min_gap = max(25, width // 30)

    for x in idx[1:]:
        if x - prev > min_gap:
            separators.append((start + prev) // 2)
            start = x
        prev = x

    separators.append((start + prev) // 2)

    separators = sorted(separators)

    if len(separators) > 6:
        step = len(separators) / 6
        separators = [separators[int(i * step)] for i in range(6)]

    while len(separators) < 6:
        if len(separators) >= 2:
            separators.append(separators[-1] + width // 5)
        else:
            return [], -1

    separators = sorted(separators[:6])

    lane_centers = [
        (separators[i] + separators[i + 1]) // 2
        for i in range(5)
    ]

    return lane_centers, -1


# =========================================================================
# V3.0 GAME-DAY EVENT DETECTION
#
# Every event is perceived from the camera pixels only (the Unity build
# exposes no telemetry — confirmed by decompiling Assembly-CSharp.dll:
# the golden-lane banner is literally drawn onto the camera stream texture
# via DrawGoldenBannerOnTexture). The thresholds below are the ones most
# likely to need a quick live tune on game day — they are all named.
# =========================================================================

# --- EV1 Darkness: front frame brightness collapses under the DarkOverlay.
# NOTE: the base scene is already a NIGHT city (~0.45 brightness in live runs),
# so the darkness event must drop well below that. Tuned off real frames.
DARK_BRIGHTNESS_ON = 0.28   # below this -> darkness active (brake fully)
DARK_BRIGHTNESS_OFF = 0.38  # above this -> darkness cleared (hysteresis)

# --- EV5 Golden Lane: one lane's tokens all turn green for 5s. We pick the
# lane holding a clear majority of the green tokens on screen.
GOLDEN_MIN_GREEN = 3        # need at least this many greens stacked in a lane
GOLDEN_MARGIN = 2           # and at least this many more than the runner-up lane

# --- EV2 police: detected by blue+red lights CO-LOCATED on the cop car. We
# dilate each colour mask then intersect them — a lone red token or blue patch
# of scenery never overlaps, only a real police car does. (Tuned after a live
# run showed bare blue/red presence false-positived almost every frame.)
POLICE_FLASH_DILATE = 21    # px: how close blue & red must be to count as one car
POLICE_OVERLAP_MIN = 120    # min overlapping px to declare the cop present
# Color-based cop detection is unreliable on this NIGHT scene (blue sky + our
# own red car + red curbs). Left OFF until replaced by reading the in-frame
# EV1-EV5 event tracker (top-centre HUD). Flip True only with a calibrated ROI.
ENABLE_POLICE_CV = False

# --- EV3/EV4 chasing cars from the BACK camera.
# OFF by default: on the night scene the rear frame is mostly dark, so a dark
# "car body" blob is found almost every frame -> 67 false dodges in a 55s live
# run. Until a headlight/sprite-specific detector is calibrated on real rear
# frames, we rely on normal driving to avoid the chasers (collision only costs
# -50% speed, whereas constant false dodging wrecks every lap).
ENABLE_CHASING_CV = False
REAR_BLOB_MIN_AREA = 1200   # min contour area to count as a vehicle
REAR_ROI_Y_FRAC = (0.25, 0.95)  # vertical band of the rear frame we scan

# --- EV1..EV5 tracker read off the top-centre HUD (green=passed, red=pending).
# Calibrated against live frames: 5 labels evenly spaced across the top.
EV_LABEL_XC = (0.336, 0.422, 0.500, 0.578, 0.656)  # label centres (frac of width)
EV_LABEL_YB = (0.110, 0.170)                        # vertical band (frac of height)
EV_LABEL_HALFW = 0.052                              # half label width (frac)


def read_event_tracker(front_frame):
    """Read EV1..EV5 pass status from the in-frame HUD.

    Returns a list of 5 values: 1 = passed (green label), 0 = pending (red
    label), -1 = unreadable (e.g. obscured during a darkness overlay). The
    caller latches the last good value when a slot reads -1.
    """
    h, w = front_frame.shape[:2]
    y1, y2 = int(h * EV_LABEL_YB[0]), int(h * EV_LABEL_YB[1])
    out = []
    for xc in EV_LABEL_XC:
        x1 = int((xc - EV_LABEL_HALFW) * w)
        x2 = int((xc + EV_LABEL_HALFW) * w)
        box = front_frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
        red = cv2.countNonZero(cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255])),
            cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))))
        grn = cv2.countNonZero(
            cv2.inRange(hsv, np.array([40, 80, 80]), np.array([85, 255, 255])))
        if grn > red and grn > 8:
            out.append(1)
        elif red > grn and red > 8:
            out.append(0)
        else:
            out.append(-1)
    return out


def frame_brightness(frame):
    """Mean V-channel brightness of the front frame, normalized to [0, 1]."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean()) / 255.0


def detect_golden_lane(tokens, lane_centers):
    """Return the lane index whose green tokens dominate the frame, else -1.

    During a Golden-Lane event every token in the chosen lane is rendered
    green, so that lane shows a tall stack of greens. We require both an
    absolute count (GOLDEN_MIN_GREEN) and a clear lead over every other lane
    (GOLDEN_MARGIN) so ordinary scattered greens never trip it.
    """
    n = len(lane_centers)
    if n == 0:
        return -1

    green_per_lane = [0] * n
    for t in tokens:
        if t['color'] != 'green':
            continue
        li = int(np.argmin([abs(t['x'] - c) for c in lane_centers]))
        green_per_lane[li] += 1

    best = int(np.argmax(green_per_lane))
    best_count = green_per_lane[best]
    if best_count < GOLDEN_MIN_GREEN:
        return -1

    runner_up = max((green_per_lane[i] for i in range(n) if i != best), default=0)
    if best_count - runner_up < GOLDEN_MARGIN:
        return -1
    return best


def _blue_red_lights(hsv):
    """Return (blue_px, red_px) counts for the blue+red police flash."""
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 90, 90]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 90, 90]), np.array([180, 255, 255])),
    )
    blue_mask = cv2.inRange(hsv, np.array([100, 90, 90]), np.array([130, 255, 255]))
    return blue_mask, red_mask


def detect_police_front(front_frame, lane_centers):
    """EV2 — the cop spawns AHEAD ('Position: N segments ahead, Lane: L') and
    flashes blue+red lights. Colliding with it is an instant GAME OVER, so we
    locate its lane to hard-avoid it while we go grab a red token.

    Returns ``(police_active, cop_lane)``; cop_lane is -1 when unknown.
    """
    if front_frame is None or len(lane_centers) == 0:
        return False, -1

    h, w = front_frame.shape[:2]
    # Restrict to the CENTRAL ROAD BAND only: above this band is the blue night
    # sky, below it is our own red car, and the left/right edges are the red
    # road curbs — all of which otherwise masquerade as police blue+red.
    y0, y1 = int(h * 0.40), int(h * 0.66)
    x0, x1 = int(w * 0.18), int(w * 0.82)
    roi = front_frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    blue_mask, red_mask = _blue_red_lights(hsv)

    # KEY: a red TOKEN is red with no blue beside it; blue scenery has no red
    # beside it. Only a police car carries blue AND red lights *co-located*.
    # So we dilate both masks and keep their OVERLAP — that is the cop, and it
    # rejects the constant red-token / blue-background false positives.
    k = np.ones((POLICE_FLASH_DILATE, POLICE_FLASH_DILATE), np.uint8)
    overlap = cv2.bitwise_and(cv2.dilate(blue_mask, k), cv2.dilate(red_mask, k))
    if cv2.countNonZero(overlap) < POLICE_OVERLAP_MIN:
        return False, -1

    contours, _ = cv2.findContours(overlap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cop_lane, best_area = -1, 0
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a <= best_area:
            continue
        x, _, bw, _ = cv2.boundingRect(cnt)
        cx = x0 + x + bw // 2          # ROI-relative -> full-frame x
        best_area = a
        cop_lane = int(np.argmin([abs(cx - c) for c in lane_centers]))
    return cop_lane >= 0, cop_lane


def detect_chasing_rear(back_frame, n_lanes):
    """EV3/EV4 — chasing cars close in from BEHIND (rear camera). Return the
    lane (0..n_lanes-1, left->right) of the nearest car behind us, or -1.

    The nearest car is the vehicle blob sitting lowest in the rear ROI
    (closest to our bumper). Any large vehicle-sized blob counts.
    """
    if back_frame is None or n_lanes == 0:
        return -1

    h, w = back_frame.shape[:2]
    ry1 = int(h * REAR_ROI_Y_FRAC[0])
    ry2 = int(h * REAR_ROI_Y_FRAC[1])
    roi = back_frame[ry1:ry2, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    blue_mask, red_mask = _blue_red_lights(hsv)
    veh_mask = cv2.bitwise_or(blue_mask, red_mask)
    veh_mask = cv2.bitwise_or(
        veh_mask, cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 70]))
    )  # also catch dark car bodies
    kernel = np.ones((5, 5), np.uint8)
    veh_mask = cv2.morphologyEx(veh_mask, cv2.MORPH_OPEN, kernel)
    veh_mask = cv2.morphologyEx(veh_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(veh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    threat_lane, lowest_y = -1, -1
    for cnt in contours:
        if cv2.contourArea(cnt) < REAR_BLOB_MIN_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        bottom = y + bh
        if bottom > lowest_y:
            lowest_y = bottom
            cx = x + bw // 2
            threat_lane = min(n_lanes - 1, int(cx / max(1, w) * n_lanes))
    return threat_lane


# ---------------------------------------------------------
# MAIN PERCEPTION TASK (OPTIMIZED CORE)
# ---------------------------------------------------------
def processing_task():

    # -----------------------------
    # 1. READ SHARED STATE FAST
    # -----------------------------
    with data_lock:
        frame_id = shared_data.get('front_frame_id', 0)
        front_frame = shared_data['latest_front_frame']

    if front_frame is None:
        return

    # -----------------------------
    # 2. INIT STATIC MEMORY
    # -----------------------------
    if not hasattr(processing_task, "last_frame_id"):
        processing_task.last_frame_id = -1
        processing_task.last_detect_time = 0
        processing_task.last_lane_time = 0
        processing_task.cached_lane_centers = []
        processing_task.cached_lane = -1
        processing_task.dark = False          # darkness hysteresis latch
        processing_task.green_seen = 0
        processing_task.red_seen = 0
        processing_task.prev_green_total = 0  # for crude collection counting
        processing_task.last_threat_time = 0  # rear/police CV runs slower
        processing_task.cached_police = False
        processing_task.cached_cop_lane = -1
        processing_task.cached_rear_lane = -1
        processing_task.dump_n = 0
        processing_task.last_dump_time = 0
        processing_task.events_passed = [False, False, False, False, False]
        processing_task.prev_passed = [False, False, False, False, False]
        processing_task.dump_burst = 0   # frames left to grab fast after an event

    # -----------------------------
    # 3. SKIP OLD FRAME (CRITICAL)
    # -----------------------------
    if frame_id == processing_task.last_frame_id:
        return

    processing_task.last_frame_id = frame_id

    now = time.time()

    # -----------------------------
    # 4. THROTTLE DETECTION (10–12 FPS)
    # -----------------------------
    if now - processing_task.last_detect_time < 0.08:
        return
    processing_task.last_detect_time = now

    # -----------------------------
    # 5. RUN DETECTION
    # -----------------------------
    detected_tokens = detect_tokens(front_frame)

    # -----------------------------
    # 6. LANE UPDATE (SLOW CACHE ~2 FPS)
    # -----------------------------
    if now - processing_task.last_lane_time > 0.5:
        lane_centers, _ = detect_lanes(front_frame)
        processing_task.cached_lane_centers = lane_centers
        processing_task.last_lane_time = now

    lane_centers = processing_task.cached_lane_centers

    # simple fallback
    current_lane = 2 if len(lane_centers) == 0 else 2

    # -----------------------------
    # 6b. V3.0 EVENT DETECTION
    # -----------------------------
    n_lanes = len(lane_centers) if len(lane_centers) > 0 else 5

    # EV1 Darkness — brightness with hysteresis so it doesn't chatter.
    bright = frame_brightness(front_frame)
    if processing_task.dark:
        if bright > DARK_BRIGHTNESS_OFF:
            processing_task.dark = False
    else:
        if bright < DARK_BRIGHTNESS_ON:
            processing_task.dark = True
    darkness = processing_task.dark

    # EV5 Golden Lane — lane dominated by green tokens.
    golden_lane = detect_golden_lane(detected_tokens, lane_centers)

    # EV2 police cop is AHEAD (front camera); EV3/EV4 chasing cars are BEHIND.
    # Vehicle CV is heavier, so it runs at ~6 FPS to protect the 40ms deadline;
    # the 5s event windows are far longer than this latency.
    if now - processing_task.last_threat_time > 0.15:
        processing_task.last_threat_time = now
        with data_lock:
            back_frame = shared_data.get('latest_back_frame')
        if ENABLE_POLICE_CV:
            processing_task.cached_police, processing_task.cached_cop_lane = \
                detect_police_front(front_frame, lane_centers)
        else:
            processing_task.cached_police, processing_task.cached_cop_lane = False, -1
        if ENABLE_CHASING_CV:
            processing_task.cached_rear_lane = detect_chasing_rear(back_frame, n_lanes)
        else:
            processing_task.cached_rear_lane = -1
    police = processing_task.cached_police
    cop_lane = processing_task.cached_cop_lane
    rear_lane = processing_task.cached_rear_lane

    # EV1..EV5 pass tracker (cheap, every detect tick). Latch to PASSED: a
    # label only ever turns green->stays green, so we never clear a True.
    ev_raw = read_event_tracker(front_frame)
    for idx, v in enumerate(ev_raw):
        if v == 1:
            processing_task.events_passed[idx] = True
    events_passed = list(processing_task.events_passed)

    # When any EV label flips to PASSED, an event just resolved -> trigger a
    # burst capture so we grab the police/chasing car that was on screen.
    if events_passed != processing_task.prev_passed:
        processing_task.dump_burst = 5
        processing_task.prev_passed = list(events_passed)

    # Crude token-flow counter for the operator HUD (best effort).
    green_total = sum(1 for t in detected_tokens if t['color'] == 'green')
    if green_total < processing_task.prev_green_total:
        processing_task.green_seen += (processing_task.prev_green_total - green_total)
    processing_task.prev_green_total = green_total

    # -----------------------------
    # 7. WRITE BACK SHARED DATA
    # -----------------------------
    with data_lock:
        shared_data['detected_tokens'] = detected_tokens
        shared_data['lane_centers'] = lane_centers
        shared_data['current_lane'] = current_lane
        shared_data['brightness'] = bright
        shared_data['event_darkness'] = darkness
        shared_data['event_golden_lane'] = golden_lane
        shared_data['event_police'] = police
        shared_data['police_lane'] = cop_lane
        shared_data['rear_threat_lane'] = rear_lane
        shared_data['green_seen'] = processing_task.green_seen
        shared_data['events_passed'] = events_passed

    # -----------------------------
    # 7b. OPTIONAL FRAME DUMP (CV tuning) — env RTSE_DUMP=<dir>
    # -----------------------------
    bursting = processing_task.dump_burst > 0
    due = bursting or now - processing_task.last_dump_time > _DUMP_GAP
    if _DUMP_DIR and processing_task.dump_n < _DUMP_N and due:
        processing_task.last_dump_time = now
        if bursting:
            processing_task.dump_burst -= 1
        try:
            os.makedirs(_DUMP_DIR, exist_ok=True)
            n = processing_task.dump_n
            cv2.imwrite(os.path.join(_DUMP_DIR, f"front_{n}.png"), front_frame)
            # Grab the REAR frame too (chasing cars approach from behind).
            with data_lock:
                back_frame = shared_data.get('latest_back_frame')
            if back_frame is not None:
                cv2.imwrite(os.path.join(_DUMP_DIR, f"rear_{n}.png"), back_frame)
            greens = sum(1 for t in detected_tokens if t['color'] == 'green')
            reds = sum(1 for t in detected_tokens if t['color'] == 'red')
            passed_str = "".join("1" if p else "0" for p in events_passed)
            tag = "BURST" if bursting else "tick "
            print(f"[DUMP {tag}] n={n} bright={bright:.2f} green={greens} red={reds} "
                  f"golden={golden_lane} EVpassed={passed_str} (raw={ev_raw})")
            processing_task.dump_n += 1
        except Exception as e:
            print("[DUMP] error:", e)

    # -----------------------------
    # 8. DEBUG VIEW (OPTIONAL)
    # -----------------------------
    if not SHOW_CAMERA:
        return

    debug = front_frame.copy()

    for token in detected_tokens:
        x, y, w, h = token['box']

        color = (0, 255, 0) if token['color'] == 'green' else \
                (0, 0, 255) if token['color'] == 'red' else \
                (0, 255, 255)

        cv2.rectangle(debug, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            debug,
            token['color'],
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    debug = cv2.resize(debug, (640, 480))

    # --- CONTROL OVERLAY (for presentation) ---
    with data_lock:
        steer = shared_data.get('steering_input', 0.0)
        accel = shared_data.get('acceleration_input', 0.0)
        reason = shared_data.get('decision_reason', '')
        cur_lane = shared_data.get('current_lane', -1)
        tgt_lane = shared_data.get('target_lane', -1)

    cv2.rectangle(debug, (0, 0), (640, 112), (0, 0, 0), -1)
    cv2.putText(debug, f"STEER {steer:+.2f}   ACCEL {accel:.2f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(debug, f"lane {cur_lane}->{tgt_lane}   {reason}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    # --- EVENT STATUS LINE (game-day tactical state) ---
    ev = []
    if darkness:
        ev.append("DARK")
    if golden_lane >= 0:
        ev.append(f"GOLD L{golden_lane}")
    if police:
        ev.append(f"POLICE copL{cop_lane}")
    if rear_lane >= 0:
        ev.append(f"CHASE L{rear_lane}")
    ev_txt = " | ".join(ev) if ev else "no event"
    ev_color = (0, 0, 255) if ev else (160, 160, 160)
    cv2.putText(debug, f"EV: {ev_txt}   bright={bright:.2f}  Gseen={processing_task.green_seen}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ev_color, 1)

    # Tactical win tracker read off the HUD: which of EV1..EV5 are passed.
    passed = events_passed
    pass_txt = " ".join(
        f"EV{i+1}" + ("OK" if passed[i] else "..") for i in range(len(passed)))
    n_passed = sum(1 for p in passed if p)
    pcol = (0, 220, 0) if n_passed == 5 else (0, 200, 255)
    cv2.putText(debug, f"PASS {n_passed}/5: {pass_txt}",
                (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, pcol, 1)

    cv2.imshow("Perception + Control", debug)
    cv2.waitKey(1)