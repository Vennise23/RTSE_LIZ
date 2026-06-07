"""
Member 1 — Perception / Computer Vision (OPTIMIZED VERSION)

Key improvements:
  - Frame ID filtering (no stale frame processing)
  - Detection throttling (reduces CPU load)
  - Lane caching (no per-frame lane recomputation)
  - Always process latest frame only
"""

import select
import time

import cv2
import numpy as np

import comms
import rtos
from rtos import shared_data, data_lock


SHOW_CAMERA = False


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

    color_ranges = {
        'green': [(np.array([40, 80, 80]), np.array([85, 255, 255]))],
        'yellow': [(np.array([20, 80, 80]), np.array([35, 255, 255]))],
        'red': [
            (np.array([0, 80, 80]), np.array([10, 255, 255])),
            (np.array([170, 80, 80]), np.array([180, 255, 255]))
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
            if area < 300 or area > 8000:
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
    # 7. WRITE BACK SHARED DATA
    # -----------------------------
    with data_lock:
        shared_data['detected_tokens'] = detected_tokens
        shared_data['lane_centers'] = lane_centers
        shared_data['current_lane'] = current_lane

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
        cv2.putText(debug, token['color'], (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    debug = cv2.resize(debug, (640, 480))
    cv2.imshow("Perception", debug)
    cv2.waitKey(1)