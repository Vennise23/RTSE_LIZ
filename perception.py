"""
Member 1 — Perception / Computer Vision.

Publishes to ``rtos.shared_data``:
  - ``latest_front_frame`` / ``latest_back_frame``: decoded BGR frames
  - ``detected_tokens``: list of {color, x, y, area, box}
  - ``lane_centers``: x-pixel positions, left -> right
  - ``current_lane``: index into lane_centers, -1 if unknown
"""

import select

import cv2
import numpy as np

import comms
from rtos import shared_data, data_lock
import rtos


# Set True only when running with a non-headless OpenCV that supports cv2.imshow.
SHOW_CAMERA = False


# ---------------------------------------------------------
# Camera reading
# ---------------------------------------------------------
def _read_single_camera(sock, window_name, data_key):
    """Drain the socket for the latest available frame and publish it."""
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

        # Drain any extra queued frames so we always publish the freshest one.
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

                if SHOW_CAMERA:
                    frame_resized = cv2.resize(frame, (640, 480))
                    cv2.imshow(window_name, frame_resized)
                    cv2.waitKey(1)

    except Exception:
        pass


def read_front_camera_task():
    _read_single_camera(comms.front_camera_sock, "Front Camera", 'latest_front_frame')


def read_back_camera_task():
    _read_single_camera(comms.back_camera_sock, "Back Camera", 'latest_back_frame')


# ---------------------------------------------------------
# Token detection (green / red / yellow)
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
        'green':  [(np.array([40,  80,  80]), np.array([85,  255, 255]))],
        'yellow': [(np.array([20,  80,  80]), np.array([35,  255, 255]))],
        # Red hue wraps around 0, so use two ranges.
        'red':    [(np.array([0,   80,  80]), np.array([10,  255, 255])),
                   (np.array([170, 80,  80]), np.array([180, 255, 255]))],
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
# Lane detection
# ---------------------------------------------------------
def detect_lanes(frame):
    """
    Returns (lane_centers, current_lane).

    Strategy (top-down 2D racing view):
      1. Take a horizontal strip near the bottom (where the car is).
      2. Threshold for bright lane markings in HSV (low sat, high val).
      3. Sum the mask along Y to get a 1D column profile.
      4. Peaks above threshold (with clustering) = lane separator stripes.
      5. Lane centers = midpoints between adjacent separators.
      6. Current lane = lane center closest to image bottom-center X.
    """
    height, width, _ = frame.shape

    roi_y1 = int(height * 0.55)
    roi_y2 = int(height * 0.90)
    roi = frame[roi_y1:roi_y2, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([0,   0,   180])
    upper = np.array([180, 60,  255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    col_profile = mask.sum(axis=0).astype(np.float32)
    if col_profile.max() <= 0:
        return [], -1
    col_profile /= col_profile.max()

    peak_threshold = 0.4
    min_separation_px = max(20, width // 20)
    candidate_idx = np.where(col_profile > peak_threshold)[0]
    if len(candidate_idx) == 0:
        return [], -1

    # Cluster consecutive bright columns into single separator peaks.
    separators = []
    cluster_start = candidate_idx[0]
    prev = candidate_idx[0]
    for x in candidate_idx[1:]:
        if x - prev > min_separation_px:
            separators.append((cluster_start + prev) // 2)
            cluster_start = x
        prev = x
    separators.append((cluster_start + prev) // 2)

    if len(separators) < 2:
        return [], -1

    lane_centers = [(separators[i] + separators[i + 1]) // 2
                    for i in range(len(separators) - 1)]
    if not lane_centers:
        return [], -1

    car_x = width // 2
    current_lane = int(np.argmin([abs(c - car_x) for c in lane_centers]))
    return lane_centers, current_lane


# ---------------------------------------------------------
# Processing task (runs at MEDIUM priority)
# ---------------------------------------------------------
def processing_task():
    with data_lock:
        front_frame = shared_data['latest_front_frame']

    if front_frame is None:
        return

    detected_tokens = detect_tokens(front_frame)
    lane_centers, current_lane = detect_lanes(front_frame)

    with data_lock:
        shared_data['detected_tokens'] = detected_tokens
        shared_data['lane_centers'] = lane_centers
        shared_data['current_lane'] = current_lane

    if not SHOW_CAMERA:
        return

    debug_frame = front_frame.copy()
    h_frame = debug_frame.shape[0]

    for i, cx in enumerate(lane_centers):
        line_color = (0, 200, 0) if i == current_lane else (180, 180, 180)
        cv2.line(debug_frame, (cx, int(h_frame * 0.55)), (cx, h_frame - 1), line_color, 2)
        cv2.putText(debug_frame, f"L{i}", (cx - 10, int(h_frame * 0.58)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)
    cv2.putText(debug_frame, f"current_lane={current_lane}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for token in detected_tokens:
        x, y, w, h = token['box']
        color_name = token['color']
        if color_name == 'green':
            box_color = (0, 255, 0)
        elif color_name == 'red':
            box_color = (0, 0, 255)
        else:
            box_color = (0, 255, 255)
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(debug_frame, color_name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    debug_frame = cv2.resize(debug_frame, (640, 480))
    cv2.imshow("Token Detection - Member 1", debug_frame)
    cv2.waitKey(1)
