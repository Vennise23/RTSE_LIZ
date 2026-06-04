import socket
import threading
import struct
import cv2
import numpy as np
import time
import keyboard
import select
import ctypes
import csv

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CAMERA_HOST = '127.0.0.1'
FRONT_CAMERA_PORT = 8080
BACK_CAMERA_PORT = 8082
CONTROL_HOST = '127.0.0.1'
CONTROL_PORT = 8081
SHOW_CAMERA = False

# Shared Resources with Mutex Lock for Concurrency
shared_data = {
    'latest_front_frame': None,
    'latest_back_frame': None,
    'steering_input' : 0.0,
    'acceleration_input' : 1.0,

    # Computer Vision
    'detected_tokens': [],
    'lane_centers': [],   # x-pixel of each detected lane center, left -> right
    'current_lane': -1,   # index into lane_centers; -1 if unknown
    'rear_vehicle_close': False,  # True when a fast car is closing in from behind

    # Member 3 — driving logic decision (debug)
    'target_lane': -1,
    'decision_reason': 'init',
}
data_lock = threading.Lock()
is_running = True

# ---------------------------------------------------------
# Real-Time Scheduling Framework (Do not change this in your code)
# ---------------------------------------------------------
class TaskPriority:
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class RTTask(threading.Thread):
    """
    Real-Time Task implementing:
    - Concurrency (inherits threading.Thread)
    - Task Period (enforced in run loop)
    - Task Priority (logical priority assigned)
    """
    def __init__(self, name, period, priority, execute_func):
        super().__init__()
        self.name = name
        self.period = period
        self.priority = priority
        self.execute_func = execute_func
        self.daemon = True
        self.wcet = 0
        self.deadline_miss = 0
        self.run_count = 0

    def run(self):
        print(f"[{self.name}] Started | Period: {self.period}s | Priority: {self.priority}")
        try:
            handle = ctypes.windll.kernel32.GetCurrentThread()
            if self.priority == TaskPriority.HIGH:
                ctypes.windll.kernel32.SetThreadPriority(handle, 2)
            elif self.priority == TaskPriority.MEDIUM:
                ctypes.windll.kernel32.SetThreadPriority(handle, 0)
            elif self.priority == TaskPriority.LOW:
                ctypes.windll.kernel32.SetThreadPriority(handle, -2)
        except Exception as e:
            pass

        next_release = time.perf_counter()
        while is_running:
            self.run_count += 1
            start_time = time.perf_counter()
            self.execute_func()
            exec_time = time.perf_counter() - start_time
            next_release += self.period
            sleep_time = next_release - time.perf_counter()

            if exec_time > self.wcet:
                self.wcet = exec_time

            if exec_time > self.period:

                self.deadline_miss += 1

                print(
                    f"[MISS] {self.name} "
                    f"{exec_time*1000:.2f}ms > "
                    f"{self.period*1000:.2f}ms"
                )

            if self.run_count % 200 == 0:

                print(
                    f"[{self.name}] "
                    f"Runs={self.run_count} "
                    f"WCET={self.wcet*1000:.2f}ms "
                    f"Miss={self.deadline_miss}"
                )

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_release = time.perf_counter()
            


# ---------------------------------------------------------
# Network Connection Setup (Do not change this in your code)
# ---------------------------------------------------------
front_camera_sock = None
back_camera_sock = None
control_conn = None

def setup_cameras():
    global front_camera_sock, back_camera_sock
    
    print("Connecting to Cameras...")
    front_connected = False
    back_connected = False
    
    while is_running and not (front_connected and back_connected):
        if not front_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, FRONT_CAMERA_PORT))
                front_camera_sock = s
                print("Connected to Front Camera successfully.")
                front_connected = True
            except Exception:
                pass
                
        if not back_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, BACK_CAMERA_PORT))
                back_camera_sock = s
                print("Connected to Back Camera successfully.")
                back_connected = True
            except Exception:
                pass
                
        if not (front_connected and back_connected):
            time.sleep(1)

def setup_control_server():
    global control_conn
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((CONTROL_HOST, CONTROL_PORT))
    server_sock.listen()
    server_sock.settimeout(1.0)
    print(f"Control server listening on {CONTROL_HOST}:{CONTROL_PORT}")
    
    while is_running:
        try:
            conn, addr = server_sock.accept()
            print(f"Control client connected from {addr}")
            control_conn = conn
            break
        except socket.timeout:
            continue

# ---------------------------------------------------------
# Task Implementations (This is where you write your tasks)
# ---------------------------------------------------------

def read_single_camera(sock, window_name, data_key):
    #This function reads the latest frame from the camera socket and stores it in the shared data
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
        while len(received_bytes) < image_length and is_running:
            packet = sock.recv(image_length - len(received_bytes))
            if not packet:
                break
            received_bytes += packet
            
        if len(received_bytes) == image_length:
            latest_frame_data = received_bytes
            
        while is_running:
            readable, _, _ = select.select([sock], [], [], 0.0)
            if not readable:
                break
                
            sock.settimeout(1.0)
            length_bytes = sock.recv(4)
            if not length_bytes:
                return
            image_length = int.from_bytes(length_bytes, 'little')
            received_bytes = b''
            while len(received_bytes) < image_length and is_running:
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
                    # You may disable this if you don't need to display the frames / This could effect the fps
                    # DEFAULT DISABLED -- VENNISE
                    frame_resized = cv2.resize(frame, (640, 480))
                    cv2.imshow(window_name, frame_resized)
                    cv2.waitKey(1)
                
    except Exception as e:
        pass

def read_front_camera_task():
    read_single_camera(front_camera_sock, "Front Camera", 'latest_front_frame')

def read_back_camera_task():
    read_single_camera(back_camera_sock, "Back Camera", 'latest_back_frame')

def detect_tokens(frame):
    detected_tokens = []

    height, width, _ = frame.shape

    # ROI: Only detect the middle and front area of the road
    roi_x1 = int(width * 0.15)
    roi_x2 = int(width * 0.85)
    roi_y1 = int(height * 0.05)
    roi_y2 = int(height * 0.70)

    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    color_ranges = {
        'green': [
            (np.array([40, 80, 80]), np.array([85, 255, 255]))
        ],
        'yellow': [
            (np.array([20, 80, 80]), np.array([35, 255, 255]))
        ],
        'red': [
            (np.array([0, 80, 80]), np.array([10, 255, 255])),
            (np.array([170, 80, 80]), np.array([180, 255, 255]))
        ]
    }

    for color, ranges in color_ranges.items():
        mask = None

        for lower, upper in ranges:
            current_mask = cv2.inRange(hsv, lower, upper)
            mask = current_mask if mask is None else cv2.bitwise_or(mask, current_mask)

        kernel = np.ones((5, 5), np.uint8)
        # Remove small noise from the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # Fill small gaps inside detected objects
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 300 or area > 8000:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Ignore objects that are too small or too large
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.6 or aspect_ratio > 1.6:
                continue

            # Convert ROI coordinates back to the original frame coordinates
            x = x + roi_x1
            y = y + roi_y1

            cx = x + w // 2
            cy = y + h // 2

            detected_tokens.append({
                'color': color,
                'x': cx,
                'y': cy,
                'area': area,
                'box': (x, y, w, h)
            })

    return detected_tokens

def detect_lanes(frame):
    """
    Detect lane separator stripes and return:
        lane_centers: list of x-pixel positions (left -> right) where the
                      drivable lane *centers* are.
        current_lane: index into lane_centers that the car is currently in,
                      or -1 if it cannot be determined.

    Strategy (top-down 2D racing view):
      1. Take a horizontal strip near the bottom (where the car is).
      2. Threshold for bright lane markings in HSV (low saturation, high value).
      3. Sum the binary mask along Y to get a 1-D column profile.
      4. Peaks in that profile = lane separator stripes.
      5. Lane centers = midpoints between adjacent separators (and between
         the outer separators and the image edges if the road fills the view).
      6. Current lane = lane center closest to image center-bottom X.
    """
    height, width, _ = frame.shape

    # ROI: a band near the bottom of the frame, in front of the car.
    roi_y1 = int(height * 0.55)
    roi_y2 = int(height * 0.90)
    roi = frame[roi_y1:roi_y2, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Lane markings are typically white / light: low saturation, high value.
    lower = np.array([0, 0, 180])
    upper = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Column-wise sum: tall vertical stripes -> high values.
    col_profile = mask.sum(axis=0).astype(np.float32)
    if col_profile.max() <= 0:
        return [], -1
    col_profile /= col_profile.max()

    # Find peaks: columns above threshold that are local maxima within a window.
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

    # Lane centers = midpoints between adjacent separators.
    lane_centers = []
    for i in range(len(separators) - 1):
        lane_centers.append((separators[i] + separators[i + 1]) // 2)

    if not lane_centers:
        return [], -1

    # Current lane = whichever lane center is closest to the car (image bottom center).
    car_x = width // 2
    current_lane = int(np.argmin([abs(c - car_x) for c in lane_centers]))

    return lane_centers, current_lane


def processing_task():
    # Member 1: Get latest front camera frame
    with data_lock:
        front_frame = shared_data['latest_front_frame']

    if front_frame is None:
        return

    # Detect green, red, yellow tokens
    detected_tokens = detect_tokens(front_frame)

    # Detect lane centers + which lane we are in
    lane_centers, current_lane = detect_lanes(front_frame)

    # Save detection result for Member 3
    with data_lock:
        shared_data['detected_tokens'] = detected_tokens
        shared_data['lane_centers'] = lane_centers
        shared_data['current_lane'] = current_lane

    # Draw bounding boxes for demo/debugging
    debug_frame = front_frame.copy()
    h_frame = debug_frame.shape[0]

    # Draw lane centers (vertical green lines) and label the current lane.
    for i, cx in enumerate(lane_centers):
        line_color = (0, 200, 0) if i == current_lane else (180, 180, 180)
        cv2.line(debug_frame, (cx, int(h_frame * 0.55)), (cx, h_frame - 1), line_color, 2)
        cv2.putText(
            debug_frame,
            f"L{i}",
            (cx - 10, int(h_frame * 0.58)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            line_color,
            2,
        )
    cv2.putText(
        debug_frame,
        f"current_lane={current_lane}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

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
        cv2.putText(
            debug_frame,
            color_name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2
        )

    debug_frame = cv2.resize(debug_frame, (640, 480))
    cv2.imshow("Token Detection - Member 1", debug_frame)
    cv2.waitKey(1)

def decide_target_lane(tokens, lane_centers, current_lane, rear_close, frame_h):
    """
    Member 3 — pick the best lane to be in.

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

    # If lane detection lost us, aim for the middle lane as a safe default.
    if current_lane < 0:
        return n_lanes // 2, "no_current_lane"

    def lane_of(token_x):
        # Assign a token to the nearest lane center.
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

    # 1) Rear vehicle closing in -> swerve to an adjacent lane (prefer safer one).
    if rear_close and adjacent:
        safe = [i for i in adjacent if not danger_in_lane[i]]
        if safe:
            return safe[0], "rear_close_swerve"
        return adjacent[0], "rear_close_forced"

    # 2) Danger in current lane -> change lane.
    if danger_in_lane[current_lane]:
        safe = [i for i in adjacent if not danger_in_lane[i]]
        if safe:
            return safe[0], "avoid_danger"
        # No safe neighbor: stay put rather than crash sideways.
        return current_lane, "no_safe_neighbor"

    # 3) Green token in adjacent lane and current lane has none -> grab it.
    if not green_in_lane[current_lane]:
        green_adj = [i for i in adjacent if green_in_lane[i] and not danger_in_lane[i]]
        if green_adj:
            return green_adj[0], "chase_green"

    # 4) Otherwise hold.
    return current_lane, "hold"


def compute_controls(target_lane, lane_centers, frame_w, rear_close):
    """Convert a target lane into (steering, acceleration)."""
    if target_lane < 0 or not lane_centers:
        return 0.0, 0.7  # cruise straight, modest throttle

    target_x = lane_centers[target_lane]
    car_x = frame_w / 2.0
    # Normalise pixel offset to [-1, 1] using half-width.
    steering = (target_x - car_x) / (frame_w / 2.0)
    steering = max(-1.0, min(1.0, steering))

    # Ease throttle while turning hard, push harder when escaping a rear car.
    base = 1.0 if rear_close else 0.85
    accel = base * (1.0 - 0.4 * abs(steering))
    accel = max(0.3, min(1.0, accel))
    return float(steering), float(accel)


def driving_logic_task():
    """Member 3 — read perception state, decide controls, publish to shared_data."""
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


def send_controls_task():
    # Read latest decision from Member 3 and push it to the simulator.
    global control_conn
    if control_conn is None:
        return

    with data_lock:
        steering_input = shared_data['steering_input']
        acceleration_input = shared_data['acceleration_input']

    try:
        data = struct.pack('ff', steering_input, acceleration_input)
        control_conn.sendall(data)
    except Exception as e:
        print(f"Control send error: {e}")
        control_conn = None


# ---------------------------------------------------------
# Main (Scheduler Initialization)
# ---------------------------------------------------------
if __name__ == '__main__':
    print("Initializing RTSE Sample Drive...")
    
    # Initialize network connections
    threading.Thread(target=setup_control_server, daemon=True).start()
    threading.Thread(target=setup_cameras, daemon=True).start()
    
    print("\n--- Starting Real-Time Tasks (awaiting connections dynamically) ---\n")
    
    # This is where you define tasks with explicit Scheduling parameters (Concurrency, Priority, Period)
    # Period refers to the period of execution of the task in seconds
    # Priority refers to the priority of the task, higher priority means higher priority
    # Concurrency refers to the number of instances of the task that can run at the same time
    t_front_camera = RTTask("ReadFrontCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_front_camera_task)
    t_back_camera = RTTask("ReadBackCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_back_camera_task)
    t_processing = RTTask("Processing", period=0.02, priority=TaskPriority.MEDIUM, execute_func=processing_task)
    t_driving = RTTask("DrivingLogic", period=0.02, priority=TaskPriority.MEDIUM, execute_func=driving_logic_task)
    t_controls = RTTask("SendControls", period=0.005, priority=TaskPriority.HIGH, execute_func=send_controls_task)

    # Start tasks to run concurrently
    t_front_camera.start()
    t_back_camera.start()
    t_processing.start()
    t_driving.start()
    t_controls.start()
    
    try:
        # You need this to keep the main thread alive, otherwise the program will exit immediately
        while is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Stopping system...")
        is_running = False

    # This is to make sure that the tasks are terminated cleanly
    t_front_camera.join()
    t_back_camera.join()
    t_processing.join()
    t_driving.join()
    t_controls.join()
    
    # This is to close all the connections
    if front_camera_sock:
        front_camera_sock.close()
    if back_camera_sock:
        back_camera_sock.close()
    if control_conn:
        control_conn.close()
    cv2.destroyAllWindows()
    print("System terminated cleanly.")
