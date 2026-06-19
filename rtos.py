"""
Member 2 — RTOS / Concurrent Programming.

Provides:
  - Shared state dictionary (``shared_data``) and its mutex (``data_lock``).
  - Global ``is_running`` flag (access via ``rtos.is_running`` from other modules).
  - ``TaskPriority`` and ``RTTask`` for periodic, prioritised thread execution
    with WCET and deadline-miss tracking.
"""

import ctypes
import threading
import time


# ---------------------------------------------------------
# Shared Resources with Mutex Lock for Concurrency
# ---------------------------------------------------------
shared_data = {
    'latest_front_frame': None,
    'latest_back_frame': None,
    'steering_input': 0.0,
    'acceleration_input': 1.0,
    
    'front_frame_id': 0,
    'back_frame_id': 0,

    # Computer Vision
    'detected_tokens': [],
    'lane_centers': [],          # x-pixel of each detected lane center, left -> right
    'current_lane': -1,          # index into lane_centers; -1 if unknown
    'rear_vehicle_close': False, # True when a fast car is closing in from behind

    # ---------------------------------------------------------------
    # V3.0 Game-Day EVENT STATE (written by perception, read by driving)
    # All events are perceived from the camera streams only — the game
    # exposes no telemetry channel (confirmed by decompiling the build).
    # ---------------------------------------------------------------
    'brightness': 1.0,           # mean normalized brightness of front frame [0..1]
    'event_darkness': False,     # EV1: brightness collapsed -> must brake fully (accel=-1)
    'event_police': False,       # EV2: police car ahead -> must grab a red token, never hit cop
    'police_lane': -1,           # EV2: lane the cop occupies (HARD avoid, collision = GAME OVER)
    'event_golden_lane': -1,     # EV5: lane index that is "golden" right now, else -1
    'rear_threat_lane': -1,      # EV3/EV4: lane of a chasing car closing from behind, else -1

    # EV1..EV5 pass status read straight off the in-frame HUD tracker
    # (green label = PASSED, red label = still pending). This IS the tactical
    # win progress: the team must pass every event at least once.
    'events_passed': [False, False, False, False, False],

    # Tactical bookkeeping (best-effort, for the operator HUD)
    'green_seen': 0,
    'red_seen': 0,

    # Member 3 — driving logic decision (debug)
    'target_lane': -1,
    'decision_reason': 'init',
}
data_lock = threading.Lock()
is_running = True


# ---------------------------------------------------------
# Real-Time Scheduling Framework
# ---------------------------------------------------------
class TaskPriority:
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class RTTask(threading.Thread):
    """
    Real-Time Task implementing:
      - Concurrency (inherits threading.Thread)
      - Task Period (enforced in run loop via next_release)
      - Task Priority (Windows thread priority class)
      - WCET measurement and deadline-miss counting
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
        except Exception:
            pass

        # Import lazily so other modules can mutate is_running without circular import.
        import rtos
        next_release = time.perf_counter()
        while rtos.is_running:
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
                # Already past the deadline — re-anchor so we don't accumulate drift.
                next_release = time.perf_counter()
