"""Abstract game interface plus a Mock and Real implementation.

The control system only ever talks to the abstract ``GameInterface``.
Swapping ``MockGameInterface`` for ``RealGameInterface`` is the only
change required to drive the Unity build at SpeedTrials2D/SpeedTrials2D.exe.

The real implementation matches the protocol documented in
``sample_drive.py`` and ``test_communication.py``:

* Two camera servers (front: 8080, back: 8082). We connect as client
  and read JPEG frames prefixed by a little-endian 4-byte length.
* One control endpoint (8081). The game connects as a client to us.
  Each command is ``struct.pack('ff', steering, acceleration)`` with
  both values in ``[-1.0, 1.0]``.

We deliberately keep network setup blocking on a dedicated worker
thread so the periodic perception task can run un-blocked the moment
frames start flowing.
"""

from __future__ import annotations

import math
import random
import socket
import struct
import sys
import threading
import time
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import config
from .state import GameState, Obstacle, Token, TokenColor


# ----------------------------------------------------------------------
# Helper: Reliable TCP socket reading
# ----------------------------------------------------------------------
def recv_exact(sock: socket.socket, size: int) -> bytes:
    """Receive exactly `size` bytes from socket or raise OSError if connection breaks.
    
    Args:
        sock: The socket to read from.
        size: Number of bytes to read.
        
    Returns:
        Exactly `size` bytes of data.
        
    Raises:
        OSError: If the connection closes before all bytes are received.
    """
    buf = bytearray()
    while len(buf) < size:
        chunk = sock.recv(size - len(buf))
        if not chunk:
            raise OSError(f"Connection closed after {len(buf)}/{size} bytes")
        buf.extend(chunk)
    return bytes(buf)


# ----------------------------------------------------------------------
# Abstract base
# ----------------------------------------------------------------------
class GameInterface(ABC):
    """Symmetric Perceive/Actuate contract between control system and game."""

    @abstractmethod
    def read_state(self) -> GameState:
        """Return the most recent perception snapshot.

        Implementations must be non-blocking once the connection is
        established (Perception runs at 50 Hz). If no frame is ready,
        return a snapshot with ``perception_healthy=False`` rather than
        sleeping.
        """

    @abstractmethod
    def send_command(self, steering: float, acceleration: float) -> None:
        """Push a low-level (steering, acceleration) pair to the game.

        Both values are clamped to ``[-1.0, 1.0]`` to match the protocol.
        """

    def start(self) -> None:
        """Optional connection setup. Default is a no-op."""

    def stop(self) -> None:
        """Optional teardown. Default is a no-op."""


# ----------------------------------------------------------------------
# Mock implementation: a self-contained, deterministic-by-seed game world
# ----------------------------------------------------------------------
class MockGameInterface(GameInterface):
    """A purely Python world that spawns tokens and lets the car move.

    Designed to exercise every code path in the RTOS layer without any
    external dependency. The forward motion is modelled as an
    ever-decreasing distance for already-spawned tokens; new tokens
    appear at the horizon at a configurable rate.
    """

    def __init__(
        self,
        seed: int = 42,
        token_spawn_hz: float = 6.0,
        obstacle_spawn_hz: float = 0.5,
        steady_speed_norm: float = 0.6,
    ) -> None:
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._tokens: List[Token] = []
        self._obstacles: List[Obstacle] = []
        self._own_lane = config.LANE_CENTER_INDEX
        self._lane_float = float(config.LANE_CENTER_INDEX)
        self._speed_norm = steady_speed_norm
        self._last_step_time: Optional[float] = None
        self._token_spawn_period = 1.0 / token_spawn_hz
        self._obstacle_spawn_period = 1.0 / max(obstacle_spawn_hz, 1e-6)
        self._next_token_spawn = 0.0
        self._next_obstacle_spawn = 0.0
        self._run_started_at: Optional[float] = None
        self._low_light_active = False
        self._low_light_started_at = 0.0
        self._low_light_applied = False
        self._rear_chase_count = 0
        self._rear_chase_active = False
        self._rear_chase_started_at = 0.0
        self._rear_chase_expires_at = 0.0
        self._rear_chase_lane = -1
        self._rear_pressure = 0.0
        self._chase_collision_done = False
        self._police_active = False
        self._police_started_at = 0.0
        self._police_expires_at = 0.0
        self._police_lane = -1
        self._police_collision_done = False
        self._police_appear_at = 0.0
        self._last_player_red_token = 0.0
        self._game_over = False
        self._game_over_reason = ""
        # Effect of the last steering command: positive = steering right.
        self._steering = 0.0
        self._acceleration = 1.0
        self._started = False

    # ---- lifecycle ------------------------------------------------
    def start(self) -> None:
        now = time.perf_counter()
        self._run_started_at = now
        self._last_step_time = now
        self._next_token_spawn = now
        self._next_obstacle_spawn = now + 1.0
        self._police_appear_at = config.POLICE_CAR_MIN_APPEAR_SEC + self._rng.random() * (
            config.POLICE_CAR_MAX_APPEAR_SEC - config.POLICE_CAR_MIN_APPEAR_SEC
        )
        self._started = True

    def stop(self) -> None:
        self._started = False

    # ---- world simulation -----------------------------------------
    def _advance_world(self) -> None:
        """Move tokens toward the car, spawn new ones, integrate steering."""
        now = time.perf_counter()
        if self._last_step_time is None:
            self._last_step_time = now
            return
        dt = now - self._last_step_time
        self._last_step_time = now

        self._update_low_light(now, dt)
        self._update_chase_state(now, dt)
        self._update_police_state(now, dt)

        if self._game_over:
            self._speed_norm = 0.0
            return

        # Lane integration: steering of +/-1 sweeps one lane in ~0.4 s.
        lane_rate = self._steering * 2.5
        self._lane_float = max(0.0, min(float(config.NUM_LANES - 1),
                                        self._lane_float + lane_rate * dt))
        self._own_lane = int(round(self._lane_float))

        # Speed integration: acceleration of +1 drives speed_norm toward 1.
        self._speed_norm = max(0.0, min(1.0, self._speed_norm + 0.4 * self._acceleration * dt - 0.05 * dt))
        if self._speed_norm < 0.05:
            self._speed_norm = 0.05  # never fully stop in mock so flow stays interesting
        if self._low_light_active:
            if self._low_light_reverse:
                # reverse effect = recovery / boost instead of penalty
                self._speed_norm = min(1.0, self._speed_norm + 0.2 * dt)
            else:
                self._speed_norm *= config.LOW_LIGHT_SPEED_PENALTY_FACTOR

        if self._rear_chase_active and self._rear_chase_lane == self._own_lane and not self._chase_collision_done:
            self._speed_norm *= config.CHASE_CAR_SPEED_PENALTY_FACTOR
            self._chase_collision_done = True
            self._rear_pressure = 1.0

        # Tokens drift toward us at a rate proportional to speed_norm.
        flow = max(0.05, self._speed_norm) * 0.8
        self._tokens = [
            Token(lane=t.lane, distance=t.distance - flow * dt, color=t.color)
            for t in self._tokens
            if t.distance - flow * dt > 0.0
        ]
        self._obstacles = [
            Obstacle(lane=o.lane, distance=o.distance - flow * dt)
            for o in self._obstacles
            if o.distance - flow * dt > 0.0
        ]

        # Spawn new tokens periodically at the horizon.
        while now >= self._next_token_spawn:
            color = self._rng.choices(
                [TokenColor.GREEN, TokenColor.RED, TokenColor.YELLOW],
                weights=[0.55, 0.35, 0.10],
                k=1,
            )[0]
            self._tokens.append(Token(
                lane=self._rng.randrange(config.NUM_LANES),
                distance=1.0,
                color=color,
            ))
            self._next_token_spawn += self._token_spawn_period

        while now >= self._next_obstacle_spawn:
            self._obstacles.append(Obstacle(
                lane=self._rng.randrange(config.NUM_LANES),
                distance=1.0,
            ))
            self._next_obstacle_spawn += self._obstacle_spawn_period

    def _update_low_light(self, now: float, dt: float) -> None:
        if self._run_started_at is None:
            self._run_started_at = now

        elapsed = now - self._run_started_at

        # --- 1. ENTRY condition (more realistic range) ---
        if not self._low_light_active:
            if 0.20 <= self._last_brightness <= 0.45:
                self._low_light_active = True
                self._low_light_started_at = now
                self._low_light_reverse = True

        # --- 2. EXIT condition (hysteresis) ---
        else:
            if self._last_brightness > 0.60:
                self._low_light_active = False
                self._low_light_reverse = False

        # --- 3. optional decay effect (keep your behavior) ---
        if self._low_light_active:
            self._rear_pressure = max(0.0, self._rear_pressure - 0.2)

    def _update_police_state(self, now: float, dt: float) -> None:
        if self._run_started_at is None:
            self._run_started_at = now
        elapsed = now - self._run_started_at
        if not self._police_active and not self._game_over:
            if elapsed >= self._police_appear_at:
                self._police_active = True
                self._police_started_at = now
                self._police_expires_at = now + config.POLICE_CAR_WINDOW_SEC
                self._police_lane = self._own_lane
                self._police_collision_done = False
                self._last_player_red_token = 0.0

        if not self._police_active:
            return

        if self._police_lane == self._own_lane and not self._police_collision_done:
            self._game_over = True
            self._game_over_reason = "police_car_collision"
            return

        if self._last_player_red_token > 0.0 and (now - self._last_player_red_token) <= 1.5:
            self._police_active = False
            self._police_lane = -1
            self._police_collision_done = True
            self._speed_norm *= config.POLICE_CAR_SPEED_PENALTY_FACTOR
            return

        if now >= self._police_expires_at:
            self._police_active = False
            self._police_lane = -1
            self._speed_norm *= config.POLICE_CAR_SPEED_PENALTY_FACTOR

    def _mark_red_token_taken(self, now: float) -> None:
        self._last_player_red_token = now

    def _update_chase_state(self, now: float, dt: float) -> None:
        """Trigger the two chase-car appearances and update pressure."""
        if self._run_started_at is None:
            self._run_started_at = now
        elapsed = now - self._run_started_at

        starts = [
            (config.CHASE_CAR_FIRST_APPEAR_SEC, config.CHASE_CAR_FIRST_WINDOW_SEC),
            (config.CHASE_CAR_SECOND_APPEAR_SEC, config.CHASE_CAR_SECOND_WINDOW_SEC),
        ]

        if self._rear_chase_count < len(starts):
            appear_at, window = starts[self._rear_chase_count]
            if elapsed >= appear_at and not self._rear_chase_active:
                self._rear_chase_active = True
                self._rear_chase_started_at = now
                self._rear_chase_expires_at = now + window
                self._rear_chase_lane = self._own_lane
                self._rear_pressure = 0.0
                self._chase_collision_done = False
                self._rear_chase_count += 1

        if not self._rear_chase_active:
            self._rear_pressure = max(0.0, self._rear_pressure - 0.3 * dt)
            return

        time_left = self._rear_chase_expires_at - now
        if time_left <= 0.0:
            self._rear_chase_active = False
            self._rear_chase_lane = -1
            self._rear_pressure = 0.0
            self._chase_collision_done = False
            return

        self._rear_pressure = min(
            1.0,
            self._rear_pressure + config.CHASE_CAR_PRESSURE_RISE_PER_SEC * dt,
        )

    # ---- GameInterface impl ---------------------------------------
    def read_state(self) -> GameState:
        if not self._started:
            return GameState.empty()
        with self._lock:
            self._advance_world()
            # Defensive copies into tuples to keep GameState immutable.
            return GameState(
                timestamp=time.perf_counter(),
                own_lane=self._own_lane,
                speed_norm=self._speed_norm,
                brightness=0.1 if self._low_light_active else 1.0,
                low_light_active=self._low_light_active,
                rear_pressure=self._rear_pressure,
                rear_chase_active=self._rear_chase_active,
                rear_chase_lane=self._rear_chase_lane,
                rear_time_left=max(0.0, self._rear_chase_expires_at - time.perf_counter())
                    if self._rear_chase_active else 0.0,
                police_alert=self._police_active,
                police_lane=self._police_lane,
                police_time_left=max(0.0, self._police_expires_at - time.perf_counter())
                    if self._police_active else 0.0,
                game_over=self._game_over,
                game_over_reason=self._game_over_reason,
                tokens=tuple(self._tokens),
                obstacles=tuple(self._obstacles),
                perception_healthy=True,
            )

    def send_command(self, steering: float, acceleration: float) -> None:
        with self._lock:
            self._steering = max(-1.0, min(1.0, steering))
            self._acceleration = max(-1.0, min(1.0, acceleration))


# ----------------------------------------------------------------------
# Real implementation: TCP to the Unity build
# ----------------------------------------------------------------------
class RealGameInterface(GameInterface):
    """Drives the SpeedTrials2D Unity executable over TCP.

    OpenCV is imported lazily so the Mock-only path never pays the import
    cost (and so the project still works on machines without OpenCV
    installed).
    """

    def __init__(self, show_overlay: bool = True) -> None:
        # Networking
        self._front_sock: Optional[socket.socket] = None
        self._back_sock: Optional[socket.socket] = None
        self._back_sock: Optional[socket.socket] = None
        self._control_server: Optional[socket.socket] = None
        self._control_conn: Optional[socket.socket] = None
        self._setup_thread: Optional[threading.Thread] = None
        self._running = False

        # Latest decoded perception fields (written by the reader thread,
        # read by ``read_state``; protected by ``_perception_lock``).
        self._perception_lock = threading.Lock()
        self._latest_state: GameState = GameState.empty()

        # Reader thread keeps the camera socket drained at the speed the
        # game produces frames. ``read_state`` then returns a cached
        # snapshot in O(1).
        self._reader_thread: Optional[threading.Thread] = None
        self._back_reader_thread: Optional[threading.Thread] = None
        self._perception_thread: Optional[threading.Thread] = None
        self._vehicle_thread: Optional[threading.Thread] = None
        self._overlay_thread: Optional[threading.Thread] = None

        self._front_frame_lock = threading.Lock()
        self._front_frame: Optional[Any] = None
        self._front_frame_ts: float = 0.0
        self._back_frame_lock = threading.Lock()
        self._back_frame: Optional[Any] = None
        self._back_frame_ts: float = 0.0

        self._detection_lock = threading.Lock()
        self._last_enriched: List[Tuple[Token, Tuple[int, int, int, int]]] = []
        self._last_chasing_car: Dict[str, Any] = {
            "detected": False,
            "score": 0.0,
            "lane": -1,
            "bbox": None,
        }
        self._last_police_car: Dict[str, Any] = {
            "detected": False,
            "score": 0.0,
            "lane": -1,
            "bbox": None,
        }
        self._last_brightness = 1.0
        self._last_vehicle_detection_at = 0.0

        # Lane-of-self tracking (we never get told, so we integrate from
        # our own steering commands). 0 .. NUM_LANES-1.
        self._own_lane_float = float(config.LANE_CENTER_INDEX)
        self._last_command_at: Optional[float] = None
        self._last_steering = 0.0
        self._last_acceleration = 0.0
        
        # ---------------- Visual Effects ----------------
        self._yellow_effect_active = False
        self._yellow_effect_start = 0.0
        self._yellow_effect_duration = 2.5  # seconds
        self._yellow_flash_state = False
        self._yellow_last_toggle = 0.0

        self._low_light_reverse = False  # "reverse mode"

        # Optional debug visualization. When on, the camera reader thread
        # draws an annotated frame in an OpenCV window. Off-by-default for
        # Mock mode (this class is never used in mock); on-by-default for
        # Real mode so the operator can verify perception live. The
        # window is pinned always-on-top so it stays visible over the
        # Unity game window (since we cannot modify the Unity build to
        # draw the HUD inside it).
        self._show_overlay = show_overlay
        self._overlay_window = "RTOS Perception (RTSE)"
        self._overlay_back_window = "RTOS Perception (RTSE) - Rear"
        self._overlay_window_ready = False
        self._overlay_back_ready = False

        # Template images for vehicle detection (lazy-loaded by _load_templates).
        self._chasing_templates: Optional[List] = None
        self._police_templates: Optional[List] = None
        self._templates_loaded = False

        self._diagnostic_lock = threading.Lock()
        self._diagnostics: Dict[str, int] = {
            "front_frames": 0,
            "back_frames": 0,
            "decode_failures": 0,
            "socket_errors": 0,
            "perception_cycles": 0,
            "overlay_frames": 0,
            "token_detections": 0,
            "vehicle_detections": 0,
        }
        self._performance_metrics: Dict[str, float] = {
            "token_latency_ns": 0.0,
            "token_count": 0.0,
            "vehicle_latency_ns": 0.0,
            "vehicle_count": 0.0,
            "perception_latency_ns": 0.0,
            "perception_count": 0.0,
            "overlay_latency_ns": 0.0,
            "overlay_count": 0.0,
        }

    def _load_templates(self) -> None:
        """Lazy-load template images on first use (when cv2 is available)."""
        if self._templates_loaded:
            return
        try:
            import cv2  # type: ignore
            asset_dir = Path(__file__).parent / "assets"
            chasing_raw = [
                cv2.imread(str(asset_dir / "chasing_car_front.png")),
                cv2.imread(str(asset_dir / "chasing_car_back.png")),
            ]
            police_raw = [
                cv2.imread(str(asset_dir / "police_car_front.png")),
                cv2.imread(str(asset_dir / "police_car_back.png")),
            ]
            self._chasing_templates = [
                gray
                for image in chasing_raw
                if image is not None
                for gray in self._build_template_pyramid(image, cv2)
            ]
            self._police_templates = [
                gray
                for image in police_raw
                if image is not None
                for gray in self._build_template_pyramid(image, cv2)
            ]
        except ImportError:
            print("[RealGameInterface] OpenCV not available; template detection disabled.")
            self._chasing_templates = []
            self._police_templates = []
        finally:
            self._templates_loaded = True

    def _build_template_pyramid(self, image: Any, cv2: Any) -> List[Any]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scales = getattr(config, "REAL_GAME_TEMPLATE_SCALES", (1.2, 1.0, 0.8))
        pyramid: List[Any] = []
        for scale in scales:
            if scale == 1.0:
                resized = gray
            else:
                resized = cv2.resize(
                    gray,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
                )
            if resized.shape[0] < 16 or resized.shape[1] < 16:
                continue
            pyramid.append(resized)
        return pyramid

    def _template_detect(
        self,
        gray_frame,
        templates: List,
        cv2,
        threshold: float = 0.65,
    ) -> Tuple[bool, float, Optional[Tuple[int, int, int, int]]]:
        best_score = 0.0
        best_bbox = None

        for template in templates:
            if template is None:
                continue
            if template.shape[0] > gray_frame.shape[0] or template.shape[1] > gray_frame.shape[1]:
                continue

            result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                best_bbox = (max_loc[0], max_loc[1], template.shape[1], template.shape[0])

        return best_score >= threshold, best_score, best_bbox

    def _detect_vehicle(
        self,
        frame,
        templates: Optional[List],
        cv2,
        threshold: float = 0.70,
    ) -> Dict[str, any]:
        if not templates or all(t is None for t in templates):
            return {"detected": False, "score": 0.0, "lane": -1, "bbox": None}

        x1 = int(frame.shape[1] * config.REAL_GAME_REAR_VEHICLE_ROI_X_FRAC[0])
        x2 = int(frame.shape[1] * config.REAL_GAME_REAR_VEHICLE_ROI_X_FRAC[1])
        y1 = int(frame.shape[0] * config.REAL_GAME_REAR_VEHICLE_ROI_Y_FRAC[0])
        y2 = int(frame.shape[0] * config.REAL_GAME_REAR_VEHICLE_ROI_Y_FRAC[1])
        roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        detected, score, bbox = self._template_detect(gray_roi, templates, cv2, threshold=threshold)
        lane = -1
        if detected and bbox is not None:
            x, y, w, h = bbox
            x_full = x + x1
            y_full = y + y1
            center_x = x_full + w // 2
            lane_bounds = [int(frame.shape[1] * f) for f in config.LANE_X_BOUNDS_FRAC]
            lane = self._lane_for_x(center_x, lane_bounds)
            bbox = (x_full, y_full, w, h)

        return {"detected": detected, "score": score, "lane": lane, "bbox": bbox}

    def _detect_chasing_car(
        self,
        frame,
        cv2
    ) -> Dict[str, any]:
        """Detect chase car in frame."""
        self._load_templates()
        return self._detect_vehicle(frame, self._chasing_templates, cv2, threshold=0.70)

    def _detect_police_car(
        self,
        frame,
        cv2
    ) -> Dict[str, any]:
        """Detect police car in frame."""
        self._load_templates()
        return self._detect_vehicle(frame, self._police_templates, cv2, threshold=0.70)
    
    # ---- lifecycle ------------------------------------------------
    def start(self) -> None:
        self._running = True
        self._setup_thread = threading.Thread(
            target=self._setup_network,
            name="RealGameSetup",
            daemon=True,
        )
        self._setup_thread.start()

        self._reader_thread = threading.Thread(
            target=lambda: self._camera_reader_loop("front"),
            name="FrontCameraReader",
            daemon=True,
        )
        self._reader_thread.start()

        self._back_reader_thread = threading.Thread(
            target=lambda: self._camera_reader_loop("back"),
            name="BackCameraReader",
            daemon=True,
        )
        self._back_reader_thread.start()

        self._perception_thread = threading.Thread(
            target=self._perception_loop,
            name="RealGamePerception",
            daemon=True,
        )
        self._perception_thread.start()

        self._vehicle_thread = threading.Thread(
            target=self._vehicle_detection_loop,
            name="RealGameVehicleDetection",
            daemon=True,
        )
        self._vehicle_thread.start()

        if self._show_overlay:
            self._overlay_thread = threading.Thread(
                target=self._overlay_loop,
                name="RealGameOverlay",
                daemon=True,
            )
            self._overlay_thread.start()

    def stop(self) -> None:
        self._running = False
        for sock in (self._front_sock, self._back_sock, self._control_conn, self._control_server):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

        for thread in (
            self._reader_thread,
            self._back_reader_thread,
            self._perception_thread,
            self._vehicle_thread,
            self._overlay_thread,
            self._setup_thread,
        ):
            if thread is not None and thread.is_alive():
                thread.join(timeout=0.2)

        if self._show_overlay:
            try:
                import cv2  # type: ignore
                cv2.destroyWindow(self._overlay_window)
                if self._overlay_back_ready:
                    cv2.destroyWindow(self._overlay_back_window)
                cv2.waitKey(1)
            except Exception:
                pass

    # ---- networking ------------------------------------------------
    def _setup_network(self) -> None:
        """Connect to both cameras and accept a control connection."""
        # Front camera
        while self._running and self._front_sock is None:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((config.GAME_CAMERA_HOST, config.GAME_FRONT_CAMERA_PORT))
                s.settimeout(None)
                self._front_sock = s
                print("[RealGameInterface] Front camera connected.")
            except OSError:
                time.sleep(0.5)

        # Back camera (optional)
        while self._running and self._back_sock is None:
            try:
                s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s2.settimeout(1.0)
                s2.connect((config.GAME_CAMERA_HOST, config.GAME_BACK_CAMERA_PORT))
                s2.settimeout(None)
                self._back_sock = s2
                print("[RealGameInterface] Back camera connected.")
            except OSError:
                time.sleep(0.5)

        # Control server
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((config.GAME_CONTROL_HOST, config.GAME_CONTROL_PORT))
            srv.listen(1)
            srv.settimeout(1.0)
            self._control_server = srv
            print(f"[RealGameInterface] Control server listening on "
                  f"{config.GAME_CONTROL_HOST}:{config.GAME_CONTROL_PORT}")
            while self._running and self._control_conn is None:
                try:
                    conn, addr = srv.accept()
                    self._control_conn = conn
                    print(f"[RealGameInterface] Control client connected from {addr}")
                except socket.timeout:
                    continue
                except OSError:
                    break
        except OSError:
            self._mark_unhealthy()

    def _camera_reader_loop(self, which: Optional[str] = None) -> None:
        """Drain JPEG frames as fast as the game produces them.

        This thread is only responsible for decoding and publishing the
        newest frame. Perception and overlay work run in separate threads
        so expensive detections do not block frame acquisition.
        """
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            print("[RealGameInterface] OpenCV/numpy not installed; perception will be degraded.")
            return

        if which is None:
            which = "front"
            try:
                tname = threading.current_thread().name.lower()
                if "back" in tname:
                    which = "back"
            except Exception:
                pass

        sock_attr = "_front_sock" if which == "front" else "_back_sock"
        while self._running and getattr(self, sock_attr) is None:
            time.sleep(0.05)

        sock = getattr(self, sock_attr)
        while self._running and sock is not None:
            try:
                length_bytes = recv_exact(sock, 4)
                image_length = int.from_bytes(length_bytes, "little")
                buf = recv_exact(sock, image_length)
                np_arr = np.frombuffer(buf, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    self._mark_unhealthy()
                    continue

                max_width = getattr(config, "REAL_GAME_MAX_FRAME_WIDTH", 960)
                if frame.shape[1] > max_width:
                    scale = float(max_width) / frame.shape[1]
                    frame = cv2.resize(
                        frame,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_AREA,
                    )

                now = time.perf_counter()
                if which == "front":
                    with self._front_frame_lock:
                        self._front_frame = frame
                        self._front_frame_ts = now
                    with self._diagnostic_lock:
                        self._diagnostics["front_frames"] += 1
                else:
                    with self._back_frame_lock:
                        self._back_frame = frame
                        self._back_frame_ts = now
                    with self._diagnostic_lock:
                        self._diagnostics["back_frames"] += 1
            except OSError:
                with self._diagnostic_lock:
                    self._diagnostics["socket_errors"] += 1
                self._mark_unhealthy()
                break

    def _mark_unhealthy(self) -> None:
        """Mark perception as unhealthy when socket errors occur."""
        with self._perception_lock:
            self._latest_state = GameState(
                timestamp=time.perf_counter(),
                own_lane=int(round(self._own_lane_float)),
                speed_norm=max(0.0, min(1.0, self._last_acceleration)),
                brightness=1.0,
                low_light_active=False,
                rear_pressure=0.0,
                rear_chase_active=False,
                rear_chase_lane=-1,
                rear_time_left=0.0,
                police_alert=False,
                police_lane=-1,
                police_time_left=0.0,
                game_over=False,
                game_over_reason="",
                tokens=(),
                obstacles=(),
                perception_healthy=False,
            )

    def _estimate_brightness(self, frame, cv2, np) -> Tuple[float, bool]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray)) / 255.0

        rows, cols = getattr(config, "LOW_LIGHT_GRID_SIZE", (5, 5))
        region_means = []
        h, w = gray.shape
        for ry in range(rows):
            y0 = int(h * ry / rows)
            y1 = int(h * (ry + 1) / rows) if ry < rows - 1 else h
            for cx in range(cols):
                x0 = int(w * cx / cols)
                x1 = int(w * (cx + 1) / cols) if cx < cols - 1 else w
                region = gray[y0:y1, x0:x1]
                region_means.append(float(np.mean(region)) / 255.0)

        region_means_np = np.asarray(region_means, dtype=np.float32)
        dark_ratio = float(np.count_nonzero(region_means_np < config.LOW_LIGHT_THRESHOLD)) / region_means_np.size
        uniformity_std = float(np.std(region_means_np))
        low_light_active = (
            mean_brightness < config.LOW_LIGHT_THRESHOLD
            and dark_ratio >= getattr(config, "LOW_LIGHT_DARK_RATIO", 0.85)
            and uniformity_std < getattr(config, "LOW_LIGHT_UNIFORMITY_STD", 0.05)
        )
        return mean_brightness, low_light_active

    def _perception_loop(self) -> None:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            print("[RealGameInterface] OpenCV/numpy not installed; perception will be degraded.")
            return

        period = 1.0 / max(1.0, getattr(config, "REAL_GAME_TOKEN_DETECTION_HZ", 25.0))
        while self._running:
            start = time.perf_counter()
            front_frame = self._get_latest_frame("front")
            if front_frame is not None:
                try:
                    frame_start = time.perf_counter()
                    brightness, low_light_active = self._estimate_brightness(front_frame, cv2, np)
                    self._last_brightness = brightness
                    enriched = self._detect_tokens(front_frame, cv2, np)
                    now = time.perf_counter()

                    # detect yellow tokens
                    has_yellow = any(tok.color == TokenColor.YELLOW for tok, _ in enriched)

                    # trigger flashing effect
                    if has_yellow:
                        self._yellow_effect_active = True
                        self._yellow_effect_start = now
                        self._yellow_last_toggle = now
                    token_end = time.perf_counter()
                    # expire yellow effect
                    if self._yellow_effect_active:
                        if now - self._yellow_effect_start > self._yellow_effect_duration:
                            self._yellow_effect_active = False
                        else:
                            # toggle flash every 0.25s
                            if now - self._yellow_last_toggle > 0.25:
                                self._yellow_flash_state = not self._yellow_flash_state
                                self._yellow_last_toggle = now
                    with self._diagnostic_lock:
                        self._diagnostics["token_detections"] += 1
                        self._performance_metrics["token_latency_ns"] += (token_end - frame_start) * 1e9
                        self._performance_metrics["token_count"] += 1

                    with self._detection_lock:
                        self._last_enriched = enriched
                        self._last_brightness = brightness
                        chasing_car = dict(self._last_chasing_car)
                        police_car = dict(self._last_police_car)

                    tokens = tuple(item[0] for item in enriched)
                    self._update_state_from_perception(
                        front_frame.shape,
                        brightness,
                        tokens,
                        chasing_car,
                        police_car,
                        low_light_active=low_light_active,
                    )
                    perception_end = time.perf_counter()
                    with self._diagnostic_lock:
                        self._diagnostics["perception_cycles"] += 1
                        self._performance_metrics["perception_latency_ns"] += (perception_end - start) * 1e9
                        self._performance_metrics["perception_count"] += 1
                except Exception:
                    self._mark_unhealthy()

            sleep_time = period - (time.perf_counter() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _vehicle_detection_loop(self) -> None:
        try:
            import cv2  # type: ignore
        except ImportError:
            return

        period = 1.0 / max(1.0, getattr(config, "REAL_GAME_VEHICLE_DETECTION_HZ", 4.0))
        while self._running:
            start = time.perf_counter()
            front_frame = self._get_latest_frame("front")
            if front_frame is not None:
                try:
                    vehicle_start = time.perf_counter()
                    chasing_car = self._detect_chasing_car(front_frame, cv2)
                    police_car = self._detect_police_car(front_frame, cv2)
                    vehicle_end = time.perf_counter()
                    with self._detection_lock:
                        self._last_chasing_car = chasing_car
                        self._last_police_car = police_car
                        self._last_vehicle_detection_at = vehicle_end
                    with self._diagnostic_lock:
                        self._diagnostics["vehicle_detections"] += 1
                        self._performance_metrics["vehicle_latency_ns"] += (vehicle_end - vehicle_start) * 1e9
                        self._performance_metrics["vehicle_count"] += 1
                except Exception:
                    pass

            sleep_time = period - (time.perf_counter() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _overlay_loop(self) -> None:
        try:
            import cv2  # type: ignore
        except ImportError:
            return

        period = 1.0 / max(1.0, config.REAL_GAME_OVERLAY_FPS)
        while self._running:
            start = time.perf_counter()
            front_frame = self._get_latest_frame("front")
            back_frame = self._get_latest_frame("back")
            with self._detection_lock:
                enriched = list(self._last_enriched)
                chasing_car = dict(self._last_chasing_car)
                police_car = dict(self._last_police_car)
                cached_state = self.read_state()

            if front_frame is not None:
                try:
                    overlay_start = time.perf_counter()
                    self._render_overlay_front(
                        front_frame,
                        enriched,
                        chasing_car,
                        police_car,
                        cv2,
                        cached_state,
                    )
                    overlay_end = time.perf_counter()
                    with self._diagnostic_lock:
                        self._diagnostics["overlay_frames"] += 1
                        self._performance_metrics["overlay_latency_ns"] += (overlay_end - overlay_start) * 1e9
                        self._performance_metrics["overlay_count"] += 1
                except Exception:
                    pass

            if back_frame is not None:
                try:
                    self._render_overlay_back(back_frame, chasing_car, police_car, cv2)
                except Exception:
                    pass

            sleep_time = period - (time.perf_counter() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _get_latest_frame(self, which: str) -> Optional[Any]:
        if which == "front":
            lock = self._front_frame_lock
            frame = self._front_frame
        else:
            lock = self._back_frame_lock
            frame = self._back_frame

        with lock:
            return frame.copy() if frame is not None else None

    @staticmethod
    def _detect_tokens(frame, cv2, np) -> List[Tuple]:
        """Lane-aware HSV token detector. Mirrors sample_drive.py but
        bins detections into lanes/distances normalized to [0, 1].

        Args:
            frame: The video frame.
            cv2: OpenCV module.
            np: NumPy module.

        Returns:
            Enriched list of (Token, (x, y, w, h)) tuples so the
            overlay renderer can draw the original bounding boxes.
        """
        h, w = frame.shape[:2]
        rx1 = int(w * config.ROI_X_FRAC[0])
        rx2 = int(w * config.ROI_X_FRAC[1])
        ry1 = int(h * config.ROI_Y_FRAC[0])
        ry2 = int(h * config.ROI_Y_FRAC[1])
        roi = frame[ry1:ry2, rx1:rx2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        ranges = {
            TokenColor.GREEN: [
                (np.array([40, 80, 80]),  np.array([85, 255, 255])),
            ],
            TokenColor.YELLOW: [
                (np.array([20, 80, 80]),  np.array([35, 255, 255])),
            ],
            TokenColor.RED: [
                (np.array([0, 80, 80]),   np.array([10, 255, 255])),
                (np.array([170, 80, 80]), np.array([180, 255, 255])),
            ],
        }

        lane_bounds = [int(w * f) for f in config.LANE_X_BOUNDS_FRAC]
        out = []
        for color, bands in ranges.items():
            mask = None
            for lo, hi in bands:
                m = cv2.inRange(hsv, lo, hi)
                mask = m if mask is None else cv2.bitwise_or(mask, m)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 300 or area > 8000:
                    continue
                x, y, ww, hh = cv2.boundingRect(cnt)
                x_full = x + rx1
                y_full = y + ry1
                cx_full = x_full + ww // 2
                cy_full = y_full + hh // 2
                lane = RealGameInterface._lane_for_x(cx_full, lane_bounds)
                if lane is None:
                    continue
                # Larger cy_full == closer to bottom of frame == closer to car.
                # Map cy_full in [ry1, ry2] to distance in [1, 0].
                dist_norm = 1.0 - (cy_full - ry1) / max(1, (ry2 - ry1))
                dist_norm = max(0.0, min(1.0, dist_norm))
                tok = Token(lane=lane, distance=dist_norm, color=color)
                out.append((tok, (x_full, y_full, ww, hh)))
        return out
    
    def _draw_detection(
        self,
        frame,
        detection: Dict[str, any],
        label: str,
        color: Tuple[int, int, int],
        cv2,
    ) -> None:
        """Draw a labeled bounding box for a detection result.
        
        Args:
            frame: The image to draw on (modified in-place).
            detection: Detection result dict with 'detected', 'score', 'bbox'.
            label: Text label to display.
            color: BGR color tuple.
            cv2: OpenCV module.
        """
        if not detection["detected"]:
            return

        bbox = detection["bbox"]

        if bbox is None:
            return

        x, y, w, h = bbox

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            3,
        )

        text = f"{label} ({detection['score']:.2f})"

        cv2.putText(
            frame,
            text,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    def _update_state_from_perception(
        self,
        frame_shape: Tuple[int, ...],
        brightness: float,
        tokens: Tuple[Token, ...],
        chasing_car: Dict[str, any],
        police_car: Dict[str, any],
        low_light_active: bool,
    ) -> None:
        """Update cached GameState from perceived data.
        
        Args:
            frame_shape: Shape of the frame (h, w, channels).
            brightness: Normalized brightness [0, 1].
            tokens: Tuple of detected tokens.
            chasing_car: Chase car detection result.
            police_car: Police car detection result.
            low_light_active: Whether the frame is uniformly dark.
        """
        # Integrate own lane from steering: +1 = right at LANE_HOLD_TIME pace.
        now = time.perf_counter()
        if self._last_command_at is not None:
            dt = now - self._last_command_at
            self._own_lane_float = max(
                0.0,
                min(float(config.NUM_LANES - 1),
                    self._own_lane_float + self._last_steering * 2.5 * dt),
            )
        self._last_command_at = now

        with self._perception_lock:
            self._latest_state = GameState(
                timestamp=now,
                own_lane=int(round(self._own_lane_float)),
                # We have no telemetry for actual speed; use the requested
                # throttle as a proxy. Decision uses this only to widen
                # look-ahead, so the proxy is good enough.
                speed_norm=max(0.0, min(1.0, self._last_acceleration)),
                brightness=brightness,
                    low_light_active=low_light_active,
                game_over_reason="",
                tokens=tokens,
                obstacles=(),  # Phase-1 token game has no obstacles per se
                perception_healthy=True,
            )

    # ---- overlay --------------------------------------------------
    def _render_overlay_front(
        self,
        frame,
        enriched: List[Tuple],
        chasing_car: Dict[str, any],
        police_car: Dict[str, any],
        cv2,
        cached_state: GameState,
    ) -> None:
        """Draw an annotated copy of the front camera frame in a live OpenCV window.

        Layers (bottom -> top):
          1. Lane dividers (vertical gray lines)
          2. ROI rectangle (white outline)
          3. BRAKE_DIST reference line (red horizontal): tokens below
             this line are inside the imminent-danger window.
          4. LOOKAHEAD reference line (cyan horizontal): tokens above
             this line are outside the reward-evaluation window.
          5. Per-token bounding box, color-coded:
                RED   -> red box + outward AVOID arrow
                GREEN -> green box + TARGET arrow from the car
                YELLOW-> yellow box + WARN label
             A flashing thick border is added when a RED is in the
             center lane (in front of us) within BRAKE_DIST.
          6. Top status bar: token counts + current actuation label
             (derived from the latest steering / acceleration sent).
        """
        h, w = frame.shape[:2]
        rx1 = int(w * config.ROI_X_FRAC[0])
        rx2 = int(w * config.ROI_X_FRAC[1])
        ry1 = int(h * config.ROI_Y_FRAC[0])
        ry2 = int(h * config.ROI_Y_FRAC[1])
        lane_bounds = [int(w * f) for f in config.LANE_X_BOUNDS_FRAC]
        roi_height = max(1, ry2 - ry1)
            
        out = frame.copy()
        # ---------------- YELLOW FLASH EFFECT ----------------
        if self._yellow_effect_active and self._yellow_flash_state:
            h, w = out.shape[:2]

            # left + right black panels
            cv2.rectangle(out, (0, 0), (w // 5, h), (0, 0, 0), -1)
            cv2.rectangle(out, (w - w // 5, 0), (w, h), (0, 0, 0), -1)
        
        
        # 1. Lane dividers
        for x in lane_bounds:
            cv2.line(out, (x, ry1), (x, ry2), (180, 180, 180), 1, cv2.LINE_AA)
            # Lane labels below the ROI
        for li in range(len(lane_bounds) - 1):
            cx = (lane_bounds[li] + lane_bounds[li + 1]) // 2
            cv2.putText(out, f"L{li}", (cx - 12, ry2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        # 2. ROI rectangle
        cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (240, 240, 240), 1)

        # 3. BRAKE line (red, thick) — tokens with distance <= BRAKE_DIST
        #    map to y between brake_y and ry2.
        brake_y = int(ry2 - roi_height * config.BRAKE_DIST)
        cv2.line(out, (rx1, brake_y), (rx2, brake_y), (0, 0, 230), 2, cv2.LINE_AA)
        cv2.putText(out, "BRAKE", (rx1 + 4, brake_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 230), 1, cv2.LINE_AA)

        # 4. LOOKAHEAD line (cyan) at the base lookahead distance.
        look_y = int(ry2 - roi_height * config.LOOKAHEAD_BASE)
        cv2.line(out, (rx1, look_y), (rx2, look_y), (200, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(out, "LOOKAHEAD", (rx1 + 4, look_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1, cv2.LINE_AA)

        # 5. Token bounding boxes
        bgr_for = {
            TokenColor.GREEN:  (40, 220, 40),
            TokenColor.RED:    (40, 40, 230),
            TokenColor.YELLOW: (0, 220, 230),
        }
        car_anchor = (w // 2, h - 10)   # where the player's car sits
        counts = {TokenColor.GREEN: 0, TokenColor.RED: 0, TokenColor.YELLOW: 0}
        for tok, (bx, by, bw, bh) in enriched:
            color = bgr_for[tok.color]
            counts[tok.color] += 1
            cx = bx + bw // 2
            cy = by + bh // 2
            thickness = 2
            # Hazard flash for reds inside BRAKE in any lane.
            if tok.color is TokenColor.RED and tok.distance <= config.BRAKE_DIST:
                thickness = 4
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), color, thickness)
            label = f"{tok.color.value.upper()[0]} L{tok.lane} d{tok.distance:.2f}"
            cv2.putText(out, label, (bx, max(by - 6, ry1 + 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            if tok.color is TokenColor.RED:
                # AVOID arrow: push outward away from frame center.
                tip_x = cx + (60 if cx >= w // 2 else -60)
                cv2.arrowedLine(out, (cx, cy), (tip_x, cy), color, 2,
                                line_type=cv2.LINE_AA, tipLength=0.35)
                cv2.putText(out, "AVOID", (cx - 25, cy + bh // 2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            elif tok.color is TokenColor.GREEN:
                # TARGET arrow: from the car up to the token.
                cv2.arrowedLine(out, car_anchor, (cx, cy), color, 2,
                                line_type=cv2.LINE_AA, tipLength=0.10)
                cv2.putText(out, "TARGET", (cx - 28, cy - bh // 2 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            else:  # YELLOW
                cv2.putText(out, "WARN", (cx - 18, cy + bh // 2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # 6. Status bar (top)
        bar_h = 28
        cv2.rectangle(out, (0, 0), (w, bar_h), (28, 28, 28), -1)
        brightness = getattr(cached_state, "brightness", 1.0)
        status = (f"R={counts[TokenColor.RED]} "
                  f"G={counts[TokenColor.GREEN]} "
                  f"Y={counts[TokenColor.YELLOW]}    "
                  f"B={brightness:.2f} "
                  f"TH={config.LOW_LIGHT_THRESHOLD:.2f}    "
                  f"ACT={self._actuation_label()}    "
                  f"{self._status_label(cached_state)}")
        cv2.putText(out, status, (10, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)

        if not self._overlay_window_ready:
            self._init_overlay_window(cv2)
        cv2.imshow(self._overlay_window, out)
        cv2.waitKey(1)
        # Re-pin TOPMOST in case the user clicked Unity and stole focus.
        self._pin_topmost_win32()

    def _render_overlay_back(
        self,
        frame,
        chasing_car: Dict[str, any],
        police_car: Dict[str, any],
        cv2,
    ) -> None:
        """Draw rear view overlay with vehicle detection indicators.
        
        Args:
            frame: The rear camera frame.
            chasing_car: Detection result for chase car.
            police_car: Detection result for police car.
            cv2: OpenCV module.
        """
        out = frame.copy()
        cv2.putText(
            out, "REAR VIEW", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA
        )

        self._draw_detection(out, chasing_car, "CHASE", (255, 0, 255), cv2)
        self._draw_detection(out, police_car, "POLICE", (255, 255, 0), cv2)

        if not getattr(self, '_overlay_back_ready', False):
            try:
                cv2.namedWindow(self._overlay_back_window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self._overlay_back_window, 480, 360)
                cv2.moveWindow(self._overlay_back_window, 760, 20)
            except Exception:
                pass
            self._overlay_back_ready = True

        cv2.imshow(self._overlay_back_window, out)
        cv2.waitKey(1)
        self._pin_topmost_win32(self._overlay_back_window)

    def _init_overlay_window(self, cv2) -> None:
        """One-time setup: create the window and try to pin always-on-top.

        We prefer the OpenCV-native property if the installed build
        supports it; otherwise we fall back to Win32 ``SetWindowPos``.
        Either way, _pin_topmost_win32() re-pins on every frame so the
        window cannot be hidden behind the Unity game.
        """
        try:
            cv2.namedWindow(self._overlay_window, cv2.WINDOW_NORMAL)
            # 720x540 is a reasonable starting size for a 720p stream.
            cv2.resizeWindow(self._overlay_window, 720, 540)
            # Park it in the top-right so it doesn't cover Unity content.
            cv2.moveWindow(self._overlay_window, 20, 20)
        except Exception:
            pass
        try:
            # OpenCV >= 4.5.4 exposes a TOPMOST window property.
            cv2.setWindowProperty(self._overlay_window,
                                  cv2.WND_PROP_TOPMOST, 1.0)
        except Exception:
            pass
        self._overlay_window_ready = True

    def _pin_topmost_win32(self, window_name: Optional[str] = None) -> None:
        """Force the overlay to stay above all other windows (incl. Unity)."""
        if not sys.platform.startswith("win"):
            return
        try:
            import ctypes
            user32 = ctypes.windll.user32
            # Allow pinning either the main overlay or a named window.
            # If `window_name` is None, fall back to the main overlay.
            target = window_name or self._overlay_window
            hwnd = user32.FindWindowW(None, target)
            if not hwnd:
                return
            HWND_TOPMOST = -1
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_NOACTIVATE = 0x0010
            user32.SetWindowPos(
                hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
            )
        except Exception:
            pass

    def _actuation_label(self) -> str:
        """Map the last (steering, accel) we sent into a short HUD tag."""
        s = self._last_steering
        a = self._last_acceleration
        parts = []
        if a < 0:
            parts.append("BRAKE")
        elif a < 0.5:
            parts.append("SLOW")
        if abs(s) > 0.1:
            parts.append("LEFT" if s < 0 else "RIGHT")
        if not parts:
            parts.append("CRUISE")
        return "|".join(parts)

    def _status_label(self, cached_state: GameState) -> str:
        """Add a compact status tag for challenge states.
        
        Args:
            cached_state: The cached GameState to check for conditions.
        """
        if getattr(cached_state, "brightness", 1.0) < config.LOW_LIGHT_THRESHOLD:
            return "LOW_LIGHT"
        return "NORMAL"

    @staticmethod
    def _lane_for_x(x: int, lane_bounds: List[int]) -> Optional[int]:
        """Determine the lane index for a given x-coordinate.
        
        Args:
            x: The x-coordinate.
            lane_bounds: List of lane boundary x-coordinates (NUM_LANES+1 elements).
            
        Returns:
            Lane index or None if x is outside all lanes.
        """
        # lane_bounds has NUM_LANES+1 elements (left and right edges of each lane)
        if x < lane_bounds[0] or x >= lane_bounds[-1]:
            return None
        for i in range(len(lane_bounds) - 1):
            if lane_bounds[i] <= x < lane_bounds[i + 1]:
                return i
        return None

    # ---- GameInterface impl ---------------------------------------
    def read_state(self) -> GameState:
        with self._perception_lock:
            return self._latest_state

    def send_command(self, steering: float, acceleration: float) -> None:
        self._last_steering = max(-1.0, min(1.0, steering))
        self._last_acceleration = max(-1.0, min(1.0, acceleration))
        if self._control_conn is None:
            return
        try:
            payload = struct.pack("ff", self._last_steering, self._last_acceleration)
            self._control_conn.sendall(payload)
        except OSError:
            # Connection died; drop it and wait for the setup thread to
            # accept a new one.
            try:
                self._control_conn.close()
            except OSError:
                pass
            self._control_conn = None


__all__ = ["GameInterface", "MockGameInterface", "RealGameInterface"]
