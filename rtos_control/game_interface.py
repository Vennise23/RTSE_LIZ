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
from typing import List, Optional, Tuple

from . import config
from .state import GameState, Obstacle, Token, TokenColor


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
        if not self._low_light_active and elapsed >= config.LOW_LIGHT_TRIGGER_SEC:
            self._low_light_active = True
            self._low_light_started_at = now
            self._low_light_applied = False
            # All tokens become unknown while the light is off.
            self._tokens = tuple(
                Token(lane=t.lane, distance=t.distance, color=TokenColor.YELLOW)
                for t in self._tokens
            )
        if self._low_light_active:
            self._rear_pressure = max(self._rear_pressure, 0.2)
            if self._acceleration <= -0.9:
                self._low_light_active = False
                self._low_light_applied = True

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
        self._front_reader_thread: Optional[threading.Thread] = None
        self._back_reader_thread: Optional[threading.Thread] = None

        # Lane-of-self tracking (we never get told, so we integrate from
        # our own steering commands). 0 .. NUM_LANES-1.
        self._own_lane_float = float(config.LANE_CENTER_INDEX)
        self._last_command_at: Optional[float] = None
        self._last_steering = 0.0
        self._last_acceleration = 0.0

        # Optional debug visualization. When on, the camera reader thread
        # draws an annotated frame in an OpenCV window. Off-by-default for
        # Mock mode (this class is never used in mock); on-by-default for
        # Real mode so the operator can verify perception live. The
        # window is pinned always-on-top so it stays visible over the
        # Unity game window (since we cannot modify the Unity build to
        # draw the HUD inside it).
        self._show_overlay = show_overlay
        self._overlay_window = "RTOS Perception (RTSE)"
        self._back_overlay_window = "RTOS Back Camera (RTSE)"
        self._overlay_window_ready = False
        self._back_overlay_window_ready = False

        # Brightness smoothing so brief yellow-flash / visual effects do
        # not get misclassified as a real low-light event.
        self._brightness_history = []
        self._smoothed_brightness_history = []
        self._low_brightness_started_at = None
        self._back_frame_lock = threading.Lock()
        self._latest_back_frame = None

    # ---- lifecycle ------------------------------------------------
    def start(self) -> None:
        self._running = True
        self._setup_thread = threading.Thread(
            target=self._setup_network, name="RealGameSetup", daemon=True,
        )
        self._setup_thread.start()
        self._front_reader_thread = threading.Thread(
            target=self._camera_reader_loop, name="FrontCameraReader", daemon=True,
        )
        self._front_reader_thread.start()
        self._back_reader_thread = threading.Thread(
            target=self._back_camera_reader_loop, name="BackCameraReader", daemon=True,
        )
        self._back_reader_thread.start()

    def stop(self) -> None:
        self._running = False
        for sock in (self._front_sock, self._back_sock, self._control_conn, self._control_server):
            try:
                if sock is not None:
                    sock.close()
            except OSError:
                pass
        # Close overlay window if it was opened.
        if self._show_overlay:
            try:
                import cv2  # type: ignore
                cv2.destroyWindow(self._overlay_window)
                cv2.destroyWindow(self._back_overlay_window)
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

        # Back camera
        while self._running and self._back_sock is None:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((config.GAME_CAMERA_HOST, config.GAME_BACK_CAMERA_PORT))
                s.settimeout(None)
                self._back_sock = s
                print("[RealGameInterface] Back camera connected.")
            except OSError:
                time.sleep(0.5)

        # Control server
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

    def _camera_reader_loop(self) -> None:
        """Drain JPEG frames as fast as the game produces them.

        Each iteration decodes one frame, runs token detection, and
        updates the cached GameState. Perception (periodic) just reads
        the cache.
        """
        # Lazy imports so the mock path works without OpenCV.
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            print("[RealGameInterface] OpenCV/numpy not installed; "
                  "perception will be degraded.")
            return

        # Wait for the socket to come up.
        while self._running and self._front_sock is None:
            time.sleep(0.05)

        sock = self._front_sock
        while self._running and sock is not None:
            try:
                length_bytes = sock.recv(4)
                if not length_bytes or len(length_bytes) < 4:
                    self._mark_unhealthy()
                    break
                image_length = int.from_bytes(length_bytes, "little")
                buf = bytearray()
                while len(buf) < image_length and self._running:
                    chunk = sock.recv(image_length - len(buf))
                    if not chunk:
                        break
                    buf.extend(chunk)
                if len(buf) != image_length:
                    self._mark_unhealthy()
                    continue
                np_arr = np.frombuffer(bytes(buf), np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    self._mark_unhealthy()
                    continue
                brightness = self._estimate_brightness(frame, cv2, np)
                enriched = self._detect_tokens(frame, cv2, np)
                tokens = tuple(item[0] for item in enriched)
                self._update_state_from_perception(frame.shape, brightness, tokens)
                if self._show_overlay:
                    try:
                        self._render_overlay(frame, enriched, cv2)
                    except Exception:
                        # Never let a draw error kill the reader thread;
                        # we'd rather lose the HUD than the perception.
                        pass
            except OSError:
                self._mark_unhealthy()
                break

    def _back_camera_reader_loop(self) -> None:
        """Read the back camera stream and keep its window alive."""
        try:
            import cv2  # type: ignore
        except ImportError:
            print("[RealGameInterface] OpenCV not installed; back camera window disabled.")
            return

        while self._running and self._back_sock is None:
            time.sleep(0.05)

        sock = self._back_sock
        while self._running and sock is not None:
            try:
                length_bytes = sock.recv(4)
                if not length_bytes or len(length_bytes) < 4:
                    break
                image_length = int.from_bytes(length_bytes, "little")
                buf = bytearray()
                while len(buf) < image_length and self._running:
                    chunk = sock.recv(image_length - len(buf))
                    if not chunk:
                        break
                    buf.extend(chunk)
                if len(buf) != image_length:
                    continue
                with self._back_frame_lock:
                    self._latest_back_frame = bytes(buf)
                if self._show_overlay:
                    try:
                        self._render_back_overlay(bytes(buf), cv2)
                    except Exception:
                        pass
            except OSError:
                break

    def _mark_unhealthy(self) -> None:
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

    def _update_state_from_perception(self, frame_shape, brightness, tokens) -> None:
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

        # Median filter: ignore single-frame dips caused by token flash
        # effects or transient exposure changes.
        self._brightness_history.append(float(brightness))
        if len(self._brightness_history) > 5:
            self._brightness_history.pop(0)
        filtered_brightness = float(statistics.median(self._brightness_history)) if self._brightness_history else float(brightness)

        self._smoothed_brightness_history.append(filtered_brightness)
        if len(self._smoothed_brightness_history) > 4:
            self._smoothed_brightness_history.pop(0)

        # Only treat it as low light when the smoothed brightness is
        # below the threshold and is still trending downward.
        low_light_trending_down = False
        if len(self._smoothed_brightness_history) >= 3:
            a, b, c = self._smoothed_brightness_history[-3:]
            low_light_trending_down = a > b > c
        if filtered_brightness < config.LOW_LIGHT_THRESHOLD and low_light_trending_down:
            if self._low_brightness_started_at is None:
                self._low_brightness_started_at = now
        else:
            self._low_brightness_started_at = None

        low_light_detected = (
            self._low_brightness_started_at is not None
            and (now - self._low_brightness_started_at) >= config.LOW_LIGHT_CONFIRM_SEC
        )

        with self._perception_lock:
            self._latest_state = GameState(
                timestamp=now,
                own_lane=int(round(self._own_lane_float)),
                # We have no telemetry for actual speed; use the requested
                # throttle as a proxy. Decision uses this only to widen
                # look-ahead, so the proxy is good enough.
                speed_norm=max(0.0, min(1.0, self._last_acceleration)),
                brightness=filtered_brightness,
                low_light_active=low_light_detected,
                rear_pressure=0.0,
                rear_chase_active=False,
                rear_chase_lane=-1,
                rear_time_left=0.0,
                police_alert=False,
                police_lane=-1,
                police_time_left=0.0,
                game_over=False,
                game_over_reason="",
                tokens=tokens,
                obstacles=(),  # Phase-1 token game has no obstacles per se
                perception_healthy=True,
            )

    @staticmethod
    def _estimate_brightness(frame, cv2, np) -> float:
        """Estimate normalized brightness from the decoded front camera frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return max(0.0, min(1.0, float(np.mean(gray)) / 255.0))

    @staticmethod
    def _detect_tokens(frame, cv2, np) -> list:
        """Lane-aware HSV token detector. Mirrors sample_drive.py but
        bins detections into lanes/distances normalized to [0, 1].

        Returns an *enriched* list of ``(Token, (x, y, w, h))`` so the
        overlay renderer can draw the original bounding boxes; the
        caller converts to a plain ``tuple[Token, ...]`` for state.
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

    # ---- overlay --------------------------------------------------
    def _render_overlay(self, frame, enriched, cv2) -> None:
        """Draw an annotated copy of the frame in a live OpenCV window.

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
        snap = self.read_state()
        brightness = getattr(snap, "brightness", 1.0)
        status = (f"R={counts[TokenColor.RED]} "
                  f"G={counts[TokenColor.GREEN]} "
                  f"Y={counts[TokenColor.YELLOW]}    "
                  f"B={brightness:.2f} "
                  f"TH={config.LOW_LIGHT_THRESHOLD:.2f}    "
                  f"ACT={self._actuation_label()}    "
                  f"{self._status_label()}")
        cv2.putText(out, status, (10, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)

        if not self._overlay_window_ready:
            self._init_overlay_window(cv2)
        cv2.imshow(self._overlay_window, out)
        cv2.waitKey(1)
        # Re-pin TOPMOST in case the user clicked Unity and stole focus.
        # Cheap call (single Win32 SetWindowPos) so we do it every frame.
        self._pin_topmost_win32()

    def _render_back_overlay(self, jpeg_bytes: bytes, cv2) -> None:
        """Show the back camera as a live window so it does not vanish."""
        import numpy as np  # type: ignore

        np_arr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        if not self._back_overlay_window_ready:
            try:
                cv2.namedWindow(self._back_overlay_window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self._back_overlay_window, 640, 360)
                cv2.moveWindow(self._back_overlay_window, 760, 20)
            except Exception:
                pass
            self._back_overlay_window_ready = True
        cv2.imshow(self._back_overlay_window, frame)
        cv2.waitKey(1)

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

    def _pin_topmost_win32(self) -> None:
        """Force the overlay to stay above all other windows (incl. Unity)."""
        if not sys.platform.startswith("win"):
            return
        try:
            import ctypes
            user32 = ctypes.windll.user32
            hwnd = user32.FindWindowW(None, self._overlay_window)
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

    def _status_label(self) -> str:
        """Add a compact status tag for challenge states."""
        snap = self.read_state()
        if getattr(snap, "low_light_active", False):
            return "LOW_LIGHT"
        return "NORMAL"

    @staticmethod
    def _lane_for_x(x: int, lane_bounds) -> Optional[int]:
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
