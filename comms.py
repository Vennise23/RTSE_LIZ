"""
Network communication with the SpeedTrials2D Unity simulator.

  - Two camera sockets (front 8080, back 8082) supply JPEG frames.
  - One control server (8081) accepts a single client connection and we
    push ``struct.pack('ff', steering, acceleration)`` to it.

Other modules access the live sockets via ``comms.front_camera_sock``,
``comms.back_camera_sock``, ``comms.control_conn`` — keeping them as module
attributes lets the read tasks see updates without passing references around.
"""

import socket
import time

import rtos


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CAMERA_HOST = '127.0.0.1'
FRONT_CAMERA_PORT = 8080
BACK_CAMERA_PORT = 8082
CONTROL_HOST = '127.0.0.1'
CONTROL_PORT = 8081


# ---------------------------------------------------------
# Network connection globals (mutated by setup + tasks)
# ---------------------------------------------------------
front_camera_sock = None
back_camera_sock = None
control_conn = None


def setup_cameras():
    global front_camera_sock, back_camera_sock

    print("Connecting to Cameras...")
    front_connected = False
    back_connected = False

    while rtos.is_running and not (front_connected and back_connected):
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

    while rtos.is_running:
        try:
            conn, addr = server_sock.accept()
            print(f"Control client connected from {addr}")
            control_conn = conn
            break
        except socket.timeout:
            continue


def close_all():
    """Close every live socket. Safe to call multiple times."""
    global front_camera_sock, back_camera_sock, control_conn
    if front_camera_sock:
        try:
            front_camera_sock.close()
        except Exception:
            pass
        front_camera_sock = None
    if back_camera_sock:
        try:
            back_camera_sock.close()
        except Exception:
            pass
        back_camera_sock = None
    if control_conn:
        try:
            control_conn.close()
        except Exception:
            pass
        control_conn = None
