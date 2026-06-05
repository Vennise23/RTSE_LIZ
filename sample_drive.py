"""
Entry point — RTSE Phase 1.

Wires together:
  - rtos.py        (Member 2: RTTask, shared_data, data_lock)
  - comms.py       (Network sockets to Unity)
  - perception.py  (Member 1: camera reads, token + lane detection)
  - driving.py     (Member 3: decision logic + send controls)

Period assignments (chosen via measured WCET on this host):
  - Cameras: 50 ms. The spec's 5 ms is physically impossible because socket
    recv of one Unity frame is bounded below by Unity's ~30 fps output
    cadence (~33 ms/frame). We measured average ~30 ms with rare 100+ ms
    spikes during burst, so 50 ms gives headroom while still beating a
    20 Hz update rate.
  - Processing / DrivingLogic: 40 ms — perception runs at half the camera
    rate which is plenty for token + lane decisions.
  - SendControls: 5 ms — measured WCET 0.4 ms, easily meets deadline.
"""

import threading
import time

import cv2

import comms
import driving
import perception
import rtos
from rtos import RTTask, TaskPriority


if __name__ == '__main__':
    print("Initializing RTSE Sample Drive...")

    threading.Thread(target=comms.setup_control_server, daemon=True).start()
    threading.Thread(target=comms.setup_cameras, daemon=True).start()

    print("\n--- Starting Real-Time Tasks (awaiting connections dynamically) ---\n")

    tasks = [
        RTTask("ReadFrontCamera", period=0.050, priority=TaskPriority.HIGH,
               execute_func=perception.read_front_camera_task),
        RTTask("ReadBackCamera",  period=0.050, priority=TaskPriority.HIGH,
               execute_func=perception.read_back_camera_task),
        RTTask("Processing",      period=0.040, priority=TaskPriority.MEDIUM,
               execute_func=perception.processing_task),
        RTTask("DrivingLogic",    period=0.040, priority=TaskPriority.MEDIUM,
               execute_func=driving.driving_logic_task),
        RTTask("SendControls",    period=0.005, priority=TaskPriority.HIGH,
               execute_func=driving.send_controls_task),
    ]

    for t in tasks:
        t.start()

    try:
        while rtos.is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Stopping system...")
        rtos.is_running = False

    for t in tasks:
        t.join()

    comms.close_all()
    cv2.destroyAllWindows()
    print("System terminated cleanly.")
