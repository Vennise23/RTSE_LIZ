"""
Offline self-test for the V3.0 tactical driving logic.

Feeds synthetic tokens + event flags straight into the REAL code path
(driving.driving_logic_task) via shared_data and prints the decision, so we
can verify the priority state machine without launching the Unity game.
"""
import numpy as np

import driving
from rtos import shared_data, data_lock

LANES = [100, 200, 300, 400, 500]   # 5 lane centers, frame 640 wide
FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
CUR = 2                             # car always treated as middle lane


def tok(color, lane, y):
    return {'color': color, 'x': LANES[lane], 'y': y, 'area': 500, 'box': (0, 0, 10, 10)}


def run(name, tokens, **events):
    # reset driving memory so each scenario is independent
    driving._lane_lock = -1
    driving._lock_counter = 0
    driving._last_reason = ""
    driving._speed_mem = 0.85

    with data_lock:
        shared_data['latest_front_frame'] = FRAME
        shared_data['detected_tokens'] = tokens
        shared_data['lane_centers'] = LANES
        shared_data['current_lane'] = CUR
        shared_data['rear_vehicle_close'] = False
        shared_data['event_darkness'] = events.get('darkness', False)
        shared_data['event_police'] = events.get('police', False)
        shared_data['police_lane'] = events.get('police_lane', -1)
        shared_data['event_golden_lane'] = events.get('golden_lane', -1)
        shared_data['rear_threat_lane'] = events.get('rear_lane', -1)

    driving.driving_logic_task()

    with data_lock:
        r = shared_data['decision_reason']
        t = shared_data['target_lane']
        s = shared_data['steering_input']
        a = shared_data['acceleration_input']
    print(f"{name:<28} -> reason={r:<22} target_lane={t}  steer={s:+.2f}  accel={a:+.2f}")


print("=" * 92)
print("V3.0 TACTICAL DECISION SELF-TEST   (car is in lane 2 of 0..4)")
print("=" * 92)

# 1. Plain green farming: greens ahead in lane 3 (reachable) -> switch right.
run("1. tactical green (lane 3)", [tok('green', 3, 250), tok('green', 3, 200)])

# 1b. Green two lanes away (lane 4) is still worth crossing for now.
run("1b. green two lanes (lane 4)", [tok('green', 4, 230), tok('green', 4, 190)])

# 1c. Empty road -> hold lane, no needless zig-zag.
run("1c. empty road -> hold", [])

# 1d. Green AND a close red in the SAME lane (3) -> do NOT enter it for green.
run("1d. green+close-red lane 3", [tok('green', 3, 250), tok('red', 3, 310)])

# 1e. Green in lane 3, only a FAR red in lane 1 -> still take the green.
run("1e. green L3, far red L1", [tok('green', 3, 250), tok('green', 3, 210),
                                 tok('red', 1, 150)])

# 2. Red dead ahead in our lane -> emergency escape, never sit on it.
run("2. red in current lane", [tok('red', 2, 300)])

# 3. EV5 Golden Lane = lane 0 -> lock onto it and hold.
run("3. EV5 golden lane 0", [tok('green', 4, 250)], golden_lane=0)

# 4. EV2 Police: cop in lane 2 (our lane!), a red sits in lane 4 ->
#    must leave the kill lane and go take the red in lane 4.
run("4. EV2 police cop=L2", [tok('red', 4, 300), tok('green', 1, 250)],
    police=True, police_lane=2)

# 5. EV3/4 Chasing car behind in our lane -> dodge to a different lane.
run("5. EV34 chasing rear=L2", [tok('green', 2, 250)], rear_lane=2)

# 6. EV1 Darkness -> freeze steering, brake fully (accel = -1.0).
run("6. EV1 darkness", [tok('green', 3, 250)], darkness=True)

print("=" * 92)
print("Expected: 1->TACTICAL_GREEN L3 | 2->EMERGENCY_RED (leave L2) | "
      "3->EV5_GOLDEN_HOLD L0\n          4->EV2_POLICE_TAKE_RED L4 (avoids cop L2) | "
      "5->EV34_CHASING_DODGE | 6->accel=-1.00")
print("=" * 92)
