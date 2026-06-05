"""Central configuration for the SPEEDTRIALS2D rule-based control system.

All numeric tunables live here so they are easy to sweep during analysis
and to cite in the report. Keep this file free of imports from other
project modules so it can be loaded by every component without cycles.
"""

# ----------------------------------------------------------------------
# RTOS task parameters (uC/OS-II convention: smaller number = higher priority)
# ----------------------------------------------------------------------
# Periods are in seconds. Deadline defaults to period (implicit-deadline tasks).
TASK_WATCHDOG_PERIOD   = 0.020   # 20 ms
TASK_PERCEPTION_PERIOD = 0.020   # 20 ms
TASK_ACTUATION_PERIOD  = 0.030   # 30 ms
TASK_DECISION_PERIOD   = 0.050   # 50 ms

TASK_WATCHDOG_PRIORITY   = 1
TASK_PERCEPTION_PRIORITY = 2
TASK_ACTUATION_PRIORITY  = 3
TASK_DECISION_PRIORITY   = 4

# Command queue depth. Kept small on purpose: a fresh command should
# supersede a stale one, not pile up behind it.
COMMAND_QUEUE_MAX = 8

# ----------------------------------------------------------------------
# Game / world model
# ----------------------------------------------------------------------
NUM_LANES = 3                # 0 = left, 1 = center, 2 = right
LANE_CENTER_INDEX = 1
DISTANCE_HORIZON = 1.0       # normalized distance ahead the perception reports
TOKEN_REACHED_DISTANCE = 0.05  # tokens closer than this are considered consumed/missed

# ----------------------------------------------------------------------
# Decision logic (priority order is fixed: SAFETY > REWARD > STABILITY)
# ----------------------------------------------------------------------
# Distance below which a red token / obstacle is considered "imminent"
# and triggers the avoid branch. Set deliberately large (50 % of the
# look-ahead) so the safety branch starts evading well before impact,
# leaving the lateral lane change enough time to actually complete.
BRAKE_DIST = 0.50

# Base look-ahead distance for opportunistic green-collection decisions.
# The effective look-ahead grows with speed (see LOOKAHEAD_SPEED_GAIN).
LOOKAHEAD_BASE = 0.35
LOOKAHEAD_SPEED_GAIN = 0.40    # extra look-ahead per unit of normalized speed

# Lateral cool-down (seconds) after a lane change so the car does not
# oscillate between adjacent lanes. Kept short because back-to-back reds
# must trigger back-to-back lane changes.
SWITCH_COOLDOWN = 0.15

# Minimum reward advantage required to justify a discretionary lane change.
# Acts as hysteresis around the chosen lane. With reds/yellows weighted
# 20x the green reward, even tiny differences in red exposure dominate
# this threshold, so we can keep it small.
SWITCH_MARGIN = 0.15

# Reward / penalty weights.
#   Red:    must dominate everything — 20x green reward. Triggers the
#           SAFETY branch (emergency lane change) the moment it appears
#           within BRAKE_DIST in the current lane.
#   Yellow: undesirable but NOT worth losing a green for. Weight 2x
#           green reward, and it does NOT trigger the SAFETY branch on
#           its own. A lane with only-yellow is avoided; a lane with
#           green-then-yellow is still preferred over an empty one.
#   Green:  the only positive contributor.
GREEN_REWARD   = 1.0
RED_PENALTY    = 20.0
YELLOW_PENALTY = 2.0

# How strongly nearer tokens count vs far ones. The reward of a token is
# weight * max(0, 1 - distance / LOOKAHEAD_eff).
REWARD_DECAY_NEAR_BIAS = 1.0   # linear falloff; tune up for sharper preference for close tokens

# ----------------------------------------------------------------------
# Actuation calibration (rule-based mapping from lane decision -> floats)
# ----------------------------------------------------------------------
# steering_input domain is [-1.0, +1.0] (left .. right) for the real game.
STEER_MAGNITUDE = 0.6          # how hard we push the wheel during a lane change
ACCEL_CRUISE   = 1.0           # full throttle by default
ACCEL_SLOWDOWN = 0.1           # near-coast when boxed in by reds (buy reaction time)
ACCEL_BRAKE    = -0.5          # used by watchdog degradation
LANE_HOLD_TIME = 0.20          # how long a steering pulse lasts before centering

# ----------------------------------------------------------------------
# Watchdog policy
# ----------------------------------------------------------------------
# Consecutive perception misses before we declare the perception pipeline
# unhealthy and ask actuation to coast/centre.
WATCHDOG_PERCEPTION_MISS_LIMIT = 3
# Stale state threshold: if the last perception snapshot is older than this,
# decisions made from it are no longer trusted.
WATCHDOG_STALE_STATE_SEC = 0.15

# ----------------------------------------------------------------------
# Instrumentation
# ----------------------------------------------------------------------
LOG_DIR = "logs"
LOG_FLUSH_EVERY = 50           # flush CSV every N rows to bound IO overhead

# ----------------------------------------------------------------------
# Real game interface (matches sample_drive.py / test_communication.py)
# ----------------------------------------------------------------------
GAME_CAMERA_HOST = "127.0.0.1"
GAME_FRONT_CAMERA_PORT = 8080
GAME_BACK_CAMERA_PORT  = 8082
GAME_CONTROL_HOST = "127.0.0.1"
GAME_CONTROL_PORT = 8081

# Lane boundaries as fractions of the front-camera frame width.
# The road in SpeedTrials2D occupies roughly the middle 70 %; split into thirds.
LANE_X_BOUNDS_FRAC = (0.15, 0.38, 0.62, 0.85)  # 4 edges -> 3 lanes

# Vertical region of interest for token detection (fractions of frame height).
ROI_Y_FRAC = (0.05, 0.70)
ROI_X_FRAC = (0.15, 0.85)

# Mock-mode default duration (seconds). Real-mode runs until Ctrl+C.
MOCK_RUN_SECONDS_DEFAULT = 20.0
