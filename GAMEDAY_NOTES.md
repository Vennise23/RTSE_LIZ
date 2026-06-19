# V3.0 Game-Day Tactical Build — Notes

Priority on game day: **Tactical win > distance at 180s**.
Tactical win = net **+60 green** (green − red) **AND** pass every event ≥ once.

## How to run
1. Launch the V3.0 game: `RTSE_Phase_1_V3.0\SpeedTrials2D.exe`
2. **Click the game window and press a key (Space/Enter) to start it** — it sits
   on a "waiting" screen and does NOT auto-start from the control socket.
3. In a terminal at the repo root: `python sample_drive.py`
4. Watch the `Perception + Control` debug window. Top overlay shows:
   - `STEER / ACCEL`, current decision `reason`, `lane cur->target`
   - `EV:` currently-detected live events + `bright=` (darkness meter)
   - `PASS x/5: EV1.. EV2OK ..` — which events are passed (read off the HUD)

## Strategy (driving.py priority each tick)
`Darkness > Police > Chasing > Golden > Tactical-green`
- **EV1 Darkness** — brightness drops → `steering=0, accel=-1.0` (full brake; any
  input during darkness is penalised −10%).
- **EV2 Police** — *(detector OFF, see below)* would steer to a red token while
  hard-avoiding the cop lane (cop collision = GAME OVER).
- **EV3/EV4 Chasing** — *(detector OFF, see below)* would dodge to a free lane.
- **EV5 Golden** — lane that turns all-green → lock onto it and hold to expiry.
- **Tactical-green** — otherwise farm green / avoid red (drives net +60).

## What is reliable vs needs tuning
RELIABLE (validated on live frames):
- Darkness braking (threshold tuned to the night scene, ~0.28).
- **EV1–EV5 pass tracker** read straight off the top-centre HUD (green=passed),
  with a latch so a flicker/darkness frame never un-passes an event.
- Token detect ranges widened from real samples (pink reds @H≈11, gold yellows
  @H≈18 were being missed before).

OFF BY DEFAULT (color CV unreliable on the night scene — flip in `perception.py`
only after calibrating on real frames):
- `ENABLE_POLICE_CV` — blue sky + our own red car/curbs false-fired every frame.
- `ENABLE_CHASING_CV` — dark rear scene looked like a car every frame (67 false
  dodges in one 55s run). Normal driving avoids chasers passively for now.
  Note: in live tests EV1/EV2/EV5 still got passed from normal play.

## Tunables (all named, top of `perception.py`)
- `DARK_BRIGHTNESS_ON/OFF` — darkness trigger (night base is ~0.45).
- `GOLDEN_MIN_GREEN / GOLDEN_MARGIN` — golden-lane sensitivity.
- `EV_LABEL_XC / EV_LABEL_YB / EV_LABEL_HALFW` — HUD tracker box positions.
- `ENABLE_POLICE_CV / ENABLE_CHASING_CV` — re-enable car detectors.
- token HSV ranges + min area in `detect_tokens()`.
Driving weights (green vs corridor vs stability) are at the bottom of
`decide()` in `driving.py`.

## Debug capture (for tuning)
`RTSE_DUMP=<dir> RTSE_DUMP_N=30 RTSE_DUMP_GAP=1.5 python sample_drive.py`
saves front frames + logs `[DUMP] bright= green= red= golden= EVpassed=` so you
can calibrate against real gameplay (this is how the above was tuned).

## Offline logic test (no game needed)
`python test_tactical.py` — feeds synthetic tokens/events through the real
`driving_logic_task` and prints each event branch's decision.

## Known limits to mention in the report
- No telemetry channel: every event is perceived from camera pixels (confirmed
  by decompiling `Assembly-CSharp.dll`; the golden banner is drawn onto the
  stream via `DrawGoldenBannerOnTexture`).
- Camera read tasks occasionally miss the 50 ms deadline (~57–66 ms) — bounded
  by Unity's frame cadence, not our logic.
- GIL / no priority inheritance caveats already covered in rtos_control/README.
