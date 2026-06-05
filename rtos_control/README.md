# SPEEDTRIALS2D — Rule-Based Real-Time Control System

A multi-threaded, rule-based control system for the SPEEDTRIALS2D Unity
simulator, built for SECJ 4423 Real-Time Software Engineering. The
architecture imitates the **uC/OS-II RTOS** style: a small set of
periodic tasks with explicit priorities, communicating through a
mutex-protected shared snapshot and a bounded message queue.

The project deliberately uses **zero machine learning**. Every
decision is a deterministic if-else with documented thresholds, so the
execution time of the decision body is predictable and amenable to
classical schedulability analysis (RMS / EDF / RTA).

---

## 1. Architecture at a glance

```
                  +----------------+
                  |  GameInterface |  (abstract)
                  +--------+-------+
                           |
            +--------------+--------------+
            |                             |
  +--------------------+        +-----------------------+
  | MockGameInterface  |        |   RealGameInterface   |   <-- TCP to
  | (offline world)    |        |   ports 8080/8082/8081|       Unity .exe
  +--------------------+        +-----------------------+
            |                             |
            v                             v
   +-----------------------------------------------+
   |  Perception task (T=20ms, prio=2)             |   read_state()
   |    write -> SharedState (under mutex)         |
   +-----------------------------------------------+
                            |
                            v  (snapshot copy)
   +-----------------------------------------------+
   |  Decision task   (T=50ms, prio=4)             |   pure decide()
   |    safety > reward > stability                |
   |    push Command -> queue.Queue                |
   +-----------------------------------------------+
                            |
                            v  (latest-wins drain)
   +-----------------------------------------------+
   |  Actuation task  (T=30ms, prio=3)             |   send_command()
   |    map Command -> (steering, accel) floats    |
   +-----------------------------------------------+

   +-----------------------------------------------+
   |  Watchdog task   (T=20ms, prio=1, HIGHEST)    |
   |    watches perception health + deadline misses|
   |    on degrade: push DEGRADE command           |
   +-----------------------------------------------+
```

uC/OS-II convention: **smaller priority number = higher priority**.

| Task        | T    | Priority | Role                                                            |
|-------------|------|----------|-----------------------------------------------------------------|
| Watchdog    | 20ms | 1        | Detect deadline misses / stale state, trigger degraded behavior |
| Perception  | 20ms | 2        | `read_state()` → write snapshot under mutex                     |
| Actuation   | 30ms | 3        | Drain latest command → `send_command()`                         |
| Decision    | 50ms | 4        | Run rule-based policy, enqueue command                          |

Communication primitives (mapped onto Python's `threading`):

* `SharedState` — wraps a `threading.Lock`; this is the moral equivalent
  of a uC/OS-II `OSMutex` protecting a `OSMbox` (the latest perception
  snapshot is what readers fetch).
* `queue.Queue(maxsize=8)` — message queue between Decision and
  Actuation. Modeled on `OSQPost`/`OSQPend`. Bounded on purpose: the
  enqueue helper evicts a stale command to keep the queue fresh.

---

## 2. Files

```
rtos_control/
├── __init__.py
├── config.py                   # all tunables (periods, priorities, thresholds)
├── state.py                    # GameState / Token / SharedState (+ Lock)
├── game_interface.py           # abstract + Mock + Real (TCP) implementations
├── instrumentation.py          # per-task CSV logger + running stats
├── scheduler.py                # PeriodicTask, TaskRuntime (absolute-time loop)
├── decision.py                 # pure rule-based policy
├── tasks.py                    # Perception / Decision / Actuation / Watchdog bodies
├── main.py                     # assemble and run
├── schedulability_analysis.py  # offline RMS / EDF / RTA from CSV logs
└── README.md                   # you are here
```

Run from the **repository root** (the directory that contains the
`rtos_control/` folder).

---

## 3. Running

### 3.1 Mock mode (no game required)

```powershell
# from the repo root
python -m rtos_control.main --mode mock --seconds 20
```

You will see the four tasks start, the system run, and CSV logs land
under `logs/`. At the end, a per-task summary table is printed.

### 3.2 Real mode (drive the Unity build)

1. Launch the game:
   `SpeedTrials2D\SpeedTrials2D.exe`
2. In a second terminal, from the repo root:
   ```powershell
   python -m rtos_control.main --mode real --seconds 60
   ```
3. Stop with `Ctrl+C` to flush the logs cleanly.

Real mode needs the `opencv-python` and `numpy` packages from
`requirements.txt` (token detection on the front-camera frames).

### 3.3 Schedulability analysis

After any run (mock or real):

```powershell
python -m rtos_control.schedulability_analysis
```

This reads every `logs/*.csv` and prints, for the observed WCETs:

* per-task `U_i = WCET_i / T_i` and total `U`,
* Liu & Layland RMS bound `n * (2^(1/n) - 1)` and verdict,
* EDF bound `U ≤ 1` and verdict,
* Response-Time Analysis (RTA) per task with verdict.

Pass an alternate log directory as the first argument if needed:
```powershell
python -m rtos_control.schedulability_analysis logs_run_2026_06_05
```

---

## 4. Decision policy

A single pass through `decide(state, memory)` in `decision.py`:

1. **SAFETY (avoid red)** — if a red token or any obstacle is within
   `BRAKE_DIST` in the current lane, switch to whichever adjacent lane
   is safe; if both adjacent lanes are also hazardous, command
   `SLOW_DOWN`.
2. **REWARD (chase green)** — otherwise, score each lane within an
   adaptive look-ahead window
   `LOOKAHEAD = LOOKAHEAD_BASE + LOOKAHEAD_SPEED_GAIN * speed_norm`
   (the faster we go, the further we look). A lane is only chosen if
   its reward beats the current lane by at least `SWITCH_MARGIN` and
   the lateral `SWITCH_COOLDOWN` has expired — both serve as hysteresis
   against oscillation.
3. **STABILITY (hold)** — otherwise hold lane and cruise.

The red penalty `RED_PENALTY` is **5x** larger than `GREEN_REWARD`, so
the policy will skip a green if the same window contains a red. All
weights and thresholds live in `config.py`.

---

## 5. Instrumentation

Every release of every task appends a row to
`logs/<TaskName>.csv` with columns:

```
release_idx, release_time, exec_time_s, period_s, deadline_s,
finish_time, deadline_abs, missed_deadline, note
```

`exec_time_s` is the wall-clock body time measured with
`time.perf_counter()` around the body call. `missed_deadline` is 1
when `finish_time > deadline_abs` (deadline = period by default).

The `note` column carries short diagnostic strings: the kind of
command emitted (`MOVE_LEFT`, `HOLD`, …), or a watchdog summary
(`perception_unhealthy_x3;Decision_miss+1`), which is handy when
correlating misses with environmental events.

---

## 6. Adapting the Real interface

`RealGameInterface` already implements the protocol used by
`sample_drive.py` and `test_communication.py`:

* Connect (client) to the front-camera TCP server (port 8080).
* Listen (server) for the control client on port 8081; the game
  connects.
* Send `struct.pack('ff', steering, acceleration)` per command, both
  in `[-1.0, 1.0]`.

If the game later exposes a richer state API (lane number, speed
telemetry), only `RealGameInterface.read_state` needs to change;
nothing above the interface knows or cares.

---

## 7. Known platform limits (write-up material)

The analysis treats CPython threads as ordinary RTOS tasks. Two
caveats are worth stating explicitly in the report:

* **GIL**: at any instant only one Python thread executes bytecode,
  so logical task priorities translate into hints, not preemption.
  We still set OS thread priorities best-effort (Windows
  `SetThreadPriority`) so OS-side scheduling biases toward the
  Watchdog / Perception threads under contention.
* **No priority inheritance**: `threading.Lock` is a plain mutex. A
  Decision task holding the SharedState lock can briefly delay
  Perception. We bound the critical section to a single tuple copy
  (`SharedState.snapshot()` releases the lock immediately and returns
  the immutable `GameState`), so in practice the blocking term is
  microseconds — visible in the WCET column of the CSV.

These are the reasons our RTA verdict on the *observed* WCETs is a
useful sanity check, but the report should also discuss the gap to a
true preemptive RTOS implementation.

---

## 8. Quick reference

| Question                                  | Where to look                           |
|-------------------------------------------|-----------------------------------------|
| Change a period or priority               | `config.py` (`TASK_*_PERIOD`, `_PRIORITY`) |
| Change the decision thresholds            | `config.py` (`BRAKE_DIST`, `SWITCH_MARGIN`, ...) |
| Add a new task                            | Build a body in `tasks.py`, register in `main.py` |
| Swap mock for real                        | `--mode real` on the command line       |
| Re-tune lane → x-pixel mapping for Real   | `config.LANE_X_BOUNDS_FRAC`             |
| Inspect a single task's per-cycle behavior| `logs/<TaskName>.csv`                   |
