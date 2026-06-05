"""SPEEDTRIALS2D rule-based real-time control system.

Submodules:
    config                  — all tunable parameters in one place.
    state                   — GameState / Token / SharedState (mutex-protected).
    game_interface          — abstract + Mock + Real (Unity TCP) interfaces.
    instrumentation         — per-cycle CSV logging and stats.
    scheduler               — PeriodicTask / TaskRuntime (uC/OS-II style).
    decision                — pure rule-based policy (avoid red / chase green).
    tasks                   — Perception / Decision / Actuation / Watchdog bodies.
    main                    — assemble and run.
    schedulability_analysis — offline RMS / EDF / RTA from CSV logs.
"""
