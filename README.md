# Hack2Drive 2026

## Introduction
This lab introduces core principles of Real-Time Systems Engineering through a hands-on self-driving simulation. Participants will build a Python-based control system that:
- Receives real-time camera input from a Unity environment
- Processes perception using computer vision / ML
- Trains a Behavioural Cloning Model
- Sends control commands under strict timing constraints

## Project Structure
- **[RTSE_Phase_1_V0.5_source](./RTSE_Phase_1_V0.5_source)**: 2D racing environment with tokens and events.
- **[RTSE_Phase_2_V0.5_source](./RTSE_Phase_2_V0.5_source)**: Navigation challenge focused on self-driving capabilities and Behavioural Cloning.

---

## Phase 1
![Phase 1 Demo](./Phase_1_game.gif)

### Technical Setup
- How to control the Unity environment using the Python communication script
- How to train / use a small YOLO model
- How to use simple computer vision algos (OpenCV)

### Game Rules
**Objective:** 60 seconds to travel the furthest distance.

#### Tokens & Effects
- **Green Token:** Increase speed by +10%
- **Red Token:** Decrease speed by −20%
- **Yellow Token:**
  - 20% = Next token type / color hidden
  - 20% = Next 5 seconds, tokens become invisible
  - 20% = Next 5 seconds, camera input delay
  - 20% = Next 5 seconds, action output delay
  - 20% = Next 5 seconds, corrupted camera input

#### Events
- A faster car appears behind, and the player must switch lanes. On collision: −50% speed.
- A police car appears behind, and the player must take the next red token. If you ignore: −50% speed.
- The brightness decreases under 50%. All tokens become Yellow until the light is turned ON. While light ON, green tokens increase speed by +5% instead.

---

## Phase 2
![Phase 2 Demo](./Phase_2_game.gif)

### Technical Setup
- Introduction to self-driving
- Explaining Behavioural Cloning

### Game Rules
**Objective:** Navigate from Point A to Point B safely and efficiently.

#### Health System
Cars start with 5 health points (HP). Each violation results in −1 HP.

#### Violations
- Collision with obstacle/NPC
- Running a red light
- Ignoring a stop sign
- Not stopping for pedestrians at crossings
- Not giving priority to a police car
- Ignoring police officer signals

---

## 24 Hours Hackathon
**Introduction:** On ??/??/26, a new challenge will be revealed based on Phase 1 and Phase 2. The participants have 24 hours to submit their code / result (From 8 pm to 8 pm, Lab open 24 hours). A quiz assessing the lab content will be conducted during the 24 hours hackathon, accessible for 1 hour at a random hour of the last 8 hours of the hackathon.

### Reveal at the start of Hackathon:
- **Stage 1:** Tournament style for 1v1 Phase 1 game (multiplayer + new tokens)
- **Stage 2:** The top 10 teams go to Phase 2 game (multiplayer + new challenge)
# RTSE_LIZ
