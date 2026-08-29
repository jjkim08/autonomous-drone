
# Custom Quadcopter — Autonomous Visual Tracking

The vision and flight-control code for a self-built quadcopter that
autonomously follows a colored ball: detects it with an onboard camera,
then commands the flight controller to yaw and move to keep it centered
and at a set distance.

## How it works

1. **`ball.py`** — detects a colored sphere (calibrated for a printed blue
   or pink ball) from a live camera feed using HSV color thresholding.
   A stateful tracker locks onto the ball across frames, rejects
   false positives (sky, background) using shape and saturation checks,
   and estimates distance from a calibrated focal length. Streams the
   annotated feed over MJPEG so you can watch it live on the network.
2. **`follow.py`** — connects to the flight controller over MAVLink,
   arms, takes off, and switches to GUIDED mode. Two proportional
   controllers convert the ball's position (from `ball.py`) into
   commands: horizontal offset → yaw rate, estimated distance → forward
   velocity. Commands are streamed continuously at 10 Hz, with a
   lost-lock timeout that stops the drone if the ball disappears.
3. **`motion_test.py`** — a minimal takeoff/translate/land script used
   to validate basic GUIDED-mode velocity control before running the
   full tracking loop.

## Hardware

- Custom 3D-printed frame (PETG), 4x brushless motors
- MicoAir V2 flight controller running ArduPilot
- Raspberry Pi + USB camera for onboard vision
- RadioMaster ELRS radio link

## Running it

```bash
python ball.py --stream          # tune detection, view feed at http://<pi-ip>:8080
python follow.py --dry-run       # print commands without arming/flying
python follow.py                 # real autonomous flight
```

## Status

Working — the drone autonomously tracks and follows a ball in flight
(yaw + forward following). Project considered complete; further work
would require replacing the flight controller after a hardware failure.
