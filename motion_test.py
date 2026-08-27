#!/usr/bin/env python3
"""
motion_test.py

Minimal GUIDED motion test: take off, hover, translate forward a set
distance using body-frame velocity commands, hover, land.

Dress rehearsal for the tracking loop — same message type, same frame,
same continuous-send pattern the follow controller uses.

BEFORE FLYING:
  - Compass healthy, no prearm warnings
  - GPS 3D fix, 8+ satellites
  - Transmitter ON, thumb on the mode switch
  - Clear area, nothing within ~5 m of the flight path

Usage:
    python motion_test.py --dry-run
    python motion_test.py                       # 1.5 m at 0.3 m/s
    python motion_test.py --distance 2.0 --speed 0.4 --alt 1.5
"""

import argparse
import sys
import time

from pymavlink import mavutil

# ---------------------------------------------------------------- config

SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 921600              # must match SERIAL4_BAUD on the FC

SOURCE_SYSTEM = 1
SOURCE_COMPONENT = 195          # MAV_COMP_ID_PATHPLANNER

GUIDED_MODE = 4                 # ArduCopter mode numbers
LAND_MODE = 9

TAKEOFF_ALT = 1.5               # metres
TRAVEL_DISTANCE = 1.5           # must exceed GPS noise (~1-2 m)
TRAVEL_SPEED = 0.3              # m/s — slow and boring on purpose
HOVER_SECONDS = 3.0

MIN_SATELLITES = 8
ALT_TOLERANCE = 0.25

# The FC stops acting on velocity targets after ~3 s of silence, so
# commands must be re-sent continuously. 10 Hz gives plenty of margin.
SEND_HZ = 10

# Bits set = "ignore this field". Position ignored, velocity used,
# acceleration ignored, yaw and yaw_rate ignored.
VELOCITY_ONLY_MASK = 0b110111000111


# ------------------------------------------------------------- utilities

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def connect():
    log(f"Opening {SERIAL_PORT} at {BAUD_RATE}...")
    link = mavutil.mavlink_connection(
        SERIAL_PORT, baud=BAUD_RATE,
        source_system=SOURCE_SYSTEM,
        source_component=SOURCE_COMPONENT,
    )
    link.wait_heartbeat(timeout=15)
    if link.target_system == 0:
        raise RuntimeError("No heartbeat from flight controller.")
    log(f"Heartbeat from system {link.target_system}, "
        f"component {link.target_component}")
    return link


def request_stream(link, msg_id, hz):
    link.mav.command_long_send(
        link.target_system, link.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0, msg_id, int(1e6 / hz), 0, 0, 0, 0, 0)


def preflight(link, timeout=60):
    log("Preflight checks...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = link.recv_match(type="GPS_RAW_INT", blocking=True, timeout=5)
        if msg is None:
            continue
        if msg.fix_type >= 3 and msg.satellites_visible >= MIN_SATELLITES:
            log(f"GPS ready: fix {msg.fix_type}, "
                f"{msg.satellites_visible} sats")
            return True
        log(f"Waiting on GPS: fix {msg.fix_type}, "
            f"{msg.satellites_visible} sats")
    log("PREFLIGHT FAILED — GPS never reached a usable state.")
    return False


def set_mode(link, mode_number, name, timeout=10):
    log(f"Requesting {name}...")
    link.mav.set_mode_send(
        link.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_number)

    deadline = time.time() + timeout
    while time.time() < deadline:
        hb = link.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if hb and hb.custom_mode == mode_number:
            log(f"{name} confirmed.")
            return True
    log(f"FAILED to enter {name}. Check prearm messages in Mission Planner.")
    return False


def arm(link, timeout=10):
    log("Arming...")
    link.mav.command_long_send(
        link.target_system, link.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0)

    ack = link.recv_match(type="COMMAND_ACK", blocking=True, timeout=timeout)
    if ack and ack.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
        if ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            log("Armed.")
            return True
        log(f"Arm REJECTED (result {ack.result}) — usually a failed prearm "
            f"check. Look at the Messages tab.")
        return False
    log("No arm acknowledgement.")
    return False


def disarm(link):
    log("Disarming...")
    link.mav.command_long_send(
        link.target_system, link.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 0, 0, 0, 0, 0, 0, 0)


def takeoff(link, altitude, timeout=10):
    log(f"Takeoff to {altitude} m...")
    link.mav.command_long_send(
        link.target_system, link.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude)

    ack = link.recv_match(type="COMMAND_ACK", blocking=True, timeout=timeout)
    if ack and ack.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
        if ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            log("Takeoff accepted.")
            return True
        log(f"Takeoff REJECTED (result {ack.result}).")
        return False
    log("No takeoff acknowledgement.")
    return False


def wait_for_altitude(link, target, timeout=30):
    log(f"Climbing to {target} m...")
    deadline = time.time() + timeout
    last_report = 0.0
    while time.time() < deadline:
        msg = link.recv_match(type="GLOBAL_POSITION_INT",
                              blocking=True, timeout=2)
        if msg is None:
            continue
        alt = msg.relative_alt / 1000.0
        if time.time() - last_report > 1.0:
            log(f"  {alt:.2f} m")
            last_report = time.time()
        if alt >= target - ALT_TOLERANCE:
            log(f"Reached {alt:.2f} m.")
            return True
    log("Timed out climbing.")
    return False


# -------------------------------------------------------- velocity control

def send_velocity(link, vx, vy, vz):
    """Body-frame velocity command.

    vx forward (+) / back (-)
    vy right   (+) / left (-)
    vz DOWN    (+) / up   (-)     <- NED, so negative is climb
    """
    link.mav.set_position_target_local_ned_send(
        0,                                  # time_boot_ms (0 = now)
        link.target_system, link.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        VELOCITY_ONLY_MASK,
        0, 0, 0,                            # position (ignored)
        vx, vy, vz,                         # velocity
        0, 0, 0,                            # acceleration (ignored)
        0, 0)                               # yaw, yaw_rate (ignored)


def stream_velocity(link, vx, vy, vz, duration, label):
    """Hold a velocity for a duration, re-sending continuously."""
    log(f"{label}: vx={vx:+.2f} vy={vy:+.2f} vz={vz:+.2f} "
        f"for {duration:.1f}s")

    interval = 1.0 / SEND_HZ
    end = time.time() + duration
    next_report = time.time() + 1.0

    while time.time() < end:
        send_velocity(link, vx, vy, vz)

        if time.time() >= next_report:
            msg = link.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
            if msg:
                log(f"  alt {msg.relative_alt / 1000.0:.2f} m  "
                    f"gnd speed {(msg.vx**2 + msg.vy**2)**0.5 / 100:.2f} m/s")
            next_report = time.time() + 1.0

        time.sleep(interval)

    # Explicit stop, held briefly so the FC actually settles.
    log(f"{label}: stopping")
    stop_end = time.time() + 1.0
    while time.time() < stop_end:
        send_velocity(link, 0, 0, 0)
        time.sleep(interval)


def land_and_wait(link, timeout=60):
    log("Landing...")
    set_mode(link, LAND_MODE, "LAND")
    deadline = time.time() + timeout
    while time.time() < deadline:
        hb = link.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if hb and not (hb.base_mode &
                       mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            log("Landed and disarmed.")
            return True
        msg = link.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
        if msg:
            log(f"  {msg.relative_alt / 1000.0:.2f} m")
    log("Land timed out — disarming manually.")
    disarm(link)
    return False


# ------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="check GPS, enter GUIDED, stream commands, "
                             "never arm. Run this first with props OFF.")
    parser.add_argument("--alt", type=float, default=TAKEOFF_ALT)
    parser.add_argument("--distance", type=float, default=TRAVEL_DISTANCE,
                        help="metres forward (keep above ~1.5 m — GPS "
                             "noise swamps anything smaller)")
    parser.add_argument("--speed", type=float, default=TRAVEL_SPEED,
                        help="m/s")
    args = parser.parse_args()

    if args.distance < 1.0:
        log(f"WARNING: {args.distance} m is at or below GPS position noise. "
            f"You will not be able to tell the movement from drift.")

    travel_time = args.distance / args.speed
    log(f"Plan: climb {args.alt} m, hover {HOVER_SECONDS}s, "
        f"forward {args.distance} m at {args.speed} m/s "
        f"({travel_time:.1f}s), hover, land.")

    link = connect()
    request_stream(link, mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 4)
    request_stream(link, mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT, 2)

    if not preflight(link):
        sys.exit(1)

    if not set_mode(link, GUIDED_MODE, "GUIDED"):
        sys.exit(1)

    if args.dry_run:
        log("DRY RUN — streaming velocity commands, nothing armed.")
        log("Motors will NOT spin. Watch Mission Planner to confirm the "
            "FC is receiving these.")
        stream_velocity(link, args.speed, 0, 0, travel_time, "dry forward")
        log("DRY RUN complete. GUIDED accepted, commands sent, no arm.")
        return

    log("")
    log("=" * 54)
    log("  ARMING IN 5 SECONDS — Ctrl+C to abort")
    log("  Transmitter ready? SE switch to AltHold takes control back.")
    log("=" * 54)
    log("")
    time.sleep(5)

    if not arm(link):
        sys.exit(1)

    if not takeoff(link, args.alt):
        disarm(link)
        sys.exit(1)

    if not wait_for_altitude(link, args.alt):
        land_and_wait(link)
        sys.exit(1)

    stream_velocity(link, 0, 0, 0, HOVER_SECONDS, "settle")
    stream_velocity(link, args.speed, 0, 0, travel_time, "forward")
    stream_velocity(link, 0, 0, 0, HOVER_SECONDS, "hold")

    land_and_wait(link)
    log("Flight complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted. If airborne, TAKE MANUAL CONTROL NOW.")
        sys.exit(130)