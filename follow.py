#!/usr/bin/env python3
"""
follow.py

Autonomous visual tracking: yaw to face the ball, move forward/back
to hold a set distance.

Reuses the detector in ball.py — keep both files in the same folder.

Usage:
    python follow.py --dry-run              # props OFF, prints commands
    python follow.py --dry-run --no-fc      # no flight controller at all
    python follow.py                        # real flight
    python follow.py --yaw-only             # skip forward control
"""

import argparse
import sys
import time

from pymavlink import mavutil

import ball as detector

# ------------------------------------------------------------ fc config

SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 921600              # match SERIAL4_BAUD on the FC

SOURCE_SYSTEM = 1
SOURCE_COMPONENT = 195          # MAV_COMP_ID_PATHPLANNER

GUIDED_MODE = 4
LAND_MODE = 9

TAKEOFF_ALT = 1.5
MIN_SATELLITES = 8
ALT_TOLERANCE = 0.25

SEND_HZ = 10                    # FC gives up after ~3 s of silence

# Use velocity + yaw_rate. Position, acceleration and yaw ANGLE ignored.
VEL_YAWRATE_MASK = 0b010111000111


# ------------------------------------------------------- control tuning

# --- yaw ---
YAW_KP = 0.6                    # rad/s per unit offset — start low
YAW_DEADBAND = 0.06             # ignore small offsets, stops twitching
YAW_RATE_MAX = 0.8              # rad/s ceiling (~46 deg/s)

# --- forward ---
TARGET_DISTANCE = 2.0           # metres to hold
FWD_KP = 0.35                   # m/s per metre of error
FWD_DEADBAND = 0.35             # metres — distance is noisy, be generous
FWD_SPEED_MAX = 0.5
FWD_SPEED_MIN = -0.3            # backing up is slower than advancing

DIST_SMOOTH = 0.25              # lower = smoother but laggier

# --- safety ---
LOST_TIMEOUT = 5              # seconds without a lock before stopping
COAST_LIMIT = 2                 # frames of coasting still acted on
MAX_RUN_SECONDS = 120           # hard stop, then land


# ---------------------------------------------------------------- helpers

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clamp(value, low, high):
    return max(low, min(high, value))


# ------------------------------------------------------------ controllers

class YawController:
    """offset_x -> yaw rate. Positive rate = clockwise seen from above."""

    def __init__(self, kp=YAW_KP, deadband=YAW_DEADBAND,
                 rate_max=YAW_RATE_MAX):
        self.kp = kp
        self.deadband = deadband
        self.rate_max = rate_max

    def update(self, offset_x):
        if abs(offset_x) < self.deadband:
            return 0.0
        error = offset_x - (self.deadband if offset_x > 0 else -self.deadband)
        return clamp(self.kp * error, -self.rate_max, self.rate_max)


class ForwardController:
    """distance -> forward velocity. Holds TARGET_DISTANCE."""

    def __init__(self, target=TARGET_DISTANCE, kp=FWD_KP,
                 deadband=FWD_DEADBAND):
        self.target = target
        self.kp = kp
        self.deadband = deadband
        self.smoothed = None

    def reset(self):
        self.smoothed = None

    def update(self, distance_m):
        if distance_m is None:
            return 0.0, None

        if self.smoothed is None:
            self.smoothed = distance_m
        else:
            a = DIST_SMOOTH
            self.smoothed = a * distance_m + (1 - a) * self.smoothed

        error = self.smoothed - self.target      # positive = too far away
        if abs(error) < self.deadband:
            return 0.0, self.smoothed

        error -= self.deadband if error > 0 else -self.deadband
        speed = clamp(self.kp * error, FWD_SPEED_MIN, FWD_SPEED_MAX)
        return speed, self.smoothed


# --------------------------------------------------------- flight control

class FlightLink:
    """Thin wrapper over the MAVLink connection."""

    def __init__(self, port=SERIAL_PORT, baud=BAUD_RATE):
        log(f"Opening {port} at {baud}...")
        self.link = mavutil.mavlink_connection(
            port, baud=baud,
            source_system=SOURCE_SYSTEM,
            source_component=SOURCE_COMPONENT)
        self.link.wait_heartbeat(timeout=15)
        if self.link.target_system == 0:
            raise RuntimeError("No heartbeat from flight controller.")
        log(f"Heartbeat from system {self.link.target_system}")

        self._request(mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 4)
        self._request(mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT, 2)

    def _request(self, msg_id, hz):
        self.link.mav.command_long_send(
            self.link.target_system, self.link.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0, msg_id, int(1e6 / hz), 0, 0, 0, 0, 0)

    def preflight(self, timeout=60):
        log("Preflight checks...")
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.link.recv_match(type="GPS_RAW_INT",
                                       blocking=True, timeout=5)
            if msg is None:
                continue
            if msg.fix_type >= 3 and msg.satellites_visible >= MIN_SATELLITES:
                log(f"GPS ready: fix {msg.fix_type}, "
                    f"{msg.satellites_visible} sats")
                return True
            log(f"Waiting on GPS: fix {msg.fix_type}, "
                f"{msg.satellites_visible} sats")
        log("PREFLIGHT FAILED — GPS never usable.")
        return False

    def set_mode(self, mode_number, name, timeout=10):
        log(f"Requesting {name}...")
        self.link.mav.set_mode_send(
            self.link.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_number)
        deadline = time.time() + timeout
        while time.time() < deadline:
            hb = self.link.recv_match(type="HEARTBEAT",
                                      blocking=True, timeout=2)
            if hb and hb.custom_mode == mode_number:
                log(f"{name} confirmed.")
                return True
        log(f"FAILED to enter {name}.")
        return False

    def arm(self, timeout=10):
        log("Arming...")
        self.link.mav.command_long_send(
            self.link.target_system, self.link.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0)
        ack = self.link.recv_match(type="COMMAND_ACK",
                                   blocking=True, timeout=timeout)
        if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            log("Armed.")
            return True
        log(f"Arm REJECTED"
            f"{f' (result {ack.result})' if ack else ' — no ack'}.")
        return False

    def disarm(self):
        self.link.mav.command_long_send(
            self.link.target_system, self.link.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0)

    def takeoff(self, altitude, timeout=10):
        log(f"Takeoff to {altitude} m...")
        self.link.mav.command_long_send(
            self.link.target_system, self.link.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude)
        ack = self.link.recv_match(type="COMMAND_ACK",
                                   blocking=True, timeout=timeout)
        if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            log("Takeoff accepted.")
            return True
        log("Takeoff REJECTED.")
        return False

    def wait_for_altitude(self, target, timeout=30):
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.link.recv_match(type="GLOBAL_POSITION_INT",
                                       blocking=True, timeout=2)
            if msg and msg.relative_alt / 1000.0 >= target - ALT_TOLERANCE:
                log(f"Reached {msg.relative_alt / 1000.0:.2f} m.")
                return True
        log("Timed out climbing.")
        return False

    def send_command(self, vx, yaw_rate):
        """Body-frame forward velocity plus yaw rate."""
        self.link.mav.set_position_target_local_ned_send(
            0,
            self.link.target_system, self.link.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            VEL_YAWRATE_MASK,
            0, 0, 0,                # position (ignored)
            vx, 0, 0,               # velocity: forward only
            0, 0, 0,                # acceleration (ignored)
            0, yaw_rate)            # yaw angle (ignored), yaw rate

    def altitude(self):
        msg = self.link.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
        return msg.relative_alt / 1000.0 if msg else None

    def land(self, timeout=60):
        log("Landing...")
        self.set_mode(LAND_MODE, "LAND")
        deadline = time.time() + timeout
        while time.time() < deadline:
            hb = self.link.recv_match(type="HEARTBEAT",
                                      blocking=True, timeout=2)
            if hb and not (hb.base_mode &
                           mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
                log("Landed and disarmed.")
                return True
        log("Land timed out — disarming.")
        self.disarm()
        return False


# ------------------------------------------------------------- main loop

def follow_loop(fc, cap, tracker, hsv_vals, streamer, args):
    """Runs until the ball is gone too long, time runs out, or Ctrl+C."""
    import cv2

    yaw_ctl = YawController()
    fwd_ctl = ForwardController(target=args.distance)

    scale = detector.CAPTURE_W / detector.DETECT_W
    detect_h = int(detector.CAPTURE_H / scale)

    interval = 1.0 / SEND_HZ
    last_lock = time.time()
    start = time.time()
    next_report = 0.0
    coast_frames = 0

    while True:
        now = time.time()

        if now - start > MAX_RUN_SECONDS:
            log(f"Reached {MAX_RUN_SECONDS}s limit — stopping.")
            return "timeout"

        ok, frame = cap.read()
        if not ok:
            log("Frame grab failed — commanding stop.")
            if fc:
                fc.send_command(0.0, 0.0)
            time.sleep(interval)
            continue

        small = cv2.resize(frame, (detector.DETECT_W, detect_h),
                           interpolation=cv2.INTER_NEAREST)
        result, _ = tracker.update(small, hsv_vals)

        # --- decide whether this reading is usable -------------------
        usable = False
        if result is not None:
            if result["coasting"]:
                coast_frames += 1
                usable = coast_frames <= COAST_LIMIT
            else:
                coast_frames = 0
                usable = result["confident"]

        if usable:
            last_lock = now
            yaw_rate = yaw_ctl.update(result["offset_x"])
            if args.yaw_only:
                vx, smoothed = 0.0, None
            else:
                vx, smoothed = fwd_ctl.update(result["distance_m"])
            state = "COAST" if result["coasting"] else "TRACK"
        else:
            # No usable lock: command a stop, but KEEP SENDING. Going
            # silent for 3 s makes the FC drop out of velocity control.
            yaw_rate, vx, smoothed = 0.0, 0.0, None
            fwd_ctl.reset()
            state = "SEARCH"

            if now - last_lock > LOST_TIMEOUT:
                log(f"Ball lost for {LOST_TIMEOUT}s — stopping.")
                if fc:
                    fc.send_command(0.0, 0.0)
                return "lost"

        if fc:
            fc.send_command(vx, yaw_rate)

        # --- reporting ------------------------------------------------
        if now >= next_report:
            off = f"{result['offset_x']:+.2f}" if result else "  -- "
            dist = f"{smoothed:.2f}m" if smoothed else "--"
            alt = fc.altitude() if fc else None
            alt_s = f" alt {alt:.2f}m" if alt else ""
            log(f"{state:6s} x {off}  d {dist:>6s}  "
                f"-> vx {vx:+.2f} yaw {yaw_rate:+.2f}{alt_s}")
            next_report = now + 0.5

        if streamer:
            annotated = detector.draw(
                frame.copy(), result, tracker, scale, 0, 0,
                hsv_vals["h_min"] > hsv_vals["h_max"])
            cv2.putText(annotated,
                        f"{state}  vx {vx:+.2f}  yaw {yaw_rate:+.2f}",
                        (12, annotated.shape[0] - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            streamer.update(annotated)

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="never arm — enter GUIDED and stream commands")
    parser.add_argument("--no-fc", action="store_true",
                        help="no flight controller at all, just print "
                             "what would be commanded")
    parser.add_argument("--yaw-only", action="store_true",
                        help="disable forward control")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--alt", type=float, default=TAKEOFF_ALT)
    parser.add_argument("--distance", type=float, default=TARGET_DISTANCE,
                        help="metres to hold from the ball")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    cfg = detector.load_config()
    hsv_vals = cfg["hsv"]
    tracker = detector.BallTracker(focal_px=cfg["focal_px"])

    if cfg["focal_px"] is None and not args.yaw_only:
        log("No focal length calibrated — forward control needs distance.")
        log("Run ball.py and press 'c', or use --yaw-only.")
        sys.exit(1)

    log(f"Mode: {'YAW ONLY' if args.yaw_only else 'YAW + FORWARD'}")
    if not args.yaw_only:
        log(f"Holding {args.distance} m")

    cap = detector.open_camera(args.camera)
    if cap is None:
        log(f"Could not open camera {args.camera}.")
        sys.exit(1)

    streamer = None
    if args.stream:
        streamer = detector.MJPEGStreamer()
        streamer.start()

    fc = None
    try:
        if not args.no_fc:
            fc = FlightLink()
            if not fc.preflight():
                sys.exit(1)
            if not fc.set_mode(GUIDED_MODE, "GUIDED"):
                sys.exit(1)

        if args.no_fc or args.dry_run:
            log("")
            log("DRY RUN — motors will NOT spin.")
            log("Move the ball around and watch the commanded values.")
            log("Ctrl+C to stop.")
            log("")
            follow_loop(fc, cap, tracker, hsv_vals, streamer, args)
            return

        log("")
        log("=" * 56)
        log("  ARMING IN 5 SECONDS — Ctrl+C to abort")
        log("  Transmitter ready? SE switch to AltHold takes control back.")
        log("=" * 56)
        log("")
        time.sleep(5)

        if not fc.arm():
            sys.exit(1)
        if not fc.takeoff(args.alt):
            fc.disarm()
            sys.exit(1)
        if not fc.wait_for_altitude(args.alt):
            fc.land()
            sys.exit(1)

        log("Tracking.")
        reason = follow_loop(fc, cap, tracker, hsv_vals, streamer, args)
        log(f"Loop ended ({reason}).")
        fc.land()

    except KeyboardInterrupt:
        log("\nInterrupted.")
        if fc and not (args.dry_run or args.no_fc):
            log("TAKE MANUAL CONTROL — landing.")
            fc.send_command(0.0, 0.0)
            fc.land()
    finally:
        cap.release()
        if streamer:
            streamer.stop()


if __name__ == "__main__":
    main()