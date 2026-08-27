#!/usr/bin/env python3
"""
ball.py

Blue printed sphere detection for drone tracking.
150 mm ball, filament #0086D6.
Outdoor daylight, USB webcam (105 deg FOV), robustness first.

Self-contained — MJPEG streaming is built in, no separate module.

Keys (GUI mode):
    q   quit
    t   cycle view: normal -> mask -> rejected candidates
    s   save config
    c   distance calibration (hold ball at a known distance)
    r   reset the lock

Usage:
    python ball.py                          # laptop, tuning sliders
    python ball.py --camera 1               # pick a different camera
    python ball.py --no-gui --stream        # Pi: view at http://<pi-ip>:8080
    python ball.py --bench --no-gui         # framerate benchmark
    python ball.py --image ball.jpg         # tune against a photo

Find the Pi's IP with:  hostname -I
"""

import argparse
import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

# ---------------------------------------------------------------- config

CONFIG_FILE = "blue_ball_config.json"

CAPTURE_W, CAPTURE_H = 1920, 1080

# Detection downscale. 960 balances range against CPU.
# Drop to 640 if the Pi can't keep up.
# NOTE: focal_px is tied to this value — recalibrate if you change it.
DETECT_W = 960

BALL_DIAMETER_M = 0.15          # printed sphere, 150 mm

# --- robustness gates -------------------------------------------------
MIN_FILL_RATIO = 0.62           # reject non-round blobs
MIN_RADIUS_PX = 5
MAX_RADIUS_FRAC = 0.45          # blob this big is background, not ball
MAX_ASPECT = 1.45
# Blue ball against blue sky is the main false-positive risk outdoors.
# Sky is washed out (low saturation); the filament is fully saturated,
# so this gate separates them cheaply. Raise it if sky sneaks through.
MIN_MEAN_SAT = 110

# Large blobs get held to a stricter saturation bar. A 15 cm ball only
# fills a big fraction of the frame when it is very close; anything
# large AND less than fully saturated is background (sky, a wall, a
# blue tarp), not the ball.
LARGE_BLOB_FRAC = 0.15          # radius > this fraction of frame width
LARGE_BLOB_MIN_SAT = 180

LOCK_FRAMES = 3                 # consecutive hits before "locked"
COAST_FRAMES = 5                # misses tolerated before dropping lock
MAX_JUMP_FRAC = 0.35            # candidate must be near last position

# Filament #0086D6 converts to HSV (101, 255, 214) on the OpenCV scale.
# Hue stays roughly constant across a curved surface; saturation and
# value are what spread, so the H window is tight and S/V are wide.
# h_min < h_max here, so the normal (non-wrapping) path is used.
DEFAULT_HSV = {
    "h_min": 88,  "h_max": 118,
    "s_min": 100, "s_max": 255,
    "v_min": 50,  "v_max": 255,
}

# A matte printed surface has little specular highlight, unlike the
# glossy billiard ball. 2 is enough; raise it if the mask shows holes.
CLOSE_ITERATIONS = 2

DEFAULT_FOCAL_PX = None

# --- streaming --------------------------------------------------------
STREAM_PORT = 8080
STREAM_WIDTH = 640              # downscale before encoding — WiFi
JPEG_QUALITY = 70


# ------------------------------------------------------------- config io

def load_config():
    cfg = {"hsv": dict(DEFAULT_HSV), "focal_px": DEFAULT_FOCAL_PX}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                saved = json.load(f)
            cfg["hsv"].update(saved.get("hsv", {}))
            cfg["focal_px"] = saved.get("focal_px", cfg["focal_px"])
            print(f"Loaded {CONFIG_FILE}")
        except (json.JSONDecodeError, OSError) as e:
            print(f"Could not read {CONFIG_FILE} ({e}), using defaults")
    return cfg


def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved {CONFIG_FILE}")


# --------------------------------------------------------------- masking

def build_mask(hsv_img, v):
    """Threshold with automatic hue wraparound.

    Normal:  h_min <= h <= h_max
    Wrapped: h >= h_min OR h <= h_max   (when h_min > h_max)
    """
    if v["h_min"] <= v["h_max"]:
        lower = np.array([v["h_min"], v["s_min"], v["v_min"]])
        upper = np.array([v["h_max"], v["s_max"], v["v_max"]])
        return cv2.inRange(hsv_img, lower, upper)

    lo1 = np.array([v["h_min"], v["s_min"], v["v_min"]])
    hi1 = np.array([179,        v["s_max"], v["v_max"]])
    lo2 = np.array([0,          v["s_min"], v["v_min"]])
    hi2 = np.array([v["h_max"], v["s_max"], v["v_max"]])
    return cv2.bitwise_or(cv2.inRange(hsv_img, lo1, hi1),
                          cv2.inRange(hsv_img, lo2, hi2))


def clean_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                            iterations=CLOSE_ITERATIONS)
    return mask


# ------------------------------------------------------------- detection

class BallTracker:
    """Stateful tracker. A single frame can be fooled; a consistent
    track across several frames is much harder to fool."""

    def __init__(self, focal_px=None):
        self.focal_px = focal_px
        self.last_pos = None
        self.last_radius = None
        self.hit_streak = 0
        self.miss_streak = 0
        self.locked = False
        self.rejected = []

    def reset(self):
        self.last_pos = None
        self.last_radius = None
        self.hit_streak = 0
        self.miss_streak = 0
        self.locked = False

    def _evaluate(self, contour, hsv_img, frame_w):
        """Returns (candidate, reject_reason) — one is always None."""
        area = cv2.contourArea(contour)
        if area < 40:
            return None, "tiny"

        (cx, cy), radius = cv2.minEnclosingCircle(contour)

        if radius < MIN_RADIUS_PX:
            return None, "small"
        if radius > frame_w * MAX_RADIUS_FRAC:
            return None, "huge"

        circle_area = np.pi * radius * radius
        fill = area / circle_area if circle_area > 0 else 0.0
        if fill < MIN_FILL_RATIO:
            return None, f"shape {fill:.2f}"

        _, _, w, h = cv2.boundingRect(contour)
        aspect = max(w / h, h / w) if h > 0 and w > 0 else 99
        if aspect > MAX_ASPECT:
            return None, f"aspect {aspect:.2f}"

        blob = np.zeros(hsv_img.shape[:2], np.uint8)
        cv2.drawContours(blob, [contour], -1, 255, -1)
        _, mean_s, mean_v = cv2.mean(hsv_img, mask=blob)[:3]

        if mean_s < MIN_MEAN_SAT:
            return None, f"pale s={mean_s:.0f}"

        if radius > frame_w * LARGE_BLOB_FRAC and mean_s < LARGE_BLOB_MIN_SAT:
            return None, f"sky s={mean_s:.0f}"

        return {
            "cx": cx, "cy": cy, "radius": radius, "area": area,
            "fill": fill, "mean_s": mean_s, "mean_v": mean_v,
        }, None

    def _score(self, cand):
        score = cand["fill"] * 100.0
        score += min(cand["radius"], 80) * 0.5
        score += cand["mean_s"] * 0.1
        if self.last_pos is not None:
            dist = np.hypot(cand["cx"] - self.last_pos[0],
                            cand["cy"] - self.last_pos[1])
            score += max(0.0, 60.0 - dist * 0.6)
        return score

    def update(self, frame_small, hsv_vals):
        """Returns (result_dict_or_None, mask)."""
        h, w = frame_small.shape[:2]
        self.rejected = []

        blurred = cv2.GaussianBlur(frame_small, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = clean_mask(build_mask(hsv, hsv_vals))

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in contours:
            cand, reason = self._evaluate(c, hsv, w)
            if cand is not None:
                candidates.append(cand)
            elif reason != "tiny":
                (rx, ry), rr = cv2.minEnclosingCircle(c)
                self.rejected.append((rx, ry, rr, reason))

        if self.locked and self.last_pos is not None:
            max_jump = w * MAX_JUMP_FRAC
            near = [c for c in candidates
                    if np.hypot(c["cx"] - self.last_pos[0],
                                c["cy"] - self.last_pos[1]) < max_jump]
            if near:
                candidates = near

        if not candidates:
            return self._handle_miss(w, h), mask

        best = max(candidates, key=self._score)
        return self._handle_hit(best, w, h), mask

    def _handle_hit(self, cand, w, h):
        self.hit_streak += 1
        self.miss_streak = 0

        # Light smoothing — damps jitter without adding much lag.
        if self.last_pos is not None:
            a = 0.6
            cand["cx"] = a * cand["cx"] + (1 - a) * self.last_pos[0]
            cand["cy"] = a * cand["cy"] + (1 - a) * self.last_pos[1]
            cand["radius"] = a * cand["radius"] + (1 - a) * self.last_radius

        self.last_pos = (cand["cx"], cand["cy"])
        self.last_radius = cand["radius"]

        if self.hit_streak >= LOCK_FRAMES:
            self.locked = True

        return self._build(cand, w, h, confident=self.locked)

    def _handle_miss(self, w, h):
        self.hit_streak = 0
        self.miss_streak += 1

        if self.locked and self.miss_streak <= COAST_FRAMES:
            cand = {"cx": self.last_pos[0], "cy": self.last_pos[1],
                    "radius": self.last_radius, "fill": 0.0}
            out = self._build(cand, w, h, confident=False)
            out["coasting"] = True
            return out

        if self.miss_streak > COAST_FRAMES:
            self.reset()
        return None

    def _build(self, cand, w, h, confident):
        radius = cand["radius"]
        distance = None
        if self.focal_px and radius > 0:
            # focal_px is in DETECT-space pixels, same as radius.
            distance = (BALL_DIAMETER_M * self.focal_px) / (2.0 * radius)

        return {
            "cx": cand["cx"], "cy": cand["cy"],
            "offset_x": (cand["cx"] - w / 2) / (w / 2),
            "offset_y": (cand["cy"] - h / 2) / (h / 2),
            "radius": radius,
            "distance_m": distance,
            "fill": cand["fill"],
            "confident": confident,
            "coasting": False,
        }


# ------------------------------------------------------------- streaming

_PAGE = b"""<!DOCTYPE html>
<html><head><title>drone cam</title>
<style>
  body { margin:0; background:#111; color:#ccc;
         font-family:monospace; text-align:center; }
  img  { max-width:100%; height:auto; margin-top:8px; }
  p    { font-size:13px; }
</style></head>
<body>
  <img src="/stream.mjpg">
  <p>live detection feed &mdash; refresh if it stalls</p>
</body></html>
"""


class _FrameStore:
    """Latest frame, shared between capture loop and HTTP threads."""

    def __init__(self):
        self._jpeg = None
        self._lock = threading.Condition()
        self._seq = 0

    def set(self, jpeg_bytes):
        with self._lock:
            self._jpeg = jpeg_bytes
            self._seq += 1
            self._lock.notify_all()

    def wait_for_next(self, last_seq, timeout=5.0):
        with self._lock:
            if self._seq == last_seq:
                self._lock.wait(timeout)
            return self._jpeg, self._seq


class _Handler(BaseHTTPRequestHandler):
    frame_store = None

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(_PAGE)))
            self.end_headers()
            self.wfile.write(_PAGE)
            return

        if self.path != "/stream.mjpg":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()

        seq = -1
        try:
            while True:
                jpeg, seq = self.frame_store.wait_for_next(seq)
                if jpeg is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--FRAME\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass        # browser tab closed — normal

    def log_message(self, fmt, *args):
        pass


class _Server(ThreadingHTTPServer):
    # ThreadingHTTPServer already mixes in ThreadingMixIn — inheriting
    # both is an MRO conflict.
    allow_reuse_address = True
    daemon_threads = True


class MJPEGStreamer:
    """Non-blocking MJPEG server. update() never blocks on clients."""

    def __init__(self, port=STREAM_PORT, width=STREAM_WIDTH,
                 quality=JPEG_QUALITY, every_n=1):
        self.port = port
        self.width = width
        self.quality = quality
        self.every_n = max(1, every_n)
        self._count = 0
        self._store = _FrameStore()
        self._server = None
        self._thread = None

    def start(self):
        handler = type("Handler", (_Handler,), {"frame_store": self._store})
        self._server = _Server(("0.0.0.0", self.port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"Streaming at http://{self._local_ip()}:{self.port}")

    def update(self, frame_bgr):
        self._count += 1
        if self._count % self.every_n:
            return

        h, w = frame_bgr.shape[:2]
        if w > self.width:
            s = self.width / w
            frame_bgr = cv2.resize(frame_bgr, (self.width, int(h * s)),
                                   interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(".jpg", frame_bgr,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if ok:
            self._store.set(buf.tobytes())

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()

    @staticmethod
    def _local_ip():
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))      # no packets sent, just routing
            return s.getsockname()[0]
        except OSError:
            return "<pi-ip>"
        finally:
            s.close()


# ---------------------------------------------------------------- camera

def open_camera(index):
    """USB webcams default to raw YUYV, very slow at 1080p.
    MJPG is usually a 3-5x framerate win."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # newest frame, not queued
    except Exception:
        pass

    raw = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((raw >> (8 * i)) & 0xFF) for i in range(4))
    print(f"Camera: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
          f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} "
          f"fourcc={fourcc} fps={cap.get(cv2.CAP_PROP_FPS):.0f}")
    if fourcc.strip("\x00") != "MJPG":
        print("  WARNING: not MJPG — framerate will be poor. Check with: "
              "v4l2-ctl --list-formats-ext")
    return cap


# -------------------------------------------------------------------- gui

def make_trackbars(window, hsv):
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 420, 280)
    for key, value in hsv.items():
        limit = 179 if key.startswith("h") else 255
        cv2.createTrackbar(key, window, value, limit, lambda _: None)


def read_trackbars(window, keys):
    return {k: cv2.getTrackbarPos(k, window) for k in keys}


def draw(frame, result, tracker, scale, fps, view, wrapped):
    h, w = frame.shape[:2]
    cv2.drawMarker(frame, (w // 2, h // 2), (180, 180, 180),
                   cv2.MARKER_CROSS, 24, 1)

    if view == 2:
        for rx, ry, rr, reason in tracker.rejected:
            px, py, pr = int(rx * scale), int(ry * scale), int(rr * scale)
            cv2.circle(frame, (px, py), pr, (0, 0, 255), 1)
            cv2.putText(frame, reason, (px - 24, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    cv2.putText(frame, f"{fps:.0f} fps", (w - 130, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if wrapped:
        cv2.putText(frame, "hue: wrapped", (w - 190, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if result is None:
        cv2.putText(frame, "SEARCHING", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame

    cx, cy = int(result["cx"] * scale), int(result["cy"] * scale)
    radius = max(int(result["radius"] * scale), 2)

    if result["coasting"]:
        color, label = (0, 165, 255), "COASTING"
    elif result["confident"]:
        color, label = (0, 255, 0), "LOCKED"
    else:
        color, label = (0, 220, 220), "acquiring"

    cv2.circle(frame, (cx, cy), radius, color, 2)
    cv2.circle(frame, (cx, cy), 4, color, -1)
    cv2.line(frame, (w // 2, h // 2), (cx, cy), color, 1)

    dist = result["distance_m"]
    lines = [
        label,
        f"x {result['offset_x']:+.2f}  y {result['offset_y']:+.2f}",
        f"dist {dist:.2f} m" if dist else "dist -- (press c)",
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (12, 34 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


# ------------------------------------------------------------ calibration

def calibrate_focal(radius_detect_px):
    """Solve for focal length from one known distance.

    Everything stays in DETECT-space pixels so it matches the radius
    used in _build(). Do NOT scale to full-frame here.
    """
    try:
        known = float(input("\nHold the ball still. Distance in metres: "))
    except (ValueError, EOFError):
        print("Bad input — calibration skipped.")
        return None
    if known <= 0:
        print("Distance must be positive — calibration skipped.")
        return None

    focal = (2.0 * radius_detect_px * known) / BALL_DIAMETER_M
    print(f"focal_px = {focal:.1f}  "
          f"(radius {radius_detect_px:.1f}px at {known} m)")
    return focal


# ------------------------------------------------------------------- main

def run_image_mode(args, cfg, tracker, gui):
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Could not read {args.image}")
        return

    hsv_vals = cfg["hsv"]
    view = 0
    scale = frame.shape[1] / DETECT_W
    small = cv2.resize(frame, (DETECT_W, int(frame.shape[0] / scale)))

    while True:
        if gui:
            hsv_vals = read_trackbars("hsv", hsv_vals.keys())

        # No reset() here — the tracker needs consecutive hits to lock,
        # so resetting each iteration would pin it at "acquiring".
        result, mask = tracker.update(small, hsv_vals)
        wrapped = hsv_vals["h_min"] > hsv_vals["h_max"]

        if view == 1:
            disp = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        else:
            disp = draw(frame.copy(), result, tracker, scale, 0, view, wrapped)

        cv2.imshow("track", disp)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("t"):
            view = (view + 1) % 3
        elif k == ord("r"):
            tracker.reset()
        elif k == ord("s"):
            cfg["hsv"] = hsv_vals
            save_config(cfg)
        elif k == ord("c"):
            if result:
                focal = calibrate_focal(result["radius"])
                if focal:
                    tracker.focal_px = focal
                    cfg["focal_px"] = focal
                    cfg["hsv"] = hsv_vals
                    save_config(cfg)
            else:
                print("No ball detected — can't calibrate.")

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image", type=str)
    parser.add_argument("--no-gui", action="store_true",
                        help="headless — prints values only")
    parser.add_argument("--bench", action="store_true",
                        help="measure framerate and exit (implies --no-gui)")
    parser.add_argument("--stream", action="store_true",
                        help=f"serve frames at http://<ip>:{STREAM_PORT}")
    parser.add_argument("--stream-every", type=int, default=1,
                        help="send every Nth frame (raise if WiFi lags)")
    args = parser.parse_args()

    # Bench creates a slider window it never services, which hangs.
    gui = not (args.no_gui or args.bench)

    cfg = load_config()
    hsv_vals = cfg["hsv"]
    tracker = BallTracker(focal_px=cfg["focal_px"])
    view = 0

    if cfg["focal_px"] is None:
        print("No focal length yet — press 'c' with the ball visible.")

    if gui:
        make_trackbars("hsv", hsv_vals)

    if args.image:
        run_image_mode(args, cfg, tracker, gui)
        return

    cap = open_camera(args.camera)
    if cap is None:
        print(f"Could not open camera {args.camera}. "
              f"Try --camera 1 or --camera 2.")
        return

    streamer = None
    if args.stream:
        streamer = MJPEGStreamer(every_n=args.stream_every)
        streamer.start()

    times = deque(maxlen=30)
    scale = CAPTURE_W / DETECT_W
    detect_h = int(CAPTURE_H / scale)
    frames, start = 0, time.time()

    if gui:
        print("q=quit  t=view  s=save  c=calibrate  r=reset lock")
    else:
        print("Ctrl+C to stop.")

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed.")
                break

            small = cv2.resize(frame, (DETECT_W, detect_h),
                               interpolation=cv2.INTER_NEAREST)
            if gui:
                hsv_vals = read_trackbars("hsv", hsv_vals.keys())

            result, mask = tracker.update(small, hsv_vals)

            times.append(time.time() - t0)
            fps = 1.0 / (sum(times) / len(times)) if times else 0.0
            frames += 1
            wrapped = hsv_vals["h_min"] > hsv_vals["h_max"]

            if args.bench:
                if frames >= 150:
                    el = time.time() - start
                    print(f"\n{frames} frames in {el:.1f}s = "
                          f"{frames / el:.1f} fps (detect {DETECT_W}px)")
                    break
                continue

            # Annotated frame is needed for the GUI, the stream, or both.
            annotated = None
            if gui or streamer:
                if view == 1:
                    annotated = cv2.cvtColor(
                        cv2.resize(mask, (CAPTURE_W, CAPTURE_H)),
                        cv2.COLOR_GRAY2BGR)
                else:
                    annotated = draw(frame.copy(), result, tracker,
                                     scale, fps, view, wrapped)

            if streamer:
                streamer.update(annotated)

            if not gui:
                if result and result["confident"]:
                    d = result["distance_m"]
                    dtxt = f" d {d:.2f}m" if d else ""
                    print(f"LOCK x {result['offset_x']:+.2f} "
                          f"y {result['offset_y']:+.2f} "
                          f"r {result['radius']:.0f}{dtxt}")
                else:
                    print("no lock")
                continue

            cv2.imshow("track", annotated)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("t"):
                view = (view + 1) % 3
            elif k == ord("r"):
                tracker.reset()
                print("Lock reset.")
            elif k == ord("s"):
                cfg["hsv"] = hsv_vals
                cfg["focal_px"] = tracker.focal_px
                save_config(cfg)
            elif k == ord("c"):
                if result:
                    focal = calibrate_focal(result["radius"])
                    if focal:
                        tracker.focal_px = focal
                        cfg["focal_px"] = focal
                        cfg["hsv"] = hsv_vals
                        save_config(cfg)
                else:
                    print("No ball detected — can't calibrate.")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        if streamer:
            streamer.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()