"""
Smoking Detection System — Real-Time Edition (Smooth & Strict)
Inference runs locally in a background thread for zero network latency.
Includes a temporal frame buffer for accuracy and a cooldown timer for smooth UI.

Requirements: pip install inference opencv-python numpy
Usage:
    python smoking_detection.py                      (webcam)
    python smoking_detection.py --source video.mp4   (video)
    python smoking_detection.py --source image.jpg   (image)
Press Q to quit.
"""

import cv2
import numpy as np
import argparse
import time
import threading
import queue
import base64
import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

load_dotenv()

# ── CONFIG ──────────────────────────────
API_KEY        = os.getenv("ROBOFLOW_API_KEY", "")
MODEL_ID       = "smoking-detection-ulksv/3"

# CONFIDENCE: Balanced for better cigarette detection while relying on 
# the frame buffer below to filter out the false positives.
CIGARETTE_CONF = 0.25   # Lowered so it doesn't struggle to see real cigarettes
FACE_CONF      = 0.30   # Kept strict because face detection is flawless
FACE_EXPAND    = 1.50   # Proximity zone multiplier

# TEMPORAL BUFFER: Must see cigarette N times in a row to trigger alert.
REQUIRED_FRAMES = 2

# COOLDOWN: How long the "SMOKING DETECTED" alert stays on screen (in seconds)
# after the cigarette is temporarily hidden or drops below confidence.
COOLDOWN_SECONDS = 1.5

# Delay between consecutive inference calls (ms).
INFER_WIDTH = 416
INFER_INTERVAL_MS = 0
NUM_WORKERS = 2
API_URL = "https://serverless.roboflow.com"


def frame_to_base64(frame):
    h, w = frame.shape[:2]
    if w > INFER_WIDTH:
        scale = INFER_WIDTH / w
        frame = cv2.resize(frame, (INFER_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf).decode('utf-8')

def scale_predictions(predictions, orig_w, orig_h):
    scale = orig_w / INFER_WIDTH if orig_w > INFER_WIDTH else 1.0
    scaled = []
    for p in predictions:
        p = dict(p)
        p['x'] *= scale
        p['y'] *= scale
        p['width'] *= scale
        p['height'] *= scale
        scaled.append(p)
    return scaled

# Colors (BGR)
C_FACE  = (0, 200, 255)
C_CIG   = (0, 230,  80)
C_RED   = (30,  30, 220)
C_GREEN = (60, 180,  60)
C_WHITE = (255, 255, 255)
C_BLACK = (0,   0,   0)
C_DARK  = (20,  20,  20)
C_DIM   = (150, 150, 150)

# ── Shared state between threads ────────
class InferenceState:
    def __init__(self):
        self.lock              = threading.Lock()
        self.faces             = []
        self.cigs              = []
        self.smoking_streak    = 0
        self.last_smoking_time = 0.0
        self.frame_queue       = queue.Queue(maxsize=2)
        self.stop              = threading.Event()

STATE = InferenceState()
# ────────────────────────────────────────

def expand_box(x1, y1, x2, y2, factor, img_w, img_h):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = (x2 - x1) * factor
    h = (y2 - y1) * factor
    return (
        max(0,     int(cx - w / 2)),
        max(0,     int(cy - h / 2)),
        min(img_w, int(cx + w / 2)),
        min(img_h, int(cy + h / 2)),
    )

def boxes_overlap(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

def parse_predictions(predictions, img_w, img_h):
    faces, cigarettes = [], []
    for pred in predictions:
        cls  = pred["class"].lower()
        conf = pred["confidence"]
        cx, cy = pred["x"], pred["y"]
        w,  h  = pred["width"], pred["height"]
        
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)
        
        if cls == "face" and conf >= FACE_CONF:
            faces.append((x1, y1, x2, y2, conf))
        elif cls == "cigarette" and conf >= CIGARETTE_CONF:
            cigarettes.append((x1, y1, x2, y2, conf))
    return faces, cigarettes

def check_smoking(faces, cigarettes, img_w, img_h):
    for (fx1, fy1, fx2, fy2, _) in faces:
        ex1, ey1, ex2, ey2 = expand_box(fx1, fy1, fx2, fy2, FACE_EXPAND, img_w, img_h)
        for (cx1, cy1, cx2, cy2, _) in cigarettes:
            if boxes_overlap(ex1, ey1, ex2, ey2, cx1, cy1, cx2, cy2):
                return True
    return False

# ── Background inference workers ────────
def inference_worker():
    client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)
    while not STATE.stop.is_set():
        try:
            frame = STATE.frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        img_h, img_w = frame.shape[:2]
        try:
            b64 = frame_to_base64(frame)
            result = client.infer(b64, model_id=MODEL_ID)
            preds = result.get('predictions', [])
            preds = scale_predictions(preds, img_w, img_h)

            faces, cigs = parse_predictions(preds, img_w, img_h)
            is_smoking_now = check_smoking(faces, cigs, img_w, img_h)

            with STATE.lock:
                STATE.faces = faces
                STATE.cigs  = cigs
                if is_smoking_now:
                    STATE.smoking_streak += 1
                    if STATE.smoking_streak >= REQUIRED_FRAMES:
                        STATE.last_smoking_time = time.time()
                else:
                    STATE.smoking_streak = 0

        except Exception as e:
            print(f"[WARN] Inference error: {e}")

        if INFER_INTERVAL_MS > 0:
            time.sleep(INFER_INTERVAL_MS / 1000.0)
# ────────────────────────────────────────

def draw_label(frame, text, x, y, bg, fg=C_WHITE):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 10, y), bg, -1)
    cv2.putText(frame, text, (x + 5, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, fg, 1, cv2.LINE_AA)

def draw_ui(frame, faces, cigarettes, is_smoking_alert, fps):
    img_h, img_w = frame.shape[:2]

    for (x1, y1, x2, y2, conf) in faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), C_FACE, 2)
        draw_label(frame, f"Face {conf:.0%}", x1, y1, C_FACE, C_BLACK)

    for (x1, y1, x2, y2, conf) in cigarettes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), C_CIG, 2)
        draw_label(frame, f"Cigarette {conf:.0%}", x1, y1, C_CIG, C_BLACK)

    # Top status bar
    bar_h = 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (img_w, bar_h), C_DARK, -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    if is_smoking_alert:
        pulse_r = 10 + int(abs(np.sin(time.time() * 5)) * 5)
        cv2.circle(frame, (22, bar_h // 2), pulse_r, C_RED, -1)
        cv2.putText(frame, "SMOKING DETECTED", (42, bar_h // 2 + 7),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, C_RED, 2, cv2.LINE_AA)
        cv2.line(frame, (0, bar_h), (img_w, bar_h), C_RED, 2)
    elif cigarettes and not faces:
        cv2.circle(frame, (22, bar_h // 2), 8, C_CIG, -1)
        cv2.putText(frame, "CIGARETTE DETECTED", (42, bar_h // 2 + 7),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, C_CIG, 2, cv2.LINE_AA)
        cv2.line(frame, (0, bar_h), (img_w, bar_h), C_CIG, 2)
    else:
        cv2.circle(frame, (22, bar_h // 2), 8, C_GREEN, -1)
        cv2.putText(frame, "NO SMOKING DETECTED", (42, bar_h // 2 + 7),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, C_GREEN, 2, cv2.LINE_AA)
        cv2.line(frame, (0, bar_h), (img_w, bar_h), C_GREEN, 2)

    # FPS counter (top-right)
    fps_text = f"FPS: {fps:.1f}"
    (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(frame, fps_text, (img_w - fw - 10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_DIM, 1, cv2.LINE_AA)

    cv2.putText(frame, "smoking-detection v5 (Smooth)", (img_w - 230, img_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1, cv2.LINE_AA)

    return frame

def run_on_image(path):
    frame = cv2.imread(path)
    if frame is None:
        print("[ERROR] Cannot load image.")
        return
    img_h, img_w = frame.shape[:2]
    
    b64 = frame_to_base64(frame)
    result = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY).infer(b64, model_id=MODEL_ID)
    preds = result.get('predictions', [])
    preds = scale_predictions(preds, img_w, img_h)

    faces, cigs = parse_predictions(preds, img_w, img_h)
    smoking = check_smoking(faces, cigs, img_w, img_h)
    
    frame = draw_ui(frame, faces, cigs, smoking, fps=0)
    cv2.imshow("Smoking Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_on_video(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        return

    cap.set(cv2.CAP_PROP_FPS, 30)

    print("[INFO] Running... Press Q to quit.")

    for _ in range(NUM_WORKERS):
        t = threading.Thread(target=inference_worker, daemon=True)
        t.start()

    fps         = 0.0
    fps_timer   = time.time()
    frame_count = 0
    TARGET_MS   = 1000 // 30

    while True:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        try:
            STATE.frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass  # Workers are busy; drop this frame and use the next one

        with STATE.lock:
            faces    = STATE.faces[:]
            cigs     = STATE.cigs[:]
            
            # The alert is active if we've seen smoking within the cooldown window
            time_since_smoking = time.time() - STATE.last_smoking_time
            is_smoking_alert   = (time_since_smoking <= COOLDOWN_SECONDS)

        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps         = frame_count / elapsed
            fps_timer   = time.time()
            frame_count = 0

        frame = draw_ui(frame, faces, cigs, is_smoking_alert, fps)
        cv2.imshow("Smoking Detection", frame)

        spent_ms = int((time.time() - loop_start) * 1000)
        wait_ms  = max(1, TARGET_MS - spent_ms)
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            break

    STATE.stop.set()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0")
    args   = parser.parse_args()
    source = args.source

    if source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        run_on_image(source)
    else:
        try:
            source = int(source)
        except ValueError:
            pass
        run_on_video(source)