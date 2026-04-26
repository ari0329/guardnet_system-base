"""
GuardNet – Detection Engine
Orchestrates frame capture → preprocessing → inference → annotation → alert.
"""

import os
import sys
import time
import cv2
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    FRAME_WIDTH, FRAME_HEIGHT, SEQUENCE_LENGTH,
    VIOLENCE_THRESHOLD, WINDOW_NAME,
    DISPLAY_FPS, DISPLAY_HEATMAP, RECORD_OUTPUT, OUTPUT_VIDEO_PATH,
    COLOR_NORMAL, COLOR_VIOLENCE, COLOR_TEXT_BG,
    ENABLE_HEATMAP, HEATMAP_ALPHA, MODEL_PATH
)
from utils.preprocessing import SequenceBuilder, MotionHeatmap
from utils.alerts import get_alert_manager
from utils.person_detector import PersonDetector
from models.guardnet_model import GuardNetInference


# ─── Annotation Helpers ───────────────────────────────────────────────────────

def _draw_label(
    frame: np.ndarray,
    text: str,
    confidence: float,
    is_violent: bool,
) -> np.ndarray:
    h, w = frame.shape[:2]
    colour = COLOR_VIOLENCE if is_violent else COLOR_NORMAL
    status  = "⚠  VIOLENCE DETECTED" if is_violent else "✔  NORMAL ACTIVITY"
    conf_pct = f"{confidence * 100:.1f}%"

    # Bottom banner
    banner_h = 60
    cv2.rectangle(frame, (0, h - banner_h), (w, h), COLOR_TEXT_BG, -1)
    cv2.putText(frame, status, (12, h - 34),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, colour, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Confidence: {conf_pct}", (12, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Confidence bar (top-right corner)
    bar_w = 200
    bar_h = 14
    margin = 10
    bar_x = w - bar_w - margin
    bar_y = margin
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    filled = int(bar_w * confidence)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + filled, bar_y + bar_h), colour, -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)
    cv2.putText(frame, "Violence Risk", (bar_x, bar_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Red frame border when violent
    if is_violent:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_VIOLENCE, 4)

    return frame


def _draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


def _draw_timestamp(frame: np.ndarray) -> np.ndarray:
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    h, w = frame.shape[:2]
    cv2.putText(frame, ts, (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    return frame


# ─── Main Detection Engine ───────────────────────────────────────────────────

class DetectionEngine:

    def __init__(self, model_path: str = MODEL_PATH):
        self.inferencer = GuardNetInference(model_path)
        self.seq_builder = SequenceBuilder()
        self.alert_mgr   = get_alert_manager()
        self.person_det  = PersonDetector()

        self._last_prob  = 0.0
        self._is_violent = False
        self._fps        = 0.0
        self._heatmap    = None      # initialised on first frame

    # ── Video Loop ────────────────────────────────────────────────────────────

    def run(self, source: int | str = 0, source_name: str = "camera"):
        """
        source : 0 for default webcam, or a path string for a video file.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        fps_in  = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if RECORD_OUTPUT:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(
                OUTPUT_VIDEO_PATH, fourcc, fps_in, (frame_w, frame_h)
            )

        self._heatmap = MotionHeatmap((frame_h, frame_w))
        prev_gray     = None
        t_prev        = time.time()

        print(f"[INFO] Detection started – source: {source}")
        print(f"[INFO] Press  Q  or  ESC  to quit.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream.")
                break

            # ── FPS measurement ──
            t_now      = time.time()
            self._fps  = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev     = t_now

            # ── Heatmap update ──
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None and ENABLE_HEATMAP:
                self._heatmap.update(prev_gray, curr_gray)
            prev_gray = curr_gray

            # ── Sequence inference ──
            seq = self.seq_builder.update(frame)
            if seq is not None:
                prob = self.inferencer.predict(seq)
                self._last_prob  = prob
                self._is_violent = prob >= VIOLENCE_THRESHOLD
                if self._is_violent:
                    self.alert_mgr.trigger(prob, source=source_name)

            # ── Build display frame ──
            display = frame.copy()

            if DISPLAY_HEATMAP and ENABLE_HEATMAP and prev_gray is not None:
                display = self._heatmap.get_overlay(display, HEATMAP_ALPHA)

            # Person detection overlay
            boxes = self.person_det.detect(frame)
            if boxes:
                display = self.person_det.draw(display, boxes, self._is_violent)

            # Labels & HUD
            display = _draw_label(display, "", self._last_prob, self._is_violent)
            if DISPLAY_FPS:
                display = _draw_fps(display, self._fps)
            display = _draw_timestamp(display)

            # ── Show / record ──
            cv2.imshow(WINDOW_NAME, display)
            if writer:
                writer.write(display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):   # Q or ESC
                break
            elif key == ord("h"):
                # Toggle heatmap with 'H'
                import config.config as cfg
                cfg.DISPLAY_HEATMAP = not cfg.DISPLAY_HEATMAP

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Detection stopped.")


# ─── Convenience Wrappers ────────────────────────────────────────────────────

def run_webcam(cam_index: int = 0, model_path: str = MODEL_PATH):
    engine = DetectionEngine(model_path)
    engine.run(source=cam_index, source_name=f"webcam:{cam_index}")


def run_video(video_path: str, model_path: str = MODEL_PATH):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    engine = DetectionEngine(model_path)
    engine.run(source=video_path, source_name=os.path.basename(video_path))
