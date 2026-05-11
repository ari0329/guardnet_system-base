"""
GuardNet – Real-Time Inference  (Threaded, Queue-Based, Multi-Camera Edition)
==============================================================================
Fixes:
  Problem 2 – Webcam buffering / lag / low FPS

Optimisations applied
  ✓ Threaded capture: dedicated thread per camera reads frames non-blocking
  ✓ Queue-based buffer: fixed-size queue drops stale frames automatically
  ✓ Frame skipping: model only runs on every N-th frame (configurable)
  ✓ Async inference thread: model.predict() never blocks the display loop
  ✓ LSTM sequence: collections.deque keeps a rolling window of SEQ_LEN frames
  ✓ Resize-before-inference: 112×112 for model, original res for display
  ✓ GPU memory growth: avoids OOM on low-VRAM cards
  ✓ waitKey(1) instead of waitKey(0): maximum display throughput
  ✓ Multi-camera: pass multiple indices / RTSP URLs via --cameras
  ✓ FPS counter overlay on each stream
  ✓ Graceful shutdown on 'q' or Ctrl+C

Usage:
    python demo.py --model models/guardnet_v3.h5
    python demo.py --model models/guardnet_v3.h5 --source video.mp4
    python demo.py --model models/guardnet_v3.h5 --cameras 0 1        # dual webcam
    python demo.py --model models/guardnet_v3.h5 --cameras rtsp://... # CCTV
    python demo.py --model models/guardnet_v3.h5 --skip 3 --threshold 0.6
"""

import os
import sys
import time
import queue
import threading
import argparse
import collections
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Config ─────────────────────────────────────────────────────────────────
try:
    from config.config import MODEL_PATH, SEQ_LEN
except ImportError:
    MODEL_PATH = "models/guardnet_v3.h5"
    SEQ_LEN    = 16

IMG_H      = 112
IMG_W      = 112
QUEUE_SIZE = 4          # drop frames older than this many in the buffer
ALERT_SECS = 3.0        # how long to show the VIOLENCE alert banner


# ══════════════════════════════════════════════════════════════════════════════
#  Thread 1 – Frame Capture
# ══════════════════════════════════════════════════════════════════════════════

class CameraStream:
    """
    Non-blocking camera reader.

    A background thread continuously reads frames from OpenCV and puts
    them in a fixed-size queue.  If the queue is full (inference is too slow),
    old frames are discarded automatically — the display always stays live.
    """

    def __init__(self, source, cam_id: int = 0, queue_size: int = QUEUE_SIZE):
        self.source   = source
        self.cam_id   = cam_id
        self.q: queue.Queue = queue.Queue(maxsize=queue_size)
        self.stopped  = threading.Event()
        self.cap      = None

    def start(self) -> "CameraStream":
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise IOError(
                f"[Camera {self.cam_id}] Cannot open source: {self.source}"
            )
        # Reduce internal OpenCV buffer to 1 frame → lower latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()
        return self

    def _reader(self):
        while not self.stopped.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self.stopped.set()
                break
            # If queue is full, remove the oldest frame before adding new one
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            return True, self.q.get(timeout=0.5)
        except queue.Empty:
            return False, None

    def stop(self):
        self.stopped.set()
        if self.cap:
            self.cap.release()


# ══════════════════════════════════════════════════════════════════════════════
#  Thread 2 – Async Inference
# ══════════════════════════════════════════════════════════════════════════════

class InferenceEngine:
    """
    Runs model.predict() in a background thread so the capture + display
    loop is never blocked by heavy CNN/LSTM computation.

    Flow:
        Display thread     →  puts a [SEQ_LEN, H, W, 3] sequence into in_q
        Inference thread   →  calls model.predict() and puts result into out_q
        Display thread     →  reads latest result from out_q (non-blocking)
    """

    def __init__(self, model, seq_len: int = SEQ_LEN):
        self.model    = model
        self.seq_len  = seq_len
        self.in_q:  queue.Queue = queue.Queue(maxsize=1)
        self.out_q: queue.Queue = queue.Queue(maxsize=1)
        self.stopped  = threading.Event()
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def _worker(self):
        while not self.stopped.is_set():
            try:
                sequence = self.in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            # Predict (blocking call, but in its own thread)
            inp   = np.expand_dims(sequence, axis=0)          # [1, T, H, W, 3]
            prob  = float(self.model.predict(inp, verbose=0)[0][0])
            # Overwrite stale result if not consumed yet
            if self.out_q.full():
                try:
                    self.out_q.get_nowait()
                except queue.Empty:
                    pass
            self.out_q.put(prob)

    def submit(self, sequence: np.ndarray):
        """Non-blocking submit. Drops silently if inference is still running."""
        if self.in_q.full():
            try:
                self.in_q.get_nowait()
            except queue.Empty:
                pass
        self.in_q.put_nowait(sequence)

    def get_result(self) -> Optional[float]:
        """Return latest prediction probability, or None if not ready yet."""
        try:
            return self.out_q.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.stopped.set()


# ══════════════════════════════════════════════════════════════════════════════
#  Per-Camera Processing State
# ══════════════════════════════════════════════════════════════════════════════

class CameraState:
    """Tracks frame buffer + prediction history for one camera."""

    def __init__(
        self,
        cam_id:    int,
        seq_len:   int   = SEQ_LEN,
        skip:      int   = 2,
        threshold: float = 0.55,
    ):
        self.cam_id    = cam_id
        self.seq_len   = seq_len
        self.skip      = skip          # process every `skip`-th frame
        self.threshold = threshold

        self.frame_buffer = collections.deque(maxlen=seq_len)
        self.frame_count  = 0
        self.last_prob    = 0.0
        self.alert_until  = 0.0        # epoch time when alert expires
        self.fps_times: collections.deque = collections.deque(maxlen=30)

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize to model input and normalise to [0, 1]."""
        small = cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
        small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        return small.astype(np.float32) / 255.0

    def update_fps(self):
        now = time.time()
        self.fps_times.append(now)

    def get_fps(self) -> float:
        if len(self.fps_times) < 2:
            return 0.0
        return len(self.fps_times) / (self.fps_times[-1] - self.fps_times[0] + 1e-6)


# ══════════════════════════════════════════════════════════════════════════════
#  Overlay drawing
# ══════════════════════════════════════════════════════════════════════════════

def draw_overlay(
    frame:     np.ndarray,
    prob:      float,
    threshold: float,
    fps:       float,
    cam_id:    int,
    alert:     bool,
) -> np.ndarray:
    """Draw probability bar, FPS counter, and alert banner onto frame."""
    h, w = frame.shape[:2]
    out  = frame.copy()

    # ── Probability bar ────────────────────────────────────────────────────
    bar_w = int(w * 0.4)
    bar_h = 18
    bar_x = 10
    bar_y = 40

    # Background
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (50, 50, 50), -1)
    # Fill
    fill  = int(bar_w * prob)
    color = (0, 200, 0) if prob < threshold else (0, 0, 220)
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h),
                  color, -1)
    # Border
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (200, 200, 200), 1)
    # Label
    label = f"Violence: {prob * 100:.1f}%"
    cv2.putText(out, label, (bar_x + bar_w + 8, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Camera ID + FPS ────────────────────────────────────────────────────
    status_txt = f"Cam {cam_id}  |  {fps:.1f} FPS"
    cv2.putText(out, status_txt, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Alert banner ───────────────────────────────────────────────────────
    if alert:
        banner_h = 55
        overlay  = out.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)
        cv2.putText(out, "⚠  VIOLENCE DETECTED", (12, h - 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Red border when violence ───────────────────────────────────────────
    if alert:
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 220), 4)

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Main loop
# ══════════════════════════════════════════════════════════════════════════════

def run(
    sources:   List,
    model_path: str,
    skip:       int   = 2,
    threshold:  float = 0.55,
    seq_len:    int   = SEQ_LEN,
    display_w:  int   = 640,
    display_h:  int   = 480,
):
    # ── Load model ──────────────────────────────────────────────────────────
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    print(f"[INFO] Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("[INFO] Model loaded.")

    # ── Start capture threads ───────────────────────────────────────────────
    streams = []
    for idx, src in enumerate(sources):
        try:
            src_int = int(src)      # webcam index
        except (ValueError, TypeError):
            src_int = src           # path / RTSP URL string
        stream = CameraStream(src_int, cam_id=idx).start()
        streams.append(stream)
        print(f"[INFO] Camera {idx} opened: {src}")

    # ── One shared inference engine (single GPU/CPU) ────────────────────────
    engine = InferenceEngine(model, seq_len=seq_len)

    # ── Per-camera state ───────────────────────────────────────────────────
    states = [
        CameraState(cam_id=i, seq_len=seq_len, skip=skip, threshold=threshold)
        for i in range(len(streams))
    ]

    print("[INFO] Press 'q' to quit, 's' to save a snapshot.")
    print(f"[INFO] Frame skip={skip} | Threshold={threshold} | Seq={seq_len}")

    # ── Grid layout for multi-camera display ───────────────────────────────
    n_cams = len(streams)
    cols   = min(n_cams, 2)
    rows   = (n_cams + cols - 1) // cols

    # Cache for last valid display frames (one per camera)
    display_frames = [
        np.zeros((display_h, display_w, 3), dtype=np.uint8)
        for _ in range(n_cams)
    ]

    snapshot_dir = "snapshots"

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                os.makedirs(snapshot_dir, exist_ok=True)
                ts = int(time.time())
                for ci, df in enumerate(display_frames):
                    path = os.path.join(snapshot_dir, f"cam{ci}_{ts}.jpg")
                    cv2.imwrite(path, df)
                    print(f"[INFO] Saved snapshot: {path}")

            # ── Read + process each camera ─────────────────────────────────
            for ci, (stream, state) in enumerate(zip(streams, states)):
                ret, raw_frame = stream.read()
                if not ret or raw_frame is None:
                    continue

                state.update_fps()
                state.frame_count += 1

                # Push pre-processed small frame into rolling buffer
                small = state.preprocess_frame(raw_frame)
                state.frame_buffer.append(small)

                # ── Submit to inference every `skip` frames ────────────────
                if (state.frame_count % state.skip == 0 and
                        len(state.frame_buffer) == seq_len):
                    sequence = np.stack(list(state.frame_buffer), axis=0)
                    # Only camera 0 drives the engine (simplest multi-cam
                    # approach; for true parallel use one engine per camera)
                    if ci == 0:
                        engine.submit(sequence)

                # ── Collect latest prediction ──────────────────────────────
                if ci == 0:
                    prob = engine.get_result()
                    if prob is not None:
                        state.last_prob = prob
                        if prob >= threshold:
                            state.alert_until = time.time() + ALERT_SECS

                # ── Draw overlay ───────────────────────────────────────────
                alert = time.time() < state.alert_until
                display = cv2.resize(raw_frame, (display_w, display_h))
                display = draw_overlay(
                    display,
                    prob=state.last_prob,
                    threshold=threshold,
                    fps=state.get_fps(),
                    cam_id=ci,
                    alert=alert,
                )
                display_frames[ci] = display

            # ── Build grid canvas ──────────────────────────────────────────
            if n_cams == 1:
                canvas = display_frames[0]
            else:
                row_imgs = []
                for r in range(rows):
                    row_tiles = []
                    for c in range(cols):
                        idx = r * cols + c
                        tile = (
                            display_frames[idx]
                            if idx < n_cams
                            else np.zeros((display_h, display_w, 3), dtype=np.uint8)
                        )
                        row_tiles.append(tile)
                    row_imgs.append(np.hstack(row_tiles))
                canvas = np.vstack(row_imgs)

            cv2.imshow("GuardNet – Real-Time Violence Detection", canvas)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        for s in streams:
            s.stop()
        engine.stop()
        cv2.destroyAllWindows()
        print("[INFO] Shutdown complete.")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="GuardNet real-time violence detection")
    p.add_argument("--model",     default=MODEL_PATH,
                   help="Path to trained .h5 model")
    p.add_argument("--source",    default=None,
                   help="Single video source (webcam index or file path)")
    p.add_argument("--cameras",   nargs="+", default=None,
                   help="One or more camera sources: --cameras 0 1  or  --cameras rtsp://...")
    p.add_argument("--skip",      type=int,   default=2,
                   help="Run inference every N frames (higher = faster FPS)")
    p.add_argument("--threshold", type=float, default=0.55,
                   help="Violence probability threshold (0–1)")
    p.add_argument("--seq_len",   type=int,   default=SEQ_LEN,
                   help="Number of frames in the LSTM sequence")
    p.add_argument("--width",     type=int,   default=640,
                   help="Display width per camera tile")
    p.add_argument("--height",    type=int,   default=480,
                   help="Display height per camera tile")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.cameras:
        sources = args.cameras
    elif args.source is not None:
        sources = [args.source]
    else:
        sources = [0]          # default: webcam 0

    run(
        sources=sources,
        model_path=args.model,
        skip=args.skip,
        threshold=args.threshold,
        seq_len=args.seq_len,
        display_w=args.width,
        display_h=args.height,
    )