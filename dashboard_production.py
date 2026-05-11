"""
GuardNet – Production CCTV Dashboard  v4  (Low-Latency Edition)
===============================================================
Run:  streamlit run dashboard_production.py

Key fixes applied in this version
──────────────────────────────────
Problem 1 — Frame buffer buildup / stale-frame lag
  ✓ Three decoupled daemon threads per camera:
        Thread-1  CaptureThread   — ONLY reads frames from OpenCV
        Thread-2  InferenceThread — ONLY runs the TF model
        Thread-3  FlowThread      — ONLY computes optical-flow heatmap
  ✓ LatestFrame slot (from preprocessing.py) replaces frame_q:
        display thread reads the NEWEST frame — zero queue buildup possible
  ✓ CAP_PROP_BUFFERSIZE=1  eliminates OS-level frame accumulation
  ✓ CAP_PROP_FOURCC=MJPG   faster decode for USB webcams
  ✓ Inference queue maxsize=1 with put_nowait + drop:
        if TF is slow the capture thread silently drops the inference frame
        and continues — live video is never blocked
  ✓ INFER_EVERY=4 frame skip:  model runs at ~source_fps/4, display at full fps
  ✓ Flow computed at 320×240 regardless of camera resolution
  ✓ Adaptive sleep in Streamlit loop targets 30 fps ceiling, never blocks

Problem 2 — Heatmap inaccurate / invisible
  ✓ MotionHeatmap from preprocessing.py v4:
        EMA smoothing, percentile normalisation, dilation, mask-blend
        → stable red/orange zones over moving humans, no flicker

Architecture per camera
  [CaptureThread]  cap.read() → LatestFrame.write()
                              → infer_q.put_nowait() (every 4th frame)
  [InferenceThread] infer_q.get() → TF predict → result dict
  [FlowThread]      LatestFrame.read() → MotionHeatmap.update()
                                       → result dict
  [Streamlit loop]  LatestFrame.read_copy() + result dict → annotate → display
"""

import os
import sys
import time
import threading
import queue
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config.config import (
    MODEL_PATH, LOG_FILE, VIOLENCE_THRESHOLD,
    SEQUENCE_LENGTH, FRAME_SKIP, HEATMAP_ALPHA,
    CLIPS_DIR, PRE_BUFFER_SECONDS, POST_BUFFER_SECONDS,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GuardNet CCTV",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;500;700;900&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [data-testid="stAppViewContainer"] {
  background: #040810 !important;
  font-family: 'Exo 2', sans-serif;
}
[data-testid="stSidebar"] {
  background: #060d1a !important;
  border-right: 1px solid #0d2137;
}
[data-testid="stSidebar"] * { color: #c9d8e8 !important; }

h1, h2, h3 { font-family: 'Exo 2', sans-serif; }

.feed-cell {
  background: #060d1a;
  border: 1px solid #0d2a40;
  border-radius: 6px;
  overflow: hidden;
  position: relative;
}
.feed-label-normal {
  background: rgba(0,200,80,0.18);
  border: 1px solid #00c850;
  border-radius: 4px;
  color: #00e85a;
  font-family: 'Share Tech Mono', monospace;
  font-size: 13px;
  padding: 5px 12px;
  text-align: center;
  font-weight: bold;
  letter-spacing: 1px;
}
.feed-label-violence {
  background: rgba(220,30,30,0.25);
  border: 1px solid #f04040;
  border-radius: 4px;
  color: #ff5555;
  font-family: 'Share Tech Mono', monospace;
  font-size: 13px;
  padding: 5px 12px;
  text-align: center;
  font-weight: bold;
  letter-spacing: 1px;
  animation: blink 0.8s step-start infinite;
}
@keyframes blink { 50% { opacity: 0.4; } }

.stat-card {
  background: #070f1e;
  border: 1px solid #0d2a40;
  border-radius: 8px;
  padding: 16px 12px;
  text-align: center;
}
.stat-val {
  font-family: 'Share Tech Mono', monospace;
  font-size: 28px;
  color: #00d4ff;
  line-height: 1;
}
.stat-lbl {
  font-size: 10px;
  color: #5a7a94;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  margin-top: 6px;
}
.alert-card {
  background: rgba(220,30,30,0.15);
  border: 1px solid #f04040;
  border-radius: 8px;
  padding: 10px 14px;
  color: #ff6666;
  font-family: 'Share Tech Mono', monospace;
  font-size: 12px;
  margin-bottom: 6px;
}
.section-head {
  font-family: 'Share Tech Mono', monospace;
  font-size: 11px;
  letter-spacing: 2px;
  color: #2e5a74;
  text-transform: uppercase;
  border-bottom: 1px solid #0d2137;
  padding-bottom: 6px;
  margin-bottom: 10px;
}
.no-events {
  color: #2e5a74;
  font-family: 'Share Tech Mono', monospace;
  font-size: 12px;
  text-align: center;
  padding: 20px;
}
div[data-testid="stImage"] img {
  border-radius: 4px;
  width: 100% !important;
}
</style>
""", unsafe_allow_html=True)


# ── Annotation helper ─────────────────────────────────────────────────────────

def _annotate_frame(frame: np.ndarray, prob: float, is_violent: bool,
                    boxes=None, heatmap_overlay=None,
                    cam_id: str = "CAM-1", fps: float = 0.0) -> np.ndarray:
    """Draw all overlays onto a BGR frame. Returns annotated BGR frame."""
    # Use heatmap overlay as base if available, otherwise copy raw frame
    display = heatmap_overlay if heatmap_overlay is not None else frame.copy()
    h, w    = display.shape[:2]

    # YOLOv8 person bounding boxes
    if boxes:
        colour = (0, 30, 220) if is_violent else (255, 165, 0)
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(display, (x1, y1), (x2, y2), colour, 2)
            cv2.rectangle(display, (x1, y1 - 18), (x1 + 52, y1), colour, -1)
            cv2.putText(display, "Person", (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA)

    # Top bar
    cv2.rectangle(display, (0, 0), (w, 32), (5, 10, 20), -1)
    cv2.putText(display, cam_id, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 200, 80) if not is_violent else (60, 60, 220),
                2, cv2.LINE_AA)
    ts_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(display, ts_str, (w - 210, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 140, 180), 1, cv2.LINE_AA)

    # Confidence bar
    bw, bh = 160, 10
    bx, by = w - bw - 8, 38
    cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (20, 30, 40), -1)
    bar_col = (0, 30, 220) if is_violent else (0, 180, 80)
    cv2.rectangle(display, (bx, by), (bx + int(bw * prob), by + bh), bar_col, -1)
    cv2.putText(display, f"{prob * 100:.1f}%", (bx, by + bh + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, bar_col, 1, cv2.LINE_AA)

    # FPS counter
    cv2.putText(display, f"FPS:{fps:.0f}", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 90, 120), 1, cv2.LINE_AA)

    # Bottom status banner
    banner_colour = (8, 10, 40) if is_violent else (5, 20, 10)
    cv2.rectangle(display, (0, h - 34), (w, h), banner_colour, -1)
    label  = "!  VIOLENCE DETECTED" if is_violent else "OK NORMAL ACTIVITY"
    colour = (60, 80, 240) if is_violent else (0, 200, 80)
    cv2.putText(display, label, (10, h - 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, colour, 2, cv2.LINE_AA)

    # Red border
    if is_violent:
        cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 40, 220), 4)

    return display


# ══════════════════════════════════════════════════════════════════════════════
# Per-camera state container
# ══════════════════════════════════════════════════════════════════════════════

class CameraState:
    """
    Holds all shared state for one camera.

    Thread safety
    -------------
    latest_frame  → LatestFrame uses its own internal lock
    result        → protected by result_lock
    counters (total, events, fps) → written only by their respective threads;
                                    read by Streamlit on rerun (acceptable
                                    eventual-consistency for display counters)
    stop_event    → threading.Event (thread-safe by design)
    """

    def __init__(self, cam_id: str, source, threshold: float):
        self.cam_id    = cam_id
        self.source    = source
        self.threshold = threshold

        # ── Optimised frame transport (replaces frame_q) ─────────────────
        # Import here so the class is importable even before the module is
        # added to sys.path in the thread.
        from utils.preprocessing import LatestFrame
        self.latest_frame = LatestFrame()

        # ── Result bus ───────────────────────────────────────────────────
        self.result: dict = {
            "prob":       0.0,
            "is_violent": False,
            "heatmap":    None,   # BGR overlay (full-res) or None
            "boxes":      [],
        }
        self.result_lock = threading.Lock()

        # ── Control ──────────────────────────────────────────────────────
        self.stop_event = threading.Event()

        # ── Display counters (eventual consistency is fine) ───────────────
        self.capture_fps: float = 0.0   # measured by capture thread
        self.total:       int   = 0     # frames captured
        self.events:      int   = 0     # violence events logged
        self.active:      bool  = True


# ══════════════════════════════════════════════════════════════════════════════
# Thread-1 — Capture thread
# ══════════════════════════════════════════════════════════════════════════════

def _capture_thread(state: CameraState, infer_q: queue.Queue) -> None:
    """
    Reads frames from camera as fast as possible.
    Writes every frame to LatestFrame (zero-latency display).
    Sends every INFER_EVERY-th frame to the inference queue (non-blocking).

    Why CAP_PROP_BUFFERSIZE = 1
    ---------------------------
    OpenCV's VideoCapture maintains an internal OS-level ring buffer
    (default 4-10 frames for RTSP/USB).  When inference takes 300 ms the
    capture loop stalls; hardware keeps filling the buffer; when cap.read()
    resumes it returns stale frames.  Setting buffersize=1 minimises this
    hardware-level accumulation so cap.read() returns a frame that is at
    most one decode cycle old.

    Why INFER_EVERY = 4
    --------------------
    The TF model processes SEQ_LEN=16 frames and needs those 16 frames
    to accumulate before a prediction is possible anyway.  Running inference
    on every captured frame would not increase prediction frequency — it
    would only keep the inference thread permanently busy.  Sampling every
    4th frame also provides natural temporal diversity for the sequence.
    """
    INFER_EVERY = 4

    cap = cv2.VideoCapture(state.source)
    if not cap.isOpened():
        state.active = False
        return

    # ── Low-latency capture settings ─────────────────────────────────────
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # MJPEG decode is typically 2-3× faster than H264 for USB webcams
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    # For RTSP streams, TCP transport avoids UDP packet loss causing
    # frame corruption which forces re-reads:
    # cap.set(cv2.CAP_PROP_RTSP_TRANSPORT, "tcp")  # uncomment for RTSP

    frame_idx = 0
    t_prev    = time.perf_counter()

    while not state.stop_event.is_set():
        ret, frame = cap.read()

        if not ret:
            # File loop or dead stream
            if isinstance(state.source, str) and state.source != "0":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            time.sleep(0.01)
            continue

        # ── Measure real capture FPS ──────────────────────────────────────
        t_now          = time.perf_counter()
        state.capture_fps = 0.9 * state.capture_fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev         = t_now
        state.total   += 1
        frame_idx     += 1

        # ── Write to latest-frame slot (always, unconditionally) ──────────
        # The display thread reads this; it always gets the newest frame.
        state.latest_frame.write(frame)

        # ── Non-blocking send to inference queue ──────────────────────────
        # put_nowait raises queue.Full if the inference thread is still
        # processing the previous sequence — we simply skip this frame.
        # The camera never stalls because of inference latency.
        if frame_idx % INFER_EVERY == 0:
            try:
                infer_q.put_nowait(frame.copy())
            except queue.Full:
                pass   # inference is behind; drop this frame, keep capturing

    cap.release()
    state.active = False


# ══════════════════════════════════════════════════════════════════════════════
# Thread-2 — Inference thread
# ══════════════════════════════════════════════════════════════════════════════

def _inference_thread(
    state:      CameraState,
    infer_q:    queue.Queue,
    model_path: str,
    alert_email: str,
    alert_mgr,
    logger,
) -> None:
    """
    Runs the TF model.  Completely decoupled from frame capture.

    Because it reads from infer_q (maxsize=1) it is naturally throttled:
    if inference takes 400 ms and frames arrive every 33 ms, the queue
    will be full 92% of the time and those frames are dropped in the
    capture thread — but the DISPLAY is always fresh because the display
    reads from LatestFrame, not from here.
    """
    from utils.preprocessing   import SequenceBuilder
    from utils.person_detector import PersonDetector
    from utils.clip_extractor  import ClipExtractor
    from models.guardnet_model import GuardNetInference

    inferencer  = GuardNetInference(model_path)
    seq_builder = SequenceBuilder()
    person_det  = PersonDetector()
    clipper     = ClipExtractor(fps=15.0, camera_id=state.cam_id)

    while not state.stop_event.is_set():
        try:
            frame = infer_q.get(timeout=0.5)
        except queue.Empty:
            continue

        # ── Build sequence and run model ──────────────────────────────────
        seq  = seq_builder.update(frame)
        prob = float(inferencer.predict(seq)) if seq is not None else (
            state.result["prob"]   # keep last known prob until seq fills
        )

        is_v = prob >= state.threshold

        # ── Person detection (runs at inference rate, not capture rate) ────
        boxes = person_det.detect(frame)

        # ── Clip extraction ───────────────────────────────────────────────
        if is_v and not clipper.is_recording():
            clipper.start_recording(frame)
        clip_path = clipper.push(frame) if clipper.is_recording() else None

        # ── Alert ─────────────────────────────────────────────────────────
        if is_v:
            alert_mgr.trigger(
                prob,
                camera_id   = state.cam_id,
                alert_email = alert_email,
                clip_path   = clip_path or "",
            )
            logger.log(state.cam_id, prob, clip_path or "")
            state.events += 1

        # ── Write results to shared result bus ────────────────────────────
        with state.result_lock:
            state.result["prob"]       = prob
            state.result["is_violent"] = is_v
            state.result["boxes"]      = boxes


# ══════════════════════════════════════════════════════════════════════════════
# Thread-3 — Optical-flow / heatmap thread
# ══════════════════════════════════════════════════════════════════════════════

def _flow_thread(state: CameraState) -> None:
    """
    Computes optical-flow heatmap at ~30 fps on DOWNSCALED frames.

    This thread reads from LatestFrame (same as the display thread) so it
    always works on current frames regardless of inference speed.

    Why it is a separate thread
    ---------------------------
    In the original code, Farnebäck was called inside the main camera loop
    on the full-resolution frame — up to 200 ms per call at 1080p.  Moving
    it to its own thread lets the capture loop run at full camera FPS while
    the heatmap updates independently at whatever rate the CPU allows.
    """
    from utils.preprocessing import MotionHeatmap

    FLOW_TARGET_FPS = 30.0
    FLOW_INTERVAL   = 1.0 / FLOW_TARGET_FPS

    heatmap_acc: MotionHeatmap | None = None
    prev_gray:   np.ndarray   | None = None

    while not state.stop_event.is_set():
        t_start = time.perf_counter()

        frame = state.latest_frame.read()
        if frame is None:
            time.sleep(FLOW_INTERVAL)
            continue

        h, w = frame.shape[:2]

        # Initialise heatmap with correct display shape on first valid frame
        if heatmap_acc is None:
            heatmap_acc = MotionHeatmap(
                display_shape=(h, w),
                ema_alpha=0.25,
                flow_size=(320, 240),  # compute flow at 320×240 — fast
                noise_floor=0.08,
                dilation_iter=2,
                colormap=cv2.COLORMAP_INFERNO,
            )

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            heatmap_acc.update(prev_gray, curr_gray)
            overlay = heatmap_acc.get_overlay(frame, alpha=HEATMAP_ALPHA)
            with state.result_lock:
                state.result["heatmap"] = overlay

        prev_gray = curr_gray

        # Adaptive sleep to maintain target FPS without busy-waiting
        elapsed = time.perf_counter() - t_start
        sleep_t = max(0.0, FLOW_INTERVAL - elapsed)
        if sleep_t > 0:
            time.sleep(sleep_t)


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point — launch all threads for one camera
# ══════════════════════════════════════════════════════════════════════════════

def _start_camera(
    state:       CameraState,
    model_path:  str,
    alert_email: str,
    alert_mgr,
    logger,
) -> None:
    """
    Launch the three daemon threads for one camera.

    Thread topology
    ───────────────
    CaptureThread  ──write──►  LatestFrame  ──read──►  FlowThread
                   ──put──►   infer_q      ──get──►   InferenceThread
    FlowThread     ──write──►  result["heatmap"]
    InferenceThread──write──►  result["prob", "is_violent", "boxes"]
    Streamlit loop ──read──►   LatestFrame  +  result dict
    """
    # Bounded queue — maxsize=1 ensures inference always sees a near-current
    # frame and cannot accumulate a backlog.
    infer_q = queue.Queue(maxsize=1)

    for target, kwargs in [
        (_capture_thread,   {"state": state, "infer_q": infer_q}),
        (_inference_thread, {"state": state, "infer_q": infer_q,
                              "model_path": model_path,
                              "alert_email": alert_email,
                              "alert_mgr": alert_mgr,
                              "logger": logger}),
        (_flow_thread,      {"state": state}),
    ]:
        t = threading.Thread(target=target, kwargs=kwargs, daemon=True)
        t.start()


# ══════════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ══════════════════════════════════════════════════════════════════════════════

def _ss(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

_ss("cameras",     {})       # cam_id → CameraState
_ss("running",     False)
_ss("alert_email", "")
_ss("prob_hist",   {})       # cam_id → list[float]
_ss("alert_mgr",   None)
_ss("logger",      None)


# ── Lazy singletons ───────────────────────────────────────────────────────────

def get_alert_mgr():
    if st.session_state.alert_mgr is None:
        from utils.alerts import AlertManager
        st.session_state.alert_mgr = AlertManager()
    return st.session_state.alert_mgr

def get_logger():
    if st.session_state.logger is None:
        from utils.alerts import EventLogger
        st.session_state.logger = EventLogger()
    return st.session_state.logger


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 8px'>
      <span style='font-size:36px'>🛡️</span><br>
      <span style='font-family:Share Tech Mono,monospace;color:#00d4ff;
                   font-size:18px;letter-spacing:3px'>GUARDNET</span><br>
      <span style='font-size:10px;color:#2e5a74;letter-spacing:2px'>
        CCTV INTELLIGENCE PLATFORM</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Alert email
    st.markdown('<div class="section-head">Alert Email</div>',
                unsafe_allow_html=True)
    alert_email = st.text_input(
        "Send alerts to", value=st.session_state.alert_email,
        placeholder="security@yourcompany.com",
        label_visibility="collapsed",
    )
    st.session_state.alert_email = alert_email
    if alert_email:
        st.success(f"Alerts → {alert_email}")
    else:
        st.info("Enter email to receive violence alerts")

    st.divider()

    # Camera sources
    st.markdown('<div class="section-head">Camera Sources</div>',
                unsafe_allow_html=True)
    num_cams = st.number_input("Number of cameras", 1, 6, 1)

    sources = []
    for i in range(num_cams):
        src_type = st.radio(
            f"Camera {i+1} type",
            ["Webcam", "Video File", "IP Stream"],
            key=f"src_type_{i}", horizontal=True,
        )
        if src_type == "Webcam":
            idx = st.number_input(f"Cam {i+1} index", 0, 8, i,
                                  key=f"cam_idx_{i}")
            sources.append(("webcam", int(idx), f"CAM-{i+1}"))
        elif src_type == "Video File":
            up = st.file_uploader(
                f"Upload for Cam {i+1}",
                type=["mp4", "avi", "mov", "mkv"],
                key=f"upload_{i}",
            )
            if up:
                p = Path(tempfile.gettempdir()) / up.name
                p.write_bytes(up.read())
                sources.append(("file", str(p), f"CAM-{i+1}"))
                st.success(f"{up.name}")
            else:
                sources.append(None)
        else:  # IP stream
            url = st.text_input(
                f"RTSP URL Cam {i+1}",
                placeholder="rtsp://192.168.1.x/stream",
                key=f"ip_{i}",
            )
            sources.append(("ip", url, f"CAM-{i+1}") if url else None)

    st.divider()

    model_path = MODEL_PATH

    c1, c2 = st.columns(2)
    start_clicked = c1.button("START", use_container_width=True, type="primary")
    stop_clicked  = c2.button("STOP",  use_container_width=True)

    st.divider()

    st.markdown('<div class="section-head">Active Alerts</div>',
                unsafe_allow_html=True)
    alert_log_ph = st.empty()


# ══════════════════════════════════════════════════════════════════════════════
# Start / Stop logic
# ══════════════════════════════════════════════════════════════════════════════

if start_clicked:
    if not os.path.exists(model_path):
        st.sidebar.error(
            "Trained model not found!\n\n"
            f"Expected: `{model_path}`\n\n"
            "Run `python train.py --data_dir ./data` first."
        )
    else:
        # Tear down any existing cameras
        for cs in st.session_state.cameras.values():
            cs.stop_event.set()
        st.session_state.cameras   = {}
        st.session_state.prob_hist = {}

        valid_sources = [s for s in sources if s is not None]
        if not valid_sources:
            st.sidebar.error("No valid camera source configured.")
        else:
            alert_mgr = get_alert_mgr()
            logger    = get_logger()
            for (stype, src, cam_id) in valid_sources:
                cs = CameraState(cam_id, src, VIOLENCE_THRESHOLD)
                st.session_state.cameras[cam_id]   = cs
                st.session_state.prob_hist[cam_id] = []
                _start_camera(cs, model_path,
                              st.session_state.alert_email,
                              alert_mgr, logger)
            st.session_state.running = True

if stop_clicked:
    for cs in st.session_state.cameras.values():
        cs.stop_event.set()
    st.session_state.running = False


# ══════════════════════════════════════════════════════════════════════════════
# Top analytics bar
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    "<h1 style='font-family:Share Tech Mono,monospace;color:#00d4ff;"
    "font-size:22px;letter-spacing:4px;margin-bottom:4px'>"
    "GUARDNET — CCTV INTELLIGENCE DASHBOARD</h1>",
    unsafe_allow_html=True,
)

active_cams  = sum(1 for cs in st.session_state.cameras.values() if cs.active)
total_events = sum(cs.events for cs in st.session_state.cameras.values())
total_frames = sum(cs.total  for cs in st.session_state.cameras.values())
any_violent  = any(cs.result.get("is_violent", False)
                   for cs in st.session_state.cameras.values())
sys_status   = ("THREAT DETECTED" if any_violent else
                ("ALL CLEAR" if st.session_state.running else "STANDBY"))

mc1, mc2, mc3, mc4, mc5 = st.columns(5)

mc1.markdown(f"""
<div class="stat-card">
  <div class="stat-val" style="color:{'#f85149' if any_violent else '#00d4ff'}">{active_cams}</div>
  <div class="stat-lbl">Active Cameras</div>
</div>""", unsafe_allow_html=True)

mc2.markdown(f"""
<div class="stat-card">
  <div class="stat-val" style="color:#f85149">{total_events}</div>
  <div class="stat-lbl">Violence Events</div>
</div>""", unsafe_allow_html=True)

mc3.markdown(f"""
<div class="stat-card">
  <div class="stat-val">{total_frames:,}</div>
  <div class="stat-lbl">Frames Processed</div>
</div>""", unsafe_allow_html=True)

mc4.markdown(f"""
<div class="stat-card">
  <div class="stat-val" style="color:{'#f85149' if any_violent else '#3fb950'};font-size:16px">
    {sys_status}
  </div>
  <div class="stat-lbl">System Status</div>
</div>""", unsafe_allow_html=True)

avg_fps = (
    sum(cs.capture_fps for cs in st.session_state.cameras.values())
    / max(active_cams, 1)
    if active_cams else 0.0
)
mc5.markdown(f"""
<div class="stat-card">
  <div class="stat-val" style="color:#d29922">{avg_fps:.1f}</div>
  <div class="stat-lbl">Avg Capture FPS</div>
</div>""", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Camera feed grid
# ══════════════════════════════════════════════════════════════════════════════

cameras = st.session_state.cameras
n_cams  = max(len(cameras), 1)
n_cols  = min(n_cams, 3)
cam_ids = list(cameras.keys())

if not st.session_state.running or not cameras:
    st.markdown(
        "<div style='background:#060d1a;border:2px dashed #0d2137;"
        "border-radius:8px;height:420px;display:flex;align-items:center;"
        "justify-content:center;flex-direction:column;gap:16px'>"
        "<span style='font-size:56px'>📷</span>"
        "<span style='font-family:Share Tech Mono,monospace;color:#2e5a74;"
        "font-size:14px;letter-spacing:2px'>CONFIGURE CAMERAS AND PRESS START</span>"
        "</div>",
        unsafe_allow_html=True,
    )
else:
    rows = [cam_ids[i:i + n_cols] for i in range(0, len(cam_ids), n_cols)]
    feed_placeholders = {}

    for row_ids in rows:
        cols = st.columns(len(row_ids))
        for col, cam_id in zip(cols, row_ids):
            with col:
                st.markdown(
                    f"<div class='section-head'>{cam_id}</div>",
                    unsafe_allow_html=True,
                )
                img_ph   = st.empty()
                label_ph = st.empty()
                chart_ph = st.empty()
                feed_placeholders[cam_id] = (img_ph, label_ph, chart_ph)

    # ── Render each camera ────────────────────────────────────────────────
    for cam_id, (img_ph, label_ph, chart_ph) in feed_placeholders.items():
        cs = cameras[cam_id]

        # Read latest frame — NEVER blocks; returns None if not ready yet
        frame = cs.latest_frame.read_copy()

        if frame is None:
            # Camera not yet streaming — show placeholder
            img_ph.markdown(
                "<div style='background:#060d1a;height:300px;border:1px solid"
                " #0d2a40;border-radius:4px;display:flex;align-items:center;"
                "justify-content:center'>"
                "<span style='color:#2e5a74;font-family:Share Tech Mono,"
                "monospace;font-size:12px'>BUFFERING...</span></div>",
                unsafe_allow_html=True,
            )
            continue

        # Read latest inference result (non-blocking — uses last known values)
        with cs.result_lock:
            prob       = cs.result["prob"]
            is_v       = cs.result["is_violent"]
            heatmap    = cs.result["heatmap"]
            boxes      = cs.result["boxes"]

        # Annotate
        annotated = _annotate_frame(
            frame, prob, is_v, boxes, heatmap,
            cam_id, cs.capture_fps,
        )
        # BGR → RGB for Streamlit
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img_ph.image(rgb, channels="RGB",
                     use_container_width=True, clamp=True)

        # Status label
        if is_v:
            label_ph.markdown(
                f'<div class="feed-label-violence">'
                f'VIOLENCE DETECTED &nbsp;|&nbsp; {prob*100:.1f}%</div>',
                unsafe_allow_html=True,
            )
        else:
            label_ph.markdown(
                f'<div class="feed-label-normal">'
                f'NORMAL ACTIVITY &nbsp;|&nbsp; {prob*100:.1f}%</div>',
                unsafe_allow_html=True,
            )

        # Mini probability chart
        hist = st.session_state.prob_hist.get(cam_id, [])
        hist.append(prob)
        if len(hist) > 80:
            hist.pop(0)
        st.session_state.prob_hist[cam_id] = hist

        if hist:
            fig = go.Figure(go.Scatter(
                y=hist, mode="lines",
                line=dict(color="#f85149" if is_v else "#3fb950", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(248,81,73,0.12)" if is_v
                          else "rgba(63,185,80,0.10)",
            ))
            fig.add_hline(y=VIOLENCE_THRESHOLD, line_dash="dot",
                          line_color="#d29922", line_width=1)
            fig.update_layout(
                paper_bgcolor="#040810", plot_bgcolor="#060d1a",
                margin=dict(l=0, r=0, t=0, b=0), height=80,
                yaxis=dict(range=[0, 1], showticklabels=False,
                           gridcolor="#0d2137"),
                xaxis=dict(showticklabels=False, gridcolor="#0d2137"),
                showlegend=False,
            )
            chart_ph.plotly_chart(fig, use_container_width=True,
                                  key=f"chart_{cam_id}")


# ══════════════════════════════════════════════════════════════════════════════
# Event log + sidebar alerts
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-head'>VIOLENCE EVENT LOG"
    " (only detected events stored)</div>",
    unsafe_allow_html=True,
)

logger  = get_logger()
records = logger.read_all()

if records:
    df = pd.DataFrame(records)
    df.columns = ["Timestamp", "Camera", "Confidence", "Clip Path"]
    df = df.iloc[::-1].head(50)
    st.dataframe(
        df, use_container_width=True, hide_index=True,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0, max_value=1, format="%.2f"
            ),
            "Clip Path": st.column_config.TextColumn("Clip"),
        },
    )
else:
    st.markdown(
        '<div class="no-events">No violence events logged yet.</div>',
        unsafe_allow_html=True,
    )

# Sidebar: recent alerts
recent = records[-5:] if records else []
if recent:
    alerts_html = ""
    for r in reversed(recent):
        alerts_html += (
            f"<div class='alert-card'>"
            f"{r.get('timestamp','')}<br>"
            f"CAM {r.get('camera_id','')}&nbsp;&nbsp;"
            f"CONF {float(r.get('confidence') or 0)*100:.1f}%"
            f"</div>"
        )
    alert_log_ph.markdown(alerts_html, unsafe_allow_html=True)
else:
    alert_log_ph.markdown(
        '<div class="no-events">No alerts yet.</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Auto-refresh loop
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.running:
    # Adaptive sleep: target 30 fps render ceiling.
    # We do NOT wait for a frame here — the LatestFrame slot is always
    # populated by the capture thread so st.rerun() will always have
    # a fresh frame to display.
    time.sleep(1.0 / 30.0)
    st.rerun()