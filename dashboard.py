"""
GuardNet – Streamlit Dashboard
Run with:  streamlit run dashboard.py
"""

import os
import sys
import time
import threading
import queue
import csv
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Streamlit import ──────────────────────────────────────────────────────────
try:
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    print("Install dashboard deps:  pip install streamlit pandas plotly")
    sys.exit(1)

from config.config import (
    MODEL_PATH, LOG_FILE, VIOLENCE_THRESHOLD,
    SEQUENCE_LENGTH, FRAME_SKIP
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GuardNet Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d1117; }
  [data-testid="stSidebar"] { background: #161b22; }
  h1,h2,h3 { color: #e6edf3; }
  .metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 16px; text-align: center;
  }
  .alert-box {
    background: #3d1515; border: 2px solid #f85149;
    border-radius: 8px; padding: 12px; color: #f85149;
    font-weight: bold; text-align: center;
  }
  .normal-box {
    background: #0d2818; border: 2px solid #3fb950;
    border-radius: 8px; padding: 12px; color: #3fb950;
    font-weight: bold; text-align: center;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = dict(
        running=False,
        prob_history=[],
        violence_count=0,
        total_frames=0,
        last_prob=0.0,
        is_violent=False,
        frame_queue=queue.Queue(maxsize=4),
        stop_event=threading.Event(),
        model_loaded=False,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/fluency/96/shield.png",
    width=72
)
st.sidebar.title("GuardNet 🛡️")
st.sidebar.markdown("AI Violence Detection System")
st.sidebar.divider()

source_type = st.sidebar.radio("Input Source", ["Webcam", "Video File"])
if source_type == "Webcam":
    cam_idx = st.sidebar.number_input("Camera Index", 0, 8, 0)
    video_source = int(cam_idx)
else:
    uploaded = st.sidebar.file_uploader(
        "Upload Video", type=["mp4", "avi", "mov", "mkv"]
    )
    if uploaded:
        tmp = Path("/tmp") / uploaded.name
        tmp.write_bytes(uploaded.read())
        video_source = str(tmp)
    else:
        video_source = None

threshold = st.sidebar.slider(
    "Violence Threshold", 0.1, 0.99, VIOLENCE_THRESHOLD, 0.01
)
show_heatmap = st.sidebar.checkbox("Motion Heatmap Overlay", value=True)
model_path = st.sidebar.text_input("Model Path", MODEL_PATH)
demo_mode  = st.sidebar.checkbox("Demo Mode (untrained model)", value=True)

st.sidebar.divider()
col_s1, col_s2 = st.sidebar.columns(2)


# ─────────────────────────────────────────────────────────────────────────────
# Background detection thread
# ─────────────────────────────────────────────────────────────────────────────
def _detection_thread(
    source, model_path, threshold, show_heatmap, stop_event, frame_q, demo_mode
):
    """Runs in a background thread; pushes (frame, prob) to frame_q."""
    import config.config as cfg
    cfg.VIOLENCE_THRESHOLD = threshold

    if demo_mode or not os.path.exists(model_path):
        from models.guardnet_model import create_demo_model
        create_demo_model(model_path)

    from models.guardnet_model import GuardNetInference
    from utils.preprocessing import SequenceBuilder, MotionHeatmap
    from utils.alerts import get_alert_manager
    from utils.person_detector import PersonDetector

    inferencer  = GuardNetInference(model_path)
    seq_builder = SequenceBuilder()
    alert_mgr   = get_alert_manager()
    person_det  = PersonDetector()
    prob        = 0.0
    prev_gray   = None
    heatmap_acc = None

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        h, w = frame.shape[:2]
        if heatmap_acc is None:
            heatmap_acc = MotionHeatmap((h, w))

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None and show_heatmap:
            heatmap_acc.update(prev_gray, curr_gray)
        prev_gray = curr_gray

        seq = seq_builder.update(frame)
        if seq is not None:
            prob = inferencer.predict(seq)
            if prob >= threshold:
                alert_mgr.trigger(prob, source="dashboard")

        display = frame.copy()
        if show_heatmap and prev_gray is not None:
            display = heatmap_acc.get_overlay(display, 0.35)

        boxes = person_det.detect(frame)
        if boxes:
            display = person_det.draw(display, boxes, prob >= threshold)

        display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

        if not frame_q.full():
            frame_q.put_nowait((display, prob))

        st.session_state.prob_history.append(prob)
        if len(st.session_state.prob_history) > 300:
            st.session_state.prob_history.pop(0)
        st.session_state.total_frames += 1
        if prob >= threshold:
            st.session_state.violence_count += 1
        st.session_state.last_prob  = prob
        st.session_state.is_violent = prob >= threshold

    cap.release()


def start_detection():
    if video_source is None and source_type == "Video File":
        st.sidebar.error("Please upload a video file first.")
        return
    st.session_state.stop_event.clear()
    st.session_state.running = True
    t = threading.Thread(
        target=_detection_thread,
        args=(
            video_source, model_path, threshold,
            show_heatmap,
            st.session_state.stop_event,
            st.session_state.frame_queue,
            demo_mode,
        ),
        daemon=True,
    )
    t.start()


def stop_detection():
    st.session_state.stop_event.set()
    st.session_state.running = False


with col_s1:
    if st.button("▶ Start", use_container_width=True, type="primary"):
        start_detection()

with col_s2:
    if st.button("■ Stop", use_container_width=True):
        stop_detection()


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────
st.title("🛡️ GuardNet – Real-Time Violence Detection Dashboard")
st.divider()

col_feed, col_stats = st.columns([2, 1], gap="large")

with col_feed:
    st.subheader("📷 Live Feed")
    video_placeholder = st.empty()
    status_placeholder = st.empty()

with col_stats:
    st.subheader("📊 Analytics")
    m1, m2 = st.columns(2)
    frames_ph   = m1.empty()
    violence_ph = m2.empty()
    conf_ph     = st.empty()
    chart_ph    = st.empty()

st.divider()
st.subheader("📋 Event Log")
log_ph = st.empty()

# ─────────────────────────────────────────────────────────────────────────────
# Render loop (Streamlit re-runs on every interaction; we poll frame_queue)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.running:
    try:
        frame, prob = st.session_state.frame_queue.get_nowait()
        video_placeholder.image(frame, use_container_width=True)
    except queue.Empty:
        pass

    is_v = st.session_state.is_violent
    if is_v:
        status_placeholder.markdown(
            '<div class="alert-box">⚠️ VIOLENCE DETECTED</div>',
            unsafe_allow_html=True
        )
    else:
        status_placeholder.markdown(
            '<div class="normal-box">✔ Normal Activity</div>',
            unsafe_allow_html=True
        )

    frames_ph.metric("Total Frames", st.session_state.total_frames)
    violence_ph.metric("Violence Events", st.session_state.violence_count)

    conf_ph.progress(
        float(st.session_state.last_prob),
        text=f"Confidence: {st.session_state.last_prob*100:.1f}%"
    )

    if st.session_state.prob_history:
        hist = st.session_state.prob_history[-100:]
        fig = go.Figure(go.Scatter(
            y=hist, mode="lines",
            line=dict(color="#f85149" if is_v else "#3fb950", width=2),
            fill="tozeroy",
            fillcolor="rgba(248,81,73,0.15)" if is_v else "rgba(63,185,80,0.15)",
        ))
        fig.add_hline(y=threshold, line_dash="dash",
                      line_color="#d29922", annotation_text="Threshold")
        fig.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font_color="#e6edf3", margin=dict(l=0, r=0, t=0, b=0),
            height=180, yaxis=dict(range=[0, 1]),
            xaxis=dict(showticklabels=False),
        )
        chart_ph.plotly_chart(fig, use_container_width=True, key="prob_chart")

# Event log table
if os.path.exists(LOG_FILE):
    try:
        df = pd.read_csv(LOG_FILE)
        if not df.empty:
            log_ph.dataframe(
                df.tail(20).iloc[::-1],
                use_container_width=True,
                hide_index=True,
            )
        else:
            log_ph.info("No violence events logged yet.")
    except Exception:
        log_ph.info("Log file empty or unreadable.")
else:
    log_ph.info("No events logged yet.")

# Auto-refresh while running
if st.session_state.running:
    time.sleep(0.04)   # ~25 FPS refresh
    st.rerun()
