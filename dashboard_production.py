"""
GuardNet – Production CCTV Dashboard
Multi-camera, real optical flow, real model, email alerts, clip extraction.
Run:  streamlit run dashboard.py
"""
import os, sys, time, threading, queue, tempfile, csv
from datetime import datetime
from pathlib import Path
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config.config import (
    MODEL_PATH, LOG_FILE, VIOLENCE_THRESHOLD,
    SEQUENCE_LENGTH, FRAME_SKIP, HEATMAP_ALPHA,
    CLIPS_DIR, PRE_BUFFER_SECONDS, POST_BUFFER_SECONDS
)

# ── Page ──────────────────────────────────────────────────────────────────────
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

/* Feed grid cells */
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _annotate_frame(frame, prob, is_violent, boxes=None,
                    heatmap_overlay=None, cam_id="CAM-1", fps=0.0):
    """Draw all overlays onto a BGR frame. Returns annotated BGR frame."""
    display = frame.copy()
    h, w    = display.shape[:2]

    # Heatmap overlay
    if heatmap_overlay is not None:
        display = heatmap_overlay

    # YOLOv8 boxes
    if boxes:
        colour = (0,30,220) if is_violent else (255,165,0)
        for (x1,y1,x2,y2) in boxes:
            cv2.rectangle(display, (x1,y1), (x2,y2), colour, 2)
            cv2.rectangle(display, (x1, y1-18), (x1+52, y1), colour, -1)
            cv2.putText(display, "Person", (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)

    # Top bar
    cv2.rectangle(display, (0,0), (w, 32), (5,10,20), -1)
    cv2.putText(display, cam_id, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0,200,80) if not is_violent else (60,60,220), 2, cv2.LINE_AA)
    ts_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(display, ts_str, (w-210, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,140,180), 1, cv2.LINE_AA)

    # Confidence bar (top-right)
    bw, bh = 160, 10
    bx, by = w - bw - 8, 38
    cv2.rectangle(display, (bx,by), (bx+bw, by+bh), (20,30,40), -1)
    bar_col = (0,30,220) if is_violent else (0,180,80)
    cv2.rectangle(display, (bx,by), (bx+int(bw*prob), by+bh), bar_col, -1)
    cv2.putText(display, f"{prob*100:.1f}%", (bx, by+bh+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, bar_col, 1, cv2.LINE_AA)

    # FPS
    cv2.putText(display, f"FPS:{fps:.0f}", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60,90,120), 1, cv2.LINE_AA)

    # Bottom status banner
    banner_colour = (8,10,40) if is_violent else (5,20,10)
    cv2.rectangle(display, (0, h-34), (w, h), banner_colour, -1)
    label  = "⚠  VIOLENCE DETECTED" if is_violent else "✔  NORMAL ACTIVITY"
    colour = (60,80,240) if is_violent else (0,200,80)
    cv2.putText(display, label, (10, h-10),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, colour, 2, cv2.LINE_AA)

    # Red border when violent
    if is_violent:
        cv2.rectangle(display, (0,0), (w-1,h-1), (0,40,220), 4)

    return display


# ── Per-Camera Stream State ───────────────────────────────────────────────────

class CameraState:
    def __init__(self, cam_id, source, threshold):
        self.cam_id     = cam_id
        self.source     = source
        self.threshold  = threshold
        self.frame_q    = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.prob       = 0.0
        self.is_violent = False
        self.fps        = 0.0
        self.total      = 0
        self.events     = 0
        self.active     = True
        self.last_alert = {}


# ── Detection Thread (one per camera) ────────────────────────────────────────

def _camera_thread(state: CameraState, model_path: str,
                   alert_email: str, alert_mgr, logger):
    from utils.preprocessing   import SequenceBuilder, MotionHeatmap
    from utils.person_detector import PersonDetector
    from utils.clip_extractor  import ClipExtractor
    from models.guardnet_model import GuardNetInference

    inferencer  = GuardNetInference(model_path)
    seq_builder = SequenceBuilder()
    person_det  = PersonDetector()
    clipper     = ClipExtractor(fps=15.0, camera_id=state.cam_id)

    cap = cv2.VideoCapture(state.source)
    if not cap.isOpened():
        state.active = False
        return

    src_fps     = cap.get(cv2.CAP_PROP_FPS) or 15.0
    prev_gray   = None
    heatmap_acc = None
    prob        = 0.0
    t_prev      = time.time()

    while not state.stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            if isinstance(state.source, str) and state.source != "0":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                seq_builder.reset()
                continue
            break

        t_now      = time.time()
        state.fps  = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev     = t_now
        h, w       = frame.shape[:2]

        if heatmap_acc is None:
            heatmap_acc = MotionHeatmap((h, w))

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            heatmap_acc.update(prev_gray, curr_gray)
        prev_gray = curr_gray

        seq = seq_builder.update(frame)
        if seq is not None:
            prob = inferencer.predict(seq)

        is_v = prob >= state.threshold

        # Clip extraction
        if is_v and not clipper.is_recording():
            clipper.start_recording(frame)
        clip_path = clipper.push(frame) if clipper.is_recording() else None

        # Alert
        if is_v:
            alert_mgr.trigger(
                prob,
                camera_id  = state.cam_id,
                alert_email= alert_email,
                clip_path  = clip_path or "",
            )

        # Heatmap overlay
        hmap = heatmap_acc.get_overlay(frame, HEATMAP_ALPHA) if prev_gray is not None else frame

        # Person boxes
        boxes = person_det.detect(frame)

        # Annotate
        annotated = _annotate_frame(
            frame, prob, is_v, boxes, hmap, state.cam_id, state.fps
        )

        # BGR → RGB for Streamlit
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        try:
            state.frame_q.put_nowait((rgb, prob, is_v))
        except queue.Full:
            try:
                state.frame_q.get_nowait()
                state.frame_q.put_nowait((rgb, prob, is_v))
            except Exception:
                pass

        state.prob      = prob
        state.is_violent= is_v
        state.total    += 1
        if is_v:
            state.events += 1

    cap.release()
    state.active = False


# ── Session init ──────────────────────────────────────────────────────────────

def _ss(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

_ss("cameras",     {})      # cam_id → CameraState
_ss("running",     False)
_ss("alert_email", "")
_ss("prob_hist",   {})      # cam_id → list
_ss("alert_log",   [])      # recent alerts shown in sidebar
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


# ── Sidebar ───────────────────────────────────────────────────────────────────

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

    # ── Alert email ──
    st.markdown('<div class="section-head">📧 Alert Email</div>',
                unsafe_allow_html=True)
    alert_email = st.text_input(
        "Send alerts to", value=st.session_state.alert_email,
        placeholder="security@yourcompany.com",
        label_visibility="collapsed"
    )
    st.session_state.alert_email = alert_email
    if alert_email:
        st.success(f"✅ Alerts → {alert_email}")
    else:
        st.info("Enter email to receive violence alerts")

    st.divider()

    # ── Camera sources ──
    st.markdown('<div class="section-head">📹 Camera Sources</div>',
                unsafe_allow_html=True)

    num_cams = st.number_input("Number of cameras", 1, 6, 1)

    sources = []
    for i in range(num_cams):
        src_type = st.radio(f"Camera {i+1} type",
                            ["Webcam", "Video File", "IP Stream"],
                            key=f"src_type_{i}", horizontal=True)
        if src_type == "Webcam":
            idx = st.number_input(f"Cam {i+1} index", 0, 8, i,
                                  key=f"cam_idx_{i}")
            sources.append(("webcam", int(idx), f"CAM-{i+1}"))
        elif src_type == "Video File":
            up = st.file_uploader(f"Upload for Cam {i+1}",
                                  type=["mp4","avi","mov","mkv"],
                                  key=f"upload_{i}")
            if up:
                p = Path(tempfile.gettempdir()) / up.name
                p.write_bytes(up.read())
                sources.append(("file", str(p), f"CAM-{i+1}"))
                st.success(f"✅ {up.name}")
            else:
                sources.append(None)
        else:  # IP stream
            url = st.text_input(f"RTSP URL Cam {i+1}",
                                placeholder="rtsp://192.168.1.x/stream",
                                key=f"ip_{i}")
            sources.append(("ip", url, f"CAM-{i+1}") if url else None)

    st.divider()

    # ── Model path (hidden from user but configurable) ──
    model_path = MODEL_PATH  # use default; not shown to user

    # ── Start / Stop ──
    c1, c2 = st.columns(2)
    start_clicked = c1.button("▶ START", use_container_width=True, type="primary")
    stop_clicked  = c2.button("■ STOP",  use_container_width=True)

    st.divider()

    # ── Live alert log in sidebar ──
    st.markdown('<div class="section-head">🚨 Active Alerts</div>',
                unsafe_allow_html=True)
    alert_log_ph = st.empty()


# ── Start / Stop logic ────────────────────────────────────────────────────────

if start_clicked:
    if not os.path.exists(model_path):
        st.sidebar.error(
            "❌ Trained model not found!\n\n"
            f"Expected: `{model_path}`\n\n"
            "Run `python train.py --data_dir ./data` first."
        )
    else:
        # Stop existing
        for cs in st.session_state.cameras.values():
            cs.stop_event.set()
        st.session_state.cameras  = {}
        st.session_state.prob_hist= {}

        valid_sources = [s for s in sources if s is not None]
        if not valid_sources:
            st.sidebar.error("No valid camera source configured.")
        else:
            alert_mgr = get_alert_mgr()
            logger    = get_logger()
            for (stype, src, cam_id) in valid_sources:
                cs = CameraState(cam_id, src, VIOLENCE_THRESHOLD)
                st.session_state.cameras[cam_id]  = cs
                st.session_state.prob_hist[cam_id] = []
                t = threading.Thread(
                    target=_camera_thread,
                    args=(cs, model_path,
                          st.session_state.alert_email,
                          alert_mgr, logger),
                    daemon=True,
                )
                t.start()
            st.session_state.running = True

if stop_clicked:
    for cs in st.session_state.cameras.values():
        cs.stop_event.set()
    st.session_state.running = False


# ── Top analytics bar ─────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-family:Share Tech Mono,monospace;color:#00d4ff;"
    "font-size:22px;letter-spacing:4px;margin-bottom:4px'>"
    "🛡️ GUARDNET — CCTV INTELLIGENCE DASHBOARD</h1>",
    unsafe_allow_html=True
)

active_cams   = sum(1 for cs in st.session_state.cameras.values() if cs.active)
total_events  = sum(cs.events for cs in st.session_state.cameras.values())
total_frames  = sum(cs.total  for cs in st.session_state.cameras.values())
any_violent   = any(cs.is_violent for cs in st.session_state.cameras.values())
sys_status    = "🔴 THREAT DETECTED" if any_violent else (
                "🟢 ALL CLEAR" if st.session_state.running else "⚫ STANDBY")

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
  <div class="stat-val" style="color:{'#f85149' if any_violent else '#3fb950'}">
    {sys_status}
  </div>
  <div class="stat-lbl">System Status</div>
</div>""", unsafe_allow_html=True)

avg_fps = (sum(cs.fps for cs in st.session_state.cameras.values()) / max(active_cams,1)
           if active_cams else 0)
mc5.markdown(f"""
<div class="stat-card">
  <div class="stat-val" style="color:#d29922">{avg_fps:.1f}</div>
  <div class="stat-lbl">Avg FPS</div>
</div>""", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ── Camera grid ───────────────────────────────────────────────────────────────

cameras = st.session_state.cameras
n_cams  = max(len(cameras), 1)
n_cols  = min(n_cams, 3)       # max 3 columns
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
        unsafe_allow_html=True
    )
else:
    # Create placeholders in a grid
    rows = [cam_ids[i:i+n_cols] for i in range(0, len(cam_ids), n_cols)]
    feed_placeholders = {}   # cam_id → (img_ph, label_ph, conf_ph, chart_ph)

    for row_ids in rows:
        cols = st.columns(len(row_ids))
        for col, cam_id in zip(cols, row_ids):
            with col:
                st.markdown(
                    f"<div class='section-head'>{cam_id}</div>",
                    unsafe_allow_html=True
                )
                img_ph   = st.empty()
                label_ph = st.empty()
                chart_ph = st.empty()
                feed_placeholders[cam_id] = (img_ph, label_ph, chart_ph)

    # Pull one frame per camera and render
    for cam_id, (img_ph, label_ph, chart_ph) in feed_placeholders.items():
        cs = cameras[cam_id]
        try:
            rgb, prob, is_v = cs.frame_q.get(timeout=0.8)
            img_ph.image(rgb, channels="RGB", use_container_width=True,
                         clamp=True)
            if is_v:
                label_ph.markdown(
                    f'<div class="feed-label-violence">'
                    f'⚠ VIOLENCE DETECTED &nbsp;|&nbsp; {prob*100:.1f}%</div>',
                    unsafe_allow_html=True
                )
            else:
                label_ph.markdown(
                    f'<div class="feed-label-normal">'
                    f'✔ NORMAL ACTIVITY &nbsp;|&nbsp; {prob*100:.1f}%</div>',
                    unsafe_allow_html=True
                )
            # Mini chart
            hist = st.session_state.prob_hist.get(cam_id, [])
            hist.append(prob)
            if len(hist) > 80:
                hist.pop(0)
            st.session_state.prob_hist[cam_id] = hist
            if hist:
                fig = go.Figure(go.Scatter(
                    y=hist, mode="lines",
                    line=dict(
                        color="#f85149" if is_v else "#3fb950",
                        width=1.5
                    ),
                    fill="tozeroy",
                    fillcolor="rgba(248,81,73,0.12)" if is_v else "rgba(63,185,80,0.10)",
                ))
                fig.add_hline(y=VIOLENCE_THRESHOLD, line_dash="dot",
                              line_color="#d29922", line_width=1)
                fig.update_layout(
                    paper_bgcolor="#040810", plot_bgcolor="#060d1a",
                    margin=dict(l=0,r=0,t=0,b=0), height=80,
                    yaxis=dict(range=[0,1], showticklabels=False,
                               gridcolor="#0d2137"),
                    xaxis=dict(showticklabels=False, gridcolor="#0d2137"),
                    showlegend=False,
                )
                chart_ph.plotly_chart(fig, use_container_width=True,
                                      key=f"chart_{cam_id}")
        except queue.Empty:
            img_ph.markdown(
                "<div style='background:#060d1a;height:300px;border:1px solid"
                " #0d2a40;border-radius:4px;display:flex;align-items:center;"
                "justify-content:center'>"
                "<span style='color:#2e5a74;font-family:Share Tech Mono,"
                "monospace;font-size:12px'>⏳ BUFFERING…</span></div>",
                unsafe_allow_html=True
            )

# ── Event log + sidebar alerts ────────────────────────────────────────────────
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-head'>📋 VIOLENCE EVENT LOG"
    " (only detected events stored)</div>",
    unsafe_allow_html=True
)

logger  = get_logger()
records = logger.read_all()

if records:
    df = pd.DataFrame(records)
    df.columns = ["Timestamp","Camera","Confidence","Clip Path"]
    df = df.iloc[::-1].head(50)
    st.dataframe(
        df, use_container_width=True, hide_index=True,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0, max_value=1, format="%.2f"
            ),
            "Clip Path": st.column_config.TextColumn("Clip"),
        }
    )
else:
    st.markdown(
        '<div class="no-events">No violence events logged yet.</div>',
        unsafe_allow_html=True
    )

# Sidebar: recent alerts
recent = records[-5:] if records else []
if recent:
    alerts_html = ""
    for r in reversed(recent):
        alerts_html += (
            f"<div class='alert-card'>"
            f"⚠ {r.get('timestamp','')}<br>"
            f"📷 {r.get('camera_id','')}&nbsp;&nbsp;"
            f"🎯 {float(r.get('confidence',0))*100:.1f}%"
            f"</div>"
        )
    alert_log_ph.markdown(alerts_html, unsafe_allow_html=True)
else:
    alert_log_ph.markdown(
        '<div class="no-events">No alerts yet.</div>',
        unsafe_allow_html=True
    )

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if st.session_state.running:
    time.sleep(0.04)
    st.rerun()
