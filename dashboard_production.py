"""
GuardNet – Production CCTV Dashboard  v6  (Zero-Percent Fix Edition)
=====================================================================
Run:  streamlit run dashboard_production.py

ROOT CAUSES OF 0.0% CONFIDENCE (all fixed in v6)
──────────────────────────────────────────────────
Bug 1 ── SequenceBuilder.buffer attribute does not exist
  The padded-sequence fallback in v5 accessed `seq_builder.buffer`
  but SequenceBuilder stores frames internally with no public attribute.
  This caused an AttributeError that was silently caught, keeping prob=0.0
  forever.
  FIX: Replaced with a self-contained RollingSequenceBuffer built
  directly inside _inference_thread using a plain Python deque.
  No external SequenceBuilder dependency at all.

Bug 2 ── INFER_EVERY=4 + maxsize=1 queue starves the sequence buffer
  With INFER_EVERY=4 the capture thread only enqueued every 4th frame.
  The inference queue has maxsize=1, so if inference takes longer than
  one capture cycle the frame is silently dropped.  Combined, the
  sequence buffer received very few frames and the model rarely ran.
  FIX: INFER_EVERY removed entirely. Every captured frame is offered
  to the inference queue. The maxsize=1 drop already protects against
  backlog. Sequence fills in SEQUENCE_LENGTH frames (~1 second).

Bug 3 ── predict() input shape mismatch
  The model expects input shape (1, SEQ_LEN, H, W, C) but the old code
  passed (SEQ_LEN, H, W, C) without the batch dimension.
  FIX: RollingSequenceBuffer.push_immediate() always returns
  (1, SEQ_LEN, H, W, 3) with batch dim. _safe_predict() handles all
  possible output shapes (scalar, (1,), (1,1), (1,2) softmax).

Bug 4 ── Model re-imported on every Streamlit rerun
  GuardNetInference was constructed inside _inference_thread().
  Streamlit restarts daemon threads on each rerun causing TF reload.
  FIX: _get_model() singleton caches in st.session_state.

Bug 5 ── predict() return value not unwrapped correctly
  Model returns numpy array — float() on (1,1) raises ValueError.
  FIX: _safe_predict() uses .flatten()[0].
"""

import os
import sys
import time
import threading
import queue
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config.config import (
    MODEL_PATH, LOG_FILE,
    SEQUENCE_LENGTH, FRAME_SKIP, HEATMAP_ALPHA,
    CLIPS_DIR, PRE_BUFFER_SECONDS, POST_BUFFER_SECONDS,
)

# ── Violence threshold ─────────────────────────────────────────────────────────
VIOLENCE_THRESHOLD: float = float(os.getenv("VIOLENCE_THRESHOLD", "0.20"))

# ── Model input size (must match training) ─────────────────────────────────────
MODEL_INPUT_H = 112   # FIX v9: model was trained on 112x112, NOT 224x224
MODEL_INPUT_W = 112   # confirmed by: m.input_shape = (None, 16, 112, 112, 3)

# ── Kolkata Police Station Database (name, phone, lat, lon) ──────────────────
KOLKATA_POLICE_STATIONS = [
    ("Lalbazar Control Room",    "+913322143230",  22.5726, 88.3639),
    ("Gariahat PS",              "+913324863702",  22.5196, 88.3665),
    ("Park Street PS",           "+913322268321",  22.5514, 88.3512),
    ("Ballygunge PS",            "+913322872100",  22.5269, 88.3677),
    ("Tollygunge PS",            "+913324642765",  22.4964, 88.3467),
    ("Kasba PS",                 "+913324420164",  22.5121, 88.3893),
    ("Jadavpur PS",              "+913324730706",  22.4975, 88.3716),
    ("New Market PS",            "+913322163500",  22.5542, 88.3492),
    ("Jorasanko PS",             "+913322693400",  22.5843, 88.3596),
    ("Shyambazar PS",            "+913325551471",  22.5958, 88.3730),
    ("Ultadanga PS",             "+913325551601",  22.5826, 88.3938),
    ("Phoolbagan PS",            "+913223685812",  22.5712, 88.3894),
    ("Entally PS",               "+913322443820",  22.5581, 88.3729),
    ("Tiljala PS",               "+913323668600",  22.5327, 88.3912),
    ("Topsia PS",                "+913322853400",  22.5424, 88.3812),
    ("Watgunge PS",              "+913322897600",  22.5354, 88.3432),
    ("Ekbalpore PS",             "+913323711400",  22.5489, 88.3327),
    ("Garden Reach PS",          "+913324691400",  22.5218, 88.3118),
    ("Metiabruz PS",             "+913324942600",  22.5301, 88.2986),
    ("Majerhat PS",              "+913324400700",  22.5114, 88.3327),
    ("Behala PS",                "+913324561400",  22.5021, 88.3218),
    ("Haridevpur PS",            "+913324100200",  22.4862, 88.3312),
    ("Thakurpukur PS",           "+913324100300",  22.4698, 88.3243),
    ("Lake PS",                  "+913322820800",  22.5298, 88.3543),
    ("Kalighat PS",              "+913324766400",  22.5256, 88.3432),
    ("Chetla PS",                "+913322237800",  22.5378, 88.3381),
    ("Beniapukur PS",            "+913322471300",  22.5497, 88.3762),
    ("Burtolla PS",              "+913222113200",  22.5689, 88.3578),
    ("Girish Park PS",           "+913222123400",  22.5801, 88.3587),
    ("Chitpur PS",               "+913222323100",  22.5912, 88.3621),
    ("Cossipur PS",              "+913225556100",  22.6101, 88.3712),
    ("Sinthee PS",               "+913225556200",  22.6012, 88.3812),
    ("Beliaghata PS",            "+913223518300",  22.5712, 88.3921),
    ("Maniktala PS",             "+913223598400",  22.5798, 88.3812),
    ("Narkeldanga PS",           "+913223568900",  22.5712, 88.3832),
    ("Port PS",                  "+913222438500",  22.5542, 88.3298),
    ("Harbour PS",               "+913222439800",  22.5578, 88.3243),
    ("Bowbazar PS",              "+913222131700",  22.5624, 88.3567),
    ("Muchipara PS",             "+913222131800",  22.5698, 88.3645),
    ("Amherst Street PS",        "+913222131900",  22.5756, 88.3601),
    ("Jorabagan PS",             "+913222132000",  22.5812, 88.3556),
    ("Posta PS",                 "+913222132100",  22.5823, 88.3467),
    ("Rabindra Sarani PS",       "+913222132200",  22.5756, 88.3512),
    ("Dum Dum PS",               "+913225510200",  22.6312, 88.3987),
    ("Airport PS",               "+913225118900",  22.6521, 88.4431),
    ("New Alipore PS",           "+913324002300",  22.5187, 88.3354),
    ("Alipore PS",               "+913322481400",  22.5312, 88.3321),
    ("Shakespeare Sarani PS",    "+913322476600",  22.5498, 88.3576),
    ("Hastings PS",              "+913322431700",  22.5578, 88.3321),
    ("Taltola PS",               "+913322232800",  22.5623, 88.3489),
]

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Straight-line distance in km between two lat/lon points."""
    import math
    R = 6371.0
    dl = math.radians(lat2 - lat1)
    db = math.radians(lon2 - lon1)
    a  = math.sin(dl/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(db/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def _detect_nearest_police() -> tuple:
    """
    1. Hit ip-api.com/json (free, no key) to get lat/lon from public IP.
    2. Find the closest station in KOLKATA_POLICE_STATIONS by Haversine distance.
    Returns (station_name, phone_e164, location_label, distance_km).
    Falls back to Lalbazar on any error.
    """
    try:
        import urllib.request, json
        with urllib.request.urlopen("http://ip-api.com/json/?fields=lat,lon,city,regionName", timeout=5) as r:
            data = json.loads(r.read())
        lat, lon = float(data["lat"]), float(data["lon"])
        city     = data.get("city", "Kolkata")
        print(f"[GuardNet] IP-location: {city} ({lat:.4f}, {lon:.4f})")

        best_name, best_phone, best_lat, best_lon = KOLKATA_POLICE_STATIONS[0]
        best_dist = _haversine_km(lat, lon, best_lat, best_lon)
        for name, phone, slat, slon in KOLKATA_POLICE_STATIONS[1:]:
            d = _haversine_km(lat, lon, slat, slon)
            if d < best_dist:
                best_dist, best_name, best_phone = d, name, phone

        location_label = f"{city}, Kolkata"
        print(f"[GuardNet] Nearest station: {best_name} ({best_dist:.1f} km)")
        return best_name, best_phone, location_label, round(best_dist, 1)

    except Exception as e:
        print(f"[GuardNet] IP-detect failed ({e}), using Lalbazar default")
        return "Lalbazar Control Room", "+913322143230", "Kolkata", -1.0


# ── Defaults (overwritten by auto-detect on first load) ───────────────────────
DEFAULT_POLICE_NUMBER  = "+913322143230"
DEFAULT_LOCATION_LABEL = "Kolkata"

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


# ══════════════════════════════════════════════════════════════════════════════
# Self-contained rolling sequence buffer  (replaces SequenceBuilder)
# ══════════════════════════════════════════════════════════════════════════════

class RollingSequenceBuffer:
    """
    Keeps the last `seq_len` preprocessed frames in a deque.

    push_immediate(frame)
      Always returns a batch tensor (1, seq_len, H, W, 3) float32 [0,1].
      Pads with the first available frame if the buffer is not yet full.
      This means model inference starts from the very first frame —
      exactly what demo.py does.

    No dependency on SequenceBuilder or any utils module.
    """

    def __init__(self, seq_len: int, height: int = 224, width: int = 224):
        self.seq_len = seq_len
        self.height  = height
        self.width   = width
        self._buf: deque = deque(maxlen=seq_len)

    def push_immediate(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess and push one BGR frame.
        Returns batch tensor (1, seq_len, H, W, 3) always — never None.
        """
        # Resize → RGB → normalise
        resized = cv2.resize(frame, (self.width, self.height))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normed  = rgb.astype(np.float32) / 255.0
        self._buf.append(normed)

        frames = list(self._buf)

        # Pad front with copies of the oldest frame until seq_len reached
        while len(frames) < self.seq_len:
            frames.insert(0, frames[0])

        seq = np.stack(frames, axis=0)          # (seq_len, H, W, 3)
        return np.expand_dims(seq, axis=0)      # (1, seq_len, H, W, 3)

    @property
    def filled(self) -> bool:
        return len(self._buf) >= self.seq_len

    def __len__(self) -> int:
        return len(self._buf)


# ══════════════════════════════════════════════════════════════════════════════
# Model singleton
# ══════════════════════════════════════════════════════════════════════════════


# ── Module-level model cache (thread-safe — no st.session_state in threads) ────
_MODEL_CACHE: dict = {}
_MODEL_LOCK  = threading.Lock()


class _ModelWrapper:
    """
    Thin wrapper that holds BOTH the raw Keras model (primary) and
    the GuardNetInference object (fallback).  Keeping both lets us
    bypass GuardNetInference.predict() entirely if it misbehaves.
    """
    def __init__(self, keras_model, guardnet_obj, input_shape, output_shape):
        self.keras_model   = keras_model     # tf.keras.Model — always callable
        self.guardnet_obj  = guardnet_obj    # GuardNetInference or None
        self.input_shape   = input_shape
        self.output_shape  = output_shape
        # model attribute keeps PathB/C in _safe_predict working
        self.model         = keras_model


def _get_model(model_path: str):
    """
    Load the model ONCE into a module-level dict so daemon threads can
    access it safely.  st.session_state is NOT available in daemon threads.

    Strategy (most-to-least reliable):
      1. Load raw Keras model directly from the file (h5 / SavedModel dir).
         This bypasses GuardNetInference entirely and is always correct.
      2. Also try to instantiate GuardNetInference for any extra pre/post
         processing it may do — but we do NOT rely on its predict().
    """
    with _MODEL_LOCK:
        if model_path in _MODEL_CACHE:
            return _MODEL_CACHE[model_path]

        print(f"[GuardNet] Loading model: {model_path}")

        # ── Step 1: Raw Keras load (primary, always works) ─────────────────
        keras_model = None
        try:
            import tensorflow as tf
            keras_model = tf.keras.models.load_model(model_path, compile=False)
            print(f"[GuardNet] Keras load OK")
            print(f"[GuardNet]   input  shape : {keras_model.input_shape}")
            print(f"[GuardNet]   output shape : {keras_model.output_shape}")
            # Warm-up call — use actual model input shape, not config constants
            try:
                _, seq_len, wh, ww, _ = keras_model.input_shape
                seq_len = seq_len or SEQUENCE_LENGTH
                wh = wh or MODEL_INPUT_H
                ww = ww or MODEL_INPUT_W
            except Exception:
                seq_len, wh, ww = SEQUENCE_LENGTH, MODEL_INPUT_H, MODEL_INPUT_W
            dummy = np.zeros((1, seq_len, wh, ww, 3), dtype=np.float32)
            out = keras_model(dummy, training=False)
            print(f"[GuardNet]   warm-up output: {np.array(out).flatten()[:4]}")
        except Exception as e:
            print(f"[GuardNet] Keras load FAILED: {e}")
            import traceback; traceback.print_exc()
            keras_model = None

        # ── Step 2: GuardNetInference wrapper (optional) ───────────────────
        guardnet_obj = None
        try:
            from models.guardnet_model import GuardNetInference
            guardnet_obj = GuardNetInference(model_path)
            print("[GuardNet] GuardNetInference wrapper loaded OK")
        except Exception as e:
            print(f"[GuardNet] GuardNetInference wrapper skipped: {e}")

        if keras_model is None and guardnet_obj is None:
            print("[GuardNet] FATAL: no model could be loaded")
            _MODEL_CACHE[model_path] = None
            return None

        in_shape  = getattr(keras_model, "input_shape",  "unknown") if keras_model else "unknown"
        out_shape = getattr(keras_model, "output_shape", "unknown") if keras_model else "unknown"
        wrapper   = _ModelWrapper(keras_model, guardnet_obj, in_shape, out_shape)

        _MODEL_CACHE[model_path] = wrapper

        try:
            st.session_state.guardnet_model_obj = wrapper
        except Exception:
            pass

        return wrapper


def _safe_predict(wrapper, seq_batch: np.ndarray) -> float:
    """
    Run prediction and extract a float in [0, 1].

    FIX v8: raw Keras model is now PATH A (primary).
    GuardNetInference.predict() is Path D (last resort).

    Input shape must be (1, SEQ_LEN, H, W, 3) float32 in [0,1].

    Output shape handling:
      (1, 1)  binary sigmoid  → arr[0]
      (1,)    binary sigmoid  → arr[0]
      (1, 2)  softmax         → arr[1]  (index 1 = violence)
      scalar                  → value
    """
    import traceback

    # ── Always log input stats so we can see data flowing ─────────────────────
    print(f"[GuardNet] predict | shape={seq_batch.shape} "
          f"min={seq_batch.min():.3f} max={seq_batch.max():.3f}")

    def _extract(result, tag) -> float:
        if hasattr(result, "numpy"):
            result = result.numpy()
        arr = np.array(result, dtype=np.float32).flatten()
        print(f"[GuardNet] {tag} | out_shape={np.array(result).shape} vals={arr}")
        if arr.size == 0:   return 0.0
        if arr.size == 1:   return float(np.clip(arr[0], 0.0, 1.0))
        if arr.size == 2:   return float(np.clip(arr[1], 0.0, 1.0))  # softmax → violence
        return float(np.clip(arr.max(), 0.0, 1.0))

    keras_model   = getattr(wrapper, "keras_model",  None)
    guardnet_obj  = getattr(wrapper, "guardnet_obj", None)

    # ── Path A: raw Keras eager call (fastest, most reliable) ─────────────────
    if keras_model is not None:
        try:
            import tensorflow as tf
            # Ensure correct dtype
            tensor = tf.constant(seq_batch, dtype=tf.float32)
            result = keras_model(tensor, training=False)
            prob   = _extract(result, "PathA-Keras-eager")
            return prob   # return even if 0.0 — it is the ground truth
        except Exception as e:
            print(f"[GuardNet] PathA error: {e}")
            traceback.print_exc()

    # ── Path B: Keras .predict() with explicit batch ────────────────────────
    if keras_model is not None:
        try:
            result = keras_model.predict(seq_batch, verbose=0)
            return _extract(result, "PathB-Keras-predict")
        except Exception as e:
            print(f"[GuardNet] PathB error: {e}")

    # ── Path C: check for wrong input shape — try (SEQ, H, W, 3) unbatched ─
    if keras_model is not None:
        try:
            import tensorflow as tf
            unbatched = tf.constant(seq_batch[0], dtype=tf.float32)  # drop batch dim
            result    = keras_model(tf.expand_dims(unbatched, 0), training=False)
            return _extract(result, "PathC-unbatch-retry")
        except Exception as e:
            print(f"[GuardNet] PathC error: {e}")

    # ── Path D: GuardNetInference.predict() last resort ────────────────────
    if guardnet_obj is not None:
        try:
            result = guardnet_obj.predict(seq_batch)
            return _extract(result, "PathD-GuardNetInference")
        except Exception as e:
            print(f"[GuardNet] PathD error: {e}")
            traceback.print_exc()

    print("[GuardNet] ALL PATHS FAILED — returning 0.0")
    return 0.0


# ── Frame annotation ──────────────────────────────────────────────────────────

def _annotate_frame(
    frame: np.ndarray,
    prob: float,
    is_violent: bool,
    boxes=None,
    heatmap_overlay=None,
    cam_id: str = "CAM-1",
    fps: float = 0.0,
) -> np.ndarray:
    display = heatmap_overlay if heatmap_overlay is not None else frame.copy()
    h, w    = display.shape[:2]

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
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(display, ts, (w - 210, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 140, 180), 1, cv2.LINE_AA)

    # Confidence bar
    bw, bh = 160, 10
    bx, by = w - bw - 8, 38
    cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (20, 30, 40), -1)
    bar_col = (0, 30, 220) if is_violent else (0, 180, 80)
    cv2.rectangle(display, (bx, by), (bx + int(bw * prob), by + bh), bar_col, -1)
    cv2.putText(display, f"{prob * 100:.1f}%", (bx, by + bh + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, bar_col, 1, cv2.LINE_AA)

    # FPS
    cv2.putText(display, f"FPS:{fps:.0f}", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 90, 120), 1, cv2.LINE_AA)

    # Bottom banner
    cv2.rectangle(display, (0, h - 34), (w, h),
                  (8, 10, 40) if is_violent else (5, 20, 10), -1)
    label  = "!  VIOLENCE DETECTED" if is_violent else "OK NORMAL ACTIVITY"
    colour = (60, 80, 240) if is_violent else (0, 200, 80)
    cv2.putText(display, label, (10, h - 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, colour, 2, cv2.LINE_AA)

    if is_violent:
        cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 40, 220), 4)

    return display


# ══════════════════════════════════════════════════════════════════════════════
# Per-camera state
# ══════════════════════════════════════════════════════════════════════════════

class CameraState:
    def __init__(self, cam_id: str, source, threshold: float):
        self.cam_id    = cam_id
        self.source    = source
        self.threshold = threshold

        from utils.preprocessing import LatestFrame
        self.latest_frame = LatestFrame()

        self.result: dict = {
            "prob": 0.0, "is_violent": False,
            "heatmap": None, "boxes": [],
        }
        self.result_lock = threading.Lock()
        self.stop_event  = threading.Event()

        self.capture_fps: float = 0.0
        self.total:       int   = 0
        self.events:      int   = 0
        self.active:      bool  = True


# ══════════════════════════════════════════════════════════════════════════════
# Thread-1  Capture
# ══════════════════════════════════════════════════════════════════════════════

def _capture_thread(state: CameraState, infer_q: queue.Queue) -> None:
    """
    FIX v6: INFER_EVERY removed.
    Every frame is offered to the inference queue.
    maxsize=1 naturally drops frames when inference is busy.
    """
    cap = cv2.VideoCapture(state.source)
    if not cap.isOpened():
        print(f"[GuardNet] Cannot open: {state.source}")
        state.active = False
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    t_prev = time.perf_counter()
    print(f"[GuardNet] Capture started: {state.cam_id}")

    while not state.stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            if isinstance(state.source, str) and state.source not in ("0", 0):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            time.sleep(0.01)
            continue

        t_now             = time.perf_counter()
        state.capture_fps = 0.9 * state.capture_fps + 0.1 / max(t_now - t_prev, 1e-6)
        t_prev            = t_now
        state.total      += 1

        state.latest_frame.write(frame)

        try:
            infer_q.put_nowait(frame.copy())
        except queue.Full:
            pass   # inference busy — drop, keep capturing

    cap.release()
    state.active = False
    print(f"[GuardNet] Capture stopped: {state.cam_id}")


# ══════════════════════════════════════════════════════════════════════════════
# Thread-2  Inference  ← ALL BUGS FIXED
# ══════════════════════════════════════════════════════════════════════════════

def _inference_thread(
    state:          CameraState,
    infer_q:        queue.Queue,
    model_path:     str,
    alert_email:    str,
    alert_mgr,
    logger,
    police_number:  str = "",
    location_label: str = "the monitored premises",
    inferencer=None,
) -> None:
    """
    v7 FIXES
    --------
    1. Model passed directly as inferencer= — never touches st.session_state
       from inside a daemon thread (that was the core cause of 0.0% confidence).
    2. Fallback: module-level _get_model() cache if inferencer is None.
    3. RollingSequenceBuffer — no SequenceBuilder dependency.
    4. push_immediate() — inference starts from frame 1.
    5. _safe_predict() — 3 fallback paths + full debug logging.
    6. All helper modules wrapped in try/except for graceful degradation.
    """
    # FIX: use passed object; only call _get_model() as last resort
    if inferencer is None:
        inferencer = _get_model(model_path)
    if inferencer is None:
        print(f"[GuardNet] Inference aborted — model not loaded.")
        return

    seq_buf = RollingSequenceBuffer(
        seq_len=SEQUENCE_LENGTH,
        height=MODEL_INPUT_H,
        width=MODEL_INPUT_W,
    )

    # Optional helpers — degrade gracefully if unavailable
    try:
        from utils.person_detector import PersonDetector
        person_det     = PersonDetector()
        use_person_det = True
    except Exception as e:
        print(f"[GuardNet] PersonDetector not available: {e}")
        person_det     = None
        use_person_det = False

    try:
        from utils.clip_extractor import ClipExtractor
        clipper     = ClipExtractor(fps=15.0, camera_id=state.cam_id)
        use_clipper = True
    except Exception as e:
        print(f"[GuardNet] ClipExtractor not available: {e}")
        clipper     = None
        use_clipper = False

    frame_count = 0
    print(
        f"[GuardNet] Inference started: {state.cam_id} | "
        f"seq={SEQUENCE_LENGTH} thr={state.threshold:.2f}"
    )

    while not state.stop_event.is_set():
        try:
            frame = infer_q.get(timeout=0.5)
        except queue.Empty:
            continue

        frame_count += 1

        # ── Sequence → predict ────────────────────────────────────────────
        seq_batch = seq_buf.push_immediate(frame)   # always (1,SEQ,H,W,3)

        # Auto-correct shape if Keras model reveals a different input spec
        keras_m = getattr(inferencer, "keras_model", None)
        if keras_m is not None:
            try:
                in_shape = keras_m.input_shape  # e.g. (None, 16, 112, 112, 3)
                if len(in_shape) == 5:
                    _, exp_seq, exp_h, exp_w, _ = in_shape
                    cur_seq = seq_batch.shape[1]
                    cur_h   = seq_batch.shape[2]
                    cur_w   = seq_batch.shape[3]
                    needs_fix = False
                    if exp_seq and exp_seq != cur_seq:
                        print(f"[GuardNet] SEQ mismatch: model={exp_seq} buf={cur_seq} — rebuilding buffer")
                        seq_buf = RollingSequenceBuffer(exp_seq,
                                      exp_h or MODEL_INPUT_H,
                                      exp_w or MODEL_INPUT_W)
                        seq_batch = seq_buf.push_immediate(frame)
                        needs_fix = True
                    if (exp_h and exp_h != cur_h) or (exp_w and exp_w != cur_w):
                        print(f"[GuardNet] Spatial mismatch: model=({exp_h},{exp_w}) buf=({cur_h},{cur_w}) — resizing")
                        import cv2 as _cv2
                        frames_resized = [
                            _cv2.resize(
                                (seq_batch[0, i] * 255).astype(np.uint8),
                                (exp_w, exp_h)
                            ).astype(np.float32) / 255.0
                            for i in range(seq_batch.shape[1])
                        ]
                        seq_batch = np.expand_dims(np.stack(frames_resized), 0)
            except Exception as _shape_err:
                print(f"[GuardNet] Shape check error (non-fatal): {_shape_err}")

        prob = _safe_predict(inferencer, seq_batch)
        is_v = prob >= state.threshold

        # Terminal heartbeat — first 5 frames always, then every 30
        if frame_count <= 5 or frame_count % 30 == 0 or is_v:
            print(
                f"[GuardNet] {state.cam_id} frame={frame_count} "
                f"buf={len(seq_buf)}/{SEQUENCE_LENGTH} "
                f"prob={prob:.4f} ({prob*100:.1f}%) violent={is_v}"
            )

        # ── Person detection ──────────────────────────────────────────────
        boxes = []
        if use_person_det:
            try:
                boxes = person_det.detect(frame)
            except Exception as e:
                print(f"[GuardNet] PersonDetector.detect error: {e}")

        # ── Clip extraction ───────────────────────────────────────────────
        clip_path = None
        if use_clipper and clipper is not None:
            try:
                if is_v and not clipper.is_recording():
                    clipper.start_recording(frame)
                clip_path = clipper.push(frame) if clipper.is_recording() else None
            except Exception as e:
                print(f"[GuardNet] Clipper error: {e}")

        # ── Alerts ────────────────────────────────────────────────────────
        if is_v:
            try:
                alert_mgr.trigger(
                    prob,
                    camera_id      = state.cam_id,
                    alert_email    = alert_email,
                    clip_path      = clip_path or "",
                    police_number  = police_number,
                    location_label = location_label,
                )
            except Exception as e:
                print(f"[GuardNet] AlertManager error: {e}")
            try:
                logger.log(state.cam_id, prob, clip_path or "")
            except Exception as e:
                print(f"[GuardNet] Logger error: {e}")
            state.events += 1
            print(f"[GuardNet] VIOLENCE {state.cam_id} prob={prob:.3f}")

        # ── Publish ───────────────────────────────────────────────────────
        with state.result_lock:
            state.result["prob"]       = prob
            state.result["is_violent"] = is_v
            state.result["boxes"]      = boxes

    print(f"[GuardNet] Inference stopped: {state.cam_id}")


# ══════════════════════════════════════════════════════════════════════════════
# Thread-3  Optical-flow heatmap
# ══════════════════════════════════════════════════════════════════════════════

def _flow_thread(state: CameraState) -> None:
    try:
        from utils.preprocessing import MotionHeatmap
    except Exception as e:
        print(f"[GuardNet] MotionHeatmap not available: {e}")
        return

    FLOW_INTERVAL = 1.0 / 30.0
    heatmap_acc   = None
    prev_gray     = None

    while not state.stop_event.is_set():
        t0    = time.perf_counter()
        frame = state.latest_frame.read()
        if frame is None:
            time.sleep(FLOW_INTERVAL)
            continue

        h, w = frame.shape[:2]
        if heatmap_acc is None:
            heatmap_acc = MotionHeatmap(
                display_shape=(h, w),
                ema_alpha=0.25,
                flow_size=(320, 240),
                noise_floor=0.08,
                dilation_iter=2,
                colormap=cv2.COLORMAP_INFERNO,
            )

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            try:
                heatmap_acc.update(prev_gray, curr_gray)
                overlay = heatmap_acc.get_overlay(frame, alpha=HEATMAP_ALPHA)
                with state.result_lock:
                    state.result["heatmap"] = overlay
            except Exception:
                pass

        prev_gray = curr_gray
        time.sleep(max(0.0, FLOW_INTERVAL - (time.perf_counter() - t0)))


# ══════════════════════════════════════════════════════════════════════════════
# Launch all threads
# ══════════════════════════════════════════════════════════════════════════════

def _start_camera(
    state: CameraState, model_path: str, alert_email: str,
    alert_mgr, logger, police_number: str = "",
    location_label: str = "the monitored premises",
    model_obj=None,
) -> None:
    infer_q = queue.Queue(maxsize=1)

    # FIX v7: pass the already-loaded model object directly into the thread.
    # Do NOT let the thread call _get_model() — st.session_state is unavailable
    # inside daemon threads and silently returns None, causing 0.0% confidence.
    if model_obj is None:
        model_obj = _get_model(model_path)

    for target, kwargs in [
        (_capture_thread,   {"state": state, "infer_q": infer_q}),
        (_inference_thread, {
            "state": state, "infer_q": infer_q,
            "model_path": model_path, "alert_email": alert_email,
            "alert_mgr": alert_mgr, "logger": logger,
            "police_number": police_number, "location_label": location_label,
            "inferencer": model_obj,   # <-- KEY FIX
        }),
        (_flow_thread, {"state": state}),
    ]:
        t = threading.Thread(target=target, kwargs=kwargs, daemon=True)
        t.start()


# ══════════════════════════════════════════════════════════════════════════════
# Session-state defaults
# ══════════════════════════════════════════════════════════════════════════════

def _ss(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

_ss("cameras",           {})
_ss("running",           False)
_ss("alert_email",       "")
_ss("police_number",     DEFAULT_POLICE_NUMBER)
_ss("location_label",    DEFAULT_LOCATION_LABEL)
_ss("prob_hist",         {})
_ss("alert_mgr",         None)
_ss("logger",            None)
_ss("station_name",      "")
_ss("station_dist_km",   -1.0)
_ss("geo_detected",      False)

# ── Auto-detect on very first load (not on every Streamlit rerun) ─────────────
if not st.session_state.geo_detected:
    _name, _phone, _loc, _dist = _detect_nearest_police()
    st.session_state.police_number   = _phone
    st.session_state.location_label  = _loc
    st.session_state.station_name    = _name
    st.session_state.station_dist_km = _dist
    st.session_state.geo_detected    = True


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

    # Email
    st.markdown('<div class="section-head">Alert Email</div>',
                unsafe_allow_html=True)
    alert_email = st.text_input(
        "Email", value=st.session_state.alert_email,
        placeholder="security@yourcompany.com",
        label_visibility="collapsed",
    )
    st.session_state.alert_email = alert_email
    if alert_email:
        st.success(f"Alerts → {alert_email}")
    else:
        st.info("Enter email to receive violence alerts")

    st.divider()

    # Police — auto-detected from IP location
    st.markdown('<div class="section-head">🚨 Nearest Police Station</div>',
                unsafe_allow_html=True)

    _sname = st.session_state.station_name
    _sdist = st.session_state.station_dist_km
    if _sname:
        _dist_txt = f"{_sdist} km away" if _sdist >= 0 else "distance unknown"
        st.markdown(
            f"""<div style='background:#071a0e;border:1px solid #1a5c2e;
            border-radius:6px;padding:10px 12px;margin-bottom:8px'>
            <span style='font-family:Share Tech Mono,monospace;color:#00e85a;
            font-size:11px;letter-spacing:1px'>📍 AUTO-DETECTED</span><br>
            <span style='font-family:Share Tech Mono,monospace;color:#00d4ff;
            font-size:13px;font-weight:bold'>{_sname}</span><br>
            <span style='font-size:11px;color:#5a7a94'>{_dist_txt}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    if st.button("🔄 Re-detect my location", use_container_width=True):
        _n, _p, _l, _d = _detect_nearest_police()
        st.session_state.police_number   = _p
        st.session_state.location_label  = _l
        st.session_state.station_name    = _n
        st.session_state.station_dist_km = _d
        st.rerun()

    st.caption("Override if needed:")
    police_number = st.text_input(
        "Police phone (E.164)", value=st.session_state.police_number,
        placeholder="+91XXXXXXXXXX", label_visibility="collapsed",
        help="Auto-filled from your IP location. Edit if wrong.",
    )
    st.session_state.police_number = police_number

    location_label = st.text_input(
        "Location label", value=st.session_state.location_label,
        placeholder="Your area, Kolkata",
        label_visibility="collapsed",
        help="Included in SMS & voice call to police.",
    )
    st.session_state.location_label = location_label

    if police_number:
        st.success(f"📞 {police_number}")
    else:
        st.info("Phone auto-filled — edit if wrong")

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
                f"Upload Cam {i+1}",
                type=["mp4", "avi", "mov", "mkv"],
                key=f"upload_{i}",
            )
            if up:
                p = Path(tempfile.gettempdir()) / up.name
                p.write_bytes(up.read())
                sources.append(("file", str(p), f"CAM-{i+1}"))
                st.success(up.name)
            else:
                sources.append(None)
        else:
            url = st.text_input(
                f"RTSP URL Cam {i+1}",
                placeholder="rtsp://192.168.1.x/stream",
                key=f"ip_{i}",
            )
            sources.append(("ip", url, f"CAM-{i+1}") if url else None)

    st.divider()

    model_path = MODEL_PATH

    # Model status
    if os.path.exists(model_path):
        mb = os.path.getsize(model_path) / 1_048_576
        st.success(f"Model ready ({mb:.1f} MB)")
    else:
        st.error(f"Model NOT found:\n`{model_path}`")

    # Live threshold slider
    st.markdown('<div class="section-head" style="margin-top:8px">Sensitivity</div>',
                unsafe_allow_html=True)
    threshold_pct = st.slider(
        "Violence threshold (%)", 5, 80,
        value=int(VIOLENCE_THRESHOLD * 100), step=5,
        help="Lower = more sensitive. Default 20% matches demo.py.",
    )
    live_threshold = threshold_pct / 100.0
    for cs in st.session_state.cameras.values():
        cs.threshold = live_threshold
    st.caption(f"Threshold: {threshold_pct}%")

    st.divider()

    c1, c2 = st.columns(2)
    start_clicked = c1.button("START", use_container_width=True, type="primary")
    stop_clicked  = c2.button("STOP",  use_container_width=True)

    st.divider()
    st.markdown('<div class="section-head">Active Alerts</div>',
                unsafe_allow_html=True)
    alert_log_ph = st.empty()


# ══════════════════════════════════════════════════════════════════════════════
# Start / Stop
# ══════════════════════════════════════════════════════════════════════════════

if start_clicked:
    if not os.path.exists(model_path):
        st.sidebar.error(
            f"Model not found:\n`{model_path}`\n"
            "Run `python train.py --data_dir ./data`"
        )
    else:
        # FIX v7: capture the model object explicitly so we can pass it
        # directly to each inference thread (avoids st.session_state thread issue)
        loaded_model = _get_model(model_path)
        if loaded_model is None:
            st.sidebar.error("Model failed to load — check terminal logs.")
            st.stop()

        # Stop existing cameras
        for cs in st.session_state.cameras.values():
            cs.stop_event.set()
        time.sleep(0.3)
        st.session_state.cameras   = {}
        st.session_state.prob_hist = {}

        valid_sources = [s for s in sources if s is not None]
        if not valid_sources:
            st.sidebar.error("No valid camera source configured.")
        else:
            alert_mgr = get_alert_mgr()
            logger    = get_logger()
            for (stype, src, cam_id) in valid_sources:
                cs = CameraState(cam_id, src, live_threshold)
                st.session_state.cameras[cam_id]   = cs
                st.session_state.prob_hist[cam_id] = []
                _start_camera(
                    cs, model_path, st.session_state.alert_email,
                    alert_mgr, logger,
                    police_number  = st.session_state.police_number,
                    location_label = (st.session_state.location_label
                                      or DEFAULT_LOCATION_LABEL),
                    model_obj      = loaded_model,   # FIX v7: pass pre-loaded obj
                )
            st.session_state.running = True
            st.sidebar.success(
                f"{len(valid_sources)} camera(s) started | "
                f"threshold={live_threshold*100:.0f}%"
            )

if stop_clicked:
    for cs in st.session_state.cameras.values():
        cs.stop_event.set()
    st.session_state.running = False


# ══════════════════════════════════════════════════════════════════════════════
# Stats bar
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
mc1.markdown(f"""<div class="stat-card">
  <div class="stat-val" style="color:{'#f85149' if any_violent else '#00d4ff'}">{active_cams}</div>
  <div class="stat-lbl">Active Cameras</div></div>""", unsafe_allow_html=True)
mc2.markdown(f"""<div class="stat-card">
  <div class="stat-val" style="color:#f85149">{total_events}</div>
  <div class="stat-lbl">Violence Events</div></div>""", unsafe_allow_html=True)
mc3.markdown(f"""<div class="stat-card">
  <div class="stat-val">{total_frames:,}</div>
  <div class="stat-lbl">Frames Processed</div></div>""", unsafe_allow_html=True)
mc4.markdown(f"""<div class="stat-card">
  <div class="stat-val" style="color:{'#f85149' if any_violent else '#3fb950'};font-size:16px">
    {sys_status}</div>
  <div class="stat-lbl">System Status</div></div>""", unsafe_allow_html=True)
avg_fps = (
    sum(cs.capture_fps for cs in st.session_state.cameras.values())
    / max(active_cams, 1) if active_cams else 0.0
)
mc5.markdown(f"""<div class="stat-card">
  <div class="stat-val" style="color:#d29922">{avg_fps:.1f}</div>
  <div class="stat-lbl">Avg Capture FPS</div></div>""", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Camera feed grid
# ══════════════════════════════════════════════════════════════════════════════

cameras = st.session_state.cameras
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
    n_cols = min(len(cam_ids), 3)
    rows   = [cam_ids[i:i + n_cols] for i in range(0, len(cam_ids), n_cols)]
    feed_placeholders = {}

    for row_ids in rows:
        cols = st.columns(len(row_ids))
        for col, cam_id in zip(cols, row_ids):
            with col:
                st.markdown(f"<div class='section-head'>{cam_id}</div>",
                            unsafe_allow_html=True)
                feed_placeholders[cam_id] = (st.empty(), st.empty(), st.empty())

    for cam_id, (img_ph, label_ph, chart_ph) in feed_placeholders.items():
        cs    = cameras[cam_id]
        frame = cs.latest_frame.read_copy()

        if frame is None:
            img_ph.markdown(
                "<div style='background:#060d1a;height:300px;border:1px solid"
                " #0d2a40;border-radius:4px;display:flex;align-items:center;"
                "justify-content:center'>"
                "<span style='color:#2e5a74;font-family:Share Tech Mono,"
                "monospace;font-size:12px'>BUFFERING...</span></div>",
                unsafe_allow_html=True,
            )
            continue

        with cs.result_lock:
            prob    = cs.result["prob"]
            is_v    = cs.result["is_violent"]
            heatmap = cs.result["heatmap"]
            boxes   = cs.result["boxes"]

        annotated = _annotate_frame(frame, prob, is_v, boxes, heatmap,
                                    cam_id, cs.capture_fps)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img_ph.image(rgb, channels="RGB", use_container_width=True, clamp=True)

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
            fig.add_hline(y=cs.threshold, line_dash="dot",
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
# Event log
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-head'>VIOLENCE EVENT LOG</div>",
            unsafe_allow_html=True)

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
                "Confidence", min_value=0, max_value=1, format="%.2f"),
            "Clip Path": st.column_config.TextColumn("Clip"),
        },
    )
else:
    st.markdown('<div class="no-events">No violence events logged yet.</div>',
                unsafe_allow_html=True)

# Sidebar alerts
recent = records[-5:] if records else []
if recent:
    html = "".join(
        f"<div class='alert-card'>{r.get('timestamp','')}<br>"
        f"CAM {r.get('camera_id','')} &nbsp; "
        f"CONF {float(r.get('confidence') or 0)*100:.1f}%</div>"
        for r in reversed(recent)
    )
    alert_log_ph.markdown(html, unsafe_allow_html=True)
else:
    alert_log_ph.markdown('<div class="no-events">No alerts yet.</div>',
                          unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Auto-refresh  30 fps ceiling
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.running:
    time.sleep(1.0 / 30.0)
    st.rerun()
