"""
GuardNet – Preprocessing Pipeline
"""
import cv2
import numpy as np
from collections import deque
from typing import List, Tuple, Optional
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import FRAME_WIDTH, FRAME_HEIGHT, SEQUENCE_LENGTH, FRAME_SKIP


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


class SequenceBuilder:
    def __init__(self, seq_len=SEQUENCE_LENGTH):
        self.seq_len  = seq_len
        self.buffer   = deque(maxlen=seq_len)
        self._counter = 0

    def update(self, frame: np.ndarray) -> Optional[np.ndarray]:
        self._counter += 1
        if self._counter % FRAME_SKIP != 0:
            return None
        self.buffer.append(preprocess_frame(frame))
        if len(self.buffer) == self.seq_len:
            seq = np.array(self.buffer, dtype=np.float32)
            return np.expand_dims(seq, axis=0)
        return None

    def reset(self):
        self.buffer.clear()
        self._counter = 0


class MotionHeatmap:
    """Optical flow heatmap with proper colour mapping."""
    def __init__(self, shape: Tuple[int, int]):
        self.heatmap = np.zeros(shape, dtype=np.float32)
        self.decay   = 0.88

    def update(self, prev_gray: np.ndarray, curr_gray: np.ndarray):
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        magnitude = cv2.resize(magnitude, (self.heatmap.shape[1], self.heatmap.shape[0]))
        # Exponential moving average
        self.heatmap = self.heatmap * self.decay + magnitude * (1 - self.decay) * 20

    def get_overlay(self, frame: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        h, w = frame.shape[:2]
        hm   = cv2.resize(self.heatmap, (w, h))
        # Normalise to 0-255
        norm = np.clip(hm, 0, None)
        mn, mx = norm.min(), norm.max()
        if mx - mn > 1e-6:
            norm = ((norm - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            norm = np.zeros((h, w), dtype=np.uint8)
        # COLORMAP_JET: blue (cold/no motion) → red (hot/aggressive motion)
        coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        # Only overlay where there is actual motion (avoid painting still regions)
        motion_mask = (norm > 15).astype(np.float32)[:, :, np.newaxis]
        blended = frame.astype(np.float32) * (1 - alpha * motion_mask) + \
                  coloured.astype(np.float32) * alpha * motion_mask
        return np.clip(blended, 0, 255).astype(np.uint8)

    def reset(self):
        self.heatmap[:] = 0


def extract_frames_from_video(video_path, max_frames=200, seq_len=SEQUENCE_LENGTH):
    cap    = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame))
    cap.release()
    sequences = []
    for start in range(0, len(frames) - seq_len + 1, seq_len):
        seq = np.array(frames[start:start + seq_len], dtype=np.float32)
        if seq.shape[0] == seq_len:
            sequences.append(seq)
    return sequences


def load_dataset(data_dir, seq_len=SEQUENCE_LENGTH):
    classes = {"non-violence": 0, "violence": 1}
    X, y   = [], []
    for cls, label in classes.items():
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for vf in os.listdir(cls_dir):
            if not vf.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue
            try:
                seqs = extract_frames_from_video(
                    os.path.join(cls_dir, vf), seq_len=seq_len)
                for s in seqs:
                    X.append(s); y.append(label)
            except Exception as e:
                print(f"[WARN] {vf}: {e}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)
