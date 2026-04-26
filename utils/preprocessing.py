"""
GuardNet – Preprocessing Pipeline
Handles frame extraction, normalization, and sequence building.
"""

import cv2
import numpy as np
import os
from collections import deque
from typing import List, Tuple, Optional, Generator

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    FRAME_WIDTH, FRAME_HEIGHT, SEQUENCE_LENGTH, FRAME_SKIP
)


# ─── Frame Utilities ──────────────────────────────────────────────────────────

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Resize, convert colour, and normalise a single BGR frame."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized   = cv2.resize(frame_rgb, (FRAME_WIDTH, FRAME_HEIGHT),
                           interpolation=cv2.INTER_AREA)
    normalised = resized.astype(np.float32) / 255.0
    return normalised


def build_optical_flow(prev_gray: np.ndarray,
                       curr_gray: np.ndarray) -> np.ndarray:
    """Return a 2-channel (dx, dy) optical-flow map normalised to [0, 1]."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    # Normalise each channel independently to [0, 1]
    for c in range(2):
        ch = flow[:, :, c]
        mn, mx = ch.min(), ch.max()
        if mx - mn > 1e-6:
            flow[:, :, c] = (ch - mn) / (mx - mn)
        else:
            flow[:, :, c] = 0.0
    return flow.astype(np.float32)


# ─── Sequence Builder ─────────────────────────────────────────────────────────

class SequenceBuilder:
    """
    Maintains a rolling window of preprocessed frames.
    Returns a ready-to-infer batch whenever the buffer is full.
    """

    def __init__(self, seq_len: int = SEQUENCE_LENGTH):
        self.seq_len  = seq_len
        self.buffer   = deque(maxlen=seq_len)
        self._counter = 0

    def update(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Push one raw BGR frame.
        Returns np.ndarray of shape (1, seq_len, H, W, 3) when ready,
        else None.
        """
        self._counter += 1
        if self._counter % FRAME_SKIP != 0:
            return None

        processed = preprocess_frame(frame)
        self.buffer.append(processed)

        if len(self.buffer) == self.seq_len:
            sequence = np.array(self.buffer, dtype=np.float32)   # (T, H, W, 3)
            return np.expand_dims(sequence, axis=0)              # (1, T, H, W, 3)
        return None

    def reset(self):
        self.buffer.clear()
        self._counter = 0


# ─── Dataset Preprocessing ────────────────────────────────────────────────────

def extract_frames_from_video(
    video_path: str,
    max_frames: int = 200,
    seq_len: int = SEQUENCE_LENGTH
) -> List[np.ndarray]:
    """
    Extract non-overlapping frame sequences from a video file.
    Returns list of arrays, each shape (seq_len, H, W, 3).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame))
    cap.release()

    # Build non-overlapping windows
    sequences = []
    for start in range(0, len(frames) - seq_len + 1, seq_len):
        seq = np.array(frames[start:start + seq_len], dtype=np.float32)
        if seq.shape[0] == seq_len:
            sequences.append(seq)
    return sequences


def load_dataset(
    data_dir: str,
    seq_len: int = SEQUENCE_LENGTH
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a dataset structured as:
        data_dir/
            violence/      ← .mp4/.avi clips
            non-violence/  ← .mp4/.avi clips

    Returns X of shape (N, seq_len, H, W, 3) and y of shape (N,).
    """
    classes = {"non-violence": 0, "violence": 1}
    X, y = [], []

    for class_name, label in classes.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"[WARN] Directory not found: {class_dir}")
            continue

        video_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
        print(f"[INFO] Loading {len(video_files)} videos from '{class_name}'")

        for vf in video_files:
            path = os.path.join(class_dir, vf)
            try:
                sequences = extract_frames_from_video(path, seq_len=seq_len)
                for seq in sequences:
                    X.append(seq)
                    y.append(label)
            except Exception as e:
                print(f"[WARN] Skipping {vf}: {e}")

    if not X:
        raise RuntimeError("No sequences loaded. Check data_dir structure.")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ─── Heatmap Accumulator ──────────────────────────────────────────────────────

class MotionHeatmap:
    """Accumulates per-frame optical-flow magnitude into a running heatmap."""

    def __init__(self, shape: Tuple[int, int]):
        self.heatmap = np.zeros(shape, dtype=np.float32)
        self.decay   = 0.92

    def update(self, prev_gray: np.ndarray, curr_gray: np.ndarray):
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        magnitude = cv2.resize(magnitude, (self.heatmap.shape[1],
                                           self.heatmap.shape[0]))
        self.heatmap = self.heatmap * self.decay + magnitude

    def get_overlay(self, frame: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        norm = cv2.normalize(self.heatmap, None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)
        coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 1 - alpha, coloured, alpha, 0)

    def reset(self):
        self.heatmap[:] = 0
