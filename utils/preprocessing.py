"""
GuardNet – Preprocessing Pipeline  (Memory-Safe Edition)
=========================================================
Changes vs original:
  • Added scan_dataset()       — scans file paths only, zero RAM usage
  • Added VideoDataGenerator   — Keras generator, loads 1 batch at a time
  • load_dataset() kept        — legacy, only for tiny datasets < 30 clips
  • MotionHeatmap fully fixed  — motion-masked JET overlay (blue→red)
  • SequenceBuilder unchanged  — used by detection engine at runtime
"""

import cv2
import numpy as np
import os
import sys
import random
from collections import deque
from pathlib import Path
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    FRAME_WIDTH, FRAME_HEIGHT, SEQUENCE_LENGTH, FRAME_SKIP
)

FRAME_SIZE = (FRAME_HEIGHT, FRAME_WIDTH)   # (H, W)


# ══════════════════════════════════════════════════════════════════════════════
# Frame Utilities
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Resize, convert colour, and normalise a single BGR frame → float32 [0,1]."""
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (FRAME_WIDTH, FRAME_HEIGHT),
                         interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


# ══════════════════════════════════════════════════════════════════════════════
# Sequence Builder  (used by detection engine at runtime)
# ══════════════════════════════════════════════════════════════════════════════

class SequenceBuilder:
    """
    Rolling window of preprocessed frames.
    Returns (1, SEQ_LEN, H, W, 3) batch when buffer is full.
    """

    def __init__(self, seq_len: int = SEQUENCE_LENGTH):
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


# ══════════════════════════════════════════════════════════════════════════════
# Motion Heatmap  (optical flow → JET colormap overlay)
# ══════════════════════════════════════════════════════════════════════════════

class MotionHeatmap:
    """
    Farneback dense optical flow → magnitude → COLORMAP_JET overlay.
    Blue  = no/low motion
    Red   = aggressive/high-speed motion
    """

    def __init__(self, shape: Tuple[int, int]):
        self.heatmap = np.zeros(shape, dtype=np.float32)
        self.decay   = 0.88

    def update(self, prev_gray: np.ndarray, curr_gray: np.ndarray):
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        magnitude = cv2.resize(
            magnitude, (self.heatmap.shape[1], self.heatmap.shape[0])
        )
        self.heatmap = self.heatmap * self.decay + magnitude * (1 - self.decay) * 20

    def get_overlay(self, frame: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        h, w = frame.shape[:2]
        hm   = cv2.resize(self.heatmap, (w, h))
        norm = np.clip(hm, 0, None)
        mn, mx = norm.min(), norm.max()
        if mx - mn > 1e-6:
            norm = ((norm - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            return frame.copy()

        coloured     = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        motion_mask  = (norm > 15).astype(np.float32)[:, :, np.newaxis]
        blended = (
            frame.astype(np.float32) * (1 - alpha * motion_mask)
            + coloured.astype(np.float32) * alpha * motion_mask
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    def reset(self):
        self.heatmap[:] = 0


# ══════════════════════════════════════════════════════════════════════════════
# Internal helper — extract one sequence from a video file
# ══════════════════════════════════════════════════════════════════════════════

def _extract_one_sequence(
    video_path: str,
    seq_len:    int = SEQUENCE_LENGTH,
) -> Optional[np.ndarray]:
    """
    Sample `seq_len` frames evenly across the whole clip.
    Returns float32 (seq_len, H, W, 3) or None if video is too short.
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < seq_len:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, seq_len, dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame))
    cap.release()

    return (
        np.array(frames, dtype=np.float32)
        if len(frames) == seq_len
        else None
    )


def extract_frames_from_video(
    video_path: str,
    max_frames: int = 200,
    seq_len:    int = SEQUENCE_LENGTH,
) -> List[np.ndarray]:
    """
    Extract non-overlapping frame sequences from a video.
    Returns list of arrays each shaped (seq_len, H, W, 3).
    Used by legacy load_dataset only.
    """
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


# ══════════════════════════════════════════════════════════════════════════════
# Memory-Safe Dataset Scanner  ← NEW
# ══════════════════════════════════════════════════════════════════════════════

def scan_dataset(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Scan dataset folders and return (file_paths, labels).
    Does NOT load any frames — zero RAM usage.

        data_dir/violence/      → label 1
        data_dir/non-violence/  → label 0
    """
    exts      = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    class_map = {"violence": 1, "non-violence": 0}
    all_paths:  List[str] = []
    all_labels: List[int] = []

    for cls_name, label in class_map.items():
        cls_dir = Path(data_dir) / cls_name
        if not cls_dir.exists():
            print(f"[WARN] Missing directory: {cls_dir}")
            continue
        clips = [p for p in cls_dir.iterdir() if p.suffix.lower() in exts]
        print(f"  {cls_name}: {len(clips)} clips found")
        for clip in clips:
            all_paths.append(str(clip))
            all_labels.append(label)

    if not all_paths:
        raise RuntimeError(
            f"No valid video clips found in '{data_dir}'.\n"
            "Expected sub-folders:  violence/  and  non-violence/"
        )
    return all_paths, all_labels


# ══════════════════════════════════════════════════════════════════════════════
# Memory-Safe Keras Data Generator  ← NEW
# ══════════════════════════════════════════════════════════════════════════════

import tensorflow as tf  # imported here so top-level is TF-free if not needed


class VideoDataGenerator(tf.keras.utils.Sequence):
    """
    Keras Sequence generator — loads ONE BATCH at a time from disk.

    Solves the OOM error:  instead of holding 1 GB in RAM, only
    `batch_size` sequences exist in memory at any moment.

    Args:
        paths      : list of video file paths
        labels     : corresponding integer labels (0 or 1)
        batch_size : keep ≤ 2 on 8 GB RAM,  ≤ 4 on 16 GB RAM
        seq_len    : frames per sequence (must match model input)
        shuffle    : shuffle order each epoch
        augment    : random horizontal flip + brightness jitter
    """

    def __init__(
        self,
        paths:      List[str],
        labels:     List[int],
        batch_size: int  = 2,
        seq_len:    int  = SEQUENCE_LENGTH,
        shuffle:    bool = True,
        augment:    bool = False,
    ):
        self.paths      = paths
        self.labels     = labels
        self.batch_size = batch_size
        self.seq_len    = seq_len
        self.shuffle    = shuffle
        self.augment    = augment
        self.indices    = list(range(len(paths)))
        if shuffle:
            random.shuffle(self.indices)

    # ── Keras Sequence interface ───────────────────────────────────────────

    def __len__(self) -> int:
        return max(1, len(self.paths) // self.batch_size)

    def __getitem__(self, batch_idx: int):
        start = batch_idx * self.batch_size
        end   = min(start + self.batch_size, len(self.paths))
        batch_idx_list = self.indices[start:end]

        X_batch, y_batch = [], []
        for i in batch_idx_list:
            seq = _extract_one_sequence(self.paths[i], self.seq_len)
            if seq is None:
                # Corrupt/short video — pad with zeros so batch shape stays valid
                seq = np.zeros(
                    (self.seq_len, FRAME_HEIGHT, FRAME_WIDTH, 3),
                    dtype=np.float32,
                )
            if self.augment:
                seq = self._augment(seq)
            X_batch.append(seq)
            y_batch.append(self.labels[i])

        return (
            np.array(X_batch, dtype=np.float32),
            np.array(y_batch,  dtype=np.int32),
        )

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.indices)

    # ── Augmentation ──────────────────────────────────────────────────────

    @staticmethod
    def _augment(seq: np.ndarray) -> np.ndarray:
        """Random horizontal flip + brightness jitter across all frames."""
        if np.random.rand() < 0.5:
            seq = seq[:, :, ::-1, :]           # flip all frames consistently
        delta = np.random.uniform(-0.08, 0.08)
        seq   = np.clip(seq + delta, 0.0, 1.0)
        return seq


# ══════════════════════════════════════════════════════════════════════════════
# Legacy loader — kept for backwards compatibility
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(
    data_dir: str,
    seq_len:  int = SEQUENCE_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ⚠️  LEGACY — loads ALL sequences into RAM at once.
    Fine for tiny datasets (< 20 clips total).
    For anything larger use VideoDataGenerator + scan_dataset instead.
    """
    classes = {"non-violence": 0, "violence": 1}
    X, y = [], []

    for cls_name, label in classes.items():
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"[WARN] Directory not found: {cls_dir}")
            continue
        video_files = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
        print(f"[INFO] Loading {len(video_files)} videos from '{cls_name}'")
        for vf in video_files:
            path = os.path.join(cls_dir, vf)
            try:
                seqs = extract_frames_from_video(path, seq_len=seq_len)
                for seq in seqs:
                    X.append(seq)
                    y.append(label)
            except Exception as e:
                print(f"[WARN] Skipping {vf}: {e}")

    if not X:
        raise RuntimeError("No sequences loaded. Check data_dir structure.")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)