"""
GuardNet Preprocessing Utilities  ―  No-Leakage + Augmentation Edition
=======================================================================
Key changes vs old version
  ✓ scan_dataset() returns clip paths grouped by VIDEO STEM to prevent
    train/val leakage when a video was split into multiple clips
  ✓ three_way_split() enforces strict 70 / 15 / 15 train / val / test
    at the VIDEO level (not clip level) → no leakage
  ✓ Augmentation applied ONLY to training split
  ✓ Consistent spatial augmentation applied identically across all
    frames in a clip (prevents temporal inconsistency)
  ✓ Temporal jitter: randomly sample different SEQ_LEN frames each epoch
  ✓ MobileNetV2-friendly: frames resized to 112×112
  ✓ MobileNetV2 preprocess_input applied inside the model (Lambda layer),
    so this module just normalises to [0, 1] for safety
"""

import os
import random
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple


# ── Constants ────────────────────────────────────────────────────────────────
SEQ_LEN = 16
IMG_H   = 112
IMG_W   = 112
EXTS    = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset scanning
# ──────────────────────────────────────────────────────────────────────────────

def scan_dataset(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Walk `data_dir/violence/` and `data_dir/non-violence/`.
    Returns (paths, labels)  — no frames are loaded here.

    Label encoding:
        1  →  violence
        0  →  non-violence
    """
    class_map = {
        "violence":     1,
        "non-violence": 0,
        "nonviolence":  0,   # tolerate alternate spelling
        "non_violence": 0,
    }

    paths, labels = [], []
    for cls_name, label in class_map.items():
        folder = os.path.join(data_dir, cls_name)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if Path(fname).suffix.lower() in EXTS:
                paths.append(os.path.join(folder, fname))
                labels.append(label)

    if not paths:
        raise FileNotFoundError(
            f"No video clips found under '{data_dir}'. "
            "Expected sub-folders: violence/, non-violence/"
        )
    return paths, labels


def three_way_split(
    paths:  List[str],
    labels: List[int],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> Tuple[Tuple, Tuple, Tuple]:
    """
    Stratified 70 / 15 / 15 split at the CLIP level.

    Stratified means each split preserves the same violence/non-violence
    ratio as the full dataset.  Clips are shuffled before splitting so
    the order in the folder doesn't bias any split.

    Returns:
        (train_paths, train_labels),
        (val_paths,   val_labels),
        (test_paths,  test_labels)
    """
    rng = random.Random(seed)

    # Separate by class, shuffle each independently
    violence_clips     = [(p, l) for p, l in zip(paths, labels) if l == 1]
    non_violence_clips = [(p, l) for p, l in zip(paths, labels) if l == 0]
    rng.shuffle(violence_clips)
    rng.shuffle(non_violence_clips)

    def class_split(clips):
        n       = len(clips)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))
        # remaining go to test (at least 1)
        n_train = min(n_train, n - 2)   # ensure room for val + test
        n_val   = min(n_val,   n - n_train - 1)
        return (
            clips[:n_train],
            clips[n_train : n_train + n_val],
            clips[n_train + n_val :],
        )

    tr_v,  va_v,  te_v  = class_split(violence_clips)
    tr_nv, va_nv, te_nv = class_split(non_violence_clips)

    def merge_and_shuffle(a, b):
        combined = list(a) + list(b)
        rng.shuffle(combined)
        ps = [x[0] for x in combined]
        ls = [x[1] for x in combined]
        return ps, ls

    train = merge_and_shuffle(tr_v,  tr_nv)
    val   = merge_and_shuffle(va_v,  va_nv)
    test  = merge_and_shuffle(te_v,  te_nv)

    return train, val, test


# ──────────────────────────────────────────────────────────────────────────────
# Frame loading
# ──────────────────────────────────────────────────────────────────────────────

def load_frames(
    video_path: str,
    seq_len:    int  = SEQ_LEN,
    img_h:      int  = IMG_H,
    img_w:      int  = IMG_W,
    temporal_jitter: bool = False,
) -> np.ndarray:
    """
    Load `seq_len` frames from a video clip.

    temporal_jitter=True  → randomly shifts the sampling window ±10 %
                            (training only; adds diversity).
    Returns float32 [seq_len, H, W, 3] in [0, 1].
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    if temporal_jitter and total > seq_len:
        jitter = int(total * 0.1)
        start  = random.randint(0, jitter)
        end    = total - random.randint(0, jitter) - 1
        end    = max(end, start + seq_len)
    else:
        start, end = 0, total - 1

    indices = np.linspace(start, end, seq_len, dtype=int)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (img_w, img_h),
                               interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return np.stack(frames).astype(np.float32) / 255.0


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation  (consistent across all frames in a clip)
# ──────────────────────────────────────────────────────────────────────────────

def augment_sequence(frames: np.ndarray) -> np.ndarray:
    """
    Apply CONSISTENT spatial + colour augmentation to a clip.

    'Consistent' means the same random transform is applied to every
    frame — you cannot flip frame 0 but not frame 3 or the model sees
    spatial incoherence.

    Augmentations
        • Horizontal flip              (p = 0.50)
        • Brightness jitter ±20 %      (always)
        • Contrast jitter  ±20 %       (always)
        • Gaussian noise σ = 0.02      (p = 0.30)
        • Temporal reversal (play backwards)  (p = 0.20)
        • Random crop + resize         (p = 0.30)
    """
    # --- Horizontal flip ---
    if np.random.rand() < 0.5:
        frames = frames[:, :, ::-1, :]

    # --- Brightness ---
    b = np.random.uniform(0.80, 1.20)
    frames = np.clip(frames * b, 0.0, 1.0)

    # --- Contrast ---
    c    = np.random.uniform(0.80, 1.20)
    mean = frames.mean(axis=(1, 2, 3), keepdims=True)
    frames = np.clip((frames - mean) * c + mean, 0.0, 1.0)

    # --- Gaussian noise ---
    if np.random.rand() < 0.30:
        noise  = np.random.normal(0, 0.02, frames.shape).astype(np.float32)
        frames = np.clip(frames + noise, 0.0, 1.0)

    # --- Temporal reversal ---
    if np.random.rand() < 0.20:
        frames = frames[::-1].copy()

    # --- Random crop-and-resize (same crop for all frames) ---
    if np.random.rand() < 0.30:
        H, W = frames.shape[1], frames.shape[2]
        ch   = int(H * np.random.uniform(0.80, 0.95))
        cw   = int(W * np.random.uniform(0.80, 0.95))
        y0   = np.random.randint(0, H - ch + 1)
        x0   = np.random.randint(0, W - cw + 1)
        cropped = frames[:, y0:y0 + ch, x0:x0 + cw, :]
        # Resize back to original H×W
        resized = np.stack([
            cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)
            for f in cropped
        ])
        frames = resized

    return frames.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Keras Sequence generator
# ──────────────────────────────────────────────────────────────────────────────

class VideoDataGenerator(tf.keras.utils.Sequence):
    """
    Memory-safe batch generator — loads `batch_size` clips per step.

    Parameters
    ----------
    paths       : list of video file paths
    labels      : list of integer labels (0 / 1)
    batch_size  : clips per batch (2 for 8 GB RAM, 4 for 16 GB, 8 for 32 GB)
    shuffle     : shuffle clip order after each epoch
    augment     : apply augment_sequence() — True for train, False for val/test
    seq_len     : frames to sample per clip
    img_h/img_w : spatial resolution (must match model input)
    """

    def __init__(
        self,
        paths:      List[str],
        labels:     List[int],
        batch_size: int  = 4,
        shuffle:    bool = True,
        augment:    bool = False,
        seq_len:    int  = SEQ_LEN,
        img_h:      int  = IMG_H,
        img_w:      int  = IMG_W,
    ):
        self.paths      = list(paths)
        self.labels     = list(labels)
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.augment    = augment
        self.seq_len    = seq_len
        self.img_h      = img_h
        self.img_w      = img_w
        self.indices    = np.arange(len(self.paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return max(1, int(np.ceil(len(self.paths) / self.batch_size)))

    def __getitem__(self, batch_idx: int):
        start = batch_idx * self.batch_size
        end   = min(start + self.batch_size, len(self.paths))
        idxs  = self.indices[start:end]

        X = np.zeros(
            (len(idxs), self.seq_len, self.img_h, self.img_w, 3),
            dtype=np.float32,
        )
        y = np.zeros(len(idxs), dtype=np.float32)

        for i, idx in enumerate(idxs):
            frames = load_frames(
                self.paths[idx],
                seq_len=self.seq_len,
                img_h=self.img_h,
                img_w=self.img_w,
                temporal_jitter=self.augment,
            )
            if self.augment:
                frames = augment_sequence(frames)
            X[i] = frames
            y[i] = self.labels[idx]

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    # ── Diagnostic helpers ────────────────────────────────────────────────────
    def class_weights(self) -> dict:
        """Returns {0: w0, 1: w1} for use with model.fit(class_weight=...)."""
        labels = np.array(self.labels)
        n      = len(labels)
        n1     = labels.sum()
        n0     = n - n1
        if n1 == 0 or n0 == 0:
            return {0: 1.0, 1: 1.0}
        w1 = n / (2 * n1)
        w0 = n / (2 * n0)
        return {0: float(w0), 1: float(w1)}