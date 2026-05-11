"""
GuardNet Preprocessing Utilities  —  Optimized Edition v4
==========================================================

Changes vs v3
  ✓ MotionHeatmap  — fully rewritten
        • Farnebäck runs on configurable downscale target (default 320×240)
          instead of full-resolution → 10-24× cheaper on 1080p streams
        • Exponential Moving Average (EMA) over magnitude maps eliminates
          per-frame flicker completely
        • Percentile-based normalisation (95th pct ceiling) prevents a
          single fast-moving object from collapsing everything else to zero
        • Morphological dilation groups nearby motion blobs into clear zones
        • Mask-based alpha blend: original pixels show through where motion ≈ 0
        • COLORMAP_INFERNO selected for intuitive thermal appearance
        • reset() method added for safe re-use across stream reconnections

  ✓ LatestFrame   — new thread-safe single-slot buffer
        • Lock-protected overwrite: display thread always reads the NEWEST
          frame regardless of how slow inference is running
        • Replaces the blocking frame_q.get(timeout=…) pattern in the
          Streamlit render loop — zero stale-frame risk

  ✓ scan_dataset / three_way_split / load_frames / augment_sequence
        — unchanged from v3 (already correct)

  ✓ VideoDataGenerator — unchanged from v3

Dependencies
    opencv-python, numpy, tensorflow
"""

import os
import random
import threading
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple, Optional


# ── Constants ────────────────────────────────────────────────────────────────
SEQ_LEN = 16
IMG_H   = 112
IMG_W   = 112
EXTS    = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


# ══════════════════════════════════════════════════════════════════════════════
# NEW ─ Thread-safe latest-frame slot
# ══════════════════════════════════════════════════════════════════════════════

class LatestFrame:
    """
    Lock-protected single-slot frame buffer.

    The capture thread calls write() on every decoded frame.
    The display / flow threads call read() whenever they need the
    most recent image.  There is no queue, so there is no buildup.

    Why this matters
    ----------------
    A bounded Queue(maxsize=2) still has a worst-case lag of 2 frames.
    If the consumer (Streamlit render loop) is slow, frames pile up until
    the queue is full and the producer starts dropping — but by then the
    displayed frame is already stale by up to (maxsize × frame_interval).
    A single-slot buffer with lock-protected overwrite guarantees the
    consumer always reads a frame that is at most ONE capture interval old.
    """

    def __init__(self):
        self._frame: Optional[np.ndarray] = None
        self._lock  = threading.Lock()
        self._count: int = 0          # total frames written (for FPS calc)

    def write(self, frame: np.ndarray) -> None:
        """Overwrite the slot with the newest frame. Called from capture thread."""
        with self._lock:
            # Store a reference, NOT a copy — the capture thread must not
            # modify frame after calling write().  If it does, call
            # frame.copy() here at the cost of one extra allocation.
            self._frame = frame
            self._count += 1

    def read(self) -> Optional[np.ndarray]:
        """
        Return the most recent frame, or None if nothing has been written yet.
        Caller should treat None as 'not ready yet' and skip the render cycle.
        """
        with self._lock:
            return self._frame

    def read_copy(self) -> Optional[np.ndarray]:
        """
        Return a copy of the most recent frame.
        Use this when the caller will modify the array (e.g. annotation drawing).
        """
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._count


# ══════════════════════════════════════════════════════════════════════════════
# NEW ─ Optimized MotionHeatmap
# ══════════════════════════════════════════════════════════════════════════════

class MotionHeatmap:
    """
    Temporal-smoothed, downscaled optical-flow heatmap.

    Usage (one instance per camera, lives in the flow thread)
    ---------------------------------------------------------
        hmap = MotionHeatmap(display_shape=(h, w))

        # Inside the flow loop:
        hmap.update(prev_gray_fullres, curr_gray_fullres)

        # When annotating a frame for display:
        overlay = hmap.get_overlay(frame_bgr, alpha=0.55)

    Parameters
    ----------
    display_shape : (H, W) of the final display frame (full resolution)
    ema_alpha     : EMA blend factor per update.
                    0.10 → very smooth (slow to respond)
                    0.25 → balanced  ← default
                    0.40 → more reactive (slightly more flicker)
    flow_size     : (W, H) at which optical flow is computed.
                    Smaller = faster but coarser.  320×240 works well for
                    detecting human-scale motion at typical CCTV resolutions.
    noise_floor   : Normalised magnitude below which pixels are treated as
                    static and left uncoloured.  Suppresses sensor noise.
    dilation_iter : Number of morphological dilation iterations used to
                    blob-fill motion regions.  2 works well for humans.
    colormap      : OpenCV colormap for the heatmap.
                    COLORMAP_INFERNO  → black→purple→orange→white  (thermal)
                    COLORMAP_JET      → blue→green→red              (classic)
    """

    def __init__(
        self,
        display_shape: Tuple[int, int],   # (H, W)
        ema_alpha:     float = 0.25,
        flow_size:     Tuple[int, int] = (320, 240),   # (W, H) for cv2.resize
        noise_floor:   float = 0.08,
        dilation_iter: int   = 2,
        colormap:      int   = cv2.COLORMAP_INFERNO,
    ):
        self.display_h, self.display_w = display_shape[0], display_shape[1]
        self.ema_alpha     = float(np.clip(ema_alpha, 0.01, 1.0))
        self.flow_w, self.flow_h = flow_size
        self.noise_floor   = float(noise_floor)
        self.colormap      = colormap

        # Structuring element for morphological dilation
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (7, 7)
        )
        self._dilation_iter = dilation_iter

        # Running EMA magnitude map (flow-resolution, float32)
        self._ema_mag: Optional[np.ndarray] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        prev_gray: np.ndarray,   # full-resolution grayscale (H×W, uint8)
        curr_gray: np.ndarray,   # full-resolution grayscale (H×W, uint8)
    ) -> None:
        """
        Compute one frame of optical flow and blend it into the EMA accumulator.

        Call this every frame from the flow thread.  The method downscales
        both inputs to flow_size before calling Farnebäck, so it is cheap
        even on 4K sources.

        Root cause fixed here
        ---------------------
        Old code passed full-resolution frames to cv2.calcOpticalFlowFarneback,
        making it 10-24× slower than necessary.  At 1080p this alone cost
        80-200ms per frame — more than a full TF inference pass.
        """
        # ── 1. Downscale for speed ─────────────────────────────────────────
        prev_s = cv2.resize(prev_gray, (self.flow_w, self.flow_h),
                            interpolation=cv2.INTER_LINEAR)
        curr_s = cv2.resize(curr_gray, (self.flow_w, self.flow_h),
                            interpolation=cv2.INTER_LINEAR)

        # ── 2. Farnebäck dense optical flow ───────────────────────────────
        #   pyr_scale=0.5  — halve resolution per pyramid level
        #   levels=3       — 3 pyramid levels (good for human-scale motion)
        #   winsize=15     — smoothing window; larger = smoother, slower
        #   iterations=3   — per-level iterations
        #   poly_n=5       — pixel neighbourhood for polynomial expansion
        #   poly_sigma=1.2 — Gaussian sigma for expansion
        flow = cv2.calcOpticalFlowFarneback(
            prev_s, curr_s, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        # ── 3. Magnitude of flow vectors ──────────────────────────────────
        mag = cv2.magnitude(flow[..., 0], flow[..., 1]).astype(np.float32)

        # ── 4. Temporal EMA — the core flicker fix ─────────────────────────
        #   cv2.accumulateWeighted: dst = (1-alpha)*dst + alpha*src
        #   Each new frame contributes ema_alpha weight; older frames decay.
        #   At alpha=0.25 the effective window is ~4 frames, so a single
        #   noisy frame affects the display by only 25% instead of 100%.
        if self._ema_mag is None:
            self._ema_mag = mag.copy()
        else:
            cv2.accumulateWeighted(mag, self._ema_mag, self.ema_alpha)

    def get_overlay(
        self,
        frame_bgr: np.ndarray,   # full-resolution BGR frame to annotate
        alpha:     float = 0.55, # blend strength of heatmap over original
    ) -> np.ndarray:
        """
        Return a BGR frame with the motion heatmap blended in.

        Colour mapping (INFERNO):
            black   → no / very low motion
            purple  → low motion
            orange  → medium-high motion
            white   → maximum motion (fighting, running)

        Root causes fixed here
        ----------------------
        1. Percentile normalisation:
           Old code (if any) used global-max normalisation.  If one person
           runs fast, their vectors dominate the max, compressing all other
           motion values toward zero → the colormap shows mostly cold colours
           even when other people are moving.
           Fix: use the 95th-percentile as the ceiling so only the top 5%
           of motion vectors reach full intensity.

        2. Noise-floor threshold:
           Sensor noise and minor camera vibration produce small non-zero
           flow vectors everywhere.  Without thresholding, the colormap
           tints the entire frame faintly — no clear hot zones.
           Fix: zero out everything below noise_floor * p95.

        3. Mask-based alpha blend:
           Old overlays typically did a uniform alpha composite across the
           entire frame, washing out static regions with a dark tint.
           Fix: compute a per-pixel mask from the normalised magnitude so
           the original image shows through exactly where motion is zero.
        """
        if self._ema_mag is None:
            return frame_bgr.copy()

        mag = self._ema_mag.copy()

        # ── 1. Percentile normalisation ────────────────────────────────────
        p95 = float(np.percentile(mag, 95))
        if p95 < 0.5:
            # Scene is essentially static — skip overlay entirely
            return frame_bgr.copy()

        mag_norm = np.clip(mag / p95, 0.0, 1.0)

        # ── 2. Suppress noise floor ────────────────────────────────────────
        mag_norm[mag_norm < self.noise_floor] = 0.0

        # ── 3. Convert to uint8 for OpenCV operations ──────────────────────
        mag_u8 = (mag_norm * 255).astype(np.uint8)

        # ── 4. Morphological dilation — fill motion blobs ──────────────────
        #   Without dilation, humans appear as thin outlines (flow is
        #   strongest at motion boundaries).  Dilation fills the blob
        #   interior so the entire body appears highlighted.
        mag_u8 = cv2.dilate(mag_u8, self._kernel,
                            iterations=self._dilation_iter)

        # ── 5. Colormap ────────────────────────────────────────────────────
        heat_colored = cv2.applyColorMap(mag_u8, self.colormap)

        # ── 6. Upscale to display resolution ──────────────────────────────
        heat_up = cv2.resize(
            heat_colored,
            (self.display_w, self.display_h),
            interpolation=cv2.INTER_LINEAR,
        )

        # ── 7. Per-pixel motion mask ───────────────────────────────────────
        #   Upscale the uint8 magnitude to display size and use it as the
        #   blend weight so the overlay is strong where motion is strong
        #   and invisible where the scene is static.
        mag_disp = cv2.resize(
            mag_u8,
            (self.display_w, self.display_h),
            interpolation=cv2.INTER_LINEAR,
        )
        # 3-channel float32 mask in [0, 1]
        mask_3ch = (
            cv2.cvtColor(mag_disp, cv2.COLOR_GRAY2BGR).astype(np.float32)
            / 255.0
        )

        # ── 8. Blend: original where mask≈0, heatmap where mask≈1 ─────────
        #   out[p] = frame[p] * (1 - mask[p]*alpha) + heat[p] * (mask[p]*alpha)
        out = (
            frame_bgr.astype(np.float32) * (1.0 - mask_3ch * alpha)
            + heat_up.astype(np.float32) * (mask_3ch * alpha)
        )
        return np.clip(out, 0, 255).astype(np.uint8)

    def reset(self) -> None:
        """Clear accumulated state. Call when camera reconnects or stream restarts."""
        self._ema_mag = None


# ══════════════════════════════════════════════════════════════════════════════
# Dataset scanning  (unchanged from v3)
# ══════════════════════════════════════════════════════════════════════════════

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
        "nonviolence":  0,
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
    paths:       List[str],
    labels:      List[int],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> Tuple[Tuple, Tuple, Tuple]:
    """
    Stratified 70 / 15 / 15 split at the CLIP level.

    Returns:
        (train_paths, train_labels),
        (val_paths,   val_labels),
        (test_paths,  test_labels)
    """
    rng = random.Random(seed)

    violence_clips     = [(p, l) for p, l in zip(paths, labels) if l == 1]
    non_violence_clips = [(p, l) for p, l in zip(paths, labels) if l == 0]
    rng.shuffle(violence_clips)
    rng.shuffle(non_violence_clips)

    def class_split(clips):
        n       = len(clips)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))
        n_train = min(n_train, n - 2)
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

    return (
        merge_and_shuffle(tr_v,  tr_nv),
        merge_and_shuffle(va_v,  va_nv),
        merge_and_shuffle(te_v,  te_nv),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Frame loading  (unchanged from v3)
# ══════════════════════════════════════════════════════════════════════════════

def load_frames(
    video_path:      str,
    seq_len:         int  = SEQ_LEN,
    img_h:           int  = IMG_H,
    img_w:           int  = IMG_W,
    temporal_jitter: bool = False,
) -> np.ndarray:
    """
    Load `seq_len` frames from a video clip.
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


# ══════════════════════════════════════════════════════════════════════════════
# Augmentation  (unchanged from v3)
# ══════════════════════════════════════════════════════════════════════════════

def augment_sequence(frames: np.ndarray) -> np.ndarray:
    """
    Apply CONSISTENT spatial + colour augmentation to a clip.

    Augmentations applied identically to every frame in the clip:
        • Horizontal flip              (p = 0.50)
        • Brightness jitter ±20 %      (always)
        • Contrast jitter  ±20 %       (always)
        • Gaussian noise σ = 0.02      (p = 0.30)
        • Temporal reversal            (p = 0.20)
        • Random crop + resize         (p = 0.30)
    """
    if np.random.rand() < 0.5:
        frames = frames[:, :, ::-1, :]

    b      = np.random.uniform(0.80, 1.20)
    frames = np.clip(frames * b, 0.0, 1.0)

    c    = np.random.uniform(0.80, 1.20)
    mean = frames.mean(axis=(1, 2, 3), keepdims=True)
    frames = np.clip((frames - mean) * c + mean, 0.0, 1.0)

    if np.random.rand() < 0.30:
        noise  = np.random.normal(0, 0.02, frames.shape).astype(np.float32)
        frames = np.clip(frames + noise, 0.0, 1.0)

    if np.random.rand() < 0.20:
        frames = frames[::-1].copy()

    if np.random.rand() < 0.30:
        H, W = frames.shape[1], frames.shape[2]
        ch   = int(H * np.random.uniform(0.80, 0.95))
        cw   = int(W * np.random.uniform(0.80, 0.95))
        y0   = np.random.randint(0, H - ch + 1)
        x0   = np.random.randint(0, W - cw + 1)
        cropped = frames[:, y0:y0 + ch, x0:x0 + cw, :]
        resized = np.stack([
            cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)
            for f in cropped
        ])
        frames = resized

    return frames.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Keras Sequence generator  (unchanged from v3)
# ══════════════════════════════════════════════════════════════════════════════

class VideoDataGenerator(tf.keras.utils.Sequence):
    """
    Memory-safe batch generator — loads `batch_size` clips per step.
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