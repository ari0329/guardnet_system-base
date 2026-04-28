"""
GuardNet – Clip Extractor
Saves only the violence-detected segment (pre + post buffer) as a short .avi clip.
"""
import cv2
import os
import time
import threading
import numpy as np
from collections import deque
from datetime import datetime
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CLIPS_DIR, PRE_BUFFER_SECONDS, POST_BUFFER_SECONDS


class ClipExtractor:
    """
    Maintains a rolling pre-event frame buffer.
    When violence is detected, begins writing frames.
    After POST_BUFFER_SECONDS, finalises and saves the clip.
    """

    def __init__(self, fps: float = 15.0, camera_id: str = "cam0"):
        self.fps         = max(fps, 1.0)
        self.camera_id   = camera_id
        self.pre_buf_len = int(self.fps * PRE_BUFFER_SECONDS)
        self.post_buf_len= int(self.fps * POST_BUFFER_SECONDS)
        self.pre_buffer  = deque(maxlen=self.pre_buf_len)
        self._recording  = False
        self._post_count = 0
        self._writer     = None
        self._clip_path  = None
        self._lock       = threading.Lock()

    def push(self, frame: np.ndarray) -> str | None:
        """
        Push one annotated BGR frame.
        Returns clip path when a clip is finalised, else None.
        """
        with self._lock:
            if not self._recording:
                self.pre_buffer.append(frame.copy())
                return None
            else:
                if self._writer:
                    self._writer.write(frame)
                self._post_count += 1
                if self._post_count >= self.post_buf_len:
                    return self._finalise()
                return None

    def start_recording(self, frame: np.ndarray):
        """Call when violence first detected."""
        with self._lock:
            if self._recording:
                return
            self._recording  = True
            self._post_count = 0
            h, w = frame.shape[:2]
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._clip_path = os.path.join(
                CLIPS_DIR, f"{self.camera_id}_{ts}.avi"
            )
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self._writer = cv2.VideoWriter(
                self._clip_path, fourcc, self.fps, (w, h)
            )
            # Write pre-buffer frames first
            for f in self.pre_buffer:
                self._writer.write(f)
            self.pre_buffer.clear()

    def _finalise(self) -> str:
        path = self._clip_path
        if self._writer:
            self._writer.release()
            self._writer = None
        self._recording  = False
        self._post_count = 0
        self._clip_path  = None
        return path

    def is_recording(self) -> bool:
        return self._recording
