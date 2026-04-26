"""
GuardNet – Person Detection (YOLOv8)
Wraps Ultralytics YOLOv8 for optional multi-person bounding-box overlay.
Falls back gracefully if ultralytics is not installed.
"""

import os
import sys
import numpy as np
import cv2
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    YOLO_WEIGHTS, YOLO_CONFIDENCE, YOLO_IOU, COLOR_BOX, ENABLE_YOLO
)

# Type alias
BBox = Tuple[int, int, int, int]   # x1, y1, x2, y2


class PersonDetector:
    """
    YOLOv8-based person detector.
    Auto-downloads the nano model on first run if the weights file is absent.
    """

    PERSON_CLASS_ID = 0   # COCO class 0 = person

    def __init__(self, weights: str = YOLO_WEIGHTS):
        self.enabled = ENABLE_YOLO
        self.model   = None

        if not self.enabled:
            return

        try:
            from ultralytics import YOLO
            if not os.path.exists(weights):
                print("[INFO] YOLOv8 weights not found – auto-downloading yolov8n …")
                self.model = YOLO("yolov8n.pt")
                os.makedirs(os.path.dirname(weights), exist_ok=True)
                self.model.save(weights)
            else:
                self.model = YOLO(weights)
            print("[INFO] YOLOv8 person detector ready.")
        except ImportError:
            print("[WARN] ultralytics not installed – person detection disabled.")
            self.enabled = False
        except Exception as e:
            print(f"[WARN] YOLO init failed ({e}) – person detection disabled.")
            self.enabled = False

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[BBox]:
        """
        Run inference on a BGR frame.
        Returns list of (x1, y1, x2, y2) boxes for every detected person.
        """
        if not self.enabled or self.model is None:
            return []

        results = self.model(
            frame,
            conf=YOLO_CONFIDENCE,
            iou=YOLO_IOU,
            classes=[self.PERSON_CLASS_ID],
            verbose=False,
        )
        boxes: List[BBox] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append((x1, y1, x2, y2))
        return boxes

    def draw(
        self,
        frame: np.ndarray,
        boxes: List[BBox],
        violence: bool = False,
    ) -> np.ndarray:
        """Annotate frame with bounding boxes. Returns annotated copy."""
        colour = (0, 30, 220) if violence else COLOR_BOX
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            label = "Person"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
            cv2.putText(
                frame, label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA
            )
        return frame
