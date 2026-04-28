"""
GuardNet – YOLOv8 Person Detector
"""
import os
import sys
import numpy as np
import cv2
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import YOLO_WEIGHTS, YOLO_CONFIDENCE, YOLO_IOU, ENABLE_YOLO

BBox = Tuple[int, int, int, int]


class PersonDetector:
    PERSON_CLASS = 0

    def __init__(self):
        self.enabled = ENABLE_YOLO
        self.model   = None

        if not self.enabled:
            return

        try:
            from ultralytics import YOLO
            if os.path.exists(YOLO_WEIGHTS):
                self.model = YOLO(YOLO_WEIGHTS)
            else:
                print("[INFO] Downloading YOLOv8 nano weights...")
                self.model = YOLO("yolov8n.pt")
            print("[INFO] YOLOv8 person detector ready.")
        except ImportError:
            print("[WARN] ultralytics not installed – person detection disabled.")
            print("[WARN] Run: pip install ultralytics")
            self.enabled = False
        except Exception as e:
            print(f"[WARN] YOLO init failed: {e} – person detection disabled.")
            self.enabled = False

    def detect(self, frame: np.ndarray) -> List[BBox]:
        """
        Run YOLOv8 on frame.
        Returns list of (x1, y1, x2, y2) boxes for each person found.
        """
        if not self.enabled or self.model is None:
            return []

        try:
            results = self.model(
                frame,
                conf=YOLO_CONFIDENCE,
                iou=YOLO_IOU,
                classes=[self.PERSON_CLASS],
                verbose=False,
            )
            boxes = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    boxes.append((x1, y1, x2, y2))
            return boxes
        except Exception as e:
            print(f"[WARN] Detection error: {e}")
            return []

    def draw(self, frame: np.ndarray, boxes: List[BBox],
             violence: bool = False) -> np.ndarray:
        """
        Draw bounding boxes on frame.
        Orange = normal, Red = violent
        """
        colour = (0, 30, 220) if violence else (255, 165, 0)

        for (x1, y1, x2, y2) in boxes:
            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(
                "Person", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame,
                (x1, y1 - th - 6),
                (x1 + tw + 4, y1),
                colour, -1
            )

            # Label text
            cv2.putText(
                frame, "Person",
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA
            )

        return frame