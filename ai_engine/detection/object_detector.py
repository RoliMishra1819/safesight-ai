"""
SafeSight AI - Object Detection Module
Uses YOLOv8 for real-time weapon and person detection.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None


CLASS_NAMES = {
    0: "person",
    1: "weapon_gun",
    2: "weapon_knife",
    3: "weapon_bat",
    4: "suspicious_bag",
}

THREAT_CLASSES = {"weapon_gun", "weapon_knife", "weapon_bat"}


class ObjectDetector:
    """
    YOLOv8-based object detector for SafeSight AI.
    Detects persons, weapons, and suspicious objects.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8_model.pt",
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load YOLOv8 model from disk."""
        try:
            from ultralytics import YOLO
            import torch.serialization
            torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            logger.info(f"Model loaded from {self.model_path} on {self.device}")
        except FileNotFoundError:
            logger.warning(f"Model file not found at {self.model_path}. Using pretrained YOLOv8n.")
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
            self.model.to(self.device)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run object detection on a single frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of Detection objects
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = CLASS_NAMES.get(class_id, result.names.get(class_id, "unknown"))

                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                    )
                )

        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Run detection on a batch of frames."""
        return [self.detect(frame) for frame in frames]

    def filter_threat_detections(self, detections: List[Detection]) -> List[Detection]:
        """Return only detections that are threat-related."""
        return [d for d in detections if d.class_name in THREAT_CLASSES]

    def draw_detections(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.

        Args:
            frame: BGR image
            detections: List of Detection objects

        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            is_threat = det.class_name in THREAT_CLASSES
            color = (0, 0, 255) if is_threat else (0, 255, 0)
            label = f"{det.class_name} {det.confidence:.2f}"
            if det.track_id is not None:
                label = f"ID:{det.track_id} {label}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        return annotated

    def get_weapon_count(self, detections: List[Detection]) -> int:
        return sum(1 for d in detections if d.class_name in THREAT_CLASSES)

    def get_person_count(self, detections: List[Detection]) -> int:
        return sum(1 for d in detections if d.class_name == "person")
