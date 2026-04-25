"""
SafeSight AI - Frame Utilities
Image processing helpers for annotation, saving, and preprocessing.
"""

import cv2
import numpy as np
import os
import time
from typing import Tuple, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

SNAPSHOT_DIR = Path("snapshots")
SNAPSHOT_DIR.mkdir(exist_ok=True)


def resize_frame(
    frame: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    inter: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize frame while maintaining aspect ratio."""
    h, w = frame.shape[:2]
    if width is None and height is None:
        return frame
    if width is None:
        ratio = height / h
        new_w, new_h = int(w * ratio), height
    elif height is None:
        ratio = width / w
        new_w, new_h = width, int(h * ratio)
    else:
        new_w, new_h = width, height
    return cv2.resize(frame, (new_w, new_h), interpolation=inter)


def draw_risk_overlay(
    frame: np.ndarray,
    risk_score: float,
    threat_level: str,
    person_count: int,
    fps: float = 0.0,
    camera_id: str = "",
) -> np.ndarray:
    """
    Draw a HUD-style risk overlay on the frame.

    Args:
        frame: BGR frame
        risk_score: 0.0 to 1.0 risk score
        threat_level: "none" | "low" | "medium" | "high" | "critical"
        person_count: Number of persons detected
        fps: Current FPS for display
        camera_id: Camera identifier

    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Color by threat level
    level_colors = {
        "none": (0, 200, 0),
        "low": (0, 220, 80),
        "medium": (0, 165, 255),
        "high": (0, 60, 255),
        "critical": (0, 0, 220),
    }
    color = level_colors.get(threat_level.lower(), (200, 200, 200))

    # Semi-transparent header bar
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

    # Risk bar
    bar_width = int((w - 20) * risk_score)
    cv2.rectangle(annotated, (10, 40), (w - 10, 48), (60, 60, 60), -1)
    if bar_width > 0:
        cv2.rectangle(annotated, (10, 40), (10 + bar_width, 48), color, -1)

    # Text HUD
    timestamp_str = time.strftime("%H:%M:%S")
    lines = [
        f"SAFESIGHT AI  {camera_id}  {timestamp_str}",
        f"THREAT: {threat_level.upper()}  RISK: {risk_score:.2%}  PERSONS: {person_count}  FPS: {fps:.1f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            annotated, line,
            (10, 16 + i * 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            color, 1, cv2.LINE_AA,
        )

    # Critical flashing border
    if threat_level.lower() == "critical":
        thickness = 4 if int(time.time() * 2) % 2 == 0 else 2
        cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), (0, 0, 255), thickness)

    return annotated


def save_snapshot(
    frame: np.ndarray,
    camera_id: str,
    incident_id: str,
    quality: int = 92,
) -> str:
    """
    Save a frame snapshot to disk.

    Returns:
        Path to saved snapshot
    """
    cam_dir = SNAPSHOT_DIR / camera_id
    cam_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{incident_id}_{int(time.time())}.jpg"
    filepath = str(cam_dir / filename)
    cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    logger.info(f"Snapshot saved: {filepath}")
    return filepath


def preprocess_for_detection(
    frame: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
) -> np.ndarray:
    """
    Preprocess frame for YOLO inference.
    Resizes and normalizes to [0, 1] as float32 NCHW tensor.
    """
    resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(tensor, 0)               # CHW -> NCHW


def is_nighttime(frame: np.ndarray, brightness_threshold: int = 60) -> bool:
    """Estimate if a frame was captured at night using brightness."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    return mean_brightness < brightness_threshold


def crop_bbox(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: int = 10,
) -> np.ndarray:
    """Crop a bounding box region from a frame with optional padding."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return frame[y1:y2, x1:x2]


def draw_zones(
    frame: np.ndarray,
    zones: dict,
    alpha: float = 0.15,
) -> np.ndarray:
    """Visualize named zones as colored overlays."""
    overlay = frame.copy()
    colors = [(255, 100, 0), (0, 100, 255), (100, 255, 0), (255, 0, 200)]
    for i, (name, (x1, y1, x2, y2)) in enumerate(zones.items()):
        color = colors[i % len(colors)]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.putText(overlay, name, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def encode_frame_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """Encode frame as JPEG bytes (for HTTP streaming)."""
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes()
