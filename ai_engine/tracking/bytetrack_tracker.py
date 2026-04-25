"""
SafeSight AI - ByteTrack Multi-Object Tracker
Tracks persons and objects across video frames using ByteTrack algorithm.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    class_name: str
    confidence: float
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    history: List[Tuple[int, int, int, int]] = field(default_factory=list)
    velocity: Tuple[float, float] = (0.0, 0.0)

    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


def compute_iou(bbox1: Tuple, bbox2: Tuple) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def hungarian_match(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """Simple greedy matching (approximation of Hungarian algorithm)."""
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return list(zip(row_ind.tolist(), col_ind.tolist()))


class ByteTracker:
    """
    ByteTrack-inspired multi-object tracker for SafeSight AI.
    Associates detections with existing tracks using IoU and confidence.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        high_conf_threshold: float = 0.6,
        low_conf_threshold: float = 0.1,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold

        self._tracks: List[Track] = []
        self._next_id = 1
        self.frame_count = 0

    def update(self, detections) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: List of Detection objects from ObjectDetector

        Returns:
            List of active Track objects with track_ids assigned
        """
        self.frame_count += 1

        # Separate high and low confidence detections
        high_dets = [d for d in detections if d.confidence >= self.high_conf_threshold]
        low_dets = [d for d in detections if self.low_conf_threshold <= d.confidence < self.high_conf_threshold]

        # Predict new positions for existing tracks
        for track in self._tracks:
            track.time_since_update += 1

        active_tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

        # --- First association: high confidence detections ---
        matched, unmatched_dets, unmatched_tracks = self._associate(
            high_dets, active_tracks
        )

        for track_idx, det_idx in matched:
            track = active_tracks[track_idx]
            det = high_dets[det_idx]
            track.history.append(track.bbox)
            if len(track.history) > 30:
                track.history.pop(0)
            cx_old, cy_old = track.center()
            track.bbox = det.bbox
            track.confidence = det.confidence
            cx_new, cy_new = track.center()
            track.velocity = (cx_new - cx_old, cy_new - cy_old)
            track.hits += 1
            track.time_since_update = 0
            track.age += 1

        remaining_tracks = [active_tracks[i] for i in unmatched_tracks]

        # --- Second association: low confidence detections with remaining tracks ---
        matched2, _, _ = self._associate(low_dets, remaining_tracks)
        for track_idx, det_idx in matched2:
            track = remaining_tracks[track_idx]
            det = low_dets[det_idx]
            track.bbox = det.bbox
            track.confidence = det.confidence
            track.hits += 1
            track.time_since_update = 0
            track.age += 1

        # Create new tracks for unmatched high-confidence detections
        for det_idx in unmatched_dets:
            det = high_dets[det_idx]
            new_track = Track(
                track_id=self._next_id,
                bbox=det.bbox,
                class_name=det.class_name,
                confidence=det.confidence,
            )
            self._next_id += 1
            self._tracks.append(new_track)

        # Remove dead tracks
        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

        # Assign track IDs back to detections
        confirmed = [
            t for t in self._tracks
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits
        ]

        for det in detections:
            for track in confirmed:
                if compute_iou(det.bbox, track.bbox) > 0.5:
                    det.track_id = track.track_id
                    break

        return confirmed

    def _associate(
        self,
        detections,
        tracks: List[Track],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not detections or not tracks:
            return [], list(range(len(detections))), list(range(len(tracks)))

        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for t_idx, track in enumerate(tracks):
            for d_idx, det in enumerate(detections):
                iou = compute_iou(track.bbox, det.bbox)
                cost_matrix[t_idx, d_idx] = 1 - iou

        try:
            matches_raw = hungarian_match(cost_matrix)
        except Exception:
            return [], list(range(len(detections))), list(range(len(tracks)))

        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))

        for t_idx, d_idx in matches_raw:
            if cost_matrix[t_idx, d_idx] <= 1 - self.iou_threshold:
                matched.append((t_idx, d_idx))
                if d_idx in unmatched_dets:
                    unmatched_dets.remove(d_idx)
                if t_idx in unmatched_tracks:
                    unmatched_tracks.remove(t_idx)

        return matched, unmatched_dets, unmatched_tracks

    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        for track in self._tracks:
            if track.track_id == track_id:
                return track
        return None

    def reset(self):
        self._tracks = []
        self._next_id = 1
        self.frame_count = 0
