"""
SafeSight AI - Crowd Detection
Detects crowd formation, density anomalies, and sudden gatherings.
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CrowdAlert:
    zone: Optional[str]
    person_count: int
    density: float          # persons per 1000 sq pixels
    center: Tuple[float, float]
    timestamp: float
    risk_score: float
    alert_type: str         # "overcrowding" | "sudden_gathering" | "dense_cluster"


class CrowdDetector:
    """
    Detects crowd-related threats including overcrowding and sudden gatherings
    using density estimation and temporal analysis.
    """

    def __init__(
        self,
        crowd_threshold: int = 5,
        density_threshold: float = 0.005,   # persons per pixel
        sudden_gather_delta: int = 3,        # person increase in window
        time_window: float = 5.0,            # seconds for sudden gathering check
        zones: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
        frame_area: int = 640 * 480,
    ):
        self.crowd_threshold = crowd_threshold
        self.density_threshold = density_threshold
        self.sudden_gather_delta = sudden_gather_delta
        self.time_window = time_window
        self.zones = zones or {}
        self.frame_area = frame_area

        self._history: List[Tuple[float, int]] = []  # (timestamp, count)
        self._alerts: List[CrowdAlert] = []

    def update(self, tracks, timestamp: Optional[float] = None) -> List[CrowdAlert]:
        """
        Analyze current tracks for crowd conditions.

        Args:
            tracks: List of Track objects
            timestamp: Current time

        Returns:
            List of new CrowdAlert objects
        """
        now = timestamp or time.time()
        persons = [t for t in tracks if t.class_name == "person"]
        count = len(persons)

        self._history.append((now, count))
        # Keep only recent history
        self._history = [(t, c) for t, c in self._history if now - t <= self.time_window * 3]

        new_alerts = []

        # --- Per-zone analysis ---
        zone_persons = self._group_by_zone(persons)

        for zone_name, zone_persons_list in zone_persons.items():
            z_count = len(zone_persons_list)
            zone_area = self._zone_area(zone_name)
            density = z_count / zone_area if zone_area > 0 else 0.0
            center = self._cluster_center(zone_persons_list)

            if z_count >= self.crowd_threshold:
                risk = self._compute_risk(z_count, density, "overcrowding")
                alert = CrowdAlert(
                    zone=zone_name,
                    person_count=z_count,
                    density=density,
                    center=center,
                    timestamp=now,
                    risk_score=risk,
                    alert_type="overcrowding",
                )
                new_alerts.append(alert)
                logger.warning(f"Crowd detected in zone '{zone_name}': {z_count} persons")

        # --- Global frame analysis ---
        global_density = count / self.frame_area
        if global_density > self.density_threshold:
            risk = self._compute_risk(count, global_density, "dense_cluster")
            centers = self._cluster_center(persons)
            alert = CrowdAlert(
                zone=None,
                person_count=count,
                density=global_density,
                center=centers,
                timestamp=now,
                risk_score=risk,
                alert_type="dense_cluster",
            )
            new_alerts.append(alert)

        # --- Sudden gathering detection ---
        window_start = now - self.time_window
        recent = [c for t, c in self._history if t >= window_start]
        if len(recent) >= 2:
            delta = recent[-1] - recent[0]
            if delta >= self.sudden_gather_delta:
                risk = self._compute_risk(count, global_density, "sudden_gathering")
                centers = self._cluster_center(persons)
                alert = CrowdAlert(
                    zone=None,
                    person_count=count,
                    density=global_density,
                    center=centers,
                    timestamp=now,
                    risk_score=risk,
                    alert_type="sudden_gathering",
                )
                new_alerts.append(alert)
                logger.warning(f"Sudden gathering detected: +{delta} persons in {self.time_window}s")

        self._alerts.extend(new_alerts)
        return new_alerts

    def _group_by_zone(self, persons) -> Dict[str, list]:
        result: Dict[str, list] = {}
        for zone_name, (x1, y1, x2, y2) in self.zones.items():
            in_zone = [
                p for p in persons
                if x1 <= p.center()[0] <= x2 and y1 <= p.center()[1] <= y2
            ]
            if in_zone:
                result[zone_name] = in_zone
        return result

    def _zone_area(self, zone_name: str) -> float:
        if zone_name in self.zones:
            x1, y1, x2, y2 = self.zones[zone_name]
            return max(1.0, float((x2 - x1) * (y2 - y1)))
        return float(self.frame_area)

    def _cluster_center(self, persons) -> Tuple[float, float]:
        if not persons:
            return (0.0, 0.0)
        xs = [p.center()[0] for p in persons]
        ys = [p.center()[1] for p in persons]
        return (float(np.mean(xs)), float(np.mean(ys)))

    def _compute_risk(self, count: int, density: float, alert_type: str) -> float:
        base = min(count / 20.0, 1.0)
        density_factor = min(density / self.density_threshold, 2.0) * 0.3
        type_multiplier = {"overcrowding": 0.8, "dense_cluster": 0.9, "sudden_gathering": 1.0}.get(alert_type, 0.8)
        return round(min((base + density_factor) * type_multiplier, 1.0), 3)

    def get_alerts(self) -> List[CrowdAlert]:
        return list(self._alerts)

    def get_current_count(self) -> int:
        return self._history[-1][1] if self._history else 0
