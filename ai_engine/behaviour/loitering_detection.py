"""
SafeSight AI - Loitering Detection
Detects persons who remain in a zone for an extended period (potential threat indicator).
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class LoiterRecord:
    track_id: int
    first_seen: float
    last_seen: float
    zone: Optional[str]
    positions: List[Tuple[float, float]] = field(default_factory=list)
    alert_triggered: bool = False

    @property
    def duration(self) -> float:
        return self.last_seen - self.first_seen

    def average_position(self) -> Tuple[float, float]:
        if not self.positions:
            return (0.0, 0.0)
        xs = [p[0] for p in self.positions]
        ys = [p[1] for p in self.positions]
        return (float(np.mean(xs)), float(np.mean(ys)))

    def displacement(self) -> float:
        """How far has this person moved overall (low = loitering)."""
        if len(self.positions) < 2:
            return 0.0
        first = np.array(self.positions[0])
        last = np.array(self.positions[-1])
        return float(np.linalg.norm(last - first))


@dataclass
class LoiterAlert:
    track_id: int
    duration: float
    zone: Optional[str]
    average_position: Tuple[float, float]
    timestamp: float
    risk_score: float


class LoiteringDetector:
    """
    Detects loitering behavior by tracking how long persons remain
    in a given area without significant movement.
    """

    def __init__(
        self,
        loiter_threshold_seconds: float = 15.0,
        movement_threshold_pixels: float = 50.0,
        zones: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
        cleanup_interval: float = 60.0,
    ):
        """
        Args:
            loiter_threshold_seconds: Duration before flagging as loitering
            movement_threshold_pixels: Max movement to still be considered loitering
            zones: Named zones as {name: (x1, y1, x2, y2)} for zone-based alerts
            cleanup_interval: How often to purge stale records (seconds)
        """
        self.loiter_threshold = loiter_threshold_seconds
        self.movement_threshold = movement_threshold_pixels
        self.zones = zones or {}
        self.cleanup_interval = cleanup_interval

        self._records: Dict[int, LoiterRecord] = {}
        self._alerts: List[LoiterAlert] = []
        self._last_cleanup = time.time()

    def update(self, tracks, timestamp: Optional[float] = None) -> List[LoiterAlert]:
        """
        Update loitering records for current tracks.

        Args:
            tracks: List of Track objects from ByteTracker
            timestamp: Current timestamp (defaults to time.time())

        Returns:
            List of new LoiterAlert objects triggered this update
        """
        now = timestamp or time.time()
        new_alerts = []
        active_ids = set()

        for track in tracks:
            if track.class_name != "person":
                continue

            track_id = track.track_id
            active_ids.add(track_id)
            cx, cy = track.center()
            zone = self._get_zone(cx, cy)

            if track_id not in self._records:
                self._records[track_id] = LoiterRecord(
                    track_id=track_id,
                    first_seen=now,
                    last_seen=now,
                    zone=zone,
                    positions=[(cx, cy)],
                )
            else:
                record = self._records[track_id]
                record.last_seen = now
                record.positions.append((cx, cy))
                if len(record.positions) > 300:
                    record.positions = record.positions[-150:]

                # Check if truly loitering (long duration + low movement)
                if (
                    record.duration >= self.loiter_threshold
                    and record.displacement() <= self.movement_threshold
                    and not record.alert_triggered
                ):
                    risk_score = self._compute_risk(record)
                    alert = LoiterAlert(
                        track_id=track_id,
                        duration=record.duration,
                        zone=record.zone,
                        average_position=record.average_position(),
                        timestamp=now,
                        risk_score=risk_score,
                    )
                    record.alert_triggered = True
                    self._alerts.append(alert)
                    new_alerts.append(alert)
                    logger.warning(
                        f"Loitering alert: track_id={track_id}, "
                        f"duration={record.duration:.1f}s, zone={zone}"
                    )

        # Cleanup stale records
        if now - self._last_cleanup > self.cleanup_interval:
            stale = [tid for tid in self._records if tid not in active_ids]
            for tid in stale:
                del self._records[tid]
            self._last_cleanup = now

        return new_alerts

    def _get_zone(self, cx: float, cy: float) -> Optional[str]:
        for zone_name, (x1, y1, x2, y2) in self.zones.items():
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return zone_name
        return None

    def _compute_risk(self, record: LoiterRecord) -> float:
        """
        Compute a 0-1 risk score for loitering.
        Longer duration + restricted zone = higher risk.
        """
        duration_score = min(record.duration / 120.0, 1.0)  # cap at 2 mins
        zone_multiplier = 1.5 if record.zone and "restricted" in record.zone.lower() else 1.0
        raw = duration_score * zone_multiplier
        return round(min(raw, 1.0), 3)

    def get_all_loiterers(self) -> List[LoiterRecord]:
        return [
            r for r in self._records.values()
            if r.duration >= self.loiter_threshold and r.displacement() <= self.movement_threshold
        ]

    def get_alerts(self) -> List[LoiterAlert]:
        return list(self._alerts)

    def clear_alerts(self):
        self._alerts.clear()

    def reset(self):
        self._records.clear()
        self._alerts.clear()
