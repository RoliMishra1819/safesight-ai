"""
SafeSight AI - Running/Panic Detection
Detects rapid movement patterns that may indicate panic, chase, or fleeing behavior.
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class RunAlert:
    track_id: int
    speed: float              # pixels per second
    direction: Tuple[float, float]
    position: Tuple[float, float]
    timestamp: float
    risk_score: float
    alert_type: str           # "running" | "panic_dispersion" | "chase"


class RunningDetector:
    """
    Detects abnormal movement speeds indicating running, panic, or chase scenarios.
    Uses per-track velocity history from the ByteTracker.
    """

    def __init__(
        self,
        fps: float = 30.0,
        run_speed_threshold: float = 80.0,        # pixels/second
        panic_dispersion_threshold: float = 3,    # number of simultaneous runners
        chase_alignment_threshold: float = 0.85,  # cosine similarity for chase
        history_window: int = 10,                 # frames to average speed
    ):
        self.fps = fps
        self.run_speed_threshold = run_speed_threshold
        self.panic_threshold = panic_dispersion_threshold
        self.chase_threshold = chase_alignment_threshold
        self.history_window = history_window

        self._speed_history: Dict[int, List[float]] = {}
        self._velocity_history: Dict[int, List[Tuple[float, float]]] = {}
        self._alerts: List[RunAlert] = []

    def update(self, tracks, timestamp: Optional[float] = None) -> List[RunAlert]:
        """
        Analyze track velocities for running/panic detection.

        Args:
            tracks: List of Track objects (with .velocity set by ByteTracker)
            timestamp: Current timestamp

        Returns:
            List of new RunAlert objects
        """
        now = timestamp or time.time()
        new_alerts = []
        runners = []

        for track in tracks:
            if track.class_name != "person":
                continue

            tid = track.track_id
            vx, vy = track.velocity
            speed = float(np.sqrt(vx**2 + vy**2) * self.fps)

            # Update speed history
            if tid not in self._speed_history:
                self._speed_history[tid] = []
                self._velocity_history[tid] = []
            self._speed_history[tid].append(speed)
            self._velocity_history[tid].append((vx, vy))

            if len(self._speed_history[tid]) > self.history_window:
                self._speed_history[tid].pop(0)
                self._velocity_history[tid].pop(0)

            avg_speed = float(np.mean(self._speed_history[tid]))

            if avg_speed > self.run_speed_threshold:
                runners.append(track)
                unit_vel = self._unit_vector((vx, vy))
                risk = self._compute_risk(avg_speed, "running")
                alert = RunAlert(
                    track_id=tid,
                    speed=avg_speed,
                    direction=unit_vel,
                    position=track.center(),
                    timestamp=now,
                    risk_score=risk,
                    alert_type="running",
                )
                new_alerts.append(alert)

        # Panic dispersion: multiple people running in different directions
        if len(runners) >= self.panic_threshold:
            directions = []
            for r in runners:
                if r.track_id in self._velocity_history and self._velocity_history[r.track_id]:
                    v = self._velocity_history[r.track_id][-1]
                    directions.append(self._unit_vector(v))

            if len(directions) >= 2:
                avg_alignment = self._average_pairwise_similarity(directions)
                if avg_alignment < (1 - self.chase_threshold):  # low alignment = panic/scatter
                    risk = self._compute_risk(len(runners), "panic_dispersion")
                    center = self._group_center(runners)
                    alert = RunAlert(
                        track_id=-1,
                        speed=float(np.mean([self._avg_speed(r.track_id) for r in runners])),
                        direction=(0.0, 0.0),
                        position=center,
                        timestamp=now,
                        risk_score=risk,
                        alert_type="panic_dispersion",
                    )
                    new_alerts.append(alert)
                    logger.warning(f"Panic dispersion detected: {len(runners)} runners")

                elif avg_alignment >= self.chase_threshold and len(runners) >= 2:
                    risk = self._compute_risk(len(runners), "chase")
                    center = self._group_center(runners)
                    alert = RunAlert(
                        track_id=-1,
                        speed=float(np.mean([self._avg_speed(r.track_id) for r in runners])),
                        direction=directions[0] if directions else (0.0, 0.0),
                        position=center,
                        timestamp=now,
                        risk_score=risk,
                        alert_type="chase",
                    )
                    new_alerts.append(alert)
                    logger.warning(f"Chase pattern detected: {len(runners)} aligned runners")

        self._alerts.extend(new_alerts)
        return new_alerts

    def _unit_vector(self, v: Tuple[float, float]) -> Tuple[float, float]:
        mag = np.sqrt(v[0]**2 + v[1]**2)
        if mag < 1e-6:
            return (0.0, 0.0)
        return (v[0] / mag, v[1] / mag)

    def _average_pairwise_similarity(self, directions: List[Tuple[float, float]]) -> float:
        sims = []
        for i in range(len(directions)):
            for j in range(i + 1, len(directions)):
                sim = directions[i][0] * directions[j][0] + directions[i][1] * directions[j][1]
                sims.append(sim)
        return float(np.mean(sims)) if sims else 0.0

    def _group_center(self, tracks) -> Tuple[float, float]:
        xs = [t.center()[0] for t in tracks]
        ys = [t.center()[1] for t in tracks]
        return (float(np.mean(xs)), float(np.mean(ys)))

    def _avg_speed(self, track_id: int) -> float:
        history = self._speed_history.get(track_id, [0.0])
        return float(np.mean(history))

    def _compute_risk(self, value: float, alert_type: str) -> float:
        if alert_type == "running":
            base = min(value / (self.run_speed_threshold * 3), 1.0)
        elif alert_type == "panic_dispersion":
            base = min(value / 10.0, 1.0) * 1.2
        else:  # chase
            base = min(value / 5.0, 1.0) * 1.1
        return round(min(base, 1.0), 3)

    def get_alerts(self) -> List[RunAlert]:
        return list(self._alerts)
