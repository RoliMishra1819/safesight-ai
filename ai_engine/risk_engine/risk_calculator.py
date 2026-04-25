"""
SafeSight AI - Risk Calculator
Aggregates threat signals into a unified risk score and generates incident reports.
"""

import time
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import deque
import numpy as np
import logging

from ai_engine.threat_classifier.classifier import ThreatClassification, ThreatLevel

logger = logging.getLogger(__name__)


@dataclass
class IncidentReport:
    incident_id: str
    timestamp: float
    threat_level: str
    risk_score: float
    confidence: float
    contributing_factors: List[str]
    weapon_detected: bool
    person_count: int
    camera_id: str
    location: Optional[str]
    frame_snapshot_path: Optional[str] = None
    resolved: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RiskSummary:
    current_risk: float
    rolling_avg_risk: float
    peak_risk: float
    threat_level: ThreatLevel
    active_incidents: int
    total_incidents_today: int
    last_updated: float


class RiskCalculator:
    """
    Computes real-time and rolling risk scores, manages incident logging,
    and provides aggregated risk summaries for the dashboard.
    """

    def __init__(
        self,
        camera_id: str = "cam_01",
        location: Optional[str] = None,
        rolling_window: int = 30,          # frames
        incident_cooldown: float = 10.0,   # seconds between same-type incidents
        log_path: str = "logs/incident_logs.json",
        escalation_threshold: float = 0.65,
    ):
        self.camera_id = camera_id
        self.location = location
        self.rolling_window = rolling_window
        self.incident_cooldown = incident_cooldown
        self.log_path = log_path
        self.escalation_threshold = escalation_threshold

        self._risk_history: deque = deque(maxlen=rolling_window)
        self._incidents: List[IncidentReport] = []
        self._last_incident_time: Dict[str, float] = {}
        self._incident_counter = 0
        self._today_count = 0

        self._load_existing_logs()

    def _load_existing_logs(self):
        try:
            with open(self.log_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    today = time.strftime("%Y-%m-%d")
                    self._today_count = sum(
                        1 for inc in data
                        if time.strftime("%Y-%m-%d", time.localtime(inc.get("timestamp", 0))) == today
                    )
                    self._incident_counter = len(data)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def update(
        self,
        classification: ThreatClassification,
        person_count: int = 0,
        frame_path: Optional[str] = None,
    ) -> Optional[IncidentReport]:
        """
        Process a new threat classification and potentially generate an incident.

        Args:
            classification: ThreatClassification from ThreatClassifier
            person_count: Number of persons in frame
            frame_path: Optional path to saved frame snapshot

        Returns:
            IncidentReport if a new incident was created, else None
        """
        self._risk_history.append(classification.raw_score)
        new_incident = None

        if (
            classification.raw_score >= self.escalation_threshold
            and classification.threat_level not in (ThreatLevel.NONE, ThreatLevel.LOW)
        ):
            factor_key = "_".join(sorted(classification.contributing_factors)) or "unknown"
            now = classification.timestamp
            last = self._last_incident_time.get(factor_key, 0.0)

            if now - last >= self.incident_cooldown:
                self._incident_counter += 1
                self._today_count += 1
                self._last_incident_time[factor_key] = now

                report = IncidentReport(
                    incident_id=f"INC-{self._incident_counter:05d}",
                    timestamp=now,
                    threat_level=classification.threat_level.value,
                    risk_score=classification.raw_score,
                    confidence=classification.confidence,
                    contributing_factors=classification.contributing_factors,
                    weapon_detected=classification.weapon_detected,
                    person_count=person_count,
                    camera_id=self.camera_id,
                    location=self.location,
                    frame_snapshot_path=frame_path,
                )
                self._incidents.append(report)
                new_incident = report
                self._save_incident(report)
                logger.warning(
                    f"INCIDENT {report.incident_id}: {report.threat_level} | "
                    f"score={report.risk_score:.3f} | factors={report.contributing_factors}"
                )

        return new_incident

    def get_summary(self) -> RiskSummary:
        history = list(self._risk_history)
        current = history[-1] if history else 0.0
        rolling_avg = float(np.mean(history)) if history else 0.0
        peak = float(np.max(history)) if history else 0.0

        # Derive level from rolling average
        from ai_engine.threat_classifier.classifier import THREAT_THRESHOLDS
        level = ThreatLevel.NONE
        for lvl, (lo, hi) in THREAT_THRESHOLDS.items():
            if lo <= rolling_avg < hi:
                level = lvl
                break

        active = sum(1 for i in self._incidents if not i.resolved)

        return RiskSummary(
            current_risk=round(current, 4),
            rolling_avg_risk=round(rolling_avg, 4),
            peak_risk=round(peak, 4),
            threat_level=level,
            active_incidents=active,
            total_incidents_today=self._today_count,
            last_updated=time.time(),
        )

    def resolve_incident(self, incident_id: str, notes: str = "") -> bool:
        for inc in self._incidents:
            if inc.incident_id == incident_id:
                inc.resolved = True
                inc.notes = notes
                return True
        return False

    def get_recent_incidents(self, n: int = 10) -> List[IncidentReport]:
        return list(reversed(self._incidents[-n:]))

    def _save_incident(self, report: IncidentReport):
        try:
            try:
                with open(self.log_path, "r") as f:
                    existing = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing = []
            existing.append(report.to_dict())
            with open(self.log_path, "w") as f:
                json.dump(existing, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save incident log: {e}")

    def get_risk_trend(self) -> List[float]:
        return list(self._risk_history)

    def reset_daily_count(self):
        self._today_count = 0
