"""
SafeSight AI - Pydantic Schemas
Request/Response models for the FastAPI backend.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class ThreatLevelEnum(str, Enum):
    none = "none"
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


# ─── Incident Schemas ─────────────────────────────────────────────────────────

class IncidentOut(BaseModel):
    incident_id: str
    timestamp: float
    threat_level: ThreatLevelEnum
    risk_score: float
    confidence: float
    contributing_factors: List[str]
    weapon_detected: bool
    person_count: int
    camera_id: str
    location: Optional[str] = None
    frame_snapshot_path: Optional[str] = None
    resolved: bool = False
    notes: str = ""

    class Config:
        from_attributes = True


class ResolveIncidentIn(BaseModel):
    notes: str = Field(default="", max_length=1000)


# ─── Risk Schemas ─────────────────────────────────────────────────────────────

class RiskSummaryOut(BaseModel):
    current_risk: float = Field(..., ge=0.0, le=1.0)
    rolling_avg_risk: float = Field(..., ge=0.0, le=1.0)
    peak_risk: float = Field(..., ge=0.0, le=1.0)
    active_incidents: int
    total_incidents_today: int
    last_updated: float


class RiskTrendPoint(BaseModel):
    timestamp: float
    risk_score: float


# ─── Camera Schemas ───────────────────────────────────────────────────────────

class CameraStreamIn(BaseModel):
    camera_id: str = Field(..., min_length=1, max_length=64)
    stream_url: str = Field(..., description="RTSP/HTTP stream URL or device index")
    location: Optional[str] = Field(default=None, max_length=128)


class CameraOut(BaseModel):
    camera_id: str
    stream_url: str
    location: Optional[str] = None
    active: bool = True


# ─── System Schemas ───────────────────────────────────────────────────────────

class SystemStatusOut(BaseModel):
    status: str
    uptime: float
    cpu_percent: float
    memory_percent: float
    version: str
