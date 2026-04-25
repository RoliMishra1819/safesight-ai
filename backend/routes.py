"""
SafeSight AI - API Routes
REST endpoints for incidents, risk summary, streams, and system control.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse
from typing import List, Optional
import time
import asyncio
import json

from backend.schemas import (
    IncidentOut,
    RiskSummaryOut,
    ResolveIncidentIn,
    CameraStreamIn,
    SystemStatusOut,
)
from backend.database import get_db, Database

router = APIRouter()


# ─── Dependency ──────────────────────────────────────────────────────────────

def get_database() -> Database:
    return get_db()


# ─── Incidents ───────────────────────────────────────────────────────────────

@router.get("/incidents", response_model=List[IncidentOut], tags=["Incidents"])
async def list_incidents(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    threat_level: Optional[str] = Query(default=None),
    camera_id: Optional[str] = Query(default=None),
    resolved: Optional[bool] = Query(default=None),
    db: Database = Depends(get_database),
):
    """List recent incidents with optional filters."""
    incidents = await db.get_incidents(
        limit=limit,
        offset=offset,
        threat_level=threat_level,
        camera_id=camera_id,
        resolved=resolved,
    )
    return incidents


@router.get("/incidents/{incident_id}", response_model=IncidentOut, tags=["Incidents"])
async def get_incident(incident_id: str, db: Database = Depends(get_database)):
    """Get a specific incident by ID."""
    incident = await db.get_incident_by_id(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")
    return incident


@router.patch("/incidents/{incident_id}/resolve", tags=["Incidents"])
async def resolve_incident(
    incident_id: str,
    body: ResolveIncidentIn,
    db: Database = Depends(get_database),
):
    """Mark an incident as resolved."""
    success = await db.resolve_incident(incident_id, notes=body.notes)
    if not success:
        raise HTTPException(status_code=404, detail="Incident not found")
    return {"status": "resolved", "incident_id": incident_id}


@router.delete("/incidents/{incident_id}", tags=["Incidents"])
async def delete_incident(incident_id: str, db: Database = Depends(get_database)):
    """Delete an incident record."""
    success = await db.delete_incident(incident_id)
    if not success:
        raise HTTPException(status_code=404, detail="Incident not found")
    return {"status": "deleted", "incident_id": incident_id}


# ─── Risk Summary ─────────────────────────────────────────────────────────────

@router.get("/risk/summary", response_model=RiskSummaryOut, tags=["Risk"])
async def risk_summary(db: Database = Depends(get_database)):
    """Get the current risk summary across all cameras."""
    summary = await db.get_risk_summary()
    return summary


@router.get("/risk/trend", tags=["Risk"])
async def risk_trend(
    camera_id: Optional[str] = Query(default=None),
    window: int = Query(default=30, ge=5, le=300),
    db: Database = Depends(get_database),
):
    """Get risk score trend data for charts."""
    trend = await db.get_risk_trend(camera_id=camera_id, window=window)
    return {"camera_id": camera_id or "all", "trend": trend}


# ─── Real-Time SSE Stream ─────────────────────────────────────────────────────

@router.get("/stream/alerts", tags=["Streaming"])
async def stream_alerts(camera_id: Optional[str] = Query(default=None)):
    """
    Server-Sent Events stream for real-time alert delivery.
    Connect with EventSource in frontend.
    """
    async def event_generator():
        from backend.database import alert_queue
        while True:
            try:
                event = await asyncio.wait_for(alert_queue.get(), timeout=15.0)
                if camera_id and event.get("camera_id") != camera_id:
                    continue
                data = json.dumps(event)
                yield f"data: {data}\n\n"
            except asyncio.TimeoutError:
                # Keepalive ping
                yield ": keepalive\n\n"
            except Exception:
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Camera Management ────────────────────────────────────────────────────────

@router.get("/cameras", tags=["Cameras"])
async def list_cameras(db: Database = Depends(get_database)):
    """List all registered camera streams."""
    cameras = await db.get_cameras()
    return {"cameras": cameras}


@router.post("/cameras", tags=["Cameras"])
async def add_camera(body: CameraStreamIn, db: Database = Depends(get_database)):
    """Register a new camera stream."""
    camera = await db.add_camera(body.camera_id, body.stream_url, body.location)
    return {"status": "registered", "camera": camera}


@router.delete("/cameras/{camera_id}", tags=["Cameras"])
async def remove_camera(camera_id: str, db: Database = Depends(get_database)):
    """Remove a camera stream."""
    await db.remove_camera(camera_id)
    return {"status": "removed", "camera_id": camera_id}


# ─── System ───────────────────────────────────────────────────────────────────

@router.get("/system/status", response_model=SystemStatusOut, tags=["System"])
async def system_status():
    """Return system health and uptime info."""
    import psutil
    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    return SystemStatusOut(
        status="running",
        uptime=time.time(),
        cpu_percent=cpu,
        memory_percent=mem.percent,
        version="1.0.0",
    )
