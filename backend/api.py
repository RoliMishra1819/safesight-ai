"""
SafeSight AI - FastAPI Backend
"""

import sys
import os
import traceback
import uuid
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(ROOT)  # Fix relative model paths
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import init_db, get_db
from main import process_video

DATASET_DIR = Path(r"C:\Users\roli5\OneDrive\Desktop\SafeSight\safesight-ai\datasets")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    print(f"[SafeSight API] MongoDB connected")
    print(f"[SafeSight API] DATASET_DIR exists = {DATASET_DIR.exists()}")
    if DATASET_DIR.exists():
        for f in DATASET_DIR.iterdir():
            if f.is_dir():
                count = len(list(f.rglob("*.mp4")))
                print(f"[SafeSight API]   {f.name}: {count} mp4 files")
    yield

app = FastAPI(title="SafeSight AI API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class AnalyseRequest(BaseModel):
    video_name: str

def _dominant_threat(metrics: dict) -> str:
    dist = metrics.get("threat_distribution", {})
    if not dist:
        return "Normal"
    return max(dist, key=lambda k: dist[k].get("count", 0))

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/debug")
def debug():
    try:
        if not DATASET_DIR.exists():
            return {"error": f"Does not exist: {DATASET_DIR}"}
        folder_info = {}
        for folder in sorted(DATASET_DIR.iterdir()):
            if folder.is_dir():
                videos = list(folder.rglob("*.mp4"))
                folder_info[folder.name] = {"count": len(videos), "example": str(videos[0]) if videos else None}
        return {"dataset_dir": str(DATASET_DIR), "exists": True, "folders": folder_info}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

@app.get("/videos")
def list_videos():
    grouped = {}
    try:
        if not DATASET_DIR.exists():
            return {"videos": [], "grouped": {}, "count": 0, "error": f"Directory not found: {DATASET_DIR}"}
        for folder in sorted(DATASET_DIR.iterdir()):
            if not folder.is_dir():
                continue
            mp4_files = sorted(folder.rglob("*.mp4"))
            if not mp4_files:
                continue
            videos = [str(f.relative_to(folder)).replace("\\", "/") for f in mp4_files]
            grouped[folder.name] = sorted(videos)
    except Exception as e:
        print(f"[ERROR] list_videos: {e}")
        traceback.print_exc()
        return {"videos": [], "grouped": {}, "count": 0, "error": str(e)}
    all_videos = [f"{cat}/{v}" for cat, vids in grouped.items() for v in vids]
    print(f"[SafeSight API] /videos -> {len(all_videos)} total")
    return {"videos": all_videos, "grouped": grouped, "count": len(all_videos)}

@app.post("/analyse")
async def analyse_video(req: AnalyseRequest):
    video_path = DATASET_DIR / req.video_name.replace("/", os.sep)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
    print(f"[SafeSight API] Starting pipeline: {req.video_name}")
    metrics = process_video(str(video_path), display=False)
    if metrics is None:
        raise HTTPException(status_code=500, detail="Pipeline returned None.")
    dominant = _dominant_threat(metrics)
    risk_pct = metrics.get("risk_scores", {}).get("overall_risk_pct", 0)
    incident = {
        "incident_id": str(uuid.uuid4()), "video_name": req.video_name,
        "camera_id": req.video_name, "filename": req.video_name.split("/")[-1],
        "category": req.video_name.split("/")[0], "analysed_at": datetime.utcnow().isoformat(),
        "timestamp": datetime.utcnow().timestamp(), "threat_level": dominant,
        "risk_score": round(risk_pct / 100, 4), "total_frames": metrics.get("total_frames", 0),
        "threat_distribution": metrics.get("threat_distribution", {}),
        "anomaly_metrics": metrics.get("anomaly_metrics", {}),
        "classifier_metrics": metrics.get("classifier_metrics", {}),
        "risk_scores": metrics.get("risk_scores", {}),
        "session_start": metrics.get("session_start"), "resolved": False, "notes": "",
    }
    db = get_db()
    await db.save_incident(incident)
    await db.save_risk_snapshot(camera_id=req.video_name, risk_score=incident["risk_score"])
    print(f"[SafeSight API] Saved incident_id={incident['incident_id']}")
    return incident

@app.get("/sessions")
async def get_all_sessions(limit: int = 20):
    return await get_db().get_incidents(limit=limit)

@app.get("/session/{video_name:path}")
async def get_session(video_name: str):
    db = get_db()
    results = await db.get_incidents(limit=1, camera_id=video_name)
    if results:
        return results[0]
    filename = video_name.split("/")[-1]
    for s in await db.get_incidents(limit=100):
        if s.get("filename") == filename or s.get("camera_id", "").endswith(filename):
            return s
    raise HTTPException(status_code=404, detail=f"No session found for: {video_name}")

@app.get("/risk-summary")
async def risk_summary():
    return await get_db().get_risk_summary()

@app.get("/analytics")
async def analytics():
    return await get_db().get_incident_stats()

@app.get("/incidents")
async def incidents(limit: int = 50, threat_level: str = None, resolved: bool = None):
    return await get_db().get_incidents(limit=limit, threat_level=threat_level, resolved=resolved)

@app.put("/incident/{incident_id}/resolve")
async def resolve_incident(incident_id: str, notes: str = ""):
    ok = await get_db().resolve_incident(incident_id, notes)
    if not ok:
        raise HTTPException(404, detail="Incident not found")
    return {"resolved": True, "incident_id": incident_id}


