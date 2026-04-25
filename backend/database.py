"""
SafeSight AI - MongoDB Database Layer
Async MongoDB via Motor driver for incident, camera, and risk snapshot storage.
"""

import asyncio
import time
import logging
from typing import List, Optional, Dict, Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import DESCENDING, ASCENDING, IndexModel
import os

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB",  "safesight")

# Global alert queue for SSE streaming
alert_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

_db_instance: Optional["Database"] = None


async def init_db():
    global _db_instance
    _db_instance = Database(MONGO_URI, MONGO_DB)
    await _db_instance.initialize()


def get_db() -> "Database":
    if _db_instance is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _db_instance


# ─── Database Class ───────────────────────────────────────────────────────────

class Database:
    def __init__(self, uri: str = MONGO_URI, db_name: str = MONGO_DB):
        self.uri     = uri
        self.db_name = db_name
        self._client: Optional[AsyncIOMotorClient] = None
        self._db:     Optional[AsyncIOMotorDatabase] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self):
        self._client = AsyncIOMotorClient(self.uri)
        self._db     = self._client[self.db_name]

        # Create indexes
        await self._db.incidents.create_indexes([
            IndexModel([("timestamp",    DESCENDING)]),
            IndexModel([("camera_id",    ASCENDING)]),
            IndexModel([("threat_level", ASCENDING)]),
            IndexModel([("incident_id",  ASCENDING)], unique=True),
            IndexModel([("resolved",     ASCENDING)]),
        ])
        await self._db.cameras.create_indexes([
            IndexModel([("camera_id", ASCENDING)], unique=True),
        ])
        await self._db.risk_snapshots.create_indexes([
            IndexModel([("camera_id", ASCENDING), ("timestamp", DESCENDING)]),
        ])

        logger.info(f"MongoDB connected → {self.uri} / db={self.db_name}")

    async def close(self):
        if self._client:
            self._client.close()

    # ── Incidents ─────────────────────────────────────────────────────────────

    async def save_incident(self, report_dict: Dict[str, Any]):
        """Upsert an incident document into MongoDB."""
        doc = {**report_dict}
        doc.setdefault("resolved", False)
        doc.setdefault("notes",    "")

        await self._db.incidents.update_one(
            {"incident_id": doc["incident_id"]},
            {"$set": doc},
            upsert=True,
        )

        # Push to SSE live-stream queue
        try:
            alert_queue.put_nowait({**report_dict, "event": "new_incident"})
        except asyncio.QueueFull:
            pass

        logger.debug(f"Incident saved: {doc['incident_id']}")

    async def get_incidents(
        self,
        limit:        int            = 50,
        offset:       int            = 0,
        threat_level: Optional[str]  = None,
        camera_id:    Optional[str]  = None,
        resolved:     Optional[bool] = None,
    ) -> List[Dict]:
        filt: Dict[str, Any] = {}
        if threat_level:
            filt["threat_level"] = threat_level
        if camera_id:
            filt["camera_id"] = camera_id
        if resolved is not None:
            filt["resolved"] = resolved

        cursor = (
            self._db.incidents
            .find(filt, {"_id": 0})
            .sort("timestamp", DESCENDING)
            .skip(offset)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    async def get_incident_by_id(self, incident_id: str) -> Optional[Dict]:
        return await self._db.incidents.find_one(
            {"incident_id": incident_id}, {"_id": 0}
        )

    async def resolve_incident(self, incident_id: str, notes: str = "") -> bool:
        result = await self._db.incidents.update_one(
            {"incident_id": incident_id},
            {"$set": {"resolved": True, "notes": notes}},
        )
        return result.modified_count > 0

    async def delete_incident(self, incident_id: str) -> bool:
        result = await self._db.incidents.delete_one({"incident_id": incident_id})
        return result.deleted_count > 0

    # ── Risk Summary & Trend ──────────────────────────────────────────────────

    async def get_risk_summary(self) -> Dict:
        since = time.time() - 300  # last 5 minutes

        pipeline = [
            {"$match": {"timestamp": {"$gte": since}}},
            {"$group": {
                "_id":       None,
                "avg_risk":  {"$avg": "$risk_score"},
                "peak_risk": {"$max": "$risk_score"},
            }},
        ]
        agg = await self._db.risk_snapshots.aggregate(pipeline).to_list(1)
        avg_risk  = agg[0]["avg_risk"]  if agg else 0.0
        peak_risk = agg[0]["peak_risk"] if agg else 0.0

        active = await self._db.incidents.count_documents({"resolved": False})

        today_start = time.mktime(
            time.strptime(time.strftime("%Y-%m-%d"), "%Y-%m-%d")
        )
        today_count = await self._db.incidents.count_documents(
            {"timestamp": {"$gte": today_start}}
        )

        return {
            "current_risk":          round(avg_risk  or 0, 4),
            "rolling_avg_risk":      round(avg_risk  or 0, 4),
            "peak_risk":             round(peak_risk or 0, 4),
            "active_incidents":      active,
            "total_incidents_today": today_count,
            "last_updated":          time.time(),
        }

    async def get_risk_trend(
        self,
        camera_id: Optional[str] = None,
        window:    int           = 30,
    ) -> List[Dict]:
        filt: Dict[str, Any] = {}
        if camera_id:
            filt["camera_id"] = camera_id

        cursor = (
            self._db.risk_snapshots
            .find(filt, {"_id": 0, "timestamp": 1, "risk_score": 1})
            .sort("timestamp", DESCENDING)
            .limit(window)
        )
        docs = await cursor.to_list(length=window)
        return list(reversed(docs))

    async def save_risk_snapshot(self, camera_id: str, risk_score: float):
        await self._db.risk_snapshots.insert_one({
            "camera_id":  camera_id,
            "timestamp":  time.time(),
            "risk_score": risk_score,
        })

    # ── Cameras ───────────────────────────────────────────────────────────────

    async def add_camera(
        self,
        camera_id:  str,
        stream_url: str,
        location:   Optional[str] = None,
    ) -> Dict:
        doc = {
            "camera_id":  camera_id,
            "stream_url": stream_url,
            "location":   location,
            "active":     True,
            "added_at":   time.time(),
        }
        await self._db.cameras.update_one(
            {"camera_id": camera_id},
            {"$set": doc},
            upsert=True,
        )
        return doc

    async def get_cameras(self) -> List[Dict]:
        cursor = self._db.cameras.find({"active": True}, {"_id": 0})
        return await cursor.to_list(length=100)

    async def remove_camera(self, camera_id: str):
        await self._db.cameras.update_one(
            {"camera_id": camera_id},
            {"$set": {"active": False}},
        )

    # ── Analytics ─────────────────────────────────────────────────────────────

    async def get_incident_stats(self) -> Dict:
        """Aggregated counts by threat level and camera for dashboard charts."""
        by_level = await self._db.incidents.aggregate([
            {"$group": {"_id": "$threat_level", "count": {"$sum": 1}}},
        ]).to_list(10)

        by_camera = await self._db.incidents.aggregate([
            {"$group": {"_id": "$camera_id", "count": {"$sum": 1}}},
            {"$sort": {"count": DESCENDING}},
        ]).to_list(20)

        hourly = await self._db.incidents.aggregate([
            {"$match": {"timestamp": {"$gte": time.time() - 86400}}},
            {"$group": {
                "_id": {
                    "$toInt": {"$divide": ["$timestamp", 3600]}
                },
                "count": {"$sum": 1},
                "avg_risk": {"$avg": "$risk_score"},
            }},
            {"$sort": {"_id": ASCENDING}},
        ]).to_list(24)

        return {
            "by_level":  [{"level":  d["_id"], "count": d["count"]} for d in by_level],
            "by_camera": [{"camera": d["_id"], "count": d["count"]} for d in by_camera],
            "hourly_24h": [
                {
                    "hour":     d["_id"],
                    "count":    d["count"],
                    "avg_risk": round(d["avg_risk"], 4),
                }
                for d in hourly
            ],
        }
