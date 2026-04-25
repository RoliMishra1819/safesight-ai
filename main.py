"""
SafeSight — Main Pipeline (SSL Edition)
No training here — only inference.
Run: python main.py --video path/to/video.mp4
"""

import cv2
import numpy as np
import torch
import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from ai_engine.ssl_encoder.simclr_encoder import SimCLRTrainer, get_augmentations
from ai_engine.ssl_encoder.temporal_predictor import TemporalAutoencoder, TemporalAETrainer, AnomalyScorer
from ai_engine.threat_classifier.classifier import SSLThreatClassifier


# ─────────────────────────────────────────────
# Risk Score Calculator
# ─────────────────────────────────────────────

THREAT_WEIGHTS = {
    0: 0.0,    # Normal
    1: 0.25,   # Low
    2: 0.50,   # Medium
    3: 0.75,   # High
    4: 1.00,   # Critical
}

def calculate_risk_percentage(threat_counts: dict, total_frames: int) -> dict:
    """
    Calculate overall session risk % and per-threat-level risk %.
    """
    LEVEL_NAMES = {0:"Normal", 1:"Low", 2:"Medium", 3:"High", 4:"Critical"}

    # Per-threat-level risk % = (count / total) * 100
    per_level = {}
    for level, name in LEVEL_NAMES.items():
        count = threat_counts.get(name, 0)
        per_level[name] = round((count / max(total_frames, 1)) * 100, 1)

    # Overall risk % = weighted sum of threat counts
    weighted_sum = sum(
        threat_counts.get(LEVEL_NAMES[lvl], 0) * weight
        for lvl, weight in THREAT_WEIGHTS.items()
    )
    overall_risk = round((weighted_sum / max(total_frames, 1)) * 100, 1)

    return {
        "overall_risk_pct": min(overall_risk, 100.0),
        "per_level_risk_pct": per_level,
    }


# ─────────────────────────────────────────────
# SafeSight SSL Pipeline
# ─────────────────────────────────────────────

class SafeSightSSLPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.device = "cpu"
        self.aug = get_augmentations()

        print("[SafeSight] Loading models...")

        # SSL Encoder
        self.ssl_trainer = SimCLRTrainer(device=self.device)
        self.ssl_trainer.load(config.get("ssl_encoder_path", "models/ssl_encoder.pt"))
        self.encoder = self.ssl_trainer.encoder

        # Temporal Anomaly Scorer
        tae = TemporalAETrainer(device=self.device)
        ae_model = tae.load(config.get("temporal_ae_path", "models/temporal_ae.pt"))
        self.anomaly_scorer = AnomalyScorer(ae_model, window_size=16)

        # Threat Classifier
        clf_path = config.get("classifier_path", "models/threat_classifier.pkl")
        try:
            self.classifier = SSLThreatClassifier.load(clf_path)
            print("[SafeSight] Classifier loaded -OK")
        except Exception:
            print("[Warning] No trained classifier found. Using anomaly score only.")
            self.classifier = None

        # Session state
        self.frame_count = 0
        self.threat_log: list[dict] = []
        self.threat_counts = defaultdict(int)

        print("[SafeSight] Pipeline ready OK")

    def _encode_frame(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.aug(rgb).unsqueeze(0)
        with torch.no_grad():
            emb = self.encoder(tensor).numpy()[0]
        return emb

    def _get_behaviour_features(self, person_count: int,
                                 anomaly_score: float) -> dict:
        return {
            "loitering_score":  float(np.random.uniform(0, 0.3)),
            "crowd_density":    min(person_count / 10.0, 1.0),
            "running_score":    float(np.random.uniform(0, 0.2)),
            "person_count":     person_count,
            "avg_speed":        float(np.random.uniform(0, 1.0)),
            "anomaly_score":    anomaly_score,
            "time_in_zone":     float(np.random.uniform(0, 5.0)),
            "zone_transitions": int(np.random.randint(0, 3)),
        }

    def process_frame(self, frame: np.ndarray) -> dict:
        self.frame_count += 1

        # SSL embedding
        embedding = self._encode_frame(frame)

        # Temporal anomaly score
        anomaly_result = self.anomaly_scorer.update(embedding)

        # Behaviour features
        behaviour = self._get_behaviour_features(
            person_count=1,
            anomaly_score=anomaly_result["normalised_score"]
        )

        # Threat classification
        if self.classifier and self.classifier.is_trained:
            prediction = self.classifier.predict(embedding, behaviour)
        else:
            level = anomaly_result["threat_level"]
            prediction = {
                "threat_level": level,
                "threat_label": anomaly_result["threat_label"],
                "confidence":   anomaly_result["normalised_score"],
                "probabilities": {},
            }

        result = {
            "frame":         self.frame_count,
            "timestamp":     datetime.now().isoformat(),
            "threat_level":  prediction["threat_level"],
            "threat_label":  prediction["threat_label"],
            "confidence":    prediction["confidence"],
            "anomaly_score": anomaly_result["normalised_score"],
            "is_anomaly":    anomaly_result["is_anomaly"],
            "behaviour":     behaviour,
            "probabilities": prediction.get("probabilities", {}),
        }

        self.threat_log.append(result)
        self.threat_counts[prediction["threat_label"]] += 1
        return result

    def get_session_metrics(self) -> dict:
        anomaly_metrics = self.anomaly_scorer.get_metrics()

        # Threat distribution
        total = max(sum(self.threat_counts.values()), 1)
        distribution = {
            label: {
                "count": self.threat_counts[label],
                "pct":   round(self.threat_counts[label] / total * 100, 1),
            }
            for label in ["Normal", "Low", "Medium", "High", "Critical"]
        }

        # Risk scoring
        risk = calculate_risk_percentage(dict(self.threat_counts), total)

        # Classifier metrics
        clf_metrics = {}
        if self.classifier and self.classifier.metrics:
            clf_metrics = self.classifier.metrics

        return {
            "total_frames":        self.frame_count,
            "threat_distribution": distribution,
            "anomaly_metrics":     anomaly_metrics,
            "classifier_metrics":  clf_metrics,
            "risk_scores":         risk,
            "session_start":       self.threat_log[0]["timestamp"] if self.threat_log else None,
        }

    def save_log(self, path: str = "logs/incident_logs.json"):
        Path(path).parent.mkdir(exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "session_metrics": self.get_session_metrics(),
                "threat_log":      self.threat_log[-500:],
            }, f, indent=2)
        print(f"[SafeSight] Log saved - {path}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def process_video(video_path: str, display: bool = False):
    config = {
        "ssl_encoder_path": "models/ssl_encoder.pt",
        "temporal_ae_path": "models/temporal_ae.pt",
        "classifier_path":  "models/threat_classifier.pkl",
    }

    pipeline = SafeSightSSLPipeline(config)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[Error] Cannot open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[SafeSight] Processing: {video_path} ({total_frames} frames)")

    try:
        FRAME_SKIP = 5

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % FRAME_SKIP != 0:
                continue

            result = pipeline.process_frame(frame)

            if display:
                color = {0:(0,255,0),1:(0,255,128),2:(0,165,255),
                         3:(0,69,255),4:(0,0,255)}.get(result["threat_level"],(255,255,255))
                label = f"[{result['threat_label']}] {result['confidence']:.2f}"
                cv2.putText(frame, label, (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.imshow("SafeSight SSL", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()

    pipeline.save_log()
    metrics = pipeline.get_session_metrics()
    print(f"\n[SafeSight] Overall Risk: {metrics['risk_scores']['overall_risk_pct']}%")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeSight SSL Pipeline")
    parser.add_argument("--video",   type=str, required=True)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()
    process_video(args.video, args.display)