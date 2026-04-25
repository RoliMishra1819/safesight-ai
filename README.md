# 🛡️ SafeSight AI — Pre-Incident Risk & Threat Detection

SafeSight AI is a real-time video surveillance intelligence system that detects potential threats **before** incidents escalate. It fuses object detection, multi-object tracking, and behavioral analysis into a unified risk score — all streamed live to a monitoring dashboard.

---

## 🏗️ Architecture

```
safesight-ai/
├── ai_engine/
│   ├── detection/          # YOLOv8 object & weapon detection
│   ├── tracking/           # ByteTrack multi-object tracker
│   ├── behavior/           # Loitering, crowd, running detectors
│   ├── threat_classifier/  # Multi-signal threat scorer
│   └── risk_engine/        # Incident logging & risk aggregation
├── backend/                # FastAPI REST + SSE API
├── dashboard/              # Streamlit live monitoring UI
├── utils/                  # Video capture & frame helpers
├── datasets/               # Training datasets
├── models/                 # Trained model weights
├── notebooks/              # Training & preprocessing notebooks
├── logs/                   # Incident JSON logs
└── main.py                 # Pipeline entry point
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the AI pipeline (webcam)

```bash
python main.py --source 0 --display
```

### 3. Start the backend API

```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Launch the dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

---

## 🔍 Detection Capabilities

| Signal | Description |
|---|---|
| **Weapon Detection** | Guns, knives, bats via YOLOv8 |
| **Loitering** | Person stationary >15s in zone |
| **Running/Panic** | High-speed movement, scatter patterns |
| **Crowd Anomaly** | Sudden gatherings, overcrowding |
| **Night Activity** | Darkness-adjusted risk multiplier |

## ⚠️ Threat Levels

| Level | Score Range | Color |
|---|---|---|
| None | 0–15% | 🟢 Green |
| Low | 15–35% | 🟡 Yellow-green |
| Medium | 35–60% | 🟠 Orange |
| High | 60–80% | 🔴 Red |
| Critical | 80–100% | 🆘 Dark Red |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/incidents` | List incidents |
| GET | `/api/v1/incidents/{id}` | Get incident details |
| PATCH | `/api/v1/incidents/{id}/resolve` | Resolve incident |
| GET | `/api/v1/risk/summary` | Current risk summary |
| GET | `/api/v1/risk/trend` | Risk score history |
| GET | `/api/v1/stream/alerts` | SSE real-time alert stream |
| GET | `/api/v1/cameras` | List cameras |
| POST | `/api/v1/cameras` | Register camera |

---

## 🧠 Model Training

See notebooks:
- `notebooks/dataset_preprocessing.ipynb` — Prepare weapon & behavior datasets
- `notebooks/yolo_training.ipynb` — Fine-tune YOLOv8 on custom data
- `notebooks/threat_classifier_training.ipynb` — Train the risk classifier

---

## 🛠️ Configuration

Key parameters in `main.py` and module constructors:

```python
# Detection sensitivity
ObjectDetector(confidence_threshold=0.45)

# Loitering threshold
LoiteringDetector(loiter_threshold_seconds=15.0)

# Crowd alert threshold
CrowdDetector(crowd_threshold=5)

# Running speed threshold
RunningDetector(run_speed_threshold=80.0)  # pixels/second
```

---

## 📜 License

MIT License — see LICENSE for details.
