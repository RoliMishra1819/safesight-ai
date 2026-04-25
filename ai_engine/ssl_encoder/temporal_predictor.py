"""
SafeSight Temporal Predictor — Anomaly Detection via Prediction Error
The model learns 'normal' temporal patterns from SSL embeddings.
High reconstruction/prediction error → anomaly → elevated threat score.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import pickle
import os


# ─────────────────────────────────────────────
# 1. Temporal Autoencoder (LSTM-based)
# ─────────────────────────────────────────────

class TemporalAutoencoder(nn.Module):
    """
    Encodes a window of SSL frame embeddings, then reconstructs them.
    Trained on NORMAL sequences only → anomalies produce high recon error.

    Input:  (B, T, D) — batch of T-frame windows, each frame D-dimensional
    Output: (B, T, D) — reconstructed sequence
    """
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64,
                 num_layers: int = 2, window_size: int = 16):
        super().__init__()
        self.window_size = window_size
        self.input_dim = input_dim

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=0.1)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=0.1)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor):
        # Encode
        _, (h, c) = self.encoder(x)
        # Decode: use zero input, let hidden state drive reconstruction
        dec_input = torch.zeros(x.size(0), x.size(1),
                                self.encoder.hidden_size).to(x.device)
        dec_out, _ = self.decoder(dec_input, (h, c))
        reconstruction = self.output_layer(dec_out)
        return reconstruction


# ─────────────────────────────────────────────
# 2. Anomaly Scorer
# ─────────────────────────────────────────────

class AnomalyScorer:
    """
    Maintains a sliding window of SSL embeddings.
    Computes frame-level anomaly scores using reconstruction error.
    Maps anomaly score → SafeSight threat level (0–4).
    """

    THREAT_MAP = {
        (0.00, 0.20): (0, "Normal"),
        (0.20, 0.40): (1, "Low"),
        (0.40, 0.60): (2, "Medium"),
        (0.60, 0.80): (3, "High"),
        (0.80, 1.01): (4, "Critical"),
    }

    def __init__(self, model: TemporalAutoencoder, window_size: int = 16,
                 threshold_percentile: float = 95, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.window_size = window_size
        self.threshold_percentile = threshold_percentile
        self.window: deque = deque(maxlen=window_size)
        self.threshold: float = 0.5          # calibrated after training
        self.score_history: list[float] = []

    def calibrate(self, normal_embeddings: np.ndarray):
        """
        Compute threshold from normal validation embeddings.
        Call this AFTER training on normal data.
        """
        errors = []
        for i in range(len(normal_embeddings) - self.window_size):
            window = normal_embeddings[i:i + self.window_size]
            score = self._score_window(window)
            errors.append(score)
        self.threshold = float(np.percentile(errors, self.threshold_percentile))
        print(f"[AnomalyScorer] Threshold calibrated: {self.threshold:.4f} "
              f"(p{self.threshold_percentile})")

    def _score_window(self, window: np.ndarray) -> float:
        tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(
            self.device)
        with torch.no_grad():
            recon = self.model(tensor)
        error = F.mse_loss(recon, tensor).item()
        return error

    def update(self, embedding: np.ndarray) -> dict:
        """
        Feed one frame embedding. Returns anomaly info dict.
        """
        self.window.append(embedding)
        if len(self.window) < self.window_size:
            return {"anomaly_score": 0.0, "normalised_score": 0.0,
                    "threat_level": 0, "threat_label": "Normal",
                    "is_anomaly": False}

        raw_score = self._score_window(np.array(self.window))
        norm_score = min(raw_score / (self.threshold * 2), 1.0)
        self.score_history.append(norm_score)

        threat_level, threat_label = self._map_threat(norm_score)
        return {
            "anomaly_score": raw_score,
            "normalised_score": round(norm_score, 4),
            "threat_level": threat_level,
            "threat_label": threat_label,
            "is_anomaly": raw_score > self.threshold,
        }

    def _map_threat(self, norm_score: float):
        for (lo, hi), (level, label) in self.THREAT_MAP.items():
            if lo <= norm_score < hi:
                return level, label
        return 4, "Critical"

    def get_metrics(self) -> dict:
        """Aggregate metrics for dashboard."""
        if not self.score_history:
            return {}
        arr = np.array(self.score_history)
        return {
            "mean_anomaly_score": float(np.mean(arr)),
            "max_anomaly_score": float(np.max(arr)),
            "anomaly_rate": float(np.mean(arr > 0.5)),
            "score_history": self.score_history[-200:],    # last 200 frames
        }


# ─────────────────────────────────────────────
# 3. Trainer for Temporal Autoencoder
# ─────────────────────────────────────────────

class TemporalAETrainer:
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64,
                 window_size: int = 16, lr: float = 1e-3, device: str = "cpu"):
        self.device = device
        self.window_size = window_size
        self.model = TemporalAutoencoder(input_dim, hidden_dim,
                                         window_size=window_size).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_history: list[float] = []

    def _make_windows(self, embeddings: np.ndarray) -> np.ndarray:
        """Slide a window over embedding sequence."""
        windows = []
        for i in range(len(embeddings) - self.window_size):
            windows.append(embeddings[i:i + self.window_size])
        return np.array(windows)

    def train(self, normal_embeddings: np.ndarray,
              epochs: int = 20, batch_size: int = 32) -> list[float]:
        """Train ONLY on normal behaviour embeddings."""
        windows = self._make_windows(normal_embeddings)
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(windows, dtype=torch.float32))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                recon = self.model(batch)
                loss = F.mse_loss(recon, batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg = epoch_loss / len(loader)
            self.loss_history.append(avg)
            print(f"[TemporalAE] Epoch {epoch+1}/{epochs}  Loss: {avg:.6f}")

        return self.loss_history

    def save(self, path: str = "models/temporal_ae.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[TemporalAE] Model saved → {path}")

    def load(self, path: str = "models/temporal_ae.pt") -> TemporalAutoencoder:
        self.model.load_state_dict(
            torch.load(path, map_location=self.device))
        self.model.eval()
        return self.model