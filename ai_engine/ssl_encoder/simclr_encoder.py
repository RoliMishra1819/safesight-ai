"""
SafeSight SSL Encoder — SimCLR-style Contrastive Learning
Trains a lightweight CNN encoder on unlabeled video frames.
No labels required. Learns rich visual representations from augmentations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import pickle
import os


# ─────────────────────────────────────────────
# 1. Lightweight Encoder (CPU-friendly)
# ─────────────────────────────────────────────

class SafeSightEncoder(nn.Module):
    """
    Lightweight CNN encoder optimised for CPU inference.
    Input:  (B, 3, 64, 64) normalised RGB frame
    Output: (B, 128) L2-normalised embedding
    """
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                          # 32x32

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                          # 16x16

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),                             # 4x4
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)   # L2-normalise for contrastive loss


# ─────────────────────────────────────────────
# 2. NT-Xent Contrastive Loss (SimCLR)
# ─────────────────────────────────────────────

class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)                       # (2B, D)
        sim = torch.mm(z, z.T) / self.temperature             # (2B, 2B)

        # Mask out self-similarities
        mask = torch.eye(2 * B, dtype=torch.bool)
        sim.masked_fill_(mask, float('-inf'))

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([torch.arange(B) + B,
                            torch.arange(B)]).to(z.device)
        loss = F.cross_entropy(sim, labels)
        return loss


# ─────────────────────────────────────────────
# 3. Augmentation Pipeline
# ─────────────────────────────────────────────

def get_augmentations(image_size: int = 64) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────
# 4. Frame Dataset (unlabeled)
# ─────────────────────────────────────────────

class UnlabeledFrameDataset(torch.utils.data.Dataset):
    """
    Loads raw frames (numpy arrays or file paths) without any labels.
    Returns two augmented views of the same frame for contrastive learning.
    """
    def __init__(self, frames: list, image_size: int = 64):
        self.frames = frames
        self.aug = get_augmentations(image_size)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        if isinstance(frame, str):
            import cv2
            frame = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB)
        frame = np.array(frame, dtype=np.uint8)
        return self.aug(frame), self.aug(frame)   # two different augmentations


# ─────────────────────────────────────────────
# 5. Trainer
# ─────────────────────────────────────────────

class SimCLRTrainer:
    def __init__(self, embedding_dim: int = 128, temperature: float = 0.5,
                 lr: float = 3e-4, device: str = "cpu"):
        self.device = device
        self.encoder = SafeSightEncoder(embedding_dim).to(device)
        self.criterion = NTXentLoss(temperature)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.loss_history: list[float] = []

    def train(self, frames: list, epochs: int = 10,
              batch_size: int = 32) -> list[float]:
        dataset = UnlabeledFrameDataset(frames)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        self.encoder.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for view1, view2 in loader:
                view1, view2 = view1.to(self.device), view2.to(self.device)
                z1, z2 = self.encoder(view1), self.encoder(view2)
                loss = self.criterion(z1, z2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg = epoch_loss / len(loader)
            self.loss_history.append(avg)
            print(f"[SSL] Epoch {epoch+1}/{epochs}  Loss: {avg:.4f}")

        return self.loss_history

    def save(self, path: str = "models/ssl_encoder.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.encoder.state_dict(), path)
        print(f"[SSL] Encoder saved → {path}")

    def load(self, path: str = "models/ssl_encoder.pt"):
        self.encoder.load_state_dict(
            torch.load(path, map_location=self.device))
        self.encoder.eval()

    def encode_frame(self, frame: np.ndarray) -> np.ndarray:
        """Encode a single BGR frame → 128-d numpy embedding."""
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        aug = get_augmentations()
        tensor = aug(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.encoder(tensor)
        return emb.cpu().numpy()[0]

