import torch
from torch import nn

class FrameEncoder(nn.Module):
    """Small CNN applied independently to each sampled video frame."""
    def __init__(self, feature_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, feature_dim), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.features(x)

class DeepfakeAttentionModel(nn.Module):
    """CNN frame encoder + temporal self-attention + binary classifier."""
    def __init__(self, feature_dim=128, heads=4):
        super().__init__()
        self.encoder = FrameEncoder(feature_dim)
        self.temporal = nn.MultiheadAttention(feature_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.25), nn.Linear(64, 2)
        )
    def forward(self, video):
        # video: [batch, frames, channels, height, width]
        b, t, c, h, w = video.shape
        x = video.reshape(b * t, c, h, w)
        x = self.encoder(x).reshape(b, t, -1)
        attended, _ = self.temporal(x, x, x, need_weights=False)
        x = self.norm(x + attended)
        return self.classifier(x.mean(dim=1))
