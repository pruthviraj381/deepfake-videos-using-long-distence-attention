import torch
from torch import nn


class FrameEncoder(nn.Module):

    def __init__(self, feature_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(64, feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.features(x)


class DeepfakeAttentionModel(nn.Module):

    def __init__(self, feature_dim=128, heads=4, max_frames=8):
        super().__init__()

        self.encoder = FrameEncoder(feature_dim)

        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_frames, feature_dim)
        )

        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        self.temporal = nn.MultiheadAttention(
            feature_dim,
            heads,
            batch_first=True,
            dropout=0.1
        )

        self.norm1 = nn.LayerNorm(feature_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, feature_dim)
        )

        self.norm2 = nn.LayerNorm(feature_dim)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, video):

        batch, frames, channels, height, width = video.shape

        x = video.reshape(
            batch * frames,
            channels,
            height,
            width
        )

        x = self.encoder(x)

        x = x.reshape(batch, frames, -1)

        x = x + self.position_embedding[:, :frames, :]

        attended, _ = self.temporal(
            x, x, x,
            need_weights=False
        )

        x = self.norm1(x + attended)

        feed_forward_output = self.feed_forward(x)

        x = self.norm2(x + feed_forward_output)

        x = x.mean(dim=1)

        return self.classifier(x)