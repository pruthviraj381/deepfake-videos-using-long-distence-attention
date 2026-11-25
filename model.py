"""
Long-Distance Attention Model for Deepfake Video Detection
MCA Final Year Academic Project

Architecture:
1. CNN-based Frame Encoder: Extracts spatial facial features from individual frames.
2. Temporal Positional Encoding: Encodes sequential and temporal distance between frames.
3. Multi-Head Long-Distance Attention: Models temporal dependencies and detects manipulation artifacts.
4. Temporal Feature Aggregation: Pools temporal sequence representations.
5. Classification Head: Outputs probabilities for REAL (class 0) and FAKE (class 1).
"""

import math
from torch_loader import torch, nn, F, TORCH_AVAILABLE, TORCH_ERROR

if nn is None:
    class _DummyModule:
        def __init__(self, *args, **kwargs): pass
        def parameters(self): return []
        def eval(self): return self
        def train(self): return self
    class _DummyNN:
        Module = _DummyModule
        Linear = _DummyModule
        Conv2d = _DummyModule
        BatchNorm2d = _DummyModule
        MaxPool2d = _DummyModule
        AdaptiveAvgPool2d = _DummyModule
        LayerNorm = _DummyModule
        Dropout = _DummyModule
        MultiheadAttention = _DummyModule
        Sequential = _DummyModule
        Parameter = lambda x: None
    nn = _DummyNN


class FrameEncoder(nn.Module):
    """
    CNN Frame Encoder designed for CPU execution and low-memory environments (4GB RAM).
    Processes normalized frames of resolution 128x128 (or 64x64) and produces a feature vector of dimension d_model.
    """
    def __init__(self, d_model=128):
        super(FrameEncoder, self).__init__()
        self.d_model = d_model

        # Block 1: 128x128 -> 64x64
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2: 64x64 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3: 32x32 -> 16x16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Block 4: 16x16 -> 8x8
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        # Linear projection to d_model
        self.fc = nn.Linear(128 * 2 * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: (B * T, 3, 128, 128)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.adaptive_pool(x)

        x = torch.flatten(x, 1)  # (B * T, 512)
        x = self.fc(x)  # (B * T, d_model)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class LongDistanceAttentionBlock(nn.Module):
    """
    Multi-Head Long-Distance Attention Block.
    Captures temporal dependencies across distant frames in video sequences to identify
    inconsistencies, facial boundary blurring, and temporal warping typical in deepfakes.
    """
    def __init__(self, d_model=128, num_heads=4, dropout=0.1):
        super(LongDistanceAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x):
        # x: (Batch, Frames, d_model)
        # Self-attention across temporal frames
        attn_out, attn_weights = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, attn_weights


class LongDistanceAttentionDeepfakeDetector(nn.Module):
    """
    Complete Deepfake Detection Model using Long-Distance Attention.
    
    Inputs:
        x: Tensor of shape (Batch, Frames, 3, 128, 128)
    Outputs:
        logits: Tensor of shape (Batch, 2) where 0 = REAL, 1 = FAKE
    """
    def __init__(self, num_frames=4, image_size=128, d_model=128, num_heads=4, num_classes=2):
        super(LongDistanceAttentionDeepfakeDetector, self).__init__()
        self.num_frames = num_frames
        self.image_size = image_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_classes = num_classes

        # 1. Spatial Frame Encoder (CNN)
        self.encoder = FrameEncoder(d_model=d_model)

        # 2. Learnable Temporal Positional Encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_frames, d_model))
        if hasattr(nn, 'init'):
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # 3. Multi-Head Long-Distance Temporal Attention
        self.attention_block1 = LongDistanceAttentionBlock(d_model=d_model, num_heads=num_heads)
        self.attention_block2 = LongDistanceAttentionBlock(d_model=d_model, num_heads=num_heads)

        # 4. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, return_attention=False):
        """
        Forward pass.
        Args:
            x: (B, T, 3, H, W) where T=num_frames, H=W=64
            return_attention: whether to return attention weights for explainability
        """
        B, T, C, H, W = x.shape

        # Fold batch and time to pass frames through CNN encoder
        x_reshaped = x.view(B * T, C, H, W)
        frame_features = self.encoder(x_reshaped)  # (B * T, d_model)

        # Reshape back to temporal sequence: (B, T, d_model)
        temporal_sequence = frame_features.view(B, T, self.d_model)

        # Add positional embedding (sliced up to T if sequence length varies)
        if T <= self.pos_embedding.shape[1]:
            temporal_sequence = temporal_sequence + self.pos_embedding[:, :T, :]
        else:
            # Interpolate or expand positional encoding dynamically if more frames
            pos = F.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=T,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            temporal_sequence = temporal_sequence + pos

        # Apply Long-Distance Attention blocks
        attended_seq, weights1 = self.attention_block1(temporal_sequence)
        attended_seq, weights2 = self.attention_block2(attended_seq)

        # Temporal pooling (Global Average Pooling across frame dimension)
        video_representation = torch.mean(attended_seq, dim=1)  # (B, d_model)

        # Classification logits (REAL vs FAKE)
        logits = self.classifier(video_representation)  # (B, 2)

        if return_attention:
            return logits, weights2
        return logits

    def predict_proba(self, x):
        """
        Returns Softmax probabilities for REAL (class 0) and FAKE (class 1).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities

    def count_parameters(self):
        """Returns total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_frames=4, image_size=128, d_model=128, num_heads=4):
    """
    Factory helper to instantiate the detector with standard configuration.
    """
    return LongDistanceAttentionDeepfakeDetector(
        num_frames=num_frames,
        image_size=image_size,
        d_model=d_model,
        num_heads=num_heads,
        num_classes=2
    )


if __name__ == "__main__":
    model = create_model()
    dummy_input = torch.randn(2, 4, 3, 64, 64)
    out = model(dummy_input)
    proba = model.predict_proba(dummy_input)
    print(f"Model created successfully!")
    print(f"Trainable parameters: {model.count_parameters():,}")
    print(f"Output logits shape: {out.shape}")
    print(f"Predicted probabilities shape: {proba.shape}")
    print(f"Sample prediction probs [REAL, FAKE]: {proba[0].tolist()}")
