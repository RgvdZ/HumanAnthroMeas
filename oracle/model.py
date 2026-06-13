import torch
import torch.nn as nn
import torchvision.models as models
from config import NUM_MEASUREMENTS

class Oracle(nn.Module):
    def __init__(self, num_measurements=NUM_MEASUREMENTS):
        super().__init__()
        # Load MobileNetV3-Small, modify first conv to accept 3 channels (mask + warp)
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        # Replace first conv: original is 3 channels, we keep it as 3.
        # No change needed because our input is exactly 3 channels.
        in_features = self.backbone.classifier[0].in_features
        # Remove classifier layers
        self.backbone.classifier = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_measurements)
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        features = self.backbone(x)          # (B, in_features)
        delta = self.head(features)          # (B, num_measurements)
        return delta
