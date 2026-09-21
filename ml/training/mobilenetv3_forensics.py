import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class MobileNetV3Forensics(nn.Module):
    """
    Lightweight convolutional neural network backbone for in-browser image forensics.
    - Parameter count: ~2.5M
    - Input shape: [B, 3, 224, 224]
    - Output shape: [B, 3] (authentic, fully_generated, ai_edited)
    """

    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base_model = mobilenet_v3_small(weights=weights)

        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Multi-task classification head
        in_features = base_model.classifier[0].in_features  # 576 for mobilenet_v3_small
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 224, 224]
        feats = self.features(x)
        pooled = self.avgpool(feats)
        flattened = torch.flatten(pooled, 1)
        logits = self.classifier(flattened)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
