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

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.2,
        weights_path: str | None = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        if weights_path:
            base_model = mobilenet_v3_small(weights=None)
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            base_model.load_state_dict(state_dict)
        elif pretrained:
            base_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        else:
            base_model = mobilenet_v3_small(weights=None)

        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

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
