from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class MobileNetV3FeatureExtractor(nn.Module):
    """Visual feature extractor backbone for Forensics Web Lab research model.

    Extracts 576-dimensional penultimate features from MobileNetV3-small.
    Input: [B, 3, 224, 224] normalized image tensor.
    Output: [B, 576] visual feature vector (after AdaptiveAvgPool2d and flatten).
    """

    def __init__(self, weights_path: str | Path | None = None) -> None:
        super().__init__()
        base = mobilenet_v3_small(weights=None)
        if weights_path:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            base.load_state_dict(state_dict)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 224, 224]
        feats = self.features(x)
        pooled = self.avgpool(feats)
        return torch.flatten(pooled, 1)


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    opset_version: int = 17,
    input_name: str = "input",
    output_name: str = "output",
) -> Path:
    """Exports a PyTorch model to ONNX format with dynamic batch dimension."""
    model.eval()
    dummy_input = torch.randn(*input_shape, requires_grad=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes={
            input_name: {0: "batch_size"},
            output_name: {0: "batch_size"},
        },
    )

    return output_path


def export_research_backbone_fp32(
    weights_path: Path | str = "models/research/pretrained/mobilenet_v3_small-047dcff4.pth",
    output_path: Path | str = "models/research/onnx/mobilenet_v3_small_backbone_fp32.onnx",
    opset_version: int = 17,
) -> Path:
    """Exports the official research MobileNetV3-small backbone to ONNX FP32 format."""
    weights_p = Path(weights_path)
    output_p = Path(output_path)
    model = MobileNetV3FeatureExtractor(weights_path=weights_p)
    return export_to_onnx(
        model=model,
        output_path=output_p,
        input_shape=(1, 3, 224, 224),
        opset_version=opset_version,
        input_name="input",
        output_name="visual_features",
    )


if __name__ == "__main__":
    out = export_research_backbone_fp32()
    print(f"Exported research backbone to {out} (size: {out.stat().st_size} bytes)")
