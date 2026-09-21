from pathlib import Path
import torch
import torch.nn as nn
from ml.training.mobilenetv3_forensics import MobileNetV3Forensics

def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_shape=(1, 3, 224, 224),
    opset_version: int = 17,
) -> Path:
    """
    Exports a PyTorch model to ONNX format with dynamic batch dimension.
    """
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
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    return output_path
