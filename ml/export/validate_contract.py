from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

def validate_onnx_contract(
    model: nn.Module,
    onnx_path: Path,
    tolerance: float = 1e-4,
    batch_size: int = 2,
) -> bool:
    """
    Validates numerical equivalence between PyTorch model and ONNX Runtime CPU session.
    """
    model.eval()
    test_tensor = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        torch_out = model(test_tensor).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": test_tensor.numpy()})[0]

    max_diff = np.max(np.abs(torch_out - onnx_out))
    print(f"[CONTRACT TEST] Maximum absolute discrepancy: {max_diff:.6e} (tolerance: {tolerance})")

    return bool(max_diff < tolerance)
