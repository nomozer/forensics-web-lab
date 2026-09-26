from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_onnx_model(
    input_onnx_path: Path,
    output_quantized_path: Path,
    weight_type=QuantType.QInt8,
) -> Path:
    """
    Applies dynamic INT8 post-training quantization to reduce binary footprint < 10 MB.
    """
    output_quantized_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(input_onnx_path),
        model_output=str(output_quantized_path),
        weight_type=weight_type,
    )
    return output_quantized_path
