# Model Artifacts & Registry

This directory contains configuration, schemas, and metadata for machine learning models deployed in **Forensics Web Lab**.

## Policy on Model Weights

1. **No Dummy/Mock Models in Git**: We strictly adhere to scientific reproducibility. No fabricated binary model weights are checked into source control just to make the UI "appear" functional.
2. **Installation Verification**: In development mode without a verified downloaded or locally trained model, the web application explicitly surfaces `Model not installed`.
3. **Execution Gracefulness**: Metadata parsing (EXIF, XMP), provenance checking (C2PA adapter), and signal-based forensic feature extractors (FFT/DCT frequency statistics, JPEG block artifact analysis, noise residuals, ELA) execute independently in the browser without requiring the ML model.
4. **Model Checksum & Integrity**: Every registered model artifact is verified via SHA-256 before inference session instantiation in the browser Web Worker. Any checksum mismatch results in execution abort.

## Model Registry Specification

The file [`registry.json`](file:///d:/Documents/forensics-web-lab/models/registry.json) tracks available models, their paths, SHA-256 hashes, quantized sizes, input shapes, opset versions, and lifecycle status (`not-trained`, `trained`, `quantized`, `deprecated`).

See [`MODEL_CARD.md`](file:///d:/Documents/forensics-web-lab/models/MODEL_CARD.md) for full architecture and evaluation specifications.
