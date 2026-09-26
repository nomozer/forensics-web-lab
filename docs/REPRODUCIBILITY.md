# Reproducibility Guide: Forensics Web Lab

## 1. System Requirements & Prerequisites

* **Node.js**: Version $\ge 20.x$ (tested on v24.x LTS)
* **Package Manager**: `pnpm` $\ge 9.x$
* **Python**: Version $\ge 3.12$
* **Operating Systems Supported**: Linux (Ubuntu 22.04+), macOS (13+), Windows 11 (PowerShell / WSL2)

---

## 2. Web Workspace Setup & Build

```bash
# 1. Clone the repository and switch to the development branch
git checkout feat/production-ai-image-forensics

# 2. Install workspace dependencies
pnpm install

# 3. Run linting and type-checking across all packages
pnpm lint
pnpm typecheck

# 4. Run automated test suites
pnpm test

# 5. Build production bundle
pnpm build
```

---

## 3. Python ML & Research Pipeline Setup

```bash
# 1. Navigate to ML directory
cd ml

# 2. Create and activate a dedicated virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 3. Install dependencies in editable mode
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run Python unit and contract tests
pytest tests/ -v
```

---

## 4. Reproducing Research Experiments

All experiments are governed by declarative configuration files in `ml/configs/`:

```bash
# Run baseline training experiment (when authorized data is available)
python -m ml.training.train --config ml/configs/baseline_mobilenetv3.yaml

# Run evaluation suite across all benchmark matrices
python -m ml.evaluation.evaluate --model-path checkpoints/best_model.pt --config ml/configs/eval_matrix.yaml

# Export verified ONNX model and INT8 quantization
python -m ml.export.export_onnx --checkpoint checkpoints/best_model.pt --output ../models/global-local-v1.onnx --quantize int8
```

*Note on Seeds*: All PyTorch, NumPy, and Python standard library random seeds are locked to `42` by default in config files to ensure bitwise reproducibility of data splits and batching.
