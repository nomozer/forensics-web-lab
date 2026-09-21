# ADR 0006: Research and Product Data Track Isolation (Dual-Track Policy)

## Status
Accepted (Phase 4A.0)

## Context
Forensics Web Lab operates simultaneously as:
1. **An academic scientific research initiative**: Investigating multi-generator synthetic artifacts, benchmark robustness, and calibration on established academic computer vision datasets (e.g. `GenImage`, `SAGI-D`, `RAID`).
2. **A client-side web product**: Providing in-browser zero-server-egress media inspection tools for end users, journalists, and forensic analysts.

Crucially, major academic vision datasets are governed by restrictive non-commercial terms. Specifically:
* **GenImage** is licensed under `CC BY-NC-SA 4.0 with additional dataset terms`, explicitly forbidding commercial usage of both the dataset and any **derivative works** (such as trained neural network weights).
* Other datasets (`SAGI-D`, `RealHD`) have ambiguous redistribution rights or pending availability.

If a neural network trained on non-commercial academic data were bundled into the web product, it would legally contaminate the product, creating copyright and licensing violations. Conversely, completely forbidding academic datasets would cripple scientific benchmarking against established literature.

Therefore, we require a formal architectural policy that strictly decouples the **Research Track** from the **Product Track**.

## Decision

We establish an immutable **Dual-Track Data & Model Governance Policy**:

### 1. Research Track (`research`)
* **Permitted**:
  - Non-commercial academic research datasets (e.g. `GenImage`, `SAGI-D`, `RAID`).
  - Benchmarking, ablation studies, and publication-oriented experiments.
  - Experimental checkpoints stored exclusively in `models/research/` and evaluation artifacts in `artifacts/research/`.
* **Strictly Prohibited**:
  - Registering research checkpoints into `models/registry.json` (the production registry).
  - Bundling research checkpoints into the web client build (`apps/web/dist`).
  - Distributing research weights as a production model release.
  - Using research benchmark metrics to advertise commercial product performance without independent real-world product evaluation.

### 2. Product Track (`product`)
* **Permitted**:
  - Exclusively project-owned datasets, public-domain imagery, or data with explicit, verified commercial training and model distribution grants (`product-eligible`).
  - Purely synthetic geometric/signal fixtures (`ml/tests/fixtures/generated-smoke/`) for technical pipeline verification.
  - Verified checkpoints stored in `models/product/` that pass all 17 criteria of `docs/MODEL_ACQUISITION_GATE.md`.
* **Strictly Prohibited**:
  - Ingesting any dataset with `commercialUse: prohibited` or `derivativeWeights: prohibited`.
  - Promoting any model lacking verifiable dataset provenance, cryptographic checksums, or complete lineage manifests.

### 3. Directory Layout & Isolation Conventions
```text
data/
├── research/              # Academic / non-commercial data (quarantined from Git)
│   └── README.md
└── product/               # Product-eligible data (quarantined from Git)
    └── README.md

artifacts/
├── research/              # Research evaluation metrics, curves, calibration plots
│   └── README.md
└── product/               # Production calibration tables, runtime profiles
    └── README.md

models/
├── research/              # Checkpoints trained on academic data (quarantined from Git)
│   └── README.md
└── product/               # Production-eligible ONNX models (quarantined from Git)
    └── README.md
```

### 4. Git Version Control Invariance
* Under no circumstances are raw image archives, extracted dataset splits, or binary model weights committed to the Git repository.
* Git tracks **only**:
  - Structural `README.md` files.
  - Machine-readable schemas (`docs/schemas/`).
  - Dataset registry (`datasets/registry.json`).
  - Cryptographic manifests (SHA-256 hashes, source groupings, sample IDs).
  - License audit evidence and reproduction scripts.

### 5. Automated Contamination Guards
* Automated validation tooling (in `@forensics/shared` and `ml/datasets/`) must enforce:
  1. Rejection of `research-only` datasets in product acquisition or training commands.
  2. Rejection of research checkpoints in production registry validation.
  3. Rejection of `blocked` datasets from automated retrieval.
  4. Failure to promote any model lacking a verified, clean manifest hash.

## Consequences
* **Legal Safety**: The production web client remains 100% legally clean, preventing any intellectual property or licensing infringement.
* **Scientific Rigor**: Academic researchers can reproduce and evaluate state-of-the-art algorithms on established benchmarks without compromising the product.
* **Auditability**: Complete lineage tracking ensures that every production weight is mathematically and legally traceable to its source data.
