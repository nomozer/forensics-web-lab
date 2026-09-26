# ADR 0001: Monorepo Structure and Tooling Selection

## Status
Accepted

## Context
Forensics Web Lab is both a scientific research initiative and an end-user web product. The repository encompasses:
1. Client-side web application (React, TypeScript, Vite, Web Workers, Canvas API).
2. Specialized TypeScript modules: `inference` (ONNX Runtime Web), `forensics` (DSP & frequency algorithms), `provenance` (EXIF/XMP/C2PA adapter), `report` (JSON/PDF generator), `shared` (contracts, schemas, utilities).
3. Machine learning research code: dataset preparation, PyTorch training pipelines, evaluation harnesses, ONNX export, and quantization scripts.
4. Model artifacts, experiment logs, and research documentation.

We require a coherent monorepo structure with reproducible tooling, strict typing, and zero cross-pollution between browser packages and heavy ML environments.

## Decision
1. **Repository Layout**: Monorepo with explicit workspace separation:
   * `apps/web`: Vite + React UI application.
   * `packages/inference`: ONNX Runtime Web session management, Web Worker abstraction, fallback orchestration.
   * `packages/forensics`: Pure TypeScript signal processing (2D-DCT/FFT, JPEG block boundary grid, noise residual variance, ELA).
   * `packages/provenance`: Lightweight metadata extraction and C2PA adapter interface.
   * `packages/report`: Schema validator and printable DOM report generator.
   * `packages/shared`: Shared types, constants, and schema definitions.
   * `ml/`: Python 3.12 workspace managed with standard virtual environments (`configs/`, `datasets/`, `training/`, `evaluation/`, `export/`, `tests/`).
   * `models/`: Versioned model registry (`registry.json`) and model cards.
   * `docs/`: Comprehensive architecture, research plans, and operational documentation.
2. **Package Manager**: Use `pnpm` (`pnpm-workspace.yaml`) for web workspaces due to strict dependency isolation, content-addressable storage, and fast deterministic installs.
3. **TypeScript**: Strict mode enabled across all packages (`noImplicitAny`, `strictNullChecks`, `exactOptionalPropertyTypes`).
4. **Python Tooling**: Python >= 3.12 with standard `pyproject.toml` configuration, `ruff` for linting/formatting, `pytest` for ML pipeline testing.

## Consequences
* Clean decoupling of client browser code from Python machine learning scripts.
* Shared types ensure zero drift between ML export schemas and browser runtime deserialization.
* Local installs are fast, deterministic, and easily validated via CI/CD.
