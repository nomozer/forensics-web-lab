# Data Licenses & Intellectual Property Audit

This document tracks the licensing terms, academic attribution requirements, and usage restrictions for all image datasets referenced by **Forensics Web Lab**.

## 1. Regulatory Overview

Forensics Web Lab is conducted strictly as non-commercial scientific research into digital provenance and media verification.

### Core Principles
1. **Academic Attribution**: Every dataset utilized must receive full academic citation in publications, technical reports, and model cards.
2. **License Compliance**: Datasets bearing `Non-Commercial` (NC) restrictions must never be redistributed or bundled into commercial offerings.
3. **No Direct Hosting**: The repository hosts only download automation adapters and manifest parsers. We do not re-host or distribute raw copyrighted image archives.

---

## 2. Dataset License Registry

| Dataset | Primary Authors / Institution | Declared License | Usage Restriction | Commercial Use Allowed? |
| :--- | :--- | :--- | :--- | :--- |
| **GenImage** | Zhu et al. (NeurIPS 2023) | CC-BY-NC 4.0 | Academic & Research Only | No |
| **SAGI-D** | Open Inpainting Research | CC-BY 4.0 / Academic | Research & Benchmark | License specific |
| **RealHD** | Deepfake Forensics Consortium | Research Use License | Non-commercial evaluation | No |
| **RAID Benchmark** | Academic consortium | Open Academic License | Evaluation only | No |
| **COCO (MS-COCO)** | Microsoft / COCO Consortium | CC-BY 4.0 | Non-commercial research | Dependent on images |
| **RAISE** | Dang-Nguyen et al. | Research License | Scientific research only | No |
| **ImageNet (Val)** | Stanford / Princeton | Research and Educational | Educational & Research | No |

---

## 3. Human Gatekeeper Protocol for Data Download

Whenever automated scripts in `ml/datasets/` are invoked to retrieve data:
1. Verify license compatibility with current research objectives.
2. Display the formal terms to the terminal operator.
3. Request explicit typed confirmation before streaming network bytes.
4. Record download audit trail in `research/experiments/data_access_log.jsonl`.
