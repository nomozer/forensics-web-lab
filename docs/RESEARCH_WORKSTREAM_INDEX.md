# Research workstream index

Use descriptive workstream names in current guides, notebook titles, runner help, and new storage paths. Phase codes remain historical trace identifiers in evidence metadata and `STATUS_LEDGER.md`; sealed artifacts and historical run paths are never renamed.

| Phase trace | Workstream / job | Primary documentation | Runner / notebook |
|---|---|---|---|
| `4C.7A` | Independent validation preparation: endpoint lock, model bindings, bootstrap/evaluator preflight | `research/evidence/phase-4c.7a/PHASE_REPORT.md`, `ml/configs/independent_validation_protocol.yaml` | `ml/evaluation/independent_evaluator.py` (preflight only until a human-approved cohort exists) |
| `4C.7B` | Independent cohort acquisition: licensed source catalog, content-grounded edit plan, production inpainting, Technical QC, and Human Content QC handoff | `research/evidence/phase-4c.7b/PHASE_REPORT.md`, `research/evidence/phase-4c.7b/CONTENT_GROUNDED_EDITING_AMENDMENT.md` | `scripts/research/run_cohort_acquisition.py`, `notebooks/independent_cohort_acquisition_colab.ipynb` |

## Storage naming

New acquisition runs use:

`MyDrive/Colab Notebooks/forensics-web-lab/independent_cohort_acquisition/runs/<RUN_ID>`

Historical paths remain valid and must be supplied as their original `--output-root` when resumed or audited:

- `pilot-20261007T093824Z`: `MyDrive/forensics-web-lab/phase_4c7b_runs/`
- `pilot-20261007T132003Z`: `MyDrive/Colab Notebooks/forensics-web-lab/phase_4c7b/runs/`

Do not move either historical run automatically. Its binding contains the original code, protocol, catalog, plan, and run identifiers.

## Terminology

- **Allocation plan**: the 440 licensed/disjoint source candidates plus locked source/tool/modification/mask quotas. It does not authorize generation by itself.
- **Content-grounded edit plan**: reviewed prompt, target/placement description, and normalized-canvas geometry for a bounded set of attempts.
- **Technical QC**: machine-checkable dimensions, modes, binary mask, mask-area class, non-blank image, masked change, and outside-mask invariance. It is not semantic approval.
- **Human Content QC**: explicit human review of prompt/operation match and visible edit quality. Agent observations never satisfy this gate.
