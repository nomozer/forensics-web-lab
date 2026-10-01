# Human Authorization Template & Checklist (Phase 4C.2G)

> [!WARNING]
> This document is a template and guideline for human research governance review.
> **DO NOT** execute evaluation or unseal locked-test until a designated human reviewer completes and signs the formal authorization artifact `HUMAN_UNSEALING_AUTHORIZATION.json` adhering to `docs/schemas/human-unsealing-authorization.v1.schema.json`.

---

## 1. Governance Review Checklist (All 10 Points Required)

Before authorizing locked-test unsealing, the human reviewer must verify:

1. [ ] **Protocol Review**: Reviewed preregistered protocol in Phase 4C.2E (`research/evidence/phase-4c.2e/`).
2. [ ] **Evaluator Review**: Reviewed evaluator implementation frozen in Phase 4C.2G.0 at commit `3cf75c2bf0c9835dd58897b7b36982732cab40ab` (`ml/evaluation/locked_test_evaluator.py`, `confirmatory_metrics.py`).
3. [ ] **Execution Package Commit**: Confirmed `execution_package_commit` matches exact sealed commit `2826a8274cb89ec548d6fac5c8ae50c1c2836202`, and verified preferred execution mode (clean detached Git worktree/checkout at `2826a8274cb89ec548d6fac5c8ae50c1c2836202` with clean working tree).
4. [ ] **Single Unsealing Session**: Agreed to strictly **1** unsealing session (`maximum_unsealing_sessions = 1`).
5. [ ] **Attempt Accounting**: Agreed to strictly **5** model evaluation attempts (`maximum_model_evaluation_attempts = 5`, 1 attempt per seed).
6. [ ] **No Post-Unsealing Tuning**: Acknowledged that zero tuning, hyperparameter search, or prompt adjustment may take place after unsealing.
7. [ ] **No Silent Retries**: Acknowledged that if an evaluation attempt fails or crashes post-reservation, no silent retries are permitted; the attempt is counted as consumed.
8. [ ] **Confirmatory Decision Rule**: Agreed to the exact preregistered decision criterion: **95% Percentile Bootstrap CI lower bound > 0.5000** (`balanced_binary_uninformative_reference`).
9. [ ] **Unfavorable Outcome Reporting**: Understood and agreed that unfavorable, inconclusive, or negative results **MUST** be transparently reported; locked-test evaluation is prospective and immutable.
10. [ ] **Phase 4C.2G.1 Authorization**: Explicitly authorized transition to Phase 4C.2G.1 under strict offline air-gapped runtime conditions.

---

## 2. Artifact Creation Instructions

To authorize execution, the human reviewer creates the artifact:
`HUMAN_UNSEALING_AUTHORIZATION.json` (outside Git, or in designated authorization path)

Structure required by schema `docs/schemas/human-unsealing-authorization.v1.schema.json`:

```json
{
  "status": "AUTHORIZED",
  "authorization_id": "auth-2026-phase4c2g-001",
  "authorized_by": "<Human Reviewer Full Name and Title>",
  "authorized_at_utc": "<ISO-8601 Timestamp>",
  "candidate_protocol": "stage1_frozen_backbone_linear_probe",
  "sample_size": 250,
  "exact_seeds": [42, 1337, 2025, 3407, 9001],
  "checkpoint_sha256s": {
    "42": "c941f42ed00adb098962ddb43c0f2f7d897e48e20957cddc0c491243f8730c92",
    "1337": "69e706f9062f050ccbfd2affb72d63d5dde9d12f02c1e64c762b0ea6719073f1",
    "2025": "92ce5ee986d487fdf14374842067d0684fc7669f5ecb32c7d603f6813942ceee",
    "3407": "4897821ef0a97df9f1c51acde8d4ac01d383aa3a7b9e7ee55bbc3b3108847868",
    "9001": "5f0f8803adcb7eec88d47e398ef8b2002b46e740c263b12abc2ba392822a91c3"
  },
  "evaluator_effective_commit": "3cf75c2bf0c9835dd58897b7b36982732cab40ab",
  "evaluator_component_hashes": {
    "confirmatory_metrics": "bacbb9dc0a230359f1c3bbaece1ce8c19e30ddafa807313480d173a808bde808",
    "locked_test_evaluator": "b311e010535600781e52014fb6b675d32d78a558c5fe7e67bd3ec00b6989a5e1",
    "run_evaluator_cli": "37ed02ff44d8c523150ebb968d2777bba9c2dee8597b3df60efa1fb5bd1e9041"
  },
  "maximum_unsealing_sessions": 1,
  "maximum_model_evaluation_attempts": 5,
  "expiry_policy": {
    "policy": "single_session_only"
  },
  "authorization_purpose": "Confirmatory prospective evaluation on locked-test partition.",
  "no_tuning_acknowledgment": true,
  "execution_package_commit": "2826a8274cb89ec548d6fac5c8ae50c1c2836202",
  "execution_package_tree_clean": true
}
```
