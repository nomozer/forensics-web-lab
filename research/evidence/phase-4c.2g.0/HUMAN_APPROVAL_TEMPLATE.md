# Human Authorization Template & Checklist (Phase 4C.2G)

> [!WARNING]
> This document is a template and guideline for human research governance review.
> **DO NOT** execute evaluation or unseal locked-test until a designated human reviewer completes and signs the formal authorization artifact `HUMAN_UNSEALING_AUTHORIZATION.json` adhering to `docs/schemas/human-unsealing-authorization.v1.schema.json`.

---

## 1. Governance Review Checklist (All 10 Points Required)

Before authorizing locked-test unsealing, the human reviewer must verify:

1. [ ] **Protocol Review**: Reviewed preregistered protocol in Phase 4C.2E (`research/evidence/phase-4c.2e/`).
2. [ ] **Evaluator Review**: Reviewed evaluator implementation in Phase 4C.2F.2 (`ml/evaluation/locked_test_evaluator.py`, `confirmatory_metrics.py`).
3. [ ] **Execution Package Commit**: Confirmed `execution_package_commit` matches exact Git HEAD and working tree is completely clean.
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
    "42": "26038e788bc5fca39d67566d5885c3dbb9cf19a4e32d56a3103fe319d672ea4c",
    "1337": "53086eb0717208d1f855d045fb9f237efb99e71ec912ba09756b3fa20ae77196",
    "2025": "c644d6786a345517f8a70a8039775080fb0ae9fa2f33f1fe904033c5e88e7be1",
    "3407": "f444c4fae0a2ea9c98efd4ba2195f001cbe65b2ea24a259c15b169543e55c3c0",
    "9001": "d79ab0b606fbf42442cf282d02c7717466542718ef0c36b8e210543666b4c10c"
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
  "execution_package_commit": "<EXACT_40_CHAR_TERMINAL_SEAL_COMMIT>",
  "execution_package_tree_clean": true
}
```
