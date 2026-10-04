# Phase 4C.2G.0 Final Report: Execution Package Preparation, Offline Runtime Readiness, and Human Authorization Request

## 1. Executive Summary

Phase 4C.2G.0 establishes the final sealed execution package, formalizes offline runtime constraints, and prepares the human unsealing authorization request for the locked-test confirmatory evaluation.

In accordance with strict research governance invariants:
- **Locked-Test Access**: ZERO accesses to the locked-test partition (`locked_test_real_accesses = 0`).
- **Unsealing Sessions**: ZERO unsealing sessions executed (`completed_real_unsealing_sessions = 0`).
- **Model Inferences**: ZERO real model evaluations executed (`completed_real_model_evaluations = 0`, `evaluation_attempts = 0`).
- **GPU Calls & Training**: ZERO GPU inference calls, zero training runs.
- **Authorization Artifact**: Strictly NO `HUMAN_UNSEALING_AUTHORIZATION.json` with status `AUTHORIZED` in Git or local storage. `HUMAN_AUTHORIZATION_REQUEST.json` is created strictly in `PENDING_HUMAN_APPROVAL` status.

---

## 2. Commit Lineage and Sealed Execution Package

| Role | Commit / Identifier | Description |
| :--- | :--- | :--- |
| **Base Main Commit** | `8a379665bc8db3722a46618e47db4806a6ea7244` | Candidate selection and protocol baseline |
| **Parent Phase 4C.2F Commit** | `35f430f410c77247b1e8bb9bb9559ac615b0cd5b` | Sealed Phase 4C.2F.2 evaluator and evidence |
| **Final Effective Evaluator Commit** | `3cf75c2bf0c9835dd58897b7b36982732cab40ab` | Evaluator source code frozen with execution package commit verification |
| **Execution Package Commit** | `2826a8274cb89ec548d6fac5c8ae50c1c2836202` | Terminal evidence seal commit containing sealed package, schema, and contracts |
| **Current Working Branch** | `research/phase-4c2g-locked-test-execution` | Isolated evaluation execution branch |

Evaluator source code was frozen at commit `3cf75c2bf0c9835dd58897b7b36982732cab40ab`. No evaluator source modifications have been or will be made following that commit.

### Preferred Execution Mode
- Create a clean detached Git worktree or checkout at exact execution package commit: `2826a8274cb89ec548d6fac5c8ae50c1c2836202`.
- Verify working tree is clean via `git status --porcelain` (must be completely empty).
- Verify three evaluator components match sealed SHA-256 digests.
- Do NOT run directly from the HEAD of evidence hotfix commits.
- The 29,823-byte archive is a delivery and backup sealed artifact; it does not replace checking the exact execution-package commit.

---

## 3. Component Integrity & Cryptographic Registry

| Component | Relative Path | Bytes | SHA-256 |
| :--- | :--- | :--- | :--- |
| **Confirmatory Metrics** | `ml/evaluation/confirmatory_metrics.py` | 15,199 | `bacbb9dc0a230359f1c3bbaece1ce8c19e30ddafa807313480d173a808bde808` |
| **Locked Test Evaluator** | `ml/evaluation/locked_test_evaluator.py` | 47,708 | `b311e010535600781e52014fb6b675d32d78a558c5fe7e67bd3ec00b6989a5e1` |
| **Evaluator CLI Wrapper** | `ml/evaluation/run_phase_4c2f_evaluator.py` | 5,739 | `37ed02ff44d8c523150ebb968d2777bba9c2dee8597b3df60efa1fb5bd1e9041` |
| **Authorization Schema** | `docs/schemas/human-unsealing-authorization.v1.schema.json` | 6,143 | `15291643b2ca2b3d6e3c135957db140e33f7cc65bc11aebc4668f3e105af6734` |
| **Candidate Checkpoints Binding** | `research/evidence/phase-4c.2e/candidate_checkpoint_binding.json` | 3,519 | `b1363cb69e393927c6e2340a38e10865d16749939c1c391b28f2842666b31cfb` |

### Canonical Candidate Checkpoints (Phase 4C.2E Preregistered Lineage)
- **Seed 42**: `c941f42ed00adb098962ddb43c0f2f7d897e48e20957cddc0c491243f8730c92`
- **Seed 1337**: `69e706f9062f050ccbfd2affb72d63d5dde9d12f02c1e64c762b0ea6719073f1`
- **Seed 2025**: `92ce5ee986d487fdf14374842067d0684fc7669f5ecb32c7d603f6813942ceee`
- **Seed 3407**: `4897821ef0a97df9f1c51acde8d4ac01d383aa3a7b9e7ee55bbc3b3108847868`
- **Seed 9001**: `5f0f8803adcb7eec88d47e398ef8b2002b46e740c263b12abc2ba392822a91c3`

---

## 4. Offline Runtime Readiness & Device Policy

1. **Air-Gap Verification**:
   - Evaluator includes passive inspection verifying: zero default routes outside loopback, absence of `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, absence of active DNS queries, and zero outbound socket probes.
   - Tested and verified via synthetic fixture (`test_phase_4c2g_preparation.py`).
   - Current environment is a development host; `actual_execution_network_isolation` is recorded as `NOT_YET_VERIFIED`.
   - Read-only mount verification must be strictly non-mutating (statvfs, metadata, disjoint realpath); canary writes into the locked-test partition are strictly forbidden.
2. **Device Independence & Determinism**:
   - Evaluator supports both CPU and CUDA inference paths.
   - CPU được ưu tiên để giảm phụ thuộc CUDA và cải thiện khả năng tái lập trong một runtime đã niêm phong. Bitwise parity giữa các host, BLAS, PyTorch hoặc kiến trúc CPU khác nhau không được giả định nếu chưa được kiểm chứng.
   - Preprocessing, thresholding, and metric logic are strictly invariant to device selection.

---

## 5. Human Authorization Request & Approval Requirements

The formal request artifact `research/evidence/phase-4c.2g.0/HUMAN_AUTHORIZATION_REQUEST.json` is generated with `status: PENDING_HUMAN_APPROVAL`.

Before authorizing execution in Phase 4C.2G.1, the human reviewer must verify all 10 governance criteria:
1. Protocol review (Phase 4C.2E).
2. Evaluator code review (frozen commit `3cf75c2bf0c9835dd58897b7b36982732cab40ab`).
3. Execution package commit verification (`execution_package_commit: 2826a8274cb89ec548d6fac5c8ae50c1c2836202` matching Git HEAD and clean working tree).
4. Strictly 1 unsealing session.
5. Strictly 5 model evaluation attempts (1 per seed).
6. Zero post-unsealing tuning.
7. No silent retries upon reservation failure.
8. Preregistered decision criterion: **95% Percentile Bootstrap CI lower bound > 0.5000**.
9. Unfavorable results reporting obligation.
10. Explicit offline air-gapped runtime authorization.

---

## 6. Pre-Authorization Go/No-Go Verdict

- **Verdict**: `RUNTIME_PREPARATION_REQUIRED`
- **Rationale**: While the execution package, evaluator code, test suite (60/60 passing), and authorization request are fully prepared and sealed, physical network isolation on the execution host has not yet been established. Execution must pause until the offline runtime is verified and explicit human authorization is granted.
