# Phase 4C.2G.0.13 — Authorized Recovery Evaluator Session

## Outcome

`LOCKED_TEST_CONFIRMATORY_EVALUATION_COMPLETE_INSUFFICIENT_CONFIRMATORY_EVIDENCE`

The directly approved recovery authorization was materialized without claiming a cryptographic signature and consumed by exactly one recovery evaluator session. The exact `0658dce` package ran on CPU against the preserved read-only VHDX. All five preregistered Stage 1 N=250 checkpoints completed once, in canonical seed order, with no retry, tuning, training, calibration fitting, best-seed selection, or ensemble.

## Confirmatory result

| Seed | Macro-F1 | Balanced accuracy | AUROC | Brier | ECE |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.524055881 | 0.527696793 | 0.542155054 | 0.248549606 | 0.004062124 |
| 1337 | 0.471988306 | 0.508746356 | 0.534666678 | 0.249749445 | 0.029746964 |
| 2025 | 0.530576335 | 0.530612245 | 0.542044556 | 0.248422403 | 0.003241363 |
| 3407 | 0.507791924 | 0.521865889 | 0.537590630 | 0.249095251 | 0.002439887 |
| 9001 | 0.534762985 | 0.534985423 | 0.546557982 | 0.248385477 | 0.012318747 |

Primary arithmetic mean Macro-F1 is `0.513835085980773`. The preregistered 10,000-replicate unique-source cluster bootstrap using `PCG64(20261002)` produced percentile 95% CI `[0.49976282750638307, 0.5273360991798881]`. Because the lower bound is not strictly greater than `0.5000`, confirmatory success is false and the verdict is `INSUFFICIENT_CONFIRMATORY_EVIDENCE`.

## Authorization and counters

- Approved request SHA-256: `52861ac098abd3dc7e6b45d30fa7c887dd921eba355b945b3436261d9c773134`
- Recovery authorization: `CONSUMED`; user identity `dung`; no cryptographic-signature claim.
- Historical authorization/session remains consumed and preserved: 1 interrupted access session, 0 attempts, 0 model evaluations.
- Recovery session: 1 access session, 5 attempts, 5 completed checkpoint evaluations.
- Cumulative history: 2 evaluator access sessions, 5 attempts, 5 completed checkpoint evaluations; no counter reset.

The recovery ledger has 12 valid hash-chained entries. Both output checksum indices pass, all five prediction files contain exactly 686 records, and the reported arithmetic mean and decision rule were independently recomputed.

## Runtime and cleanup

The same VHDX was reused without copying or rebuilding, mounted with OS-enforced read-only status, and used under verified zero-route network isolation. The sealed controller exited 0 and its cleanup receipt reports network restoration and watchdog removal. Independent elevated read-back confirmed two active default routes, Wi-Fi and Radmin VPN both Up, watchdog absent, VHDX detached, and `R:` absent.

The outer custody wrapper exited 1 only because it checked a non-contract filename, `confirmatory_result.json`; the sealed controller correctly published `aggregate_confirmatory_metrics.json` and `confirmatory_decision.json`. This is a wrapper false negative, not a sealed evaluator failure. No retry was performed or permitted.

Raw authorization, manifest, dataset, checkpoints, prediction files, and raw ledgers remain outside Git. Git contains only redacted bindings, hashes, aggregate/per-checkpoint metrics, counters, and cleanup evidence.
