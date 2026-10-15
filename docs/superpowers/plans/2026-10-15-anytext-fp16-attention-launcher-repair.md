# AnyText fp16 Attention Repair + Launcher Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** Remove the verified cause of `expected scalar type Half but found Float` in the AnyText Colab path and replace the 7-cell, payload-embedding notebook with a short launcher that fetches hash-verified, revision-pinned sources.

**Spec:** user request of 2026-10-15 (AnyText fp16 repair + launcher); `research/evidence/anytext_downloader_local/HANDOFF_ANYTEXT_DOWNLOADER.md`.

## Verified root cause (systematic-debugging Phase 1-3)

Reproduced LOCALLY with the real `ms_wrapper.py` (ModelScope v1.1.3, 25,783 B), the real runner `run_anytext_inference`, upstream code at `3d4a260c`, torch 2.0.1+cu118, **random-initialised weights** (dtype errors do not depend on weight values). Isolated Python 3.10 venv, no checkpoint downloaded.

| Run | Change vs previous (one variable) | Result |
| :-- | :-- | :-- |
| 1 | Colab-equivalent deps (no xformers) | `RuntimeError: expected scalar type Half but found Float` at `ldm/modules/attention.py:192` `einsum('b i j, b j d -> b i d', sim, v)`, inside `ControlNet.forward` -> `input_blocks` (first DDIM step) |
| 2 | `ATTN_PRECISION=fp16` | inference completes (diagnostic only) |
| 3 | `pip install xformers==0.0.20`, no env override | inference completes; `XFORMERS_IS_AVAILBLE=True` |

Mechanism: `CrossAttention.forward` upcasts q,k to fp32 when `_ATTN_PRECISION == "fp32"` (default), so `sim` is fp32 while `v` is fp16 under `model.half()` without autocast. Upstream `environment.yaml` pins `xformers==0.0.20`; with it `BasicTransformerBlock` selects `MemoryEfficientCrossAttention`, which has no upcast. The Colab environment omitted xformers. **Not verified:** the original Colab traceback itself (artifacts are not on LOCAL); real-weight image quality; xformers 0.0.20 wheel behaviour on the Colab GPU (verified only on Windows GTX 1650).

## Precision policy

Unchanged: fp16 weights/activations, no autocast, same checkpoint/prompt/seed/scheduler/steps. Only change: install upstream-pinned `xformers==0.0.20` so the attention implementation matches upstream's reference environment. The runner fails closed if xformers attention is not active (no silent fall-back to an env-var override).

## Tasks

1. **Single dependency source** `scripts/research/anytext_requirements.txt` (adds `xformers==0.0.20`, `diffusers==0.10.2`, `huggingface_hub==0.25.2`, `sentencepiece==0.1.99`, keeps `ujson==5.9.0`).
2. **Setup/launch module** `scripts/research/anytext_colab_setup.py`: `bootstrap` verification of source hashes, `env` (isolated py3.10, pinned torch, requirements, `pip check`), `smoke` (venv interpreter: full inference import graph, glyph generation, CUDA fp16 xformers attention + `BasicTransformerBlock` class check), `run` (spawn runner, concise console, full log), `result` (locate the current run's archive).
3. **Runner**: fail-closed attention precondition; structured failure receipts (stage, attempt id, traceback, failing frame, tensor dtype/device, module parameter dtypes); config fingerprint required for attempt reuse; per-run archive name + `latest_run.json`; stop deleting failure receipts; `--run-label`.
4. **Builder + notebook**: 1 markdown + 4 short code cells, source pinned to a full commit SHA (builder refuses a SHA whose blobs differ from the working tree), no embedded payloads.
5. **Tests** (behavioural): real half/float einsum failure produces the diagnostic receipt; fingerprint reuse; stream filter; archive selection; builder pin verification. Replace the embedded-payload tests that no longer apply.
6. **Docs/evidence**: continuity files, repro evidence (`research/evidence/anytext_fp16_repair/`), handoff update. Run `pnpm continuity:check`.
7. **Commit + push** on `research/independent-cohort-acquisition`; notebook pins the pushed code commit.

## Review Focus
- Stale ZIP from an older run chosen for download.
- Old failure receipts deleted/overwritten by the repair run.
- Attempt reused although config/source changed.
- xformers wheel missing on the Colab GPU: must stop with a clear message, not fall back.
