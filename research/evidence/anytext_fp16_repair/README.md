# AnyText fp16 attention repair - evidence

Written for: the researcher continuing the AnyText document feasibility pilot.

## What was established, and how

| Claim | Evidence | Status |
| :-- | :-- | :-- |
| Two Colab attempts started inference and failed with `expected scalar type Half but found Float` | User report only. Colab failure receipts/logs are **not on LOCAL**, so the original traceback is not preserved here. | reported, not verified |
| The error is reproducible with the real wrapper | `repro_logs/run1_no_xformers_FAIL_half_float.log`: `ldm/modules/attention.py:192 einsum('b i j, b j d -> b i d', sim, v)` inside ControlNet input blocks, first DDIM step | verified LOCAL |
| Root cause = fp32 upcast in `CrossAttention` (no xformers) vs fp16 `v` | read of upstream `attention.py` at `3d4a260c` + runs 1-3 (one variable each) | verified LOCAL |
| `xformers==0.0.20` (upstream `environment.yaml`) removes the failure without a precision-policy change | `repro_logs/run3_xformers_0.0.20_PASS.log` | verified LOCAL |
| `ATTN_PRECISION=fp16` alone also passes | `repro_logs/run2_attn_precision_env_diagnostic_PASS.log` | diagnostic only, NOT used (it would change the policy) |
| A second bug (`KeyError: target_bbox_in_canvas`) would have occurred right after successful inference | `local_stub_e2e_run1_attemptA_failure_receipt_KeyError.json`, `repro_logs/local_stub_e2e_run1_before_bbox_fix.log` | verified LOCAL, fixed |
| Fixed pipeline mechanics (both attempts, outside-mask diff 0, run-specific archive, reuse) | `local_stub_e2e_run2_COMPLETE_run_manifest.json` | verified LOCAL with random weights |

## Method and limits

- Environment: isolated Python 3.10.19 venv, torch 2.0.1+cu118, packages in `local_windows_pip_freeze.txt` (all `scripts/research/anytext_requirements.txt` pins match; `pip check` clean). Windows 11, GeForce GTX 1650 4 GB.
- Weights are **randomly initialised** (`repro_make_random_stub.py`); the real `ms_wrapper.py` (SHA-256 `ed6a6059...`) and `configuration.json` and real CLIP tokenizer files from ModelScope v1.1.3 were used. No checkpoint was downloaded. Dtype errors do not depend on weight values; image content from these runs is noise and has **no scientific value**.
- `peak_vram_mb` in the logs (~6.6 GB) exceeds the 4 GB card because Windows spilled to shared memory; it is not a Colab measurement.
- Not verified: the original Colab traceback; `xformers==0.0.20` on the Colab GPU/driver; real-weight output quality; Human Content QC (`PENDING_HUMAN_REVIEW`).
- Reproduce: `repro_make_random_stub.py` then `repro_infer.py <stub_model_dir>` with cwd = AnyText checkout at `3d4a260c67d92c0e27d6e18ed69990e19a40de6d` and `PYTHONPATH=.`.
