# Phase 4C.2H.1 Colab pilot instructions

The pilot is development-only. It executes exactly two already-budgeted inner fits: one `stage1_frozen` fit and one `stage1_frozen_pair_ranking` fit, both at seed `42`, outer fold `0`, inner fold `0`. It does not access Phase 4C.2G or any locked-test data.

## Upload exact inputs

Upload these files without renaming them:

1. Local `D:\Documents\forensics-web-lab-local-artifacts\phase_4c2h\inputs\phase_4c2h_code_75568d1.tar.gz` to Google Drive at `MyDrive/Colab Notebooks/forensics-web-lab/phase_4c2h/inputs/phase_4c2h_code_75568d1.tar.gz`.
   - bytes: `9,509,956`
   - SHA-256: `2bd393873be3291084f86a329a34566bd9aa46c0efd8aa9f155e47571b132c18`
2. Ensure the existing development archive is at `MyDrive/Colab Notebooks/forensics-web-lab/phase_4c1/inputs/phase_4c1_binary_n250_reusable.tar`.
   - bytes: `724,633,600`
   - SHA-256: `d49a106f0c4991ca8d79776277cbf7331df209157725c438288720dc42226a27`
3. Upload/open `notebooks/phase_4c2h_development_colab.ipynb` in Colab.

The code snapshot already contains the exact pretrained weights. Do not separately download or substitute weights.

## Run the pilot

1. Select a GPU runtime in Colab.
2. Leave `MODE = 'pilot'`, `ALLOW_FULL_150_FITS = False`, and `DEVICE = 'cuda'` unchanged.
3. Run all cells in order.
4. Keep the output directory unchanged: `phase_4c2h/runs/execution_75568d1`. This is the resume namespace.
5. Save the cell output plus the complete Drive output directory after the run.

Expected completed fit IDs:

- `inner__recipe-stage1_frozen__seed-42__outer-0__inner-0`
- `inner__recipe-stage1_frozen_pair_ranking__seed-42__outer-0__inner-0`

A successful pilot must report `completed_inner_fits = 2`, `completed_outer_refits = 0`, `completed_total_fits = 2`, and preserve a valid `fit_receipt.json` plus hash-bound `history.json`, `metrics.json`, and `classifier_checkpoint.pt` for each fit. Record each receipt's `wall_seconds`; feature-cache build time is recorded separately and is not a classifier-fit runtime.

## Resume and later full execution

Rerunning with the same exact inputs and output path validates every completed artifact hash and skips valid fits. Any mismatched receipt or artifact fails closed; do not delete or edit evidence to force a resume.

Only after reviewing pilot runtime/artifacts should a separately intentional full launch set `MODE = 'all'` and `ALLOW_FULL_150_FITS = True`. The two valid pilot fits then count within the locked 120 inner fits; the runner completes the remaining inner fits before any of the 30 outer refits. This phase does not authorize or perform that full launch.
