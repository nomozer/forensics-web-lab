# Phase 4C.1 Colab Execution Guide

## Prerequisites

1. **Google Colab Pro/Pro+** recommended for GPU access (T4/A100)
2. **Google Drive** with sufficient space (~10 GB for bundle + artifacts)
3. **GitHub Personal Access Token** with `repo` scope for private repo access
3. **Local bundle creation** (run export script locally first)

---

## Step 1: Create Bundle Locally

```bash
cd forensics-web-lab
python ml/datasets/export_phase_4c1_bundle.py --execute --output ./phase_4c1_bundle
```

This creates `./phase_4c1_bundle/` with:
- 4 Option P archives (5.88 GB total)
- `manifest_pilot_a_option_p.csv`
- `checkpoint-receipt.json`
- `pilot_a_binary_preregistered.yaml`

Upload the entire `phase_4c1_bundle/` folder to Google Drive at:
`MyDrive/forensics-web-lab/phase_4c1_bundle/`

---

## Step 2: Configure Colab Secrets

1. Open Colab notebook: `notebooks/phase_4c1_learning_curve_colab.ipynb`
2. Click **Settings** (gear icon) → **Secrets**
3. Add secret:
   - Name: `GITHUB_TOKEN`
   - Value: Your GitHub PAT with `repo` scope
4. Enable "Notebook access" for this secret

---

## Step 3: Open & Run Notebook

1. Open the notebook in Colab:
   - File → Open notebook → GitHub → nomozer/forensics-web-lab → notebooks/phase_4c1_learning_curve_colab.ipynb
2. **Runtime → Change runtime type** → GPU → T4 (or A100 if available)
3. Run cells sequentially:
   - Cell 1: Environment check
   - Cell 2: Mount Drive & load token
   - Cell 3: Clone repo at exact commit
   - Cell 4: Install dependencies
   - Cell 5: Mount Drive & copy bundle
   - Cell 6: Validate bundle
   - Cell 7: Dry-run matrix (verify everything)
   - Cell 8: **Set EXECUTE = True to enable training**
   - Cell 9-11: Training execution (only if EXECUTE=True)

---

## Configuration

### Key Parameters (in `ml/configs/phase_4c1_learning_curve.yaml`)

| Parameter | Value |
|---|---|
| Sample sizes | N=50, 100, 250 |
| Seeds | [42, 1337, 2025, 3407, 9001] |
| Stage 1 runs | 15 (3 sizes × 5 seeds) |
| Baselines | 6 × 15 = 90 runs |
| Stage 2 | Conditional (gated) |
| Locked test | SEALED |

### Gating Rules

- **Smoke run first**: N=50, seed=42, Stage 1
- **Stage 2 gate**: Stage 1 Macro-F1 > Dummy AND > Metadata-Only on inner_validation
- **Locked test**: SEALED (0 evaluations until separate approval)

---

## Artifact Persistence

All outputs saved to Google Drive:
```
MyDrive/forensics-web-lab/phase_4c1_artifacts/
├── checkpoints/          # .pt files
├── metrics/              # JSON/CSV metrics
├── logs/                 # Training logs
├── bootstrap/            # Bootstrap CI results
└── run_registry.json     # Resume tracker
```

---

## Resume Capability

The notebook supports resume via `run_registry.json`:
- Tracks completed (size, seed, stage) tuples
- Skips completed runs on re-run
- Run `EXECUTE = True` again to resume

---

## Monitoring

Monitor training in real-time:
- **Colab output**: Live logs
- **TensorBoard**: `%load_ext tensorboard && %tensorboard --logdir /content/drive/MyDrive/forensics-web-lab/phase_4c1_artifacts/logs`
- **Drive**: Check artifact folder for checkpoints/metrics

---

## Troubleshooting

| Issue | Solution |
|---|---|
| GPU OOM | Reduce batch_size in config (32 → 16) |
| Drive quota | Request more space or clean old artifacts |
| Private repo access | Verify GITHUB_TOKEN has `repo` scope |
| Bundle missing | Re-run export script and re-upload |
| CUDA OOM on T4 | Use gradient accumulation (accumulation_steps=2) |

---

## Next Steps After Training

1. **Statistical analysis**: Run bootstrap CI on completed runs
2. **Stage 2 gate**: Evaluate if Stage 1 > Dummy AND > Metadata-only
3. **Locked-test evaluation**: Separate approval required
4. **Report generation**: Compile results for Phase 4C.1B report

---

## Safety Checklist

- [ ] Bundle uploaded to Drive
- [ ] GITHUB_TOKEN in Colab Secrets
- [ ] EXECUTE = False (dry-run first)
- [ ] Drive has 10+ GB free
- [ ] GPU runtime selected
- [ ] Notebook copied to own Drive (File → Save copy in Drive)