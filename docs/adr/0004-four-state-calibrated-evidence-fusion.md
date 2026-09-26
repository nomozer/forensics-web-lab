# ADR 0004: Four-State Calibrated Evidence Fusion and Epistemic Uncertainty

## Status
Accepted

## Context
Standard binary classification ("Real vs AI") is epistemically flawed in digital forensics:
1. "Real" is an unfalsifiable claim. Failing to find known synthetic artifacts only proves that no detectable AI footprint was identified with current tools; it does not verify absolute historical truth.
2. In-the-wild images frequently undergo benign transformations (social media re-compression, artistic post-processing, screenshotting, resizing) that attenuate synthetic high-frequency signatures.
3. Localized AI edits (inpainting, background replacement, face retouching) must be distinguished from entirely generated images.
4. Independent forensic analyzers (metadata, C2PA, classical frequency/block statistics, deep neural networks) can yield conflicting signals.

## Decision
1. **Four Canonical Verdict States**:
   * `no_ai_evidence`: No clear indicators of generative AI synthesis or local manipulation found within the detector's capability.
   * `fully_generated`: Strong, coherent multi-signal evidence that the image was synthesized end-to-end by a generative model.
   * `ai_edited`: Evidence that specific localized spatial regions were synthesized, inpainted, or modified using AI techniques, while the remainder exhibits organic or traditional camera noise patterns.
   * `uncertain`: Signals are contradictory, confidence is below calibrated acceptance thresholds, or severe degradation renders reliable forensic inference impossible.

2. **Calibrated Evidence Fusion Formula**:
   * **Stage 1 (Probability Calibration)**: Deep model logits are scaled via Temperature Scaling $T > 0$ fitted on held-out validation data.
   * **Stage 2 (Signal Aggregation)**:
     Let $P_{\text{model}} = [p_{\text{none}}, p_{\text{gen}}, p_{\text{edit}}]$ be calibrated model probabilities.
     Let $S_{\text{patch}}$ be the suspicious patch area ratio ($0.0 \le S_{\text{patch}} \le 1.0$) with peak score $M_{\text{patch}}$.
     Let $F_{\text{freq}}$ be the high-frequency spectral anomaly score ($0.0 \le F_{\text{freq}} \le 1.0$).
     Let $F_{\text{noise}}$ be the localized noise inconsistency variance score.
     Let $C_{\text{provenance}}$ be C2PA / generator metadata modifier (if generator tag found, sets prior strongly towards generated/edited).
   * **Stage 3 (Conflict & Uncertainty Thresholding)**:
     * If $\max(P_{\text{model}}) < \tau_{\text{conf}}$ (default $\tau_{\text{conf}} = 0.65$), flag as `uncertain`.
     * If $p_{\text{gen}} \approx p_{\text{none}}$ with margin $< \delta_{\text{margin}}$ (default $0.15$), flag as `uncertain`.
     * If global model predicts `no_ai_evidence` ($p_{\text{none}} > 0.85$) BUT local patch scan discovers a contiguous dense region with $M_{\text{patch}} > 0.80$ and area ratio $\in [0.03, 0.40]$, classify as `ai_edited`.
     * If C2PA cryptographic signature indicates tampering or revocation, flag in provenance report and lower overall confidence.
     * When quality factors indicate extreme degradation (e.g. dimensions $< 256\text{ px}$ or severe multi-pass compression), widen the `uncertain` band.

3. **Strict Ban on False Absolutes**:
   * The user interface and exported reports must NEVER render the phrase "Real Image" or "Authentic Photo" as an absolute forensic conclusion.
   * The label is invariably rendered as `Chưa tìm thấy bằng chứng AI (no_ai_evidence)`.

## Consequences
* High forensic integrity: prevents false accusations on degraded or traditionally edited media.
* Transparent confidence calibration with explicit rationale in the forensic report.
