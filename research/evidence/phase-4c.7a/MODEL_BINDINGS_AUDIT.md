# Candidate Model Bindings Audit Report: Phase 4C.7A

> **Audit Status**: PASSED_ALL_VERIFIED<br>
> **Timestamp (UTC)**: 2026-10-06T14:30:00Z<br>
> **Source Experiment**: Phase 4C.6B Controlled DSP Augmentation<br>

---

## 1. Kết Quả Đối Soát Five Outer-Fold Models

| Outer Fold | Kích thước (bytes) | SHA-256 Model | Receipt Cross-Check | Trạng thái |
| :---: | :---: | :--- | :---: | :---: |
| Fold 0 | 58418 | `232d0994775a6e6cfdfb1bb3fd79f28557439e372e15e4760d0ffbbdba72f609` | MATCH | **VERIFIED** |
| Fold 1 | 58030 | `c07c582322f7671f3b9f352f5d5388f5a39cec5fa11c80d8441cbb366a3fc15f` | MATCH | **VERIFIED** |
| Fold 2 | 58465 | `8a979e35339f1ab2728f6c96d94ce2ef828ad13aad66e0e3ae8339a1be8fb324` | MATCH | **VERIFIED** |
| Fold 3 | 58455 | `60358c2fbdfd54d96dee68a1ce43106560ee4da5792343c0d14d85beb1d5fca3` | MATCH | **VERIFIED** |
| Fold 4 | 58390 | `efe517db09c5aecdd219eabc878b6f538e908c03b8b8c0648b2d0f6ec4356a6c` | MATCH | **VERIFIED** |

---

## 2. Đối Chiếu Hợp Đồng Đặc Trưng (Feature Contracts)

- **Visual Modality**:
  * Architecture: MobileNetV3-Small (penultimate avgpool)
  * Dimension: 576
  * Pretrained weights SHA-256: `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`
- **DSP Modality**:
  * Source module: `ml/training/dsp_features.py`
  * Extractor: `extract_dsp_features`
  * Dimension: 16
  * Feature order (16 dimensions):
    1. `fft_high_freq_energy_ratio`
    2. `fft_spectral_peak_count`
    3. `fft_radial_anomaly_score`
    4. `fft_spectral_decay_slope`
    5. `dct_mean_high_freq_energy`
    6. `dct_ac_energy_variance`
    7. `dct_anomaly_score`
    8. `noise_global_level`
    9. `noise_max_to_min_ratio`
    10. `noise_std_of_variances`
    11. `noise_coef_variation`
    12. `noise_inconsistency_score`
    13. `jpeg_boundary_difference_ratio`
    14. `jpeg_periodic_grid_strength`
    15. `jpeg_artifact_score`
    16. `laplacian_variance`

---

## 3. Giải Trình Sự Khác Biệt Giữa Tin Nhắn Báo Cáo và File Lưu Trữ

The SHA-256 hashes in 'candidate_model_bindings.json' match 100% byte-for-byte with the physical fold_model.json files produced by Phase 4C.6B. The alternate hashes appearing in the prior conversation chat summary text (e.g., outer_0 starting with e5c1d6...) were hallucinated text in the Markdown response and never existed in any physical file, git commit, or receipt in the repository.

**Kết luận**: File `candidate_model_bindings.json` và code nạp model `ml/evaluation/independent_model_bindings.py` hoàn toàn chính xác và nhất quán với các file mô hình vật lý trên đĩa. Không có bất kỳ refit hay can thiệp nào.
