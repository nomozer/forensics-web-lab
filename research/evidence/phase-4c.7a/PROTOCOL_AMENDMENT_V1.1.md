# Protocol Amendment Version 1.1: Phase 4C.7A

> **Document Class**: Protocol Amendment<br>
> **Predecessor Version**: Protocol v1.0.0 (`independent_validation_protocol.yaml`)<br>
> **Amendment Version**: v1.1.0<br>
> **Date (UTC)**: 2026-10-06T14:45:00Z<br>
> **Phase**: Phase 4C.7A — Independent Validation Preparation<br>
> **Status**: `AMENDMENT_LOCKED_PRE_ACQUISITION`

---

## 1. Lý Do và Phạm Vi Điều Chỉnh

Trong quá trình chuẩn bị kỹ thuật trước khi thu thập dữ liệu độc lập, chúng tôi thực hiện ba điểm chuẩn hóa và làm rõ trong giao thức kiểm định:

1. **Chuẩn hóa Thứ Tự 16 Đặc Trưng DSP (DSP Feature Order Reconciliation)**:
   - *Trước điều chỉnh*: Bảng hợp đồng đặc trưng trong bản phác thảo v1.0.0 ghi nhận danh sách tên tạm thời chưa đồng bộ.
   - *Sau điều chỉnh*: Khóa chặt thứ tự 16 đặc trưng DSP khớp 100% byte-for-byte với hàm trích xuất lịch sử `ml/training/dsp_features.py::DSP_FEATURE_NAMES`:
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
   - *Ảnh hưởng*: Đảm bảo các mô hình phân loại DSP đã đóng băng từ Phase 4C.6B nhận chính xác ma trận đặc trưng tương ứng với weights và scaler.

2. **Khóa Cỡ Mẫu Mục Tiêu Dựa Trên Mô Phỏng Monte Carlo ($N_{\text{target}} = 400$)**:
   - *Trước điều chỉnh*: Luận chứng cỡ mẫu v1.0.0 dựa trên các công thức z-test xấp xỉ giả định phương sai tuyến tính.
   - *Sau điều chỉnh*: Thay thế hoàn toàn bằng kết quả mô phỏng Monte Carlo trực tiếp trên cấu trúc ghép cặp cụm nguồn và đúng estimand trung bình số học 5 Macro-F1 (`simulate_sample_size_planning.py`).
   - *Khóa chính thức*:
     * **Cỡ mẫu mục tiêu**: $N_{\text{target}} = 400$ paired sources ($800$ ảnh hợp lệ).
     * **Ngân sách thu thập dự phòng**: $N_{\text{buffer}} = 440$ pairs (+10% cho mẫu lỗi kỹ thuật).
     * **Quy tắc dừng**: Dừng thu thập ngay khi đạt đủ $N_{\text{target}} = 400$ pairs hợp lệ. Tuyệt đối không dừng sớm hoặc thu thập thêm dựa trên kết quả trung gian.

3. **Cơ Chế Source-Disjoint Guard Đa Tầng**:
   - *Bổ sung*: Không chỉ so sánh `source_id`, hệ thống bắt buộc đối soát:
     * `image_sha256` của từng ảnh đối chiếu với toàn bộ 684 historical Option P images (phát hiện ảnh đổi tên).
     * `origin_id` / `instance_id` đối chiếu với metadata lịch sử.
     * Độ phân giải (canvas resolution) giữa authentic và ai_edited của cùng một source phải khớp tuyệt đối (ngăn ngừa shortcut kích thước).

---

## 2. Các Ràng Buộc Bất Biến Giữ Nguyên

Mọi nguyên tắc cốt lõi của Protocol v1.0.0 được duy trì nguyên vẹn:
- Giữ nguyên 5 outer-fold models đóng băng của cả 2 recipes (`visual_calibrated` và `late_fusion_dsp_augmented`).
- 6 điều kiện kiểm thử: `original`, `jpeg_q95`, `jpeg_q75`, `jpeg_q50`, `resize_0.5`, `resize_0.5_jpeg_q75`.
- Primary endpoint: $\Delta \overline{\text{Macro-F1}}$ tại `jpeg_q75` = Augmented − Visual.
- Paired source-cluster bootstrap 10,000 replicates, seed PCG64 `20261007`, 95% Percentile CI.
- Success khi CI lower bound $> 0$.
- Nghiêm cấm lấy trung bình xác suất trước khi tính Macro-F1; không tạo ensemble.
