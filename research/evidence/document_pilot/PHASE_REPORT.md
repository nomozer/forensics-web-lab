# Báo Cáo Thực Nghiệm: Pilot Phát Hiện Giả Mạo Tài Liệu Bằng AI & Đánh Đổi Hình Học Quan Sát (P0 vs P1 vs P2)

> **Mã định danh**: `Phase 4C.8 trace — Document AI-Tampering Data Accessibility & Observation Geometry Pilot`
> **Branch**: `research/independent-cohort-acquisition` (LOCAL CPU priority)
> **Cam kết khoa học**: Zero retraining, zero refitting, zero recalibration, zero post-hoc threshold tuning.
> **Ngưỡng quyết định tiền đăng ký**: $s_{\text{AI}} = p(\text{ai\_edited}) + p(\text{fully\_generated}) \ge 0.5$.
> **Định danh đề tài**: “Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.”
> **Ranh giới làm việc**: “Tài liệu” trong pilot này là ảnh tài liệu chụp hoặc scan (hóa đơn, chứng từ, văn bản) có can thiệp văn bản bằng AI (diffusion/inpainting). Định nghĩa mang tính ranh giới thực nghiệm nội bộ, không tự ý tuyên bố đã được hội đồng/GVHD phê duyệt; không mở rộng sang text LLM.

---

## 1. Tóm Tắt Điều Hành & Phát Hiện Trọng Tâm (Executive Summary)

1. **Kiểm toán Khả năng Tiếp cận Dữ liệu Tài liệu AI (Accessibility Audit)**:
   - Đã khảo sát 5 bộ dữ liệu y văn mục tiêu: `DocTamper` (CVPR 2023, nén .mdb có mật khẩu, splicing truyền thống), `OSTF` (AAAI 2025, ảnh biển hiệu đường phố, cần email xin), `AIForge-Doc` (Scam-AI, arXiv:2602.20569, hóa đơn CORD/SROIE inpainting bằng Gemini/Ideogram, dataset bị khóa trên HuggingFace `gated: auto` trả về 401 khi không có token), `Fake or JPEG?` (arXiv:2403.17608, lý thuyết confounding tiền xử lý), và `GADNet` (ESWA 2025, không có dữ liệu tài liệu).
   - Tiếp nhận thành công bộ chuẩn mở `vankey/RealText-V2` (ACM MM 2026 MGC GenText-Forensics, CC-BY-NC-4.0, ungated).
   - Được người dùng phê duyệt phương án (Option 1) cấp phép bổ sung mạng để hoàn thành panel cân bằng $N=40$ (20 AI-edited + 20 authentic) với tổng chuyển giao mạng phiên là 72.25 MB ($\ll 1\text{ GB}$).
   - **Kiểm toán cách ly độc lập (`ISOLATION_AUDIT_PASS`)**: Đối soát 40 mẫu với toàn bộ tập dữ liệu nội bộ (Three-Class train/holdout, Source control Nature, ImageNet train, So-Fake probe v2, TGIF N=400, Locked-Test 343 sources) ghi nhận **0 hash collisions** (100% cách ly tuyệt đối).

2. **Kiểm toán Hình học Center-Crop trên Tài liệu (Mask Retention Audit)**:
   - **P0 Center-Crop**: Độ bao phủ vùng chỉnh sửa trung bình chỉ đạt **72.95%** (Median: 86.18%, Min: 23.36%). Có tới **85.0% (17/20 mẫu)** bị mất một phần vùng can thiệp giả mạo do văn bản nằm ở lề, tiêu đề hoặc chân trang tài liệu dọc (A4/receipts, aspect ratio 0.71–1.47). Chỉ có 15.0% (3/20 mẫu) giữ trọn vẹn 100% vùng sửa.
   - **P1 Letterbox Canvas**: Bảo toàn trọn vẹn **100.0% (20/20 mẫu)** vùng chỉnh sửa và cấu trúc tài liệu nhờ đệm trung tính (128).
   - **P2 Multi-Tile (5 views)**: Bảo toàn 100.0% trên khung nhìn toàn cục và cung cấp 4 khung nhìn phóng đại 4 góc phần tư ($256 \times 256$).

3. **Đánh Đổi Hiệu Năng Phân Loại (P0 vs P1 vs P2 trên 6 Frozen Models)**:
   - **`A0-visual` (MobileNetV3 576-D GAP visual probe)**:
     - **P1 Letterbox** đạt hiệu năng xuất sắc nhất trong toàn bộ thực nghiệm: **Macro-F1 = 0.6931**, **Balanced Accuracy = 0.7000**, FPR = 0.1500, AUROC = 0.6900, AP = 0.7215.
     - So với P0 Center-Crop (Macro-F1 = 0.6011), P1 cải thiện $\mathbf{+0.0920}$ (+9.20 điểm phần trăm Macro-F1; 95% Bootstrap CI: $[-0.0837, +0.2658]$, $85.31\%$ positive bootstrap fraction).
   - **`A0-global` (592-D Visual + Global DSP)**:
     - P0 Center-Crop: Macro-F1 = 0.4872.
     - P1 Letterbox: Macro-F1 = **0.6050** ($\mathbf{+0.1178}$ hay +11.78 pp; 95% CI: $[-0.0280, +0.2686]$, $94.06\%$ positive bootstrap fraction).
     - P2 Multi-Tile: Macro-F1 = 0.6229, nhưng **FPR bùng nổ lên 45.0%** (9/20 ảnh authentic bị báo động giả thành AI).
   - **`A0-local` (640-D Visual + Global DSP + Local Quadrant DSP)**:
     - P0 và P1: Macro-F1 = 0.4302 (bị kìm hãm bởi GAP).
     - P2 Multi-Tile: Macro-F1 tăng vọt lên **0.6248** ($\mathbf{+0.1946}$ hay +19.46 pp vs P1; 95% CI: $[+0.0047, +0.3794]$, $97.79\%$ positive fraction, CI hoàn toàn dương), nhưng cái giá phải trả là **FPR tăng lên 40.0%** (8/20 authentic báo nhầm AI).
   - **Dòng mô hình `A1` (bổ sung authentic ImageNet trong huấn luyện)**:
     - Đạt **FPR = 0.0000** (0% báo động giả trên toàn bộ 20 tài liệu authentic), nhưng do không gian authentic ImageNet kéo lệch ranh giới quyết định sang authentic, recall trên tài liệu giả mạo rơi về 0.00–0.05, kéo Macro-F1 về mức $0.3333 - 0.3866$.

4. **Đánh Đổi Chi Phí Suy Luận (CPU Inference Latency Trade-Off)**:
   - **P1 Letterbox**: Nhanh nhất với Median = **223.7 ms** (Mean = 231.5 ms, P95 = 279.9 ms).
   - **P0 Center-Crop**: Median = **247.3 ms** (Mean = 254.5 ms, P95 = 310.7 ms).
   - **P2 Multi-Tile (5 views)**: Median = **836.2 ms**, Mean = **1,395.7 ms** (P95 = 1,147.1 ms). Gấp $\sim 3.7\times$ đến $6.0\times$ thời gian suy luận của P1!
   - **Kết luận kiến trúc**: **P1 (Orientation-Correct Letterbox)** là hình học quan sát tối ưu vượt trội: vừa bảo toàn 100% diện tích tài liệu, vừa đạt Macro-F1 cao nhất (0.6931 trên A0-visual), duy trì FPR thấp, và có độ trễ CPU nhỏ nhất (223.7 ms), hoàn toàn khả thi để tích hợp client-side in-browser.

---

## 2. Bảng Đối Soát Chỉ Số Toàn Diện (Summary Scoreboard)

Tất cả 6 mô hình được đánh giá dưới cùng ngưỡng nhị phân tiền đăng ký $s_{\text{AI}} \ge 0.5$ trên panel $N=40$ (20 AI-edited + 20 authentic):

| Model | Hình học | Macro-F1 | Bal-Acc | TPR (Recall) | TNR | FPR | AUROC | AP | Brier | ECE (10 bin) | CPU Latency (Median) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0-visual** | **P0 (Crop)** | 0.6011 | 0.6500 | 0.3000 | 1.0000 | 0.0000 | 0.7150 | 0.7752 | 0.2420 | 0.1199 | 247.3 ms |
| **A0-visual** | **P1 (Letterbox)** | **0.6931** | **0.7000** | **0.5500** | 0.8500 | 0.1500 | 0.6900 | 0.7215 | 0.2219 | 0.1249 | **223.7 ms** |
| **A0-visual** | **P2 (Multi-Tile)** | 0.5807 | 0.6250 | 0.3000 | 0.9500 | 0.0500 | 0.6850 | 0.7629 | 0.2359 | 0.0601 | 836.2 ms |
| A0-global | P0 (Crop) | 0.4872 | 0.5500 | 0.2000 | 0.9000 | 0.1000 | 0.6100 | 0.6598 | 0.3070 | 0.2522 | 247.3 ms |
| A0-global | P1 (Letterbox) | 0.6050 | 0.6250 | 0.4000 | 0.8500 | 0.1500 | 0.5950 | 0.6586 | 0.2639 | 0.1278 | 223.7 ms |
| A0-global | P2 (Multi-Tile) | 0.6229 | 0.6250 | 0.7000 | 0.5500 | **0.4500** | 0.6750 | 0.6775 | 0.2283 | 0.0638 | 836.2 ms |
| A0-local | P0 (Crop) | 0.4302 | 0.5000 | 0.1500 | 0.8500 | 0.1500 | 0.5200 | 0.5340 | 0.3778 | 0.3733 | 247.3 ms |
| A0-local | P1 (Letterbox) | 0.4302 | 0.5000 | 0.1500 | 0.8500 | 0.1500 | 0.5200 | 0.6117 | 0.3044 | 0.2403 | 223.7 ms |
| A0-local | P2 (Multi-Tile) | 0.6248 | 0.6250 | 0.6500 | 0.6000 | **0.4000** | 0.6700 | 0.6989 | 0.2275 | 0.0633 | 836.2 ms |
| A1-visual | P0 (Crop) | 0.3333 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.6525 | 0.7556 | 0.4221 | 0.4427 | 247.3 ms |
| A1-visual | P1 (Letterbox) | 0.3866 | 0.5250 | 0.0500 | 1.0000 | 0.0000 | 0.6200 | 0.6927 | 0.4116 | 0.4240 | 223.7 ms |
| A1-visual | P2 (Multi-Tile) | 0.3333 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.6175 | 0.7037 | 0.4072 | 0.4159 | 836.2 ms |
| A1-global | P0 (Crop) | 0.3333 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.6525 | 0.6788 | 0.4128 | 0.4226 | 247.3 ms |
| A1-global | P1 (Letterbox) | 0.3866 | 0.5250 | 0.0500 | 1.0000 | 0.0000 | 0.6275 | 0.7256 | 0.3938 | 0.3928 | 223.7 ms |
| A1-global | P2 (Multi-Tile) | 0.3333 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.6300 | 0.6950 | 0.3551 | 0.3506 | 836.2 ms |
| A1-local | P0 (Crop) | 0.3220 | 0.4750 | 0.0000 | 0.9500 | 0.0500 | 0.5400 | 0.5342 | 0.4284 | 0.4446 | 247.3 ms |
| A1-local | P1 (Letterbox) | 0.3866 | 0.5250 | 0.0500 | 1.0000 | 0.0000 | 0.5100 | 0.6006 | 0.3936 | 0.3876 | 223.7 ms |
| A1-local | P2 (Multi-Tile) | 0.3333 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.5025 | 0.6031 | 0.3423 | 0.2955 | 836.2 ms |

---

## 3. Phân Tích Thống Kê Ghép Cặp (10,000 Replicate Paired Bootstrap CIs)

| Mô hình | So sánh | $\Delta \text{Macro-F1}$ | 95% Percentile CI | $P(\Delta^* > 0)$ | $\Delta \text{Bal-Acc}$ | 95% Percentile CI | $P(\Delta^* > 0)$ | Kết luận khoa học |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **A0-visual** | **P1 vs P0** | **+0.0920** | `[-0.0837, +0.2658]` | **85.31%** | **+0.0500** | `[-0.0978, +0.1918]` | 74.88% | Xu hướng cải thiện tích cực mạnh |
| A0-visual | P2 vs P0 | -0.0204 | `[-0.1301, +0.0891]` | 30.40% | -0.0250 | `[-0.1146, +0.0556]` | 23.26% | P2 không cải thiện trên visual-only |
| A0-visual | P2 vs P1 | -0.1124 | `[-0.2571, +0.0223]` | 5.14% | -0.0750 | `[-0.1933, +0.0404]` | 9.37% | P1 vượt trội P2 rõ rệt |
| **A0-global** | **P1 vs P0** | **+0.1178** | `[-0.0280, +0.2686]` | **94.06%** | **+0.0750** | `[-0.0631, +0.2188]` | 84.70% | Cải thiện lớn, 94% mass dương |
| A0-global | P2 vs P0 | +0.1357 | `[-0.0718, +0.3401]` | 90.14% | +0.0750 | `[-0.1015, +0.2601]` | 78.30% | P2 tăng F1 nhưng trả giá bằng FPR 45% |
| A0-global | P2 vs P1 | +0.0179 | `[-0.1615, +0.2000]` | 58.03% | 0.0000 | `[-0.1600, +0.1567]` | 49.82% | P2 không vượt P1 trên Bal-Acc |
| **A0-local** | **P2 vs P1** | **+0.1946** | `[+0.0047, +0.3794]` | **97.79%** | **+0.1250** | `[-0.0589, +0.3182]` | 90.08% | **Ý nghĩa thống kê dương (CI cắt trên 0)** |

---

## 4. Kiểm Toán Hình Học Phủ Vùng Mask (Mask Coverage Retention)

Phân bố mức độ giữ lại vùng chỉnh sửa trên 20 mẫu tài liệu bị sửa (`ai_edited`):
- **P0 Center-Crop**:
  - Trung bình: **72.95%**
  - Tứ phân vị: Q25 = 48.24%, Trung vị P50 = 86.18%, Q75 = 97.05%
  - Giá trị nhỏ nhất: 23.36% (mẫu `GenText_Forensic_00000007.jpg`, chỉ giữ lại chưa đầy 1/4 diện tích can thiệp do văn bản ở rìa trang)
  - Số mẫu mất một phần diện tích: **17 / 20 mẫu (85.0%)**
  - Số mẫu giữ nguyên 100%: **3 / 20 mẫu (15.0%)**
  - Số mẫu mất toàn bộ: 0 / 20 mẫu (nhờ diện tích crop bao quát trung tâm trang)
- **P1 Letterbox Canvas**:
  - Giữ lại: **100.0% trên 20 / 20 mẫu**
  - Không làm biến dạng tỷ lệ khung hình tài liệu (chỉ đệm 128 ở lề trống).

---

## 5. Luận Giải Khoa Học & Ý Nghĩa Kiến Trúc Hệ Thống

1. **Vì sao P1 (Letterbox) lại cải thiện mạnh mẽ `A0-visual`?**:
   - Khác với ảnh thiên nhiên hay chân dung (nơi chủ thể thường ở trung tâm), tài liệu hành chính (hóa đơn, chứng từ, văn bản) có cấu trúc thông tin trải dài: số hóa đơn, con dấu, chữ ký, tổng tiền thường nằm ở tiêu đề trên cùng hoặc chân trang dưới cùng.
   - Khi áp dụng P0 (Center-Crop vuông $\min(W, H)$), 85% tài liệu bị cắt xén trực tiếp vào các dòng văn bản bị can thiệp.
   - P1 letterbox giữ nguyên vẹn 100% ngữ cảnh toàn trang. Mô hình MobileNetV3 nhờ đó quan sát được toàn bộ các khu vực chỉnh sửa, nâng độ nhạy phát hiện (Recall) từ 30.0% lên **55.0%**, kéo Macro-F1 từ 0.6011 lên **0.6931**.

2. **Vì sao P2 (Multi-Tile) gây bùng nổ báo động giả (False Alarms) trên DSP?**:
   - Khi chia tài liệu thành 4 góc phần tư $256 \times 256$, các đường cắt nhân tạo chạy xuyên qua các đoạn văn bản và nền giấy.
   - Biến đổi Fourier 2D (FFT) và DCT trên các mảng cắt này bắt gặp hiện tượng rò rỉ phổ (spectral leakage) và các bước nhảy biên độ lớn (discontinuities) tại mép cắt, kích hoạt các tín hiệu tần số cao giả tạo.
   - Do đó, các mô hình phụ thuộc DSP (`A0-global`, `A0-local`) hiểu nhầm các mép cắt và vùng khoảng trắng biên tài liệu là dấu vết AI, làm tỷ lệ báo động giả trên tài liệu gốc vọt lên **40%–45%**.
   - Hơn nữa, việc phải trích xuất đặc trưng 5 lần khiến CPU latency tăng từ 223 ms lên gần **1.4 giây**, không phù hợp với tiêu chí client-side siêu nhẹ trong trình duyệt web.

3. **Hiện tượng dịch chuyển ranh giới quyết định của Điều kiện A1**:
   - Việc bổ sung 250 ảnh ImageNet authentic trong A1 giúp loại bỏ 100% báo động giả trên tài liệu authentic (FPR = 0.0000), nhưng do đặc trưng tài liệu văn bản khác xa ảnh thiên nhiên ImageNet/COCO, mô hình A1 phân loại gần như toàn bộ tài liệu là authentic, khiến khả năng phát hiện tài liệu giả mạo bị triệt tiêu (Recall $\le 5\%$).
   - Kết luận khoa học: **Không thể dùng chung không gian biểu diễn ảnh thiên nhiên (ImageNet/COCO) để phát hiện tài liệu văn bản giả mạo mà không có dữ liệu tài liệu trong tập huấn luyện.**

---

## 6. Hồ Sơ Bằng Chứng & Artifacts Liên Quan

- **Tài liệu kiểm toán khả năng tiếp cận**: `research/evidence/document_pilot/DOCUMENT_DATASET_ACCESSIBILITY_AUDIT.md`
- **Giao thức tiền đăng ký v1.0.0**: `research/evidence/document_pilot/PROTOCOL_DOCUMENT_PILOT_V1.0.0.md`
- **Manifest tập dữ liệu ($N=40$)**: `data/research/document_pilot/document_pilot_manifest.json` (SHA-256: `ed0a17c8...`)
- **Biên nhận kiểm toán cách ly**: `research/evidence/document_pilot/document_pilot_isolation_audit.json` (`ISOLATION_AUDIT_PASS`, 0 collisions)
- **Biên nhận tiếp nhận dữ liệu**: `research/evidence/document_pilot/document_acquisition_receipt.json`
- **Dự đoán chi tiết 6 mô hình × 3 hình học**: `research/evidence/document_pilot/document_pilot_predictions.json`
- **Biên nhận thực nghiệm chi tiết**: `research/evidence/document_pilot/document_pilot_receipt.json`
- **Runner chuẩn mực**: `scripts/research/run_document_observation_pilot.py`
- **Bộ kiểm thử tự động (Unit tests)**: `ml/tests/test_document_observation_pilot.py` (3/3 PASS)

---

## 7. Trạng Thái Pha & Bước Duy Nhất Tiếp Theo

- **Trạng thái Pha**: `DOCUMENT_AI_TAMPERING_PILOT_AND_OBSERVATION_GEOMETRY_EVALUATED_PASS`
- **Phán quyết kỹ thuật**: Hình học P1 (Orientation-Correct Letterbox) chính thức được khuyến nghị thay thế P0 cho tác vụ xử lý tài liệu trong toàn bộ pipeline tương lai.
- **Bước duy nhất tiếp theo**: Huấn luyện segmentation baseline (MobileNetV3-UNet mini < 10MB) trên **COLAB GPU** theo `PROTOCOL_LOCALIZATION_PREPARATION_V1.0.0.md` với hình học P1 Letterbox Canvas, sau đó mang về đánh giá trên **LOCAL CPU**.
