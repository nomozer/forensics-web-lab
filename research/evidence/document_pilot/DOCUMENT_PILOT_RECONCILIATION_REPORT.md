# Báo Cáo Đối Soát & Đính Chính: Document Pilot Reconciliation, Taxonomy/Compression Confounding & Paired Replay (C0 vs C1)

> **Mã định danh**: `Phase 4C.8 trace — Document Pilot Reconciliation & Replay Audit`<br>
> **Tên đề tài chính thức**: “Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.”<br>
> **Branch**: `research/independent-cohort-acquisition` (LOCAL CPU)<br>
> **Cam kết khoa học bất biến**: Zero retraining, zero refitting, zero recalibration, zero post-hoc threshold tuning. Sáu mô hình locked giữ nguyên trạng thái đóng băng tuyệt đối. Không tự động đổi RQ, không coi localization là bắt buộc của đề tài.

---

## 1. Tóm Tắt Điều Hành & Phán Quyết Khoa Học (Executive Summary)

Giai đoạn đối soát này tập trung chấn chỉnh tính trung thực khoa học, đính chính 6 phát biểu vượt bằng chứng của báo cáo pilot ban đầu, kiểm toán tận gốc phân loại học (taxonomy) và nguy cơ gây nhiễu định dạng/nén (format/compression confounding) trên bộ dữ liệu `RealText-V2`, đồng thời hoàn thành thực nghiệm replay ghép cặp có kiểm soát giữa C0 (Original decoding) và C1 (Symmetric JPEG `Q=75`, `subsampling=2`, `optimize=False`) qua 3 hình học (P0, P1, P2) trên toàn bộ 6 locked models.

### Các Kết Quả Then Chốt:
1. **Kiểm toán Taxonomy & Provenance ($N=40$)**:
   - **0/40 mẫu** trong panel có bằng chứng được xác minh là can thiệp bằng Generative AI (diffusion, GAN, LLM inpainting).
   - Toàn bộ 20 mẫu giả mạo (`black`) thuộc nhóm **`manipulation provenance unverified`**: báo cáo sơ cấp ghi nhận các can thiệp số cấp ký tự/từ ngữ (xóa che mosaic clumsy, sửa đổi ngày tháng, thay thế từ ngữ pháp lý, vi chỉnh phông chữ), nhưng không có bằng chứng công cụ tạo sinh cụ thể.
   - Toàn bộ 20 mẫu nguyên bản (`white`) là tài liệu độc lập riêng biệt, hoàn toàn **không ghép cặp nguồn 1:1** với tập forged (0 stem overlap).
   - **Phán quyết khoa học bắt buộc**: Pilot này strictly chỉ đánh giá năng lực phân loại giả mạo / nguyên bản (**forgery vs pristine**) theo nhãn nguồn; **tuyệt đối không tuyên bố đã đo lường năng lực phát hiện tài liệu do AI tạo sinh / chỉnh sửa (AI-edited)**.
2. **Kiểm toán Định Dạng & Lịch Sử Nén (Format × Label Confounding)**:
   - **100% tương quan định dạng với nhãn nguồn**: 20/20 mẫu forged là file JPEG (magic `FF D8 FF E0`, chroma subsampling 4:2:0, bảng lượng tử hóa luminance Q0 có mean đồng nhất 5.77); ngược lại 20/20 mẫu authentic là file PNG (magic `89 50 4E 47`, RGB lossless, không có nén mất dữ liệu).
   - Tồn tại nguy cơ shortcut nghiêm trọng: các mô hình có thể phân loại dựa trên lưới khối DCT 8×8 hoặc subsampling 4:2:0 của JPEG thay vì phát hiện dấu vết sửa đổi văn bản.
3. **Đính Chính Kiểm Toán Cách Ly (Isolation Audit Rectification)**:
   - Script kiểm toán chỉ so sánh mã băm byte SHA-256 đối với các manifest nội bộ đã niêm phong. Kết quả 0 xung đột mã băm chỉ chứng minh **không trùng lặp byte dữ liệu**, **không chứng minh được tính độc lập nguồn (source independence)**. Xóa bỏ hoàn toàn cụm từ "100% cách ly tuyệt đối".
4. **Kết Quả Replay Ghép Cặp C0 vs C1**:
   - Nén JPEG đối xứng C1 trước khi áp dụng tiền xử lý chuẩn tắc bảo toàn hình học không làm thay đổi các quyết định phân loại trên `A0-visual`, `A0-global`, `A1-global`, `A1-visual` (0 label flips trên toàn bộ 40 mẫu).
   - P1 Letterbox duy trì Macro-F1 cao hơn P0 Center-Crop (+0.0920 trên A0-visual, +0.1178 trên A0-global) ở cả hai điều kiện C0 và C1.
   - P2 Multi-Tile tiếp tục bùng nổ báo động giả authentic trên các mô hình phụ thuộc DSP (FPR 35%–45%) ở cả hai điều kiện C0 và C1.
   - **Ranh giới nhận thức**: Phép thử C1 là chẩn đoán độ nhạy với một lần nén bổ sung, không có nghĩa là đã xóa sạch lịch sử nén trước đó của ảnh hay tạo ra thiết kế đối chứng nguồn 2×2.
5. **Bóc Tách Độ Trễ CPU Thực Tế**:
   - Con số độ trễ lịch sử giống nhau giữa 6 mô hình thực chất là thời gian trích xuất đặc trưng dùng chung (shared feature extraction), bị sao chép sang từng dòng mô hình.
   - Thời gian tính điểm thực tế của từng mô hình (linear scoring + temperature scaling) là siêu nhỏ: **0.13 – 0.21 ms** cho P0/P1 và **0.62 – 0.69 ms** cho P2.
   - Hơn 99.8% độ trễ pipeline nằm ở tiền xử lý và trích xuất đặc trưng dùng chung. CPU latency không đại diện cho Web Worker / browser environment.

---

## 2. Sổ Cái Đính Chính Sáu Phát Biểu Vượt Bằng Chứng (Epistemic Rectification Ledger)

| STT | Phát biểu cũ trong Phase Report | Đính chính khoa học chuẩn mực | Bằng chứng thực nghiệm |
| :---: | :--- | :--- | :--- |
| **1** | *"P1 tối ưu vượt trội"* | P1 cải thiện Macro-F1 tại ngưỡng tiền đăng ký 0.5 trên panel hiện tại, nhưng có trade-off: P1 có thể làm tăng FPR (từ 0.0% lên 15.0% trên A0-visual) và làm giảm nhẹ AUROC (từ 0.7150 xuống 0.6900) và AP (từ 0.7752 xuống 0.7215). Không dùng từ "tối ưu vượt trội". | Raw predictions & receipt: AUROC P0 (0.7150) > P1 (0.6900); FPR P0 (0.00) < P1 (0.15). |
| **2** | *"P1 bảo toàn 100% dấu vết forensic"* | P1 bảo toàn 100% diện tích hình học (không cắt bỏ lề/chân trang), nhưng việc thu nhỏ toàn trang (downsampling tài liệu kích thước lớn ~2550×3300 về canvas 512×512) làm suy giảm độ phân giải và làm mờ các chi tiết văn bản vi mô (micro-text details). | Hình học pixel: tỷ lệ co nhỏ ~5.0x đến 6.5x làm mất thông tin biên chữ sắc nét. |
| **3** | *"P2 báo nhầm vì spectral leakage"* | Mép cắt góc phần tư tạo gián đoạn biên và kích hoạt tần số cao là một **giả thuyết giải thích hợp lý**, chưa phải kết luận nhân quả cô lập hoàn toàn. Báo động giả của P2 còn chịu ảnh hưởng từ kích thước khối DCT và tương tác cấu trúc văn bản. | C1 (nén JPEG đối xứng) vẫn ghi nhận P2 FPR = 45% (A0-global) và 35% (A0-local). |
| **4** | *"A1 loại bỏ 100% báo động giả"* | Việc A1 đạt FPR = 0.0% trên 20 tài liệu authentic thực chất là do ranh giới quyết định bị kéo lệch hoàn toàn về phía authentic, dẫn đến việc mô hình bỏ sót gần như toàn bộ tài liệu giả mạo (Recall $\le 5\%$, Macro-F1 chỉ $0.33 - 0.38$). Đây là sự sụp đổ độ nhạy chứ không phải thành tựu phát hiện. | Scoreboard A1: TPR chỉ 0/20 hoặc 1/20 tài liệu giả mạo được phát hiện. |
| **5** | *"Không thể dùng backbone ảnh tự nhiên cho tài liệu"* | Đây là khẳng định quá mức. Thực tế A0-visual (sử dụng hoàn toàn backbone ảnh tự nhiên MobileNetV3 ImageNet đóng băng) vẫn đạt Macro-F1 = 0.6931 và Bal-Acc = 0.7000 trên nhãn nguồn panel này, cao nhất trong toàn bộ 6 mô hình. | A0-visual P1 đạt Macro-F1 0.6931, vượt trội các mô hình DSP. |
| **6** | *"Hoàn toàn khả thi trên browser"* | Báo cáo chưa hề đo lường trên trình duyệt (in-browser Web Worker / WASM). Đo lường trên Python LOCAL CPU đơn luồng không đại diện cho độ trễ, mức tiêu thụ bộ nhớ hoặc giới hạn Web Worker của trình duyệt. | Khẳng định browser feasibility là mục tiêu kỹ thuật chưa được kiểm chứng trong pilot này. |

---

## 3. Kiểm Toán Phân Loại Học (Taxonomy & Provenance Audit, $N=40$)

Chi tiết toàn diện lưu tại: [`research/evidence/document_pilot/document_sample_taxonomy_audit.json`](document_sample_taxonomy_audit.json).

### Phân Bổ Danh Mục Provenance:
- **Verified AI manipulation**: **0 / 40 mẫu (0.0%)**. Không có mẫu nào có bằng chứng được tạo ra bởi diffusion/inpainting/AIGC.
- **Verified non-AI manipulation**: **0 / 40 mẫu (0.0%)**. Không xác minh được công cụ phần mềm chỉnh sửa cụ thể.
- **Manipulation provenance unverified**: **20 / 40 mẫu (50.0%)**. Toàn bộ 20 mẫu giả mạo (`black`, `GenText_Forensic_00000000` đến `00000019`).
- **Authentic pristine**: **20 / 40 mẫu (50.0%)**. Toàn bộ 20 mẫu nguyên bản (`white`, `GenText_Forensic_00002000` đến `00002024`).

### Quan Hệ Ghép Cặp (Source Relations):
- **Cặp nguồn 1:1**: **0 cặp**. 20 mẫu forged và 20 mẫu authentic là các tài liệu độc lập hoàn toàn, không có chung nguồn gốc (0 stem overlap).
- **Mask Annotation Semantics**: Mặt nạ điểm ảnh là binary mask (pixel=255 biểu thị vùng văn bản bị can thiệp, pixel=0 biểu thị nền nguyên bản).

---

## 4. Kiểm Toán Định Dạng & Lịch Sử Nén (Format × Compression Audit)

Chi tiết toàn diện lưu tại: [`research/evidence/document_pilot/document_format_compression_audit.json`](document_format_compression_audit.json).

### Ma Trận Định Dạng × Nhãn Nguồn:
| Nhãn Nguồn (`raw_type`) | Đuôi File | Magic Bytes | PIL Format | Chroma Subsampling | Mean Q0 (Luminance) | PNG Chunks | Số Lượng Mẫu |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **black (forged)** | `.jpg` | `FF D8 FF E0` | `JPEG` | **4:2:0** | **5.77** | Không có | **20 / 20 (100%)** |
| **white (authentic)** | `.png` | `89 50 4E 47` | `PNG` | Không nén | Không có | `IDAT, IEND, IHDR` | **20 / 20 (100%)** |

### Kết Luận Về Rủi Ro Gây Nhiễu:
1. Có sự gắn kết hoàn hảo giữa nhãn nguồn và định dạng lưu trữ: toàn bộ ảnh giả mạo là JPEG có nén mất dữ liệu (lossy 4:2:0), trong khi toàn bộ ảnh nguyên bản là PNG không nén (lossless RGB).
2. Tuy nhiên, việc ảnh authentic lưu dưới dạng PNG không chứng minh ảnh chưa từng trải qua nén JPEG trong quá trình tạo lập ban đầu (chẳng hạn chụp từ máy ảnh/máy scan rồi xuất sang PNG).
3. Không thể kết luận các bộ dò chỉ học đặc trưng nén nếu không có bộ dữ liệu đối chứng ghép cặp 2×2 thực sự trên cùng nguồn tài liệu.

---

## 5. Bảng Đối Soát Replay Ghép Cặp (C0 vs C1 trên 6 Locked Models)

Toàn bộ 6 mô hình được đánh giá dưới ngưỡng tiền đăng ký $s_{\text{AI}} \ge 0.5$ trên $N=40$ mẫu:
- **C0**: Original decoding (orientation-correct RGB).
- **C1**: Symmetric JPEG recompression (`quality=75`, `subsampling=2`, `optimize=False`) thực hiện một lần duy nhất trên RGB trước khi đưa vào các hình học P0, P1, P2.

| Mô hình | Điều kiện | Hình học | Macro-F1 | Bal-Acc | TPR (Recall) | TNR | FPR | AUROC | AP | Binary Brier (so 0.25) | Multiclass Brier (so 0.67) | ECE (10-bin) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0-visual** | **C0** | P0 | 0.6011 | 0.6500 | 0.3000 | 1.0000 | 0.0000 | 0.7150 | 0.7752 | 0.2420 | 0.6938 | 0.1199 |
| **A0-visual** | **C0** | **P1** | **0.6931** | **0.7000** | **0.5500** | 0.8500 | 0.1500 | 0.6900 | 0.7215 | **0.2219** | **0.6379** | 0.1249 |
| **A0-visual** | **C0** | P2 | 0.5807 | 0.6250 | 0.3000 | 0.9500 | 0.0500 | 0.6850 | 0.7629 | 0.2359 | 0.6611 | 0.0601 |
| **A0-visual** | **C1** | P0 | 0.6011 | 0.6500 | 0.3000 | 1.0000 | 0.0000 | 0.7125 | 0.7773 | 0.2428 | 0.6943 | 0.1204 |
| **A0-visual** | **C1** | **P1** | **0.6931** | **0.7000** | **0.5500** | 0.8500 | 0.1500 | 0.6925 | 0.7260 | **0.2221** | **0.6385** | 0.1248 |
| **A0-visual** | **C1** | P2 | 0.5807 | 0.6250 | 0.3000 | 0.9500 | 0.0500 | 0.6850 | 0.7629 | 0.2365 | 0.6614 | 0.0603 |
| A0-global | C0 | P0 | 0.4872 | 0.5500 | 0.2000 | 0.9000 | 0.1000 | 0.6100 | 0.6598 | 0.3070 | 0.7486 | 0.2522 |
| A0-global | C0 | P1 | 0.6050 | 0.6250 | 0.4000 | 0.8500 | 0.1500 | 0.5950 | 0.6586 | 0.2639 | 0.6513 | 0.1278 |
| A0-global | C0 | P2 | 0.6229 | 0.6250 | 0.7000 | 0.5500 | 0.4500 | 0.6750 | 0.6775 | 0.2283 | 0.7090 | 0.0638 |
| A0-global | C1 | P0 | 0.4872 | 0.5500 | 0.2000 | 0.9000 | 0.1000 | 0.6100 | 0.6598 | 0.3072 | 0.7487 | 0.2522 |
| A0-global | C1 | P1 | 0.6050 | 0.6250 | 0.4000 | 0.8500 | 0.1500 | 0.5950 | 0.6586 | 0.2640 | 0.6524 | 0.1278 |
| A0-global | C1 | P2 | 0.6229 | 0.6250 | 0.7000 | 0.5500 | 0.4500 | 0.6700 | 0.6750 | 0.2283 | 0.7075 | 0.0744 |
| A0-local | C0 | P0 | 0.4302 | 0.5000 | 0.1500 | 0.8500 | 0.1500 | 0.5200 | 0.5340 | 0.3778 | 0.8037 | 0.3733 |
| A0-local | C0 | P1 | 0.4302 | 0.5000 | 0.1500 | 0.8500 | 0.1500 | 0.5200 | 0.6117 | 0.3044 | 0.7259 | 0.2403 |
| A0-local | C0 | P2 | 0.6248 | 0.6250 | 0.6500 | 0.6000 | 0.4000 | 0.6700 | 0.6989 | 0.2275 | 0.7088 | 0.0633 |
| A0-local | C1 | P0 | 0.4302 | 0.5000 | 0.1500 | 0.8500 | 0.1500 | 0.5150 | 0.5317 | 0.3778 | 0.8037 | 0.3722 |
| A0-local | C1 | P1 | 0.4302 | 0.5000 | 0.1500 | 0.8500 | 0.1500 | 0.5150 | 0.6097 | 0.3043 | 0.7262 | 0.2401 |
| A0-local | C1 | P2 | 0.6500 | 0.6500 | 0.6500 | 0.6500 | 0.3500 | 0.6650 | 0.6946 | 0.2281 | 0.7080 | 0.0584 |
| A1-global | C0 | P0 | 0.3333 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.6525 | 0.6788 | 0.4128 | 0.8306 | 0.4226 |
| A1-global | C0 | P1 | 0.3866 | 0.5250 | 0.0500 | 1.0000 | 0.0000 | 0.6275 | 0.7256 | 0.3938 | 0.8143 | 0.3928 |
| A1-global | C0 | P2 | 0.3333 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.6300 | 0.6950 | 0.3551 | 0.7816 | 0.3506 |
| A1-global | C1 | P0 | 0.3333 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.6525 | 0.6788 | 0.4137 | 0.8315 | 0.4235 |
| A1-global | C1 | P1 | 0.3866 | 0.5250 | 0.0500 | 1.0000 | 0.0000 | 0.6300 | 0.7327 | 0.3941 | 0.8148 | 0.3933 |
| A1-global | C1 | P2 | 0.3333 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.6200 | 0.6879 | 0.3563 | 0.7828 | 0.3520 |
| A1-local | C0 | P0 | 0.3220 | 0.4750 | 0.0000 | 0.9500 | 0.0500 | 0.5400 | 0.5342 | 0.4284 | 0.8492 | 0.4446 |
| A1-local | C0 | P1 | 0.3866 | 0.5250 | 0.0500 | 1.0000 | 0.0000 | 0.5100 | 0.6006 | 0.3936 | 0.8130 | 0.3876 |
| A1-local | C0 | P2 | 0.3333 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.5025 | 0.6031 | 0.3423 | 0.7709 | 0.2955 |
| A1-local | C1 | P0 | 0.3220 | 0.4750 | 0.0000 | 0.9500 | 0.0500 | 0.5400 | 0.5342 | 0.4290 | 0.8497 | 0.4451 |
| A1-local | C1 | P1 | 0.3866 | 0.5250 | 0.0500 | 1.0000 | 0.0000 | 0.5100 | 0.6161 | 0.3938 | 0.8131 | 0.3879 |
| A1-local | C1 | P2 | 0.3333 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.5025 | 0.6028 | 0.3428 | 0.7712 | 0.2991 |
| A1-visual | C0 | P0 | 0.3333 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.6525 | 0.7556 | 0.4221 | 0.8358 | 0.4427 |
| A1-visual | C0 | P1 | 0.3866 | 0.5250 | 0.0500 | 1.0000 | 0.0000 | 0.6200 | 0.6927 | 0.4116 | 0.8258 | 0.4240 |
| A1-visual | C0 | P2 | 0.3333 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.6175 | 0.7037 | 0.4072 | 0.8206 | 0.4159 |
| A1-visual | C1 | P0 | 0.3333 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.6525 | 0.7556 | 0.4227 | 0.8362 | 0.4431 |
| A1-visual | C1 | P1 | 0.3866 | 0.5250 | 0.0500 | 1.0000 | 0.0000 | 0.6200 | 0.6927 | 0.4117 | 0.8259 | 0.4240 |
| A1-visual | C1 | P2 | 0.3333 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.6125 | 0.7009 | 0.4075 | 0.8208 | 0.4159 |

---

## 6. Phân Tích Thống Kê Ghép Cặp (Paired Bootstrap CIs, 10,000 Replicates)

- **Phương pháp**: Paired image bootstrap (10,000 resamples cùng chỉ số ảnh giữa các điều kiện). Do các mẫu trong RealText-V2 không có thông tin nhóm nguồn được xác minh (unpaired independent documents), phân tích này áp dụng giả định mẫu ghép cặp đơn lẻ (single paired sample), không gọi là source-cluster bootstrap.
- **Quy tắc diễn giải**: Tỷ lệ replicate có delta dương ($P(\Delta^* > 0)$) được báo cáo rõ ràng là **Bootstrap Positive Fraction**, tuyệt đối không trình bày như p-value hay xác suất giả thuyết đúng.

### Độ Nhạy Nén (C1 vs C0 Trong Cùng Hình Học):
- `A0-visual` (P0, P1, P2): $\Delta \text{Macro-F1} = +0.0000$ (95% CI: `[0.0000, 0.0000]`), $\Delta \text{Bal-Acc} = +0.0000$ (95% CI: `[0.0000, 0.0000]`). 0 flips / 40 mẫu. Điểm số trôi trung bình: $\Delta s_{\text{AI}} = -0.0016$ (P0), $+0.0006$ (P1), $+0.0004$ (P2).
- `A0-global` (P0, P1, P2): $\Delta \text{Macro-F1} = +0.0000$ (95% CI: `[0.0000, 0.0000]`), 0 flips. $\Delta s_{\text{AI}} = -0.0005$ (P0), $+0.0002$ (P1), $+0.0005$ (P2).
- `A0-local`:
  - P0, P1: 0 flips, $\Delta \text{Macro-F1} = 0.0000$.
  - P2: 1 flip (từ False Positive thành True Negative trên mẫu authentic), $\Delta \text{Macro-F1} = +0.0252$ (95% CI: `[0.0000, +0.0829]`, $P > 0: 64.0\%$), $\Delta \text{Bal-Acc} = +0.0250$.
- `Dòng A1`: 0 flips trên toàn bộ 6 cấu hình (P0, P1, P2 của A1-global, A1-local, A1-visual).

### So Sánh Hình Học (P1 vs P0 và P2 vs P0):
- **`A0-visual` (P1 vs P0)**:
  - C0: $\Delta \text{Macro-F1} = \mathbf{+0.0920}$ (95% CI: `[-0.0837, +0.2658]`, $P(\Delta^* > 0) = 85.3\%$), $\Delta \text{Bal-Acc} = +0.0500$ (95% CI: `[-0.0978, +0.1918]`).
  - C1: $\Delta \text{Macro-F1} = \mathbf{+0.0920}$ (95% CI: `[-0.0837, +0.2658]`, $P(\Delta^* > 0) = 85.3\%$), $\Delta \text{Bal-Acc} = +0.0500$.
  - **Kết luận**: Hiệu ứng cải thiện Macro-F1 của P1 Letterbox so với P0 Center-Crop duy trì hoàn toàn vững chắc sau nén JPEG đối xứng C1. Khoảng tin cậy 95% CI cắt 0 nhẹ ở cận dưới (-0.0837), mang tính xu hướng tích cực mạnh trong phạm vi panel $N=40$.
- **`A0-global` (P1 vs P0)**:
  - C0: $\Delta \text{Macro-F1} = \mathbf{+0.1178}$ (95% CI: `[-0.0280, +0.2686]`, $P(\Delta^* > 0) = 94.1\%$).
  - C1: $\Delta \text{Macro-F1} = \mathbf{+0.1178}$ (95% CI: `[-0.0280, +0.2686]`, $P(\Delta^* > 0) = 94.1\%$).

---

## 7. Bóc Tách Độ Trễ CPU Thực Tế (Isolated CPU Latency Benchmark)

Đo lường trên panel đối chứng tất định gồm 4 tài liệu (2 forged + 2 authentic), 2 lượt warm-up, 5 lần lặp độc lập trên LOCAL CPU:

| Hình học | Giai đoạn thực thi | Median (ms) | Mean (ms) | P95 (ms) | Min (ms) | Max (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **P0 (Crop)** | Tiền xử lý (Transform) | 181.70 ms | 184.44 ms | 225.43 ms | 149.20 ms | 225.68 ms |
| | Trích xuất đặc trưng dùng chung (Shared Extraction) | 247.90 ms | 256.76 ms | 360.77 ms | 134.11 ms | 365.17 ms |
| | Tính điểm từng mô hình (Scoring A0-visual) | **0.15 ms** | 0.19 ms | 0.36 ms | 0.11 ms | 0.47 ms |
| | **Tổng toàn pipeline (End-to-End)** | **432.94 ms** | 444.49 ms | 585.77 ms | 286.80 ms | 587.12 ms |
| **P1 (Letterbox)** | Tiền xử lý (Transform) | 87.98 ms | 91.73 ms | 116.24 ms | 81.62 ms | 117.80 ms |
| | Trích xuất đặc trưng dùng chung (Shared Extraction) | 138.90 ms | 142.27 ms | 165.27 ms | 128.48 ms | 176.62 ms |
| | Tính điểm từng mô hình (Scoring A0-visual) | **0.13 ms** | 0.15 ms | 0.25 ms | 0.09 ms | 0.28 ms |
| | **Tổng toàn pipeline (End-to-End)** | **226.48 ms** | 234.90 ms | 272.67 ms | 211.40 ms | 272.70 ms |
| **P2 (Multi-Tile)** | Tiền xử lý (Transform 5 views) | 124.33 ms | 125.66 ms | 144.80 ms | 105.24 ms | 150.39 ms |
| | Trích xuất đặc trưng dùng chung (5 views) | 698.84 ms | 701.39 ms | 755.88 ms | 627.40 ms | 782.80 ms |
| | Tính điểm từng mô hình (Scoring A0-visual 5 views) | **0.62 ms** | 0.64 ms | 0.77 ms | 0.49 ms | 1.01 ms |
| | **Tổng toàn pipeline (End-to-End)** | **826.49 ms** | 830.98 ms | 901.15 ms | 763.80 ms | 921.90 ms |

### Phát Hiện Về Sai Lệch Báo Cáo Lịch Sử:
1. Trong báo cáo trước, con số ~223 ms (P1), ~247 ms (P0), ~836 ms (P2) được sao chép y nguyên vào tất cả 6 dòng mô hình trong bảng tổng hợp, gây ấn tượng sai lệch rằng đó là thời gian suy luận độc lập của từng mô hình.
2. Thực tế, thời gian tính điểm của mỗi mô hình chỉ tốn **0.13 – 0.21 ms**. Hơn 99.8% thời gian là tiền xử lý hình ảnh và trích xuất đặc trưng dùng chung (MobileNetV3 backbone + DSP 16-D).
3. Do đó, việc chạy thêm 5 mô hình tuyến tính chỉ làm tăng thêm khoảng ~1 ms tổng thời gian suy luận.

---

## 8. Trả Lời Năm Câu Hỏi Trọng Tâm

1. **Panel thực sự có bao nhiêu mẫu AI manipulation được xác minh?**
   - **Chính xác là 0 mẫu**. Cả 20 mẫu giả mạo chỉ có báo cáo mô tả can thiệp văn bản số (redaction, date alteration, word substitution); không có metadata hay bằng chứng nào chứng minh có mô hình Generative AI (diffusion/inpainting) được sử dụng.
2. **Có nguy cơ format/history confounding không?**
   - **Có, nguy cơ rất cao**. 100% mẫu forged là JPEG 4:2:0 và 100% mẫu authentic là PNG lossless. Mô hình có thể dựa vào các dấu vết tần số hoặc khối nén để phân loại nhãn nguồn thay vì nhận diện can thiệp nội dung văn bản.
3. **Sau symmetric JPEG (C1), các quan sát P0/P1/P2 còn tồn tại đến đâu?**
   - **Vẫn tồn tại vững chắc**. Việc nén JPEG đối xứng trước khi resize không làm thay đổi các quyết định phân loại (0 flips trên 5/6 mô hình; 1 flip trên A0-local P2). Hiệu ứng vượt trội của P1 Letterbox so với P0 Center-Crop (+0.0920 F1) giữ nguyên 100%. P2 Multi-Tile vẫn bùng nổ báo động giả (FPR 35%–45%).
4. **Có lỗi implementation/scoring/timing thực sự không?**
   - **Implementation logic**: Khớp đúng khế ước toán học và cấu trúc hình học.
   - **Timing error**: Lỗi diễn giải nghiêm trọng khi sao chép thời gian trích xuất dùng chung sang từng dòng mô hình, che giấu sự thật là scoring chỉ tốn < 0.2 ms.
   - **Epistemic errors**: 6 phát biểu khẳng định vượt quá bằng chứng thực nghiệm đã được lập sổ cái đính chính đầy đủ tại Mục 2.
5. **Dữ liệu đã đủ để đề xuất training chưa, và cho bài toán nào?**
   - **CHƯA ĐỦ**. Dữ liệu hiện tại ($N=40$ từ RealText-V2) hoàn toàn chưa xác minh được nguồn gốc AI tạo sinh, không có cấu trúc ghép cặp 1:1, và bị nhiễu định dạng 100% (JPEG vs PNG). Chưa đủ cơ sở khoa học để đề xuất huấn luyện mô hình phát hiện tài liệu AI hay chuyển đổi bài toán sang segmentation.

---

## 9. Đúng Một Đề Xuất Tiếp Theo Dựa Trên Bằng Chứng

**Khuyến nghị chính thức**:
Hoàn thiện thủ tục tiếp cận có kiểm soát nguồn tài liệu **AIForge-Doc** (Scam-AI, arXiv:2602.20569, hóa đơn CORD/SROIE có ground-truth inpainting bằng Gemini/Ideogram và cặp authentic đối chứng gốc) qua token truy cập hợp lệ, hoặc thiết lập quy trình tạo sinh tài liệu inpainting có kiểm soát (controlled document inpainting generation) trên tập hóa đơn SROIE/CORD sẵn có, nhằm loại bỏ triệt để nhiễu định dạng và xác minh 100% nguồn gốc AI trước khi tiến hành bất kỳ đợt huấn luyện nào.
