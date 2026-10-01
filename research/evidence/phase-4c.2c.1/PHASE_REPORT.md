# Phase 4C.2C.1 — Reconcile Stage 2 Metric Lineage and Calibration Semantics Report

> **Giai đoạn**: Phase 4C.2C.1
> **Mục tiêu**: Kiểm toán và giải quyết mâu thuẫn số liệu giữa báo cáo Phase 4C.2C.0, các artifact thô Stage 2 (`run_receipt.json`, `metrics.json`, `predictions.json`, `epoch_history.json`, `best_checkpoint.pt`), và kết quả phân tích Phase 4C.2C; chuẩn hóa ngữ nghĩa hiệu chuẩn (calibration-in-the-large vs signed confidence calibration gap).
> **Branch**: `research/phase-4c2-finetuning`
> **Evaluation Partition**: `inner_validation` ($91$ unique sources, $182$ balanced samples).
> **Niêm phong Locked-Test**: $0$ truy cập, $0$ evaluation, giữ nguyên trạng thái niêm phong.
> **Huấn luyện mới**: $0$ lượt chạy (0 GPU calls).
> **Sửa đổi raw artifacts**: $0$ tệp thô bị sửa đổi (raw Colab outputs 100% byte-identical).
> **Phán quyết chính thức**: **`PHASE_4C2C_ANALYSIS_RECONCILED`**

---

## 1. Nguồn gốc và Nguyên nhân Chính xác của Chênh lệch C.0 và C

### 1.1. Hiện tượng Quan sát được
Tại giai đoạn đối soát Phase 4C.2C.1, kiểm tra fail-closed tự động đối chiếu giữa bảng Mục 7 của `research/evidence/phase-4c.2c.0/PHASE_REPORT.md` (C.0) và tệp dữ liệu chuẩn tắc `research/evidence/phase-4c.2c/paired_run_metrics.csv` (C) đã phát hiện sai lệch ở cả 15 runs Stage 2, tiêu biểu:
- **N50 seed 42**:
  - Báo cáo C.0 ghi: `best_epoch = 16`, `epochs_completed = 20`, `val_macro_f1 = 0.5401`.
  - Phân tích C ghi: `best_epoch = 7`, `epochs_completed = 12`, `val_macro_f1 = 0.5549316185128159`.
- **N50 seed 1337**:
  - Báo cáo C.0 ghi: `best_epoch = 8`, `epochs_completed = 13`, `val_macro_f1 = 0.5057`.
  - Phân tích C ghi: `best_epoch = 12`, `epochs_completed = 17`, `val_macro_f1 = 0.5546089005710142`.
- **N250 seed 42**:
  - Báo cáo C.0 ghi: `best_epoch = 4`, `epochs_completed = 9`, `val_macro_f1 = 0.5627`.
  - Phân tích C ghi: `best_epoch = 9`, `epochs_completed = 14`, `val_macro_f1 = 0.5578231292517006`.

### 1.2. Kiểm toán Nguồn gốc Nhị phân và Kết luận Nguyên nhân
Kiểm toán độc lập trực tiếp trên archive gốc `execution_9ee7fdb_complete_results.tar.gz` (SHA-256 `609a14bbe683a23b82db5fe98ae3954e984393244746d374bccd23e5c78e43e2`) và 15 thư mục run thô tại `execution_9ee7fdb` đã chứng minh dứt khoát:
1. **Các artifact thô Colab thực tế chưa bao giờ chứa các con số trong bảng C.0**:
   - Tệp `run_receipt.json`, `metrics.json`, `predictions.json`, `epoch_history.json` và `best_checkpoint.pt` trong archive gốc lẫn thư mục giải nén đều ghi nhận đồng nhất: `N50 seed 42` có `best_epoch = 7`, `epochs_completed = 12`, `val_macro_f1 = 0.5549316185128159`.
2. **Bản chất của mâu thuẫn là lỗi soạn thảo văn bản (Markdown Presentation / Drafting Error)**:
   - Trong Phase 4C.2C.0, công cụ ingestion `ingest_phase_4c2_results.py` chỉ làm nhiệm vụ xác minh bảo mật TAR, tính toán băm SHA-256, kiểm tra cấu trúc thư mục và bảo đảm không rò rỉ locked-test; công cụ này không tự động sinh bảng Markdown của Mục 7.
   - Bảng Mục 7 trong `phase-4c.2c.0/PHASE_REPORT.md` được soạn thảo thủ công trong phiên làm việc trước, trong đó người/agent soạn thảo đã ghi các con số giữ chỗ hoặc nháp từ một lần chạy thử nghiệm khác, không trích xuất từ receipts thực tế.
3. **Quá trình Ingestion và Kiểm toán Checksum hoàn toàn hợp lệ**:
   - 100% tệp thô trong archive gốc nguyên vẹn từng bit;
   - Không có hiện tượng đọc nhầm thư mục hay tráo đổi artifact giữa các lần chạy;
   - Báo cáo lịch sử C.0 được bảo toàn nguyên trạng và đi kèm văn bản đính chính chính thức tại `research/evidence/phase-4c.2c.1/PHASE_4C2C0_ERRATUM.md`.

---

## 2. Kiểm chứng Hợp đồng Runner (Runner Contract Trace)

Khảo sát mã nguồn thực thi chính xác tại `ml/training/run_phase_4c2.py` để làm rõ cơ chế sinh và quan hệ giữa các artifact:

1. **Checkpoint được lưu ở epoch nào?**
   *Bằng chứng dòng mã:* Dòng 794–812 trong `run_phase_4c2.py`:
   ```python
   if val_metrics["macro_f1"] > best_val_macro_f1:
       best_val_macro_f1 = val_metrics["macro_f1"]
       best_epoch = epoch
       save_checkpoint(..., epoch=epoch + 1, metrics=val_metrics, path=checkpoint_path, ...)
   ```
   Checkpoint được lưu tại mỗi epoch mà `val_metrics["macro_f1"]` vượt qua giá trị cao nhất trước đó. Thuộc tính `epoch` ghi vào checkpoint là 1-based (`epoch + 1`).

2. **`best_val_macro_f1` lấy từ đâu?**
   *Bằng chứng dòng mã:* Khởi tạo bằng `0.0` tại dòng 751, và được cập nhật bằng `val_metrics["macro_f1"]` tại dòng 795 khi tìm thấy giá trị validation Macro-F1 mới cao hơn.

3. **Sau early stopping, runner có reload best checkpoint không?**
   *Bằng chứng dòng mã:* Dòng 822–825:
   ```python
   # Load best checkpoint for final evaluation
   if checkpoint_path.exists():
       checkpoint = torch.load(checkpoint_path, map_location=device)
       model.load_state_dict(checkpoint["model_state_dict"])
   ```
   **CÓ**. Runner chủ động nạp lại `checkpoint["model_state_dict"]` từ `best_checkpoint.pt` vào model trước khi thực hiện đánh giá cuối cùng và thu thập dự đoán.

4. **`predictions.json` được tạo từ best checkpoint hay model state cuối?**
   *Bằng chứng dòng mã:* Dòng 825–829. Do `model.load_state_dict(checkpoint["model_state_dict"])` được gọi ngay trước dòng 829 (`predictions_data = collect_predictions(model, val_loader, device)`), `predictions.json` **được tạo 100% từ best checkpoint đã nạp lại**, không phải từ trạng thái mô hình tại epoch dừng sớm.

5. **`metrics.json` được tạo từ predictions nào?**
   *Bằng chứng dòng mã:* Dòng 826 và dòng 928–937. Hàm `evaluate(model, val_loader, device, criterion)` được gọi tại dòng 826 ngay trên model đã reload best checkpoint để tạo ra `final_metrics`, sau đó `final_metrics` được serialize thành `metrics.json`. Giá trị `macro_f1` trong `metrics.json` hoàn toàn trùng khớp với giá trị tính lại từ `predictions.json`.

6. **`run_receipt.final_metrics` lấy từ đâu?**
   *Bằng chứng dòng mã:* Dòng 896: `"final_metrics": final_metrics`. Cùng chung nguồn đối tượng từ lời gọi `evaluate()` trên best checkpoint tại dòng 826.

7. **`best_epoch` dùng zero-based hay one-based indexing?**
   *Bằng chứng dòng mã:* Trong vòng lặp huấn luyện, biến `best_epoch = epoch` là zero-based (dòng 796). Khi ghi vào checkpoint metadata (dòng 801) và `run_receipt.json` (dòng 894: `"best_epoch": best_epoch + 1`), nó được chuyển đổi chuẩn hóa thành **one-based indexing**.

8. **`epochs_completed` được tính thế nào?**
   *Bằng chứng dòng mã:* Dòng 893: `"epochs_completed": len(epoch_history)`. Đây là tổng số epoch đã hoàn thành trước khi cơ chế early stopping kích hoạt (hoặc chạm `max_epochs = 20`). Vì `patience = 5`, công thức quy luật thực nghiệm là:
   $$\text{epochs\_completed} = \min(20, \text{best\_epoch} + 5)$$

9. **Có khả năng stale file hoặc resume ghép artifacts từ hai lần chạy khác nhau không?**
   *Bằng chứng dòng mã:* Không. Script vận hành `scripts/phase_4c2_execute_all.sh` thực thi từng run trong một thư mục đầu ra biệt lập, xóa bỏ trạng thái dở dang trước khi chạy runner, và mỗi run chạy đơn nhất từ đầu đến cuối trong một tiến trình Python duy nhất.

10. **`checksums.json` chứng minh các file thuộc cùng run publication như thế nào?**
    *Bằng chứng dòng mã:* Dòng 972–980. Ngay trước khi kết thúc run, runner lặp qua tất cả các file đã sinh trong thư mục (`best_checkpoint.pt`, `run_receipt.json`, `epoch_history.json`, `predictions.json`, `training_history.csv`, `metrics.json`, `environment.json`, `environment-binding.json`, `trainable-parameter-inventory.json`), tính toán kích thước byte và mã băm SHA-256 nguyên tử, rồi niêm phong vào `checksums.json`. Mọi thay đổi hay ghép tệp từ lần chạy khác đều sẽ gây vi phạm checksum ngay lập tức.

---

## 3. Quyết định Điểm cuối Chuẩn tắc (Canonical Endpoint Decision)

* **Điểm cuối chính đã đăng ký**: Inner-validation Macro-F1 của checkpoint được chọn (`best_checkpoint.pt`).
* **Nguồn số liệu chuẩn tắc (Canonical Source)**: Số liệu tính toán trực tiếp từ `predictions.json`, hoàn toàn trùng khớp với `metrics.json` và `run_receipt.json` (chênh lệch $< 10^{-6}$).
* **Kiểm chứng Parity 7 Nguồn**:
  - `receipt.best_val_macro_f1 == receipt.final_metrics.macro_f1`: **15/15 MATCH**
  - `receipt.final_metrics.macro_f1 == metrics.json.macro_f1`: **15/15 MATCH**
  - `metrics.json.macro_f1 == recomputed(predictions.json).macro_f1`: **15/15 MATCH**
  - `metrics.json.macro_f1 == max(epoch_history.val_macro_f1)`: **15/15 MATCH**
  - `metrics.json.macro_f1 == checkpoint.metrics.macro_f1`: **15/15 MATCH**
  - `receipt.best_epoch == checkpoint.epoch`: **15/15 MATCH**
  - `receipt.best_epoch == argmax(epoch_history.val_macro_f1)`: **15/15 MATCH**
* **Kết luận**: Các artifact của cả 15 runs Stage 2 đạt mức toàn vẹn và nhất quán nội bộ 100%. Phân tích đối chứng ghép cặp trong Phase 4C.2C đã đọc đúng nguồn chuẩn tắc này.

---

## 4. Bảng Đối chiếu Chi tiết 15 Runs Stage 2 (Trước và Sau Đối soát)

Dưới đây là bảng đối chiếu giữa số liệu trình bày sai trong báo cáo C.0 và số liệu thực tế chuẩn tắc trong các artifact thô và Phase 4C.2C:

| Run ID | Cohort $N$ | Seed | C.0 (Mục 7) Best Epoch | Canonical Best Epoch | C.0 Epochs Done | Canonical Epochs Done | C.0 Val Macro-F1 | Canonical / Phase 4C.2C Macro-F1 | Checkpoint SHA-256 (8 ký tự đầu) | Parity Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `n50_seed_42` | 50 | 42 | 16 | **7** | 20 | **12** | 0.5401 | **0.554932** | `5dc5a62d` | VERIFIED |
| `n50_seed_1337` | 50 | 1337 | 8 | **12** | 13 | **17** | 0.5057 | **0.554609** | `cca31fa3` | VERIFIED |
| `n50_seed_2025` | 50 | 2025 | 1 | **4** | 6 | **9** | 0.5445 | **0.537959** | `020b2245` | VERIFIED |
| `n50_seed_3407` | 50 | 3407 | 1 | **1** | 6 | **6** | 0.5366 | **0.520806** | `876d7585` | VERIFIED |
| `n50_seed_9001` | 50 | 9001 | 2 | **5** | 7 | **10** | 0.5387 | **0.539816** | `c9d0a015` | VERIFIED |
| `n100_seed_42` | 100 | 42 | 2 | **5** | 7 | **10** | 0.5739 | **0.565921** | `13a216fc` | VERIFIED |
| `n100_seed_1337` | 100 | 1337 | 15 | **15** | 20 | **20** | 0.5694 | **0.575028** | `b019ee67` | VERIFIED |
| `n100_seed_2025` | 100 | 2025 | 4 | **8** | 9 | **13** | 0.5518 | **0.609312** | `15e4f4f5` | VERIFIED |
| `n100_seed_3407` | 100 | 3407 | 3 | **7** | 8 | **12** | 0.5471 | **0.560386** | `3fe9eb80` | VERIFIED |
| `n100_seed_9001` | 100 | 9001 | 4 | **9** | 9 | **14** | 0.5692 | **0.572261** | `8c6ff7a5` | VERIFIED |
| `n250_seed_42` | 250 | 42 | 4 | **9** | 9 | **14** | 0.5627 | **0.557823** | `9153ea3e` | VERIFIED |
| `n250_seed_1337` | 250 | 1337 | 3 | **3** | 8 | **8** | 0.5543 | **0.567195** | `5a74ef4b` | VERIFIED |
| `n250_seed_2025` | 250 | 2025 | 7 | **12** | 12 | **17** | 0.5619 | **0.568787** | `90cf613d` | VERIFIED |
| `n250_seed_3407` | 250 | 3407 | 2 | **2** | 7 | **7** | 0.5684 | **0.586902** | `687f5d47` | VERIFIED |
| `n250_seed_9001` | 250 | 9001 | 3 | **6** | 8 | **11** | 0.5539 | **0.564870** | `2ea238e8` | VERIFIED |

---

## 5. Chuẩn hóa Ngữ nghĩa Hiệu chuẩn (Calibration Semantics Hotfix)

Nhằm đảm bảo sự chặt chẽ về mặt khoa học và toán học, hai khái niệm hiệu chuẩn trước đây thường bị dùng lẫn lộn đã được phân tách rõ ràng:

### 5.1. Định nghĩa Toán học
1. **Calibration-in-the-large**:
   $$\text{CITL} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i) = \text{mean}(p_{\text{positive}} - y)$$
   - Đo lường độ lệch trung bình của xác suất lớp dương so với tỷ lệ xuất hiện thực tế (prevalence).
   - Tuyệt đối **không gọi chỉ số này là bằng chứng về overconfidence hoặc underconfidence**.
2. **Signed confidence calibration gap**:
   $$\text{Gap} = \frac{1}{N} \sum_{i=1}^N (\text{confidence}_i - \text{correctness}_i)$$
   - Trong đó: $\text{confidence}_i = \max(p_i, 1 - p_i)$ và $\text{correctness}_i = \mathbb{I}(\hat{y}_i = y_i)$.
   - **Chỉ chỉ số này mới được dùng để kết luận về độ tự tin của mô hình**:
     - $\text{Gap} > 0$: Mô hình có xu hướng tự tin quá mức (*overconfidence*);
     - $\text{Gap} < 0$: Mô hình có xu hướng thận trọng / thiếu tự tin (*underconfidence / conservatism*);
     - $\text{Gap} \approx 0$: Độ tin cậy trung bình phản ánh sát độ chính xác thực nghiệm.

### 5.2. Kết quả Đo lường Chuẩn tắc
| Cohort $N$ | Stage 1 CITL | Stage 2 CITL | $\Delta$ CITL | Stage 1 Conf Gap | Stage 2 Conf Gap | $\Delta$ Conf Gap | Stage 1 ECE | Stage 2 ECE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **N = 50** | $-0.0052 \pm 0.0096$ | $+0.0034 \pm 0.0080$ | $+0.0086$ | $-0.0138 \pm 0.0091$ | $-0.0167 \pm 0.0134$ | $-0.0029$ | $0.0171 \pm 0.0060$ | $0.0185 \pm 0.0118$ |
| **N = 100** | $-0.0016 \pm 0.0066$ | $+0.0059 \pm 0.0123$ | $+0.0076$ | $-0.0450 \pm 0.0085$ | $-0.0375 \pm 0.0224$ | $+0.0075$ | $0.0450 \pm 0.0085$ | $0.0418 \pm 0.0210$ |
| **N = 250** | $+0.0013 \pm 0.0129$ | $+0.0012 \pm 0.0142$ | $-0.0001$ | $-0.0480 \pm 0.0158$ | $-0.0332 \pm 0.0194$ | $+0.0148$ | $0.0480 \pm 0.0158$ | $0.0421 \pm 0.0122$ |

*Nhận xét*: Không quan sát thấy xu hướng overconfidence trung bình trên inner-validation; signed confidence calibration gap âm gợi ý xu hướng underconfidence nhẹ. Đây là bằng chứng phát triển mô hình, không phải kết luận confirmatory.

### 5.3. Cải tiến Biểu đồ Độ tin cậy (Reliability Diagrams)
Biểu đồ `figures/calibration_comparison.svg` và `.png` đã được nâng cấp toàn diện:
- Bổ sung thanh sai số $\pm 1$ SD (Seed Variability) cho từng bin không rỗng trên cả SVG và PNG;
- Cắt đứt đường nối qua các bin rỗng (không vẽ đường nối giả tạo qua khoảng không có dữ liệu);
- Bổ sung chú thích cấu hình chia bin tường minh ($10$ uniform bins $[0.0, 0.1), \dots, [0.9, 1.0]$);
- Đánh dấu bin thưa mẫu ($\text{count} < 2$);
- Legend rõ ràng cho Stage 1 (Frozen), Stage 2 (Fine-Tuning), Perfect Calibration, và Error Bar ($\pm 1$ SD across 5 seeds).

---

## 6. Chuẩn hóa Ngôn từ về Baseline (Baseline Wording Alignment)

1. **Uninformative Metadata Placeholder Baseline (0.5000)**:
   - Con số $0.5000$ được định danh chuẩn tắc là **uninformative metadata placeholder baseline**, phản ánh giá trị phỏng đoán ngẫu nhiên trên phân phối nhãn cân bằng khi chưa có mô hình metadata.
   - Không được gọi đây là "mô hình metadata hoàn chỉnh".
   - Không sử dụng baseline này để kết luận metadata pháp chứng là vô dụng.
2. **Stratified Dummy Baseline (0.4749)**:
   - Nguồn gốc thực nghiệm rõ ràng: Giá trị Macro-F1 trung bình của Dummy classifier ngẫu nhiên phân tầng theo tỷ lệ lớp trên tập `inner_validation` qua 5 hạt giống ngẫu nhiên ($0.4749 \pm 0.0325$: seed 42: $0.4940$, 1337: $0.4173$, 2025: $0.4820$, 3407: $0.4930$, 9001: $0.4883$).

---

## 7. Kết quả Thống kê Điểm cuối Chính Ghép cặp Cuối cùng

Dựa trên dữ liệu chuẩn tắc, kết quả kiểm định điểm cuối chính $\Delta \text{ Macro-F1 paired} = \text{Macro-F1}_{\text{Stage 2}} - \text{Macro-F1}_{\text{Stage 1}}$ qua $n=5$ paired seeds:
- **$N = 50$**:
  - $\Delta = +0.0094 \pm 0.0082$, 95% CI $[-0.0008, +0.0196]$
  - Paired $t$-test: $t = 2.58$, $p_{\text{raw}} = 0.0628$, $p_{\text{Holm}} = 0.1884$
  - Exact sign-flip permutation test: $p_{\text{raw}} = 0.1250$, $p_{\text{Holm}} = 0.3750$
  - Seed count: 4 Stage 2 tốt hơn / 1 Stage 1 tốt hơn
- **$N = 100$**:
  - $\Delta = +0.0038 \pm 0.0194$, 95% CI $[-0.0202, +0.0279]$
  - Paired $t$-test: $t = 0.44$, $p_{\text{raw}} = 0.6833$, $p_{\text{Holm}} = 0.8925$
  - Exact sign-flip permutation test: $p_{\text{raw}} = 0.8750$, $p_{\text{Holm}} = 0.8750$
  - Seed count: 2 Stage 2 tốt hơn / 3 Stage 1 tốt hơn
- **$N = 250$**:
  - $\Delta = +0.0064 \pm 0.0169$, 95% CI $[-0.0147, +0.0275]$
  - Paired $t$-test: $t = 0.84$, $p_{\text{raw}} = 0.4463$, $p_{\text{Holm}} = 0.8925$
  - Exact sign-flip permutation test: $p_{\text{raw}} = 0.3125$, $p_{\text{Holm}} = 0.6250$
  - Seed count: 4 Stage 2 tốt hơn / 1 Stage 1 tốt hơn

*Kết luận thống kê*: Mọi 95% CI đều cắt 0, mọi giá trị $p$ (thô và hiệu chỉnh Holm) đều lớn hơn 0.05. Bằng chứng thực nghiệm trên `inner_validation` không cho thấy sự vượt trội có ý nghĩa thống kê của quy trình Stage 2 partial fine-tuning so với Stage 1 frozen backbone linear probe.

---

## 8. Bảng Tổng kết Ràng buộc Kỹ thuật (Invariants Accounting)

| Chỉ số / Ràng buộc | Giá trị Ghi nhận | Trạng thái Tuân thủ |
| :--- | :---: | :---: |
| Raw artifacts modified | **0** | Đạt chuẩn bất biến |
| New training runs | **0** | Đạt chuẩn bất biến |
| GPU inference calls | **0** | Đạt chuẩn bất biến |
| Locked-test evaluations | **0** | Niêm phong tuyệt đối |
| Absolute Windows paths in Git evidence | **0** | Đạt chuẩn portable |
| 15 Stage 2 runs parity across 6 raw sources | **15/15 (100%)** | Parity verified |
| C.0 Erratum published | **Yes** (`PHASE_4C2C0_ERRATUM.md`) | Đạt chuẩn trung thực |
| Final Phase Verdict | **`PHASE_4C2C_ANALYSIS_RECONCILED`** | **HOÀN THÀNH** |
