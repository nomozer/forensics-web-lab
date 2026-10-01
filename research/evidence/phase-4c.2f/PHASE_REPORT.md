# BÁO CÁO NGHIÊN CỨU PHASE 4C.2F — XÂY DỰNG, KIỂM THỬ VÀ NIÊM PHONG LOCKED-TEST CONFIRMATORY EVALUATOR

---

## 1. Mục tiêu và Định vị Nghiên cứu

Phase 4C.2F kế thừa trực tiếp kết quả đăng ký trước (preregistration) từ Phase 4C.2E đã được hợp nhất vào nhánh `main` qua Pull Request #4 (merge commit `8a37966`). Mục tiêu cốt lõi của Phase 4C.2F là:
1. Xây dựng công cụ đánh giá xác nhận (`confirmatory evaluator`) tuân thủ nghiêm ngặt giao thức khoa học đã đăng ký trước.
2. Thiết lập cơ chế kiểm soát truy cập và phòng vệ lỗi đóng (`fail-closed access guard`) ngăn chặn tuyệt đối việc mở niêm phong dữ liệu locked-test khi chưa có phê duyệt chính thức từ con người (`human authorization`).
3. Khóa toàn bộ định nghĩa toán học của các chỉ số (Macro-F1, Balanced Accuracy, AUROC, Brier Score, ECE 10 uniform bins, CITL, Signed Confidence Calibration Gap).
4. Khóa thuật toán lấy mẫu tái lập cụm nguồn (source-cluster bootstrap) 10,000 replicates với bộ sinh số ngẫu nhiên PCG64 seed `20261002`, bảo toàn tính đa bội cụm (multiplicity) và duy trì 686 quan sát mỗi replicate.
5. **Tuyệt đối không mở, đọc, mount, duyệt, băm hoặc đánh giá locked-test trong phase này.** Giữ nguyên vẹn các bộ đếm: `locked_test_accesses = 0`, `completed_unsealing_sessions = 0`, `completed_model_evaluations = 0`, `new_training_runs = 0`, `gpu_inference_calls = 0`.

---

## 2. Lịch sử Tích hợp & Ba Tầng Provenance

Toàn bộ công việc được thực hiện trên nhánh độc lập:
* **Nhánh nghiên cứu**: `research/phase-4c2f-locked-test-evaluator`
* **Commit gốc trên main**: `8a379665bc8db3722a46618e47db4806a6ea7244` (PR #4 merge commit)
* **Ba tầng Provenance lịch sử**:
  1. *Integration Baseline*: `8a379665bc8db3722a46618e47db4806a6ea7244` (tích hợp Phase 4C.2E).
  2. *Candidate Stage 1 Snapshot*: `79bb11527d900fd387de1f41f2010c4152b7fea7` (môi trường huấn luyện sinh ra 5 checkpoint candidate Stage 1 N=250).
  3. *Historical Stage 2 Snapshot*: `9ee7fdbb88fad16167f5790b5105867747801372` (giao thức fine-tuning lịch sử đã được kiểm định và loại trừ khỏi ứng viên chính).

---

## 3. Bảo toàn Giao thức Đăng ký trước (Preserved Preregistration)

Toàn bộ các quyết định khoa học từ Phase 4C.2E được bảo toàn nguyên vẹn:
* **Mô hình candidate chính thức**: `stage1_frozen_backbone_linear_probe`
* **Cỡ mẫu huấn luyện (Sample Size)**: $N = 250$ theo nguyên tắc phát triển dữ liệu tối đa (`maximum_development_data`), loại trừ việc chọn $N=100$ dựa trên điểm inner-validation cao hơn.
* **Danh sách hạt giống (Seeds)**: Đúng 5 hạt giống Stage 1 $N=250$: `[42, 1337, 2025, 3407, 9001]`.
* **Số lượng checkpoints**: 5 checkpoints, đối soát khớp 100% mã băm SHA-256 và kích thước 5,627,375 bytes.
* **Cấm chọn hạt giống tốt nhất**: Không cherry-picking, không loại seed xấu.
* **Cấm tổ hợp xác suất hậu nghiệm**: Không ensemble probability trước khi đánh giá.
* **Cấm huấn luyện lại**: Không dùng inner-validation để cập nhật trọng số.
* **Primary Endpoint**: Trung bình số học (arithmetic mean) của Macro-F1 trên 5 checkpoints.
* **Bootstrap**: 10,000 replicates, PCG64 seed `20261002`, resampling theo `unique_source_id`.
* **Tiêu chuẩn thành công**: Cận dưới khoảng tin cậy 95% bootstrap (CI lower bound) $> 0.5000$.
* **Ngưỡng tham chiếu**: $0.5000$ được định danh nghiêm ngặt là `balanced_binary_uninformative_reference` (không gọi là metadata baseline).
* **Secondary Endpoints**: Chỉ mang tính mô tả / khám phá (`descriptive / exploratory`).
* **Hiệu chuẩn (Calibration)**: Chỉ đánh giá (`evaluation-only`), cấm fitting, cấm Temperature Scaling, cấm tối ưu ngưỡng.
* **Giới hạn phiên**: Tối đa 1 phiên mở niêm phong (`unsealing session`), đúng 5 lần đánh giá mô hình (`model evaluations`).

---

## 4. Định nghĩa Chỉ số và Hợp đồng Đánh giá (Metric & Evaluator Contracts)

1. **Gán nhãn nhị phân**: `authentic = 0`, `ai_edited = 1`.
2. **Xác suất dương (Positive Probability)**: $p_1 = \text{softmax}(\text{logits}, \text{dim}=1)[:, 1]$.
3. **Phân lớp dự đoán (Predicted Class)**: $\hat{y} = \text{argmax}(\text{logits})$, tương đương ngưỡng $0.5$ trên xác suất dương. Cấm tối ưu hóa ngưỡng.
4. **Macro-F1**: `sklearn.metrics.f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)`.
5. **Balanced Accuracy & AUROC**: Định nghĩa chuẩn tắc; fail-closed nếu cohort thiếu một trong hai lớp.
6. **Confusion Matrix**: `confusion_matrix(y_true, y_pred, labels=[0, 1])` xuất ma trận $[[TN, FP], [FN, TP]]$.
7. **Độ nhạy và độ đặc hiệu**:
   - Lớp 0 (`authentic`): $\text{sensitivity} = TN / (TN + FP)$, $\text{specificity} = TP / (TP + FN)$.
   - Lớp 1 (`ai_edited`): $\text{sensitivity} = TP / (TP + FN)$, $\text{specificity} = TN / (TN + FP)$.
8. **Expected Calibration Error (ECE)**:
   - Đúng 10 bins đều nhau từ $0.0$ đến $1.0$ với bin edges: `[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`.
   - 9 bins đầu: $[e_i, e_{i+1})$; bin cuối: $[0.9, 1.0]$.
   - Bins rỗng không đóng góp vào ECE.
   - ECE là tổng có trọng số của khoảng cách tuyệt đối giữa độ tự tin trung bình của lớp dự đoán và độ chính xác thực tế.
9. **Calibration-in-the-large (CITL)**: $\text{mean}(p_1 - y_{\text{true}})$.
10. **Signed Confidence Calibration Gap**: $\text{mean}(\text{confidence} - \text{correctness})$.

---

## 5. Thuật toán Lấy mẫu Tái lập Cụm Nguồn (Source-Cluster Bootstrap)

1. **Bộ sinh số ngẫu nhiên**: `numpy.random.Generator(numpy.random.PCG64(20261002))`.
2. **Đơn vị lấy mẫu**: `unique_source_id` (343 cụm nguồn).
3. **Bảo toàn cấu trúc cụm và tính đa bội**:
   - Khi một source được lấy mẫu $k$ lần, cả 2 mẫu (authentic và ai_edited) của source đó đều xuất hiện đúng $k$ lần.
   - Tuyệt đối không dùng `set()` hay `isin()` làm mất tính lặp lại.
   - Mỗi replicate có đúng 686 dòng quan sát.
4. **Quy trình tính toán từng replicate**:
   - Tính riêng Macro-F1 cho từng checkpoint trong số 5 checkpoints.
   - Lấy trung bình số học của 5 giá trị Macro-F1.
   - Tuyệt đối không lấy trung bình xác suất trước khi tính F1.
5. **Khoảng tin cậy 95%**: `numpy.quantile(bootstrap_means, [0.025, 0.975], method="linear")`.
6. **Phạm vi suy diễn khoa học**: Khoảng tin cậy thể hiện độ bất định do lấy mẫu cụm nguồn (`source-sampling uncertainty`) có điều kiện trên 5 checkpoints cố định; không bao phủ toàn bộ độ biến thiên do huấn luyện hạt giống.

---

## 6. Kiến trúc Bảo vệ Lỗi đóng (Fail-Closed Evaluator)

Module `ml/evaluation/locked_test_evaluator.py` và CLI `ml/evaluation/run_phase_4c2f_evaluator.py` thực hiện các hàng rào an ninh:
1. **Kiểm tra phê duyệt con người (`Human Authorization Gate`)**:
   - Bắt buộc phải có file `HUMAN_UNSEALING_AUTHORIZATION.json` hợp lệ với trạng thái `AUTHORIZED`, đúng protocol, sample size 250, 5 seeds, và mã định danh người duyệt. Nếu thiếu, chương trình dừng ngay lập tức (`PermissionError`).
2. **Kiểm tra cô lập mạng (`Network Isolation Guard`)**:
   - Thực hiện kiểm tra socket ra mạng ngoài. Nếu kết nối thành công, chương trình lập tức dừng với `RuntimeError` yêu cầu môi trường air-gap hoàn toàn.
3. **Kiểm tra gắn đĩa chỉ đọc (`Read-Only Data Mount Guard`)**:
   - Thử nghiệm canary write trên thư mục locked-test. Nếu ghi được, dừng ngay với `RuntimeError` yêu cầu mount read-only.
4. **Kiểm tra toàn vẹn checkpoint (`Checkpoint Integrity Gate`)**:
   - Giải quyết 5 checkpoints và kiểm tra kích thước đúng 5,627,375 bytes và mã băm SHA-256 khớp 100% với danh mục đăng ký.
5. **Sổ nhật ký truy cập nối tiếp có băm chuỗi (`Append-Only Hash-Chained Access Ledger`)**:
   - File `locked_test_access_ledger.jsonl` lưu vết từng sự kiện kèm mã băm SHA-256 chuỗi (`prev_entry_hash`).
   - Tự động phát hiện và chặn đứng mọi hành vi chỉnh sửa hoặc chèn log trái phép.
6. **Thời điểm tăng bộ đếm**:
   - `completed_unsealing_sessions` tăng từ 0 lên 1 ngay trước lần đọc đầu tiên từ locked-test.
   - `completed_model_evaluations` tăng ngay sau khi inference của một checkpoint xuất predictions thành công.
   - Chặn đứng lần đánh giá thứ 6 (`max=5`) và phiên mở niêm phong thứ 2 (`max=1`).
7. **Ghi dự đoán nguyên tử và phân biệt sự cố (`Atomic Staging & Crash Discrimination`)**:
   - Predictions ghi vào file tạm `.part`, flush, fsync, rồi rename nguyên tử qua `os.replace`.
   - Pre-inference crash (trước khi predictions sinh ra): cho phép retry mà không tăng bộ đếm đánh giá.
   - Post-inference crash (predictions đã tồn tại): fail-closed, tuyệt đối cấm chạy lại âm thầm.

---

## 7. Kết quả Kiểm thử & Chạy thử Giả lập (Synthetic Dry Run)

1. **Test Suite Độc lập (`ml/tests/test_phase_4c2f_evaluator.py`)**:
   - Đạt **30/30 tests PASS** (26 unit/contract tests + 4 security & fault-injection tests).
   - 0 truy cập locked-test, 0 cuộc gọi mạng, 0 training runs.
2. **Kiểm thử Giả lập Đầu cuối (`Synthetic Dry Run`)**:
   - Thực hiện trên 343 mock sources (686 mock samples).
   - Hoàn thành đầy đủ 5 model evaluations, ghi nhận ledger, tính bootstrap CI (1,000 replicates), và xuất receipt hợp lệ tại `research/evidence/phase-4c.2f/synthetic_dry_run_receipt.json`.

---

## 8. Bảng Tổng kết Các Bộ đếm & Trạng thái Khóa

| Chỉ số / Bộ đếm | Trạng thái Phase 4C.2F | Giới hạn Cho phép | Trạng thái Tuân thủ |
| :--- | :--- | :--- | :--- |
| `locked_test_accesses` | **0** | 0 (trước phê duyệt) | **TUÂN THỦ TUYỆT ĐỐI** |
| `completed_unsealing_sessions` | **0** | 0 (trước phê duyệt) | **TUÂN THỦ TUYỆT ĐỐI** |
| `completed_model_evaluations` | **0** | 0 (trước phê duyệt) | **TUÂN THỦ TUYỆT ĐỐI** |
| `new_training_runs` | **0** | 0 | **TUÂN THỦ TUYỆT ĐỐI** |
| `gpu_inference_calls` | **0** | 0 | **TUÂN THỦ TUYỆT ĐỐI** |
| Checkpoints resolved | **5/5** | 5 | **100% SHA-256 MATCH** |
| Windows paths in Git evidence | **0** | 0 | **TUÂN THỦ TUYỆT ĐỐI** |

---

## 9. Phán quyết Pre-Unsealing

```
READY_FOR_HUMAN_ONE_TIME_UNSEALING_APPROVAL
```
Mọi thành phần kỹ thuật của evaluator đã được xây dựng, kiểm thử và niêm phong mật mã. Hệ thống ở trạng thái đóng băng, sẵn sàng chờ quyết định phê duyệt bằng văn bản từ con người để bước vào Phase 4C.2G (thực thi mở niêm phong một lần và đánh giá xác nhận).
