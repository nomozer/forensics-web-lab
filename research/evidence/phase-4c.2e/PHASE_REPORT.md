# Báo cáo Nghiên cứu: PHASE 4C.2E — LOCKED-TEST CONFIRMATORY PROTOCOL PREREGISTRATION

> **Tài liệu đăng ký trước chính thức cho đánh giá kiểm chuẩn trên tập locked-test (Confirmatory Protocol Preregistration).**<br>
> **Thời điểm đăng ký**: Post model development — strictly prior to locked-test unsealing.<br>
> **Branch**: `research/phase-4c2e-locked-test-preregistration`<br>
> **Integration Baseline Commit**: `143e02cac2d2fa7a49b34f94271f2e19378a9c0f`<br>
> **Trạng thái Locked-Test Partition**: `SEALED` (0 accesses to date)<br>
> **Phán quyết**: **`READY_FOR_HUMAN_UNSEALING_APPROVAL`**

---

## 1. Mục tiêu & Nguyên tắc Đăng ký Trước (Preregistration Principles)

Mục tiêu của Phase 4C.2E là thiết lập một biên bản đăng ký trước bất biến (*prospective confirmatory evaluation protocol*) cho đánh giá mô hình trên phân vùng kiểm thử niêm phong (`locked_test`), nhằm ngăn chặn triệt để mọi hình thức p-hacking, cherry-picking, lựa chọn cỡ mẫu hoặc seed hậu nghiệm, rò rỉ dữ liệu hoặc điều chỉnh siêu tham số sau khi dữ liệu kiểm thử được mở niêm phong.

### Ba nguyên tắc bất biến:
1. **Zero Access During Preregistration**: Không đọc, mount, giải nén, duyệt thư mục hoặc tính lại mã băm nội dung của `locked_test` trong phase này. Số lượt truy cập `locked_test_accesses` duy trì bằng 0 tuyệt đối.
2. **Frozen Protocol & Checkpoints**: Đóng băng chính xác candidate protocol, cỡ mẫu $N=250$, 5 checkpoint hạt giống Stage 1 và danh sách mã băm SHA-256 đối soát.
3. **Pre-specified Statistical Inference**: Đăng ký trước chỉ số chính (Primary Endpoint), phương pháp khoảng tin cậy 95% Bootstrap cụm nguồn (Source-Cluster Bootstrap), seed ngẫu nhiên xác định và ngưỡng tham chiếu khoa học không phụ thuộc kết quả quan sát.

---

## 2. Phân Tầng Nguồn Gốc và Phiên Bản (Three-Layer Provenance)

Hệ thống phân định rạch ròi 3 lớp nguồn gốc để bảo toàn tính độc lập của các snapshot lịch sử:

* **Integration / Main Baseline**: `143e02cac2d2fa7a49b34f94271f2e19378a9c0f`<br>
  Điểm neo tích hợp chính thức của repository sau khi hoàn thành Phase 4C.2D và kiểm toán ranh giới kiểm thử hermetic CI (PR #3).
* **Candidate Stage 1 Training Snapshot**: `79bb11527d900fd387de1f41f2010c4152b7fea7`<br>
  Commit snapshot gốc chứa mã nguồn runner và môi trường huấn luyện của 15 runs Stage 1 thực thi trên Google Colab T4.
* **Historical Stage 2 Snapshot**: `9ee7fdbb88fad16167f5790b5105867747801372`<br>
  Snapshot môi trường thực thi của 15 runs Stage 2 fine-tuning (đã hoàn thành, phân tích và không được chọn làm primary candidate).

Tuyệt đối không sử dụng các thay đổi dependencies tại HEAD để tính lại hoặc ghi đè ổ khóa môi trường khoa học của các runs lịch sử.

---

## 3. Khóa Candidate Protocol và Checkpoint Binding

### 3.1. Quyết định lựa chọn Protocol
* **Candidate Protocol Chính thức**: `stage1_frozen_backbone_linear_probe`
* **Loại bỏ Stage 2**: Protocol Stage 2 (pre-registered partial fine-tuning) không được đưa vào đánh giá confirmatory trên locked-test do không chứng minh được sự vượt trội có ý nghĩa thống kê trên tập inner-validation tại bất kỳ cỡ mẫu nào (toàn bộ 95% CIs cắt 0, mọi Holm $p > 0.05$) trong Phase 4C.2C và 4C.2D.

### 3.2. Chính sách Cỡ Mẫu (Sample-Size Policy: $N=250$)
* **Lựa chọn cỡ mẫu**: Khóa duy nhất cỡ mẫu $N=250$.
* **Cơ sở khoa học**: Lựa chọn $N=250$ dựa trên nguyên tắc dữ liệu phát triển tối đa (*maximum-development-data principle*), vì đây là tập dữ liệu huấn luyện lớn nhất đã đăng ký trước ($N=250$ sources) khả dụng trước khi mở locked-test.
* **Minh bạch phương pháp luận**: Không chọn $N=100$ dù giá trị trung bình trên inner-validation của $N=100$ có thể cao hơn một chút (0.5728 vs 0.5627). Việc chọn $N=250$ là chính sách tiến cứu (*prospective policy*) được đóng băng trước khi tiếp cận dữ liệu kiểm thử.

### 3.3. Chính sách Hạt Giống & Checkpoints
* **Bộ hạt giống**: Đánh giá toàn bộ 5 hạt giống Stage 1 tại $N=250$: `[42, 1337, 2025, 3407, 9001]`.
* **Cấm chọn lọc**: Tuyệt đối không chọn "best seed", không loại bỏ "bad seed", không ensemble hậu nghiệm xác suất trừ khi đã có quy tắc từ trước. Báo cáo độc lập 5 mô hình và chỉ số tổng hợp đã đăng ký.
* **Không huấn luyện lại**: Không huấn luyện lại bằng `inner_validation`, không tạo thêm checkpoint mới (`new_training_runs = 0`).

### 3.4. Bảng Ràng Buộc Mã Băm 5 Candidate Checkpoints

| Seed | Run ID | Checkpoint SHA-256 | Kích thước (bytes) | Best Epoch | Val Macro-F1 | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 42 | `phase4c1-stage1-n250-seed42` | `c941f42ed00adb098962ddb43c0f2f7d897e48e20957cddc0c491243f8730c92` | 5,627,375 | 6 | 0.576910 | completed |
| 1337 | `phase4c1-stage1-n250-seed1337` | `69e706f9062f050ccbfd2affb72d63d5dde9d12f02c1e64c762b0ea6719073f1` | 5,627,375 | 5 | 0.547321 | completed |
| 2025 | `phase4c1-stage1-n250-seed2025` | `92ce5ee986d487fdf14374842067d0684fc7669f5ecb32c7d603f6813942ceee` | 5,627,375 | 5 | 0.544930 | completed |
| 3407 | `phase4c1-stage1-n250-seed3407` | `4897821ef0a97df9f1c51acde8d4ac01d383aa3a7b9e7ee55bbc3b3108847868` | 5,627,375 | 4 | 0.583371 | completed |
| 9001 | `phase4c1-stage1-n250-seed9001` | `5f0f8803adcb7eec88d47e398ef8b2002b46e740c263b12abc2ba392822a91c3` | 5,627,375 | 6 | 0.561039 | completed |

* Dataset Bundle Archive SHA-256: `d49a106f0c4991ca8d79776277cbf7331df209157725c438288720dc42226a27`
* Dataset Bundle Content SHA-256: `c365c812cc814097f11b9e5ed5c82e672015e2ba093f09f975df0a2a01229e9b`
* Dataset Bundle Manifest SHA-256: `411e35da80843a10f7bb22e33efac8a4d1db70110dbda91cadbbf39532c9312d`
* Configuration SHA-256: `e03c07dae05a2402416abb0c60480a8cd0a35d060de3bb138a09d0bec0a85dc9`

---

## 4. Kế Hoạch Định Lượng & Suy Luận Thống Kê (Confirmatory Metrics Plan)

### 4.1. Primary Endpoint
* **Chỉ số đo lường chính**: Macro-F1 trên tập `locked_test`.
* **Ước lượng tổng hợp (Aggregate Primary Estimate)**: Trung bình số học (Arithmetic Mean) của Macro-F1 qua đúng 5 checkpoint Stage 1 N=250:
  $$\overline{\text{Macro-F1}} = \frac{1}{5} \sum_{k=1}^5 \text{Macro-F1}_k$$
* **Không áp dụng Paired t-test giữa 5 seeds**: Cả 5 mô hình đều được đánh giá trên cùng một tập dữ liệu kiểm thử locked-test, vi phạm giả định quan sát độc lập cần thiết cho suy luận kiểm chuẩn thông qua t-test.

### 4.2. Khoảng Tin Cậy 95% Bootstrap Cụm Nguồn (Source-Cluster Bootstrap CI)
* **Đơn vị tái lấy mẫu (Resampling Unit)**: `unique source_id` (343 sources).
* **Bảo toàn cặp ghép (Pair-Aware Preservation)**: Khi một `source_id` được chọn ngẫu nhiên có hoàn lại, toàn bộ cặp ảnh ghép (`authentic` và `ai_edited`) của source đó được giữ nguyên vẹn trong tập mẫu bootstrap.
* **Cố định 5 Checkpoints**: 5 checkpoints được giữ cố định; chỉ có tập dữ liệu kiểm thử được tái lấy mẫu.
* **Số lượt tái lấy mẫu**: 10,000 bootstrap replicates.
* **Seed ngẫu nhiên đóng băng**: `RNG seed = 20261002`.
* **Khoảng tin cậy**: Percentile 95% CI (lấy phân vị 2.5% và 97.5% của 10,000 giá trị $\overline{\text{Macro-F1}}$).
* **Quy tắc bất biến**: Tuyệt đối không thay đổi phương pháp ước lượng khoảng tin cậy sau khi quan sát kết quả.

### 4.3. Ngưỡng Tham Chiếu Kiểm Chuẩn (Confirmatory Reference: 0.5000)
* **Tên gọi chuẩn tắc**: `balanced binary uninformative reference` (Ngưỡng tham chiếu không mang thông tin của bài toán nhị phân cân bằng).
* **Tuyệt đối cấm**: Không được gọi 0.5000 là "trained metadata baseline".
* **Quy tắc quyết định kiểm chuẩn (Confirmatory Decision Rule)**:
  * **Thành công (Confirmatory Success)**: Cận dưới của khoảng tin cậy 95% Bootstrap ($\text{CI}_{\text{lower}}$) của $\overline{\text{Macro-F1}}$ phải lớn hơn 0.5000 một cách nghiêm ngặt:
    $$\text{CI}_{\text{lower}} > 0.5000$$
  * **Chưa đạt (Inconclusive / Lack of Confirmatory Evidence)**: Nếu $\text{CI}_{\text{lower}} \le 0.5000$, kết luận chưa có bằng chứng kiểm chuẩn xác nhận mô hình phân biệt tốt hơn ngưỡng đoán ngẫu nhiên trên phân vùng kiểm thử chưa thấy.
* Không được nới lỏng hay thay đổi ngưỡng sau khi mở niêm phong.

### 4.4. Secondary Endpoints (Mô tả & Khám phá)
* **Danh mục chỉ số phụ**: Balanced Accuracy, AUROC, Brier Score, ECE, Ma trận nhầm lẫn (Confusion Matrix), Sensitivity/Recall từng lớp, Specificity từng lớp, các chỉ số riêng lẻ theo từng checkpoint, độ lệch chuẩn giữa các hạt giống (between-seed SD).
* **Phân loại**: Toàn bộ chỉ số phụ được phân loại là **`descriptive_exploratory`** (không đăng ký kiểm định giả thuyết chính thức, không áp dụng hiệu chỉnh đa bội).
* **Giao thức Hiệu chuẩn (Calibration Protocol)**:
  * Chế độ: **Đánh giá thuần túy (Evaluation-only)**, cấm mọi hành vi khớp tham số (no fitting).
  * Cấm Temperature Scaling, cấm Threshold Optimization (áp dụng ngưỡng mặc định argmax / 0.5), cấm Platt scaling hoặc Isotonic regression.
  * Khóa trước 10 bins đều nhau trên $[0, 1]$ với các cạnh: `[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`.
  * Báo cáo tách bạch hai khái niệm: Sai số hiệu chuẩn quy mô lớn ($\text{CITL} = \text{mean}(p - y)$) và Khoảng lệch độ tin cậy có dấu ($\text{Signed Confidence Gap} = \text{mean}(\text{conf} - \text{acc})$).

---

## 5. Rào Chắn Toàn Vẹn & Chống Rò Rỉ (Integrity & Leakage Gates)

Evaluator thực thi trong Phase 4C.2F bắt buộc phải vượt qua các cổng kiểm chuẩn trước khi ghi nhận kết quả:
1. **Kiểm tra số lượng nguồn**: Phải khớp chính xác **343 unique `source_id`** đã niêm phong trong metadata.
2. **Kiểm tra ghép cặp**: Đúng 2 mẫu/nguồn (1 authentic + 1 ai_edited), tổng cộng 686 mẫu.
3. **Kiểm tra nhãn**: Đúng cặp nhãn $\{0, 1\}$ cho mỗi nguồn.
4. **Kiểm tra cách ly phân vùng**: 0 trùng lặp với `development_train` (250 sources) và 0 trùng lặp với `inner_validation` (91 sources).
5. **Kiểm tra chiều ghi**: Tuyệt đối không có đường dẫn ghi đè từ output vào dữ liệu kiểm thử.
6. **Kiểm tra mã băm**: Checkpoint hashes phải khớp 100% với `candidate_checkpoint_binding.json`.
7. **Kiểm tra cách ly mạng**: 0 bytes truyền tải qua mạng trong quá trình suy luận.
8. **Cấm chạy lại tùy tiện**: Không được đánh giá lại mô hình sau khi đã tạo xong predictions hợp lệ.

---

## 6. Giao Thức Mở Niêm Phong Một Lần (One-Time Unsealing Protocol)

Phân tách rạch ròi 2 biến số kế toán truy cập:
* `locked_test_unsealing_sessions`: Số phiên mở niêm phong (Dự kiến tối đa: 1; Hiện tại: **0**).
* `locked_test_model_evaluations`: Số lượt checkpoint được đánh giá (Dự kiến: 5; Hiện tại: **0**).
* `locked_test_accesses`: **0 tuyệt đối**.

### Quy tắc vận hành & Xử lý lỗi (Fault Tolerance Matrix):
1. **Phê duyệt của con người (Human Approval)**: Bắt buộc phải có sự chấp thuận trực tiếp từ người dùng trước khi kích hoạt quy trình unsealing.
2. **Mount chỉ đọc**: Phân vùng locked-test chỉ được mount ở chế độ `read-only`.
3. **Xuất kết quả nguyên tử**: Predictions được ghi vào thư mục tạm `.part`, kiểm tra tính toàn vẹn rồi mới đổi tên nguyên tử sang thư mục run chính thức.
4. **Receipt đồng bộ**: Ghi nhận `run_receipt.json` ngay sau khi mỗi checkpoint hoàn thành.
5. **Sổ cái bất biến**: Nhật ký truy cập (`access ledger`) theo cơ chế chỉ ghi thêm (`append-only`).
6. **Xử lý sự cố (Crash Handling)**:
   * **Crash trước khi suy luận (Pre-inference runtime crash)**: Cho phép dọn dẹp thư mục tạm và thử lại mà không tính vào số lượt đánh giá mô hình; nguyên nhân được ghi vào incident log.
   * **Crash sau khi đã có predictions (Post-inference crash)**: **Fail-closed**. Cấm tự động chạy lại âm thầm. Đóng băng artifacts và dừng hệ thống để người dùng kiểm tra.
7. **Khóa hậu nghiệm**: Sau khi 5 evaluations hoàn tất, niêm phong toàn bộ outputs; chỉ thực thi các phân tích thống kê đã đăng ký trước.

---

## 7. Trạng Thái Hiện Tại & Phán Quyết Trước Thực Thi

* `preregistration_complete`: **`true`**
* `locked_test_partition_status`: **`SEALED`**
* `completed_unsealing_sessions`: **`0`**
* `completed_model_evaluations`: **`0`**
* `locked_test_accesses`: **`0`**
* `new_training_runs`: **`0`**
* `gpu_inference_calls`: **`0`**
* **Phán quyết**: **`READY_FOR_HUMAN_UNSEALING_APPROVAL`**
