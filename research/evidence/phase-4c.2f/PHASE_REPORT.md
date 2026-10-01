# BÁO CÁO NGHIÊN CỨU PHASE 4C.2F.1 — HOTFIX AN TOÀN TIỀN MỞ NIÊM PHONG VÀ RÀNG BUỘC NGUỒN GỐC (FINAL PRE-UNSEALING SAFETY & PROVENANCE HOTFIX)

---

## 1. Mục tiêu và Định vị Nghiên cứu

Phase 4C.2F.1 là đợt hoàn thiện an toàn và kiểm toán nguồn gốc cuối cùng cho công cụ đánh giá xác nhận (`confirmatory evaluator`) trước khi chuyển giao quyền mở niêm phong cho con người. Mục tiêu cốt lõi:
1. **Phân định rõ ràng ba tầng commit**:
   - `base_main_commit`: `8a379665bc8db3722a46618e47db4806a6ea7244` (PR #4 merge commit từ main).
   - `evaluator_functional_commit`: `656529f04ee8dfcf26e7bb46c757f5cba279326e` (commit chức năng chuẩn tắc của evaluator).
   - `audited_through_commit`: `656529f04ee8dfcf26e7bb46c757f5cba279326e`.
   - Áp dụng mô hình evidence seal không tự tham chiếu: Git commit đóng gói là terminal evidence seal theo ngữ nghĩa phi vòng lặp.
2. **Chuẩn hóa Synthetic Dry-Run Receipt**:
   - Thay các trường dễ gây nhầm lẫn bằng: `synthetic_sessions_simulated = 1`, `synthetic_model_evaluations_simulated = 5`, `completed_real_unsealing_sessions = 0`, `completed_real_model_evaluations = 0`, `locked_test_real_accesses = 0`, `verdict = "SYNTHETIC_PIPELINE_PASS"`.
   - Cấm hoàn toàn các phán quyết khoa học (`CONFIRMATORY_SUCCESS`, `CONFIRMATORY_FAILURE`, `LOCKED_TEST_PASS`, `LOCKED_TEST_FAIL`) trong synthetic receipt.
   - Chạy lại bootstrap giả lập đúng 10,000 replicates bằng `numpy.random.Generator(numpy.random.PCG64(20261002))`.
   - Ghi rõ: fixture giả lập, không phải kết quả khoa học, không đưa vào báo cáo hiệu năng thực tế.
3. **Loại bỏ hoàn toàn Outbound Socket Probe**:
   - Cấm evaluator chủ động kết nối DNS, HTTP/HTTPS hoặc bất kỳ địa chỉ mạng ngoài nào để kiểm tra cô lập.
   - Thay bằng cơ chế kiểm tra cục bộ thụ động, không truyền dữ liệu: biến môi trường proxy, bảng định tuyến, network namespace, và runtime isolation receipt.
   - Ghi nhận `network_guard_tests_passed = true` và `actual_execution_network_isolation = "PENDING_RUNTIME_VERIFICATION"`.
4. **Loại bỏ hoàn toàn Canary Write**:
   - Xóa bỏ việc tạo/xóa file `.canary_write_test` trên thư mục locked-test.
   - Xác minh read-only bằng phương pháp hoàn toàn không thay đổi dữ liệu: đối chiếu disjoint giữa `realpath` dữ liệu và output, kiểm tra mount options (`ro`) qua `/proc/self/mountinfo` hoặc `/proc/mounts`, cờ `os.statvfs` `ST_RDONLY`.
   - Evaluator chỉ mở file dataset bằng chế độ đọc nhị phân (`"rb"`).
5. **Chuẩn hóa kế toán truy cập (Access Accounting)**:
   - Tách biệt hai bộ đếm: `evaluation_attempts` và `completed_model_evaluations`.
   - Trình tự nghiêm ngặt cho mỗi checkpoint:
     1. Xác minh authorization, commit, checkpoint, runtime.
     2. Ghi ledger event `EVALUATION_RESERVED` và fsync.
     3. Tăng `evaluation_attempts` trước model forward đầu tiên.
     4. Chạy inference.
     5. Ghi predictions `.part`, flush và fsync.
     6. Atomic rename qua `os.replace`.
     7. Tăng `completed_model_evaluations`.
     8. Ghi `EVALUATION_COMPLETED` và fsync.
   - Sự cố sau `EVALUATION_RESERVED`: ghi `EVALUATION_ATTEMPT_INTERRUPTED`, cấm tự động retry, bắt buộc can thiệp con người.
   - Chặn đứng: attempt thứ 6, completed evaluation thứ 6, session thứ 2.
6. **Khóa quy tắc Tie-Breaking**:
   - Quy tắc chính thức: $\text{predicted\_class} = \text{argmax}(\text{logits}, \text{axis}=1)$.
   - Đối với hai lớp: $p_1 > 0.5 \implies 1$; $p_1 < 0.5 \implies 0$; $p_1 == 0.5 \implies 0$ (theo cơ chế ưu tiên chỉ số đầu tiên của argmax). Không quy đổi tương đương $p_1 \ge 0.5$.
7. **Xây dựng Schema phê duyệt con người**:
   - Tạo `docs/schemas/human-unsealing-authorization.v1.schema.json` khóa chặt các trường bắt buộc.
   - Tuyệt đối không tạo file artifact mang trạng thái `AUTHORIZED`.
8. **Chuẩn hóa thuật ngữ Ledger**:
   - Sử dụng định danh chính xác: `tamper-evident hash-chained ledger`.
   - Mỗi entry chứa đầy đủ: `sequence_number`, `prev_entry_hash`, `entry_hash`, `timestamp_utc`, `event_type`, `session_id`, `checkpoint_seed`, `evaluator_functional_commit`, các bộ đếm tích lũy, và metadata.
   - Ghi `tip_entry_hash` vào receipt có fsync.

---

## 2. Ràng buộc Nguồn gốc và Phân định Commit

* **Nhánh nghiên cứu**: `research/phase-4c2f-locked-test-evaluator`
* **Base main commit**: `8a379665bc8db3722a46618e47db4806a6ea7244` (PR #4 merge commit)
* **Evaluator functional commit**: `656529f04ee8dfcf26e7bb46c757f5cba279326e`
* **Audited through commit**: `656529f04ee8dfcf26e7bb46c757f5cba279326e`
* **Mô hình niêm phong bằng chứng (Evidence Seal Semantics)**:
  - Áp dụng mô hình terminal Git commit đóng gói toàn bộ artifacts, không nhúng mã băm của chính commit đó vào file nhằm triệt tiêu nghịch lý tự tham chiếu (circular reference).

---

## 3. Kiến trúc Bảo vệ Lỗi đóng Cải tiến (Hardened Fail-Closed Guards)

1. **Passive Airgap Guard (Không phát gói tin ra ngoài)**:
   - Hàm `LockedTestEvaluator.verify_network_isolation()` không còn mở socket kết nối tới 8.8.8.8 hay bất kỳ máy chủ nào.
   - Kiểm tra thụ động cục bộ: phát hiện proxy lạ qua biến môi trường (`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`), kiểm tra bảng định tuyến Linux (`/proc/net/route` không có default route ngoài loopback), và kiểm tra network namespace `/proc/self/ns/net`.
   - Nếu không có chứng cứ cô lập từ runtime container/namespace: fail-closed trước khi truy cập dữ liệu.
2. **Non-Invasive Read-Only Mount Guard (Không canary write)**:
   - Hàm `LockedTestEvaluator.verify_read_only_mount()` không tạo, đổi tên, ghi hoặc xóa bất kỳ file nào dưới locked-test root.
   - Xác thực tính read-only bằng kiểm tra mount options (`ro`) trong `/proc/self/mountinfo` hoặc `/proc/mounts`, kiểm tra cờ hệ thống `os.statvfs` `ST_RDONLY`, và kiểm tra tính disjoint của đường dẫn canonical `realpath` so với output directory.
3. **Kế toán đặt chỗ trước suy luận (Pre-Forward Reservation)**:
   - Ghi nhận `EVALUATION_RESERVED` và tăng `evaluation_attempts` trước khi chuyển dữ liệu vào mô hình.
   - Nếu quá trình suy luận hoặc validation gặp sự cố, hệ thống ghi nhận `EVALUATION_ATTEMPT_INTERRUPTED` và khóa dừng ngay lập tức. Mọi nỗ lực tái chạy checkpoint đó mà không có phê duyệt con người đều bị từ chối (`fail-closed`).
   - Giới hạn cứng: đúng 5 attempts, đúng 5 completed evaluations, đúng 1 unsealing session.
4. **Tamper-Evident Hash-Chained Ledger**:
   - `locked_test_access_ledger.jsonl` duy trì chuỗi băm liên tục với `sequence_number`, `prev_entry_hash`, và `entry_hash`.
   - Mọi thao tác sửa đổi nội dung, đảo thứ tự, hoặc chèn dòng đều bị phát hiện ngay lập tức khi khởi tạo đối tượng `AccessLedger`.
   - `tip_entry_hash` được nhúng trực tiếp vào receipt của từng checkpoint và fsync xuống đĩa.

---

## 4. Định nghĩa Toán học và Tie-Breaking Chuẩn tắc

1. **Phân lớp dự đoán (Predicted Class)**:
   $$\hat{y} = \text{argmax}(\text{logits}, \text{axis}=1)$$
   - Nếu $p_1 > 0.5 \implies \hat{y} = 1$ (`ai_edited`).
   - Nếu $p_1 < 0.5 \implies \hat{y} = 0$ (`authentic`).
   - Nếu $p_1 == 0.5 \implies \hat{y} = 0$ (theo cơ chế ưu tiên chỉ số đầu tiên `first-index` của hàm `argmax`).
   - Tuyệt đối không diễn giải là "tương đương $p_1 \ge 0.5$" vì tại $0.5$ ngưỡng $\ge$ sẽ chọn nhãn 1.
2. **Source-Cluster Bootstrap (10,000 Replicates)**:
   - Bộ sinh số ngẫu nhiên: `numpy.random.Generator(numpy.random.PCG64(20261002))`.
   - Lấy mẫu có hoàn lại 343 cụm nguồn `unique_source_id`.
   - Bảo toàn tính đa bội cụm (multiplicity): khi nguồn được chọn $k$ lần, cả 2 mẫu xuất hiện đúng $k$ lần (đúng 686 dòng/replicate).
   - Tối ưu hóa tính toán F1 qua bincount nhị phân đạt tốc độ 1.1s cho 10,000 replicates, bảo toàn sự tương đương số học tuyệt đối với `sklearn.metrics.f1_score` đến $10^{-15}$.

---

## 5. Kết quả Kiểm thử Toàn diện (Verification Suite)

Đã bổ sung và hoàn thiện bộ test suite độc lập `ml/tests/test_phase_4c2f_evaluator.py`:
* **Tổng số tests**: **30/30 PASS**.
* **Các ca kiểm thử quan trọng**:
  1. PRE gate khóa đúng functional commit `656529f04ee8dfcf26e7bb46c757f5cba279326e` và phân biệt với `base_main_commit`.
  2. Synthetic dry-run receipt không chứa phán quyết khoa học xác nhận; verdict là `SYNTHETIC_PIPELINE_PASS`.
  3. Mọi bộ đếm thực tế (`completed_real_unsealing_sessions`, `completed_real_model_evaluations`, `locked_test_real_accesses`) luôn bằng 0.
  4. Bootstrap giả lập chạy đúng 10,000 replicates với PCG64 seed 20261002.
  5. Kiểm tra cô lập mạng thụ động, 0 cuộc gọi socket outbound.
  6. Kiểm tra gắn đĩa chỉ đọc thụ động, 0 canary writes, 0 sửa đổi filesystem.
  7. Đặt chỗ đánh giá (`EVALUATION_RESERVED`) diễn ra trước model forward.
  8. Crash giữa chừng đánh dấu `EVALUATION_ATTEMPT_INTERRUPTED` và chặn đứng việc tự động retry.
  9. Chặn đứng lần attempt thứ 6, completed evaluation thứ 6, và session thứ 2.
  10. Tie-breaking $p_1 == 0.5$ chọn class 0.
  11. Schema phê duyệt con người từ chối nếu thiếu hoặc sai evaluator commit, sai checkpoint hashes, hoặc thiếu cam kết không tuning.
  12. Phát hiện lập tức mọi hành vi sửa đổi chuỗi băm trong ledger.
  13. `locked_test_accesses` duy trì bằng 0 tuyệt đối trong toàn bộ test suite.

---

## 6. Bảng Tổng kết Các Bộ đếm & Trạng thái Khóa

| Chỉ số / Bộ đếm | Trạng thái Hiện tại | Giới hạn Cho phép | Trạng thái Tuân thủ |
| :--- | :--- | :--- | :--- |
| `locked_test_accesses` | **0** | 0 (trước phê duyệt) | **TUÂN THỦ TUYỆT ĐỐI** |
| `completed_real_unsealing_sessions` | **0** | 0 (trước phê duyệt) | **TUÂN THỦ TUYỆT ĐỐI** |
| `completed_real_model_evaluations` | **0** | 0 (trước phê duyệt) | **TUÂN THỦ TUYỆT ĐỐI** |
| `new_training_runs` | **0** | 0 | **TUÂN THỦ TUYỆT ĐỐI** |
| `gpu_inference_calls` | **0** | 0 | **TUÂN THỦ TUYỆT ĐỐI** |
| Checkpoints resolved | **5/5** | 5 | **100% SHA-256 MATCH** |
| Outbound network probes | **0** | 0 | **TUÂN THỦ TUYỆT ĐỐI** |
| Canary writes to locked-test | **0** | 0 | **TUÂN THỦ TUYỆT ĐỐI** |
| Windows paths in Git evidence | **0** | 0 | **TUÂN THỦ TUYỆT ĐỐI** |

---

## 7. Phán quyết Pre-Unsealing Hotfix

```
READY_FOR_HUMAN_ONE_TIME_UNSEALING_APPROVAL
```
Toàn bộ các yêu cầu an toàn, kiểm toán nguồn gốc, kế toán truy cập và loại bỏ các tác vụ xâm lấn (outbound socket, canary write) đã được xử lý triệt để. Hệ thống hoàn toàn sẵn sàng cho bước cấp phép một lần bằng văn bản từ con người.
