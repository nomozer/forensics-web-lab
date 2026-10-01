# BÁO CÁO NGHIÊN CỨU PHASE 4C.2F.2 — HOTFIX KHÓA COMMIT THỰC THI CUỐI VÀ CHUẨN HÓA ĐẶC TẢ PHÊ DUYỆT (EFFECTIVE EVALUATOR COMMIT & AUTHORIZATION SCHEMA EXACTNESS HOTFIX)

---

## 1. Mục tiêu và Định vị Nghiên cứu

Phase 4C.2F.2 là bước chuẩn hóa chuẩn xác cuối cùng (exactness hotfix) nhằm khóa chặt commit thực thi evaluator thực tế, hoàn thiện schema phê duyệt mở niêm phong của con người theo cơ chế không thể suy diễn hay sai lệch, tích hợp kiểm tra tự xác minh toàn vẹn schema trước khi đọc locked-test, và chuẩn hóa toàn bộ dấu thời gian UTC thực tế.

Mục tiêu cốt lõi:
1. **Khóa chặt phân tầng Commit (Commit Hierarchy)**:
   - `base_main_commit`: `8a379665bc8db3722a46618e47db4806a6ea7244` (PR #4 merge commit trên nhánh main).
   - `original_evaluator_commit` / `pre_hotfix_evaluator_commit`: `656529f04ee8dfcf26e7bb46c757f5cba279326e` (commit mã nguồn evaluator ban đầu).
   - `safety_hotfix_commit`: `959139e847f56fddd61e7615759f05897d7f5d9b` (hotfix Phase 4C.2F.1).
   - `effective_evaluator_commit`: `97851a3008c03e2cdeeed22a3cdf896380605a0b` (commit chức năng chứa toàn bộ code thực thi cuối cùng của evaluator).
   - `audited_through_commit`: `97851a3008c03e2cdeeed22a3cdf896380605a0b`.
   - `evidence_seal_semantics`: `non_circular_terminal_git_commit` (commit đóng gói bằng chứng cuối cùng theo ngữ nghĩa phi tự tham chiếu).
   - Tuyệt đối cấm sử dụng `656529f` làm commit chức năng hiện hành.
2. **Khóa chặt Schema Phê duyệt con người (`human-unsealing-authorization.v1.schema.json`)**:
   - `exact_seeds`: Khóa danh sách đúng 5 seed `[42, 1337, 2025, 3407, 9001]` theo đúng giá trị và thứ tự chuẩn tắc bằng từ khóa `const`.
   - `checkpoint_sha256s`: Khóa toàn bộ object 5 hash SHA-256 bằng `const` và `additionalProperties: false`.
   - `evaluator_component_hashes`: Khóa toàn bộ 3 hash SHA-256 của các module evaluator bằng `const` và `additionalProperties: false`.
   - `evaluator_effective_commit`: Khóa bằng `const: "97851a3008c03e2cdeeed22a3cdf896380605a0b"`.
   - `additionalProperties: false` cho toàn bộ các nested object và root.
   - Định dạng `date-time`: Kiểm tra định dạng thời gian ISO-8601 UTC bằng `jsonschema.FormatChecker` tại runtime validator.
   - Chính sách hết hạn `expiry_policy`: Khóa bằng cấu trúc `oneOf` cho 3 trường hợp:
     * `single_session_only`: `expires_at_utc` phải là `null` hoặc không tồn tại.
     * `explicit_expiry_timestamp`: `expires_at_utc` bắt buộc là chuỗi ISO UTC date-time hợp lệ.
     * `explicit_no_expiry`: `expires_at_utc` bắt buộc là `null`.
   - Cấm tạo bất kỳ file `HUMAN_UNSEALING_AUTHORIZATION.json` thật nào.
3. **Tự xác minh Schema phía Evaluator (Schema Self-Verification)**:
   - Đưa thông tin `authorization_schema` (`relative_path`, `bytes`, `sha256`) vào `evaluator_source_binding.json`.
   - Evaluator tự đọc và kiểm tra độ dài byte và mã băm SHA-256 của file schema trước khi dùng để kiểm tra quyền truy cập. Nếu sai lệch dù 1 byte: lập tức fail-closed trước khi đọc locked-test.
4. **Chuẩn hóa Timestamp UTC thực tế**:
   - Mọi timestamp UTC trong evidence được sinh từ `datetime.now(timezone.utc).isoformat()`.
   - Loại bỏ hoàn toàn tình trạng gán giờ địa phương kèm nhãn `+00:00`.
   - Kiểm tra mọi timestamp không được nằm trong tương lai.

---

## 2. Ràng buộc Nguồn gốc và Bảng Khóa Mã Băm

### 2.1. Phân tầng Commit
* **Nhánh nghiên cứu**: `research/phase-4c2f-locked-test-evaluator`
* **Base main commit**: `8a379665bc8db3722a46618e47db4806a6ea7244`
* **Original evaluator commit**: `656529f04ee8dfcf26e7bb46c757f5cba279326e`
* **Safety hotfix commit**: `959139e847f56fddd61e7615759f05897d7f5d9b`
* **Effective evaluator commit**: `97851a3008c03e2cdeeed22a3cdf896380605a0b`
* **Audited through commit**: `97851a3008c03e2cdeeed22a3cdf896380605a0b`
* **Evidence seal semantics**: `non_circular_terminal_git_commit`

### 2.2. Khóa Thành phần Mã Nguồn Evaluator (`evaluator_source_binding.json`)
| Thành phần | Đường dẫn | Kích thước (bytes) | SHA-256 |
| :--- | :--- | :--- | :--- |
| `confirmatory_metrics` | `ml/evaluation/confirmatory_metrics.py` | 15,199 | `bacbb9dc0a230359f1c3bbaece1ce8c19e30ddafa807313480d173a808bde808` |
| `locked_test_evaluator` | `ml/evaluation/locked_test_evaluator.py` | 45,226 | `70d1196c501147d6944a799e3a287635b5ae7042c01cc2ed776dcf3c58940927` |
| `run_evaluator_cli` | `ml/evaluation/run_phase_4c2f_evaluator.py` | 5,739 | `37ed02ff44d8c523150ebb968d2777bba9c2dee8597b3df60efa1fb5bd1e9041` |
| `authorization_schema` | `docs/schemas/human-unsealing-authorization.v1.schema.json` | 5,686 | `29eb1a477e449ab35e07020a0218c6f2eda7b407b186e842834e347e5eb8eb84` |
| `evaluator_test_suite` | `ml/tests/test_phase_4c2f_evaluator.py` | 48,939 | `927d229ecb06ef7968542408e8f043a2a0eae67e90acf6ebdf899f52314323a8` |

### 2.3. Khóa Checkpoints Ứng viên (Stage 1 N=250 Lineage)
| Seed | Checkpoint File | Checkpoint SHA-256 |
| :--- | :--- | :--- |
| 42 | `checkpoints/best_checkpoint.pt` | `c941f42ed00adb098962ddb43c0f2f7d897e48e20957cddc0c491243f8730c92` |
| 1337 | `checkpoints/best_checkpoint.pt` | `69e706f9062f050ccbfd2affb72d63d5dde9d12f02c1e64c762b0ea6719073f1` |
| 2025 | `checkpoints/best_checkpoint.pt` | `92ce5ee986d487fdf14374842067d0684fc7669f5ecb32c7d603f6813942ceee` |
| 3407 | `checkpoints/best_checkpoint.pt` | `4897821ef0a97df9f1c51acde8d4ac01d383aa3a7b9e7ee55bbc3b3108847868` |
| 9001 | `checkpoints/best_checkpoint.pt` | `5f0f8803adcb7eec88d47e398ef8b2002b46e740c263b12abc2ba392822a91c3` |

---

## 3. Kết quả Kiểm thử Hồi quy Toàn diện (Regression Suite)

Bộ test suite mở rộng `ml/tests/test_phase_4c2f_evaluator.py` đạt **45/45 PASS**:
* **30 test cơ sở Phase 4C.2F / 4C.2F.1**:
  - Parity số học Macro-F1, balanced accuracy, ECE 10 uniform bins với sklearn fixture.
  - Source-cluster bootstrap 10,000 replicates PCG64 (seed 20261002), 686 rows/replicate.
  - Fail-closed khi thiếu authorization, hỏng chuỗi băm ledger, quá 1 session hoặc quá 5 attempts.
  - Đặt chỗ `EVALUATION_RESERVED` trước forward, sự cố đánh dấu `EVALUATION_ATTEMPT_INTERRUPTED`.
  - Kiểm tra cô lập mạng thụ động (0 socket connects), kiểm tra read-only mount thụ động (0 canary writes).
  - Synthetic receipt verdict `SYNTHETIC_PIPELINE_PASS` với 0 real counters.
* **15 ca kiểm thử hồi quy mới Phase 4C.2F.2 (`TestPhase4C2F2SchemaExactnessAndProvenance`)**:
  1. `test_f01_exact_seeds_differing_element_fails`: Sai lệch dù 1 seed lập tức FAIL.
  2. `test_f02_exact_seeds_out_of_order_fails`: Đổi thứ tự seed lập tức FAIL.
  3. `test_f03_checkpoint_hash_wrong_value_fails`: Checkpoint hash sai giá trị lập tức FAIL.
  4. `test_f04_evaluator_component_hash_wrong_value_fails`: Evaluator hash sai giá trị lập tức FAIL.
  5. `test_f05_authorization_locks_to_effective_evaluator_commit`: Khóa đúng `97851a3008c03e2cdeeed22a3cdf896380605a0b`.
  6. `test_f06_original_evaluator_commit_656529f_rejected`: Commit cũ `656529f` bị từ chối với `PermissionError`.
  7. `test_f07_authorization_schema_sha_mismatch_fails`: Sửa đổi schema dù 1 byte lập tức FAIL khi evaluator tự kiểm tra.
  8. `test_f08_nested_additional_property_fails`: Thêm trường lạ vào nested object lập tức FAIL (`additionalProperties: false`).
  9. `test_f09_invalid_datetime_fails_with_format_checker`: Format date-time không hợp lệ lập tức FAIL qua FormatChecker.
  10. `test_f10_expiry_policy_and_expires_at_utc_inconsistency_fails`: Không nhất quán giữa expiry policy và timestamp lập tức FAIL.
  11. `test_f11_timestamp_utc_not_in_future`: Timestamp UTC trong mọi bằng chứng có timezone aware và không ở tương lai.
  12. `test_f12_pre_unsealing_gate_locks_effective_evaluator_commit`: PRE gate khóa đúng `effective_evaluator_commit`.
  13. `test_f13_synthetic_receipt_zero_real_counters`: Toàn bộ bộ đếm thực tế trong synthetic receipt bằng 0.
  14. `test_f14_no_real_authorization_artifact_created`: Không có file authorization thật trong repository.
  15. `test_f15_locked_test_real_accesses_remains_zero`: Số lần truy cập locked-test bằng 0 tuyệt đối.

Các kiểm thử toàn repo:
* `ml/tests/test_phase_4c2e_preregistration.py`: **30/30 PASS**.
* Toàn bộ test hermetic Python (`pytest ml/tests -m "not requires_research_artifact"`): **427 passed, 131 deselected**.
* Frontend test (`pnpm test`): **34/34 TS passed, 13/13 continuity unit tests passed**.
* Typecheck (`pnpm typecheck`): **0 errors**.
* Web build (`pnpm build`): **Build thành công**.
* Git diff check (`git diff --check`): **Sạch 100%, không có whitespace error**.

---

## 4. Bảng Tổng kết Các Bộ đếm & Trạng thái Khóa

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
| Real authorization artifacts | **0** | 0 | **TUÂN THỦ TUYỆT ĐỐI** |
| Windows paths in Git evidence | **0** | 0 | **TUÂN THỦ TUYỆT ĐỐI** |

---

## 5. Phán quyết Pre-Unsealing Hotfix

```
READY_FOR_HUMAN_ONE_TIME_UNSEALING_APPROVAL
```

Tất cả các điều kiện tiên quyết cho việc phê duyệt mở niêm phong một lần duy nhất đã được thỏa mãn đầy đủ và xác minh nghiêm ngặt. Hệ thống sẵn sàng tiếp nhận văn bản phê duyệt chuẩn tắc từ con người để chuyển sang Phase 4C.2G.
