# Báo Cáo Nghiên Cứu Phase 4C.2G.0.2B: Chuẩn Hóa Lập Luận Root-Cause Timestamp và Chuẩn Bị Wrapper Thực Thi Offline

> **Phase**: Phase 4C.2G.0.2B — Correct Timestamp Root-Cause Wording and Prepare the Exact Offline Command<br>
> **Thời điểm niêm phong UTC**: `2026-10-02T01:27:03.664665+00:00`<br>
> **Clock Source**: `SYSTEM_UTC_RUNTIME` (timezone_aware=true, future_check=PASS)<br>
> **Mục tiêu**: Loại bỏ suy diễn chưa được chứng minh về nguyên nhân sai lệch timestamp; chuẩn hóa phân loại root-cause khoa học; xây dựng script wrapper `RUN_PHASE4C2G_OFFLINE_VERIFIER.ps1` giải quyết đường dẫn thực tế; ban hành cẩm nang người dùng offline `OFFLINE_USER_RUNBOOK.md`.<br>
> **Branch**: `research/phase-4c2g-locked-test-execution`<br>
> **Effective Evaluator Commit**: `3cf75c2bf0c9835dd58897b7b36982732cab40ab`<br>
> **Execution Package Commit**: `2826a8274cb89ec548d6fac5c8ae50c1c2836202`<br>
> **Verdict Cuối Cùng**: `READY_FOR_USER_PHYSICAL_NETWORK_DISCONNECTION`<br>
> **Tuyên bố bảo vệ phân vùng kiểm chuẩn**: **`Phân vùng locked-test được niêm phong tuyệt đối; 0 lượt đọc, 0 mount, 0 duyệt, 0 giải nén, 0 băm file; 0 inference CPU/GPU; 0 training; 0 tạo artifact AUTHORIZED; locked_test_real_accesses = 0.`**

---

## 1. Chuẩn Hóa Lập Luận Nguyên Nhân Gốc Rễ (Root Cause Exactness)

1. **Phân loại khoa học trung thực**:
   - **Phân loại chính thức**: `MANUAL_OR_STATIC_TIMESTAMP_WITHOUT_RUNTIME_CLOCK_BINDING`.
   - **Cơ chế lịch sử**: `INDETERMINATE`.
   - **Wording chuẩn**:
     > “The stale timestamp was produced without a verifiable runtime UTC clock binding and appeared 6,552 seconds in the future when independently checked. Available evidence does not establish whether it originated from local-time relabeling, a manually entered value, or another clock conversion error. The exact historical mechanism is therefore classified as indeterminate.”
2. **Bác bỏ về mặt toán học đối với giả thuyết cục bộ UTC+7**:
   - Thời điểm `02:55:00 UTC+7` quy đổi chuẩn xác thành `19:55:00 UTC` của ngày hôm trước (`2026-10-01`), chênh lệch hơn 5 giờ so với thời điểm quan sát thực tế độc lập `2026-10-02T01:05:48Z`.
   - Việc khẳng định `02:55 local UTC+7` bị gán nhãn thành `02:55+00:00` là một giả định không có bằng chứng mã nguồn (source diff), log hoặc kiểm chứng đồng hồ xác thực. Do đó, việc quy kết này đã bị bãi bỏ triệt để.

---

## 2. Wrapper Thực Thi Offline với Đường Dẫn Thực Tế

Đã xây dựng script wrapper PowerShell:
`scripts/research/RUN_PHASE4C2G_OFFLINE_VERIFIER.ps1`
- **Kích thước**: 5,480 bytes
- **SHA-256**: `e475c4a402bfec5993e5b462d98563b00145c63732f5770dfff08483263bbb4e`
- **Các tham số tự động giải quyết (0 placeholder)**:
  - `ExecutionWorktree`: `../forensics-web-lab-offline-execution` (detached HEAD tại commit `2826a8274cb89ec548d6fac5c8ae50c1c2836202`).
  - `CheckpointRoot`: `../forensics-web-lab-local-artifacts/phase_4c1/runs/extracted_15_runs` (chứa đúng 5 thư mục `n250_seed_*`).
  - `OutputDir`: `data/research/evaluation-outputs/phase_4c2g` (rỗng, ghi được).
  - `PlannedMountpoint`: `data/research/locked-test-mount` (rỗng, chưa mount).
  - `ReceiptPath`: `data/research/local-artifacts/phase-4c.2g/offline_verifier_execution_receipt.json`.
- **Hành vi an toàn**:
  - Không mở socket, không bật/tắt adapter mạng của người dùng.
  - Không gọi evaluator, không mount locked-test, không tạo authorization.
  - In đầy đủ thông tin: Current UTC time, Python exe, Git worktree HEAD, receipt path, và cảnh báo `DO NOT RUN WHILE NETWORK IS CONNECTED`.
  - Quy ước exit codes: Exit `0` khi `READY_FOR_HUMAN_AUTHORIZATION_REVIEW`; exit `2` khi còn default route / adapter kết nối; exit `1` khi phát hiện mismatch.

---

## 3. Cẩm Nang Vận Hành Cho Người Dùng (Runbook)

Đã ban hành cẩm nang 11 bước:
`research/evidence/phase-4c.2g.0.2b/OFFLINE_USER_RUNBOOK.md`
Quy định chặt chẽ:
- Người dùng mở PowerShell trước khi ngắt mạng;
- Tắt cáp Ethernet, Wi-Fi, VPN, Bluetooth PAN;
- Thực thi wrapper;
- Kiểm tra receipt nhưng chưa chạy evaluator;
- Nếu tái kết nối mạng sau đó, receipt bị vô hiệu hóa và phải thực hiện lại toàn bộ quy trình ngắt mạng trước phiên ủy quyền chính thức.

---

## 4. Ma Trận Trạng Thái & Bộ Đếm Khoa Học

| Tiêu chí | Trạng thái / Giá trị |
| :--- | :---: |
| **Execution Worktree HEAD** | `2826a8274cb89ec548d6fac5c8ae50c1c2836202` |
| **Worktree Porcelain Cleanliness** | `CLEAN` |
| **Evaluator Effective Commit** | `3cf75c2bf0c9835dd58897b7b36982732cab40ab` |
| **5 Checkpoints Integrity** | `VERIFIED_BITWISE_EXACT` (5,627,375 bytes/file) |
| **Dependencies Pip Check** | `PASS` |
| **Pip Freeze Checksum** | `0e11c90b41f22f3f302a4c1a9860c6e3ffba109d6e465f361d447f71a3709a0c` |
| **Network Isolation Status** | `USER_PHYSICAL_ACTION_REQUIRED` |
| **Planned Locked-Test Mount State** | `UNMOUNTED` |
| **Locked-test Real Accesses** | **`0`** |
| **Completed Real Unsealing Sessions** | **`0`** |
| **Completed Real Model Evaluations** | **`0`** |
| **Evaluation Attempts** | **`0`** |
| **CPU Inference Calls** | **`0`** |
| **GPU Inference Calls** | **`0`** |
| **New Training Runs** | **`0`** |
| **HUMAN_UNSEALING_AUTHORIZATION.json** | **`CHƯA TỒN TẠI`** (`PENDING_HUMAN_APPROVAL`) |
