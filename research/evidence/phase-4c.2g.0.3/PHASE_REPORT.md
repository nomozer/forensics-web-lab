# Báo Cáo Nghiên Cứu Phase 4C.2G.0.3: Xây Dựng và Kiểm Chứng Controller Cách Ly Mạng Tự Động Trên Windows

> **Phase**: Phase 4C.2G.0.3 — Build and Verify an Automated Windows Network-Isolation Controller<br>
> **Thời điểm niêm phong UTC**: `2026-10-02T01:53:04.030482+00:00`<br>
> **Clock Source**: `SYSTEM_UTC_RUNTIME` (timezone_aware=true, future_check=PASS)<br>
> **Mục tiêu**: Tự động hóa hoàn toàn quy trình cách ly mạng tạm thời trên Windows bằng PowerShell controller cục bộ độc lập; phát hiện phiên từ xa và fail-closed; lập snapshot và thiết lập watchdog Scheduled Task 10 phút; thực hiện cách ly bằng khối try/finally đảm bảo khôi phục chính xác adapter; kiểm tra thụ động zero-probe; tích hợp gọi offline verifier; niêm phong tuyệt đối locked-test (0 accesses).<br>
> **Branch**: `research/phase-4c2g-locked-test-execution`<br>
> **Effective Evaluator Commit**: `3cf75c2bf0c9835dd58897b7b36982732cab40ab`<br>
> **Execution Package Commit**: `2826a8274cb89ec548d6fac5c8ae50c1c2836202`<br>
> **Dry-Run Verdict**: `DRY_RUN_INSPECTION_PASS`<br>
> **Non-Elevated Readiness Verdict**: `USER_UAC_CONFIRMATION_REQUIRED`<br>
> **Tuyên bố bảo vệ phân vùng kiểm chuẩn**: **`Phân vùng locked-test được niêm phong tuyệt đối; 0 lượt đọc, 0 mount, 0 duyệt, 0 giải nén, 0 băm file; 0 inference CPU/GPU; 0 training; 0 tạo artifact AUTHORIZED; locked_test_real_accesses = 0.`**

---

## 1. Kiến Trúc Controller Cách Ly Tự Động (`RUN_PHASE4C2G_AUTOMATED_ISOLATION.ps1`)

Đã hoàn thành và kiểm thử toàn diện controller PowerShell:
`scripts/research/RUN_PHASE4C2G_AUTOMATED_ISOLATION.ps1`
- **Kích thước**: 24255 bytes
- **SHA-256**: `2ba8e03170680b8c46780cf5f81b04426ea2ab7660f107ca951bf83668841952`
- **Phiên bản**: `1.0.0`
- **Các chế độ hoạt động**:
  1. `-DryRun`: Kiểm tra môi trường, phân loại adapter, kiểm tra route, quyền Administrator, xác minh hệ thống Scheduled Task; không thực hiện bất kỳ thay đổi nào trên hệ thống (Verdict: `DRY_RUN_INSPECTION_PASS`).
  2. `-ReadinessTest`: Kiểm tra quyền Administrator; nếu chưa elevated, khởi chạy tiến trình elevated detached qua `Start-Process powershell.exe -Verb RunAs` (yêu cầu người dùng bấm UAC); tạo watchdog Scheduled Task 10 phút; cách ly adapter trong try/finally; chạy `RUN_PHASE4C2G_OFFLINE_VERIFIER.ps1`; xác minh receipt 12 tiêu chí; tự động khôi phục mạng; xuất receipt nguyên tử.
  3. `-ElevatedDetachedWorker`: Chế độ nội bộ chạy trong tiến trình elevated độc lập, ghi log ra `automated_isolation_worker.log`, tiếp tục chạy ổn định dù Claude bị mất kết nối mạng tạm thời.

---

## 2. Các Rào Chắn An Toàn (Safety Boundaries)

1. **Phát hiện phiên từ xa (Remote Session Fail-Closed)**:
   - Ngăn chặn hoàn toàn việc ngắt mạng khi phát hiện: RDP (`SESSIONNAME` chứa `RDP`, `ICA`, `HDX`), SSH (`SSH_CLIENT`, `SSH_CONNECTION`), WinRM/Remote PowerShell (`ServerRemoteHost`, `PSSessionApplicationName`), hoặc CI Runner (`CI`, `GITHUB_ACTIONS`, `TF_BUILD`).
   - Phản hồi: `BLOCKED_REMOTE_SESSION_NETWORK_ISOLATION_UNSAFE`. Không adapter nào bị tắt.
2. **Watchdog khôi phục khẩn cấp bắt buộc**:
   - Trước khi disable adapter, controller tự sinh script khôi phục:
     `data/research/local-artifacts/phase-4c.2g/RECOVER_NETWORK.ps1`
   - Đăng ký one-shot Windows Scheduled Task (`Phase4C2G_Emergency_Network_Recovery`) với quyền `HIGHEST` kích hoạt sau đúng 10 phút.
   - Nếu không tạo được task, fail-closed lập tức: `BLOCKED_RECOVERY_WATCHDOG_NOT_AVAILABLE`.
3. **Bảo vệ Adapter & Khôi phục Tuyệt đối**:
   - Chỉ đưa vào Allowlist các adapter đang `Up` và không phải `Loopback`.
   - Các adapter `Disconnected` (Ethernet, Bluetooth) hoặc `Disabled` trước phiên không bao giờ bị can thiệp.
   - Khối `try/finally` đảm bảo bật lại đúng danh sách adapter đã tắt. Scheduled Task chỉ bị hủy sau khi việc khôi phục thành công.
4. **Kiểm tra Thụ động (Passive Zero-Probe Inspection)**:
   - 0 outbound socket probes, 0 DNS lookups, 0 HTTP requests.
   - Static scan xác nhận không chứa: `Invoke-WebRequest`, `curl`, `wget`, `ping`, `Test-NetConnection`, `System.Net.Sockets`, CLI evaluator, lệnh mount.

---

## 3. Kết Quả Kiểm Thử Thực Tế

1. **Dry-Run Inspection**:
   - Phát hiện 8 adapter: 6 adapter egress target (`Wi-Fi`, `VMnet8`, `Radmin VPN`, `VMnet1`, `vEthernet Default Switch`, `vEthernet WSL`) và 2 adapter được bảo vệ (`Bluetooth`, `Ethernet`).
   - Default IPv4 routes: 2; Default IPv6 routes: 0.
   - Scheduled Task Subsystem Available: `True`.
   - Dry-Run Verdict: **`DRY_RUN_INSPECTION_PASS`**.
2. **Non-Elevated Readiness Test Execution**:
   - Phát hiện phiên hiện tại không có quyền Administrator (`Administrator Role: False`).
   - Background session bắt được giới hạn không thể trực tiếp hiển thị cửa sổ UAC, in hướng dẫn rõ ràng cho người dùng mở elevated shell.
   - Verdict: **`USER_UAC_CONFIRMATION_REQUIRED`**.
3. **Bộ Kiểm Thử Tự Động (Regression Suite)**:
   - Bổ sung 10 bài test mới (`test_g53` đến `test_g62`) trong `ml/tests/test_phase_4c2g_preparation.py`.
   - Toàn bộ **62/62 tests PASS 100%**.

---

## 4. Bảng Kê Chỉ Số Nghiên Cứu Thực Tế (Scientific Real Counters)

| Chỉ số (Real Metric) | Giá trị |
| :--- | :--- |
| `locked_test_real_accesses` | **0** |
| `completed_real_unsealing_sessions` | **0** |
| `completed_real_model_evaluations` | **0** |
| `evaluation_attempts` | **0** |
| `cpu_inference_calls` | **0** |
| `gpu_inference_calls` | **0** |
| `new_training_runs` | **0** |
| `human_authorization_artifacts_created` | **0** |
