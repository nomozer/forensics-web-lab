# Báo Cáo Nghiên Cứu Phase 4C.2G.0.3.1: Thu Hẹp Mục Tiêu Cách Ly Mạng và Hoàn Thiện Quy Trình Single-UAC

> **Phase**: Phase 4C.2G.0.3.1 — Minimize Windows Network-Isolation Targets and Harden the Single-UAC Readiness Workflow<br>
> **Thời điểm niêm phong UTC**: `2026-10-02T02:15:32.620118+00:00`<br>
> **Clock Source**: `SYSTEM_UTC_RUNTIME` (timezone_aware=true, future_check=PASS)<br>
> **Mục tiêu**: Khắc phục nguyên nhân gốc rễ việc chọn adapter quá rộng của Phase 4C.2G.0.3; chuyển dịch hoàn toàn sang mô hình chọn adapter tối thiểu dựa trên `ifIndex` và quyền sở hữu route thực tế; bảo vệ tuyệt đối loopback, adapter host-only (`VMnet1`), và các switch ảo nội bộ (`WSL`, `Default Switch`); nâng cấp watchdog khôi phục lên 15 phút với kiểm tra cú pháp và truy vấn xác minh `schtasks /query`; bổ sung cơ chế cách ly lặp fail-closed phát hiện ambiguous route owner; thiết lập quy trình single-UAC (1 lệnh duy nhất, 1 lần bấm UAC); niêm phong tuyệt đối locked-test (0 accesses).<br>
> **Branch**: `research/phase-4c2g-locked-test-execution`<br>
> **Effective Evaluator Commit**: `3cf75c2bf0c9835dd58897b7b36982732cab40ab`<br>
> **Execution Package Commit**: `2826a8274cb89ec548d6fac5c8ae50c1c2836202`<br>
> **Dry-Run Verdict**: `DRY_RUN_INSPECTION_PASS`<br>
> **Phase Verdict**: `READY_FOR_SINGLE_UAC_READINESS_TEST`<br>
> **Tuyên bố bảo vệ phân vùng kiểm chuẩn**: **`Phân vùng locked-test được niêm phong tuyệt đối; 0 lượt đọc, 0 mount, 0 duyệt, 0 giải nén, 0 băm file; 0 inference CPU/GPU; 0 training; 0 tạo artifact AUTHORIZED; locked_test_real_accesses = 0.`**

---

## 1. Phân Tích Nguyên Nhân Gốc Rễ và Khắc Phục (Root Cause & Resolution)

1. **Vấn đề của Phase 4C.2G.0.3**:
   - Controller cũ sử dụng tiêu chí quá rộng: `Status == 'Up' AND Name -notmatch 'Loopback'`.
   - Tiêu chí này khiến 6 adapter bị đưa vào allowlist disable, bao gồm cả `VMnet1` (host-only), `VMnet8`, `vEthernet WSL`, và `vEthernet Default Switch`, dù chúng không sở hữu bất kỳ default/egress route nào ra Internet.
2. **Thuật toán Chọn Tối Thiểu (Minimal Selection by ifIndex & Route Ownership)**:
   - Dùng `InterfaceIndex` (`ifIndex`) làm định danh kỹ thuật cốt lõi.
   - Một adapter chỉ được đưa vào initial disable allowlist nếu:
     1. Sở hữu active IPv4 default route (`0.0.0.0/0`); hoặc
     2. Sở hữu active IPv6 default route (`::/0`); hoặc
     3. Là VPN/tunnel adapter đang hoạt động và sở hữu non-loopback egress route.
   - Kết quả trên máy hiện tại:
     - Default route owners: **đúng 2 adapter** (`Wi-Fi` - ifIndex 21 và `Radmin VPN` - ifIndex 12).
     - Protected internal adapters: **đúng 4 adapter** (`VMnet8`, `VMnet1`, `Default Switch`, `WSL`).
     - Protected disconnected adapters: **2 adapter** (`Ethernet`, `Bluetooth`).
     - Initial disable target count giảm từ **6 xuống đúng 2**.

---

## 2. Hoàn Thiện Watchdog & Cách Ly Lặp Fail-Closed

1. **Watchdog 15 Phút Khắc Phục Nguy Cơ Phục Hồi Sớm**:
   - Tăng thời gian chờ từ 10 phút lên 15 phút (`(Get-Date).AddMinutes(15)`), bảo đảm đủ thời gian cho toàn bộ chu trình verifier hash 5 checkpoint (mỗi checkpoint ~5.6 MB).
   - Kiểm tra cú pháp script phục hồi trước khi đăng ký bằng `[System.Management.Automation.Language.Parser]::ParseFile`.
   - Đọc lại Scheduled Task bằng lệnh `schtasks.exe /query /tn Phase4C2G_Emergency_Network_Recovery` để xác minh chắc chắn sự tồn tại của task trước khi disable bất kỳ adapter nào.
2. **Vòng Lặp Cách Ly Đa Vòng (Iterative Rescan) & Ambiguous Owner Guard**:
   - Sau khi disable các default route owners ban đầu, controller rescan lại route table.
   - Nếu phát hiện default route mới phát sinh, controller tra cứu `InterfaceIndex`, cập nhật script phục hồi, và disable bổ sung (tối đa 3 vòng).
   - Nếu phát hiện route không thể gán duy nhất cho adapter đang hoạt động (`unidentified/ambiguous route owner`), controller lập tức kích hoạt fail-closed: tự động khôi phục toàn bộ adapter đã disable và thoát an toàn với `BLOCKED_AMBIGUOUS_ROUTE_OWNER`.

---

## 3. Quy Trình Vận Hành Single-UAC Cho Người Dùng

Người dùng chỉ cần thực hiện **đúng 1 thao tác**:
1. Mở PowerShell với quyền **Run as Administrator** (hoặc chạy lệnh bên dưới và bấm **Yes** một lần duy nhất tại hộp thoại Windows UAC):
   ```powershell
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\Documents\forensics-web-lab\scripts\research\RUN_PHASE4C2G_AUTOMATED_ISOLATION.ps1" -ReadinessTest
   ```
2. Controller tự động hoàn toàn:
   - Snapshot & chọn đúng 2 adapter (`Wi-Fi`, `Radmin VPN`);
   - Sinh `RECOVER_NETWORK.ps1` và xác minh cú pháp;
   - Đăng ký Scheduled Task watchdog 15 phút và kiểm tra query;
   - Disable 2 adapter;
   - Chạy `RUN_PHASE4C2G_OFFLINE_VERIFIER.ps1`;
   - Xác minh receipt offline verifier;
   - Khôi phục lại đúng 2 adapter trong khối `finally`;
   - Xác nhận mạng đã phục hồi;
   - Hủy Scheduled Task watchdog;
   - Ghi receipt nguyên tử ra `data/research/local-artifacts/phase-4c.2g/automated_isolation_readiness_receipt.json`.

---

## 4. Kết Quả Kiểm Thử Chất Lượng (Quality Gates)

1. `python -m pytest ml/tests/test_phase_4c2g_preparation.py -v`: **71/71 tests PASS** (bổ sung 9 bài test mới `test_g63` đến `test_g71`).
2. `python -m pytest ml/tests -m "not requires_research_artifact" -q`: **489 passed, 131 deselected**.
3. `pnpm test`: **34/34 TS unit tests PASS + 13/13 continuity checker tests PASS**.
4. `pnpm typecheck`: **0 errors**.
5. `pnpm build`: **Build thành công 100%**.
6. `pnpm continuity:check`: **`CONTINUITY_CHECK: PASS`**.
7. `git diff --check`: **Clean (0 errors, 0 trailing whitespaces, LF normalized)**.

---

## 5. Bảng Kê Chỉ Số Nghiên Cứu Thực Tế (Scientific Real Counters)

| Chỉ số nghiên cứu | Giá trị |
| :--- | :--- |
| `locked_test_real_accesses` | **0** |
| `completed_real_unsealing_sessions` | **0** |
| `completed_real_model_evaluations` | **0** |
| `evaluation_attempts` | **0** |
| `cpu_inference_calls` | **0** |
| `gpu_inference_calls` | **0** |
| `new_training_runs` | **0** |
| `human_authorization_artifacts_created` | **0** |

---

## 6. Phán Quyết Kết Thúc Phase 4C.2G.0.3.1
**`READY_FOR_SINGLE_UAC_READINESS_TEST`**
