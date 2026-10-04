# Hướng Dẫn Vận Hành Kiểm Chuẩn Môi Trường Offline (Offline User Runbook)

> **Tài liệu hướng dẫn**: `research/evidence/phase-4c.2g.0.2b/OFFLINE_USER_RUNBOOK.md`
> **Phase áp dụng**: Phase 4C.2G.0.2B
> **Mục tiêu**: Hướng dẫn người dùng thao tác ngắt kết nối mạng vật lý và chạy công cụ xác minh offline độc lập trên máy chủ thực thi trước khi xem xét cấp phép giải niêm phong.

---

## Các Bước Thao Tác Bắt Buộc

1. **Mở PowerShell trước khi ngắt mạng**: Khởi chạy cửa sổ PowerShell (hoặc PowerShell 7) trong thư mục gốc của repository `forensics-web-lab` trước khi thực hiện ngắt kết nối mạng.
2. **Ghi lại đường dẫn wrapper**: Xác định rõ đường dẫn thực thi của tệp script wrapper:
   ```powershell
   scripts\research\RUN_PHASE4C2G_OFFLINE_VERIFIER.ps1
   ```
3. **Rút cáp Ethernet**: Rút toàn bộ dây cáp mạng LAN/Ethernet kết nối vào máy chủ.
4. **Tắt Wi-Fi**: Vô hiệu hóa hoặc ngắt kết nối hoàn toàn Wi-Fi trên hệ thống.
5. **Ngắt VPN**: Tắt toàn bộ các ứng dụng kết nối mạng riêng ảo (Radmin VPN, WireGuard, OpenVPN, v.v.).
6. **Tắt Bluetooth PAN / Mobile Hotspot**: Đảm bảo không có chia sẻ kết nối Internet qua Bluetooth hoặc điểm phát sóng di động.
7. **Không đóng PowerShell**: Giữ nguyên cửa sổ PowerShell đã mở trước đó để tránh các tiến trình khởi tạo lại kết nối ngầm.
8. **Chạy wrapper**: Thực thi wrapper kiểm toán bằng lệnh:
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\research\RUN_PHASE4C2G_OFFLINE_VERIFIER.ps1
   ```
9. **Tuyệt đối không mount locked-test**: Trong suốt quá trình chạy verifier, phân vùng `locked-test` phải duy trì trạng thái `UNMOUNTED`.
10. **Kiểm tra receipt khi exit 0**: Nếu wrapper trả về exit code `0` (`READY_FOR_HUMAN_AUTHORIZATION_REVIEW`), tiến hành kiểm tra receipt tại:
    ```
    data/research/local-artifacts/phase-4c.2g/offline_verifier_execution_receipt.json
    ```
    Tuyệt đối **CHƯA** chạy evaluator tại bước này.
11. **Quy định vô hiệu hóa receipt khi tái kết nối mạng**: Nếu máy chủ kết nối mạng trở lại sau khi chạy verifier, receipt đó **không còn giá trị** để phục vụ phiên thực thi chính thức. Bắt buộc phải thực hiện lại quy trình ngắt mạng từ đầu và chạy lại verifier ngay trước khi xem xét ủy quyền (`HUMAN_UNSEALING_AUTHORIZATION.json`) và thực thi đánh giá.
