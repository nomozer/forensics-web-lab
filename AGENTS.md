# Instructions for AI Assistants (AGENTS.md)

Chào bạn, đây là các quy tắc bất biến dành cho mọi tác nhân AI làm việc trên repository `forensics-web-lab`.

## 1. Kiểm tra Git trước khi làm việc
Mọi phiên làm việc bắt buộc phải bắt đầu bằng việc kiểm tra trạng thái Git hiện tại (chạy riêng từng lệnh, không gộp nếu được yêu cầu):
* `pwd`
* `git status --short`
* `git branch --show-current`
* `git rev-parse HEAD`
* `git rev-parse main`
* `git remote -v`
* `git log --oneline -5`

Không commit trực tiếp lên `main`, không sửa lịch sử `main`, không force push, không thay đổi remote.

## 2. Đọc tài liệu ngữ cảnh bắt buộc
Trước khi thực hiện bất kỳ sửa đổi nào, tác nhân PHẢI đọc:
1. `docs/PROJECT_STATE.md` — Nắm rõ trạng thái hiện tại, phase đã hoàn thành và blocker.
2. `docs/CODE_MAP.md` — Bản đồ cấu trúc code thực tế và trạng thái xác minh từng file.
3. `docs/SESSION_HANDOFF.md` — Biên bản bàn giao phiên gần nhất.
4. Các ADR liên quan trong `docs/adr/` trước khi thay đổi bất kỳ quyết định kiến trúc nào.

## 3. Quy tắc trung thực kỹ thuật và nghiên cứu (Scientific Honesty)
1. **Không tự động tải dataset/checkpoint**: Mọi hành động tải dữ liệu lớn (> 50 MB) hoặc checkpoint bên ngoài phải có sự xác nhận của người dùng.
2. **Không tự huấn luyện khi chưa có dữ liệu/GPU**: Không giả lập huấn luyện.
3. **Không tạo số liệu/metric giả**: Nếu chưa đo thật thì ghi `not measured` hoặc `not evaluated`. Tuyệt đối không bịa đặt chỉ số accuracy, F1, ECE.
4. **Không mô tả interface hoặc placeholder là tính năng đã hoạt động**: Phải trung thực tuyệt đối giữa contract/interface và implementation thực sự đã test.
5. **Trạng thái không có model (No-Model State)**: Khi chưa cài model thật, hệ thống KHÔNG được sinh xác suất 3 lớp ngẫu nhiên hay giả lập, KHÔNG gán nhãn `fully_generated` hoặc `ai_edited`. Trạng thái bắt buộc là `uncertain` và hiển thị rõ `Model not installed`.
6. **Cập nhật tài liệu trạng thái**: Bất kỳ khi nào code thay đổi hoặc phase mới hoàn thành, phải cập nhật `docs/PROJECT_STATE.md`, `docs/CODE_MAP.md`, và `docs/SESSION_HANDOFF.md`.
