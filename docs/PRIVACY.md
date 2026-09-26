# Privacy Specification: Forensics Web Lab

> **Dự án**: `forensics-web-lab`  
> **Trạng thái bảo đảm**: `architecture-supported`  
> **Xác minh mạng thời gian chạy (Runtime Network Verification)**: `unverified` (đang có backlog kiểm thử E2E)

---

## 1. Tuyên ngôn Quyền riêng tư Kiến trúc (Architectural Privacy Manifesto)

**Forensics Web Lab** được thiết kế từ gốc theo kiến trúc bảo vệ quyền riêng tư cục bộ:
> **Zero User Data Transmission**: Hình ảnh số, tensor trích xuất, cấu trúc metadata EXIF/XMP và báo cáo giám định được xử lý trực tiếp trên thiết bị cá nhân của người dùng, không phụ thuộc máy chủ phân tích trung gian.

---

## 2. Cơ chế Kỹ thuật Bảo vệ Quyền riêng tư (Privacy-by-Design Technical Mechanisms)

### 2.1 Xử lý Hoàn toàn tại Client (Complete Local Processing)
* **Client-Side Runtime**: Giải mã ảnh, trích xuất metadata, suy luận mô hình học sâu ONNX, biến đổi Fourier 2D và phân tích khối nén JPEG block grid thực thi hoàn toàn trong Web Worker và OffscreenCanvas.
* **Không dùng API suy luận đám mây**: Hệ thống không tích hợp bất kỳ API thị giác máy tính bên ngoài nào (OpenAI, Google Cloud Vision, AWS Rekognition).
* **Không tích hợp Telemetry**: Không chèn mã tracking pixels, Google Analytics hay beacon chẩn đoán gửi ra ngoài.

### 2.2 Vòng đời Dữ liệu Phù du (Ephemeral Session Lifecycle)
* **Không lưu trữ ảnh vĩnh viễn**: Ảnh của người dùng tuyệt đối không lưu vào `localStorage`, `IndexedDB` hay cookies.
* **Thu hồi URL Blob (Object URL Revocation)**: Các URL blob tạo ra để hiển thị ảnh preview được giải phóng ngay qua `URL.revokeObjectURL` sau khi nạp vào bộ nhớ đồ họa canvas.
* **Nút xóa phiên (Clear Session Control)**: Giao diện người dùng cung cấp tính năng xóa sạch phiên làm việc:
  1. Hủy sạch buffer pixel trong RAM.
  2. Giải phóng mảng tensor Float32.
  3. Đặt lại trạng thái Web Worker.
  4. Xóa preview trên DOM.

### 2.3 Bảo mật khi Xuất Báo cáo (Report Export Privacy)
* **Báo cáo JSON chuẩn hóa**: Chỉ chứa các chỉ số thống kê toán học, điểm số tín hiệu, hash SHA-256 và danh sách cảnh báo.
* **Không nhúng base64 ảnh gốc mặc định**: Tránh việc dữ liệu ảnh bị rò rỉ khi người dùng chia sẻ file kết quả.
* **Không lộ đường dẫn cục bộ**: Báo cáo không chứa bất kỳ đường dẫn filesystem hay định danh cá nhân của máy trạm.

---

## 3. Trạng thái Thẩm định Mạng Thực tế (Empirical Network Verification Status)

* **Tình trạng hiện tại**: `architecture-supported`. Toàn bộ mã nguồn client không có lệnh `fetch` hay `XMLHttpRequest` tải dữ liệu ảnh lên bất kỳ máy chủ nào.
* **Giới hạn kiểm chứng**: `runtime-network-verification: unverified`. Việc kiểm thử tự động ghi nhận toàn bộ network frames trong suốt chu trình phân tích trên trình duyệt thực tế (thông qua Chrome DevTools Protocol / Playwright Network Interception) chưa được tích hợp vào CI/CD.
* **Kế hoạch kiểm định**: Đã đưa vào Backlog nhiệm vụ thiết lập kịch bản kiểm thử E2E tự động chặn và khẳng định $0$ outbound network requests ngoài việc nạp các tài nguyên tĩnh (`.wasm`, fonts) ban đầu.
