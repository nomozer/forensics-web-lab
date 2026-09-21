# Forensics Web Lab

> **Đề tài nghiên cứu**: *Nghiên cứu và xây dựng công cụ nhẹ phát hiện và định vị hình ảnh do AI tạo sinh và chỉnh sửa trên nền tảng web*  
> **Product Name**: **Forensics Web Lab**

[![License Status](https://img.shields.io/badge/license-pending%20decision-lightgrey)](docs/LICENSING.md)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x%20strict-blue.svg)](https://www.typescriptlang.org/)
[![Runtime](https://img.shields.io/badge/Runtime-ONNX%20Runtime%20Web-orange.svg)](https://onnxruntime.ai/)
[![Privacy](https://img.shields.io/badge/Privacy-100%25%20Client--Side%20Zero--Egress-success.svg)](docs/PRIVACY.md)

---

## 1. Giới thiệu dự án & Tuyên bố khoa học (Scientific Disclaimer)

**Forensics Web Lab** là một hệ thống nghiên cứu thực nghiệm và công cụ web chuyên sâu hỗ trợ phát hiện, phân loại và định vị các dấu hiệu ảnh do AI tạo sinh (fully generated) hoặc can thiệp chỉnh sửa cục bộ (AI-edited / inpainting).

### Tuyên bố quan trọng về bản chất giám định
* **Công cụ hỗ trợ điều tra, không phải chứng chỉ xác minh tuyệt đối**: Hệ thống này được thiết kế để phát hiện các dấu vết đặc trưng của mô hình tạo sinh (dấu vết tần số, bất thường nhiễu hạt, lưới biến dạng, cấu trúc pixel). Việc không phát hiện thấy dấu vết không đồng nghĩa với việc khẳng định bức ảnh là "ảnh thật 100%".
* **Không sử dụng từ "ảnh thật" như một kết luận tuyệt đối**: Trạng thái âm tính của hệ thống được định nghĩa chặt chẽ là `no_ai_evidence` (*Chưa tìm thấy bằng chứng AI trong phạm vi nhận biết của bộ phân tích*).
* **Bảo vệ quyền riêng tư tuyệt đối (Zero Server Egress)**: Mọi thao tác giải mã file, trích xuất EXIF/metadata, suy luận mô hình AI (ONNX Runtime Web), biến đổi Fourier 2D (FFT), và tạo bản đồ nhiệt (Heatmap) đều chạy **100% trực tiếp trong trình duyệt người dùng** thông qua Web Worker (WASM / WebGPU). **Không có bất kỳ byte dữ liệu ảnh nào bị gửi lên máy chủ**.

---

## 2. Bốn trạng thái kết luận chuẩn mực

Hệ thống tổng hợp đa nguồn bằng chứng và trả về một trong bốn trạng thái chính quy:

| Trạng thái | Mã định danh | Ý nghĩa pháp lý / kỹ thuật |
| :--- | :--- | :--- |
| **Chưa tìm thấy bằng chứng AI** | `no_ai_evidence` | Chưa tìm thấy bằng chứng rõ ràng cho thấy ảnh được tạo hoặc sửa bằng AI trong phạm vi công cụ. Không khẳng định tính nguyên bản tuyệt đối. |
| **Tạo hoàn toàn bằng AI** | `fully_generated` | Có bằng chứng đa nguồn nhất quán cho thấy toàn bộ nội dung ảnh được tạo sinh nhân tạo bởi mô hình AI (Diffusion, GAN, v.v.). |
| **Chỉnh sửa cục bộ bằng AI** | `ai_edited` | Có dấu hiệu một phần ảnh bị can thiệp, inpainting, xóa vật thể, thay thế đối tượng hoặc tái tạo bằng AI, trong khi các phần khác giữ đặc tính gốc. |
| **Không chắc chắn / Mâu thuẫn** | `uncertain` | Bằng chứng chưa đủ độ tin cậy thống kê, các bộ phân tích cho tín hiệu xung đột, hoặc ảnh bị nén/suy giảm quá nặng khiến không thể kết luận an toàn. |

---

## 3. Kiến trúc hệ sinh thái sản phẩm

Hệ thống được tổ chức theo mô hình Monorepo chặt chẽ:

```text
forensics-web-lab/
├── apps/
│   └── web/               # Ứng dụng web React + TypeScript + Vite + Canvas + Tailwind/Vanilla CSS
├── packages/
│   ├── inference/         # Quản lý ONNX Runtime Web, Web Worker pool, WebGPU/WASM fallback
│   ├── forensics/         # Thuật toán DSP thuần TS: 2D-FFT, DCT, Noise Residual, JPEG Block Grid, ELA
│   ├── provenance/        # Bộ phân tích EXIF, XMP, IPTC và adapter C2PA Content Credentials
│   ├── report/            # Tạo báo cáo JSON Schema v1.0 và giao diện in PDF chuyên nghiệp
│   └── shared/            # Định nghĩa kiểu dữ liệu, hằng số, schema và contracts dùng chung
├── ml/
│   ├── configs/           # Cấu hình huấn luyện và kiểm thử có thể tái lập (YAML)
│   ├── datasets/          # Manifest generator và adapter cho GenImage, SAGI-D, RealHD, RAID
│   ├── training/          # Pipeline huấn luyện PyTorch (MobileNetV3 backbone, Focal Loss)
│   ├── evaluation/        # Framework đánh giá: In-domain, Unseen Generator, Robustness, ECE
│   ├── export/            # Xuất ONNX (Opset 17) và lượng tử hóa INT8
│   └── tests/             # Unit test cho pipeline ML và contract test PyTorch vs ONNX
├── models/
│   ├── registry.json      # Sổ đăng ký model chính thức (kích thước, SHA-256, opset, trạng thái)
│   ├── MODEL_CARD.md      # Model Card chi tiết theo tiêu chuẩn nghiên cứu quốc tế
│   └── README.md          # Hướng dẫn quản lý trọng số mô hình
├── research/
│   ├── experiments/       # Nhật ký thực nghiệm và tracking tham số
│   └── results/           # Dữ liệu đo lường thực tế (tuyệt đối không dùng số liệu giả)
├── docs/                  # Toàn bộ tài liệu kiến trúc, kế hoạch nghiên cứu, bảo mật và pháp lý
└── .github/workflows/     # CI/CD tự động kiểm tra lint, typecheck, unit test và build
```

---

## 4. Tài liệu nghiên cứu & Đặc tả kỹ thuật

* [Kiến trúc hệ thống toàn diện](docs/ARCHITECTURE.md) (`docs/ARCHITECTURE.md`)
* [Kế hoạch nghiên cứu & Ma trận thực nghiệm](docs/RESEARCH_PLAN.md) (`docs/RESEARCH_PLAN.md`)
* [Danh mục dữ liệu & Chuẩn hóa Manifest](docs/DATASETS.md) (`docs/DATASETS.md`)
* [Kiểm toán bản quyền dữ liệu](docs/DATA_LICENSES.md) (`docs/DATA_LICENSES.md`)
* [Đặc tả huấn luyện mô hình nhẹ](docs/TRAINING.md) (`docs/TRAINING.md`)
* [Quy trình đánh giá & Chuẩn đo lường](docs/EVALUATION.md) (`docs/EVALUATION.md`)
* [Chính sách bảo mật quyền riêng tư (Zero-Egress)](docs/PRIVACY.md) (`docs/PRIVACY.md`)
* [Hướng dẫn bảo mật phòng thủ](docs/SECURITY.md) (`docs/SECURITY.md`)
* [Mô hình mối đe dọa (Threat Model)](docs/THREAT_MODEL.md) (`docs/THREAT_MODEL.md`)
* [Giới hạn kỹ thuật & Cảnh báo](docs/LIMITATIONS.md) (`docs/LIMITATIONS.md`)
* [Hướng dẫn triển khai tĩnh](docs/DEPLOYMENT.md) (`docs/DEPLOYMENT.md`)
* [Hướng dẫn tái lập thực nghiệm](docs/REPRODUCIBILITY.md) (`docs/REPRODUCIBILITY.md`)
* [Các quyết định cấp phép bản quyền](docs/LICENSING.md) (`docs/LICENSING.md`)
* [Backlog công việc & Ma trận phụ thuộc](docs/BACKLOG.md) (`docs/BACKLOG.md`)
* [Sổ đăng ký mô hình & Model Card](models/MODEL_CARD.md) (`models/MODEL_CARD.md`)

---

## 5. Bắt đầu nhanh (Quickstart)

### Yêu cầu môi trường
* Node.js $\ge 20.x$
* pnpm $\ge 9.x$
* Python $\ge 3.12$ (cho nghiên cứu và huấn luyện ML)

### Khởi chạy ứng dụng Web
```bash
# Cài đặt dependencies
pnpm install

# Khởi chạy máy chủ phát triển
pnpm dev
```

### Chạy kiểm thử tự động
```bash
# Kiểm tra TypeScript strict mode
pnpm typecheck

# Chạy unit tests cho toàn bộ monorepo
pnpm test

# Build ứng dụng cho môi trường production
pnpm build
```