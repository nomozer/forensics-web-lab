# Dataset Registry & Track Governance (`datasets/`)

> **Dự án**: `forensics-web-lab`  
> **Schema chuẩn**: `docs/schemas/dataset-registry.v1.schema.json`  
> **Sổ đăng ký chính thức**: `datasets/registry.json`

---

## 1. Mục đích của Thư mục

Thư mục `datasets/` đóng vai trò là cơ sở dữ liệu kiểm soát bản quyền và nguồn gốc tập dữ liệu:
* Lưu trữ sổ đăng ký dataset có thể kiểm tra bằng máy (`registry.json`).
* Quản lý phân luồng dữ liệu theo **Dual-Track Policy** (ADR-0006).
* Chặn đứng việc đưa dữ liệu phi thương mại vào sản phẩm web.
* Lưu trữ các bảng băm SHA-256 (manifests) và văn bản chứng minh giấy phép (license evidence).

> [!CAUTION]
> **Tuyệt đối không lưu ảnh vào thư mục này**. Toàn bộ dữ liệu nhị phân đã được cấu hình loại trừ trong `.gitignore`. Thư mục này chỉ theo dõi metadata, schema và contracts.

---

## 2. Quy tắc Phân luồng Dữ liệu (Track Governance)

Mọi dataset được đăng ký đều phải được xếp vào một trong 3 luồng:

1. **`research-only`**:
   - Dữ liệu học thuật phi thương mại (ví dụ: `genimage` mang giấy phép `CC BY-NC-SA 4.0 with additional dataset terms`).
   - Được phép dùng cho nghiên cứu, huấn luyện nội bộ tại `data/research/` và xuất checkpoint tại `models/research/`.
   - **Nghiêm cấm** xuất hiện trong cấu hình huấn luyện của sản phẩm hoặc đưa checkpoint vào `models/registry.json`.

2. **`product-eligible`**:
   - Dữ liệu tự sở hữu hoặc dữ liệu có giấy phép thương mại đầy đủ (`commercialUse: allowed`, `derivativeWeights: allowed`).
   - Đủ điều kiện để huấn luyện và phát hành mô hình cho web client.

3. **`blocked`**:
   - Dữ liệu chưa đủ nguồn chính thức, chưa có văn bản giấy phép rõ ràng (ví dụ: `realhd` hiển thị "Coming soon").
   - Bị khóa hoàn toàn; mọi lệnh tải hoặc nạp dữ liệu đều bị từ chối tự động.

---

## 3. Kiểm định Tự động (Automated Validation)

Sổ đăng ký `datasets/registry.json` được kiểm tra tính toàn vẹn trong cả TypeScript test suite và Python test suite:

```bash
# Kiểm tra bằng TypeScript contracts
pnpm --filter @forensics/shared test

# Kiểm tra bằng Python CLI
python -m ml.datasets.acquire --validate-registry
```
