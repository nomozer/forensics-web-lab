# Licensing Decisions & Considerations: Forensics Web Lab

## 1. Current Repository Licensing Status

* **Status**: **Unlicensed / Proprietary (Pending Maintainer Decision)**.
* **Directive**: In strict accordance with project governance rules, no open-source or commercial license file (e.g. `LICENSE`, `LICENSE.md`) has been created for the root repository. Formal licensing is deferred to the repository owner / project leads.

---

## 2. Key Decisions for Repository Owners

When determining the open release or distribution terms for Forensics Web Lab, the project leads should consider the following options and tradeoffs:

### Option A: Permissive Open-Source (Apache 2.0 or MIT)
* **Pros**: Maximizes academic reach, community contributions, and adoption by journalists, researchers, and public interest non-profits.
* **Considerations**: Apache 2.0 provides an explicit patent grant and trademark restrictions, offering strong legal protection for academic research tools.

### Option B: Copyleft Open-Source (GNU GPLv3 or AGPLv3)
* **Pros**: Guarantees that any derivative tools, proprietary forks, or SaaS modifications must remain openly accessible with source code.
* **Considerations**: Imposes reciprocal obligations that may restrict commercial enterprise integrations.

### Option C: Dual Licensing (Academic Free / Commercial Enterprise)
* **Pros**: Free for verified non-commercial research, academic institutions, and investigative journalism, with commercial usage requiring separate licensing.
* **Considerations**: Requires governance overhead to manage commercial inquiries and compliance.

---

## 3. Third-Party Dependency & Dataset License Considerations

1. **Client-Side Web Dependencies**:
   * All chosen npm libraries (React, Vite, `onnxruntime-web`) are distributed under standard permissive licenses (MIT, Apache 2.0).
2. **Machine Learning Datasets**:
   * Academic datasets (e.g. `GenImage`, `RealHD`) are published under non-commercial licenses (`CC-BY-NC 4.0`).
   * Models trained exclusively on non-commercial academic datasets cannot be licensed or deployed for direct commercial monetization without acquiring commercial rights to underlying training data.

---

## 4. Model Weights Licensing Status & Distribution Governance

1. **Current Status**: **`not-applicable`**
   * Trong giai đoạn hiện tại, chưa có trọng số mô hình hoặc file checkpoint nhị phân nào được huấn luyện hay lưu trữ trong repository. Do đó, việc gán bất kỳ giấy phép mở (như MIT, Apache-2.0) cho trọng số mô hình là không có cơ sở thực tế.

2. **Future Distribution Constraints**:
   * Khi mô hình được huấn luyện trong tương lai, quyền phân phối và cấp phép trọng số phụ thuộc đồng thời vào 3 thành tố pháp lý bắt buộc:
     * **Mã nguồn dự án**: Theo giấy phép được chủ sở hữu quyết định ở Mục 2.
     * **Pretrained Backbone**: Giấy phép của checkpoint nền khởi tạo (ví dụ: trọng số ImageNet của torchvision/timm).
     * **Dữ liệu huấn luyện**: Bản quyền và điều khoản sử dụng của tập dữ liệu huấn luyện (ví dụ: nếu sử dụng dữ liệu học thuật phi thương mại như `CC-BY-NC 4.0`, toàn bộ checkpoint phái sinh chỉ được phát hành dưới điều khoản phi thương mại tương ứng).
   * Tuyệt đối không suy diễn hoặc tự ý gán giấy phép cho mô hình khi chưa thẩm định đầy đủ tính pháp lý của cả 3 thành tố trên.
