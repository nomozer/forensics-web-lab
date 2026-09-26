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
   * Mỗi tập dữ liệu học thuật sở hữu giấy phép và điều khoản riêng biệt, không dùng một giấy phép chung:
     * **GenImage**: Được phát hành dưới điều khoản `CC BY-NC-SA 4.0 with additional dataset terms`, cấm sử dụng thương mại đối với cả dataset lẫn các sản phẩm phái sinh (derivative works). Việc checkpoint có cấu thành derivative work hay không chưa có kết luận pháp lý thống nhất (`legal_interpretation.trained_weights_status: unclear`). Theo chính sách bảo thủ của dự án, toàn bộ mô hình huấn luyện từ GenImage bị cấm đưa vào sản phẩm (`productionPromotion: prohibited-by-project-policy`) và chỉ lưu tại **Research Track**.
     * **RealHD**: Trạng thái hiện tại là `unavailable-or-pending` ("Coming soon" trên GitHub), giấy phép `unverified`, tình trạng `blocked`. Tuyệt đối không suy diễn giấy phép khi chưa có nguồn chính thức.
     * **SAGI-D / RAID**: Đang ở trạng thái `blocked` (`status: proposed`, `licenseStatus: unverified`, `acquisitionEnabled: false`) cho đến khi thu thập đủ URL bằng chứng bản quyền chính thức.
     * **Synthetic Smoke Fixture**: Dữ liệu giả lập hình học/gradient/nhiễu sinh cục bộ bằng code, xếp vào **`fixture-only`** (`purpose: fixture`, `ownership: project-generated`, `licenseStatus: pending-project-license-decision`, `commercialUse: internal-testing-only`). Không thuộc product model lineage.
   * Trọng số mô hình huấn luyện từ các tập dữ liệu phi thương mại hoặc thử nghiệm trên **tuyệt đối không được đưa vào ứng dụng web sản phẩm (`models/product/`)**.
   * Mô hình sản phẩm chỉ được huấn luyện từ tập dữ liệu thuộc **Product Track** (dữ liệu tự sở hữu hoặc có quyền thương mại rõ ràng).

---

## 4. Model Weights Licensing Status & Distribution Governance

1. **Current Status**: **`not-applicable`**
   * Trong giai đoạn hiện tại, chưa có trọng số mô hình hoặc file checkpoint nhị phân nào được huấn luyện hay lưu trữ trong repository. Do đó, việc gán bất kỳ giấy phép mở (như MIT, Apache-2.0) cho trọng số mô hình là không có cơ sở thực tế.

2. **Dual-Track Weight Distribution Governance**:
   * Khi mô hình được huấn luyện trong tương lai, quy chế phân phối tuân thủ ranh giới hai luồng:
     * **Research Track Checkpoints (`models/research/`)**: Phục vụ công bố học thuật nội bộ; kế thừa các ràng buộc phi thương mại / chia sẻ tương tự (Share-Alike) của dữ liệu nguồn (ví dụ: GenImage). Tuyệt đối không được đóng gói vào bản phát hành sản phẩm.
     * **Product Track Checkpoints (`models/product/`)**: Chỉ được nạp vào client web khi đáp ứng đồng thời 3 điều kiện:
       1. Mã nguồn dự án có giấy phép cho phép phân phối.
       2. Pretrained backbone (nếu có) có quyền sử dụng thương mại.
       3. Toàn bộ dữ liệu huấn luyện là `product-eligible` (không chứa bất kỳ mẫu nào mang điều khoản `commercialUse: prohibited`).
   * Tuyệt đối không suy diễn hoặc tự ý gán giấy phép cho mô hình khi chưa thẩm định đầy đủ tính pháp lý của cả 3 thành tố trên.
