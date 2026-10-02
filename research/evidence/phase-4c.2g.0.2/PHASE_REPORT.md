# Báo Cáo Nghiên Cứu Phase 4C.2G.0.2: Chuẩn Bị và Xác Minh Runtime Offline Vật Lý Trước Khi Phê Duyệt Mở Niêm Phong

> **Phase**: Phase 4C.2G.0.2 — Prepare and Verify the Physical Offline Runtime Before Human Authorization<br>
> **Thời điểm niêm phong UTC**: `2026-10-02T01:05:48.000000+00:00`<br>
> **Mục tiêu**: Chuẩn bị và xác minh toàn diện runtime CPU offline thực tế (detached execution worktree, 4 evaluator components, 5 canonical checkpoints, frozen dependencies, filesystem separation); kiểm tra thụ động cách ly mạng; bảo toàn nghiêm ngặt phân vùng locked-test chưa mở.<br>
> **Branch**: `research/phase-4c2g-locked-test-execution`<br>
> **Effective Evaluator Commit**: `3cf75c2bf0c9835dd58897b7b36982732cab40ab`<br>
> **Execution Package Commit**: `2826a8274cb89ec548d6fac5c8ae50c1c2836202`<br>
> **Hotfix Evidence Commit**: `54572ae4d6e11dc49110932a0d79e4685a335a0a`<br>
> **Verdict Runtime Gate**: `USER_PHYSICAL_ACTION_REQUIRED`<br>
> **Tuyên bố bảo vệ phân vùng kiểm chuẩn**: **`Phân vùng locked-test được niêm phong tuyệt đối; 0 lượt đọc, 0 mount, 0 duyệt, 0 giải nén, 0 băm file; 0 inference CPU/GPU; 0 training; 0 tạo artifact AUTHORIZED; locked_test_real_accesses = 0.`**

---

## 1. Tóm Tắt Thực Thi & Kết Quả Xác Minh Thực Tế

Trong phase này, toàn bộ điều kiện cần thiết cho môi trường thực thi offline đã được dựng và kiểm tra đối chiếu trực tiếp trên máy chủ thử nghiệm trước khi người có thẩm quyền ký duyệt văn bản mở niêm phong:

1. **Detached Execution Worktree**:
   - Được khởi tạo tại đúng execution package commit `2826a8274cb89ec548d6fac5c8ae50c1c2836202` tách biệt hoàn toàn khỏi working tree chính.
   - Trạng thái git porcelain hoàn toàn sạch (`git status --porcelain` rỗng).
   - Tuyệt đối không chạy từ commit hotfix HEAD (`54572ae`).

2. **Xác Minh 4 Evaluator Components**:
   - `ml/evaluation/confirmatory_metrics.py`: 15,199 bytes, SHA-256 `bacbb9dc0a230359f1c3bbaece1ce8c19e30ddafa807313480d173a808bde808` (**PASS**)
   - `ml/evaluation/locked_test_evaluator.py`: 47,708 bytes, SHA-256 `b311e010535600781e52014fb6b675d32d78a558c5fe7e67bd3ec00b6989a5e1` (**PASS**)
   - `ml/evaluation/run_phase_4c2f_evaluator.py`: 5,739 bytes, SHA-256 `37ed02ff44d8c523150ebb968d2777bba9c2dee8597b3df60efa1fb5bd1e9041` (**PASS**)
   - `docs/schemas/human-unsealing-authorization.v1.schema.json`: 6,143 bytes, SHA-256 `15291643b2ca2b3d6e3c135957db140e33f7cc65bc11aebc4668f3e105af6734` (**PASS**)

3. **Xác Minh 5 Checkpoints Chuẩn Tắc (Zero Inference)**:
   - Toàn bộ 5 checkpoint Stage 1 $N=250$ thuộc giao thức đăng ký `stage1_frozen_backbone_linear_probe` được kiểm tra byte count và streaming SHA-256 trực tiếp từ tệp lưu trữ ngoài Git.
   - Không nạp checkpoint vào PyTorch (`torch.load`), không chạy forward mô hình, 0 lượt inference.
   - Mỗi file có kích thước chính xác **5,627,375 bytes** và khớp hoàn toàn 100% với canonical SHA-256 đã khóa:
     - Seed 42: `c941f42ed00adb098962ddb43c0f2f7d897e48e20957cddc0c491243f8730c92`
     - Seed 1337: `69e706f9062f050ccbfd2affb72d63d5dde9d12f02c1e64c762b0ea6719073f1`
     - Seed 2025: `92ce5ee986d487fdf14374842067d0684fc7669f5ecb32c7d603f6813942ceee`
     - Seed 3407: `4897821ef0a97df9f1c51acde8d4ac01d383aa3a7b9e7ee55bbc3b3108847868`
     - Seed 9001: `5f0f8803adcb7eec88d47e398ef8b2002b46e740c263b12abc2ba392822a91c3`

4. **Khóa Môi Trường Dependencies & Pip Check**:
   - `python -m pip check`: Hoàn thành kiểm tra không xung đột dependency (`No broken requirements found.`).
   - Python: `3.12.10`, Torch: `2.5.1+cu121`, Torchvision: `0.20.1+cu121`, Numpy: `2.5.3`, Scikit-learn: `1.9.1`.
   - Pip freeze digest: `0e11c90b41f22f3f302a4c1a9860c6e3ffba109d6e465f361d447f71a3709a0c` (54 packages).
   - Kiểm tra import các module lõi thành công mà không load model hay dữ liệu.

5. **Kiểm Tra Cách Ly Mạng Thụ Động (Passive Inspection)**:
   - Thực hiện kiểm tra nội bộ 100% thụ động (0 socket probes, 0 DNS lookups, 0 HTTP requests).
   - Các biến môi trường proxy (`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`) đều rỗng.
   - Bảng định tuyến hiện tại vẫn còn default Internet route (`0.0.0.0` qua gateway mặc định).
   - **Kết luận**: Môi trường chưa được ngắt kết nối mạng vật lý. Theo quy định khoa học trung thực, hệ thống ghi nhận chính xác trạng thái này và yêu cầu can thiệp vật lý (`USER_PHYSICAL_ACTION_REQUIRED`).

6. **Tách Biệt Filesystem & An Toàn Mount**:
   - Thư mục xuất kết quả dự kiến `data/research/evaluation-outputs/phase_4c2g` và điểm gắn kết `data/research/locked-test-mount` được xác minh tách biệt hoàn toàn (`VERIFIED_DISJOINT`).
   - Trạng thái mount phân vùng locked-test: `UNMOUNTED`.
   - Kiểm tra mount read-only được giữ ở trạng thái: `PENDING_HUMAN_AUTHORIZATION`.
   - Loại bỏ yêu cầu canary write xâm lấn; bắt buộc kiểm tra read-only bằng statvfs/metadata không đột biến.

---

## 2. Bảng Ma Trận Trạng Thái & Bộ Đếm Khoa Học Tuyệt Đối

| Chỉ số / Điều kiện | Giá trị / Trạng thái | Tiêu chuẩn Đánh giá |
| :--- | :---: | :---: |
| **Detached Worktree Commit** | `2826a8274cb89ec548d6fac5c8ae50c1c2836202` | Khớp tuyệt đối |
| **Worktree Working Tree** | `CLEAN` | Không tệp bẩn |
| **Evaluator Effective Commit** | `3cf75c2bf0c9835dd58897b7b36982732cab40ab` | Đóng băng |
| **Evaluator Components Parity** | `4/4 PASS` | Khớp 100% SHA-256 & byte count |
| **Canonical Checkpoints Parity** | `5/5 PASS` | Khớp 100% SHA-256 & byte count |
| **Dependency Environment Lock** | `0e11c90...` | `pip check` PASS |
| **Network Passive Check** | 0 outbound probes | Pass |
| **Actual Physical Airgap** | `USER_PHYSICAL_ACTION_REQUIRED` | Còn default route Internet |
| **Locked-test Mount State** | `UNMOUNTED` | Chưa mount |
| **Read-only Mount Check** | `PENDING_HUMAN_AUTHORIZATION` | Chờ sau authorization |
| **Human Authorization Status** | `PENDING_HUMAN_APPROVAL` | Chưa ký |
| **`locked_test_real_accesses`** | **`0`** | Không chạm locked-test |
| **`completed_real_unsealing_sessions`** | **`0`** | Chưa unseal |
| **`completed_real_model_evaluations`** | **`0`** | Chưa đánh giá |
| **`evaluation_attempts`** | **`0`** | Chưa thử nghiệm |
| **`cpu_inference_calls`** | **`0`** | Chưa chạy inference |
| **`gpu_inference_calls`** | **`0`** | Chưa chạy inference |
| **`new_training_runs`** | **`0`** | Không huấn luyện |

---

## 3. Quyết Định Tiếp Theo (Verdict)

```text
USER_PHYSICAL_ACTION_REQUIRED
```

Người dùng / chuyên gia thực hiện cần tiến hành ngắt kết nối mạng vật lý (rút dây mạng, tắt Wi-Fi hoặc chuyển sang network namespace chỉ có loopback) trên máy chủ dự kiến chạy đánh giá trước khi lập và ký văn bản `HUMAN_UNSEALING_AUTHORIZATION.json`. Toàn bộ hệ thống dừng tại đây và bảo toàn nghiêm ngặt các rào cản niêm phong.
