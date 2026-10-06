# Phase 4C.7B — Independent Cohort Acquisition: Protocol Amendment, Automated Pipeline, and Feasibility Alignment

> **Phase**: Phase 4C.7B — Independent Cohort Acquisition & Feasibility Alignment<br>
> **Status**: `ACQUISITION_IN_PROGRESS_WITH_EXACT_BLOCKER`<br>
> **Findings status**: `NOT_MEASURED` (zero detector scoring executed on candidate cohort)<br>
> **Evidence class**: `independent_validation_amendment_and_pipeline`<br>
> **Candidate models verified**: 5 / 5 outer-fold models bound and SHA-256 verified (unchanged)<br>
> **Training runs**: 0 new fits, 0 refits, 0 recalibrations, 0 hyperparameter tuning<br>
> **Detector scoring**: 0 predictions (Detector Isolation Invariant strictly enforced)<br>
> **Cohort size target**: $N_{\text{target}} = 400$ source pairs (800 images: 400 authentic, 400 ai_edited) + 400 binary masks<br>
> **Buffer candidate pool**: 440 pre-ordered candidates (110 per stratum)<br>
> **Task scope**: Binary classification (`authentic` = 0 vs. `ai_edited` = 1)<br>
> **Exact blocker**: Generation of 400 images using SDXL 1.0 inpainting requires GPU with $\ge 12$ GB VRAM (fp16 model requires ~6.6 GB VRAM and ~8 GB system RAM; local machine has NVIDIA GTX 1650 4.00 GB VRAM and 2.44 GB free RAM). Generation pipeline and Colab notebook are prepared for execution on Google Colab T4 GPU.

---

## 1. Bối Cảnh và Căn Cứ Điều Chỉnh (Protocol Amendment v1.2)

Tại Phase 4C.7A, quá trình chuẩn bị kiểm định độc lập kết thúc với trạng thái `PREPARATION_COMPLETE_PENDING_INDEPENDENT_COHORT` và blocker thực địa:
1. Thiếu 160 ảnh tự chụp từ người dùng (Field Collection).
2. Thiếu tài khoản trả phí và API thực thi khả dụng cho Adobe Firefly.

Người dùng đã chính thức đồng ý điều chỉnh kế hoạch thu thập theo nguồn lực thực tế, tuân thủ nguyên tắc trung thực khoa học:
- **Thay thế ảnh tự chụp** bằng ảnh chụp công cộng có giấy phép và provenance rõ ràng:
  * **COCO 2017 Dataset (Phân vùng sạch ngoài Option P)**: 50% (200 pairs). Thẩm tra giấy phép Flickr theo từng ảnh (`CC-BY 2.0`, `CC-BY-SA 2.0`, `CC0`, `Flickr Commons`), không mặc định CC-BY 4.0 cho toàn bộ ảnh.
  * **Unsplash Verified Open Collection (Chụp/Công bố trước 2022)**: 50% (200 pairs). Giấy phép Unsplash License, lưu trữ photo ID, tác giả và ngày công bố. Nhận định khoa học rõ ràng: thời điểm trước 2022 và EXIF không phải bằng chứng tuyệt đối của việc chưa từng chỉnh sửa, mà là bộ lọc vận hành hợp lý chống can thiệp text-to-image AI hiện đại.
- **Thay thế Adobe Firefly** bằng 2 công cụ inpainting mã nguồn mở (open-weights) tiêu biểu:
  * **Stable Diffusion 2 Inpainting (`stabilityai/stable-diffusion-2-inpainting`)**: 50% (200 pairs), CreativeML OpenRAIL-M.
  * **SDXL Inpainting 1.0 (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`)**: 50% (200 pairs), CreativeML OpenRAIL++.
- **Ma trận phân bổ trực giao cân bằng $2 \times 2$ (Orthogonal Balanced Matrix)**:
  * COCO x SD2: 100 pairs (buffer: 110)
  * COCO x SDXL: 100 pairs (buffer: 110)
  * Unsplash x SD2: 100 pairs (buffer: 110)
  * Unsplash x SDXL: 100 pairs (buffer: 110)
  * Tổng: 400 pairs chính thức, 440 candidates trong buffer pool.
  * *Làm rõ về mặt khoa học*: Bỏ cách diễn đạt "triệt tiêu mọi confounding". Phân bổ cân bằng chỉ kiểm soát các yếu tố đã thiết kế giữa 2 nguồn và 2 công cụ; không triệt tiêu các biến số nhiễu tự nhiên tiềm ẩn.
- **Quy tắc thay thế mẫu lỗi theo Strata (Stratum-Preserving Error Replacement)**:
  Nếu một candidate trong ô $(S, T)$ bị loại do QC kỹ thuật hoặc disjoint guard, nó chỉ được phép thay thế bằng candidate tiếp theo trong **cùng ô quota $(S, T)$** theo danh sách candidates đã khóa ngẫu nhiên từ trước. Tuyệt đối không hoán đổi quota giữa các ô.
- **Ngữ nghĩa của Mask**: Mask nhị phân $512 \times 512$ PNG xác định **vùng yêu cầu chỉnh sửa (inpainting request area)** cung cấp cho diffusion pipeline, không mặc nhiên đồng nhất tuyệt đối với việc mọi pixel bên ngoài mask đều bất biến 100% do đặc tính nén của VAE/latent space.

---

## 2. Kết Quả Đo Lường Hạ Tầng Thực Tế (Resource & Runtime Assessment)

Đo lường trực tiếp trên máy trạm cục bộ qua script kiểm tra hệ thống:
- **GPU**: NVIDIA GeForce GTX 1650 (4.00 GB GDDR6 VRAM, Compute Capability 7.5).
- **System RAM**: 15.78 GB Total, **2.44 GB Available / Free**.
- **Storage Free**: Ổ C: 52.1 GB, Ổ D: 79.0 GB.
- **Ước tính bộ nhớ tải checkpoints**:
  * SD2 Inpainting (fp16): ~3.5 GB.
  * SDXL Inpainting (fp16): ~7.25 GB.
  * Tổng dung lượng tải weights: **~10.75 GB**.
- **Ước tính dung lượng lưu trữ Cohort 400 pairs**: ~610 MB (hoàn toàn an toàn trong 79.0 GB đĩa D:).
- **Phân tích Peak GPU/RAM Memory**:
  * SDXL fp16 inpainting pipeline yêu cầu tối thiểu **6.5 GB – 8.0 GB VRAM**.
  * Chạy SDXL trên GPU 4.00 GB sẽ gây ra lỗi CUDA Out of Memory (OOM) 100%.
  * Dùng Sequential CPU Offload cho SDXL đòi hỏi tối thiểu **12 GB RAM khả dụng**, trong khi hệ thống chỉ còn 2.44 GB RAM khả dụng.
  * Do đó, việc cố ép chạy toàn bộ 400 ảnh trên local workstation là không khả thi (infeasible) và vi phạm nguyên tắc trung thực khoa học.
- **Giải pháp chuyển giao khả thi**: Chuẩn bị notebook Google Colab T4 (`notebooks/independent_cohort_acquisition_colab.ipynb`) chạy trực tiếp trên GPU 15.0 GB VRAM miễn phí, thời gian sinh dự kiến ~37–45 phút cho toàn bộ 400 pairs.

---

## 3. Kiến Trúc Pipeline Thu Thập và Quality Control (Acquisition Pipeline)

Mã nguồn thực thi được hoàn thiện tại:
- `ml/evaluation/independent_cohort_acquisition.py`: Module xử lý trung tâm.
- `scripts/research/run_cohort_acquisition.py`: Giao diện dòng lệnh CLI.
- `notebooks/independent_cohort_acquisition_colab.ipynb`: Notebook Colab thực thi GPU.

### 3.1. Các Cơ Chế Bảo Vệ Được Cài Đặt (Guards & Controls)
1. **Four-Level Disjoint Guard**:
   - Đối chiếu với toàn bộ 684 historical Option P sources (bao gồm cả 343 retired locked-test sources).
   - Kiểm tra: `source_id`, `origin_id` (Flickr / Unsplash ID), `raw_sha256` (ảnh gốc trước chuẩn hóa), `master_sha256` (ảnh sau chuẩn hóa 512x512).
2. **Canvas and Codec Parity**:
   - Tự động chuẩn hóa về $512 \times 512$ RGB PNG lossless.
   - Mask nhị phân $512 \times 512$ PNG với pixel $\in \{0, 255\}$.
3. **Rigorous Technical QC**:
   - Không có lỗi decode PIL.
   - Không phải ảnh đơn sắc/blank (std dev $> 2.0$), không chứa NaN/Inf.
   - Mask area ratio khớp với bracket chỉ định: Small ($< 10\%$), Medium ($10\% - 30\%$), Large ($> 30\%$).
   - Generation delta check: ảnh edited phải khác ảnh authentic trong vùng mask (mean absolute difference $> 3.0$).
   - Decompression bomb guard: giới hạn pixel $\le 50,000,000$.
4. **Stratum-Preserving Replacement**:
   - Tự động ghi nhận mọi nỗ lực vào `attempt_ledger.jsonl`.
   - Nếu candidate lỗi, loại bỏ và lấy candidate kế tiếp trong cùng ô stratum.
   - Raise `StratumQuotaDeficitError` nếu cạn buffer pool 110 candidates mà chưa đủ 100 valid pairs.
5. **Detector Isolation Invariant**:
   - Kiểm tra nghiêm ngặt `assert_detector_isolation()`.
   - Cấm tải bất kỳ module detector nào (`phase_4c2g_model`, `locked_test_evaluator`, `independent_evaluator`).
   - Zero detector scoring / zero predictions trong toàn bộ pipeline.
6. **Content QC Preservation**:
   - Trạng thái nội dung được gán cứng là `PENDING_CONTENT_QC`.
   - Tuyệt đối không công bố cohort hoàn tất chỉ vì ảnh giải mã được về mặt kỹ thuật.

---

## 4. Bằng Chứng Thực Nghiệm và Kiểm Chứng Kỹ Thuật (Empirical Verification)

### 4.1. Technical Smoke Test
Lệnh thực thi:
```bash
python scripts/research/run_cohort_acquisition.py --smoke-test
```
Kết quả ghi nhận tại `research/evidence/phase-4c.7b/acquisition_smoke_receipt.json`:
- **Trạng thái**: `PASS` (thời gian chạy: 2.63 giây).
- **Detector Isolation Invariant**: PASSED.
- **Plan Generation**: PASSED (440 candidates, 110 per stratum, 0 historical overlap).
- **Pipeline Execution trên fixture tạm (non-cohort)**: PASSED (16 images, 8 masks, all QC passed).
- **Manifest & Checksum Verification**: PASSED (`1b95ef5687cbfd68...`).
- **Stratum-preserving replacement logic**: Đã tiêm nhiễu làm hỏng ứng viên đầu tiên của stratum `coco_sd2`; hệ thống phát hiện chính xác lỗi, từ chối ứng viên hỏng, thay thế bằng ứng viên tiếp theo trong cùng stratum và hoàn thành đủ quota 8 pairs.

### 4.2. Candidate Acquisition Plan Verification
Lệnh thực thi:
```bash
python scripts/research/run_cohort_acquisition.py --verify-plan
```
Kết quả:
- `total_candidates`: 440 (110 per stratum: `coco_sd2`: 110, `coco_sdxl`: 110, `unsplash_sd2`: 110, `unsplash_sdxl`: 110).
- `modification_type_counts`: `object_replacement`: 176 (40%), `object_removal_and_infill`: 132 (30%), `object_insertion`: 132 (30%).
- `mask_area_counts`: `medium_10_to_30pct`: 176 (40%), `large_over_30pct`: 132 (30%), `small_under_10pct`: 132 (30%).
- `tool_counts`: SD2: 220, SDXL: 220.
- `source_origin_counts`: COCO 2017: 220, Unsplash Verified: 220.
- `historical_overlap_detected`: `false` (0 overlapping sources, 0 overlapping origins).

### 4.3. Test Suites
- **Bộ test mới** `ml/tests/test_independent_cohort_acquisition.py`: **10 / 10 tests PASSED** (21.03s).
- **Bộ test hồi quy** `ml/tests/test_independent_validation_preparation.py`: **25 / 25 tests PASSED** (39.13s).

---

## 5. Hướng Dẫn Chuyển Giao và Bước Kế Tiếp (Handoff & Next Steps)

- **Trạng thái kết thúc Phase 4C.7B**: `ACQUISITION_IN_PROGRESS_WITH_EXACT_BLOCKER`.
- **Nút thắt kỹ thuật cụ thể**: Cần môi trường GPU $\ge 12$ GB VRAM (Google Colab T4) để chạy suy luận inpainting cho 400 cặp ảnh bằng 2 checkpoints SD2 và SDXL thật.
- **Thao tác tối thiểu cần thực hiện trên Colab**:
  1. Mở file `notebooks/independent_cohort_acquisition_colab.ipynb` trên Google Colab.
  2. Chọn `Runtime` $\rightarrow$ `Change runtime type` $\rightarrow$ `T4 GPU`.
  3. Chọn `Runtime` $\rightarrow$ `Run all`.
  4. Sau khi sinh xong (~37–45 phút), notebook sẽ tự động kiểm tra Technical QC, tạo `manifest_independent_cohort.csv`, tính SHA-256 checksum, và tải về file `independent_cohort_phase4c7b.zip`.
  5. Đặt thư mục giải nén vào `data/research/independent_cohort/` trên workspace để chuyển tiếp sang khâu Content QC và Evaluation.

---

## 6. Trạng Thái Đóng và Đạo Đức Nghiên Cứu

| Hạng Mục | Trạng Thái Ghi Nhận |
| :--- | :--- |
| **Giao thức nghiên cứu** | Protocol Amendment v1.2 đã khóa trước thu thập (`AMENDMENT_LOCKED_PRE_ACQUISITION`) |
| **Mô hình candidate** | 5 outer-fold models giữ nguyên 100% trọng số và SHA-256 hash đã kiểm toán |
| **Detector isolation** | Tuyệt đối tuân thủ, zero detector calls trong thu thập và QC |
| **Independent Performance** | Tiếp tục giữ trạng thái **`NOT_MEASURED`** (chưa đánh giá) |
| **Quy tắc Git** | Làm việc trên branch `research/independent-cohort-acquisition`, **không tạo Pull Request** |
| **Trạng thái Phase** | **`ACQUISITION_IN_PROGRESS_WITH_EXACT_BLOCKER`** |
