# Phase 4C.7B — Independent Cohort Acquisition: Protocol Amendment, Automated Pipeline, and Feasibility Alignment

> **Phase**: Phase 4C.7B — Independent Cohort Acquisition & Feasibility Alignment (revision 2: provenance, notebook binding, fail-closed correction)<br>
> **Status**: `BLOCKED_PENDING_ELIGIBLE_CANDIDATE_CATALOG` — code, tests and pinned notebook are ready; the real pilot cannot start until an eligible catalog v2 exists (see §R3). Not `READY_FOR_COLAB_REAL_PILOT`.<br>
> **Findings status**: `NOT_MEASURED` (0 detector calls, 0 cohort evaluation)<br>
> **Functional commit**: `d487d1e79fcc626333c9f1b23e7854ef60f6a820` (notebook `EXPECTED_COMMIT`)<br>
> **Real pilot**: `NOT_RUN` (no Colab GPU / user Drive in this environment; no real images generated)<br>
> **Training runs**: 0 fits, 0 refits; frozen models untouched; retired locked-test not accessed (only its 343 source IDs are read for the disjoint guard)<br>

---

## R. Revision 2 — Corrections (2026-10-07)

The revision-1 report below (kept unchanged as history) claimed `READY_FOR_COLAB_REAL_ACQUISITION_PILOT`, 17/17 tests PASS and a verified 440-candidate catalog. Re-checking the actual source showed these claims did not hold.

### R1. Defects reproduced from source before fixing
| # | Defect (commit `dc00895`) | Evidence |
| :--- | :--- | :--- |
| 1 | Notebook declared `PINNED_COMMIT = 4ea65df…` but never used it; an existing clone was only fetched, so stale code could run | Cell 2 source |
| 2 | Notebook used `!` shell commands (exit status ignored) and fell back to `/content` scratch when the Drive mount failed | Cells 1–3 source |
| 3 | Audit cell accepted any existing manifest/ZIP, unbound to the run, commit, protocol, catalog or plan | Cell 4 source |
| 4 | Disjoint guard required `data/research/tgif/manifests/manifest_pilot_a_option_p.csv`, which is excluded from Git: `--verify-plan`, the pilot and 7 tests crash on any fresh clone (Colab included) | CI run [37564295722](https://github.com/nomozer/forensics-web-lab/actions/runs/37564295722) on `dc00895`: **7 failed**, reproduced locally |
| 5 | All 220 COCO `author` values are generated labels `flickr_contributor_<coco_id>`; `published_date` holds COCO `date_captured`; no Flickr photo ID or page; 91 entries are BY-ND/BY-NC-ND | `catalog_eligibility_audit.json` |
| 6 | Unsplash candidates came from the Unsplash Lite dataset (HF mirror). Dataset Terms §2.A grant download/store and *internal ML training*; §3.A–B prohibit disseminating/redistributing the data. Creating and sharing an edited cohort is not covered; the generic Unsplash License was assumed instead. Download URLs request a processed rendition (`auto=format&fit=crop&w=600&q=80`) | `https://raw.githubusercontent.com/unsplash/datasets/master/TERMS.md` (read 2026-10-07) |
| 7 | Per-stratum quota counters were updated only on resume, and modification/mask labels were permuted jointly over all 110 slots, so a full 400-pair run could not meet the 40/30/30 × 30/40/30 quotas | `test_full_mode_quota_preservation_counts_new_acceptances` (failed with 99/100, then fixed) |
| 8 | `--engine mock` was accepted in production mode (real photos + mock edits labelled production); `--verify-plan` printed FAIL but exited 0; the smoke test wrote run directories into the system temp root | CLI source |

### R2. Changes
- **Notebook** (`notebooks/independent_cohort_acquisition_colab.ipynb`): `checkout_pinned` clones/fetches and checks out **detached** at the full `EXPECTED_COMMIT`, asserts `git rev-parse HEAD`, and stops (without modifying anything) when the clone has local or untracked changes. All Git/CLI calls use `subprocess.run(..., check=True)`. Drive-mount failure stops before any run directory exists. Each session gets a new `RUN_ID`; Cell 4 audits only a run completed in the same session and never overwrites a ZIP. Full mode is not launched from the notebook.
- **Eligibility (pre-registered, detector-independent)**: `evaluate_candidate_eligibility` requires a creator verified from a source response (marker + evidence URL; placeholder patterns rejected), versioned CC license, labelled date provenance, documented download rendition, namespaced source keys and a check timestamp. ND is excluded for edited-cohort export unless separate permission is recorded (this does not claim all private research use is prohibited). BY-NC / BY-SA / BY-NC-SA are eligible with an explicit usage policy (attribution, indicate changes, NonCommercial and/or ShareAlike). Channels whose terms do not cover editing+sharing (Unsplash Lite dataset) are excluded.
- **Disjoint guard**: historical keys `coco:<id>` from the committed `research/evidence/phase-4c.0/real-variant-map.json` (684 sources); candidates carry `coco:`/`flickr:`/`unsplash:` keys. Flickr-ID disjointness is reported `NOT_CHECKED` (historical Flickr IDs are not available); byte-hash comparison only runs when the local manifest exists and only detects byte-identical files.
- **Run binding & audit**: `run_binding.json` (run id, commit, protocol/catalog/plan SHA-256, mode, quota); resume refuses foreign/unbound artifacts; `run_receipt.json` is written only after the quota is met and records per-stratum wall time, peak VRAM (`torch.cuda.max_memory_allocated`, `NOT_MEASURED` without CUDA), engine checkpoint + resolved Hugging Face revision (loaded with that exact `revision`), prompt/seed/steps/guidance/scheduler; every attempt carries `run_id` and elapsed time; receipts add download time, raw size, rendition, license URL and usage policy. `audit_acquisition_run` fails on any binding, checksum, ledger or count mismatch.
- **Plan layout**: first 100 slots per stratum carry exactly 40/30/30 and 30/40/30; the 10 buffer slots hold each of the 9 (modification, mask) cells once plus one extra (object_replacement, medium) — marginals 4/3/3 and 3/4/3, totals 44/33/33 unchanged — so a failed core slot of any cell can be replaced without breaking a quota (tested with one QC failure per stratum in a SYNTHETIC full run).
- **Historical byte-hash guard**: only active when the Git-excluded Option P manifest is present; on Colab it is inactive, printed as a NOTICE and recorded in `run_receipt.json`. It only detects byte-identical files (TGIF PNG hashes vs downloaded JPEG bytes), so it was never a meaningful guard for re-encoded copies.
- **CLI**: `--mode pilot|full` requires `--run-id`, `--expected-commit` (clean checkout at that SHA) and `--output-root`; checks GPU policy (CUDA, ≥ 12 GiB VRAM); refuses existing run dirs unless `--resume`; `--audit-run`; mock engine refused without `--allow-synthetic`.
- **Catalog builder v2** (`scripts/research/build_verified_candidate_catalog.py`): Flickr photo ID from COCO `flickr_url`, creator and current license from Flickr oEmbed (request URL, response SHA-256, timestamp stored), license must agree at COCO and Flickr, excluded entries kept with reasons. Unit-tested only against SYNTHETIC responses; **not run against the live services** (egress to flickr.com, huggingface.co, unsplash.com and images.cocodataset.org is denied in this environment).

### R3. Catalog eligibility (superseded v1 → v2)
| | COCO | Unsplash | Total |
| :--- | ---: | ---: | ---: |
| v1 candidates evaluated | 220 | 220 | 440 |
| Eligible under pre-registered rules | 0 | 0 | **0** |
| Excluded: creator placeholder / unverified | 220 | 220 (unverified) | 440 |
| Excluded: ND licence (also) | 91 | — | 91 |
| Excluded: channel terms do not cover edited export | — | 220 | 220 |
| Replaced by verified candidates | 0 | 0 | **0** |
| Still unverified (v2 not built) | 220 needed | 220 needed | 440 |

`verified_candidate_catalog.json`, `candidate_acquisition_plan.json` and `acquisition_smoke_receipt.json` are kept unchanged as **SUPERSEDED** history; the eligibility gate rejects the v1 catalog. The default binding now points to `verified_candidate_catalog_v2.json`, which does not exist yet, so `--verify-plan` fails closed.

**Blockers (user decisions/actions):**
1. **Unsplash strata (220 candidates)**: the Lite-dataset channel is not usable for an exported edited cohort. Options: (a) protocol amendment v1.3 replacing the source with a channel whose terms cover editing+sharing and that exposes verifiable creator/license metadata (e.g. Flickr/Wikimedia Commons CC BY / BY-SA photos), or (b) obtain written permission from Unsplash, or (c) discover photos via the Unsplash API under the Unsplash License (needs an API key and a review of the API terms). Not decided here.
2. **COCO strata**: run the v2 builder on a host with network access (Colab, or this environment after allowing `images.cocodataset.org` and `www.flickr.com`), review the output and commit it; then re-pin the notebook.

### R4. Verification actually run (this environment, CPU, no GPU)
- Targeted: `ml/tests/test_independent_cohort_bindings.py` + `ml/tests/test_independent_cohort_acquisition.py` + `ml/tests/test_independent_validation_preparation.py` — see CURRENT_STATE for counts.
- Full hermetic suite `pytest ml/tests -m "not requires_research_artifact"` — see CURRENT_STATE.
- `python scripts/research/run_cohort_acquisition.py --smoke-test` (SYNTHETIC, writes `acquisition_smoke_receipt_v2.json`).
- `python scripts/research/build_verified_candidate_catalog.py --audit-legacy` (offline) → `catalog_eligibility_audit.json`.
- `pnpm continuity:check`, `git diff --check`.

### R5. NOT_MEASURED
Real download/provenance of any authentic image, inpainting on real images, generation time, peak VRAM, QC pass rates, contact sheet of real pairs, content QC, Flickr oEmbed behaviour against the live API, independent detector performance.

---

## Revision 1 (SUPERSEDED — kept as history; claims corrected in §R)

> **Phase**: Phase 4C.7B — Independent Cohort Acquisition & Feasibility Alignment<br>
> **Status**: `READY_FOR_COLAB_REAL_ACQUISITION_PILOT`<br>
> **Findings status**: `NOT_MEASURED` (zero detector scoring executed on candidate cohort)<br>
> **Evidence class**: `independent_validation_amendment_and_pipeline`<br>
> **Candidate models verified**: 5 / 5 outer-fold models bound and SHA-256 verified (unchanged)<br>
> **Training runs**: 0 new fits, 0 refits, 0 recalibrations, 0 hyperparameter tuning<br>
> **Detector scoring**: 0 predictions (Detector Isolation Invariant strictly enforced)<br>
> **Cohort size target**: $N_{\text{target}} = 400$ source pairs (800 images: 400 authentic, 400 ai_edited) + 400 binary masks<br>
> **Buffer candidate pool**: 440 pre-ordered candidates (110 per stratum: verified real photographic sources)<br>
> **Task scope**: Binary classification (`authentic` = 0 vs. `ai_edited` = 1)<br>
> **Exact blocker**: Full inpainting generation of 400 pairs using SDXL 1.0 requires GPU with $\ge 12$ GB VRAM (local workstation has NVIDIA GTX 1650 4.00 GB VRAM and 2.44 GB free RAM, empirically measured). Acquisition pipeline, real candidate catalog, fail-closed downloader, and canonical 5-cell Colab launcher are verified and ready for execution on Google Colab T4 GPU (defaulting to 8-pair Technical Pilot).

## 1. Bối Cảnh và Căn Cứ Điều Chỉnh (Protocol Amendment v1.2)

Tại Phase 4C.7A, quá trình chuẩn bị kiểm định độc lập kết thúc với trạng thái `PREPARATION_COMPLETE_PENDING_INDEPENDENT_COHORT` và blocker thực địa:
1. Thiếu 160 ảnh tự chụp từ người dùng (Field Collection).
2. Thiếu tài khoản trả phí và API thực thi khả dụng cho Adobe Firefly.

Người dùng đã chính thức đồng ý điều chỉnh kế hoạch thu thập theo nguồn lực thực tế, tuân thủ nguyên tắc trung thực khoa học:
- **Thay thế ảnh tự chụp** bằng ảnh chụp công cộng có giấy phép và provenance rõ ràng:
  * **COCO 2017 Dataset (Phân vùng sạch ngoài Option P)**: 50% (200 pairs). Thẩm tra giấy phép Flickr theo từng ảnh từ metadata chính thức `image_info_test2017.zip` (`CC-BY 2.0`, `CC-BY-SA 2.0`, `Attribution-NoDerivs`, `Attribution-NonCommercial`, v.v.), không mặc định toàn bộ ảnh cùng license.
  * **Unsplash Verified Open Collection (Công bố trước 2022)**: 50% (200 pairs). Giấy phép Unsplash License, lưu trữ photo ID, tác giả và ngày công bố ($\le 2021$). Nhận định khoa học rõ ràng: thời điểm trước 2022 và EXIF không phải bằng chứng tuyệt đối của việc chưa từng chỉnh sửa, mà là bộ lọc vận hành hợp lý chống can thiệp text-to-image AI hiện đại; ngày công bố trên Unsplash là submission date, không suy diễn thành capture date.
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
- **Quy tắc thay thế mẫu lỗi bảo toàn đa chiều (Multi-Dimensional Quota-Preserving Replacement)**:
  Nếu một candidate trong ô $(S, T)$ bị loại do lỗi download hoặc QC kỹ thuật, hệ thống chỉ chọn candidate tiếp theo trong buffer cùng ô $(S, T)$ có loại thao tác (`modification_type`) và diện tích mask (`mask_area_class`) còn thiếu quota. Tuyệt đối không làm lệch ma trận quota đã khóa.
- **Ngữ nghĩa của Mask**: Mask nhị phân $512 \times 512$ PNG xác định **vùng yêu cầu chỉnh sửa (inpainting request area)** cung cấp cho diffusion pipeline, không mặc nhiên đồng nhất tuyệt đối với việc mọi pixel bên ngoài mask đều bất biến 100% do đặc tính nén của VAE/latent space.

---

## 2. Kết Quả Đo Lường Hạ Tầng Thực Tế & Ước Tính (Resource Assessment)

### 2.1. Đo lường thực tế trên máy trạm cục bộ (Empirically Measured)
- **GPU**: NVIDIA GeForce GTX 1650 (4.00 GB GDDR6 VRAM, Compute Capability 7.5).
- **System RAM**: 15.78 GB Total, **2.44 GB Available / Free**.
- **Storage Free**: Ổ C: 52.1 GB, Ổ D: 79.0 GB.
- **Kết luận**: Khả năng chạy SDXL fp16 cục bộ là không khả thi do thiếu VRAM ($\ge 6.6$ GB) và thiếu RAM ($\ge 8$ GB cho offload).

### 2.2. Thông số ước tính cho môi trường thực thi từ xa (Estimated)
- **Môi trường mục tiêu**: Google Colab Free Tier (Tesla T4 GPU 15.0 GB VRAM) hoặc A100 GPU.
- **Thời gian sinh dự kiến**: **37–45 phút** cho toàn bộ 400 pairs (ESTIMATED, chưa đo thực nghiệm trên Colab).
- **Dung lượng lưu trữ dự kiến**: ~610 MB cho toàn bộ 400 pairs (800 ảnh + 400 masks).

---

## 3. Kiến Trúc Pipeline Thu Thập và Quality Control (Acquisition Pipeline)

Mã nguồn thực thi gồm một runner chuẩn tắc duy nhất:
- `ml/evaluation/independent_cohort_acquisition.py`: Core pipeline, downloader thật, QC kỹ thuật, durable resume.
- `scripts/research/run_cohort_acquisition.py`: Production CLI runner duy nhất (`--export-plan`, `--verify-plan`, `--smoke-test`, `--mode [pilot|full]`, `--contact-sheet`).
- `scripts/research/build_verified_candidate_catalog.py`: Khám phá và đóng băng danh mục 440 candidates thật từ COCO và Unsplash.
- `notebooks/independent_cohort_acquisition_colab.ipynb`: Notebook Colab 5 cells ngắn gọn, gọi CLI chuẩn tắc, mount Drive bền vững, không sao chép logic.

### 3.1. Các Cơ Chế Bảo Vệ Được Cài Đặt (Guards & Controls)
1. **Four-Level Disjoint Guard**:
   - Đối chiếu với toàn bộ 684 historical Option P sources (bao gồm cả 343 retired locked-test sources).
   - Kiểm tra: `source_id`, `origin_id` (Flickr / Unsplash ID), `raw_sha256` (ảnh gốc trước chuẩn hóa), `master_sha256` (ảnh sau chuẩn hóa 512x512).
2. **Fail-Closed Real Image Downloader**:
   - Chỉ tải từ official URLs đã xác minh.
   - Tuyệt đối cấm fallback sang ảnh synthetic, random noise hoặc placeholder khi download thất bại.
   - Thất bại download được ghi ngay vào `attempt_ledger.jsonl` và thay thế bằng candidate tiếp theo trong buffer.
3. **Canvas and Codec Parity**:
   - Chuẩn hóa về $512 \times 512$ RGB PNG lossless.
   - Mask nhị phân $512 \times 512$ PNG với pixel $\in \{0, 255\}$.
4. **Rigorous Technical QC**:
   - Không lỗi decode PIL, không ảnh solid/blank ($\text{std} > 2.0$), không chứa NaN/Inf.
   - Diện tích mask khớp bracket chỉ định (Small $< 10\%$, Medium $10\% - 30\%$, Large $> 30\%$).
   - Delta check: masked mean absolute difference $> 3.0$.
5. **Detector Isolation Invariant**:
   - `assert_detector_isolation()` fail-closed nếu bất kỳ module detector nào được nạp vào bộ nhớ.
   - Zero detector scoring / zero predictions trong toàn bộ pipeline.
6. **Durable On-Disk Resume & Tamper Detection**:
   - Mọi attempt và receipt ghi trực tiếp xuống `attempt_ledger.jsonl` và `provenance_ledger.jsonl`.
   - Resume tự động kiểm tra SHA-256 toàn bộ ảnh và mask đã có; fail-closed với `TamperDetectedError` nếu phát hiện sai lệch.
7. **Content QC Preservation**:
   - Sinh HTML contact sheet (`content_qc_contact_sheet.html`) hiển thị authentic, mask và ai_edited song song.
   - Zero detector scores hiển thị; tất cả mẫu giữ nguyên trạng thái `PENDING_CONTENT_QC` cho đến khi con người thẩm định.

---

## 4. Bằng Chứng Thực Nghiệm và Kiểm Chứng Kỹ Thuật (Empirical Verification)

### 4.1. Khám Phá & Đóng Băng 440 Candidates Thật
- File catalog: `research/evidence/phase-4c.7b/verified_candidate_catalog.json` (440 candidates: 220 COCO, 220 Unsplash).
- File plan chuẩn tắc: `research/evidence/phase-4c.7b/candidate_acquisition_plan.json` (`evidence_class: "verified_real_catalog"`, `eligible_for_independent_cohort: true`).
- File fixture tổng hợp cách ly: `research/evidence/phase-4c.7b/fixtures/synthetic_candidate_plan_fixture.json` (`evidence_class: "synthetic"`, `eligible_for_independent_cohort: false`).
- Kiểm chứng download thật trên máy trạm: Tải và decode thành công ảnh COCO (`(640, 480) RGB`) và Unsplash (`(600, 336) RGB`) với 0 lỗi.

### 4.2. Candidate Acquisition Plan Verification
Lệnh thực thi:
```bash
python scripts/research/run_cohort_acquisition.py --verify-plan
```
Kết quả kiểm tra:
- `total_candidates`: 440 (110 per stratum: `coco_sd2`: 110, `coco_sdxl`: 110, `unsplash_sd2`: 110, `unsplash_sdxl`: 110).
- `modification_type_counts`: `object_replacement`: 176 (40%), `object_removal_and_infill`: 132 (30%), `object_insertion`: 132 (30%).
- `mask_area_counts`: `medium_10_to_30pct`: 176 (40%), `large_over_30pct`: 132 (30%), `small_under_10pct`: 132 (30%).
- `tool_counts`: SD2: 220, SDXL: 220.
- `source_origin_counts`: COCO 2017: 220, Unsplash Verified: 220.
- `historical_overlap_detected`: `false` (0 overlapping sources, 0 overlapping origins).
- `missing_provenance_detected`: `false`.

### 4.3. Technical Smoke Test
Lệnh thực thi:
```bash
python scripts/research/run_cohort_acquisition.py --smoke-test
```
Kết quả tại `research/evidence/phase-4c.7b/acquisition_smoke_receipt.json`:
- **Trạng thái**: `PASS` (thời gian chạy: 1.09 giây).
- **Detector Isolation Invariant**: PASSED (0 detector modules loaded).
- **Plan Verification**: PASSED (440 real candidates verified).
- **Fail-Closed Synthetic Guard**: PASSED (rejected synthetic fixture in production mode).
- **Pipeline Execution**: PASSED (16 images, all QC passed).
- **Manifest & Checksum Verification**: PASSED.
- **Durable Resume & Tamper Detection**: PASSED (phát hiện chính xác disk tampering).

### 4.4. Test Suites
- **Bộ test mới** `ml/tests/test_independent_cohort_acquisition.py`: **17 / 17 tests PASSED** (11.04s).
  * Bao gồm 7 regression tests mới: rejection of synthetic candidates, download failure fail-closed, missing provenance fail-closed, historical Option P disjoint guard, durable resume & tamper detection, provisional cohort evaluator verdict, notebook 5-cell production CLI contract.
- **Bộ test hồi quy** `ml/tests/test_independent_validation_preparation.py`: **25 / 25 tests PASSED** (15.96s).

---

## 5. Hướng Dẫn Chuyển Giao và Quy Trình Thực Thi Colab (Handoff & Next Steps)

- **Trạng thái kết thúc Phase 4C.7B**: `READY_FOR_COLAB_REAL_ACQUISITION_PILOT`.
- **Cấu hình Colab**:
  1. Mở file `notebooks/independent_cohort_acquisition_colab.ipynb` trên Google Colab.
  2. Chọn `Runtime` $\rightarrow$ `Change runtime type` $\rightarrow$ `T4 GPU`.
  3. Chạy **Cell 1** (Mount Google Drive) và **Cell 2** (Git clone branch `research/independent-cohort-acquisition`).
  4. Chạy **Cell 3 (Technical Pilot)**: Mặc định chạy pilot 2 pairs/stratum (8 pairs total) trên ảnh chụp thật để đo đạc latency, peak memory và xuất HTML contact sheet.
  5. Thẩm định kết quả Pilot: Mở file `pilot_cohort/content_qc_contact_sheet.html` trên trình duyệt để kiểm tra chất lượng inpainting.
  6. Sau khi Pilot đạt, mở khóa dòng lệnh Full Acquisition trong Cell 3 để sinh đủ 400 pairs (100 pairs/stratum).
  7. Chạy **Cell 4**: Kiểm toán manifest, SHA-256 checksum, và đóng gói ZIP lưu trữ trực tiếp vào Google Drive (`/content/drive/MyDrive/forensics-web-lab/phase_4c7b_cohort/`).
  8. Tải file ZIP về máy trạm, giải nén vào `data/research/independent_cohort/` và thực hiện Content QC trước khi chuyển sang Phase 4C.7C (Evaluation).

---

## 6. Trạng Thái Đóng và Đạo Đức Nghiên Cứu

| Hạng Mục | Trạng Thái Ghi Nhận |
| :--- | :--- |
| **Giao thức nghiên cứu** | Protocol Amendment v1.2 đã khóa trước thu thập (`AMENDMENT_LOCKED_PRE_ACQUISITION`) |
| **Mô hình candidate** | 5 outer-fold models giữ nguyên 100% trọng số và SHA-256 hash đã kiểm toán |
| **Candidate sources** | 100% ảnh chụp thật có provenance xác minh từ COCO test2017 và Unsplash pre-2022 |
| **Detector isolation** | Tuyệt đối tuân thủ, zero detector calls trong thu thập và QC |
| **Independent Performance** | Tiếp tục giữ trạng thái **`NOT_MEASURED`** (chưa đánh giá) |
| **Cohort Acquisition Status** | **`NOT_ACQUIRED`** (in progress, ready for remote GPU pilot execution) |
| **Quy tắc Git** | Làm việc trên branch `research/independent-cohort-acquisition`, **không tạo Pull Request** |
| **Trạng thái Phase** | **`READY_FOR_COLAB_REAL_ACQUISITION_PILOT`** |
