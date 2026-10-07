# Phase 4C.7B — Independent Cohort Acquisition: Protocol Amendment, Automated Pipeline, and Feasibility Alignment

> **Phase**: Phase 4C.7B — Independent Cohort Acquisition & Feasibility Alignment (revision 5: Generation Contract Enforcement, Systematic Failure Gate Hotfix, durable logging, pilot NOT_RUN)<br>
> **Status**: `READY_FOR_COLAB_REAL_ACQUISITION_PILOT` (revision 5, supersedes revision 4). Incident in pilot run `pilot-20261007T093824Z` (commit `cdacb41a5501568c1a11435dfcb00d327bc69a6d`) investigated and root cause proven: `StableDiffusionXLInpaintPipeline` defaults resolution to 1024x1024 without explicit `height`/`width` arguments, causing QC dimension failures misclassified as content QC rejections that burned the 110-candidate pool in `coco_sdxl`. Resolved via explicit `height=512, width=512` in Diffusers pipeline call, canvas contract assertions, immediate halt on `GenerationContractError`, persistent `acquisition.log`, and updated Colab notebook runs path.<br>
> **Findings status**: `NOT_MEASURED` (0 detector calls, 0 cohort evaluation)<br>
> **Functional commit**: will be pinned to functional commit SHA before notebook push<br>
> **Real pilot**: `NOT_RUN` (Colab T4 GPU execution required; 8-pair pilot ready with fresh RUN_ID)<br>
> **Training runs**: 0 fits, 0 refits; frozen models untouched; retired locked-test not accessed (only its 343 source IDs are read for the disjoint guard)<br>

---

## U. Revision 5 — Generation Contract Enforcement & Systematic Failure Gate Hotfix (2026-10-07)

### U1. Incident Report & Root Cause Analysis (`pilot-20261007T093824Z`)
- **Sự cố thực địa**: Pilot run `pilot-20261007T093824Z` (commit `cdacb41a5501568c1a11435dfcb00d327bc69a6d`) trên Google Colab T4 GPU dừng với lỗi:
  `StratumQuotaDeficitError: Stratum 'coco_sdxl' exhausted all candidates without reaching quota: 0/2 valid pairs acquired.`
- **Phân tích Ledgers**:
  * `provenance_ledger.jsonl`: Có đúng 2 records thuộc stratum `coco_sd2` (đều pass technical QC, resolution 512x512, status `ACCEPTED`).
  * `attempt_ledger.jsonl`: Có 222 records tổng cộng.
  * Phân tích theo stratum và tool:
    - Stratum `coco_sd2` (`stable_diffusion_2_inpainting`): 2 attempts đều `ACCEPTED` (đạt quota 2/2).
    - Stratum `coco_sdxl` (`sdxl_inpainting`): 110 attempts liên tiếp đều thất bại với cùng lý do:
      `QC_FAILED — Edited image size/mode invalid: (1024, 1024), RGB`.
- **Root Cause Kỹ thuật**:
  * Trong `DiffusersInpaintingEngine.inpaint()`, `self.pipeline(...)` được gọi mà không truyền tham số `height` và `width`.
  * Trong thư viện `diffusers`, `StableDiffusionXLInpaintPipeline.__call__` tính toán:
    `height = height or self.default_sample_size * self.vae_scale_factor`
    Với SDXL: `default_sample_size = 128`, `vae_scale_factor = 8` $\to 128 \times 8 = 1024$. Do đó pipeline sinh ảnh $1024 \times 1024$ mặc dù ảnh đầu vào và mask là $512 \times 512$.
    Với SD2: `default_sample_size = 64`, `vae_scale_factor = 8` $\to 64 \times 8 = 512$. Do đó SD2 tình cờ sinh đúng $512 \times 512$.
  * Lỗi generation contract hệ thống này bị hàm `evaluate_technical_qc` trả về `False, "Edited image size/mode invalid: (1024, 1024), RGB"`, khiến runner ghi nhận `status: QC_FAILED` và tiếp tục thử các ứng viên tiếp theo trong pool (`continue`), làm cạn kiệt toàn bộ 110 ứng viên của `coco_sdxl` trước khi dừng với `StratumQuotaDeficitError`.

### U2. Giải pháp Generation Contract & Systematic Failure Gate
1. **Khóa Kích thước Tường minh**:
   - Thêm `"target_height": 512, "target_width": 512` vào `INPAINTING_MODEL_REGISTRY` cho cả hai model SD2 và SDXL.
   - `DiffusersInpaintingEngine.inpaint()` nhận tường minh `height=512, width=512` và truyền trực tiếp vào `self.pipeline(..., height=height, width=width, ...)`.
   - Tuyệt đối không resize ảnh 1024 xuống 512 để vượt QC; pipeline phải tạo sinh trực tiếp ở kích thước 512x512.
2. **Kiểm tra Contract Canvas Trước và Sau Inpainting**:
   - Trước inpainting: kiểm tra `auth_canvas` (512x512 RGB) và `mask_canvas` (512x512 L). Vi phạm ném `GenerationContractError`.
   - Sau inpainting: kiểm tra `edited_canvas` (512x512 RGB). Vi phạm ném `GenerationContractError`.
3. **Systematic Failure Gate (Ngăn Cháy Pool Ứng Viên)**:
   - Phân biệt triệt để lỗi contract/cấu hình hệ thống với lỗi content QC của từng ứng viên:
     * Vi phạm generation contract (`GenerationContractError`): Ghi attempt `status: GENERATION_CONTRACT_ERROR`, xuất `out_path / "failure_receipt.json"`, ghi log lỗi với full traceback và **dừng ngay lập tức** mà không thử tiếp ứng viên khác.
     * Content QC rejection hợp lệ (ảnh đen, biến thiên thấp, mask ratio ngoài ngưỡng): Ghi `status: QC_FAILED`, tiếp tục cơ chế thay thế ứng viên trong stratum để đảm bảo quota.
   - `audit_acquisition_run` fail-closed: Từ chối ngay lập tức nếu phát hiện `failure_receipt.json`.
4. **Logging Bền vững & Colab Notebook**:
   - Tạo file log bền vững `run_dir / "acquisition.log"` ghi nhận timestamp, candidate id, stratum id, trạng thái và full exception traceback.
   - Hàm `run()` trong Colab notebook dùng `subprocess.Popen(..., stderr=subprocess.STDOUT)` stream trực tiếp cả stdout và stderr lên giao diện Colab.
   - Cập nhật đường dẫn mặc định: `/content/drive/MyDrive/Colab Notebooks/forensics-web-lab/phase_4c7b/runs/<RUN_ID>`.
   - Xóa cờ hoàn tất cũ `globals().pop("PILOT_RUN_COMPLETED", None)` tại đầu Cell 3 trước mỗi lần chạy; chỉ audit và đóng gói ZIP khi lần chạy hiện tại thành công.
   - Bảo toàn nguyên vẹn thư mục run cũ `pilot-20261007T093824Z` tại `MyDrive/forensics-web-lab/phase_4c7b_runs/`.

### U3. Kết quả Kiểm thử & Xác minh
- **Regression test suite**: Bổ sung 4 targeted tests trong `ml/tests/test_independent_cohort_bindings.py`:
  * `test_diffusers_engine_passes_explicit_height_width_and_validates_dimensions`: kiểm tra pipeline nhận đúng `height=512, width=512`, từ chối input/output sai kích thước.
  * `test_output_1024_halts_immediately_and_does_not_try_next_candidate`: kiểm tra output 1024 dừng ngay sau ứng viên đầu tiên (call count = 1), ghi failure receipt và attempt ledger `GENERATION_CONTRACT_ERROR`.
  * `test_output_correct_size_proceeds_through_qc_and_handles_content_qc_rejection`: kiểm tra output 512x512 đi tiếp qua QC, lỗi content QC tiếp tục thay thế ứng viên hợp lệ.
  * `test_generation_contract_failure_prevents_completion_receipt_and_blocks_audit`: kiểm tra run thất bại không tạo completion receipt, bị `audit_acquisition_run` từ chối và chặn đóng gói ZIP.
- **Toàn bộ test suites**: 75/75 tests PASS trong `ml/tests/test_independent_cohort_bindings.py` (58/58) và `ml/tests/test_independent_cohort_acquisition.py` (17/17).
- **Hermetic smoke test**: PASS trong 1.37s (`acquisition_smoke_receipt_v2.json`).
- **Plan verification**: PASS (440 candidates, 110 per stratum, 0 historical overlap).
- **Model preflight**: PASS cho cả hai model.
- **Tính trung thực khoa học**: Real generation pilot `NOT_RUN` ở local (do không có GPU); các chỉ số detector `NOT_MEASURED`.

---

## T. Revision 4 — Protocol Amendment v1.3.1 Model Source Hotfix, Community Mirror Qualification & Fail-Closed Preflight (2026-10-07)

### T1. Incident Report & Empirical Reproduction
- **Sự cố thực địa**: Pilot run `pilot-20261007T082516Z` (commit `3710762868a007af4fe79798bad79d086abcd5c8`) trên Google Colab T4 GPU vượt qua GPU policy check nhưng dừng tại:
  `HfApi().model_info("stabilityai/stable-diffusion-2-inpainting")`
  với ngoại lệ `RepositoryNotFoundError` / HTTP 401 Unauthorized (`{"error":"Invalid username or password."}`).
- **Nguyên nhân kỹ thuật**: Stability AI đã chuyển repo `stabilityai/stable-diffusion-2-inpainting` sang chế độ gated/restricted/deprecated. Hugging Face API trả về HTTP 401 đối với các truy vấn ẩn danh không token để tránh rò rỉ sự tồn tại của private/gated repo.
- **Trạng thái thực tế**: Attempt ledger trên đĩa được xác nhận là trống; 0 receipts được ghi nhận. Thư mục run cũ và log lỗi được bảo toàn nguyên vẹn, không chỉnh sửa binding cũ để gượng ép `--resume`.
- **Tái hiện thực nghiệm**: Tái hiện thành công qua API metadata mà không sinh ảnh cục bộ; xác nhận phân biệt giữa lỗi mạng, repo 404, repo gated/private 401 và token không hợp lệ. Không yêu cầu người dùng nhập token hoặc in token vào log/chat.

### T2. Thẩm định Community Mirror & Protocol Amendment v1.3.1
- **Thẩm định nguồn thay thế `sd2-community/stable-diffusion-2-inpainting`**:
  * Đọc model card, license, model configs và Git LFS hashes.
  * Giấy phép: `openrail++` (CreativeML OpenRAIL++).
  * Pipeline class: `StableDiffusionInpaintPipeline`.
  * Configs: `model_index.json`, `unet/config.json` (sample_size 64, in_channels 9, out_channels 4), `scheduler/scheduler_config.json` (DDIM).
  * Revision đầy đủ: `5f74973cbb64c8568780732c17f43eb269d63a0d`.
  * LFS OIDs & file sizes đã thẩm định:
    - `unet/diffusion_pytorch_model.fp16.safetensors`: `29a698f37775d5904a958c9cebed98184483dfb441729a8e5f98dd5b65df70c8` (1,731,933,536 bytes).
    - `512-inpainting-ema.safetensors`: `b29e2ed9a8fe58e76f7e801bda091d23738bd74c1da3f339bcbe2d40922fcb60` (5,214,662,094 bytes).
    - `vae/diffusion_pytorch_model.fp16.safetensors`: `3e4c08995484ee61270175e9e7a072b66a6e4eeb5f0c266667fe1f45b90daf9a` (167,335,342 bytes).
    - `text_encoder/model.fp16.safetensors`: `681c555376658c81dc273f2d737a2aeb23ddb6d1d8e5b3a7064636d359a22668` (680,821,096 bytes).
  * **Định danh minh bạch**: Được ghi nhận là `community_mirror` trong tài liệu và specification. Tuyệt đối không gọi là nguồn chính thức và không tuyên bố trọng số bit-exact mà không có bằng chứng đối sánh trực tiếp.
  * **Bảo toàn giao thức**: Giữ nguyên kiến trúc UNet 9-channel inpainting, DDIM scheduler, 50 inference steps, guidance scale 7.5, fixed seeds, exact prompts và quota ma trận $2 \times 2$ (100 cặp mục tiêu / 110 pool cho mỗi stratum).
  * **SDXL Inpainting**: Thẩm định nguồn chính thức `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` (full revision `115134f363124c53c7d878647567d04daf26e41e`, UNet fp16 LFS `6470840731e98cc16713ddf3ac7ee458c9fdbcb881a98c6727cd4a938f227d3f`).
  * **Bất biến**: Không thay thế bằng SD1.5 hoặc model khác; không cho phép fallback tự động giữa các model.

### T3. Cơ chế Preflight Gate & Load Binding
- **Hàm preflight**: `verify_model_access_preflight(tool_keys, timeout, http_opener)` kiểm tra metadata API, 3 config files chính và gửi HTTP HEAD request tới safetensors weight files.
- **Fail-closed boundary**: Chạy strictly TRƯỚC KHI tạo thư mục run trên đĩa (`run_dir = Path(args.output_root) / args.run_id`). Nếu preflight thất bại, tiến trình dừng ngay lập tức và 0 thư mục rác/mồ côi được tạo ra.
- **CLI flag**: Bổ sung `--check-models` cho phép kiểm tra độc lập nhanh chóng (< 3 giây, 0 byte weights tải về máy).
- **Revision binding**: Revision được resolve trong preflight và chuyển trực tiếp vào `DiffusersInpaintingEngine`, đảm bảo nạp đúng chính xác revision đã kiểm tra.

### T4. Kết quả Kiểm thử & Xác minh
- **Regression test suite**: Bổ sung 8 tests trong `ml/tests/test_independent_cohort_bindings.py`:
  * HTTP 401 handling (`ModelPreflightError` phân loại `AUTHENTICATION_OR_GATED_REPO`).
  * HTTP 404 handling (`REPOSITORY_NOT_FOUND`).
  * Lỗi mạng (`NETWORK_OR_TIMEOUT_ERROR`).
  * Gated repo detection.
  * Live mirror và SDXL resolution.
  * Diffusers engine binding kiểm tra revision.
  * Fail-closed preflight ngăn chặn tạo thư mục run khi thất bại.
  * CLI `--check-models` thực thi thành công.
- **Tổng kết test**: 71/71 tests PASS (`test_independent_cohort_bindings.py`: 54/54, `test_independent_cohort_acquisition.py`: 17/17).
- **Smoke test**: `python scripts/research/run_cohort_acquisition.py --smoke-test` PASS trong 2.58s (`acquisition_smoke_receipt_v2.json`).
- **Plan verification**: `python scripts/research/run_cohort_acquisition.py --verify-plan` PASS (440 candidates, 110/stratum).
- **Colab Launcher**: Cập nhật Cell 2 bổ sung `--check-models` và ghim notebook vào commit functional mới.

---

## S. Revision 3 — Protocol Amendment v1.3 & Local Verified Catalog Build (2026-10-07)

### S1. Protocol Amendment v1.3 Resolution
- **Thay thế Unsplash Lite bằng Wikimedia Commons**: Do điều khoản Unsplash Lite Dataset Terms §2.A/§3.A–B chỉ cho phép download/store cho nghiên cứu nội bộ và cấm tái phân phối dữ liệu, việc tạo và công bố tập dữ liệu chỉnh sửa (inpainting) không được bảo hộ. Protocol Amendment v1.3 chính thức thay thế Unsplash Lite bằng ảnh chụp từ **Wikimedia Commons** (`Category:Quality_images`).
- **Nguồn COCO 2017 Clean val split**: Giữ nguyên nguồn COCO với điều kiện kiểm tra creator và license qua Flickr oEmbed live API. Trích xuất metadata từ `captions_val2017.json` (chứa đầy đủ `flickr_url` và Flickr photo ID; phân vùng `test2017` không chứa trường này). Loại bỏ toàn bộ 684 Option P IDs (còn 4,316 ảnh hoàn toàn disjoint).
- **Bốn Strata đối xứng**:
  1. `coco_sd2`: 100 cặp mục tiêu (110 pool)
  2. `coco_sdxl`: 100 cặp mục tiêu (110 pool)
  3. `commons_sd2`: 100 cặp mục tiêu (110 pool)
  4. `commons_sdxl`: 100 cặp mục tiêu (110 pool)
  Tổng cộng: 400 cặp mục tiêu chính thức, 440 candidates trong buffer pool.
- **Bảo toàn hạn ngạch (Quota Preservation)**: 40% replacement (160/176), 30% removal (120/132), 30% insertion (120/132); 30% small (120/132), 40% medium (160/176), 30% large (120/132).
- **Bất biến Zero Detector Scoring**: Tuyệt đối không gọi hay suy luận bất kỳ mô hình detector nào trong quá trình tuyển chọn hoặc tạo dữ liệu.

### S2. Kết quả Chạy Production Catalog Builder v2 ở Local
Chạy `scripts/research/build_verified_candidate_catalog.py` qua mạng local có checkpointing tại `artifacts/catalog_checkpoints/`:
- **COCO 2017**: Dùng `RemoteZipFile` range-based streaming đọc trực tiếp `captions_val2017.json` (chỉ tải 805 KB thay vì toàn bộ archive annotations). Thẩm tra 220 ứng viên hợp lệ qua Flickr oEmbed live API với bounded retry và rate limiting (loại các ảnh bị xóa/private/lỗi license).
- **Wikimedia Commons**: Truy vấn MediaWiki API `Category:Quality_images`, đọc `imageinfo` extmetadata, hỗ trợ continuation token (`gcmcontinue`), lọc giấy phép CC BY / CC BY-SA, loại NoDerivs. Thêm header `User-Agent` chuẩn MediaWiki policy giải quyết lỗi HTTP 403 Forbidden.
- **Tổng kết catalog**:
  * Đạt chính xác **220 COCO + 220 Wikimedia Commons = 440 eligible candidates** trong `research/evidence/phase-4c.7b/verified_candidate_catalog_v2.json`.
  * Ghi nhận 221 excluded candidates kèm lý do chi tiết (CREATOR_UNVERIFIED, LICENSE_ND_NO_EDITED_EXPORT, NO_CC_LICENSE_AT_SOURCE, HTTP_ERROR, v.v.).
- **Xác minh Kế hoạch**: Chạy `python scripts/research/run_cohort_acquisition.py --verify-plan` đạt kết quả **PASS** (440 candidates, đúng 110 per stratum, 0 historical overlap). Xuất file kế hoạch `research/evidence/phase-4c.7b/candidate_acquisition_plan_v2.json`.

### S3. Thực nghiệm Tải và Decode Mẫu Thật Ngoài Git
Thực hiện tải và chuẩn hóa canvas thật trên 4 mẫu ảnh đại diện cho 4 strata tại `artifacts/pilot_download_test/`:
- `IND_COCO_SD2_001` (COCO): Pot Noodle, Attribution License, 200,576 bytes, tỷ lệ mask 0.0773 (small).
- `IND_COCO_SDXL_001` (COCO): mike ambs, CC BY-NC-SA, 106,547 bytes, tỷ lệ mask 0.4451 (large).
- `IND_COMMONS_SD2_001` (Commons): Moahim, CC BY-SA 4.0, 4,205,835 bytes, tỷ lệ mask 0.2397 (medium).
- `IND_COMMONS_SDXL_001` (Commons): Crisco 1492, CC BY-SA 4.0, 3,355,898 bytes, tỷ lệ mask 0.0700 (small).
Tất cả 4 mẫu giải mã PIL RGB hoàn hảo, chuẩn hóa canvas $512 \times 512$ PNG thành công.

### S4. Kết quả Kiểm thử Toàn diện
- Targeted & contract tests: 63/63 tests PASS trong `test_independent_cohort_bindings.py` (46/46) và `test_independent_cohort_acquisition.py` (17/17).
- Smoke test hermetic: `run_cohort_acquisition.py --smoke-test` PASS trong 1.22s (`acquisition_smoke_receipt_v2.json`).
- Continuity checker: `pnpm continuity:check` PASS.
- Monorepo test suite: `pnpm test` PASS (13/13 unit and contract tests).

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
