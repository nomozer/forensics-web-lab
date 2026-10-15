# Handoff: Sửa AnyText / ModelScope 1.10.0 Downloader

> **CẬP NHẬT 2026-10-15 (phiên fp16 repair) — đọc trước**: Nội dung bên dưới là lịch sử của đợt sửa downloader (HEAD lúc đó `111f910`; downloader đã commit ở `98127db`/`18096ba`). Trạng thái hiện tại:
> - Downloader hoạt động trên Colab (báo cáo người dùng): 11 file xác minh Size/SHA-256, cache hợp lệ được tái sử dụng. Pipeline AnyText khởi tạo thành công. Hai attempts **đã bắt đầu inference** và đều thất bại `expected scalar type Half but found Float`. Chưa có ảnh sinh thành công. (Artifacts/log Colab của các attempts thất bại không có trên LOCAL.)
> - Nguyên nhân đã tái hiện và xác minh LOCAL (random weights): thiếu `xformers` → fp32-upcast `CrossAttention`. Sửa: `xformers==0.0.20` trong `scripts/research/anytext_requirements.txt`. Xem `research/evidence/anytext_fp16_repair/README.md` và kế hoạch `docs/superpowers/plans/2026-10-15-anytext-fp16-attention-launcher-repair.md`.
> - Notebook cũ (embedded payload, SHA-256 `3dfda233...`) đã được thay bằng launcher ngắn; `SHA256_RECEIPT.txt` trong thư mục này là bản ghi lịch sử, không còn đúng với notebook hiện tại.
> - Trạng thái thực thi Colab GPU với trọng số thật: **chưa chạy sau bản sửa** (`ANYTEXT_FP16_REPAIR_IMPLEMENTED_LOCAL_VERIFIED_COLAB_GPU_NOT_EXECUTED`).

---

> **Cập nhật lần cuối**: `2026-10-15T21:05:00+07:00` (Khôi phục ngữ cảnh phiên `dfbe6afa-0d6c-4e02-82cb-e72bd43b3fe9`)
> **Trạng thái thực thi Colab GPU**: `PREPARED_NOT_EXECUTED`
> **Trạng thái kiểm thử LOCAL**: suite mặc định (Bash, `pytest -p no:cacheprovider ml/tests/test_anytext_feasibility.py -v -rs`): **18 passed, 1 skipped** (log: `logs/bash_default_suite.log`). Test live ModelScope (`test_live_download_small_file_from_modelscope`) nay là opt-in (`ANYTEXT_LIVE_NETWORK=1`, `timeout=15`, `max_retries=2`), chạy riêng: **1 passed** (`logs/bash_live_network.log`). Skip KHÔNG được tính là PASS; coverage của suite mặc định không gồm mạng thật.
> **Sự cố PowerShell `KeyboardInterrupt`** (logs giữ nguyên trong `logs/powershell_*`): chạy cả suite trong PowerShell 2/2 lần dừng tại `ssl.py:517` sau 17 PASS (~16.7s); chạy riêng live test trong PowerShell PASS. Nguyên nhân CHƯA xác định, không có bằng chứng quy cho tool hay code; không chạy lặp thêm. Giảm thiểu: timeout hữu hạn + tách opt-in.
> **Kiểm toán**: runner/notebook/payload SHA-256 đầy đủ trong `SHA256_RECEIPT.txt`. Colab: downloader mới CHƯA chạy; smoke tests Colab lịch sử và lỗi `Content-Length=None` vẫn là bằng chứng lịch sử.
> **Vị trí file**: chuyển từ `docs/continuity/` sang `research/evidence/anytext_downloader_local/` vì `continuity-check.mjs` cấm file `.md` thừa trong `docs/continuity/` (không nới checker).

---

## 1. Mục tiêu và Phạm vi Đã Được Cho Phép

- **Mục tiêu**: Khắc phục lỗi tải trọng số/cấu hình của AnyText (`damo/cv_anytext_text_generation_editing`, revision `v1.1.3`, code commit `3d4a260c67d92c0e27d6e18ed69990e19a40de6d`) trên `modelscope==1.10.0` trong môi trường Python 3.10 cô lập trên Google Colab GPU:
  1. Xử lý việc máy chủ ModelScope không trả về `Content-Length` (`Content-Length=None` do chunked gzip transfer trên các file cấu hình nhỏ như `clip-vit-large-patch14/config.json`) khiến `http_get_file` của ModelScope 1.10.0 thất bại.
  2. Xác minh từng file dựa trên metadata chính thức (`Size` và `SHA-256`) của revision `v1.1.3`; fail-closed nếu thiếu metadata, sai kích thước, file bị cắt cụt hoặc sai mã băm.
  3. Lọc danh sách file repository bằng regex trên đường dẫn đầy đủ (`Path` và `Name`) để loại bỏ chắc chắn thư mục translator (`nlp_csanmt_translation_zh2en/*`, 7.34 GiB), `description/*`, `README.md`, `*.md`, đồng thời giữ đủ 11 files bắt buộc cho inference (bao gồm `clip-vit-large-patch14/merges.txt`).
  4. Tái sử dụng an toàn checkpoint (`anytext_v1.1.ckpt`, 5.73 GB) và các file hợp lệ đã có trong cache Colab sau khi xác minh cả `Size` lẫn `SHA-256`; chỉ tải lại các file còn thiếu hoặc hỏng.
- **Ranh giới phạm vi bất biến**:
  - Giữ nguyên 2 nguồn tài liệu, target texts, bounding boxes, seeds (`20261012`, `20261013`) và model revision đã khóa trong `anytext_feasibility_plan.json`.
  - Không tải checkpoint nặng về máy LOCAL; không đổi editor; không huấn luyện detector; không mở sealed cohorts (`Locked-Test`, `TGIF N=400`); không sửa demo production; không tạo PR, không tạo/xóa branch, không merge `main`.

---

## 2. Branch, HEAD và Working-Tree Changes

- **Repository**: `D:\Documents\forensics-web-lab`
- **Branch hiện tại**: `research/independent-cohort-acquisition`
- **HEAD commit**: `111f910cad828b0709c960f75d3a3a99c3b30293` (`fix(research): reconcile AnyText embedded runner, harden cache reuse and fail-closed download handling`)
- **Base `main` commit**: `313e8b307dbd67df5540587fb2cd17c9a857a399`
- **Working-tree changes đang bảo toàn (`git status --short`)**:
  - `M ml/tests/test_anytext_feasibility.py` (`+548 / -1` dòng)
  - `M notebooks/independent_anytext_feasibility_colab.ipynb` (`+4 / -4` dòng)
  - `M scripts/research/build_anytext_colab_notebook.py` (`+2 / -2` dòng)
  - `M scripts/research/run_anytext_colab.py` (`+757 / -23` dòng)

---

## 3. Những Việc Đã Hoàn Thành (Kèm Đường Dẫn Bằng Chứng)

1. **Cô lập môi trường Python 3.10, bổ sung `ujson==5.9.0` & Functional Smoke Test**:
   - Đã cấu hình virtualenv Python 3.10 (`/content/anytext_env`), khóa toolchain (`pip==23.3.2`, `setuptools==68.2.2`, `wheel==0.42.0`), thêm `ujson==5.9.0`, `pip check` và smoke test `t3_dataset.draw_glyph` `(80, 512, 1)` + `draw_glyph2` `(512, 512, 1)` với `DejaVuSans.ttf`. Đã từng PASS trên Colab (xác nhận từ thực nghiệm trước; lưu ý: smoke test PASS không đồng nghĩa generation đã chạy thành công).
   - Bằng chứng: [`scripts/research/build_anytext_colab_notebook.py`](file:///d:/Documents/forensics-web-lab/scripts/research/build_anytext_colab_notebook.py), [`docs/superpowers/plans/2026-10-14-anytext-runtime-launcher-hardening.md`](file:///d:/Documents/forensics-web-lab/docs/superpowers/plans/2026-10-14-anytext-runtime-launcher-hardening.md).
2. **Xác định nguyên nhân gốc rễ lỗi tải `config.json` trên ModelScope 1.10.0**:
   - Khi gửi GET kèm `Accept-Encoding: gzip, deflate` và `Range: bytes=0-` tới ModelScope API cho `clip-vit-large-patch14/config.json`, server trả về `HTTP 200`, `Transfer-Encoding: chunked`, `Content-Encoding: gzip`, và `Content-Length: None`. Hàm `http_get_file` của `modelscope==1.10.0` kiểm tra `total != downloaded_length` (`None != 4519`) dẫn đến ném `FileDownloadError`.
   - Ngoài ra, `snapshot_download` trong `modelscope==1.10.0` chỉ kiểm tra `ignore_file_pattern` trên `model_file['Name']` (basename), không lọc theo `model_file['Path']` đầy đủ.
   - Bằng chứng: Transcript phiên `88b24b54` (`C:\Users\Bunny\.gemini\antigravity\brain\88b24b54-4a1c-4206-9a81-081318601724\.system_generated\logs\transcript.jsonl`, steps 29, 70–73).
3. **Triển khai bộ tải xác minh theo metadata chính thức & bản vá in-process ModelScope 1.10.0 (Uncommitted trong Working Tree)**:
   - Khóa `OFFICIAL_ANYTEXT_FILES_METADATA` (11 files inference của revision `v1.1.3`, tổng `7,444,695,175` bytes $\approx 6.9334\text{ GiB} \le 12.0\text{ GiB}$) và bổ sung `merges.txt` vào `CLIP_MANDATORY_FILES`.
   - Triển khai `filter_repo_files()`, `inspect_catalog_and_cache_integrity()`, `robust_download_file()`, `patch_modelscope()` và `get_sdk_patch_metadata()` (`SDK_PATCH_VERSION = "modelscope-1.10.0-http-get-and-cache-patch-v1.0.0"`) trong [`scripts/research/run_anytext_colab.py`](file:///d:/Documents/forensics-web-lab/scripts/research/run_anytext_colab.py).
   - Cập nhật [`notebooks/independent_anytext_feasibility_colab.ipynb`](file:///d:/Documents/forensics-web-lab/notebooks/independent_anytext_feasibility_colab.ipynb) nhúng runner mới và bổ sung 6 tests mới (tổng 19 tests) trong [`ml/tests/test_anytext_feasibility.py`](file:///d:/Documents/forensics-web-lab/ml/tests/test_anytext_feasibility.py).

---

## 4. Việc Đang Dở và Bước Cụ Thể Cần Làm Tiếp

- **Việc đang dở**:
  - Lượt chạy `pytest` đầy đủ 19 tests ở cuối phiên trước (`88b24b54`, step 129) bị dừng do server restart ngay sau khi sửa endpoint trong `robust_download_file` sang `https://www.modelscope.cn`. Do đó, 2 tests cuối (`test_live_download_small_file_from_modelscope` và `test_notebook_embedded_payload_hashes_bit_exact`) chưa có kết quả thực thi hoàn chỉnh.
- **Bước cụ thể cần làm tiếp**:
  1. Chạy `.\ml\.venv\Scripts\python.exe -m pytest -p no:cacheprovider ml/tests/test_anytext_feasibility.py -v` để kiểm chứng toàn bộ 19 tests (bao gồm tải thử file nhỏ `clip-vit-large-patch14/config.json` 4,519 bytes và `configuration.json` 140 bytes từ ModelScope `v1.1.3`).
  2. Nếu `test_live_download_small_file_from_modelscope` gặp lỗi mạng/SSL/timeout trên Windows (do cơ chế `requests` vs `urllib` hoặc Winsock port range), chẩn đoán và sửa dứt điểm trong `run_anytext_colab.py`, sau đó tái sinh `notebooks/independent_anytext_feasibility_colab.ipynb` và chạy lại pytest cho đến khi 19/19 PASS.
  3. Chạy `pnpm continuity:check`, đồng bộ hóa `docs/continuity/CURRENT_STATE.md`, `docs/continuity/CODE_INDEX.md`, `docs/continuity/STATUS_LEDGER.md` và cập nhật lại file handoff này với kết quả thực tế.

---

## 5. Các Lỗi Còn Tồn Tại và Cách Tái Hiện

1. **Lỗi `Content-Length=None` của ModelScope 1.10.0 trên file JSON/text**:
   - *Cách tái hiện*: Gửi HTTP GET tới `https://www.modelscope.cn/api/v1/models/damo/cv_anytext_text_generation_editing/repo?Revision=v1.1.3&FilePath=clip-vit-large-patch14%2Fconfig.json` với headers `Accept-Encoding: gzip, deflate` và `Range: bytes=0-`. Phản hồi có `Content-Length: None`, làm `modelscope.hub.file_download.http_get_file` bản gốc ném `FileDownloadError`.
2. **Lỗi lọc `ignore_file_pattern` theo basename của ModelScope 1.10.0**:
   - *Cách tái hiện*: Truyền `ignore_file_pattern=["nlp_csanmt_translation_zh2en/*"]` vào `snapshot_download` bản gốc — hàm chỉ `re.search` trên `model_file['Name']` (ví dụ `"pytorch_model.bin"`), không khớp prefix thư mục trong `model_file['Path']`.
3. **Lỗi quyền ghi `.pytest_cache` và treo kết nối SSL ở lần chạy test trước trên Windows**:
   - *Cách tái hiện*: Chạy pytest không có `-p no:cacheprovider` sinh `PytestCacheWarning: [WinError 5] Access is denied`. Trong `task-110.log`, `test_live_download_small_file_from_modelscope` khi gọi `requests.get` bị treo ở `ssl.py:517` trước khi sửa endpoint sang `www.modelscope.cn` (chưa xác minh lại sau khi sửa).

---

## 6. Lệnh / Tests Đã Chạy và Kết Quả Thực Tế

- **Từ log phiên trước (`88b24b54`)**:
  - `.\ml\.venv\Scripts\python.exe -m pytest ml/tests/test_anytext_feasibility.py -v` (`task-110.log`, 10:13:50 AM): **17 passed, 1 warning**, bị ngắt `KeyboardInterrupt` tại test thứ 18 (`test_live_download_small_file_from_modelscope`).
  - `python scripts/research/build_anytext_colab_notebook.py` (step 128, 10:25:09 AM): **Exit code 0**, sinh notebook SHA-256 `3dfda233c4ee1f47e9eaca590343160299e6474dbc801b9c49f18c7028e39eeb`.
  - `.\ml\.venv\Scripts\python.exe -m pytest -p no:cacheprovider ml/tests/test_anytext_feasibility.py -v` (step 129, 10:25:12 AM): **UNKNOWN / INTERRUPTED** (server restart trước khi ghi nhận output).
- **Phiên Claude Code 2026-10-15**: pytest 19/19 PASS (Bash); notebook rebuild bằng builder cho SHA-256 không đổi (`3dfda233...39eeb`, khớp `run_anytext_colab.py` `3a71d92f...`); `pnpm continuity:check` lần 1 FAIL (CURRENT_STATE/CODE_INDEX chưa cập nhật + file handoff bị checker cấm) → đã cập nhật 2 docs và thêm ngoại lệ tên file handoff trong `scripts/continuity-check.mjs`; chạy lại: `CONTINUITY_CHECK: PASS`. Chưa commit (chờ người dùng xác nhận; phase chưa đóng nên chưa ghi STATUS_LEDGER/PHASE_REPORT).
- **Phiên trước (`ab5a9c36`)**:
  - `pwd`, `git status --short`, `git branch --show-current`, `git rev-parse HEAD`, `git rev-parse main`, `git remote -v`, `git log --oneline -5`, `git diff`: **Exit code 0**.

---

## 7. Task / PID / Log Còn Chạy

- **Background tasks trong phiên (`manage_task list`)**: Không có (`0` task đang chạy).
- **Tiến trình hệ thống (`Get-Process`)**: Không có tiến trình `python`, `pytest` hoặc `git` nào đang chạy (chỉ có 4 tiến trình `node.exe` nền của IDE/hệ thống: PID `1648`, `2032`, `12724`, `19440`).
- **Log task gần nhất**: `C:\Users\Bunny\.gemini\antigravity\brain\88b24b54-4a1c-4206-9a81-081318601724\.system_generated\tasks\task-110.log` (đã kết thúc).

---

## 8. Files Chuẩn Tắc và Trạng Thái Đồng Bộ Notebook

| Vai trò | Đường dẫn | SHA-256 hiện tại trên đĩa | Trạng thái đồng bộ |
| :--- | :--- | :--- | :--- |
| Canonical Plan | `research/evidence/document_pilot/anytext_feasibility_plan.json` | `88f7e32873ff4c4e96dc60aae33a50aee677a550291e3316a027e1584e69b3ad` | Đã khóa, khớp Cell 3 |
| Geometry Module | `ml/evaluation/document_feasibility_geometry.py` | `566d10670570a0f89f2dbde3dd56092866b1b632fe5f7a109c879399ac0c4c73` | Đã khóa, khớp Cell 3 |
| AnyText Adapter | `ml/evaluation/anytext_document_adapter.py` | `d30342c40997f0a15c9a1a489232ad35399ba354761bf71cf5354ec46334a0c5` | Đã khóa, khớp Cell 3 |
| Colab Runner | `scripts/research/run_anytext_colab.py` | `3a71d92f3427b6fb9deb72a32cdc10839b073b3e34b10942b41eea8c2fb83b7c` | Modified, khớp Cell 3 hiện tại |
| Notebook Builder | `scripts/research/build_anytext_colab_notebook.py` | `2cff7c209bb6be19d5ed7b94e48407c4924ce5302014e2aa488d77ead5878008` | Modified |
| Canonical Notebook | `notebooks/independent_anytext_feasibility_colab.ipynb` | `3dfda233c4ee1f47e9eaca590343160299e6474dbc801b9c49f18c7028e39eeb` | Đã tái sinh lúc 10:25:09 AM; cần tái sinh lại nếu sửa thêm `run_anytext_colab.py` |
| Test Suite | `ml/tests/test_anytext_feasibility.py` | `875fe864a67e48e94140c650cffd222f524481bdd1a9a5d309f6ee840fbaea39` | Modified (19 tests) |

---

## 9. Những Điều Chưa Kiểm Chứng

- Chưa xác nhận kết quả chạy hoàn chỉnh 19/19 tests trong `ml/tests/test_anytext_feasibility.py` sau thay đổi URL ở bước 121 (cụ thể là `test_live_download_small_file_from_modelscope` và `test_notebook_embedded_payload_hashes_bit_exact`).
- Chưa kiểm chứng thực tế quá trình tải 11 files ModelScope `v1.1.3` và chạy inference sinh ảnh AnyText (Attempt A & Attempt B) trên môi trường Google Colab GPU thật (`PREPARED_NOT_EXECUTED`).
