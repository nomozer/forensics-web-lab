# Protocol Amendment v1.7.0 — Content-Grounded Pilot Plan Substitution and Extension

Metadata:

- Workstream: `independent_cohort_acquisition`
- Phase trace: `Phase 4C.7B`
- Amendment version: `1.7.0`
- Status: **`PENDING_HUMAN_REVIEW`**
- Human reviewer: `null`
- Human reviewed at: `null`
- Bound plan proposal: [`content_grounded_pilot_plan_v2_proposal.json`](research/evidence/phase-4c.7b/content_grounded_pilot_plan_v2_proposal.json) (SHA-256: `cd9086120b6b54a75d73db8482f1364ed7ee2643668c2a6fa44e66282355fb64`)
- Bound catalog extension: [`candidate_catalog_extension_v1.0.0.json`](research/evidence/phase-4c.7b/candidate_catalog_extension_v1.0.0.json) (SHA-256: `0d875b81a8044e3285ecf0331beec6646d94b085915baaaa32247e86ee9023e7`)
- Parent catalog: [`verified_candidate_catalog_v2.json`](research/evidence/phase-4c.7b/verified_candidate_catalog_v2.json) (SHA-256: `d85595c6b43d5acf8d312993a270278b4f17f481dca0f8286efdae07bcd281a5`)
- Proposed attempt budget: exactly 8 attempts (1 attempt per candidate, 0 retries)

> [!IMPORTANT]
> **Quy chuẩn Quản trị & Nghiên cứu Khoa học:**
>
> 1. Việc chuẩn bị Amendment v1.7.0 và Kế hoạch Pilot v2 Proposal **KHÔNG đồng nghĩa được phép generation**. Mọi generation phải chờ phê duyệt chính thức từ người dùng.
> 2. Kế hoạch và Catalog lịch sử (`content_grounded_pilot_plan.json`, `candidate_acquisition_plan_v2.json`, `verified_candidate_catalog_v2.json`) được niêm phong nguyên vẹn; không sửa đè.
> 3. Toàn bộ các quyết định Human Content QC lịch sử (diagnostic 6/6 REJECT, calibration 2 PENDING, pilot 8 PENDING) được bảo toàn tuyệt đối.
> 4. Full cohort ($N=400$) tiếp tục bị khóa nghiêm ngặt; số lượt gọi detector = 0; hiệu năng độc lập = `NOT_MEASURED`.

---

## 1. Bối cảnh & Bằng chứng Dẫn tới Amendment

Quá trình rà soát tiền kiểm nội dung toàn diện (Pre-Generation Screening Dossier) trên hình ảnh authentic $512 \times 512$ và phân tích thực nghiệm đã phát hiện 4 slot bị chặn (BLOCKED) trong phân bổ hiện hành:

1. **Slot 3 (`coco_sdxl`, insertion, small)**: Thực nghiệm ghi nhận 4/4 lần omission qua 3 run độc lập trên cùng candidate `IND_COCO_SDXL_002` (cà chua trên bánh mì). Diagnostic Arm B tạo được đối tượng nhưng bị Human Content QC REJECT chính thức do bước nhảy biên và vân bánh mì. Cần thay thế candidate.
2. **Slot 4 (`coco_sdxl`, insertion, large)**: Rà soát 21 candidate (9 large + 12 medium insertion) trong stratum `coco_sdxl` xác nhận không có candidate nào có $\ge 30\%$ diện tích trống sạch sẽ mà không đè lên con người, xe cộ hoặc cụm vật thể phức tạp. Candidate cũ `IND_COCO_SDXL_041` (đèn chùm) bị kéo mask $512 \times 155$ trên trần phẳng gây bước nhảy tông độ sáng mặt phẳng trần. Cần mở rộng catalog COCO.
3. **Slot 6 (`commons_sd2`, insertion, medium)**: `IND_COMMONS_SD2_002` (vali da cạnh xe đua Bugatti) có bối cảnh xe đua quá mạnh dẫn tới hallucination xe hơi đồ chơi; xe đạp `IND_COMMONS_SD2_040` phiên bản cũ có nan hoa và xích mỏng dễ gây confound hình học. Cần thay thế sang hình khối đặc vững chắc.
4. **Slot 8 (`commons_sdxl`, replacement, large)**: `IND_COMMONS_SDXL_003` (Tower Song) có mask cắt ngang 210 px lan can kim loại liên tục ở hậu cảnh, gây đứt gãy kiến trúc. Ứng viên khảo sát `IND_COMMONS_SDXL_020` (Żyletkowce) có chân tháp vướng đỉnh vòm kính tròn tiền cảnh: mask contour né vòm kính chỉ đạt 25.10% (hụt quota large) và chỉ bao phủ 91.60% target, đồng thời runner hiện tại chỉ hỗ trợ AABB hộp chữ nhật. Cần chuyển sang candidate độc lập trong cùng allocation.

---

## 2. Bảo Toàn Định Mức Phân Bổ Đã Khóa (Preserved Locked Allocation)

Amendment v1.7.0 bảo toàn 100% định mức ma trận phân bổ của Protocol Amendment v1.6.0:

- **Tổng số dòng**: Đúng 8 dòng (2 dòng mỗi stratum).
- **Phân bổ Strata**: `coco_sd2: 2`, `coco_sdxl: 2`, `commons_sd2: 2`, `commons_sdxl: 2`.
- **Phân bổ Thao tác**: `object_replacement: 3` (Slot 1, 2, 8), `object_removal_and_infill: 1` (Slot 5), `object_insertion: 4` (Slot 3, 4, 6, 7).
- **Phân bổ Định mức Diện tích**: `small_under_10pct: 3` (Slot 1, 3, 7), `medium_10_to_30pct: 2` (Slot 5, 6), `large_over_30pct: 3` (Slot 2, 4, 8).

### Bảng Phân Bổ 8 Dòng Chuẩn Tắc Pilot Plan v2 Proposal (PENDING Human Review)

| Slot | Candidate ID | Origin ID & Nguồn Gốc | Prompt Đề Xuất | Target BBox [x1, y1, x2, y2] & Diện Tích | Mask BBox [x1, y1, x2, y2] & Diện Tích | Trạng Thái Review |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| **1** | `IND_COCO_SD2_001` | `coco:397133`<br>Pot Noodle (`CC BY 2.0`) | _"a round brass wall clock mounted naturally on the kitchen wall, matching the warm indoor lighting"_ | `[207, 117, 270, 182]`<br>$63 \times 65 =$ **4.095 px** (1.56%) | `[195, 95, 280, 205]`<br>$85 \times 110 =$ **9.350 px** (3.57%) | **PENDING** |
| **2** | `IND_COCO_SD2_002` | `coco:37777`<br>larrylawfer (`CC BY-NC-SA 2.0`) | _"matte navy-blue upper kitchen cabinets with a stainless-steel range hood, realistic residential interior photograph"_ | `[145, 125, 410, 260]`<br>$265 \times 135 =$ **35.775 px** (13.65%) | `[95, 75, 415, 323]`<br>$320 \times 248 =$ **79.360 px** (30.27%) | **PENDING** |
| **3** | `IND_COCO_SDXL_042` | `coco:448076`<br>luis.leao (`CC BY 2.0`) | _"a professional black leather business briefcase standing upright on the red exhibition carpet, realistic studio floodlights and soft ground shadow"_ | `[235, 395, 335, 485]`<br>$100 \times 90 =$ **9.000 px** (3.43%) | `[220, 380, 350, 500]`<br>$130 \times 120 =$ **15.600 px** (5.95%) | **PENDING** |
| **4** | `COCO_EXT_SDXL_001` | `coco:460160`<br>PratarPersilja (`CC BY-SA 2.0`) | _"a weathered wooden picnic table with attached bench seating on the grassy coastal ground, natural overcast daylight and soft ground contact shadow"_ | `[20, 280, 330, 490]`<br>$310 \times 210 =$ **65.100 px** (24.83%) | `[10, 260, 340, 510]`<br>$330 \times 250 =$ **82.500 px** (31.47%) | **PENDING** |
| **5** | `IND_COMMONS_SD2_001` | `commons:92533678`<br>Moahim (`CC BY-SA 4.0`) | _"open sea and distant coastline continuing naturally through the removed foreground headland, photorealistic sunset landscape"_ | `[190, 308, 512, 512]`<br>$322 \times 204 =$ **65.688 px** (25.06%) | `[190, 305, 512, 512]`<br>$322 \times 207 =$ **66.654 px** (25.43%) | **PENDING** |
| **6** | `IND_COMMONS_SD2_040` | `commons:172876577`<br>Chainwit. (`CC BY 4.0`) | _"a rustic wooden barrel planter filled with vibrant blooming flowers sitting naturally on the cobblestone pavement, realistic daylight shadows and weathered wood texture matching the historic town square"_ | `[45, 375, 215, 495]`<br>$170 \times 120 =$ **20.400 px** (7.78%) | `[20, 360, 240, 512]`<br>$220 \times 152 =$ **33.440 px** (12.76%) | **PENDING** |
| **7** | `IND_COMMONS_SDXL_001` | `commons:166503140`<br>Crisco 1492 (`CC BY-SA 4.0`) | _"a small dark bird flying in the cloudy sky, distant scale and natural daylight"_ | `[395, 75, 455, 125]`<br>$60 \times 50 =$ **3.000 px** (1.14%) | `[350, 45, 500, 160]`<br>$150 \times 115 =$ **17.250 px** (6.58%) | **PENDING** |
| **8** | `IND_COMMONS_SDXL_005` | `commons:171463547`<br>Chris Woodrich (`CC BY-SA 4.0`) | _"a classical white marble commemorative column standing on a solid plinth in the public park, realistic overcast daylight and weathered stone texture"_ | `[145, 45, 355, 465]`<br>$210 \times 420 =$ **88.200 px** (33.65%) | `[135, 40, 365, 475]`<br>$230 \times 435 =$ **100.050 px** (38.17%) | **PENDING** |

---

## 3. Chi Tiết 4 Đề Xuất Thay Thế & Mở Rộng Cụ Thể (PENDING)

### 3.1. Slot 3: Thay thế `IND_COCO_SDXL_002` bằng `IND_COCO_SDXL_042`

- **Candidate thay thế**: `IND_COCO_SDXL_042` (Pool index: 41).
- **Provenance**: Origin `coco:448076` | Flickr photo `2260856815` | Tác giả: `luis.leao` | Giấy phép: `CC BY 2.0`.
- **Disjointness**: PASS (0 overlap với 684 nguồn Option P lịch sử).
- **Target mô tả**: Cặp táp doanh nhân bằng da đen đứng thẳng trên thảm đỏ gian hàng triển lãm hàng không TAM.
- **Prompt đề xuất**: _"a professional black leather business briefcase standing upright on the red exhibition carpet, realistic studio floodlights and soft ground shadow"_.
- **Hình học**: Target `[235, 395, 335, 485]` (9.000 px = 3.43%), Mask `[220, 380, 350, 500]` (15.600 px = 5.950928%, chuẩn `small_under_10pct`).
- **Phân tích hình học & che phủ**: Target bbox nằm 100% trong mask bbox ($220 \le 235 < 335 \le 350$ và $380 \le 395 < 485 \le 500$). Nền thảm đỏ phẳng đồng nhất, không có đồ vật che chắn xung quanh, lề 15-20 px cho phép tạo bóng đổ tiếp xúc chân thực.

### 3.2. Slot 4: Đề xuất mở rộng catalog COCO với `COCO_EXT_SDXL_001`

- **Candidate đề xuất**: `COCO_EXT_SDXL_001`.
- **Provenance**: Origin `coco:460160` | Flickr photo `9345977086` | Tác giả: `PratarPersilja` | Giấy phép: `CC BY-SA 2.0` (xác minh qua Flickr oEmbed).
- **Disjointness**: PASS (0 overlap với 684 nguồn Option P lịch sử).
- **Target mô tả**: Bàn dã ngoại bằng gỗ mộc có ghế băng gắn liền đặt trên bờ cỏ ven biển.
- **Prompt đề xuất**: _"a weathered wooden picnic table with attached bench seating on the grassy coastal ground, natural overcast daylight and soft ground contact shadow"_.
- **Hình học**: Target `[20, 280, 330, 490]` (65.100 px = 24.83%), Mask `[10, 260, 340, 510]` (82.500 px = 31.471252%, chuẩn `large_over_30pct` $\ge 30\%$).
- **Phân tích che phủ thực tế vs BBox containment**:
  - _BBox containment_: 100% target bbox nằm trong mask bbox ($10 \le 20 < 330 \le 340$ và $260 \le 280 < 490 \le 510$).
  - _Bản chất thao tác insertion_: Ảnh authentic là bờ biển tự nhiên, không có đối tượng cũ. Target bbox là hộp bao không gian cho bàn gỗ. Đối tượng vật lý được sinh (mặt bàn, chân bàn, ghế băng, bóng đổ) sẽ chiếm một tập con không đều bên trong target bbox.
  - _Khoảng lề biên_: Lề dưới $y \in [490, 510]$ (20 px) và lề ngang $x \in [10, 20]$ & $[330, 340]$ (10 px) nhằm dự phòng không gian cho chân bàn và bóng tiếp xúc trên mặt cỏ. Tuy nhiên, trong tạo sinh inpainting thực tế, lề biên tiền kiểm không bảo đảm mô hình sẽ tạo bóng đổ trải mềm hoặc triệt tiêu hoàn toàn bước nhảy biên.
  - _Vật thể lân cận_: Vùng bờ biển quang đãng, không có người, xe cộ hay công trình xây dựng. Bề mặt đất bờ cỏ gồ ghề giảm thiểu độ nhạy cảm so với mặt phẳng trần thạch cao đồng nhất của `_041`, song vẫn tiềm ẩn rủi ro sai lệch cấu trúc hoặc tông màu tại đường biên inpainting.

### 3.3. Slot 6: Thay thế `IND_COMMONS_SD2_002` bằng `IND_COMMONS_SD2_040` (Hình khối trụ đặc)

- **Candidate thay thế**: `IND_COMMONS_SD2_040` (Pool index: 39).
- **Provenance**: Origin `commons:172876577` | Nguồn: Wikimedia Commons | Tác giả: `Chainwit.` | Giấy phép: `CC BY 4.0`.
- **Disjointness**: PASS (0 overlap với 684 nguồn Option P lịch sử).
- **Target mô tả**: Chậu hoa thùng gỗ mộc trên mặt đường lát đá cuội quảng trường Sibiu.
- **Prompt đề xuất**: _"a rustic wooden barrel planter filled with vibrant blooming flowers sitting naturally on the cobblestone pavement, realistic daylight shadows and weathered wood texture matching the historic town square"_.
- **Hình học**: Target `[45, 375, 215, 495]` (20.400 px = 7.78%), Mask `[20, 360, 240, 512]` (33.440 px = 12.756348%, chuẩn `medium_10_to_30pct`).
- **Phân tích hình học**: Target nằm 100% trong mask. Thay thế ý tưởng xe đạp bằng chậu hoa thùng gỗ hình trụ đặc vững chắc, loại bỏ hoàn toàn nguy cơ đứt gãy nan hoa và xích xe đạp.

### 3.4. Slot 8: Thay thế `IND_COMMONS_SDXL_003` bằng `IND_COMMONS_SDXL_005`

- **Candidate thay thế**: `IND_COMMONS_SDXL_005` (Pool index: 4).
- **Provenance**: Origin `commons:171463547` | Nguồn: Wikimedia Commons | Tác giả: `Chris Woodrich` | Giấy phép: `CC BY-SA 4.0`.
- **Disjointness**: PASS (0 overlap với 684 nguồn Option P lịch sử).
- **Target mô tả**: Khối điêu khắc đồng đồ sộ "Triptych" tại Windsor Sculpture Park.
- **Prompt đề xuất**: _"a classical white marble commemorative column standing on a solid plinth in the public park, realistic overcast daylight and weathered stone texture"_.
- **Hình học**: Target `[145, 45, 355, 465]` (88.200 px = 33.647461%), Mask AABB `[135, 40, 365, 475]` (100.050 px = 38.166046%, chuẩn `large_over_30pct` $\ge 30\%$).
- **Phân tích che phủ thực tế vs BBox containment**:
  - _BBox containment_: 100% target bbox nằm trong mask bbox ($135 \le 145 < 355 \le 365$ và $40 \le 45 < 465 \le 475$).
  - _Che phủ đối tượng thực tế (Physical Object Coverage)_: Khối điêu khắc đồng "Triptych" trải dài từ $x=145$ đến $x=352$ và đỉnh cột bắt đầu tại $y=45$ đến chân bệ bê tông tại $y=465$ (bao gồm bệ đỡ bê tông chân tượng tại $y=445..465$). Target rectified `[145, 45, 355, 465]` bao trọn 100% cấu trúc vật lý thực tế của tượng và bệ đỡ; toàn bộ thân tượng và bệ đỡ đều nằm trọn 100% trong mask AABB.
  - _Tiếp xúc chân bệ & bóng đổ_: Chân bệ bê tông kết thúc ở $y \approx 465$, thảm cỏ bắt đầu từ $y \approx 470$. Khoảng lề $y \in [465, 475]$ (10 px) dự phòng không gian cho bệ móng và bóng đổ tiếp xúc trên mặt cỏ, song không bảo đảm loại trừ hoàn toàn nguy cơ vết sẹo biên hoặc bậc tương phản vi mô.
  - _Vật thể lân cận_: Không có lan can kim loại (khắc phục lỗi của Tower Song `_003`), không có vòm kính tròn (khắc phục lỗi của Żyletkowce `_020`), không có người đi bộ. Hậu cảnh là bầu trời mây và rặng cây công viên ở xa.
  - _Tính tương thích Runner/Schema_: Sử dụng hộp chữ nhật AABB chuẩn, tương thích 100% với code runner và schema hiện hành mà không cần sửa code.

- **Đính chính thẩm tra hình học offline trên `IND_COMMONS_SDXL_020` (Żyletkowce)**:
  - Mask AABB `[50, 130, 395, 360]`: Có diện tích 79.350 px = 30.269623% canvas (đạt quota large $\ge 30\%$).
  - Phân tích tọa độ chính xác: Mask này **không thiếu cạnh phải** (trục X $x \in [50, 395]$ bao trọn target $x \in [60, 385]$ với lề 10 px mỗi bên). Mask này **chỉ thiếu đúng 5 hàng đáy target** ($y \in [360, 365)$ trên chiều rộng 325 px, tương đương 1.625 px hay 2.27% diện tích target).
  - Giao cắt với vòm kính từ raster: Đỉnh khung kim loại vòm kính đạt $y=358$ tại $x \approx 208..212$. Mask $y \in [130, 360)$ cắt lẹm đúng **2 hàng đỉnh khung vòm kính** ($y=358, 359$, khoảng 14 px raster), không phải 13 px (13 px là độ sâu của mask cũ kéo tới $y=375$).
  - Kết luận: Cả mask AABB lẫn mask contour polygon của Żyletkowce đều không khả thi (AABB cắt vòm kính và thiếu chân tháp; contour hụt quota large xuống 25.10% và runner không hỗ trợ). Việc chuyển sang `IND_COMMONS_SDXL_005` là giải pháp kỹ thuật tối ưu và triệt để.

---

## 4. Ngân Sách & Quy Trình Tạo Sinh Đề Xuất (Proposed Budget & Workflow)

- **Ngân sách thực thi**: Đúng 8 attempts trên 8 candidates trong `content_grounded_pilot_plan_v2_proposal.json`. Mỗi candidate đúng 1 attempt duy nhất, zero retries.
- **Ràng buộc mã thực thi**: Ghim tại commit functional `38df28b4f7fcae0d8788057418410c27d1ca5852` (nạp catalog extension, kiểm chứng proposal v2 preflight, và bảo toàn toàn bộ pin/audit lịch sử).
- **Ràng buộc Colab**: Chỉ thực thi trên môi trường GPU Colab sau khi người dùng phê duyệt; không chạy generation cục bộ.
- **Trạng thái hiện tại**: **PENDING_HUMAN_REVIEW**.
