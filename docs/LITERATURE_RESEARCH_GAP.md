# Khảo Sát Tài Liệu và Xác Minh Khoảng Trống Nghiên Cứu (Literature & Research Gap Analysis)

> **Dự án**: `Forensics Web Lab`
> **Phạm vi**: RQ1 (Phân loại ba lớp authentic / ai_edited / fully_generated), RQ2 (Visual-only vs Visual + DSP), RQ3 (Lượng tử hóa ONNX/INT8), và RQ4 (Browser CPU/WASM Runtime)
> **Tài liệu tham chiếu chuẩn tắc**: [`docs/references.bib`](references.bib), [`docs/EVIDENCE_REGISTER.md`](EVIDENCE_REGISTER.md), [`docs/MANUSCRIPT.md`](MANUSCRIPT.md), [`docs/PILOT_PROTOCOL.md`](PILOT_PROTOCOL.md)
> **Nguyên tắc học thuật**: Tuân thủ tuyệt đối chuẩn mực trung thực khoa học (*Scientific Honesty*); không tuyên bố "đầu tiên", "chưa ai làm" hay tính mới tuyệt đối chỉ vì thêm DSP, ba lớp, hoặc chạy browser. Mọi nhận định đều được đối chiếu với các nguồn sơ cấp đã công bố.

---

## 1. Mục Tiêu và Phương Pháp Khảo Sát

Khảo sát này nhằm mục đích:
1. Đối chiếu một cách có hệ thống các công trình nghiên cứu đã xuất bản liên quan đến bài toán phát hiện ảnh AI (đặc biệt là chỉnh sửa cục bộ / text-guided inpainting), các kiến trúc gọn nhẹ, kỹ thuật kết hợp miền không gian - phổ (spatial-frequency DSP), và môi trường thực thi client-side / edge.
2. Phân định rõ ràng giữa các bài toán đã được cộng đồng nghiên cứu kỹ lưỡng và các khía cạnh kỹ thuật chưa thấy được khảo sát đầy đủ trong phạm vi rà soát tài liệu hiện hành.
3. Thiết lập cơ sở lý thuyết và các giả thuyết đo lường được cho bước nghiên cứu tiếp theo (RQ3 lượng tử hóa INT8 và RQ4 benchmark trình duyệt web client-side).

---

## 2. Bảng Đối Chiếu Các Công Trình Tiêu Biểu Trong Y Văn

Bảng dưới đây tổng hợp các công trình then chốt được trích dẫn và đối chiếu trong dự án, bao gồm:
- **LAID** \cite{chivaran2025laid}
- **TGIF / TGIF2** \cite{mareen2024tgif,mareen2026tgif2}
- **INP-X** (Nebioglu, Bilgiç, & Popescu, 2026; arXiv:2602.00192)
- **TruFor** \cite{guillaro2023trufor}
- **DIRE** \cite{wang2023dire}
- **CNN-Detection** \cite{wang2020cnndetection}
- **Universal Fake Detectors** \cite{ojha2023universal}
- **GenImage** \cite{zhu2023genimage}
- **SAGI** \cite{giakoumoglou2025sagi}
- **So-Fake** (arXiv:2505.18660; hzlsaber et al., 2025)
- **ONNX Runtime Web** \cite{onnxRuntimeWebDocs}

| Công Trình (Work) | Tập Dữ Liệu (Dataset) | Loại Can Thiệp (Manipulation) | Kiến Trúc Mô Hình (Model) | Độ Chính Xác Số (Precision) | Môi Trường Thực Thi (Runtime) | Biến Đổi Kiểm Tra (Transforms) | Chỉ Số Đánh Giá (Metrics) | Phạm Vi Đã Đánh Giá (Evaluated Scope) |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- | :--- | :--- |
| **So-Fake** (hzlsaber et al., 2025; arXiv:2505.18660) | So-Fake-Set (~2M ảnh), So-Fake-OOD (~100k ảnh) | 3 phân lớp: REAL, FULL_SYNTHETIC, TAMPERED (locally edited / inpainting) | So-Fake-R1: Vision-Language Model (VLM) lớn + RL Reasoning | FP16/BF16/FP32 | Cụm máy chủ GPU quy mô lớn | Nén mạng xã hội, làm mờ, biến đổi thực tế OOD | F1-score, Localization IoU, Rationale Explainability | **Đã thiết lập bài toán 3 lớp trên VLM lớn.** Đòi hỏi tài nguyên máy chủ GPU khổng lồ; **chưa khảo sát mô hình tích chập gọn nhẹ (<5M params), không có giải pháp chạy client-side WASM, và không khảo sát DSP tần số nhẹ.** |
| **LAID** (Chivaran & Ni, 2025) \cite{chivaran2025laid} | GenImage (8 generators) | Tạo sinh toàn phần (Fully-generated) | MobileNetV2, EfficientNet-B0, ShuffleNetV2 + Spatial/Spectral | FP32 | Python / PyTorch (GPU & CPU máy chủ) | Gaussian blur, JPEG nén (Q=70, 80, 90) | Accuracy, Average Precision (AP) | Phân loại ảnh sinh toàn phần trên mô hình nhẹ. **Chưa đánh giá inpainting cục bộ; chưa xuất ONNX/INT8; không đo trên trình duyệt client-side.** |
| **TGIF / TGIF2** (Mareen et al., 2024, 2026) \cite{mareen2024tgif,mareen2026tgif2} | TGIF (75k ảnh MS-COCO), TGIF2 (271k ảnh) | Vẽ bù cục bộ có điều khiển (Text-guided Inpainting: SD2, SDXL, Kandinsky) | ResNet-50, CLIP ViT-B/32, DIRE, TruFor, F3-Net | FP32 | Python / PyTorch (Cụm máy chủ GPU cao cấp) | Ảnh sạch, nén JPEG, nén mạng xã hội (Twitter, FB) | Balanced Acc, AUROC, Localization mIoU / Dice | Benchmark chuẩn cho inpainting. **Tập trung vào mô hình tham số lớn trên máy chủ; không khảo sát CNN nhẹ (< 5M params), không lượng tử hóa INT8, không đo WASM.** |
| **INP-X** (Nebioglu, Bilgiç, & Popescu, 2026; arXiv:2602.00192) | Inpainting exchange benchmarks | Khảo sát sự phụ thuộc dấu vết toàn ảnh trong inpainting | Deep CNN architectures | FP32 | Python / PyTorch (GPU máy chủ) | JPEG compression, filtering | Detection Accuracy, F1-score | Khảo sát các giả định về dấu vết inpainting toàn ảnh. **Không thiết kế mô hình nhẹ client-side; không đánh giá lượng tử hóa INT8 hay runtime trình duyệt.** |
| **TruFor** (Guillaro et al., 2023) \cite{guillaro2023trufor} | DSO-1, Coverage, CoMoFoD, VIPP | Splicing, copy-move, inpainting | Transformer backbone + Noiseprint++ CNN (~70M+ params) | FP32 | Python / PyTorch (GPU máy chủ, đòi hỏi VRAM lớn) | JPEG compression, resizing, social media | F1, AUROC, Localization IoU | Phát hiện và định vị giả mạo đa thành phần rất mạnh trên máy chủ. **Kích thước và chi phí tính toán vượt quá khả năng thực thi CPU/WASM client-side.** |
| **DIRE** (Wang et al., 2023) \cite{wang2023dire} | ImageNet, LSUN, Diffusion subsets | Diffusion generation & editing | ResNet-50 phân loại ảnh sai số tái cấu trúc khuếch tán | FP32 | Python / PyTorch (GPU máy chủ, cần 20-50 DDIM inversion steps) | Blur, JPEG, crop | Accuracy, AP | Khái quát hóa tốt trên diffusion models. **Cực kỳ tốn kém thời gian tính toán (vài giây/ảnh trên GPU); hoàn toàn không khả thi trên client-side.** |
| **CNN Detection** (Wang et al., 2020) \cite{wang2020cnndetection} | ProGAN, StyleGAN, BigGAN | GAN generation | ResNet-50 | FP32 | PyTorch (GPU máy chủ) | Gaussian blur, JPEG nén | Average Precision, Accuracy | Phát hiện dấu vết phổ của khối upsampling trong GAN. **Kém nhạy với Diffusion inpainting hiện đại; mô hình tham số lớn.** |
| **Universal Fake Detector** (Ojha et al., 2023) \cite{ojha2023universal} | LDM, Glide, DALL-E, GANs | Fully-generated & diffusion synthesis | Frozen CLIP-ViT feature space + Linear Probe / k-NN | FP32 | PyTorch (GPU máy chủ) | Gaussian blur, JPEG nén | Accuracy, AP | Khái quát hóa cao qua nhiều generator. **CLIP-ViT có kích thước hàng trăm MB, không tối ưu cho mô hình lightweight WASM client-side.** |
| **GenImage** (Zhu et al., 2023) \cite{zhu2023genimage} | 1.3 triệu ảnh từ 8 generators | Fully-generated | ResNet-50, DeiT, Swin-T | FP32 | PyTorch (Hạ tầng cụm máy chủ phân tán) | Gaussian blur, JPEG nén | Accuracy | Benchmark quy mô lớn cho ảnh sinh toàn phần. **Không chứa mặt nạ hay ảnh chỉnh sửa cục bộ text-guided inpainting.** |
| **ONNX Runtime Web** (Microsoft, 2026) \cite{onnxRuntimeWebDocs} | Không gắn với dataset cụ thể | Tài liệu runtime kỹ thuật | Các mô hình ONNX tổng quát (thị giác, ngôn ngữ) | FP32, FP16, INT8 | Trình duyệt Web (WebAssembly SIMD, WebGPU, WebGL) | Không áp dụng | Độ trễ suy luận (ms), mức chiếm dụng bộ nhớ (MB) | Nền tảng hạ tầng thực thi client-side. **Tài liệu xác nhận năng lực kỹ thuật của runtime; chưa có benchmark chuyên sâu cho bài toán inpainting forensic kết hợp DSP.** |
| **Forensics Web Lab** *(Công trình này)* | Option P (341 nguồn phát triển), TGIF N=400 (kiểm định độc lập) | Text-guided inpainting (Stable Diffusion 2) | MobileNetV3-small backbone (2.5M params) + 16 DSP + Stacker | **FP32** *(hiện tại)*; **INT8** *(kế hoạch RQ3)* | **LOCAL CPU** (đã đo); **WASM Web Worker** *(kế hoạch RQ4)* | 6 canonical transforms: original, jpeg_q95, jpeg_q75, jpeg_q50, resize_0.5, resize+jpeg | Macro-F1, BAcc, AUROC, Brier, ECE, Parity Discrepancies | **Đã hoàn thành kiểm định độc lập TGIF N=400 (FP32) và đối chứng parity PyTorch vs ONNX FP32 trên CPU. Đang chuẩn bị RQ3 (INT8) và RQ4 (Browser runtime).** |

---

## 3. Phân Loại Trạng Thái Nghiên Cứu và Nhận Diện Khoảng Trống (Research Gap Identification)

Dựa trên bảng khảo sát y văn và thực tiễn triển khai của dự án, các chủ đề nghiên cứu được phân loại theo ba mức độ nghiêm ngặt:

### 3.1. Các Khía Cạnh Đã Có Nghiên Cứu Sâu (Well-Studied in Literature)
- **Dấu vết tạo sinh toàn phần (Fully-generated detection)**: Được nghiên cứu rất phong phú trên GAN và Diffusion qua các tập dữ liệu như GenImage \cite{zhu2023genimage}, CNN-Detection \cite{wang2020cnndetection}, Universal Detectors \cite{ojha2023universal}, LAID \cite{chivaran2025laid}.
- **Benchmark inpainting trên mô hình máy chủ (Server-side Inpainting Benchmarks)**: TGIF \cite{mareen2024tgif} và TGIF2 \cite{mareen2026tgif2} đã thiết lập hệ quy chiếu chi tiết cho inpainting, nhưng phần lớn sử dụng các mô hình nặng (ResNet-50, CLIP, DIRE, TruFor) chạy trên GPU máy chủ.
- **Bài toán phân loại ba lớp (Real / Fully Synthetic / Tampered)**: Đã được chính thức hóa trong các công trình gần đây như So-Fake (arXiv:2505.18660) sử dụng mô hình thị giác-ngôn ngữ lớn (VLM). **Do đó, bản thân bài toán ba lớp tự thân KHÔNG PHẢI là tính mới học thuật.**
- **Kỹ thuật hiệu chuẩn hậu kỳ (Post-hoc Calibration)**: Temperature Scaling \cite{guo2017calibration} đã được chứng minh hiệu quả trên các tác vụ phân loại ảnh thông thường.
- **Hạ tầng ONNX Runtime Web**: Microsoft đã cung cấp runtime hoàn chỉnh hỗ trợ WebAssembly SIMD và WebGPU trong trình duyệt web \cite{onnxRuntimeWebDocs}.

### 3.2. Các Khía Cạnh Chưa Thấy Trong Phạm Vi Rà Soát (Not Observed in Reviewed Scope)
- **Phân loại đồng thời ba lớp trên kiến trúc tích chập gọn nhẹ (<5M tham số) kết hợp DSP client-side**:
  Trong khi các công trình như So-Fake giải quyết bài toán 3 lớp bằng VLM quy mô lớn trên cụm máy chủ GPU, chưa thấy công trình nào trong phạm vi rà soát khảo sát một không gian quyết định 3 lớp thống nhất trên cùng một backbone CNN siêu nhẹ (như MobileNetV3-small 576-d) kết hợp đặc trưng DSP 16-D để thực thi client-side trong trình duyệt web mà không cần gửi dữ liệu ảnh lên máy chủ (Zero-Egress).
- **Đóng góp của đặc trưng DSP canonical 16-D trong không gian ba lớp (RQ2)**:
  Các đặc trưng phổ tần số (DCT radial/azimuthal) thường được đề xuất để nhận diện dấu vết upsampling toàn ảnh. Tuy nhiên, hiệu quả thực sự của DSP khi đối mặt đồng thời với ảnh inpainting (vùng nhân tạo nhỏ, nền tự nhiên lấn át) và ảnh fully-generated (100% pixel nhân tạo) trên cùng một phân phối kiểm định chưa từng được đo lường cô lập và có đối chứng trên cùng cohort/budget. **Lưu ý nguyên tắc trung thực khoa học: Vector đặc trưng DSP do dự án chọn chỉ là một ứng viên khoảng trống cần xác minh thực nghiệm (candidate hypothesis), không tự coi là bằng chứng tính mới.**
- **Đánh giá lượng tử hóa INT8 trên bài toán pháp chứng số ảnh chỉnh sửa cục bộ (RQ3)**:
  Trong các bài toán thị giác máy tính truyền thống (ImageNet classification, object detection), việc suy giảm độ chính xác khi lượng tử hóa INT8 thường được coi là nhỏ trong nhiều báo cáo công nghiệp (*chưa có nguồn trực tiếp kiểm chứng trong bối cảnh forensics: unverified*). Trong lĩnh vực pháp chứng số ảnh inpainting, các dấu vết vi mô ở biên vẽ bù và phương sai nhiễu dư có đặc tính tín hiệu yếu. Hiện chưa thấy công trình nào đo đạc thực nghiệm có kiểm soát xem việc lượng tử hóa tĩnh (PTQ) hay động các tầng tích chập của mạng nhẹ có làm suy thoái các đặc trưng pháp chứng này hay không.
- **Hệ thống Zero-Egress Client-Side Browser hoàn chỉnh cho Inpainting (RQ4)**:
  Hầu hết các giải pháp phát hiện inpainting hiện nay bắt buộc người dùng tải ảnh lên máy chủ GPU. Rất hiếm công trình học thuật nào thiết kế, đóng gói và đo đạc thực nghiệm toàn diện một quy trình phát hiện inpainting chạy 100% cục bộ trên trình duyệt (Zero-Egress qua WASM Web Worker) với thời gian phản hồi ở mức tương tác người dùng (< 500 ms).
- **Tính bền vững của mô hình kết hợp DSP nhẹ qua các nguồn ảnh độc lập**:
  Như kết quả kiểm định độc lập của dự án trên TGIF N=400 đã chứng minh (`INDEPENDENT_JPEG75_INCONCLUSIVE`), các đặc trưng DSP toàn cục có thể biểu hiện mức cải thiện quan sát trên tập phát triển nhưng bị phân rã khi gặp nguồn ảnh mới độc lập. Khía cạnh này hiện được ghi nhận như một giả thuyết nghiên cứu chưa được kiểm chứng cô lập trong các công trình đề xuất đặc trưng tần số trước đây.

### 3.3. Ứng Viên Khoảng Trống Cần Xác Minh (Candidate Gaps to Verify)
Chúng tôi định vị 5 câu hỏi kỹ thuật cụ thể là các ứng viên khoảng trống cần giải quyết trong các bước tiếp theo của đề tài:

1. **Ứng viên Khoảng trống 1 (Không gian Phân loại Ba Lớp trên Mạng Nhẹ - RQ1)**:
   *Câu hỏi*: Mạng tích chập gọn nhẹ (MobileNetV4/V3) có khả năng phân biệt đồng thời 3 lớp `authentic`, `ai_edited`, và `fully_generated` với Macro-F1 $> 0.65$ và Balanced Accuracy ổn định hay không, hay sẽ bị thiên vị (bias) gộp `ai_edited` vào `authentic` do phần lớn diện tích ảnh inpainting vẫn là nội dung thật?
2. **Ứng viên Khoảng trống 2 (Đóng góp của Canonical DSP 16-D trong Không gian Ba Lớp - RQ2)**:
   *Câu hỏi*: Việc kết hợp vector đặc trưng DSP 16-D (tần số DCT + thống kê nhiễu dư) với visual backbone nhẹ có mang lại mức tăng Macro-F1 và cải thiện độ tin cậy hiệu chuẩn (Multiclass Brier Score, ECE) có ý nghĩa thống kê dưới các phép biến đổi nén JPEG (Q=90, 75, 50) và làm mờ hay không, so với visual-only baseline trên cùng một split và ngân sách đánh giá?
3. **Ứng viên Khoảng trống 3 (Trade-off Lượng tử hóa INT8 - RQ3)**:
   *Câu hỏi*: Việc lượng tử hóa tĩnh INT8 mô hình MobileNetV3-small backbone ảnh hưởng như thế nào đến độ chính xác phân loại (Macro-F1) dưới các điều kiện nén JPEG, và mức độ tiết kiệm dung lượng (MB) cùng tốc độ suy luận (ms) trên CPU là bao nhiêu?
4. **Ứng viên Khoảng trống 4 (Độ trễ và Tính khả thi Runtime Trình duyệt - RQ4)**:
   *Câu hỏi*: Một pipeline kết hợp gồm visual backbone (thực thi qua ONNX Runtime Web WASM SIMD trong Web Worker) và trích xuất DSP (thực thi ngoài graph bằng JavaScript/WASM) có đạt được độ trễ P95 $< 500$ ms và đỉnh bộ nhớ RAM $< 200$ MB trên các trình duyệt hiện đại hay không?
5. **Ứng viên Khoảng trống 5 (Sự suy giảm tín hiệu DSP khi lượng tử hóa)**:
   *Câu hỏi*: Do DSP trích xuất độc lập ngoài graph ONNX và chỉ được kết hợp ở tầng logit qua stacker tuyến tính, việc lượng tử hóa INT8 nhánh thị giác có làm thay đổi điểm cân bằng đóng góp giữa nhánh thị giác và nhánh DSP trong mô hình Late Fusion hay không?

---

## 4. Kế Hoạch Chuẩn Bị Cho Bước INT8 và Browser Benchmark

### 4.1. Nguyên Tắc Bảo Toàn Dữ Liệu và Tính Độc Lập
1. **Dữ liệu hiệu chuẩn lượng tử hóa (Calibration Dataset)**:
   - Toàn bộ dữ liệu dùng để chạy calibration cho lượng tử hóa INT8 (Post-Training Quantization - PTQ) **bắt buộc phải lấy từ tập phát triển** (Option P `development_train` gồm 250 nguồn, hoặc các fixture tổng hợp).
   - **Tuyệt đối không sử dụng tập kiểm định độc lập TGIF N=400** để chạy calibration, lựa chọn cấu hình lượng tử hóa, hay tinh chỉnh ngưỡng.
2. **Bảo toàn dữ liệu lịch sử**:
   - Toàn bộ artifacts, receipts, predictions và kết quả của Phase 4C.6B và 4C.7B được niêm phong bất biến.
   - Các thực nghiệm INT8 và browser benchmark sẽ có tệp cấu hình (`ml/configs/...`), mã nguồn (`scripts/research/...`), và biên nhận kết quả (`research/evidence/phase-4c.8/...`) hoàn toàn riêng biệt.

### 4.2. Khung Thực Nghiệm RQ3 (ONNX INT8 Quantization Benchmark)
- **Mục tiêu**: Đánh giá sự đánh đổi giữa độ chính xác phân loại, dung lượng lưu trữ và thời gian tính toán của mô hình INT8 so với mô hình tham chiếu FP32 vừa hoàn thành parity.
- **Biến kiểm soát**:
  - Trọng số backbone MobileNetV3-small cố định (`047dcff4...`).
  - 5 mô hình outer-fold checkpoints cố định.
  - Ngưỡng quyết định $\tau = 0.5$ cố định.
  - Đánh giá trên cùng panel kiểm tra chuẩn của tập phát triển.
- **Biến độc lập**: Chế độ lượng tử hóa (FP32 baseline vs Dynamic INT8 vs Static INT8 với calibration).
- **Chỉ số đo lường (Metrics)**:
  - Dung lượng file mô hình `.onnx` (MB).
  - Độ trễ suy luận trung bình và P95 trên CPU (ms/ảnh).
  - Độ sụt giảm Macro-F1 ($\Delta \text{Macro-F1} = \text{Macro-F1}_{\text{INT8}} - \text{Macro-F1}_{\text{FP32}}$).
  - Tỷ lệ sai lệch phân loại (Decision Flip Rate: % mẫu bị đổi nhãn sau lượng tử hóa).
- **Tiêu chuẩn Thành công / Thất bại (Success/Failure Criteria)**:
  - *Thành công (PASS)*: Dung lượng mô hình giảm $\ge 50\%$ (từ ~3.7 MB xuống $< 2.0$ MB), độ sụt giảm Macro-F1 $\le 0.02$, và tốc độ suy luận tăng $\ge 1.2\times$ trên CPU.
  - *Thất bại (FAIL)*: Độ sụt giảm Macro-F1 $> 0.05$ hoặc xuất hiện lỗi số học tràn số (overflow/underflow) trong quá trình lượng tử hóa.

### 4.3. Khung Thực Nghiệm RQ4 (Browser CPU/WASM Runtime Benchmark)
- **Mục tiêu**: Đo lường thực tế độ trễ và mức tiêu hao tài nguyên bộ nhớ khi chạy mô hình trên trình duyệt web người dùng thông qua ONNX Runtime Web.
- **Môi trường đo đạc**:
  - Trình duyệt: Google Chrome (Chromium V8) và Mozilla Firefox (SpiderMonkey).
  - Hạ tầng thực thi: Web Worker client-side, luồng riêng biệt chống đóng băng UI.
  - Runtime backend: ONNX Runtime Web v1.30.0 với backend WASM SIMD.
- **Chỉ số đo lường (Metrics)**:
  - Thời gian khởi tạo / nạp mô hình lạnh (Cold Start Load Time in ms).
  - Độ trễ suy luận mỗi ảnh P50, P90, P95 (Inference Latency in ms).
  - Đỉnh mức chiếm dụng bộ nhớ JS Heap và WASM Linear Memory (Peak RAM in MB).
  - Tỷ lệ suy luận thành công liên tục không rò rỉ bộ nhớ (100 ảnh liên tiếp).
- **Tiêu chuẩn Đạt yêu cầu**:
  - P95 latency $< 500$ ms trên máy tính cá nhân phổ thông.
  - Bộ nhớ WASM ổn định, không ghi nhận rò rỉ bộ nhớ (memory leak $\le 5$ MB sau 100 lượt suy luận).
