# Dataset Specification & Management: Forensics Web Lab

## 1. Overview and Core Data Policies

To support rigorous, scientifically reproducible model development and evaluation, Forensics Web Lab establishes strict protocols governing digital media ingestion, annotation, deduplication, and licensing.

### Fundamental Rules
1. **No Large Datasets in Git**: Training data, uncompressed benchmarks, raw images, and multi-gigabyte archives must NEVER be committed to the Git repository.
2. **Pre-Download Audit & Human Authorization**: Prior to downloading any external dataset:
   * The official academic or institutional source is verified.
   * The legal license terms are documented in [docs/DATA_LICENSES.md](file:///d:/Documents/forensics-web-lab/docs/DATA_LICENSES.md).
   * Exact archive download size and uncompressed storage requirements are calculated.
   * Cryptographic checksums (SHA-256 / MD5) are recorded.
   * **Explicit human confirmation is obtained before initiating large downloads.**
3. **Canonical Label Mapping**:
   * Internal dataset labels: `authentic`, `fully_generated`, `ai_edited`.
   * User interface presentation: `authentic` is strictly mapped to `no_ai_evidence` ("Chưa tìm thấy bằng chứng AI"). The term "Real image" is never presented as an absolute fact.

---

## 2. Supported Dataset Catalog

### 2.1 GenImage
* **Purpose**: Multi-generator benchmark for whole-image synthetic generation vs. authentic imagery.
* **Covered Generators**: Stable Diffusion (v1.4, v1.5), Midjourney, DALL-E 2/3, GLIDE, VQDM, BigGAN, ADM, Wukong.
* **Authentic Source**: Curated ImageNet validation subset.
* **License**: Research / Academic Use Only (refer to [docs/DATA_LICENSES.md](file:///d:/Documents/forensics-web-lab/docs/DATA_LICENSES.md)).
* **Estimated Archive Size**: ~30 GB - 120 GB (depending on generator subsets).

### 2.2 SAGI-D (Synthetic and AI-Generated Inpainting Dataset)
* **Purpose**: Localized generative inpainting detection and pixel-level boundary localization.
* **Content**: Paired authentic imagery, generative inpaint modifications, and ground-truth binary masks (`mask_path`).
* **Mask Availability**: Full pixel-level binary masks ($0 = \text{authentic}$, $255 = \text{inpainted}$).
* **Estimated Archive Size**: ~10 GB - 25 GB.

### 2.3 RealHD
* **Purpose**: High-definition realistic localized manipulation detection (inpainting, face refinement, object insertion/removal).
* **Resolution**: High-resolution ($1024 \times 1024$ and higher).
* **Mask Availability**: High-resolution ground truth masks included.
* **Estimated Archive Size**: ~15 GB.

### 2.4 RAID (Robust AI Image Detection Benchmark)
* **Purpose**: Evaluation of model robustness against real-world social media degradations, compression, and adversarial perturbations.
* **Use in Pipeline**: Reserved strictly for testing (unseen holdout) and robustness profiling; never used for training.
* **Estimated Archive Size**: ~20 GB.

### 2.5 COCO-Inpaint
* **Purpose**: Controlled synthetic inpainting on Microsoft COCO validation imagery.
* **Usage Condition**: Adopted only upon verification of official release and CC-BY 4.0 / academic license compliance.

### 2.6 Traditional Editing Hard Negatives
* **Purpose**: Non-AI manipulation benchmarks (Photoshop splicing, copy-move, retouching, color grading).
* **Forensic Role**: Essential hard negatives to verify that the detector does not falsely flag traditional manual photo editing as generative AI.

---

## 3. Standardized Manifest Schema

All datasets are normalized into a unified, machine-readable JSON/CSV manifest format with the following fields:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `sample_id` | `string` | Globally unique identifier (e.g. `genimage_sd15_000123`) |
| `source_id` | `string` | Unique identifier of the original parent image (used for group-based split isolation) |
| `image_path` | `string` | Relative path to image file within the local data root |
| `label` | `string` | One of: `authentic`, `fully_generated`, `ai_edited` |
| `generator` | `string` | Model family (e.g. `stable-diffusion`, `midjourney`, `glide`, `camera`) |
| `generator_version`| `string` | Version identifier (e.g. `1.5`, `v5.2`, `dalle-3`, `none`) |
| `edit_type` | `string` | Nature of manipulation: `none`, `full_synthesis`, `inpainting`, `face_swap`, `object_removal` |
| `mask_path` | `string` | Relative path to binary mask (or empty string if non-localized) |
| `dataset_name` | `string` | Name of origin dataset (e.g. `GenImage`, `SAGI-D`, `RealHD`) |
| `dataset_version` | `string` | Dataset version string |
| `split` | `string` | Split assignment: `train`, `val`, `test_indomain`, `test_unseen_generator`, `test_degraded` |
| `license` | `string` | SPDX license identifier or URL |
| `width` | `integer` | Pixel width |
| `height` | `integer` | Pixel height |
| `sha256` | `string` | 64-character lowercase hexadecimal SHA-256 hash |

---

## 4. Group-Based Splitting & Anti-Leakage Protocol

1. **Group Split by `source_id`**:
   * Under no circumstances may an authentic source image and its generated/inpainted derivatives reside in different splits.
   * All variants sharing a common `source_id` are placed as an atomic unit into either `train`, `val`, or `test`.
2. **Unseen Generator Isolation**:
   * At least one major generator family (e.g. Midjourney or SDXL) is completely quarantined from the training set and held out for the `test_unseen_generator` split.
3. **Deduplication Validation**:
   * Prior to finalizing splits, all images are indexed using perceptual hashing (pHash) with Hamming distance threshold $d \le 3$ and exact SHA-256 matching. All duplicate groups are identified and resolved.
