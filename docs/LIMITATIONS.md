# Limitations & Disclaimers: Forensics Web Lab

## 1. Epistemological Limitations of Image Forensics

### 1.1 "No AI Evidence" vs. "Authentic Photo"
Digital image forensics is inherently asymmetrical:
* Finding reliable synthetic artifacts (spectral spikes, unnatural blending boundaries, inconsistent noise residuals) provides evidence that an image was generated or manipulated by an AI system.
* Failing to detect these artifacts merely demonstrates that **no known AI signatures were identified within the operational capabilities of the detector**.
* Consequently, **Forensics Web Lab never labels an image as "Real" or "Authentic"**. The label is strictly defined as:
  $$\mathbf{no\_ai\_evidence} \quad (\text{"Chưa tìm thấy bằng chứng AI"})$$
* This tool is an **investigative aid**, not an infallible arbiter of historical truth.

---

## 2. Inherent Physical & Algorithmic Limitations

### 2.1 Information Loss Through Social Media Pipeline
Platforms such as Facebook, WhatsApp, X (Twitter), and Instagram re-encode images through lossy pipelines (downscaling, high JPEG compression $Q < 60$, chroma subsampling $4:2:0$). These operations act as spatial low-pass filters that frequently erase subtle high-frequency generative artifacts. When an image is heavily degraded, the detector will appropriately reflect reduced confidence or return `uncertain`.

### 2.2 Unseen Generative Families
Generative AI technologies evolve continuously. Models trained on Latent Diffusion or GAN architectures may exhibit reduced detection sensitivity against novel generation paradigms (e.g. discrete autoregressive image tokenizers, advanced flow-matching architectures) until training distributions are updated.

### 2.3 Traditional Editing & Complex Photography (Hard Negatives)
Certain organic photographic techniques can trigger localized signal discrepancies:
* Multi-exposure HDR blending.
* Focus stacking and composite astrophotography.
* Heavy digital noise reduction (DNR) or smartphone computational night modes.
* Traditional manual retouching (Photoshop clone stamp, frequency separation).
While the multi-modal evidence fusion engine penalizes conflicting signals to avoid false positives, complex composites may still occasionally register elevated suspicion or yield an `uncertain` verdict.

### 2.4 Independent Evaluation & Generalization Boundaries (TGIF N=400 Cohort)
* **Unseen Sources vs. Out-of-Distribution (OOD)**: The independent evaluation on the TGIF N=400 cohort tests detector performance on unseen source images within the same benchmark family (MS-COCO background sources from TGIF, \cite{mareen2024tgif}) and the same inpainting generator (Stable Diffusion 2). This establishes source-level generalization in-distribution, but **does not constitute empirical proof of out-of-distribution (OOD) generalization** across unseen generative architectures (such as Adobe Firefly, SDXL, Midjourney, or autoregressive/flow-matching models \cite{ojha2023universal,wang2023dire}).
* **Sample Size Allocation & Large Stratum Imbalance ($n=14$)**: The strata distribution (14 Large [3.5%], 221 Medium [55.25%], 165 Small [41.25%]) reflects benchmark segmentation constraints rather than natural manipulation prevalence. The Large stratum ($n=14$) is strictly descriptive; no separate statistical claims are made for this stratum.
* **Two-Class Scope**: The independent cohort is restricted to binary classification (`authentic` vs `ai_edited`) based on localized inpainting; it does not evaluate full-image generation (`fully_generated`, \cite{zhu2023genimage}).
* **Inconclusive Primary Endpoint at JPEG Q=75**: On unseen independent sources at `jpeg_q75`, Late Fusion DSP Augmented yields $\Delta \text{Macro-F1} = -0.0027$ with a 95% bootstrap CI $[-0.0126, +0.0072]$ spanning zero ($P(\Delta^* > 0) = 30.29\%$). Improvement is **unproven** on independent data, contrasting with exploratory development gains ($\Delta = +0.1363$).

---

## 3. Scope Boundaries (Unsupported Media)

Forensics Web Lab is engineered specifically for raster photographs in **JPEG/JPG, PNG, and WebP** formats.

### Strictly Unsupported Media:
* **Video Streams & Video Files** (MP4, AVI, WebM).
* **Audio Recordings & Voice Clones**.
* **Documents & Office Formats** (PDF, Word, Excel, PowerPoint).
* **Vector Graphics** (SVG, EPS, AI).
* **Deepfake Video & Facial Reenactment**.
* **AI-Generated Text Detection**.
