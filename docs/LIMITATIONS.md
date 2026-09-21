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
