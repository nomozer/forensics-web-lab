# Threat Model: Forensics Web Lab

## 1. System Threat Landscape

As a client-side digital forensics application operating directly in end-user web browsers, **Forensics Web Lab** faces specific adversary models:

| Threat Actor | Motivation | Attack Vectors |
| :--- | :--- | :--- |
| **Malicious Media Distributor** | Evasion or false attribution | Adversarial perturbations, metadata spoofing, double compression |
| **Exploitative Attacker** | Client device compromise or XSS | Polyglot files, parser exploits, decompression bombs, metadata injection |
| **Malicious Model Tamperer** | Subverting forensic integrity | Checksum spoofing, MITM model corruption |

---

## 2. Threat Analysis Matrix

### 2.1 Threat: MIME Type Spoofing & Polyglot Payloads
* **Description**: Attacker renames an executable, HTML file, or script payload with an image extension (`payload.jpg.exe` or `exploit.png`).
* **Risk**: High (potential code execution or parser crash if passed to naive decoders).
* **Mitigation**: Pure binary magic-byte inspection on the first 16 bytes of the raw `ArrayBuffer`. Rejection of any file failing strict binary header criteria.

### 2.2 Threat: Image Decompression Bomb (Pixel Flooding)
* **Description**: A highly compressed image (e.g. 100 KB on disk) expanding to gigabytes in memory (e.g. $50\,000 \times 50\,000$ pixels), causing browser tab memory exhaustion and crash.
* **Risk**: High (Denial of Service).
* **Mitigation**: Parse image dimensions from header chunks prior to full raster allocation. Enforce strict cap: $W \times H \le 64\,000\,000$ pixels.

### 2.3 Threat: Parser Exploits & Malformed EXIF Segments
* **Description**: Crafting corrupted TIFF/EXIF pointers, cyclic tag structures, or integer overflow triggers inside EXIF/XMP chunks.
* **Risk**: Medium (Memory corruption in native decoders, unhandled exceptions).
* **Mitigation**: Pure TypeScript parser operating on bounded typed buffers with strict bounds checking. All parser calls wrapped in defensive `try/catch` handlers returning fallback empty metadata rather than crashing.

### 2.4 Threat: Stored Cross-Site Scripting (XSS) via Metadata
* **Description**: Embedding script tags (`<script>alert(1)</script>`, `onload=...`) in IPTC caption, EXIF Make/Model/Artist, or XMP Dublin Core tags.
* **Risk**: High (Credential theft, session hijacking).
* **Mitigation**: Treat all metadata strings as untrusted data. Render strictly using React JSX text interpolation (HTML escaped by default). No use of `dangerouslySetInnerHTML`.

### 2.5 Threat: Model Tampering & Integrity Compromise
* **Description**: Attacker compromises hosting CDN or performs man-in-the-middle attack to replace genuine ONNX models with a backdoored model that always reports `no_ai_evidence`.
* **Risk**: Critical (Loss of forensic veracity).
* **Mitigation**: Pinned SHA-256 hashes in `models/registry.json`. The Web Worker computes the SHA-256 of the model `ArrayBuffer` before passing it to the ONNX session; any mismatch aborts execution.

### 2.6 Threat: Memory Exhaustion During Batch Patch Scanning
* **Description**: Very high-resolution images generating hundreds of overlapping patches, leading to memory spikes and Web Worker crashes.
* **Risk**: Medium (Tab crash).
* **Mitigation**: Hard clamp on maximum patches processed per scan ($N \le 48$). Sliding window coordinates adaptively scale stride for extreme resolutions. Intermediate patch float arrays are promptly reused or garbage collected.

### 2.7 Threat: Adversarial Noise & Counter-Forensics
* **Description**: Synthesizers intentionally applying gradient-based perturbations to evade detection while remaining perceptually identical.
* **Risk**: High (Evasion).
* **Mitigation**: Multi-modal fusion combining spatial deep CNN features with physical signal properties (noise residual consistency, frequency spikes, JPEG block grid). When signals conflict, system escalates to `uncertain` rather than a confident `no_ai_evidence`.
