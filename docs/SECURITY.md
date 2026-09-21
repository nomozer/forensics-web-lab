# Security Guidelines & Controls: Forensics Web Lab

## 1. Security Architecture Principles

Because **Forensics Web Lab** processes untrusted binary files (images) and displays user-controllable metadata strings, it implements strict defense-in-depth controls:

1. **Untrusted Input Treatment**: Every uploaded file and every extracted string (EXIF tags, IPTC comments, camera serials, file names) is treated as potentially adversarial.
2. **Defensive Parsing**: File decoders and parsers must operate with bounded memory and time complexity to prevent Denial of Service (DoS).
3. **Execution Sandbox**: Heavy data manipulation and tensor processing run inside Web Workers, isolating memory-intensive workloads from the main browsing context.

---

## 2. Defensive Security Implementations

### 2.1 Binary Signature (Magic Byte) Enforcement
* Do not rely on user-supplied file extensions (`.jpg`, `.png`, `.webp`) or browser-supplied `File.type` MIME strings.
* Read the initial 16 bytes of the file array buffer and match against strict hexadecimal magic signatures:
  * JPEG: `FF D8 FF`
  * PNG: `89 50 4E 47 0D 0A 1A 0A`
  * WebP: `52 49 46 46 [4 bytes size] 57 45 42 50`
* Reject any non-conforming or polyglot payloads prior to image decoding.

### 2.2 Decompression Bomb & Resource Exhaustion Defense
* **Raw File Size Cap**: Files $> 35\text{ MB}$ are rejected immediately.
* **Pixel Dimension Cap**: Read image headers to determine width and height prior to full canvas allocation. If $W \times H > 64\,000\,000$ pixels (e.g. $8000 \times 8000$), decoding is aborted.
* **Canvas Allocation Limits**: Offscreen rendering canvases are clamped to a maximum internal processing dimension of $2048 \times 2048$ px via proportional aspect-preserving downsampling.

### 2.3 Cross-Site Scripting (XSS) Prevention
* Metadata fields (e.g. EXIF UserComment, Make, Model, Software, Artist, XMP metadata) frequently contain attacker-crafted payloads (e.g. `<script>`, `javascript:`, malformed Unicode).
* **Escape by Default**: All metadata rendering in React utilizes text nodes (`textContent` semantics) and strict HTML escaping. Under no circumstances is `dangerouslySetInnerHTML` permitted for metadata or report rendering.
* **Filename Sanitization**: Uploaded filenames are stripped of path traversal characters (`..`, `/`, `\`), control characters, and HTML delimiters before display or export.

### 2.4 Model Integrity Verification
* Model weights downloaded over the network are verified via SHA-256 against entries in `models/registry.json`.
* If the computed hash fails to match the pinned registry hash, model initialization aborts with an explicit security alert.

### 2.5 Recommended Content Security Policy (CSP)
For production deployments, the hosting server must deliver the following headers:
```http
Content-Security-Policy: default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' blob: data:; worker-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'self';
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```
*(Note: `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` enable `SharedArrayBuffer` for high-performance multi-threaded WASM execution).*
