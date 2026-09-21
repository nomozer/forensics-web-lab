import React, { useState, useRef } from 'react';
import {
  MAX_FILE_SIZE_BYTES,
  MAX_PIXEL_COUNT,
  MAGIC_BYTES,
  SUPPORTED_MIME_TYPES,
} from '@forensics/shared';

export interface ValidatedImageData {
  file: File;
  rawBuffer: ArrayBuffer;
  rgba: Uint8ClampedArray;
  width: number;
  height: number;
  sha256: string;
  previewUrl: string;
}

interface ImageDropzoneProps {
  onImageSelected: (data: ValidatedImageData) => void;
  isProcessing: boolean;
}

export const ImageDropzone: React.FC<ImageDropzoneProps> = ({ onImageSelected, isProcessing }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  /**
   * Computes SHA-256 hash using the Web Crypto API.
   */
  const computeSha256 = async (buffer: ArrayBuffer): Promise<string> => {
    const digest = await crypto.subtle.digest('SHA-256', buffer);
    const hashArray = Array.from(new Uint8Array(digest));
    return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
  };

  /**
   * Validates true binary signatures (magic bytes) to defeat MIME spoofing.
   */
  const verifyMagicBytes = (view: DataView): string | null => {
    if (view.byteLength < 12) return null;

    // JPEG: FF D8 FF
    if (
      view.getUint8(0) === MAGIC_BYTES.JPEG[0] &&
      view.getUint8(1) === MAGIC_BYTES.JPEG[1] &&
      view.getUint8(2) === MAGIC_BYTES.JPEG[2]
    ) {
      return 'image/jpeg';
    }

    // PNG: 89 50 4E 47 0D 0A 1A 0A
    let isPng = true;
    for (let i = 0; i < 8; i++) {
      if (view.getUint8(i) !== MAGIC_BYTES.PNG[i]) {
        isPng = false;
        break;
      }
    }
    if (isPng) return 'image/png';

    // WebP: RIFF ... WEBP
    const isRiff =
      view.getUint8(0) === 0x52 &&
      view.getUint8(1) === 0x49 &&
      view.getUint8(2) === 0x46 &&
      view.getUint8(3) === 0x46;

    const isWebp =
      view.getUint8(8) === 0x57 &&
      view.getUint8(9) === 0x45 &&
      view.getUint8(10) === 0x42 &&
      view.getUint8(11) === 0x50;

    if (isRiff && isWebp) return 'image/webp';

    return null;
  };

  const processFile = async (file: File) => {
    setErrorMessage(null);

    if (file.size > MAX_FILE_SIZE_BYTES) {
      setErrorMessage(
        `Kích thước tập tin (${(file.size / (1024 * 1024)).toFixed(1)} MB) vượt quá giới hạn an toàn 35 MB.`
      );
      return;
    }

    try {
      const arrayBuffer = await file.arrayBuffer();
      const detectedMime = verifyMagicBytes(new DataView(arrayBuffer));

      if (!detectedMime || !SUPPORTED_MIME_TYPES.includes(detectedMime as (typeof SUPPORTED_MIME_TYPES)[number])) {
        setErrorMessage(
          'Định dạng tệp không được hỗ trợ hoặc chữ ký nhị phân bị giả mạo. Chỉ hỗ trợ ảnh JPEG, PNG và WebP hợp lệ.'
        );
        return;
      }

      // Compute SHA-256
      const sha256 = await computeSha256(arrayBuffer);

      // Create object URL and load image into canvas
      const objectUrl = URL.createObjectURL(file);
      const img = new Image();

      img.onload = () => {
        const totalPixels = img.naturalWidth * img.naturalHeight;
        if (totalPixels > MAX_PIXEL_COUNT) {
          URL.revokeObjectURL(objectUrl);
          setErrorMessage(
            `Kích thước ảnh (${img.naturalWidth}x${img.naturalHeight} = ${(totalPixels / 1e6).toFixed(
              1
            )} MP) vượt quá giới hạn phòng chống decompression bomb 64 Megapixels.`
          );
          return;
        }

        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (!ctx) {
          URL.revokeObjectURL(objectUrl);
          setErrorMessage('Không thể khởi tạo Canvas 2D context trong trình duyệt.');
          return;
        }

        ctx.drawImage(img, 0, 0);
        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        onImageSelected({
          file,
          rawBuffer: arrayBuffer,
          rgba: imgData.data,
          width: canvas.width,
          height: canvas.height,
          sha256,
          previewUrl: objectUrl,
        });
      };

      img.onerror = () => {
        URL.revokeObjectURL(objectUrl);
        setErrorMessage('Không thể giải mã raster hình ảnh. Tệp có thể đã bị hỏng.');
      };

      img.src = objectUrl;
    } catch (err) {
      setErrorMessage(`Lỗi đọc tệp: ${String(err)}`);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!isProcessing) setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (isProcessing) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  };

  return (
    <div style={{ marginBottom: '32px' }}>
      <input
        ref={fileInputRef}
        type="file"
        accept=".jpg,.jpeg,.png,.webp"
        style={{ display: 'none' }}
        onChange={handleFileChange}
        disabled={isProcessing}
      />

      <div
        className={`dropzone-box ${isDragging ? 'dragging' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isProcessing && fileInputRef.current?.click()}
      >
        <div style={{ maxWidth: '520px', margin: '0 auto' }}>
          <div
            style={{
              width: '56px',
              height: '56px',
              borderRadius: '50%',
              background: 'rgba(6, 182, 212, 0.1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 16px auto',
              color: 'var(--accent-cyan)',
            }}
          >
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
          </div>

          <h3 style={{ fontSize: '18px', fontWeight: '700', marginBottom: '8px' }}>
            Kéo thả ảnh cần giám định vào đây, hoặc nhấn để chọn
          </h3>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '16px' }}>
            Hỗ trợ <strong>JPEG, PNG, WebP</strong> (Tối đa 35 MB, tối đa 64 MP).
            <br />
            Kiểm tra chữ ký nhị phân tự động chống giả mạo đuôi tệp.
          </p>

          <div style={{ display: 'flex', justifyContent: 'center', gap: '8px', flexWrap: 'wrap' }}>
            <span className="badge badge-wasm" style={{ fontSize: '11px' }}>
              🔒 Xử lý cục bộ 100% trong RAM
            </span>
            <span className="badge badge-notice" style={{ fontSize: '11px' }}>
              ⚡ Không upload lên server
            </span>
          </div>
        </div>
      </div>

      {errorMessage && (
        <div
          style={{
            marginTop: '16px',
            padding: '12px 16px',
            background: 'rgba(244, 63, 94, 0.15)',
            border: '1px solid rgba(244, 63, 94, 0.3)',
            borderRadius: 'var(--radius-md)',
            color: '#fca5a5',
            fontSize: '13px',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
          }}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          {errorMessage}
        </div>
      )}
    </div>
  );
};
