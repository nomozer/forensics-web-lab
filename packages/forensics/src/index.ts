import { ForensicSignal, ForensicSummary } from '@forensics/shared';
import { compute2DFftAnalysis, FftAnalysisResult } from './fft2d.js';
import { compute8x8DctAnalysis, DctAnalysisResult } from './dct2d.js';
import { computeNoiseResidualAnalysis, NoiseAnalysisResult } from './noise-analysis.js';
import { computeJpegBlockGridAnalysis, JpegBlockGridResult } from './jpeg-block-grid.js';
import { computeElaStatistics, ElaAnalysisResult } from './ela.js';

export * from './fft2d.js';
export * from './dct2d.js';
export * from './noise-analysis.js';
export * from './jpeg-block-grid.js';
export * from './ela.js';
export * from './canonical-dsp.js';

export interface ImageBufferView {
  rgba: Uint8ClampedArray | Uint8Array;
  width: number;
  height: number;
}

/**
 * Converts RGBA image bytes into a normalized Float32 grayscale buffer [0.0, 1.0].
 */
export function rgbaToGrayscaleFloat(
  rgba: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number
): Float32Array {
  const total = width * height;
  const gray = new Float32Array(total);
  for (let i = 0; i < total; i++) {
    const idx = i * 4;
    // Standard luminance weights
    gray[i] = (0.299 * rgba[idx] + 0.587 * rgba[idx + 1] + 0.114 * rgba[idx + 2]) / 255.0;
  }
  return gray;
}

/**
 * Resamples a grayscale buffer to a fixed power-of-two square (e.g. 128x128) for 2D FFT.
 */
export function downsampleToPowerOfTwo(
  src: Float32Array,
  srcW: number,
  srcH: number,
  targetSize: number
): Float32Array {
  const dst = new Float32Array(targetSize * targetSize);
  const scaleX = srcW / targetSize;
  const scaleY = srcH / targetSize;

  for (let y = 0; y < targetSize; y++) {
    const sy = Math.min(srcH - 1, Math.floor(y * scaleY));
    for (let x = 0; x < targetSize; x++) {
      const sx = Math.min(srcW - 1, Math.floor(x * scaleX));
      dst[y * targetSize + x] = src[sy * srcW + sx];
    }
  }
  return dst;
}

/**
 * Executes the full suite of forensic DSP signal extractors.
 */
export function analyzeForensicSignals(
  image: ImageBufferView,
  recompressedRgba?: Uint8ClampedArray | Uint8Array
): ForensicSummary {
  const gray = rgbaToGrayscaleFloat(image.rgba, image.width, image.height);
  const fftInput = downsampleToPowerOfTwo(gray, image.width, image.height, 128);

  const fft = compute2DFftAnalysis(fftInput, 128);
  const dct = compute8x8DctAnalysis(gray, image.width, image.height);
  const noise = computeNoiseResidualAnalysis(gray, image.width, image.height);
  const jpeg = computeJpegBlockGridAnalysis(gray, image.width, image.height);

  const signals: ForensicSignal[] = [];
  const supportingEvidence: string[] = [];
  const refutingEvidence: string[] = [];

  // 1. FFT Frequency Signal
  signals.push({
    id: 'fft_radial_anomaly',
    name: '2D-FFT Radial Anomaly',
    category: 'frequency',
    score: fft.radialAnomalyScore,
    interpretation:
      fft.spectralPeakCount > 0
        ? `Phát hiện ${fft.spectralPeakCount} đỉnh phổ bất thường ở dải tần số cao (dấu hiệu upsampling dạng ô cờ).`
        : 'Phổ tần số tự nhiên, suy giảm đều theo quy luật 1/f.',
    details: { ...fft },
  });
  if (fft.radialAnomalyScore > 0.6) {
    supportingEvidence.push('Phổ 2D-FFT xuất hiện các gai tần số chu kỳ bất thường.');
  }

  // 2. DCT AC Energy Variance
  signals.push({
    id: 'dct_energy_distribution',
    name: '2D-DCT Block Energy Distribution',
    category: 'frequency',
    score: dct.dctAnomalyScore,
    interpretation:
      dct.dctAnomalyScore > 0.6
        ? 'Phân bố năng lượng AC biến thiên dị thường giữa các khối 8x8.'
        : 'Phân bố hệ số DCT đồng đều, phù hợp với ảnh chụp tự nhiên.',
    details: { ...dct },
  });

  // 3. Noise Residual Consistency
  signals.push({
    id: 'noise_residual_inconsistency',
    name: 'Noise Residual Spatial Inconsistency',
    category: 'noise',
    score: noise.noiseInconsistencyScore,
    interpretation:
      noise.noiseInconsistencyScore > 0.6
        ? 'Độ lệch chuẩn nhiễu hạt không đồng nhất giữa các vùng (dấu hiệu ghép hoặc inpainting).'
        : 'Nhiễu cảm biến phân bố đồng đều toàn bức ảnh.',
    details: { ...noise },
  });
  if (noise.noiseInconsistencyScore > 0.65) {
    supportingEvidence.push('Nhiễu cảm biến cục bộ không đồng nhất giữa các vùng không gian.');
  } else {
    refutingEvidence.push('Nhiễu hạt không gian đồng nhất trên toàn bức ảnh.');
  }

  // 4. JPEG Block Artifacts
  signals.push({
    id: 'jpeg_block_artifacts',
    name: 'JPEG Block Artifact Grid (BAG)',
    category: 'compression',
    score: jpeg.jpegArtifactScore,
    interpretation:
      jpeg.jpegArtifactScore > 0.7
        ? 'Lưới nén JPEG 8x8 hiển thị rõ ràng với độ tương phản cao.'
        : 'Không phát hiện lưới nén JPEG 8x8 bất thường.',
    details: { ...jpeg },
  });
  if (jpeg.jpegArtifactScore > 0.7) {
    refutingEvidence.push('Ảnh có dấu hiệu nén JPEG mạnh, có thể đã làm mờ dấu vết vi mô.');
  }

  // 5. Optional ELA
  if (recompressedRgba) {
    const ela = computeElaStatistics(image.rgba, recompressedRgba, image.width, image.height);
    signals.push({
      id: 'error_level_analysis',
      name: 'Error Level Analysis (ELA)',
      category: 'compression',
      score: ela.stats.elaScore,
      interpretation:
        ela.stats.elaScore > 0.6
          ? 'Mức sai số nén biến thiên cục bộ đáng kể.'
          : 'Mức độ sai số nén đồng nhất trên toàn bộ bề mặt.',
      details: { ...ela.stats },
    });
  }

  return {
    signals,
    supportingEvidence,
    refutingEvidence,
  };
}
