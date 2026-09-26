import { describe, it, expect } from 'vitest';
import {
  compute2DFftAnalysis,
  compute8x8DctAnalysis,
  computeNoiseResidualAnalysis,
  computeJpegBlockGridAnalysis,
  rgbaToGrayscaleFloat,
  analyzeForensicSignals,
} from '../index.js';

describe('@forensics/forensics DSP algorithms', () => {
  it('computes 2D FFT on a uniform buffer without crashing', () => {
    const size = 64;
    const uniform = new Float32Array(size * size).fill(0.5);
    const result = compute2DFftAnalysis(uniform, size);

    expect(result.highFreqEnergyRatio).toBeGreaterThanOrEqual(0);
    expect(result.radialAnomalyScore).toBeGreaterThanOrEqual(0);
    expect(result.radialAnomalyScore).toBeLessThanOrEqual(1.0);
  });

  it('detects high-frequency spikes in checkerboard synthetic pattern', () => {
    const size = 64;
    const checker = new Float32Array(size * size);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        checker[y * size + x] = (x + y) % 2 === 0 ? 1.0 : 0.0;
      }
    }
    const result = compute2DFftAnalysis(checker, size);
    expect(result.highFreqEnergyRatio).toBeGreaterThan(0.2);
  });

  it('computes 2D-DCT block statistics correctly', () => {
    const w = 32;
    const h = 32;
    const buffer = new Float32Array(w * h).fill(0.2);
    const result = compute8x8DctAnalysis(buffer, w, h);

    expect(result.dctAnomalyScore).toBeGreaterThanOrEqual(0);
    expect(result.dctAnomalyScore).toBeLessThanOrEqual(1.0);
  });

  it('measures noise residual consistency and inconsistency score', () => {
    const w = 48;
    const h = 48;
    const cleanImage = new Float32Array(w * h).fill(0.5);
    // Add subtle baseline noise
    for (let i = 0; i < w * h; i++) {
      cleanImage[i] += ((i % 5) - 2) * 0.005;
    }
    const resultClean = computeNoiseResidualAnalysis(cleanImage, w, h);
    expect(resultClean.noiseInconsistencyScore).toBeLessThan(0.3);

    // Add high localized noise to top-left corner
    const splicedImage = new Float32Array(cleanImage);
    for (let y = 0; y < 16; y++) {
      for (let x = 0; x < 16; x++) {
        splicedImage[y * w + x] += (( (x * 7 + y * 13) % 11) - 5) * 0.05;
      }
    }
    const resultSpliced = computeNoiseResidualAnalysis(splicedImage, w, h);
    expect(resultSpliced.maxToMinRatio).toBeGreaterThan(resultClean.maxToMinRatio);
    expect(resultSpliced.noiseInconsistencyScore).toBeGreaterThan(resultClean.noiseInconsistencyScore);
  });

  it('runs full analyzeForensicSignals pipeline', () => {
    const w = 64;
    const h = 64;
    const rgba = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h * 4; i += 4) {
      rgba[i] = 120;
      rgba[i + 1] = 130;
      rgba[i + 2] = 140;
      rgba[i + 3] = 255;
    }

    const summary = analyzeForensicSignals({ rgba, width: w, height: h });
    expect(summary.signals.length).toBeGreaterThanOrEqual(4);
    expect(Array.isArray(summary.supportingEvidence)).toBe(true);
    expect(Array.isArray(summary.refutingEvidence)).toBe(true);
  });
});
