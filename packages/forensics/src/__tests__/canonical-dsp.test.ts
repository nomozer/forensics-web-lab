import { describe, it, expect } from 'vitest';
import {
  extractCanonicalDspFeatures,
  CANONICAL_DSP_DIM,
  CANONICAL_DSP_FEATURE_NAMES,
  toGrayscaleFloat,
} from '../canonical-dsp.js';

describe('Canonical 16-D DSP Feature Extractor', () => {
  it('has exactly 16 feature dimensions and correct names', () => {
    expect(CANONICAL_DSP_DIM).toBe(16);
    expect(CANONICAL_DSP_FEATURE_NAMES.length).toBe(16);
    expect(CANONICAL_DSP_FEATURE_NAMES[0]).toBe('fft_high_freq_energy_ratio');
    expect(CANONICAL_DSP_FEATURE_NAMES[15]).toBe('laplacian_variance');
  });

  it('extracts valid 16-d features on synthetic 64x64 RGBA canvas', () => {
    const w = 64;
    const h = 64;
    const rgba = new Uint8ClampedArray(w * h * 4);

    // Create synthetic pattern with edges
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;
        const val = ((x ^ y) * 4) % 256;
        rgba[idx] = val;
        rgba[idx + 1] = (val + 50) % 256;
        rgba[idx + 2] = (val + 100) % 256;
        rgba[idx + 3] = 255;
      }
    }

    const feats = extractCanonicalDspFeatures(rgba, w, h);
    expect(feats).toBeInstanceOf(Float64Array);
    expect(feats.length).toBe(16);

    for (let i = 0; i < 16; i++) {
      expect(Number.isFinite(feats[i])).toBe(true);
      expect(Number.isNaN(feats[i])).toBe(false);
    }
  });

  it('handles uniform black and white images without NaN or Inf', () => {
    const w = 32;
    const h = 32;
    const black = new Uint8ClampedArray(w * h * 4); // all 0
    const white = new Uint8ClampedArray(w * h * 4).fill(255); // all 255

    const blackFeats = extractCanonicalDspFeatures(black, w, h);
    const whiteFeats = extractCanonicalDspFeatures(white, w, h);

    for (let i = 0; i < 16; i++) {
      expect(Number.isFinite(blackFeats[i])).toBe(true);
      expect(Number.isFinite(whiteFeats[i])).toBe(true);
    }
  });

  it('computes correct grayscale conversion with ITU-R BT.601 weights', () => {
    const rgba = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255]);
    const gray = toGrayscaleFloat(rgba, 3, 1);
    expect(gray[0]).toBeCloseTo(0.299, 3);
    expect(gray[1]).toBeCloseTo(0.587, 3);
    expect(gray[2]).toBeCloseTo(0.114, 3);
  });
});
