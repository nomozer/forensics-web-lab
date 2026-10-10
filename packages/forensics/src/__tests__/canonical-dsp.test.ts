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

  it('profiles 512x512 DSP subcomponents', async () => {
    const {
      computeCanonicalFft,
      computeCanonicalDct,
      computeCanonicalNoise,
      computeCanonicalJpeg,
      computeCanonicalLaplacianVariance,
    } = await import('../canonical-dsp.js');

    const gray = new Float32Array(512 * 512);
    for (let i = 0; i < gray.length; i++) {
      gray[i] = ((i * 17) % 256) / 255.0;
    }

    // Warm-up
    computeCanonicalFft(gray, 512, 512);
    computeCanonicalDct(gray, 512, 512);
    computeCanonicalNoise(gray, 512, 512);
    computeCanonicalJpeg(gray, 512, 512);
    computeCanonicalLaplacianVariance(gray, 512, 512);

    const iters = 5;
    let tFft = 0, tDct = 0, tNoise = 0, tJpeg = 0, tLap = 0;

    for (let i = 0; i < iters; i++) {
      let t0 = performance.now();
      computeCanonicalFft(gray, 512, 512);
      tFft += performance.now() - t0;

      t0 = performance.now();
      computeCanonicalDct(gray, 512, 512);
      tDct += performance.now() - t0;

      t0 = performance.now();
      computeCanonicalNoise(gray, 512, 512);
      tNoise += performance.now() - t0;

      t0 = performance.now();
      computeCanonicalJpeg(gray, 512, 512);
      tJpeg += performance.now() - t0;

      t0 = performance.now();
      computeCanonicalLaplacianVariance(gray, 512, 512);
      tLap += performance.now() - t0;
    }

    console.log('[DSP PROFILE] FFT (128x128):', (tFft / iters).toFixed(2), 'ms');
    console.log('[DSP PROFILE] DCT (8x8):', (tDct / iters).toFixed(2), 'ms');
    console.log('[DSP PROFILE] Noise Residual:', (tNoise / iters).toFixed(2), 'ms');
    console.log('[DSP PROFILE] JPEG Grid:', (tJpeg / iters).toFixed(2), 'ms');
    console.log('[DSP PROFILE] Laplacian Variance:', (tLap / iters).toFixed(2), 'ms');
    console.log('[DSP PROFILE] TOTAL:', ((tFft + tDct + tNoise + tJpeg + tLap) / iters).toFixed(2), 'ms');
  });
});
