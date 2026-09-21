import { describe, it, expect } from 'vitest';
import { PatchExtractor, HeatmapAccumulator, FusionCalibrator } from '../index.js';
import { ForensicSummary, ProvenanceResult, LocalizationResult } from '@forensics/shared';

describe('@forensics/inference components', () => {
  const dummyForensics: ForensicSummary = {
    signals: [
      {
        id: 'fft_radial_anomaly',
        name: 'FFT',
        category: 'frequency',
        score: 0.2,
        interpretation: 'Bình thường',
        details: {},
      },
    ],
    supportingEvidence: [],
    refutingEvidence: [],
  };

  const dummyProvenance: ProvenanceResult = {
    c2paStatus: 'unsupported',
    metadataSummary: [],
  };

  const dummyLocalization: LocalizationResult = {
    available: false,
    suspiciousAreaRatio: 0,
    regions: [],
  };

  it('extracts global tensor with correct NCHW dimensions', () => {
    const w = 400;
    const h = 300;
    const rgba = new Uint8ClampedArray(w * h * 4).fill(128);
    const tensor = PatchExtractor.extractGlobalTensor(rgba, w, h, 224);

    expect(tensor.length).toBe(3 * 224 * 224);
  });

  it('extracts bounded number of overlapping patches', () => {
    const w = 600;
    const h = 600;
    const rgba = new Uint8ClampedArray(w * h * 4).fill(200);
    const patches = PatchExtractor.extractPatches(rgba, w, h, 224);

    expect(patches.length).toBeGreaterThan(0);
    expect(patches.length).toBeLessThanOrEqual(48);
    expect(patches[0].tensorData.length).toBe(3 * 224 * 224);
  });

  it('accumulates patch scores into a smooth heatmap grid', () => {
    const scoredPatches = [
      {
        patch: {
          box: { x: 0.1, y: 0.1, width: 0.4, height: 0.4 },
          pixelBox: { x: 10, y: 10, width: 40, height: 40 },
          tensorData: new Float32Array(0),
        },
        score: 0.85,
      },
    ];

    const { localization, heatmapGrid } = HeatmapAccumulator.accumulate(scoredPatches, 16, 16);
    expect(localization.available).toBe(true);
    expect(localization.regions.length).toBeGreaterThanOrEqual(1);
    expect(heatmapGrid.length).toBe(16);
    expect(heatmapGrid[0].length).toBe(16);
  });

  it('calibrates logits and maps to fully_generated when generative confidence is high', () => {
    const result = FusionCalibrator.fuse({
      rawLogits: [0.1, 4.5, 0.2], // strong generative logit
      temperature: 1.0,
      localization: dummyLocalization,
      forensics: dummyForensics,
      provenance: dummyProvenance,
      hasModel: true,
    });

    expect(result.label).toBe('fully_generated');
    expect(result.confidence).toBeGreaterThan(0.9);
    expect(result.probabilities.fully_generated).toBeGreaterThan(0.9);
  });

  it('maps to uncertain when confidence is below threshold or conflicting', () => {
    const result = FusionCalibrator.fuse({
      rawLogits: [1.0, 1.05, 0.95], // near tie
      temperature: 1.0,
      localization: dummyLocalization,
      forensics: dummyForensics,
      provenance: dummyProvenance,
      hasModel: true,
    });

    expect(result.label).toBe('uncertain');
    expect(result.explanation).toContain('mâu thuẫn');
  });

  it('maps to ai_edited when localized high-suspicion cluster exists', () => {
    const localizedLoc: LocalizationResult = {
      available: true,
      suspiciousAreaRatio: 0.15,
      regions: [
        {
          box: { x: 0.2, y: 0.2, width: 0.3, height: 0.3 },
          score: 0.88,
          label: 'Vùng mô hình nghi ngờ can thiệp',
        },
      ],
    };

    const result = FusionCalibrator.fuse({
      rawLogits: [2.0, 0.5, 1.8],
      temperature: 1.0,
      localization: localizedLoc,
      forensics: dummyForensics,
      provenance: dummyProvenance,
      hasModel: true,
    });

    expect(result.label).toBe('ai_edited');
  });

  it('handles missing model gracefully without fake numbers', () => {
    const result = FusionCalibrator.fuse({
      hasModel: false,
      localization: dummyLocalization,
      forensics: dummyForensics,
      provenance: dummyProvenance,
    });

    expect(result.label).toBe('uncertain');
    expect(result.confidence).toBeNull();
    expect(result.probabilities).toBeNull();
    expect(result.modelAvailable).toBe(false);
    expect(result.modelStatus).toBe('not-installed');
    expect(result.refutingEvidence.some((e) => e.includes('Model not installed'))).toBe(true);
  });

  it('never outputs AI verdict or probabilities in no-model state even with elevated DSP or metadata', () => {
    const aiProvenance: ProvenanceResult = {
      c2paStatus: 'unsupported',
      metadataSummary: [
        {
          tag: 'Software',
          value: 'Midjourney v6.0',
          category: 'software',
          suspicionScore: 0.95,
        },
      ],
    };

    const elevatedForensics: ForensicSummary = {
      signals: [
        {
          id: 'fft_radial_anomaly',
          name: 'FFT',
          category: 'frequency',
          score: 0.88,
          interpretation: 'Phát hiện đỉnh năng lượng cao',
          details: {},
        },
        {
          id: 'noise_residual_inconsistency',
          name: 'Noise',
          category: 'noise',
          score: 0.85,
          interpretation: 'Nhiễu bất đồng nhất',
          details: {},
        },
      ],
      supportingEvidence: [],
      refutingEvidence: [],
    };

    const result = FusionCalibrator.fuse({
      hasModel: false,
      localization: dummyLocalization,
      forensics: elevatedForensics,
      provenance: aiProvenance,
    });

    // Invariant: Without deep model, verdict MUST be uncertain, probabilities MUST be null
    expect(result.label).toBe('uncertain');
    expect(result.confidence).toBeNull();
    expect(result.probabilities).toBeNull();
    expect(result.modelAvailable).toBe(false);
    expect(result.modelStatus).toBe('not-installed');
    expect(result.supportingEvidence.some((e) => e.includes('Tín hiệu khám phá sơ bộ'))).toBe(true);
  });
});
