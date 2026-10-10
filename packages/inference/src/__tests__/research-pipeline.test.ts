import { describe, it, expect } from 'vitest';
import {
  LinearScorer,
  StackerScorer,
  FoldCandidateModel,
  sigmoid,
} from '../research-scorer.js';
import { RESEARCH_CANDIDATE_MODELS } from '../research-models-data.js';
import { buildBicubicTensor224 } from '../research-pipeline.js';

describe('Research Pipeline & Algebraic Scorers', () => {
  it('correctly computes sigmoid function', () => {
    expect(sigmoid(0)).toBeCloseTo(0.5, 6);
    expect(sigmoid(100)).toBeCloseTo(1.0, 6);
    expect(sigmoid(-100)).toBeCloseTo(0.0, 6);
  });

  it('correctly calculates LinearScorer rawLogit, calibratedLogit, and probability', () => {
    const scorer = new LinearScorer({
      scaler_mean: [1.0, 2.0],
      scaler_scale: [2.0, 4.0],
      coef: [0.5, -1.0],
      intercept: 0.1,
      temperature: 1.5,
    });

    const x = new Float64Array([3.0, 6.0]);
    // x_scaled = [(3-1)/2, (6-2)/4] = [1.0, 1.0]
    // dot = 1.0*0.5 + 1.0*(-1.0) = -0.5
    // raw = -0.5 + 0.1 = -0.4
    // cal = -0.4 / 1.5 = -0.2666667
    expect(scorer.rawLogit(x)).toBeCloseTo(-0.4, 6);
    expect(scorer.calibratedLogit(x)).toBeCloseTo(-0.2666667, 5);
    expect(scorer.probability(x)).toBeCloseTo(sigmoid(-0.2666667), 5);
  });

  it('correctly calculates StackerScorer fusionLogit and probability', () => {
    const stacker = new StackerScorer({
      coef: [1.2, -0.8],
      intercept: 0.05,
    });

    // fusion = 0.5 * 1.2 + (-0.2) * (-0.8) + 0.05 = 0.6 + 0.16 + 0.05 = 0.81
    expect(stacker.fusionLogit(0.5, -0.2)).toBeCloseTo(0.81, 6);
    expect(stacker.probability(0.5, -0.2)).toBeCloseTo(sigmoid(0.81), 6);
  });

  it('loads exactly 5 frozen outer-fold models with valid parameters', () => {
    expect(RESEARCH_CANDIDATE_MODELS.length).toBe(5);
    for (let i = 0; i < 5; i++) {
      const m = RESEARCH_CANDIDATE_MODELS[i];
      expect(m.outerFold).toBe(i);
      expect(m.visualScorer.coef.length).toBe(576);
      expect(m.visualScorer.scalerMean.length).toBe(576);
      expect(m.visualScorer.temperature).toBeGreaterThan(0.0);
      expect(m.dspAugmentedScorer.coef.length).toBe(16);
      expect(m.dspAugmentedScorer.scalerMean.length).toBe(16);
      expect(m.dspAugmentedScorer.temperature).toBeGreaterThan(0.0);
      expect(m.stacker.coef.length).toBe(2);
    }
  });

  it('generates a normalized [3, 224, 224] bicubic tensor from 512x512 canvas', () => {
    const rgba = new Uint8ClampedArray(512 * 512 * 4);
    for (let i = 0; i < rgba.length; i += 4) {
      rgba[i] = 120;     // R
      rgba[i + 1] = 150; // G
      rgba[i + 2] = 180; // B
      rgba[i + 3] = 255;
    }

    const tensor = buildBicubicTensor224(rgba, 512, 512, 224);
    expect(tensor.length).toBe(3 * 224 * 224);

    for (let i = 0; i < tensor.length; i++) {
      expect(Number.isFinite(tensor[i])).toBe(true);
      expect(Number.isNaN(tensor[i])).toBe(false);
    }
  });
});
