/**
 * Pure Algebraic Research Scorers for 5 Outer Folds in TypeScript.
 *
 * Implements 100% parity with Python ml/evaluation/independent_model_bindings.py:
 * - LinearScorer with StandardScaler, linear dot product, and Temperature Scaling.
 * - StackerScorer with 2-input logistic regression stacker.
 * - FoldCandidateModel evaluating both visual_calibrated and late_fusion_dsp_augmented recipes.
 */

export interface LinearScorerParams {
  scaler_mean: number[];
  scaler_scale: number[];
  coef: number[];
  intercept: number;
  temperature: number;
}

export interface StackerParams {
  coef: [number, number];
  intercept: number;
}

export interface FoldParams {
  outer_fold: number;
  visual_scorer: LinearScorerParams;
  dsp_augmented_scorer: LinearScorerParams;
  stacker: StackerParams;
}

export function sigmoid(z: number): number {
  const clipped = Math.max(-35.0, Math.min(35.0, z));
  return 1.0 / (1.0 + Math.exp(-clipped));
}

export class LinearScorer {
  public readonly scalerMean: Float64Array;
  public readonly scalerScale: Float64Array;
  public readonly coef: Float64Array;
  public readonly intercept: number;
  public readonly temperature: number;

  constructor(params: LinearScorerParams) {
    this.scalerMean = new Float64Array(params.scaler_mean);
    this.scalerScale = new Float64Array(params.scaler_scale);
    this.coef = new Float64Array(params.coef);
    this.intercept = params.intercept;
    this.temperature = params.temperature;
  }

  public rawLogit(x: Float64Array | Float32Array): number {
    let dot = 0.0;
    const len = x.length;
    for (let i = 0; i < len; i++) {
      const scaled = (x[i] - this.scalerMean[i]) / this.scalerScale[i];
      dot += scaled * this.coef[i];
    }
    return dot + this.intercept;
  }

  public calibratedLogit(x: Float64Array | Float32Array): number {
    return this.rawLogit(x) / this.temperature;
  }

  public probability(x: Float64Array | Float32Array): number {
    return sigmoid(this.calibratedLogit(x));
  }
}

export class StackerScorer {
  public readonly coef: [number, number];
  public readonly intercept: number;

  constructor(params: StackerParams) {
    this.coef = [params.coef[0], params.coef[1]];
    this.intercept = params.intercept;
  }

  public fusionLogit(visualCalLogit: number, dspCalLogit: number): number {
    return (
      visualCalLogit * this.coef[0] +
      dspCalLogit * this.coef[1] +
      this.intercept
    );
  }

  public probability(visualCalLogit: number, dspCalLogit: number): number {
    return sigmoid(this.fusionLogit(visualCalLogit, dspCalLogit));
  }
}

export interface FoldScoringResult {
  visual_calibrated: {
    raw_logit: number;
    calibrated_logit: number;
    probability: number;
    prediction: number;
  };
  late_fusion_dsp_augmented: {
    raw_logit: number;
    calibrated_logit: number;
    probability: number;
    prediction: number;
  };
}

export class FoldCandidateModel {
  public readonly outerFold: number;
  public readonly visualScorer: LinearScorer;
  public readonly dspAugmentedScorer: LinearScorer;
  public readonly stacker: StackerScorer;

  constructor(params: FoldParams) {
    this.outerFold = params.outer_fold;
    this.visualScorer = new LinearScorer(params.visual_scorer);
    this.dspAugmentedScorer = new LinearScorer(params.dsp_augmented_scorer);
    this.stacker = new StackerScorer(params.stacker);
  }

  public score(
    visualFeatures: Float64Array | Float32Array,
    dspFeatures: Float64Array,
  ): FoldScoringResult {
    // 1. Visual Calibrated branch
    const zVRaw = this.visualScorer.rawLogit(visualFeatures);
    const zVCal = zVRaw / this.visualScorer.temperature;
    const pV = sigmoid(zVCal);

    // 2. DSP branch
    const zDRaw = this.dspAugmentedScorer.rawLogit(dspFeatures);
    const zDCal = zDRaw / this.dspAugmentedScorer.temperature;

    // 3. Late Fusion Stacker branch
    const zFusion = this.stacker.fusionLogit(zVCal, zDCal);
    const pFusion = sigmoid(zFusion);

    return {
      visual_calibrated: {
        raw_logit: zVRaw,
        calibrated_logit: zVCal,
        probability: pV,
        prediction: pV >= 0.5 ? 1 : 0,
      },
      late_fusion_dsp_augmented: {
        raw_logit: zFusion,
        calibrated_logit: zFusion,
        probability: pFusion,
        prediction: pFusion >= 0.5 ? 1 : 0,
      },
    };
  }
}
