/**
 * Research Pipeline for In-Browser FP32 ONNX Inference and Multimodal Scoring.
 *
 * Implements the exact research inference path:
 * Image -> Canvas 512x512 -> Canonical 16-d DSP -> Bicubic 224x224 Tensor [1, 3, 224, 224]
 *       -> ONNX Backbone 576-d -> 5 Outer Fold Models (visual_calibrated & late_fusion_dsp_augmented).
 */

import * as ort from 'onnxruntime-web';
import { extractCanonicalDspFeatures } from '@forensics/forensics';
import { FoldCandidateModel, FoldScoringResult } from './research-scorer.js';
import { RESEARCH_CANDIDATE_MODELS } from './research-models-data.js';

// ImageNet normalization constants
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

/**
 * Bicubic weight kernel (Keys cubic spline with a = -0.75 matching PyTorch BICUBIC).
 */
function cubicWeight(x: number): number {
  const ax = Math.abs(x);
  const a = -0.75;
  if (ax <= 1.0) {
    return (a + 2.0) * ax * ax * ax - (a + 3.0) * ax * ax + 1.0;
  }
  if (ax < 2.0) {
    return a * ax * ax * ax - 5.0 * a * ax * ax + 8.0 * a * ax - 4.0 * a;
  }
  return 0.0;
}

/**
 * Resize 512x512 RGBA to 224x224 and normalize into [1, 3, 224, 224] Float32Array tensor.
 */
export function buildBicubicTensor224(
  rgba512: Uint8ClampedArray | Uint8Array,
  srcW = 512,
  srcH = 512,
  dstSize = 224
): Float32Array {
  const tensor = new Float32Array(3 * dstSize * dstSize);
  const planeSize = dstSize * dstSize;
  const scaleX = srcW / dstSize;
  const scaleY = srcH / dstSize;

  for (let dy = 0; dy < dstSize; dy++) {
    const srcY = (dy + 0.5) * scaleY - 0.5;
    const yCenter = Math.floor(srcY);

    for (let dx = 0; dx < dstSize; dx++) {
      const srcX = (dx + 0.5) * scaleX - 0.5;
      const xCenter = Math.floor(srcX);

      let rSum = 0, gSum = 0, bSum = 0, wSum = 0;

      for (let j = -1; j <= 2; j++) {
        const sy = Math.max(0, Math.min(srcH - 1, yCenter + j));
        const wy = cubicWeight(srcY - (yCenter + j));

        for (let i = -1; i <= 2; i++) {
          const sx = Math.max(0, Math.min(srcW - 1, xCenter + i));
          const wx = cubicWeight(srcX - (xCenter + i));
          const w = wx * wy;

          const idx = (sy * srcW + sx) * 4;
          rSum += (rgba512[idx] / 255.0) * w;
          gSum += (rgba512[idx + 1] / 255.0) * w;
          bSum += (rgba512[idx + 2] / 255.0) * w;
          wSum += w;
        }
      }

      const invW = wSum > 1e-6 ? 1.0 / wSum : 1.0;
      const r = Math.max(0.0, Math.min(1.0, rSum * invW));
      const g = Math.max(0.0, Math.min(1.0, gSum * invW));
      const b = Math.max(0.0, Math.min(1.0, bSum * invW));

      const dstIdx = dy * dstSize + dx;
      tensor[dstIdx] = (r - MEAN[0]) / STD[0];
      tensor[planeSize + dstIdx] = (g - MEAN[1]) / STD[1];
      tensor[2 * planeSize + dstIdx] = (b - MEAN[2]) / STD[2];
    }
  }

  return tensor;
}

export interface ResearchInferenceResult {
  timingMs: {
    dsp: number;
    preprocessing: number;
    backbone: number;
    scoring: number;
    total: number;
  };
  features: {
    visual576: Float32Array;
    dsp16: Float64Array;
  };
  folds: FoldScoringResult[];
  summary: {
    visual_calibrated: {
      mean_probability: number;
      prediction: number;
    };
    late_fusion_dsp_augmented: {
      mean_probability: number;
      prediction: number;
    };
    note: string;
  };
}

export class ResearchPipeline {
  private session: ort.InferenceSession | null = null;
  private readonly candidateModels: FoldCandidateModel[] = RESEARCH_CANDIDATE_MODELS;

  /**
   * Initializes the ONNX Runtime Web session from a model ArrayBuffer.
   */
  public async initialize(modelBuffer: ArrayBuffer): Promise<void> {
    ort.env.wasm.wasmPaths = '/wasm/';
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;

    this.session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
  }

  public isReady(): boolean {
    return this.session !== null;
  }

  /**
   * Executes end-to-end research inference on 512x512 RGBA pixel buffer.
   */
  public async analyzeImage512(rgba512: Uint8ClampedArray | Uint8Array): Promise<ResearchInferenceResult> {
    if (!this.session) {
      throw new Error('Research ONNX session is not initialized.');
    }

    const tStart = performance.now();

    // 1. DSP extraction on 512x512 canvas
    const t0Dsp = performance.now();
    const dspFeatures = extractCanonicalDspFeatures(rgba512, 512, 512);
    const tDsp = performance.now() - t0Dsp;

    // 2. Preprocessing: bicubic resize to 224x224 and ImageNet normalization
    const t0Pre = performance.now();
    const tensor224 = buildBicubicTensor224(rgba512, 512, 512, 224);
    const tPre = performance.now() - t0Pre;

    // 3. Backbone forward pass in ONNX Runtime Web
    const t0Backbone = performance.now();
    const inputTensor = new ort.Tensor('float32', tensor224, [1, 3, 224, 224]);
    const feeds = { input: inputTensor };
    const outputs = await this.session.run(feeds);
    const visualOutput = outputs['visual_features'] || Object.values(outputs)[0];
    const visualFeatures = visualOutput.data as Float32Array;
    const tBackbone = performance.now() - t0Backbone;

    // 4. Scoring across all 5 outer-fold models
    const t0Score = performance.now();
    const foldResults: FoldScoringResult[] = [];
    let sumVisProb = 0;
    let sumFusionProb = 0;

    for (const model of this.candidateModels) {
      const res = model.score(visualFeatures, dspFeatures);
      foldResults.push(res);
      sumVisProb += res.visual_calibrated.probability;
      sumFusionProb += res.late_fusion_dsp_augmented.probability;
    }
    const tScore = performance.now() - t0Score;
    const tTotal = performance.now() - tStart;

    const meanVisProb = sumVisProb / this.candidateModels.length;
    const meanFusionProb = sumFusionProb / this.candidateModels.length;

    return {
      timingMs: {
        dsp: Math.round(tDsp * 10) / 10,
        preprocessing: Math.round(tPre * 10) / 10,
        backbone: Math.round(tBackbone * 10) / 10,
        scoring: Math.round(tScore * 10) / 10,
        total: Math.round(tTotal * 10) / 10,
      },
      features: {
        visual576: visualFeatures,
        dsp16: dspFeatures,
      },
      folds: foldResults,
      summary: {
        visual_calibrated: {
          mean_probability: meanVisProb,
          prediction: meanVisProb >= 0.5 ? 1 : 0,
        },
        late_fusion_dsp_augmented: {
          mean_probability: meanFusionProb,
          prediction: meanFusionProb >= 0.5 ? 1 : 0,
        },
        note: 'Kết quả tổng hợp là giá trị trung bình cộng mô tả 5 outer folds; nghiên cứu chính thức bảo toàn 5 folds độc lập.',
      },
    };
  }
}
