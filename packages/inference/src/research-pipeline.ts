/**
 * Research Pipeline for In-Browser FP32 ONNX Inference and Multimodal Scoring.
 *
 * Implements the exact research inference path:
 * Image -> Canvas 512x512 -> Canonical 16-d DSP -> Bicubic 224x224 Tensor [1, 3, 224, 224]
 *       -> ONNX Backbone 576-d -> 5 Outer Fold Models (visual_calibrated & late_fusion_dsp_augmented).
 */

import * as ort from "onnxruntime-web";
import { extractCanonicalDspFeatures } from "@forensics/forensics";
import { FoldCandidateModel, FoldScoringResult } from "./research-scorer.js";
import { RESEARCH_CANDIDATE_MODELS } from "./research-models-data.js";

// ImageNet normalization constants
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

const PRECISION_BITS = 22;

/**
 * Keys cubic spline filter with a = -0.5 matching Pillow's bicubic_filter.
 */
function bicubicFilter(x: number): number {
  const ax = Math.abs(x);
  const a = -0.5;
  if (ax < 1.0) {
    return ((a + 2.0) * ax - (a + 3.0)) * ax * ax + 1.0;
  }
  if (ax < 2.0) {
    return (((ax - 5.0) * ax + 8.0) * ax - 4.0) * a;
  }
  return 0.0;
}

function precomputeCoeffsBicubic(inSize: number, outSize: number) {
  const scale = inSize / outSize;
  const filterscale = scale > 1.0 ? scale : 1.0;
  const support = 2.0 * filterscale;

  const bounds: [number, number][] = [];
  const coeffs: Int32Array[] = [];
  const invFilterscale = 1.0 / filterscale;

  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let xmin = Math.floor(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.floor(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    const xmaxCount = xmax - xmin;

    const k = new Float64Array(xmaxCount);
    let ww = 0.0;
    for (let x = 0; x < xmaxCount; x++) {
      const w = bicubicFilter((x + xmin - center + 0.5) * invFilterscale);
      k[x] = w;
      ww += w;
    }

    if (ww !== 0.0) {
      for (let x = 0; x < xmaxCount; x++) {
        k[x] /= ww;
      }
    }

    const kInt = new Int32Array(xmaxCount);
    for (let x = 0; x < xmaxCount; x++) {
      const val = k[x];
      if (val < 0) {
        kInt[x] = Math.floor(val * (1 << PRECISION_BITS) - 0.5);
      } else {
        kInt[x] = Math.floor(val * (1 << PRECISION_BITS) + 0.5);
      }
    }

    bounds.push([xmin, xmaxCount]);
    coeffs.push(kInt);
  }
  return { bounds, coeffs };
}

/**
 * Resize arbitrary dimensions RGBA to 224x224 and normalize into [1, 3, 224, 224] Float32Array tensor.
 * Implements 100% bit-exact parity with torchvision.transforms.Resize((224, 224), interpolation=BICUBIC).
 */
export function buildBicubicTensor224(
  rgba: Uint8ClampedArray | Uint8Array,
  srcW = 512,
  srcH = 512,
  dstSize = 224,
): Float32Array {
  const tensor = new Float32Array(3 * dstSize * dstSize);
  const planeSize = dstSize * dstSize;
  const halfPrec = 1 << (PRECISION_BITS - 1);

  // 1. Horizontal pass: srcW -> dstSize
  const { bounds: bx, coeffs: cx } = precomputeCoeffsBicubic(srcW, dstSize);
  const temp = new Uint8Array(srcH * dstSize * 3);

  for (let xx = 0; xx < dstSize; xx++) {
    const [xmin, xmax] = bx[xx];
    const k = cx[xx];

    for (let y = 0; y < srcH; y++) {
      let ssR = halfPrec;
      let ssG = halfPrec;
      let ssB = halfPrec;
      const rowOffset = y * srcW;

      for (let x = 0; x < xmax; x++) {
        const idx = (rowOffset + xmin + x) * 4;
        const w = k[x];
        ssR += rgba[idx] * w;
        ssG += rgba[idx + 1] * w;
        ssB += rgba[idx + 2] * w;
      }

      let r = ssR >> PRECISION_BITS;
      if (r < 0) r = 0;
      else if (r > 255) r = 255;

      let g = ssG >> PRECISION_BITS;
      if (g < 0) g = 0;
      else if (g > 255) g = 255;

      let b = ssB >> PRECISION_BITS;
      if (b < 0) b = 0;
      else if (b > 255) b = 255;

      const tempIdx = (y * dstSize + xx) * 3;
      temp[tempIdx] = r;
      temp[tempIdx + 1] = g;
      temp[tempIdx + 2] = b;
    }
  }

  // 2. Vertical pass: srcH -> dstSize
  const { bounds: by, coeffs: cy } = precomputeCoeffsBicubic(srcH, dstSize);

  for (let yy = 0; yy < dstSize; yy++) {
    const [ymin, ymax] = by[yy];
    const k = cy[yy];

    for (let xx = 0; xx < dstSize; xx++) {
      let ssR = halfPrec;
      let ssG = halfPrec;
      let ssB = halfPrec;

      for (let y = 0; y < ymax; y++) {
        const tempIdx = ((ymin + y) * dstSize + xx) * 3;
        const w = k[y];
        ssR += temp[tempIdx] * w;
        ssG += temp[tempIdx + 1] * w;
        ssB += temp[tempIdx + 2] * w;
      }

      let r = ssR >> PRECISION_BITS;
      if (r < 0) r = 0;
      else if (r > 255) r = 255;

      let g = ssG >> PRECISION_BITS;
      if (g < 0) g = 0;
      else if (g > 255) g = 255;

      let b = ssB >> PRECISION_BITS;
      if (b < 0) b = 0;
      else if (b > 255) b = 255;

      // 3. Normalize into ImageNet tensor
      const dstIdx = yy * dstSize + xx;
      const rNorm = r / 255.0;
      const gNorm = g / 255.0;
      const bNorm = b / 255.0;

      tensor[dstIdx] = (rNorm - MEAN[0]) / STD[0];
      tensor[planeSize + dstIdx] = (gNorm - MEAN[1]) / STD[1];
      tensor[2 * planeSize + dstIdx] = (bNorm - MEAN[2]) / STD[2];
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
    tensorStats?: {
      min: number;
      max: number;
      mean: number;
      l1_norm: number;
      sha256_bytes?: string;
    };
    tensor224?: Float32Array;
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
  private readonly candidateModels: FoldCandidateModel[] =
    RESEARCH_CANDIDATE_MODELS;

  /**
   * Initializes the ONNX Runtime Web session from a model ArrayBuffer.
   */
  public async initialize(modelBuffer: ArrayBuffer): Promise<void> {
    ort.env.wasm.wasmPaths = "/wasm/";
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;

    this.session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  }

  public isReady(): boolean {
    return this.session !== null;
  }

  /**
   * Executes end-to-end research inference on 512x512 RGBA pixel buffer.
   */
  public async analyzeImage(
    rgba: Uint8ClampedArray | Uint8Array,
    width = 512,
    height = 512,
  ): Promise<ResearchInferenceResult> {
    if (!this.session) {
      throw new Error("Research ONNX session is not initialized.");
    }

    const tStart = performance.now();

    // 1. DSP extraction on original dimensions
    const t0Dsp = performance.now();
    const dspFeatures = extractCanonicalDspFeatures(rgba, width, height);
    const tDsp = performance.now() - t0Dsp;

    // 2. Preprocessing: bicubic resize to 224x224 and ImageNet normalization
    const t0Pre = performance.now();
    const tensor224 = buildBicubicTensor224(rgba, width, height, 224);
    let tMin = Infinity;
    let tMax = -Infinity;
    let tSum = 0;
    let tL1 = 0;
    for (let i = 0; i < tensor224.length; i++) {
      const v = tensor224[i];
      if (v < tMin) tMin = v;
      if (v > tMax) tMax = v;
      tSum += v;
      tL1 += Math.abs(v);
    }
    let sha256Bytes = "";
    if (typeof crypto !== "undefined" && crypto.subtle) {
      try {
        const hashBuf = await crypto.subtle.digest("SHA-256", tensor224 as unknown as BufferSource);
        const hashArr = Array.from(new Uint8Array(hashBuf));
        sha256Bytes = hashArr.map((b) => b.toString(16).padStart(2, "0")).join("");
      } catch {
        // Fallback
      }
    }

    const tPre = performance.now() - t0Pre;

    // 3. Backbone forward pass in ONNX Runtime Web
    const t0Backbone = performance.now();
    const inputTensor = new ort.Tensor("float32", tensor224, [1, 3, 224, 224]);
    const feeds = { input: inputTensor };
    const outputs = await this.session.run(feeds);
    const visualOutput =
      outputs["visual_features"] || Object.values(outputs)[0];
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
        tensorStats: {
          min: tMin,
          max: tMax,
          mean: tSum / tensor224.length,
          l1_norm: tL1,
          sha256_bytes: sha256Bytes,
        },
        tensor224: tensor224,
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
        note: "Kết quả tổng hợp là giá trị trung bình cộng mô tả 5 outer folds; nghiên cứu chính thức bảo toàn 5 folds độc lập.",
      },
    };
  }

  public async analyzeImage512(
    rgba512: Uint8ClampedArray | Uint8Array,
  ): Promise<ResearchInferenceResult> {
    return this.analyzeImage(rgba512, 512, 512);
  }
}
