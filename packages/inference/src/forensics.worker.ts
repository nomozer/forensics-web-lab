import {
  AnalysisProgressEvent,
  AnalysisResult,
  STANDARD_LIMITATIONS,
} from '@forensics/shared';
import { analyzeProvenance } from '@forensics/provenance';
import { analyzeForensicSignals } from '@forensics/forensics';
import { PatchExtractor } from './patch-extractor.js';
import { HeatmapAccumulator, ScoredPatch } from './heatmap-accumulator.js';
import { FusionCalibrator } from './fusion-calibrator.js';
import { OnnxSessionManager } from './session-manager.js';
import * as ort from 'onnxruntime-web';

export interface WorkerAnalyzePayload {
  rgba: Uint8ClampedArray;
  width: number;
  height: number;
  rawBuffer: ArrayBuffer;
  filename: string;
  sha256: string;
  mimeType: string;
  fileSizeBytes: number;
}

let isCancelled = false;

function postProgress(event: AnalysisProgressEvent): void {
  if (typeof self !== 'undefined' && typeof (self as unknown as Worker).postMessage === 'function') {
    self.postMessage({ type: 'PROGRESS', payload: event });
  }
}

if (typeof self !== 'undefined' && typeof (self as unknown as Worker).postMessage === 'function') {
  self.onmessage = async (e: MessageEvent) => {
  const { type, payload } = e.data;

  if (type === 'CANCEL') {
    isCancelled = true;
    postProgress({
      stage: 'cancelled',
      progress: 0,
      message: 'Tiến trình phân tích đã được người dùng hủy bỏ.',
    });
    return;
  }

  if (type === 'START_ANALYSIS') {
    isCancelled = false;
    const startTime = performance.now();
    const data = payload as WorkerAnalyzePayload;

    try {
      // 1. Validating
      postProgress({ stage: 'validating', progress: 0.1, message: 'Đang kiểm tra chữ ký và kích thước ảnh...' });
      if (isCancelled) return;

      // 2. Reading metadata & provenance
      postProgress({ stage: 'reading_metadata', progress: 0.25, message: 'Đang trích xuất metadata EXIF/XMP và C2PA...' });
      const provenance = await analyzeProvenance(data.rawBuffer);
      if (isCancelled) return;

      // 3. Calculating forensic DSP signals
      postProgress({ stage: 'calculating_forensic_signals', progress: 0.45, message: 'Đang tính toán phổ 2D-FFT, biến thiên DCT và nhiễu vi mô...' });
      const forensics = analyzeForensicSignals({
        rgba: data.rgba,
        width: data.width,
        height: data.height,
      });
      if (isCancelled) return;

      // 4. Checking and loading ONNX Model
      postProgress({ stage: 'loading_model', progress: 0.6, message: 'Đang chuẩn bị phiên suy luận ONNX...' });
      const session = OnnxSessionManager.getSession();
      const hasModel = session !== null;

      let globalLogits: [number, number, number] | undefined;
      const scoredPatches: ScoredPatch[] = [];

      if (hasModel) {
        // 5. Running global model
        postProgress({ stage: 'running_global_model', progress: 0.7, message: 'Đang chạy phân loại toàn ảnh trên ONNX...' });
        const globalTensorData = PatchExtractor.extractGlobalTensor(data.rgba, data.width, data.height);
        const inputTensor = new ort.Tensor('float32', globalTensorData, [1, 3, 224, 224]);
        const output = await session.run({ input: inputTensor });
        const outputTensor = Object.values(output)[0] as ort.Tensor;
        const outData = outputTensor.data as Float32Array;
        globalLogits = [outData[0], outData[1], outData[2]];

        // 6. Scanning overlapping patches
        postProgress({ stage: 'scanning_patches', progress: 0.8, message: 'Đang quét bản đồ nhiệt các vùng cục bộ...' });
        const patches = PatchExtractor.extractPatches(data.rgba, data.width, data.height);

        for (let i = 0; i < patches.length; i++) {
          if (isCancelled) return;
          const p = patches[i];
          const patchTensor = new ort.Tensor('float32', p.tensorData, [1, 3, 224, 224]);
          const patchOut = await session.run({ input: patchTensor });
          const patchLogits = (Object.values(patchOut)[0] as ort.Tensor).data as Float32Array;
          // Softmax suspicion of ai_edited + fully_generated
          const exp0 = Math.exp(patchLogits[0]);
          const exp1 = Math.exp(patchLogits[1]);
          const exp2 = Math.exp(patchLogits[2]);
          const sum = exp0 + exp1 + exp2 || 1;
          const suspicion = (exp1 + exp2) / sum;

          scoredPatches.push({ patch: p, score: suspicion });
        }
      }

      // 7. Heatmap accumulation
      const { localization } = HeatmapAccumulator.accumulate(scoredPatches);

      // 8. Calibrating evidence & fusion
      postProgress({ stage: 'calibrating_evidence', progress: 0.9, message: 'Đang hiệu chuẩn và hợp nhất đa nguồn bằng chứng...' });
      const fusion = FusionCalibrator.fuse({
        rawLogits: globalLogits,
        localization,
        forensics,
        provenance,
        hasModel,
      });

      // 9. Building final report
      postProgress({ stage: 'building_report', progress: 0.98, message: 'Đang tạo báo cáo giám định...' });
      const durationMs = Math.round(performance.now() - startTime);

      const finalResult: AnalysisResult = {
        schemaVersion: '1.0.0',
        timestamp: new Date().toISOString(),
        imageInfo: {
          filename: data.filename,
          fileSizeBytes: data.fileSizeBytes,
          mimeType: data.mimeType,
          width: data.width,
          height: data.height,
          sha256: data.sha256,
        },
        modelAvailable: fusion.modelAvailable,
        modelStatus: fusion.modelStatus,
        result: {
          label: fusion.label,
          confidence: fusion.confidence,
          probabilities: fusion.probabilities,
          explanation: fusion.explanation,
        },
        localization,
        provenance,
        forensics: {
          signals: forensics.signals,
          supportingEvidence: fusion.supportingEvidence,
          refutingEvidence: fusion.refutingEvidence,
        },
        runtime: {
          modelId: hasModel ? 'global-local-v1' : 'none',
          modelVersion: hasModel ? '0.1.0' : 'none',
          backend: OnnxSessionManager.getBackend(),
          durationMs,
          deviceInfo: {
            userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : '',
            hardwareConcurrency: typeof navigator !== 'undefined' ? navigator.hardwareConcurrency : 2,
          },
        },
        limitations: [...STANDARD_LIMITATIONS],
      };

      postProgress({ stage: 'done', progress: 1.0, message: 'Hoàn tất phân tích giám định.' });
      self.postMessage({ type: 'COMPLETE', payload: finalResult });
    } catch (error) {
      postProgress({
        stage: 'error',
        progress: 1.0,
        message: `Lỗi trong quá trình phân tích: ${String(error)}`,
      });
      self.postMessage({ type: 'ERROR', payload: String(error) });
    }
  }
};
}
