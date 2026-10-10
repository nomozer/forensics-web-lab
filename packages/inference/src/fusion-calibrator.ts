import {
  AnalysisVerdict,
  ClassProbabilities,
  CALIBRATION_THRESHOLDS,
  ForensicSummary,
  ProvenanceResult,
  LocalizationResult,
} from "@forensics/shared";

export interface FusionInputs {
  rawLogits?: [number, number, number] | undefined; // [authentic, fully_generated, ai_edited]
  temperature?: number | undefined;
  localization: LocalizationResult;
  forensics: ForensicSummary;
  provenance: ProvenanceResult;
  hasModel: boolean;
}

export interface FusionOutput {
  label: AnalysisVerdict;
  confidence: number | null;
  probabilities: ClassProbabilities | null;
  explanation: string;
  supportingEvidence: string[];
  refutingEvidence: string[];
  modelAvailable: boolean;
  modelStatus: "not-installed" | "installed";
}

export class FusionCalibrator {
  /**
   * Performs calibrated evidence fusion across model, signals, and provenance.
   */
  public static fuse(inputs: FusionInputs): FusionOutput {
    const supporting = [...inputs.forensics.supportingEvidence];
    const refuting = [...inputs.forensics.refutingEvidence];

    // Case 1: Model is not installed or available
    if (!inputs.hasModel || !inputs.rawLogits) {
      return this.fuseWithoutModel(inputs, supporting, refuting);
    }

    // Step 1: Calibrate raw logits via Temperature Scaling
    const T = Math.max(0.1, inputs.temperature ?? 1.0);
    const probs = this.softmax(inputs.rawLogits, T);

    const [pAuth, pGen, pEdit] = probs;
    const maxProb = Math.max(pAuth, pGen, pEdit);

    const suspiciousAreaRatio = inputs.localization.suspiciousAreaRatio ?? 0;
    const hasLocalizedCluster =
      inputs.localization.available &&
      inputs.localization.regions.length > 0 &&
      suspiciousAreaRatio >= CALIBRATION_THRESHOLDS.LOCALIZED_AREA_MIN_RATIO &&
      suspiciousAreaRatio <= CALIBRATION_THRESHOLDS.LOCALIZED_AREA_MAX_RATIO;

    // Check provenance clues
    const hasAiMetadata = inputs.provenance.metadataSummary.some(
      (m) => m.suspicionScore > 0.8,
    );
    if (hasAiMetadata) {
      supporting.push(
        "Tìm thấy chữ ký phần mềm/mô hình AI trong cấu trúc metadata.",
      );
    }

    // Step 2: Arbitration and Uncertainty Handling
    let verdict: AnalysisVerdict = "uncertain";
    let explanation = "";
    const marginDiff = Math.abs(pGen - pAuth);

    // Condition A: Localized inpainting / editing detected via patch scan or model
    if (hasLocalizedCluster || pEdit >= 0.5) {
      verdict = "ai_edited";
      explanation = `Phát hiện các vùng cục bộ có dấu hiệu can thiệp tạo sinh/inpainting (chiếm ~${Math.round(
        suspiciousAreaRatio * 100,
      )}% diện tích ảnh).`;
      supporting.push(
        `Quét không gian cục bộ phát hiện ${inputs.localization.regions.length} cụm nghi vấn có độ bất thường cao.`,
      );
    }
    // Condition B: Below confidence threshold or excessive margin conflict
    else if (
      maxProb < CALIBRATION_THRESHOLDS.UNCERTAIN_CONFIDENCE_THRESHOLD ||
      (pGen > 0.35 &&
        pAuth > 0.35 &&
        marginDiff < CALIBRATION_THRESHOLDS.MARGIN_CONFLICT_THRESHOLD)
    ) {
      verdict = "uncertain";
      explanation =
        "Bằng chứng chưa đủ độ tin cậy thống kê hoặc các tín hiệu nhận diện có sự mâu thuẫn giữa mô hình tạo sinh và ảnh gốc.";
      refuting.push(
        "Mức độ phân tách xác suất giữa các lớp quá hẹp để đưa ra kết luận an toàn.",
      );
    }
    // Condition C: Fully generated synthetic image
    else if (
      pGen > pAuth &&
      pGen >= CALIBRATION_THRESHOLDS.UNCERTAIN_CONFIDENCE_THRESHOLD
    ) {
      verdict = "fully_generated";
      explanation =
        "Dấu vết mô hình không gian và phổ tần số cho thấy xác suất cao ảnh được tạo sinh toàn phần bởi AI.";
      supporting.push(
        "Mô hình mạng nơ-ron nhận diện cấu trúc bề mặt đặc trưng của ảnh tổng hợp.",
      );
    }
    // Condition D: No AI evidence found
    else if (pAuth >= CALIBRATION_THRESHOLDS.UNCERTAIN_CONFIDENCE_THRESHOLD) {
      verdict = "no_ai_evidence";
      explanation =
        "Chưa tìm thấy bằng chứng rõ ràng cho thấy ảnh được tạo hoặc chỉnh sửa bằng AI trong phạm vi nhận biết của hệ thống.";
      refuting.push(
        "Không phát hiện các đặc trưng ô cờ hoặc vi cấu trúc tổng hợp quen thuộc.",
      );
    }

    return {
      label: verdict,
      confidence: Math.round(maxProb * 100) / 100,
      probabilities: {
        no_ai_evidence: Math.round(pAuth * 1000) / 1000,
        fully_generated: Math.round(pGen * 1000) / 1000,
        ai_edited: Math.round(pEdit * 1000) / 1000,
      },
      explanation,
      supportingEvidence: supporting,
      refutingEvidence: refuting,
      modelAvailable: true,
      modelStatus: "installed",
    };
  }

  /**
   * Fallback rule-based fusion when deep model weights are not loaded.
   * Strictly adheres to rule: No fake numbers or mock model inferences.
   * Verdict MUST be 'uncertain', confidence and probabilities MUST be null.
   */
  private static fuseWithoutModel(
    inputs: FusionInputs,
    supporting: string[],
    refuting: string[],
  ): FusionOutput {
    refuting.push(
      "Mô hình học sâu chưa được cài đặt (Model not installed); hệ thống không đưa ra xác suất dự đoán hay nhãn phân loại AI.",
    );

    const hasAiMetadata = inputs.provenance.metadataSummary.some(
      (m) => m.suspicionScore > 0.8,
    );
    const fftSignal = inputs.forensics.signals.find(
      (s) => s.id === "fft_radial_anomaly",
    );
    const noiseSignal = inputs.forensics.signals.find(
      (s) => s.id === "noise_residual_inconsistency",
    );

    if (hasAiMetadata) {
      supporting.push(
        "Tín hiệu khám phá sơ bộ: Metadata chứa định danh công cụ AI (mang tính gợi ý điều tra, không phải kết luận mô hình).",
      );
    }

    if ((fftSignal?.score ?? 0) > 0.7) {
      supporting.push(
        "Tín hiệu khám phá sơ bộ: Phổ tần số 2D-FFT có bất thường năng lượng bán kính cao.",
      );
    }

    if ((noiseSignal?.score ?? 0) > 0.75) {
      supporting.push(
        "Tín hiệu khám phá sơ bộ: Phần dư nhiễu vi mô Laplacian có sự bất đồng nhất cục bộ.",
      );
    }

    return {
      label: "uncertain",
      confidence: null,
      probabilities: null,
      explanation:
        "Mô hình học sâu chưa được cài đặt (Model not installed). Đánh giá dựa trên tín hiệu phân tích DSP và siêu dữ liệu chỉ có tính chất khám phá sơ bộ, không cấu thành kết luận mô hình.",
      supportingEvidence: supporting,
      refutingEvidence: refuting,
      modelAvailable: false,
      modelStatus: "not-installed",
    };
  }

  private static softmax(
    logits: [number, number, number],
    temperature: number,
  ): [number, number, number] {
    const maxVal = Math.max(...logits);
    const exp0 = Math.exp((logits[0] - maxVal) / temperature);
    const exp1 = Math.exp((logits[1] - maxVal) / temperature);
    const exp2 = Math.exp((logits[2] - maxVal) / temperature);
    const sum = exp0 + exp1 + exp2 || 1;
    return [exp0 / sum, exp1 / sum, exp2 / sum];
  }
}
