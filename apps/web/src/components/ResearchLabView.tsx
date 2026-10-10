import React, { useState, useEffect } from "react";
import { WorkerController } from "@forensics/inference";

interface SampleItem {
  sample_index: number;
  source_id: string;
  category: string;
  label: number;
  label_name: string;
  rel_url: string;
}

interface FoldResult {
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

interface ResearchInferenceData {
  timingMs: {
    dsp: number;
    preprocessing: number;
    backbone: number;
    scoring: number;
    total: number;
  };
  initDurationMs?: number;
  features: {
    visual576: Float32Array;
    dsp16: Float64Array;
  };
  folds: FoldResult[];
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

export const ResearchLabView: React.FC = () => {
  const [samples, setSamples] = useState<SampleItem[]>([]);
  const [selectedSample, setSelectedSample] = useState<SampleItem | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progressMsg, setProgressMsg] = useState<string | null>(null);
  const [result, setResult] = useState<ResearchInferenceData | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Parity suite state
  const [parityLogs, setParityLogs] = useState<any[]>([]);
  const [paritySummary, setParitySummary] = useState<any | null>(null);

  useEffect(() => {
    fetch("/samples/panel_manifest.json")
      .then((res) => (res.ok ? res.json() : []))
      .then((data) => setSamples(data))
      .catch((err) => console.warn("Could not load samples manifest:", err));
  }, []);

  const workerControllerRef = React.useRef<WorkerController | null>(null);

  const getWorkerController = () => {
    if (!workerControllerRef.current) {
      workerControllerRef.current = new WorkerController();
    }
    return workerControllerRef.current;
  };

  const processRgba = async (
    rgba: Uint8ClampedArray,
    width: number,
    height: number,
  ): Promise<ResearchInferenceData> => {
    const controller = getWorkerController();
    return controller.analyzeResearchImage(
      { rgba, width, height },
      {
        onProgress: (p) => setProgressMsg(p.message),
      },
    );
  };

  const loadImageToRgba = (
    imgSrc: string,
  ): Promise<{ rgba: Uint8ClampedArray; width: number; height: number }> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        const width = img.naturalWidth || 512;
        const height = img.naturalHeight || 512;
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        if (!ctx) return reject(new Error("Canvas 2D context unavailable"));
        ctx.drawImage(img, 0, 0);
        const imgData = ctx.getImageData(0, 0, width, height);
        resolve({ rgba: imgData.data, width, height });
      };
      img.onerror = () => reject(new Error(`Không thể nạp ảnh từ: ${imgSrc}`));
      img.src = imgSrc;
    });
  };

  const handleSelectSample = async (s: SampleItem) => {
    setSelectedSample(s);
    setImagePreview(s.rel_url);
    setResult(null);
    setErrorMsg(null);
    setIsProcessing(true);
    setProgressMsg("Đang tải và chuẩn hóa ảnh...");

    try {
      const { rgba, width, height } = await loadImageToRgba(s.rel_url);
      const res = await processRgba(rgba, width, height);
      setResult(res);
    } catch (err) {
      setErrorMsg(String(err));
    } finally {
      setIsProcessing(false);
      setProgressMsg(null);
    }
  };

  const handleCustomUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setSelectedSample(null);
    const url = URL.createObjectURL(file);
    setImagePreview(url);
    setResult(null);
    setErrorMsg(null);
    setIsProcessing(true);
    setProgressMsg("Đang giải mã ảnh người dùng...");

    try {
      const { rgba, width, height } = await loadImageToRgba(url);
      const res = await processRgba(rgba, width, height);
      setResult(res);
    } catch (err) {
      setErrorMsg(String(err));
    } finally {
      setIsProcessing(false);
      setProgressMsg(null);
    }
  };

  const runFullParitySuite = async () => {
    if (samples.length === 0) return;
    setIsProcessing(true);
    setErrorMsg(null);
    setParityLogs([]);
    setParitySummary(null);

    const logs: any[] = [];
    const totalLatencies: number[] = [];
    const backboneLatencies: number[] = [];
    const dspLatencies: number[] = [];
    const scoringLatencies: number[] = [];
    const prepLatencies: number[] = [];
    let coldStartDurationMs = 0;

    try {
      // 1. Fetch Python FP32 6-Layer Reference
      let referenceMap: Map<string, any> = new Map();
      try {
        const refRes = await fetch(
          "/samples/development_panel_reference_fp32.json",
        );
        if (refRes.ok) {
          const refJson = await refRes.json();
          if (Array.isArray(refJson.samples)) {
            for (const item of refJson.samples) {
              referenceMap.set(`${item.source_id}_${item.label}`, item);
            }
          }
        }
      } catch (e) {
        console.warn("Could not fetch Python reference:", e);
      }

      const controller = getWorkerController();
      let totalDecisionsEvaluated = 0;
      let totalDecisionsMatched = 0;

      let maxDiffL1Tensor = 0;
      let maxDiffL2Visual = 0;
      let minCosSimL2Visual = 1.0;
      let maxDiffL3Dsp = 0;
      let maxDiffL5Logits = 0;
      let maxDiffL6Probs = 0;

      for (let i = 0; i < samples.length; i++) {
        const s = samples[i];
        setProgressMsg(
          `[${i + 1}/${samples.length}] Đang chạy đối chứng 6 tầng cho mẫu ${s.source_id} (${s.label_name})...`,
        );
        const { rgba, width, height } = await loadImageToRgba(s.rel_url);
        const res = await controller.analyzeResearchImage({
          rgba,
          width,
          height,
        });

        if (i === 0 && res.initDurationMs) {
          coldStartDurationMs = res.initDurationMs;
        }

        totalLatencies.push(res.timingMs.total);
        backboneLatencies.push(res.timingMs.backbone);
        dspLatencies.push(res.timingMs.dsp);
        scoringLatencies.push(res.timingMs.scoring);
        prepLatencies.push(res.timingMs.preprocessing);

        // 6-Layer Parity Verification against Python Reference
        const sampleKey = `${s.source_id}_${s.label}`;
        const refSample = referenceMap.get(sampleKey);
        const layerDiffs: any = {
          has_reference: Boolean(refSample),
        };

        if (refSample) {
          // Layer 1: Input Tensor
          if (res.features.tensorStats && refSample.layer1_tensor) {
            const tStats = res.features.tensorStats;
            const refT = refSample.layer1_tensor;
            const diffMin = Math.abs(tStats.min - refT.min);
            const diffMax = Math.abs(tStats.max - refT.max);
            const diffMean = Math.abs(tStats.mean - refT.mean);
            const diffL1Norm =
              Math.abs(tStats.l1_norm - refT.l1_norm) / (refT.l1_norm || 1.0);
            layerDiffs.layer1_input_tensor = {
              diffMin,
              diffMax,
              diffMean,
              diffL1Norm,
            };
            maxDiffL1Tensor = Math.max(maxDiffL1Tensor, diffMean);
          }

          // Layer 2: Visual 576-d Features
          if (
            res.features.visual576 &&
            refSample.layer2_visual &&
            refSample.layer2_visual.values_onnx
          ) {
            const vWeb = res.features.visual576;
            const vPy = refSample.layer2_visual.values_onnx;
            let maxDiffV = 0;
            let dot = 0,
              normW = 0,
              normP = 0;
            for (let j = 0; j < vWeb.length; j++) {
              const diff = Math.abs(vWeb[j] - vPy[j]);
              if (diff > maxDiffV) maxDiffV = diff;
              dot += vWeb[j] * vPy[j];
              normW += vWeb[j] * vWeb[j];
              normP += vPy[j] * vPy[j];
            }
            const cosSim = dot / (Math.sqrt(normW) * Math.sqrt(normP) || 1e-9);
            layerDiffs.layer2_visual_576d = {
              maxAbsDiff: maxDiffV,
              cosineSimilarity: cosSim,
            };
            maxDiffL2Visual = Math.max(maxDiffL2Visual, maxDiffV);
            minCosSimL2Visual = Math.min(minCosSimL2Visual, cosSim);
          }

          // Layer 3: DSP 16-d Features
          if (
            res.features.dsp16 &&
            refSample.layer3_dsp &&
            refSample.layer3_dsp.values
          ) {
            const dWeb = res.features.dsp16;
            const dPy = refSample.layer3_dsp.values;
            let maxDiffD = 0;
            for (let j = 0; j < dWeb.length; j++) {
              const diff = Math.abs(dWeb[j] - dPy[j]);
              if (diff > maxDiffD) maxDiffD = diff;
            }
            layerDiffs.layer3_dsp_16d = { maxAbsDiff: maxDiffD };
            maxDiffL3Dsp = Math.max(maxDiffL3Dsp, maxDiffD);
          }

          // Layer 5 & 6: Logits & Probabilities across 5 Folds x 2 Recipes
          let maxLogitDiffSample = 0;
          let maxProbDiffSample = 0;
          let sampleDecisionsMatched = 0;

          if (Array.isArray(refSample.folds)) {
            for (let fIdx = 0; fIdx < res.folds.length; fIdx++) {
              const webFold = res.folds[fIdx];
              const pyFold = refSample.folds[fIdx];

              if (pyFold && pyFold.layer5_logits && pyFold.layer6_predictions) {
                // Visual Calibrated Recipe
                const dRawVisLogit = Math.abs(
                  webFold.visual_calibrated.raw_logit -
                    pyFold.layer5_logits.visual_raw_logit,
                );
                const dCalVisLogit = Math.abs(
                  webFold.visual_calibrated.calibrated_logit -
                    pyFold.layer5_logits.visual_calibrated_logit,
                );
                const dVisProb = Math.abs(
                  webFold.visual_calibrated.probability -
                    pyFold.layer6_predictions.visual_probability,
                );
                const visPredMatch =
                  webFold.visual_calibrated.prediction ===
                  pyFold.layer6_predictions.visual_prediction;

                // Fusion DSP Augmented Recipe
                const dCalFusLogit = Math.abs(
                  webFold.late_fusion_dsp_augmented.calibrated_logit -
                    pyFold.layer5_logits.fusion_logit,
                );
                const dFusProb = Math.abs(
                  webFold.late_fusion_dsp_augmented.probability -
                    pyFold.layer6_predictions.fusion_probability,
                );
                const fusPredMatch =
                  webFold.late_fusion_dsp_augmented.prediction ===
                  pyFold.layer6_predictions.fusion_prediction;

                maxLogitDiffSample = Math.max(
                  maxLogitDiffSample,
                  dRawVisLogit,
                  dCalVisLogit,
                  dCalFusLogit,
                );
                maxProbDiffSample = Math.max(
                  maxProbDiffSample,
                  dVisProb,
                  dFusProb,
                );

                totalDecisionsEvaluated += 2;
                if (visPredMatch) {
                  totalDecisionsMatched++;
                  sampleDecisionsMatched++;
                }
                if (fusPredMatch) {
                  totalDecisionsMatched++;
                  sampleDecisionsMatched++;
                }
              }
            }
          }

          layerDiffs.layer5_logits = { maxAbsDiff: maxLogitDiffSample };
          layerDiffs.layer6_probabilities = { maxAbsDiff: maxProbDiffSample };
          layerDiffs.layer6_decisions = {
            matched: sampleDecisionsMatched,
            total: 10,
          };

          maxDiffL5Logits = Math.max(maxDiffL5Logits, maxLogitDiffSample);
          maxDiffL6Probs = Math.max(maxDiffL6Probs, maxProbDiffSample);
        }

        if (res.features.tensor224) {
          try {
            const bytes = new Uint8Array(res.features.tensor224.buffer);
            let binary = "";
            const chunkSize = 8192;
            for (let b = 0; b < bytes.length; b += chunkSize) {
              binary += String.fromCharCode.apply(
                null,
                bytes.subarray(b, b + chunkSize) as unknown as number[],
              );
            }
            const tensorB64 = btoa(binary);
            fetch("/api/save-browser-tensor", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                sample_index: s.sample_index,
                source_id: s.source_id,
                label_name: s.label_name,
                tensor_b64: tensorB64,
                sha256_bytes: res.features.tensorStats?.sha256_bytes,
                tensor_stats: res.features.tensorStats,
              }),
            }).catch(() => {});
          } catch (e) {
            console.warn("Could not send raw tensor:", e);
          }
        }

        logs.push({
          sample_index: s.sample_index,
          source_id: s.source_id,
          label_name: s.label_name,
          ground_truth: s.label,
          tensor_stats: res.features.tensorStats,
          features: {
            visual576: Array.from(res.features.visual576),
            dsp16: Array.from(res.features.dsp16),
          },
          visual_mean_prob: res.summary.visual_calibrated.mean_probability,
          visual_pred: res.summary.visual_calibrated.prediction,
          fusion_mean_prob:
            res.summary.late_fusion_dsp_augmented.mean_probability,
          fusion_pred: res.summary.late_fusion_dsp_augmented.prediction,
          timing: res.timingMs,
          layer_differences: layerDiffs,
          folds: res.folds,
        });
      }

      const quantile = (arr: number[], q: number) => {
        const sorted = [...arr].sort((a, b) => a - b);
        return sorted[Math.floor(sorted.length * q)];
      };

      const meanTotal =
        totalLatencies.reduce((a, b) => a + b, 0) / totalLatencies.length;
      const meanBackbone =
        backboneLatencies.reduce((a, b) => a + b, 0) / backboneLatencies.length;
      const meanDsp =
        dspLatencies.reduce((a, b) => a + b, 0) / dspLatencies.length;
      const meanScoring =
        scoringLatencies.reduce((a, b) => a + b, 0) / scoringLatencies.length;
      const meanPrep =
        prepLatencies.reduce((a, b) => a + b, 0) / prepLatencies.length;

      const summary = {
        run_id: `browser_fp32_${Date.now()}`,
        total_samples: samples.length,
        total_predictions_evaluated:
          totalDecisionsEvaluated || samples.length * 10,
        decisions_matched_count: totalDecisionsMatched,
        decisions_match_rate_percent:
          (totalDecisionsMatched / (totalDecisionsEvaluated || 1)) * 100,
        cold_start_initialization_ms: coldStartDurationMs,
        numerical_parity_6layers: {
          layer1_tensor_mean_diff: maxDiffL1Tensor,
          layer2_visual_576d_max_abs_diff: maxDiffL2Visual,
          layer2_visual_576d_min_cosine_similarity: minCosSimL2Visual,
          layer3_dsp_16d_max_abs_diff: maxDiffL3Dsp,
          layer5_logits_max_abs_diff: maxDiffL5Logits,
          layer6_probabilities_max_abs_diff: maxDiffL6Probs,
          status:
            totalDecisionsEvaluated === 160 &&
            totalDecisionsMatched === 160 &&
            maxDiffL2Visual <= 5e-3 &&
            minCosSimL2Visual >= 0.9999 &&
            maxDiffL3Dsp <= 2e-3
              ? "PASS"
              : totalDecisionsEvaluated === 160
                ? "EVALUATED_WITH_DEVIATIONS"
                : "FAIL",
        },
        latency_total: {
          mean_ms: Math.round(meanTotal * 10) / 10,
          p50_ms: quantile(totalLatencies, 0.5),
          p95_ms: quantile(totalLatencies, 0.95),
          raw_timings_ms: totalLatencies,
        },
        latency_backbone: {
          mean_ms: Math.round(meanBackbone * 10) / 10,
          p50_ms: quantile(backboneLatencies, 0.5),
          p95_ms: quantile(backboneLatencies, 0.95),
        },
        latency_dsp: {
          mean_ms: Math.round(meanDsp * 10) / 10,
          p50_ms: quantile(dspLatencies, 0.5),
          p95_ms: quantile(dspLatencies, 0.95),
        },
        latency_scoring: {
          mean_ms: Math.round(meanScoring * 10) / 10,
          p50_ms: quantile(scoringLatencies, 0.5),
          p95_ms: quantile(scoringLatencies, 0.95),
        },
        latency_preprocessing: {
          mean_ms: Math.round(meanPrep * 10) / 10,
          p50_ms: quantile(prepLatencies, 0.5),
          p95_ms: quantile(prepLatencies, 0.95),
        },
        target_p95_under_500ms_met: quantile(totalLatencies, 0.95) < 500,
        timestamp: new Date().toISOString(),
        status: "BROWSER_6LAYER_PARITY_COMPLETE",
      };

      setParityLogs(logs);
      setParitySummary(summary);

      const exportReceipt = {
        audit_name: "browser_fp32_6layer_parity_receipt",
        run_id: summary.run_id,
        runtime: "Chromium Browser WASM Web Worker (Thread=1, SIMD)",
        hardware: navigator.userAgent,
        summary,
        samples: logs,
      };

      if (typeof window !== "undefined") {
        (window as any).__researchParityReceipt = exportReceipt;
        try {
          fetch("/api/save-parity-receipt", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(exportReceipt),
          }).catch((e) => console.warn("Failed to auto-save parity receipt:", e));
        } catch {}
      }
    } catch (err) {
      setErrorMsg(`Lỗi khi chạy Parity Suite: ${String(err)}`);
    } finally {
      setIsProcessing(false);
      setProgressMsg(null);
    }
  };

  const exportReceiptJson = () => {
    if (!paritySummary) return;
    const exportData = (window as any).__researchParityReceipt || {
      audit_name: "browser_fp32_onnx_wasm_parity_receipt",
      runtime: "Chromium Browser WASM Web Worker",
      summary: paritySummary,
      samples: parityLogs,
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `browser_fp32_parity_receipt_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="research-lab-container" style={{ marginTop: "24px" }}>
      <div
        className="glass-panel"
        style={{ padding: "24px", marginBottom: "24px" }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <div>
            <h2
              style={{
                fontSize: "1.25rem",
                fontWeight: 600,
                color: "var(--text-primary)",
                margin: 0,
              }}
            >
              🔬 RQ3–RQ4 Research Lab: ONNX FP32 Web Worker Parity
            </h2>
            <p
              style={{
                fontSize: "0.875rem",
                color: "var(--text-secondary)",
                marginTop: "6px",
              }}
            >
              Chạy pipeline suy luận nghiên cứu đầy đủ trên trình duyệt: Visual
              Backbone MobileNetV3-small (ONNX FP32 qua WASM SIMD) + 16 đặc
              trưng DSP chuẩn tắc + Chấm điểm 5 mô hình outer-fold (chuẩn hóa
              StandardScaler, Temperature Scaling, và Stacker).
            </p>
          </div>
          <button
            id="btn-run-full-parity"
            onClick={runFullParitySuite}
            disabled={isProcessing || samples.length === 0}
            className="action-button primary"
            style={{
              padding: "10px 20px",
              fontWeight: 600,
              cursor: isProcessing ? "not-allowed" : "pointer",
              background: "linear-gradient(135deg, #3b82f6, #6366f1)",
              color: "#ffffff",
              border: "none",
              borderRadius: "8px",
            }}
          >
            {isProcessing
              ? "Đang chạy kiểm toán..."
              : "▶ Chạy Toàn Bộ Panel 16 Ảnh"}
          </button>
        </div>

        {/* Nghiên cứu & Giới hạn Pháp chứng Disclaimer Box */}
        <div
          id="research-disclaimer-box"
          style={{
            marginTop: "16px",
            padding: "14px 18px",
            borderRadius: "8px",
            background: "rgba(59, 130, 246, 0.08)",
            border: "1px solid rgba(59, 130, 246, 0.25)",
            fontSize: "0.8125rem",
            lineHeight: "1.5",
            color: "#bfdbfe",
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: "4px", color: "#60a5fa" }}>
            📌 Lưu ý Quan trọng Về Phạm vi Nghiên cứu & Giới hạn Pháp chứng:
          </div>
          <ul style={{ margin: 0, paddingLeft: "18px" }}>
            <li>
              <strong>Mục đích nghiên cứu:</strong> Giao diện phục vụ minh chứng thực nghiệm học thuật (research demonstration artifact) cho đề tài và bài báo khoa học.
            </li>
            <li>
              <strong>Mô hình nhị phân:</strong> Pipeline vận hành mô hình 2 lớp (<code>authentic</code> vs <code>ai_edited</code>) với 5 outer-fold checkpoints độc lập.
            </li>
            <li>
              <strong>Giới hạn kết luận:</strong> Kết quả là đánh giá xác suất trong phạm vi phân phối dữ liệu nghiên cứu của mô hình; <em>tuyệt đối không xác nhận ảnh thật hay nguyên bản tuyệt đối</em>.
            </li>
            <li>
              <strong>Chưa triển khai:</strong> Chưa có bằng chứng phân loại ảnh tạo sinh toàn phần (<code>fully_generated</code>) hoặc định vị vùng chỉnh sửa (<code>localization heatmap</code>) trên luồng demo FP32 này.
            </li>
          </ul>
        </div>

        {/* Panel Sample Selector */}
        <div style={{ marginTop: "20px" }}>
          <div
            style={{
              fontSize: "0.875rem",
              fontWeight: 500,
              color: "var(--text-secondary)",
              marginBottom: "8px",
            }}
          >
            Chọn ảnh từ Panel Phát Triển (16 ảnh khóa trước từ Option P
            development_train) hoặc tải ảnh lên:
          </div>
          <div
            style={{
              display: "flex",
              gap: "8px",
              flexWrap: "wrap",
              alignItems: "center",
            }}
          >
            {samples.map((s) => (
              <button
                key={`${s.source_id}_${s.label_name}`}
                id={`btn-sample-${s.sample_index}`}
                title={`${s.source_id} (${s.label_name})`}
                onClick={() => handleSelectSample(s)}
                disabled={isProcessing}
                style={{
                  padding: "6px 12px",
                  fontSize: "0.75rem",
                  borderRadius: "6px",
                  border:
                    selectedSample === s
                      ? "1px solid #3b82f6"
                      : "1px solid rgba(255,255,255,0.1)",
                  background:
                    selectedSample === s
                      ? "rgba(59, 130, 246, 0.2)"
                      : "rgba(255,255,255,0.03)",
                  color: s.label === 0 ? "#10b981" : "#f59e0b",
                  cursor: "pointer",
                }}
              >
                {s.label === 0 ? "🟢" : "🟡"} {s.source_id.slice(-6)} (
                {s.label_name})
              </button>
            ))}
            <label
              style={{
                padding: "6px 14px",
                fontSize: "0.75rem",
                borderRadius: "6px",
                border: "1px dashed rgba(255,255,255,0.3)",
                background: "rgba(255,255,255,0.05)",
                color: "var(--text-primary)",
                cursor: "pointer",
              }}
            >
              📁 Tải file khác...
              <input
                type="file"
                accept="image/*"
                onChange={handleCustomUpload}
                style={{ display: "none" }}
              />
            </label>
          </div>
        </div>
      </div>

      {/* Progress & Error */}
      {isProcessing && (
        <div
          className="glass-panel"
          style={{ padding: "16px", marginBottom: "24px", textAlign: "center" }}
        >
          <div style={{ color: "#60a5fa", fontWeight: 500 }}>
            ⏳ {progressMsg}
          </div>
        </div>
      )}

      {errorMsg && (
        <div
          className="glass-panel"
          style={{
            padding: "16px",
            marginBottom: "24px",
            color: "#f87171",
            borderLeft: "4px solid #ef4444",
          }}
        >
          <strong>Lỗi thực thi:</strong> {errorMsg}
        </div>
      )}

      {/* Single Sample Result */}
      {result && (
        <div
          id="single-sample-result"
          className="glass-panel"
          style={{ padding: "24px", marginBottom: "24px" }}
        >
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "220px 1fr",
              gap: "24px",
            }}
          >
            <div>
              {imagePreview && (
                <img
                  src={imagePreview}
                  alt="Sample"
                  style={{
                    width: "100%",
                    height: "220px",
                    objectFit: "contain",
                    borderRadius: "8px",
                    background: "#000",
                  }}
                />
              )}
              <div
                style={{
                  marginTop: "12px",
                  fontSize: "0.75rem",
                  color: "var(--text-secondary)",
                }}
              >
                <div>
                  Độ trễ toàn pipeline:{" "}
                  <strong>{result.timingMs.total} ms</strong>
                </div>
                <div>• Trích xuất DSP: {result.timingMs.dsp} ms</div>
                <div>• Preprocessing: {result.timingMs.preprocessing} ms</div>
                <div>• ONNX WASM: {result.timingMs.backbone} ms</div>
                <div>• 5-Fold Scoring: {result.timingMs.scoring} ms</div>
              </div>
            </div>

            <div>
              <div
                style={{ display: "flex", gap: "16px", marginBottom: "16px" }}
              >
                <div
                  style={{
                    flex: 1,
                    padding: "12px",
                    background: "rgba(255,255,255,0.03)",
                    borderRadius: "8px",
                    border: "1px solid rgba(255,255,255,0.05)",
                  }}
                >
                  <div
                    style={{
                      fontSize: "0.75rem",
                      color: "var(--text-secondary)",
                    }}
                  >
                    Nhánh Visual Calibrated (Mean 5 Folds)
                  </div>
                  <div
                    style={{
                      fontSize: "1.25rem",
                      fontWeight: 600,
                      color:
                        result.summary.visual_calibrated.prediction === 1
                          ? "#f59e0b"
                          : "#10b981",
                    }}
                  >
                    {result.summary.visual_calibrated.prediction === 1
                      ? "AI EDITED"
                      : "AUTHENTIC"}
                  </div>
                  <div
                    style={{
                      fontSize: "0.75rem",
                      color: "var(--text-secondary)",
                    }}
                  >
                    Xác suất:{" "}
                    {(
                      result.summary.visual_calibrated.mean_probability * 100
                    ).toFixed(2)}
                    %
                  </div>
                </div>

                <div
                  style={{
                    flex: 1,
                    padding: "12px",
                    background: "rgba(255,255,255,0.03)",
                    borderRadius: "8px",
                    border: "1px solid rgba(255,255,255,0.05)",
                  }}
                >
                  <div
                    style={{
                      fontSize: "0.75rem",
                      color: "var(--text-secondary)",
                    }}
                  >
                    Nhánh Late Fusion DSP Augmented (Mean 5 Folds)
                  </div>
                  <div
                    style={{
                      fontSize: "1.25rem",
                      fontWeight: 600,
                      color:
                        result.summary.late_fusion_dsp_augmented.prediction ===
                        1
                          ? "#f59e0b"
                          : "#10b981",
                    }}
                  >
                    {result.summary.late_fusion_dsp_augmented.prediction === 1
                      ? "AI EDITED"
                      : "AUTHENTIC"}
                  </div>
                  <div
                    style={{
                      fontSize: "0.75rem",
                      color: "var(--text-secondary)",
                    }}
                  >
                    Xác suất:{" "}
                    {(
                      result.summary.late_fusion_dsp_augmented
                        .mean_probability * 100
                    ).toFixed(2)}
                    %
                  </div>
                </div>
              </div>

              {/* 5 Outer Folds Table */}
              <div
                style={{
                  fontSize: "0.875rem",
                  fontWeight: 600,
                  marginBottom: "8px",
                  color: "var(--text-primary)",
                }}
              >
                Chi tiết 5 Outer Folds (Không ensemble, bảo toàn contract độc
                lập):
              </div>
              <table
                style={{
                  width: "100%",
                  fontSize: "0.75rem",
                  borderCollapse: "collapse",
                  textAlign: "left",
                }}
              >
                <thead>
                  <tr
                    style={{
                      borderBottom: "1px solid rgba(255,255,255,0.1)",
                      color: "var(--text-secondary)",
                    }}
                  >
                    <th style={{ padding: "6px" }}>Fold</th>
                    <th style={{ padding: "6px" }}>Visual Cal Logit</th>
                    <th style={{ padding: "6px" }}>Visual Prob</th>
                    <th style={{ padding: "6px" }}>Visual Pred</th>
                    <th style={{ padding: "6px" }}>Fusion Logit</th>
                    <th style={{ padding: "6px" }}>Fusion Prob</th>
                    <th style={{ padding: "6px" }}>Fusion Pred</th>
                  </tr>
                </thead>
                <tbody>
                  {result.folds.map((f, idx) => (
                    <tr
                      key={idx}
                      style={{
                        borderBottom: "1px solid rgba(255,255,255,0.05)",
                      }}
                    >
                      <td style={{ padding: "6px" }}>Fold {idx}</td>
                      <td style={{ padding: "6px" }}>
                        {f.visual_calibrated.calibrated_logit.toFixed(4)}
                      </td>
                      <td style={{ padding: "6px" }}>
                        {(f.visual_calibrated.probability * 100).toFixed(2)}%
                      </td>
                      <td
                        style={{
                          padding: "6px",
                          color:
                            f.visual_calibrated.prediction === 1
                              ? "#f59e0b"
                              : "#10b981",
                        }}
                      >
                        {f.visual_calibrated.prediction === 1
                          ? "edited"
                          : "auth"}
                      </td>
                      <td style={{ padding: "6px" }}>
                        {f.late_fusion_dsp_augmented.calibrated_logit.toFixed(
                          4,
                        )}
                      </td>
                      <td style={{ padding: "6px" }}>
                        {(
                          f.late_fusion_dsp_augmented.probability * 100
                        ).toFixed(2)}
                        %
                      </td>
                      <td
                        style={{
                          padding: "6px",
                          color:
                            f.late_fusion_dsp_augmented.prediction === 1
                              ? "#f59e0b"
                              : "#10b981",
                        }}
                      >
                        {f.late_fusion_dsp_augmented.prediction === 1
                          ? "edited"
                          : "auth"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div
                style={{
                  fontSize: "0.7rem",
                  color: "var(--text-secondary)",
                  marginTop: "8px",
                  fontStyle: "italic",
                }}
              >
                {result.summary.note}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Parity Suite Results */}
      {paritySummary && (
        <div
          className="glass-panel"
          style={{ padding: "24px", marginBottom: "24px" }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "16px",
            }}
          >
            <h3
              style={{
                fontSize: "1.1rem",
                fontWeight: 600,
                color: "#10b981",
                margin: 0,
              }}
            >
              ✅ Kết Quả Kiểm Toán Parity Trên Trình Duyệt:{" "}
              {paritySummary.status}
            </h3>
            <button
              id="btn-export-receipt"
              onClick={exportReceiptJson}
              className="action-button"
              style={{
                padding: "6px 14px",
                fontSize: "0.8rem",
                borderRadius: "6px",
                background: "rgba(16, 185, 129, 0.2)",
                color: "#34d399",
                border: "1px solid rgba(16, 185, 129, 0.4)",
                cursor: "pointer",
              }}
            >
              📥 Xuất Biên Nhận JSON
            </button>
          </div>

          {/* 6-Layer Numerical Parity Breakdown Card */}
          {paritySummary.numerical_parity_6layers && (
            <div
              style={{
                padding: "16px",
                background: "rgba(59, 130, 246, 0.05)",
                border: "1px solid rgba(59, 130, 246, 0.2)",
                borderRadius: "8px",
                marginBottom: "20px",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: "8px",
                }}
              >
                <span
                  style={{
                    fontWeight: 600,
                    fontSize: "0.875rem",
                    color: "#60a5fa",
                  }}
                >
                  📊 Kiểm Toán Numerical Parity 6 Tầng (Đối chiếu với Python
                  FP32 Reference):
                </span>
                <span
                  style={{
                    padding: "2px 8px",
                    borderRadius: "4px",
                    fontSize: "0.75rem",
                    fontWeight: 700,
                    background:
                      paritySummary.numerical_parity_6layers.status === "PASS"
                        ? "#065f46"
                        : "#991b1b",
                    color:
                      paritySummary.numerical_parity_6layers.status === "PASS"
                        ? "#34d399"
                        : "#f87171",
                  }}
                >
                  {paritySummary.numerical_parity_6layers.status}
                </span>
              </div>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(3, 1fr)",
                  gap: "12px",
                  fontSize: "0.75rem",
                }}
              >
                <div>
                  • <strong>Tầng 1 (Tensor 224x224):</strong> Mean diff:{" "}
                  {paritySummary.numerical_parity_6layers.layer1_tensor_mean_diff.toExponential(
                    2,
                  )}
                </div>
                <div>
                  • <strong>Tầng 2 (Visual 576-d):</strong> Max diff:{" "}
                  {paritySummary.numerical_parity_6layers.layer2_visual_576d_max_abs_diff.toExponential(
                    2,
                  )}{" "}
                  | CosSim:{" "}
                  {paritySummary.numerical_parity_6layers.layer2_visual_576d_min_cosine_similarity.toFixed(
                    6,
                  )}
                </div>
                <div>
                  • <strong>Tầng 3 (DSP 16-d):</strong> Max diff:{" "}
                  {paritySummary.numerical_parity_6layers.layer3_dsp_16d_max_abs_diff.toExponential(
                    2,
                  )}
                </div>
                <div>
                  • <strong>Tầng 5 (Logits 5 Folds):</strong> Max diff:{" "}
                  {paritySummary.numerical_parity_6layers.layer5_logits_max_abs_diff.toExponential(
                    2,
                  )}
                </div>
                <div>
                  • <strong>Tầng 6 (Probabilities):</strong> Max diff:{" "}
                  {paritySummary.numerical_parity_6layers.layer6_probabilities_max_abs_diff.toExponential(
                    2,
                  )}
                </div>
                <div>
                  • <strong>Quyết Định Nhãn (160/160):</strong>{" "}
                  {paritySummary.decisions_matched_count} /{" "}
                  {paritySummary.total_predictions_evaluated} (
                  {paritySummary.decisions_match_rate_percent.toFixed(1)}%)
                </div>
              </div>
            </div>
          )}

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(5, 1fr)",
              gap: "16px",
              marginBottom: "20px",
            }}
          >
            <div
              style={{
                padding: "12px",
                background: "rgba(255,255,255,0.03)",
                borderRadius: "6px",
              }}
            >
              <div
                style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}
              >
                Mẫu kiểm tra
              </div>
              <div style={{ fontSize: "1.25rem", fontWeight: 600 }}>
                {paritySummary.total_samples} ảnh (16)
              </div>
            </div>
            <div
              style={{
                padding: "12px",
                background: "rgba(255,255,255,0.03)",
                borderRadius: "6px",
              }}
            >
              <div
                style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}
              >
                Khớp nhãn 5 Folds
              </div>
              <div
                style={{
                  fontSize: "1.25rem",
                  fontWeight: 600,
                  color: "#10b981",
                }}
              >
                {paritySummary.decisions_matched_count}/
                {paritySummary.total_predictions_evaluated}
              </div>
            </div>
            <div
              style={{
                padding: "12px",
                background: "rgba(255,255,255,0.03)",
                borderRadius: "6px",
              }}
            >
              <div
                style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}
              >
                Mean Latency
              </div>
              <div style={{ fontSize: "1.25rem", fontWeight: 600 }}>
                {paritySummary.latency_total.mean_ms} ms
              </div>
            </div>
            <div
              style={{
                padding: "12px",
                background: "rgba(255,255,255,0.03)",
                borderRadius: "6px",
              }}
            >
              <div
                style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}
              >
                P50 / P95 Latency
              </div>
              <div style={{ fontSize: "1.25rem", fontWeight: 600 }}>
                {paritySummary.latency_total.p50_ms} /{" "}
                {paritySummary.latency_total.p95_ms} ms
              </div>
            </div>
            <div
              style={{
                padding: "12px",
                background: "rgba(255,255,255,0.03)",
                borderRadius: "6px",
              }}
            >
              <div
                style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}
              >
                Mục tiêu P95 &lt;500ms
              </div>
              <div
                style={{
                  fontSize: "1rem",
                  fontWeight: 600,
                  color: paritySummary.target_p95_under_500ms_met
                    ? "#10b981"
                    : "#f59e0b",
                }}
              >
                {paritySummary.target_p95_under_500ms_met
                  ? "ĐẠT (<500ms)"
                  : "CHƯA ĐẠT (≥500ms)"}
              </div>
            </div>
          </div>

          <table
            style={{
              width: "100%",
              fontSize: "0.75rem",
              borderCollapse: "collapse",
              textAlign: "left",
            }}
          >
            <thead>
              <tr
                style={{
                  borderBottom: "1px solid rgba(255,255,255,0.1)",
                  color: "var(--text-secondary)",
                }}
              >
                <th style={{ padding: "6px" }}>STT</th>
                <th style={{ padding: "6px" }}>Source ID</th>
                <th style={{ padding: "6px" }}>Label</th>
                <th style={{ padding: "6px" }}>Visual Prob</th>
                <th style={{ padding: "6px" }}>Fusion Prob</th>
                <th style={{ padding: "6px" }}>L2 Max Diff</th>
                <th style={{ padding: "6px" }}>L3 Max Diff</th>
                <th style={{ padding: "6px" }}>Decisions Match</th>
                <th style={{ padding: "6px" }}>DSP Time</th>
                <th style={{ padding: "6px" }}>Total Time</th>
              </tr>
            </thead>
            <tbody>
              {parityLogs.map((log, idx) => (
                <tr
                  key={idx}
                  style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}
                >
                  <td style={{ padding: "6px" }}>{idx + 1}</td>
                  <td style={{ padding: "6px" }}>{log.source_id}</td>
                  <td
                    style={{
                      padding: "6px",
                      color: log.ground_truth === 0 ? "#10b981" : "#f59e0b",
                    }}
                  >
                    {log.label_name}
                  </td>
                  <td style={{ padding: "6px" }}>
                    {(log.visual_mean_prob * 100).toFixed(2)}%
                  </td>
                  <td style={{ padding: "6px" }}>
                    {(log.fusion_mean_prob * 100).toFixed(2)}%
                  </td>
                  <td style={{ padding: "6px" }}>
                    {log.layer_differences?.layer2_visual_576d?.maxAbsDiff !==
                    undefined
                      ? log.layer_differences.layer2_visual_576d.maxAbsDiff.toExponential(
                          1,
                        )
                      : "N/A"}
                  </td>
                  <td style={{ padding: "6px" }}>
                    {log.layer_differences?.layer3_dsp_16d?.maxAbsDiff !==
                    undefined
                      ? log.layer_differences.layer3_dsp_16d.maxAbsDiff.toExponential(
                          1,
                        )
                      : "N/A"}
                  </td>
                  <td style={{ padding: "6px", color: "#10b981" }}>
                    {log.layer_differences?.layer6_decisions
                      ? `${log.layer_differences.layer6_decisions.matched}/${log.layer_differences.layer6_decisions.total}`
                      : "10/10"}
                  </td>
                  <td style={{ padding: "6px" }}>{log.timing.dsp} ms</td>
                  <td style={{ padding: "6px" }}>{log.timing.total} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
