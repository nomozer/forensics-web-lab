import React, { useState, useEffect } from 'react';
import { WorkerController } from '@forensics/inference';

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
    fetch('/samples/panel_manifest.json')
      .then((res) => (res.ok ? res.json() : []))
      .then((data) => setSamples(data))
      .catch((err) => console.warn('Could not load samples manifest:', err));
  }, []);

  const workerControllerRef = React.useRef<WorkerController | null>(null);

  const getWorkerController = () => {
    if (!workerControllerRef.current) {
      workerControllerRef.current = new WorkerController();
    }
    return workerControllerRef.current;
  };

  const processRgba512 = async (rgba512: Uint8ClampedArray): Promise<ResearchInferenceData> => {
    const controller = getWorkerController();
    return controller.analyzeResearchImage(
      { rgba512 },
      {
        onProgress: (p) => setProgressMsg(p.message),
      }
    );
  };

  const loadImageToRgba512 = (imgSrc: string): Promise<Uint8ClampedArray> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');
        if (!ctx) return reject(new Error('Canvas 2D context unavailable'));
        ctx.drawImage(img, 0, 0, 512, 512);
        const imgData = ctx.getImageData(0, 0, 512, 512);
        resolve(imgData.data);
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
    setProgressMsg('Đang tải và chuẩn hóa ảnh...');

    try {
      const rgba512 = await loadImageToRgba512(s.rel_url);
      const res = await processRgba512(rgba512);
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
    setProgressMsg('Đang giải mã ảnh người dùng...');

    try {
      const rgba512 = await loadImageToRgba512(url);
      const res = await processRgba512(rgba512);
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
      const controller = getWorkerController();
      for (let i = 0; i < samples.length; i++) {
        const s = samples[i];
        setProgressMsg(`[${i + 1}/${samples.length}] Đang chạy đối chứng cho mẫu ${s.source_id} (${s.label_name})...`);
        const rgba512 = await loadImageToRgba512(s.rel_url);
        const res = await controller.analyzeResearchImage({ rgba512 });

        if (i === 0 && res.initDurationMs) {
          coldStartDurationMs = res.initDurationMs;
        }

        totalLatencies.push(res.timingMs.total);
        backboneLatencies.push(res.timingMs.backbone);
        dspLatencies.push(res.timingMs.dsp);
        scoringLatencies.push(res.timingMs.scoring);
        prepLatencies.push(res.timingMs.preprocessing);

        logs.push({
          sample_index: s.sample_index,
          source_id: s.source_id,
          label_name: s.label_name,
          ground_truth: s.label,
          visual_mean_prob: res.summary.visual_calibrated.mean_probability,
          visual_pred: res.summary.visual_calibrated.prediction,
          fusion_mean_prob: res.summary.late_fusion_dsp_augmented.mean_probability,
          fusion_pred: res.summary.late_fusion_dsp_augmented.prediction,
          timing: res.timingMs,
          folds: res.folds,
        });
      }

      const quantile = (arr: number[], q: number) => {
        const sorted = [...arr].sort((a, b) => a - b);
        return sorted[Math.floor(sorted.length * q)];
      };

      const summary = {
        total_samples: samples.length,
        total_predictions_evaluated: samples.length * 5 * 2, // 160
        cold_start_initialization_ms: coldStartDurationMs,
        latency_total: {
          p50_ms: quantile(totalLatencies, 0.5),
          p95_ms: quantile(totalLatencies, 0.95),
        },
        latency_backbone: {
          p50_ms: quantile(backboneLatencies, 0.5),
          p95_ms: quantile(backboneLatencies, 0.95),
        },
        latency_dsp: {
          p50_ms: quantile(dspLatencies, 0.5),
          p95_ms: quantile(dspLatencies, 0.95),
        },
        latency_scoring: {
          p50_ms: quantile(scoringLatencies, 0.5),
          p95_ms: quantile(scoringLatencies, 0.95),
        },
        latency_preprocessing: {
          p50_ms: quantile(prepLatencies, 0.5),
          p95_ms: quantile(prepLatencies, 0.95),
        },
        timestamp: new Date().toISOString(),
        status: 'BROWSER_PARITY_COMPLETE',
      };

      setParityLogs(logs);
      setParitySummary(summary);

      const exportReceipt = {
        audit_name: 'browser_fp32_onnx_wasm_parity_receipt',
        runtime: 'Chromium Browser WASM Web Worker (Thread=1, SIMD)',
        hardware: navigator.userAgent,
        summary,
        samples: logs,
      };

      if (typeof window !== 'undefined') {
        (window as any).__researchParityReceipt = exportReceipt;
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
    const exportData =
      (window as any).__researchParityReceipt || {
        audit_name: 'browser_fp32_onnx_wasm_parity_receipt',
        runtime: 'Chromium Browser WASM Web Worker',
        summary: paritySummary,
        samples: parityLogs,
      };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `browser_fp32_parity_receipt_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="research-lab-container" style={{ marginTop: '24px' }}>
      <div className="glass-panel" style={{ padding: '24px', marginBottom: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>
              🔬 RQ3–RQ4 Research Lab: ONNX FP32 Web Worker Parity
            </h2>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginTop: '6px' }}>
              Chạy pipeline suy luận nghiên cứu đầy đủ trên trình duyệt: Visual Backbone MobileNetV3-small (ONNX FP32 qua WASM SIMD)
              + 16 đặc trưng DSP chuẩn tắc + Chấm điểm 5 mô hình outer-fold (chuẩn hóa StandardScaler, Temperature Scaling, và Stacker).
            </p>
          </div>
          <button
            onClick={runFullParitySuite}
            disabled={isProcessing || samples.length === 0}
            className="action-button primary"
            style={{
              padding: '10px 20px',
              fontWeight: 600,
              cursor: isProcessing ? 'not-allowed' : 'pointer',
              background: 'linear-gradient(135deg, #3b82f6, #6366f1)',
              color: '#ffffff',
              border: 'none',
              borderRadius: '8px',
            }}
          >
            {isProcessing ? 'Đang chạy kiểm toán...' : '▶ Chạy Toàn Bộ Panel 16 Ảnh'}
          </button>
        </div>

        {/* Panel Sample Selector */}
        <div style={{ marginTop: '20px' }}>
          <div style={{ fontSize: '0.875rem', fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '8px' }}>
            Chọn ảnh từ Panel Phát Triển (16 ảnh khóa trước từ Option P development_train) hoặc tải ảnh lên:
          </div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
            {samples.map((s) => (
              <button
                key={`${s.source_id}_${s.label_name}`}
                onClick={() => handleSelectSample(s)}
                disabled={isProcessing}
                style={{
                  padding: '6px 12px',
                  fontSize: '0.75rem',
                  borderRadius: '6px',
                  border: selectedSample === s ? '1px solid #3b82f6' : '1px solid rgba(255,255,255,0.1)',
                  background: selectedSample === s ? 'rgba(59, 130, 246, 0.2)' : 'rgba(255,255,255,0.03)',
                  color: s.label === 0 ? '#10b981' : '#f59e0b',
                  cursor: 'pointer',
                }}
              >
                {s.label === 0 ? '🟢' : '🟡'} {s.source_id.slice(-6)} ({s.label_name})
              </button>
            ))}
            <label
              style={{
                padding: '6px 14px',
                fontSize: '0.75rem',
                borderRadius: '6px',
                border: '1px dashed rgba(255,255,255,0.3)',
                background: 'rgba(255,255,255,0.05)',
                color: 'var(--text-primary)',
                cursor: 'pointer',
              }}
            >
              📁 Tải file khác...
              <input type="file" accept="image/*" onChange={handleCustomUpload} style={{ display: 'none' }} />
            </label>
          </div>
        </div>
      </div>

      {/* Progress & Error */}
      {isProcessing && (
        <div className="glass-panel" style={{ padding: '16px', marginBottom: '24px', textAlign: 'center' }}>
          <div style={{ color: '#60a5fa', fontWeight: 500 }}>⏳ {progressMsg}</div>
        </div>
      )}

      {errorMsg && (
        <div className="glass-panel" style={{ padding: '16px', marginBottom: '24px', color: '#f87171', borderLeft: '4px solid #ef4444' }}>
          <strong>Lỗi thực thi:</strong> {errorMsg}
        </div>
      )}

      {/* Single Sample Result */}
      {result && (
        <div className="glass-panel" style={{ padding: '24px', marginBottom: '24px' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr', gap: '24px' }}>
            <div>
              {imagePreview && (
                <img
                  src={imagePreview}
                  alt="Sample"
                  style={{ width: '100%', height: '220px', objectFit: 'contain', borderRadius: '8px', background: '#000' }}
                />
              )}
              <div style={{ marginTop: '12px', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                <div>Độ trễ toàn pipeline: <strong>{result.timingMs.total} ms</strong></div>
                <div>• Trích xuất DSP: {result.timingMs.dsp} ms</div>
                <div>• Preprocessing: {result.timingMs.preprocessing} ms</div>
                <div>• ONNX WASM: {result.timingMs.backbone} ms</div>
                <div>• 5-Fold Scoring: {result.timingMs.scoring} ms</div>
              </div>
            </div>

            <div>
              <div style={{ display: 'flex', gap: '16px', marginBottom: '16px' }}>
                <div style={{ flex: 1, padding: '12px', background: 'rgba(255,255,255,0.03)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Nhánh Visual Calibrated (Mean 5 Folds)</div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 600, color: result.summary.visual_calibrated.prediction === 1 ? '#f59e0b' : '#10b981' }}>
                    {result.summary.visual_calibrated.prediction === 1 ? 'AI EDITED' : 'AUTHENTIC'}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    Xác suất: {(result.summary.visual_calibrated.mean_probability * 100).toFixed(2)}%
                  </div>
                </div>

                <div style={{ flex: 1, padding: '12px', background: 'rgba(255,255,255,0.03)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Nhánh Late Fusion DSP Augmented (Mean 5 Folds)</div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 600, color: result.summary.late_fusion_dsp_augmented.prediction === 1 ? '#f59e0b' : '#10b981' }}>
                    {result.summary.late_fusion_dsp_augmented.prediction === 1 ? 'AI EDITED' : 'AUTHENTIC'}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    Xác suất: {(result.summary.late_fusion_dsp_augmented.mean_probability * 100).toFixed(2)}%
                  </div>
                </div>
              </div>

              {/* 5 Outer Folds Table */}
              <div style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '8px', color: 'var(--text-primary)' }}>
                Chi tiết 5 Outer Folds (Không ensemble, bảo toàn contract độc lập):
              </div>
              <table style={{ width: '100%', fontSize: '0.75rem', borderCollapse: 'collapse', textAlign: 'left' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', color: 'var(--text-secondary)' }}>
                    <th style={{ padding: '6px' }}>Fold</th>
                    <th style={{ padding: '6px' }}>Visual Cal Logit</th>
                    <th style={{ padding: '6px' }}>Visual Prob</th>
                    <th style={{ padding: '6px' }}>Visual Pred</th>
                    <th style={{ padding: '6px' }}>Fusion Logit</th>
                    <th style={{ padding: '6px' }}>Fusion Prob</th>
                    <th style={{ padding: '6px' }}>Fusion Pred</th>
                  </tr>
                </thead>
                <tbody>
                  {result.folds.map((f, idx) => (
                    <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                      <td style={{ padding: '6px' }}>Fold {idx}</td>
                      <td style={{ padding: '6px' }}>{f.visual_calibrated.calibrated_logit.toFixed(4)}</td>
                      <td style={{ padding: '6px' }}>{(f.visual_calibrated.probability * 100).toFixed(2)}%</td>
                      <td style={{ padding: '6px', color: f.visual_calibrated.prediction === 1 ? '#f59e0b' : '#10b981' }}>
                        {f.visual_calibrated.prediction === 1 ? 'edited' : 'auth'}
                      </td>
                      <td style={{ padding: '6px' }}>{f.late_fusion_dsp_augmented.calibrated_logit.toFixed(4)}</td>
                      <td style={{ padding: '6px' }}>{(f.late_fusion_dsp_augmented.probability * 100).toFixed(2)}%</td>
                      <td style={{ padding: '6px', color: f.late_fusion_dsp_augmented.prediction === 1 ? '#f59e0b' : '#10b981' }}>
                        {f.late_fusion_dsp_augmented.prediction === 1 ? 'edited' : 'auth'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '8px', fontStyle: 'italic' }}>
                {result.summary.note}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Parity Suite Results */}
      {paritySummary && (
        <div className="glass-panel" style={{ padding: '24px', marginBottom: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#10b981', margin: 0 }}>
              ✅ Kết Quả Kiểm Toán Parity Trên Trình Duyệt: {paritySummary.status}
            </h3>
            <button
              onClick={exportReceiptJson}
              className="action-button"
              style={{
                padding: '6px 14px',
                fontSize: '0.8rem',
                borderRadius: '6px',
                background: 'rgba(16, 185, 129, 0.2)',
                color: '#34d399',
                border: '1px solid rgba(16, 185, 129, 0.4)',
                cursor: 'pointer',
              }}
            >
              📥 Xuất Biên Nhận JSON
            </button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '20px' }}>
            <div style={{ padding: '12px', background: 'rgba(255,255,255,0.03)', borderRadius: '6px' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Tổng mẫu kiểm tra</div>
              <div style={{ fontSize: '1.25rem', fontWeight: 600 }}>{paritySummary.total_samples} ảnh (16)</div>
            </div>
            <div style={{ padding: '12px', background: 'rgba(255,255,255,0.03)', borderRadius: '6px' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Tổng lượt dự đoán</div>
              <div style={{ fontSize: '1.25rem', fontWeight: 600 }}>{paritySummary.total_predictions_evaluated} lượt (160)</div>
            </div>
            <div style={{ padding: '12px', background: 'rgba(255,255,255,0.03)', borderRadius: '6px' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Độ trễ trung vị P50</div>
              <div style={{ fontSize: '1.25rem', fontWeight: 600 }}>{paritySummary.latency_p50_ms} ms</div>
            </div>
            <div style={{ padding: '12px', background: 'rgba(255,255,255,0.03)', borderRadius: '6px' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Độ trễ phân vị P95</div>
              <div style={{ fontSize: '1.25rem', fontWeight: 600 }}>{paritySummary.latency_p95_ms} ms</div>
            </div>
          </div>

          <table style={{ width: '100%', fontSize: '0.75rem', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', color: 'var(--text-secondary)' }}>
                <th style={{ padding: '6px' }}>STT</th>
                <th style={{ padding: '6px' }}>Source ID</th>
                <th style={{ padding: '6px' }}>Label</th>
                <th style={{ padding: '6px' }}>Visual Prob</th>
                <th style={{ padding: '6px' }}>Visual Pred</th>
                <th style={{ padding: '6px' }}>Fusion Prob</th>
                <th style={{ padding: '6px' }}>Fusion Pred</th>
                <th style={{ padding: '6px' }}>Total Time</th>
              </tr>
            </thead>
            <tbody>
              {parityLogs.map((log, idx) => (
                <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                  <td style={{ padding: '6px' }}>{idx + 1}</td>
                  <td style={{ padding: '6px' }}>{log.source_id}</td>
                  <td style={{ padding: '6px', color: log.ground_truth === 0 ? '#10b981' : '#f59e0b' }}>
                    {log.label_name}
                  </td>
                  <td style={{ padding: '6px' }}>{(log.visual_mean_prob * 100).toFixed(2)}%</td>
                  <td style={{ padding: '6px' }}>{log.visual_pred === 1 ? 'edited' : 'auth'}</td>
                  <td style={{ padding: '6px' }}>{(log.fusion_mean_prob * 100).toFixed(2)}%</td>
                  <td style={{ padding: '6px' }}>{log.fusion_pred === 1 ? 'edited' : 'auth'}</td>
                  <td style={{ padding: '6px' }}>{log.timing.total} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
