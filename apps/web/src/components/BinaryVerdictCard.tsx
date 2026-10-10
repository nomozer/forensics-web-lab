import React, { useState } from 'react';

export interface FoldResult {
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

export interface BinaryInferenceData {
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

interface BinaryVerdictCardProps {
  data: BinaryInferenceData;
  filename: string;
  fileSizeBytes: number;
  sha256: string;
  width: number;
  height: number;
  previewUrl: string;
  onNewImage: () => void;
}

export const BinaryVerdictCard: React.FC<BinaryVerdictCardProps> = ({
  data,
  filename,
  fileSizeBytes,
  sha256,
  width,
  height,
  previewUrl,
  onNewImage,
}) => {
  const [showFolds, setShowFolds] = useState(true);

  const visProb = data.summary.visual_calibrated.mean_probability;
  const fusionProb = data.summary.late_fusion_dsp_augmented.mean_probability;
  const isAiEdited = fusionProb >= 0.5;

  const verdictLabel = isAiEdited ? 'AI EDITED' : 'AUTHENTIC';
  const verdictSubtitle = isAiEdited
    ? 'Phát hiện dấu vết can thiệp chỉnh sửa bằng AI (vượt ngưỡng quyết định 0.50)'
    : 'Nhất quán với ảnh thông thường trong phạm vi phân phối dữ liệu nghiên cứu';

  const badgeColor = isAiEdited ? '#f59e0b' : '#10b981';
  const badgeBg = isAiEdited ? 'rgba(245, 158, 11, 0.15)' : 'rgba(16, 185, 129, 0.15)';
  const badgeBorder = isAiEdited ? 'rgba(245, 158, 11, 0.4)' : 'rgba(16, 185, 129, 0.4)';

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className="glass-panel" style={{ padding: '24px', marginBottom: '24px' }}>
      {/* Top Banner: Verdict & Primary Probabilities */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 1fr) auto',
          gap: '24px',
          alignItems: 'center',
          paddingBottom: '20px',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
            <span
              id="binary-verdict-badge"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                padding: '6px 16px',
                borderRadius: '8px',
                background: badgeBg,
                border: `1px solid ${badgeBorder}`,
                color: badgeColor,
                fontWeight: 800,
                fontSize: '1.25rem',
                letterSpacing: '0.05em',
              }}
            >
              <span>{isAiEdited ? '⚠️' : '✅'}</span>
              <span>{verdictLabel}</span>
            </span>

            <span
              style={{
                fontSize: '0.75rem',
                padding: '4px 8px',
                borderRadius: '6px',
                background: 'rgba(59, 130, 246, 0.15)',
                color: '#60a5fa',
                border: '1px solid rgba(59, 130, 246, 0.3)',
                fontWeight: 600,
              }}
            >
              Live FP32 Backbone (WASM SIMD)
            </span>
          </div>

          <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
            {verdictSubtitle}
          </p>
        </div>

        <button
          id="btn-upload-new-image"
          onClick={onNewImage}
          className="action-button secondary"
          style={{
            padding: '10px 18px',
            borderRadius: '8px',
            cursor: 'pointer',
            fontSize: '0.875rem',
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          <span>📁</span>
          <span>Tải Ảnh Khác</span>
        </button>
      </div>

      {/* Main Grid: Preview & Probabilities */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '260px minmax(0, 1fr)',
          gap: '24px',
          marginTop: '20px',
          alignItems: 'start',
        }}
      >
        {/* Left: Image Thumbnail & Metadata */}
        <div>
          <div
            style={{
              width: '100%',
              height: '240px',
              borderRadius: '8px',
              overflow: 'hidden',
              background: 'rgba(0, 0, 0, 0.4)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '12px',
            }}
          >
            <img
              id="inspected-image-preview"
              src={previewUrl}
              alt={filename}
              style={{
                maxWidth: '100%',
                maxHeight: '100%',
                objectFit: 'contain',
              }}
            />
          </div>

          <div
            style={{
              fontSize: '0.75rem',
              color: 'var(--text-secondary)',
              background: 'rgba(15, 23, 42, 0.5)',
              padding: '10px 12px',
              borderRadius: '6px',
              border: '1px solid rgba(255, 255, 255, 0.05)',
              lineHeight: '1.6',
            }}
          >
            <div style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
              <strong>Tệp:</strong> {filename}
            </div>
            <div>
              <strong>Kích thước:</strong> {width} × {height} px ({formatSize(fileSizeBytes)})
            </div>
            <div
              style={{
                fontFamily: 'var(--font-mono, monospace)',
                fontSize: '0.7rem',
                textOverflow: 'ellipsis',
                overflow: 'hidden',
                whiteSpace: 'nowrap',
              }}
              title={sha256}
            >
              <strong>SHA-256:</strong> {sha256 ? `${sha256.slice(0, 16)}...` : 'N/A'}
            </div>
          </div>
        </div>

        {/* Right: Ensemble Probabilities & Gauges */}
        <div>
          {/* Visual Calibrated Probability */}
          <div
            style={{
              padding: '16px',
              borderRadius: '8px',
              background: 'rgba(15, 23, 42, 0.6)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              marginBottom: '16px',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <div>
                <span style={{ fontWeight: 600, fontSize: '0.95rem', color: '#e2e8f0' }}>
                  Visual Calibrated Probability
                </span>
                <span style={{ marginLeft: '8px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  (MobileNetV3 FP32 576-D Embedding + Temperature Scaling)
                </span>
              </div>
              <span
                id="prob-visual-value"
                style={{
                  fontWeight: 800,
                  fontSize: '1.25rem',
                  fontFamily: 'var(--font-mono, monospace)',
                  color: visProb >= 0.5 ? '#f59e0b' : '#10b981',
                }}
              >
                {(visProb * 100).toFixed(2)}%
              </span>
            </div>

            {/* Gauge bar */}
            <div
              style={{
                position: 'relative',
                height: '10px',
                borderRadius: '5px',
                background: 'rgba(255, 255, 255, 0.1)',
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  width: `${Math.min(100, Math.max(0, visProb * 100))}%`,
                  height: '100%',
                  background: visProb >= 0.5
                    ? 'linear-gradient(90deg, #f59e0b, #ef4444)'
                    : 'linear-gradient(90deg, #10b981, #06b6d4)',
                  transition: 'width 0.4s ease',
                }}
              />
              <div
                style={{
                  position: 'absolute',
                  left: '50%',
                  top: 0,
                  bottom: 0,
                  width: '2px',
                  background: '#ffffff',
                  opacity: 0.7,
                }}
                title="Ngưỡng 50%"
              />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              <span>0% (Authentic)</span>
              <span>Ngưỡng 50%</span>
              <span>100% (AI Edited)</span>
            </div>
          </div>

          {/* Late Fusion DSP Augmented Probability */}
          <div
            style={{
              padding: '16px',
              borderRadius: '8px',
              background: 'rgba(15, 23, 42, 0.6)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              marginBottom: '16px',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <div>
                <span style={{ fontWeight: 600, fontSize: '0.95rem', color: '#e2e8f0' }}>
                  Late Fusion DSP Augmented Probability
                </span>
                <span style={{ marginLeft: '8px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  (Visual 576-D + 16-D Canonical DSP + Stacker)
                </span>
              </div>
              <span
                id="prob-fusion-value"
                style={{
                  fontWeight: 800,
                  fontSize: '1.25rem',
                  fontFamily: 'var(--font-mono, monospace)',
                  color: fusionProb >= 0.5 ? '#f59e0b' : '#10b981',
                }}
              >
                {(fusionProb * 100).toFixed(2)}%
              </span>
            </div>

            {/* Gauge bar */}
            <div
              style={{
                position: 'relative',
                height: '10px',
                borderRadius: '5px',
                background: 'rgba(255, 255, 255, 0.1)',
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  width: `${Math.min(100, Math.max(0, fusionProb * 100))}%`,
                  height: '100%',
                  background: fusionProb >= 0.5
                    ? 'linear-gradient(90deg, #f59e0b, #ef4444)'
                    : 'linear-gradient(90deg, #10b981, #06b6d4)',
                  transition: 'width 0.4s ease',
                }}
              />
              <div
                style={{
                  position: 'absolute',
                  left: '50%',
                  top: 0,
                  bottom: 0,
                  width: '2px',
                  background: '#ffffff',
                  opacity: 0.7,
                }}
                title="Ngưỡng 50%"
              />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              <span>0% (Authentic)</span>
              <span>Ngưỡng 50%</span>
              <span>100% (AI Edited)</span>
            </div>
          </div>

          {/* Timing Stats Bar */}
          <div
            style={{
              display: 'flex',
              gap: '12px',
              flexWrap: 'wrap',
              fontSize: '0.75rem',
              color: 'var(--text-secondary)',
              background: 'rgba(255, 255, 255, 0.02)',
              padding: '10px 14px',
              borderRadius: '6px',
              border: '1px solid rgba(255, 255, 255, 0.05)',
            }}
          >
            <div>
              <strong>Tổng độ trễ:</strong>{' '}
              <span id="timing-total" style={{ color: '#38bdf8', fontFamily: 'monospace' }}>
                {data.timingMs.total} ms
              </span>
            </div>
            <div>•</div>
            <div>
              <strong>Tiền xử lý Bicubic:</strong>{' '}
              <span style={{ fontFamily: 'monospace' }}>{data.timingMs.preprocessing} ms</span>
            </div>
            <div>•</div>
            <div>
              <strong>Backbone ONNX:</strong>{' '}
              <span style={{ fontFamily: 'monospace' }}>{data.timingMs.backbone} ms</span>
            </div>
            <div>•</div>
            <div>
              <strong>Trích xuất DSP:</strong>{' '}
              <span style={{ fontFamily: 'monospace' }}>{data.timingMs.dsp} ms</span>
            </div>
            <div>•</div>
            <div>
              <strong>Chấm điểm 5 Folds:</strong>{' '}
              <span style={{ fontFamily: 'monospace' }}>{data.timingMs.scoring} ms</span>
            </div>
          </div>
        </div>
      </div>

      {/* 5 Outer Folds Transparency Section */}
      <div style={{ marginTop: '24px' }}>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '12px',
          }}
        >
          <div style={{ fontWeight: 600, fontSize: '0.9rem', color: '#cbd5e1' }}>
            Chi Tiết 5 Outer-Fold Checkpoints Độc Lập (Frozen Ensembles):
          </div>
          <button
            onClick={() => setShowFolds(!showFolds)}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#38bdf8',
              fontSize: '0.8rem',
              cursor: 'pointer',
              fontWeight: 500,
            }}
          >
            {showFolds ? 'Ẩn bảng 5 folds ▲' : 'Hiện bảng 5 folds ▼'}
          </button>
        </div>

        {showFolds && (
          <div
            style={{
              overflowX: 'auto',
              borderRadius: '8px',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              background: 'rgba(15, 23, 42, 0.5)',
            }}
          >
            <table
              id="table-5-folds"
              style={{
                width: '100%',
                borderCollapse: 'collapse',
                fontSize: '0.8125rem',
                textAlign: 'left',
              }}
            >
              <thead>
                <tr
                  style={{
                    background: 'rgba(255, 255, 255, 0.03)',
                    borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                    color: 'var(--text-muted)',
                  }}
                >
                  <th style={{ padding: '10px 14px' }}>Fold</th>
                  <th style={{ padding: '10px 14px' }}>Visual Logit</th>
                  <th style={{ padding: '10px 14px' }}>Visual Calib.</th>
                  <th style={{ padding: '10px 14px' }}>P(Visual)</th>
                  <th style={{ padding: '10px 14px' }}>Dự đoán Vis.</th>
                  <th style={{ padding: '10px 14px' }}>Fusion Logit</th>
                  <th style={{ padding: '10px 14px' }}>Fusion Calib.</th>
                  <th style={{ padding: '10px 14px' }}>P(Fusion)</th>
                  <th style={{ padding: '10px 14px' }}>Dự đoán Fusion</th>
                </tr>
              </thead>
              <tbody>
                {data.folds.map((f, idx) => {
                  const visP = f.visual_calibrated.probability;
                  const fusP = f.late_fusion_dsp_augmented.probability;
                  return (
                    <tr
                      key={idx}
                      id={`row-fold-${idx}`}
                      style={{
                        borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                        fontFamily: 'var(--font-mono, monospace)',
                      }}
                    >
                      <td style={{ padding: '10px 14px', fontWeight: 600, color: '#38bdf8' }}>
                        Fold {idx}
                      </td>
                      <td style={{ padding: '10px 14px' }}>{f.visual_calibrated.raw_logit.toFixed(4)}</td>
                      <td style={{ padding: '10px 14px' }}>{f.visual_calibrated.calibrated_logit.toFixed(4)}</td>
                      <td style={{ padding: '10px 14px', color: visP >= 0.5 ? '#f59e0b' : '#10b981' }}>
                        {(visP * 100).toFixed(2)}%
                      </td>
                      <td style={{ padding: '10px 14px' }}>
                        {f.visual_calibrated.prediction === 1 ? 'AI_EDITED' : 'AUTHENTIC'}
                      </td>
                      <td style={{ padding: '10px 14px' }}>{f.late_fusion_dsp_augmented.raw_logit.toFixed(4)}</td>
                      <td style={{ padding: '10px 14px' }}>{f.late_fusion_dsp_augmented.calibrated_logit.toFixed(4)}</td>
                      <td style={{ padding: '10px 14px', color: fusP >= 0.5 ? '#f59e0b' : '#10b981', fontWeight: 600 }}>
                        {(fusP * 100).toFixed(2)}%
                      </td>
                      <td style={{ padding: '10px 14px', fontWeight: 600 }}>
                        {f.late_fusion_dsp_augmented.prediction === 1 ? 'AI_EDITED' : 'AUTHENTIC'}
                      </td>
                    </tr>
                  );
                })}
                {/* Summary Row */}
                <tr
                  style={{
                    background: 'rgba(59, 130, 246, 0.08)',
                    fontFamily: 'var(--font-mono, monospace)',
                    fontWeight: 700,
                  }}
                >
                  <td style={{ padding: '10px 14px', color: '#60a5fa' }}>Mean Ensemble</td>
                  <td style={{ padding: '10px 14px' }} colSpan={2}>-</td>
                  <td style={{ padding: '10px 14px', color: visProb >= 0.5 ? '#f59e0b' : '#10b981' }}>
                    {(visProb * 100).toFixed(2)}%
                  </td>
                  <td style={{ padding: '10px 14px' }}>
                    {visProb >= 0.5 ? 'AI_EDITED' : 'AUTHENTIC'}
                  </td>
                  <td style={{ padding: '10px 14px' }} colSpan={2}>-</td>
                  <td style={{ padding: '10px 14px', color: fusionProb >= 0.5 ? '#f59e0b' : '#10b981' }}>
                    {(fusionProb * 100).toFixed(2)}%
                  </td>
                  <td style={{ padding: '10px 14px' }}>
                    {verdictLabel}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
