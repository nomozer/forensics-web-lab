import React from 'react';
import { AnalysisVerdict, ClassProbabilities } from '@forensics/shared';

interface ResultVerdictCardProps {
  verdict: AnalysisVerdict;
  confidence: number | null;
  probabilities: ClassProbabilities | null;
  explanation: string;
  modelAvailable?: boolean;
  modelStatus?: string;
}

export const ResultVerdictCard: React.FC<ResultVerdictCardProps> = ({
  verdict,
  confidence,
  probabilities,
  explanation,
  modelAvailable = false,
  modelStatus = 'not-installed',
}) => {
  const isModelMissing =
    !modelAvailable ||
    modelStatus === 'not-installed' ||
    confidence === null ||
    probabilities === null;

  const getVerdictStyle = () => {
    if (isModelMissing) {
      return {
        title: 'Chưa có mô hình / Không chắc chắn (uncertain)',
        color: 'var(--accent-purple)',
        bg: 'rgba(139, 92, 246, 0.1)',
        border: 'rgba(139, 92, 246, 0.3)',
        icon: '?',
      };
    }

    switch (verdict) {
      case 'no_ai_evidence':
        return {
          title: 'Chưa tìm thấy bằng chứng AI (no_ai_evidence)',
          color: 'var(--accent-emerald)',
          bg: 'rgba(16, 185, 129, 0.1)',
          border: 'rgba(16, 185, 129, 0.3)',
          icon: '✓',
        };
      case 'fully_generated':
        return {
          title: 'Tạo hoàn toàn bằng AI (fully_generated)',
          color: 'var(--accent-rose)',
          bg: 'rgba(244, 63, 94, 0.1)',
          border: 'rgba(244, 63, 94, 0.3)',
          icon: '⚠',
        };
      case 'ai_edited':
        return {
          title: 'Chỉnh sửa cục bộ bằng AI (ai_edited)',
          color: 'var(--accent-amber)',
          bg: 'rgba(245, 158, 11, 0.1)',
          border: 'rgba(245, 158, 11, 0.3)',
          icon: '✦',
        };
      case 'uncertain':
      default:
        return {
          title: 'Không chắc chắn / Mâu thuẫn (uncertain)',
          color: 'var(--accent-purple)',
          bg: 'rgba(139, 92, 246, 0.1)',
          border: 'rgba(139, 92, 246, 0.3)',
          icon: '?',
        };
    }
  };

  const style = getVerdictStyle();

  return (
    <div
      className="glass-panel"
      style={{
        padding: '24px',
        marginBottom: '24px',
        borderLeft: `4px solid ${style.color}`,
        background: style.bg,
      }}
    >
      {isModelMissing && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '10px 14px',
            marginBottom: '16px',
            borderRadius: 'var(--radius-md)',
            background: 'rgba(245, 158, 11, 0.15)',
            border: '1px solid rgba(245, 158, 11, 0.4)',
            color: 'var(--accent-amber)',
            fontSize: '13px',
            fontWeight: '600',
          }}
        >
          <span>⚠️</span>
          <span>
            <strong>Model not installed</strong> (Mô hình học sâu chưa được cài đặt) — Hệ thống đang chạy ở chế độ giám định khám phá sơ bộ (Exploratory Forensics).
          </span>
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <span
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '12px',
              fontWeight: '700',
              color: style.color,
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
              marginBottom: '4px',
            }}
          >
            <span>{style.icon}</span> Kết luận giám định
          </span>
          <h2 style={{ fontSize: '22px', fontWeight: '800', color: style.color, margin: '2px 0 8px 0' }}>
            {style.title}
          </h2>
          <p style={{ fontSize: '14px', color: 'var(--text-primary)', maxWidth: '800px', lineHeight: 1.5 }}>
            {explanation}
          </p>
        </div>

        <div
          style={{
            background: 'rgba(15, 23, 42, 0.6)',
            border: `1px solid ${style.border}`,
            borderRadius: 'var(--radius-md)',
            padding: '12px 18px',
            textAlign: 'center',
            minWidth: '130px',
          }}
        >
          <div style={{ fontSize: '24px', fontWeight: '800', color: style.color }}>
            {confidence !== null ? `${Math.round(confidence * 100)}%` : 'N/A'}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
            {confidence !== null ? 'Độ tin cậy hiệu chuẩn' : 'Chưa có mô hình'}
          </div>
        </div>
      </div>

      <div
        style={{
          marginTop: '20px',
          paddingTop: '16px',
          borderTop: '1px solid rgba(148, 163, 184, 0.1)',
        }}
      >
        {isModelMissing ? (
          <div
            style={{
              padding: '12px 16px',
              background: 'rgba(15, 23, 42, 0.4)',
              borderRadius: 'var(--radius-md)',
              fontSize: '12px',
              color: 'var(--text-secondary)',
              lineHeight: 1.6,
            }}
          >
            <strong>Lưu ý khoa học:</strong> Xác suất dự đoán 3 lớp bị vô hiệu hóa vì chưa có checkpoint mô hình AI được cài đặt và kiểm định. Các phân tích DSP và siêu dữ liệu bên dưới là các tín hiệu khám phá điều tra sơ bộ, không phải kết luận dự đoán của mạng nơ-ron.
          </div>
        ) : (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
              gap: '16px',
            }}
          >
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Chưa có dấu vết AI</span>
                <span style={{ fontWeight: '600' }}>{Math.round(probabilities.no_ai_evidence * 100)}%</span>
              </div>
              <div className="progress-track" style={{ height: '6px' }}>
                <div
                  style={{
                    height: '100%',
                    width: `${probabilities.no_ai_evidence * 100}%`,
                    background: 'var(--accent-emerald)',
                    borderRadius: 'var(--radius-full)',
                  }}
                ></div>
              </div>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Tạo hoàn toàn bằng AI</span>
                <span style={{ fontWeight: '600' }}>{Math.round(probabilities.fully_generated * 100)}%</span>
              </div>
              <div className="progress-track" style={{ height: '6px' }}>
                <div
                  style={{
                    height: '100%',
                    width: `${probabilities.fully_generated * 100}%`,
                    background: 'var(--accent-rose)',
                    borderRadius: 'var(--radius-full)',
                  }}
                ></div>
              </div>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Chỉnh sửa cục bộ bằng AI</span>
                <span style={{ fontWeight: '600' }}>{Math.round(probabilities.ai_edited * 100)}%</span>
              </div>
              <div className="progress-track" style={{ height: '6px' }}>
                <div
                  style={{
                    height: '100%',
                    width: `${probabilities.ai_edited * 100}%`,
                    background: 'var(--accent-amber)',
                    borderRadius: 'var(--radius-full)',
                  }}
                ></div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
