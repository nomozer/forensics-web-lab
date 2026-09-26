import React from 'react';
import { AnalysisProgressEvent } from '@forensics/shared';

interface AnalysisProgressProps {
  progressEvent: AnalysisProgressEvent;
  onCancel: () => void;
}

const STAGES_ORDER = [
  { key: 'validating', label: 'Xác thực tệp' },
  { key: 'reading_metadata', label: 'Trích xuất metadata' },
  { key: 'loading_model', label: 'Khởi tạo ONNX' },
  { key: 'running_global_model', label: 'Mô hình toàn thể' },
  { key: 'scanning_patches', label: 'Quét bản đồ nhiệt' },
  { key: 'calculating_forensic_signals', label: 'Tín hiệu DSP' },
  { key: 'calibrating_evidence', label: 'Hiệu chuẩn & Hợp nhất' },
  { key: 'building_report', label: 'Tạo báo cáo' },
];

export const AnalysisProgress: React.FC<AnalysisProgressProps> = ({ progressEvent, onCancel }) => {
  const percent = Math.round(progressEvent.progress * 100);

  return (
    <div className="glass-panel" style={{ padding: '24px', marginBottom: '32px' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '16px',
        }}
      >
        <div>
          <h3 style={{ fontSize: '16px', fontWeight: '700', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span
              style={{
                width: '10px',
                height: '10px',
                borderRadius: '50%',
                background: 'var(--accent-cyan)',
                display: 'inline-block',
                boxShadow: '0 0 8px var(--accent-cyan)',
              }}
            ></span>
            Đang phân tích giám định... ({percent}%)
          </h3>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '4px' }}>
            {progressEvent.message}
          </p>
        </div>

        <button className="btn-danger" onClick={onCancel} title="Dừng tiến trình hiện tại">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
          Hủy phân tích
        </button>
      </div>

      <div className="progress-track" style={{ marginBottom: '20px' }}>
        <div className="progress-fill" style={{ width: `${percent}%` }}></div>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
          gap: '8px',
        }}
      >
        {STAGES_ORDER.map((s, idx) => {
          const isCurrent = progressEvent.stage === s.key;
          const stageIndex = STAGES_ORDER.findIndex((x) => x.key === progressEvent.stage);
          const isPassed = stageIndex > idx;

          let color = 'var(--text-muted)';
          let borderColor = 'transparent';
          let icon = '○';

          if (isPassed) {
            color = 'var(--accent-emerald)';
            icon = '✓';
          } else if (isCurrent) {
            color = 'var(--accent-cyan)';
            borderColor = 'rgba(6, 182, 212, 0.3)';
            icon = '●';
          }

          return (
            <div
              key={s.key}
              style={{
                padding: '8px 10px',
                borderRadius: 'var(--radius-sm)',
                background: isCurrent ? 'rgba(6, 182, 212, 0.08)' : 'rgba(15, 23, 42, 0.4)',
                border: `1px solid ${borderColor}`,
                fontSize: '11px',
                color,
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                transition: 'all 0.2s ease',
              }}
            >
              <span>{icon}</span>
              <span style={{ fontWeight: isCurrent ? '600' : '400' }}>{s.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
