import React, { useState } from 'react';
import { ForensicSignal } from '@forensics/shared';

interface ForensicsInspectorProps {
  signals: ForensicSignal[];
}

export const ForensicsInspector: React.FC<ForensicsInspectorProps> = ({ signals }) => {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const toggleExpand = (id: string) => {
    setExpandedId(expandedId === id ? null : id);
  };

  return (
    <div className="glass-panel" style={{ padding: '20px', marginBottom: '24px' }}>
      <div style={{ marginBottom: '14px' }}>
        <h3 style={{ fontSize: '16px', fontWeight: '700', margin: 0 }}>
          Chỉ số phân tích miền tần số và nén (DSP Forensics)
        </h3>
        <p style={{ fontSize: '11px', color: 'var(--text-muted)', margin: '4px 0 0 0' }}>
          Tín hiệu khám phá sơ bộ (Exploratory Heuristics) — Chỉ mang tính gợi ý điều tra, không cấu thành kết luận mô hình.
        </p>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {signals.map((sig) => {
          const scorePercent = Math.round(sig.score * 100);
          const isElevated = sig.score > 0.6;
          const isExpanded = expandedId === sig.id;

          return (
            <div
              key={sig.id}
              style={{
                background: 'rgba(15, 23, 42, 0.4)',
                border: '1px solid var(--border-color)',
                borderRadius: 'var(--radius-md)',
                padding: '12px 16px',
                transition: 'border-color 0.2s ease',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  cursor: 'pointer',
                }}
                onClick={() => toggleExpand(sig.id)}
              >
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontWeight: '600', fontSize: '14px' }}>{sig.name}</span>
                    <span
                      style={{
                        fontSize: '11px',
                        padding: '2px 8px',
                        borderRadius: 'var(--radius-full)',
                        background: 'rgba(148, 163, 184, 0.1)',
                        color: 'var(--text-secondary)',
                      }}
                    >
                      {sig.category}
                    </span>
                  </div>
                  <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                    {sig.interpretation}
                  </p>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <div style={{ textAlign: 'right' }}>
                    <span
                      style={{
                        fontWeight: '700',
                        fontSize: '14px',
                        color: isElevated ? 'var(--accent-rose)' : 'var(--accent-emerald)',
                      }}
                    >
                      {scorePercent}%
                    </span>
                    <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Mức nghi vấn</div>
                  </div>
                  <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
                    {isExpanded ? '▲' : '▼'}
                  </span>
                </div>
              </div>

              {isExpanded && (
                <div
                  style={{
                    marginTop: '12px',
                    paddingTop: '10px',
                    borderTop: '1px dashed var(--border-color)',
                    fontSize: '11px',
                    fontFamily: 'var(--font-mono)',
                    color: 'var(--text-secondary)',
                  }}
                >
                  <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                    {JSON.stringify(sig.details, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
