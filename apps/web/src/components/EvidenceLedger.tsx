import React from 'react';

interface EvidenceLedgerProps {
  supportingEvidence: string[];
  refutingEvidence: string[];
  limitations: string[];
}

export const EvidenceLedger: React.FC<EvidenceLedgerProps> = ({
  supportingEvidence,
  refutingEvidence,
  limitations,
}) => {
  return (
    <div className="glass-panel" style={{ padding: '20px', marginBottom: '24px' }}>
      <h3 style={{ fontSize: '16px', fontWeight: '700', marginBottom: '16px' }}>
        Sổ cái đối soát bằng chứng giám định
      </h3>

      <div className="grid-2col" style={{ marginBottom: '20px' }}>
        <div
          style={{
            background: 'rgba(16, 185, 129, 0.05)',
            border: '1px solid rgba(16, 185, 129, 0.2)',
            borderRadius: 'var(--radius-md)',
            padding: '16px',
          }}
        >
          <h4
            style={{
              fontSize: '13px',
              fontWeight: '700',
              color: 'var(--accent-emerald)',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              marginBottom: '10px',
            }}
          >
            <span>✓</span> Bằng chứng ủng hộ kết luận ({supportingEvidence.length})
          </h4>
          <ul style={{ paddingLeft: '20px', fontSize: '13px', color: 'var(--text-primary)', lineHeight: 1.6 }}>
            {supportingEvidence.length > 0 ? (
              supportingEvidence.map((e, idx) => <li key={idx}>{e}</li>)
            ) : (
              <li style={{ color: 'var(--text-muted)' }}>Chưa ghi nhận bằng chứng ủng hộ đáng kể.</li>
            )}
          </ul>
        </div>

        <div
          style={{
            background: 'rgba(244, 63, 94, 0.05)',
            border: '1px solid rgba(244, 63, 94, 0.2)',
            borderRadius: 'var(--radius-md)',
            padding: '16px',
          }}
        >
          <h4
            style={{
              fontSize: '13px',
              fontWeight: '700',
              color: 'var(--accent-rose)',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              marginBottom: '10px',
            }}
          >
            <span>⚠</span> Bằng chứng phản đối / Cảnh báo mâu thuẫn ({refutingEvidence.length})
          </h4>
          <ul style={{ paddingLeft: '20px', fontSize: '13px', color: 'var(--text-primary)', lineHeight: 1.6 }}>
            {refutingEvidence.length > 0 ? (
              refutingEvidence.map((e, idx) => <li key={idx}>{e}</li>)
            ) : (
              <li style={{ color: 'var(--text-muted)' }}>Không phát hiện tín hiệu phản bác hay mâu thuẫn.</li>
            )}
          </ul>
        </div>
      </div>

      <div
        style={{
          background: 'rgba(15, 23, 42, 0.5)',
          border: '1px solid var(--border-color)',
          borderRadius: 'var(--radius-md)',
          padding: '14px 18px',
        }}
      >
        <h4 style={{ fontSize: '12px', fontWeight: '700', color: 'var(--text-secondary)', marginBottom: '8px' }}>
          Cảnh báo giới hạn kỹ thuật của hệ thống
        </h4>
        <ul style={{ paddingLeft: '20px', fontSize: '12px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
          {limitations.map((l, idx) => (
            <li key={idx}>{l}</li>
          ))}
        </ul>
      </div>
    </div>
  );
};
