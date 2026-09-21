import React from 'react';
import { ProvenanceResult } from '@forensics/shared';

interface ProvenanceInspectorProps {
  provenance: ProvenanceResult;
}

export const ProvenanceInspector: React.FC<ProvenanceInspectorProps> = ({ provenance }) => {
  const getC2paBadge = () => {
    switch (provenance.c2paStatus) {
      case 'present_valid':
        return { text: 'C2PA Hợp lệ', color: 'var(--accent-emerald)', bg: 'rgba(16, 185, 129, 0.12)' };
      case 'present_invalid':
        return { text: 'C2PA Bị can thiệp / Lỗi', color: 'var(--accent-rose)', bg: 'rgba(244, 63, 94, 0.12)' };
      case 'unsupported':
        return { text: 'C2PA Chưa kích hoạt trên web', color: 'var(--accent-amber)', bg: 'rgba(245, 158, 11, 0.12)' };
      case 'absent':
      default:
        return { text: 'Không có chứng chỉ C2PA', color: 'var(--text-muted)', bg: 'rgba(148, 163, 184, 0.12)' };
    }
  };

  const c2pa = getC2paBadge();

  return (
    <div className="glass-panel" style={{ padding: '20px', marginBottom: '24px' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '16px',
        }}
      >
        <div>
          <h3 style={{ fontSize: '16px', fontWeight: '700' }}>Nguồn gốc & Metadata (Provenance)</h3>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
            Kiểm tra thông tin bản quyền số, EXIF máy ảnh, XMP và chữ ký phần mềm tạo sinh.
          </p>
        </div>

        <span
          style={{
            fontSize: '11px',
            fontWeight: '600',
            padding: '4px 10px',
            borderRadius: 'var(--radius-full)',
            color: c2pa.color,
            background: c2pa.bg,
            border: `1px solid ${c2pa.color}40`,
          }}
        >
          {c2pa.text}
        </span>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border-color)', textAlign: 'left', color: 'var(--text-muted)' }}>
              <th style={{ padding: '8px 12px' }}>Trường metadata</th>
              <th style={{ padding: '8px 12px' }}>Giá trị trích xuất</th>
              <th style={{ padding: '8px 12px' }}>Phân loại</th>
              <th style={{ padding: '8px 12px', textAlign: 'right' }}>Mức nghi vấn AI</th>
            </tr>
          </thead>
          <tbody>
            {provenance.metadataSummary.length > 0 ? (
              provenance.metadataSummary.map((m, idx) => {
                const isSuspicious = m.suspicionScore > 0.7;
                return (
                  <tr
                    key={idx}
                    style={{
                      borderBottom: '1px solid rgba(148, 163, 184, 0.08)',
                      background: isSuspicious ? 'rgba(244, 63, 94, 0.05)' : 'transparent',
                    }}
                  >
                    <td style={{ padding: '10px 12px', fontWeight: '600' }}>{m.tag}</td>
                    <td style={{ padding: '10px 12px', color: 'var(--text-primary)', wordBreak: 'break-all' }}>
                      {m.value}
                    </td>
                    <td style={{ padding: '10px 12px', color: 'var(--text-secondary)' }}>{m.category}</td>
                    <td
                      style={{
                        padding: '10px 12px',
                        textAlign: 'right',
                        fontWeight: '600',
                        color: isSuspicious ? 'var(--accent-rose)' : 'var(--text-muted)',
                      }}
                    >
                      {Math.round(m.suspicionScore * 100)}%
                    </td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan={4} style={{ padding: '16px', textAlign: 'center', color: 'var(--text-muted)' }}>
                  Không tìm thấy metadata EXIF/XMP trong tệp.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
