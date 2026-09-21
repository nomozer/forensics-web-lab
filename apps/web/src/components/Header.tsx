import React from 'react';
import { PRODUCT_NAME, RESEARCH_TITLE } from '@forensics/shared';

export const Header: React.FC = () => {
  return (
    <header style={{ marginBottom: '32px' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          flexWrap: 'wrap',
          gap: '16px',
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div
              style={{
                width: '36px',
                height: '36px',
                borderRadius: '8px',
                background: 'linear-gradient(135deg, #06b6d4, #3b82f6)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 0 15px rgba(6, 182, 212, 0.4)',
              }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffffff" strokeWidth="2">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                <path d="M11 8v6M8 11h6"></path>
              </svg>
            </div>
            <h1 style={{ fontSize: '26px', fontWeight: '800', letterSpacing: '-0.5px' }}>
              {PRODUCT_NAME}
            </h1>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '13px', marginTop: '6px', maxWidth: '750px' }}>
            <strong>Đề tài:</strong> {RESEARCH_TITLE}
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          <span className="badge badge-privacy" title="Ảnh xử lý 100% trong RAM trình duyệt, không gửi qua Internet">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
              <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
            </svg>
            Zero Server Egress
          </span>
          <span className="badge badge-wasm" title="Tăng tốc WebAssembly & WebGPU trên thiết bị">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
            </svg>
            ONNX WASM / WebGPU
          </span>
        </div>
      </div>
    </header>
  );
};
