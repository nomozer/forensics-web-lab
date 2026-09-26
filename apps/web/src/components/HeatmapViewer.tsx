import React, { useState, useEffect, useRef } from 'react';
import { LocalizationResult } from '@forensics/shared';
import { HeatmapAccumulator } from '@forensics/inference';

interface HeatmapViewerProps {
  previewUrl: string;
  width: number;
  height: number;
  localization: LocalizationResult;
}

export const HeatmapViewer: React.FC<HeatmapViewerProps> = ({
  previewUrl,
  width,
  height,
  localization,
}) => {
  const [showOverlay, setShowOverlay] = useState(true);
  const [opacity, setOpacity] = useState(0.65);
  const [showBoxes, setShowBoxes] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !localization.available || !localization.rawHeatmapGrid) return;

    const canvas = canvasRef.current;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    if (showOverlay) {
      const rgba = HeatmapAccumulator.renderColormapRgba(localization.rawHeatmapGrid, width, height);
      const imgData = ctx.createImageData(width, height);
      imgData.data.set(rgba);
      ctx.putImageData(imgData, 0, 0);
    }
  }, [localization, width, height, showOverlay]);

  return (
    <div className="glass-panel" style={{ padding: '20px', marginBottom: '24px' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '12px',
          marginBottom: '16px',
        }}
      >
        <div>
          <h3 style={{ fontSize: '16px', fontWeight: '700' }}>Bản đồ nhiệt định vị vùng nghi vấn</h3>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
            {localization.available
              ? `Tỷ lệ diện tích nghi vấn: ~${Math.round((localization.suspiciousAreaRatio ?? 0) * 100)}% (${
                  localization.regions.length
                } cụm không gian)`
              : 'Bản đồ nhiệt không khả dụng (Mô hình học sâu chưa được cài đặt — Model not installed)'}
          </p>
        </div>

        {localization.available && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={showOverlay}
                onChange={(e) => setShowOverlay(e.target.checked)}
              />
              Bật lớp phủ Heatmap
            </label>

            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={showBoxes}
                onChange={(e) => setShowBoxes(e.target.checked)}
              />
              Khung viền vùng nghi vấn
            </label>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Độ mờ:</span>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={opacity}
                onChange={(e) => setOpacity(parseFloat(e.target.value))}
                disabled={!showOverlay}
                style={{ width: '90px' }}
              />
              <span style={{ fontSize: '11px', minWidth: '30px' }}>{Math.round(opacity * 100)}%</span>
            </div>
          </div>
        )}
      </div>

      <div
        style={{
          position: 'relative',
          width: '100%',
          maxHeight: '520px',
          overflow: 'hidden',
          borderRadius: 'var(--radius-md)',
          background: '#040711',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          border: '1px solid var(--border-color)',
        }}
      >
        <img
          src={previewUrl}
          alt="Original preview"
          style={{
            maxWidth: '100%',
            maxHeight: '520px',
            objectFit: 'contain',
            display: 'block',
          }}
        />

        {localization.available && (
          <canvas
            ref={canvasRef}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              objectFit: 'contain',
              opacity: showOverlay ? opacity : 0,
              pointerEvents: 'none',
              transition: 'opacity 0.15s ease',
            }}
          />
        )}

        {localization.available &&
          showBoxes &&
          localization.regions.map((reg, idx) => {
            const b = reg.box;
            return (
              <div
                key={idx}
                style={{
                  position: 'absolute',
                  left: `${b.x * 100}%`,
                  top: `${b.y * 100}%`,
                  width: `${b.width * 100}%`,
                  height: `${b.height * 100}%`,
                  border: '2px solid #f59e0b',
                  backgroundColor: 'rgba(245, 158, 11, 0.15)',
                  boxShadow: '0 0 10px rgba(245, 158, 11, 0.5)',
                  pointerEvents: 'none',
                  borderRadius: '4px',
                }}
              >
                <span
                  style={{
                    position: 'absolute',
                    top: '-20px',
                    left: '0',
                    background: '#f59e0b',
                    color: '#000',
                    fontSize: '10px',
                    fontWeight: 'bold',
                    padding: '2px 6px',
                    borderRadius: '3px',
                    whiteSpace: 'nowrap',
                  }}
                >
                  Nghi vấn {Math.round(reg.score * 100)}%
                </span>
              </div>
            );
          })}
      </div>

      <div
        style={{
          marginTop: '12px',
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: '11px',
          color: 'var(--text-muted)',
        }}
      >
        <span>
          Lưu ý: Heatmap hiển thị <em>&quot;vùng mô hình nghi ngờ bất thường&quot;</em>, không phải khẳng định tuyệt đối vùng bị chỉnh sửa.
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          Thang đo: Thấp (Xanh lam)
          <span
            style={{
              display: 'inline-block',
              width: '40px',
              height: '8px',
              background: 'linear-gradient(90deg, #3b82f6, #10b981, #f59e0b, #f43f5e)',
              borderRadius: '2px',
            }}
          ></span>
          Cao (Đỏ/Hồng)
        </span>
      </div>
    </div>
  );
};
