import React from 'react';
import { AnalysisResult } from '@forensics/shared';
import { JsonReportExporter, PrintableReportGenerator } from '@forensics/report';

interface ReportActionsProps {
  result: AnalysisResult;
  onClearSession: () => void;
  onNewImage: () => void;
}

export const ReportActions: React.FC<ReportActionsProps> = ({
  result,
  onClearSession,
  onNewImage,
}) => {
  const handleDownloadJson = () => {
    JsonReportExporter.downloadJson(result, 'forensics_web_lab');
  };

  const handlePrint = () => {
    PrintableReportGenerator.printReport(result);
  };

  return (
    <div
      className="glass-panel"
      style={{
        padding: '20px',
        marginBottom: '40px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '16px',
      }}
    >
      <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
        <button className="btn-primary" onClick={handleDownloadJson} title="Tải xuống báo cáo định dạng JSON chuẩn v1.0.0">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="7 10 12 15 17 10"></polyline>
            <line x1="12" y1="15" x2="12" y2="3"></line>
          </svg>
          Xuất Báo cáo JSON
        </button>

        <button className="btn-secondary" onClick={handlePrint} title="In trực tiếp hoặc Lưu thành PDF từ trình duyệt">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="6 9 6 2 18 2 18 9"></polyline>
            <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"></path>
            <rect x="6" y="14" width="12" height="8"></rect>
          </svg>
          In / Lưu PDF
        </button>
      </div>

      <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
        <button className="btn-secondary" onClick={onNewImage} title="Phân tích một ảnh mới">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="12" y1="5" x2="12" y2="19"></line>
            <line x1="5" y1="12" x2="19" y2="12"></line>
          </svg>
          Giám định ảnh khác
        </button>

        <button
          className="btn-danger"
          onClick={onClearSession}
          title="Xóa toàn bộ ảnh và dữ liệu phân tích khỏi bộ nhớ trình duyệt"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="3 6 5 6 21 6"></polyline>
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
          </svg>
          Xóa sạch phiên
        </button>
      </div>
    </div>
  );
};
