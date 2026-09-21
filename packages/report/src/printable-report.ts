import { AnalysisResult, PRODUCT_NAME, RESEARCH_TITLE } from '@forensics/shared';

function escapeHtml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

export class PrintableReportGenerator {
  /**
   * Generates a self-contained, printable HTML document string with dedicated print styles.
   */
  public static generateHtml(result: AnalysisResult): string {
    const verdictTitle =
      result.result.label === 'no_ai_evidence'
        ? 'Chưa tìm thấy bằng chứng AI (no_ai_evidence)'
        : result.result.label === 'fully_generated'
          ? 'Tạo hoàn toàn bằng AI (fully_generated)'
          : result.result.label === 'ai_edited'
            ? 'Chỉnh sửa cục bộ bằng AI (ai_edited)'
            : 'Không chắc chắn / Mâu thuẫn (uncertain)';

    const verdictColor =
      result.result.label === 'no_ai_evidence'
        ? '#10b981'
        : result.result.label === 'fully_generated'
          ? '#ef4444'
          : result.result.label === 'ai_edited'
            ? '#f59e0b'
            : '#8b5cf6';

    const metadataRows = result.provenance.metadataSummary
      .map(
        (m) => `
        <tr>
          <td><strong>${escapeHtml(m.tag)}</strong></td>
          <td>${escapeHtml(m.value)}</td>
          <td>${escapeHtml(m.category)}</td>
          <td>${Math.round(m.suspicionScore * 100)}%</td>
        </tr>`
      )
      .join('');

    const signalsRows = result.forensics.signals
      .map(
        (s) => `
        <tr>
          <td><strong>${escapeHtml(s.name)}</strong></td>
          <td>${escapeHtml(s.category)}</td>
          <td>${Math.round(s.score * 100)}%</td>
          <td>${escapeHtml(s.interpretation)}</td>
        </tr>`
      )
      .join('');

    const supportingItems = result.forensics.supportingEvidence
      .map((e) => `<li>${escapeHtml(e)}</li>`)
      .join('');

    const refutingItems = result.forensics.refutingEvidence
      .map((e) => `<li>${escapeHtml(e)}</li>`)
      .join('');

    const limitationsItems = result.limitations
      .map((l) => `<li>${escapeHtml(l)}</li>`)
      .join('');

    const hasModel =
      result.modelAvailable &&
      result.modelStatus !== 'not-installed' &&
      result.result.confidence !== null &&
      result.result.probabilities !== null;

    return `<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>Báo cáo Giám định - ${escapeHtml(result.imageInfo.filename)} - ${PRODUCT_NAME}</title>
  <style>
    @page {
      size: A4;
      margin: 15mm;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      color: #1e293b;
      background: #ffffff;
      line-height: 1.5;
      font-size: 13px;
      margin: 0;
      padding: 20px;
    }
    .header {
      border-bottom: 2px solid #0f172a;
      padding-bottom: 12px;
      margin-bottom: 20px;
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
    }
    .header h1 {
      font-size: 22px;
      margin: 0 0 4px 0;
      color: #0f172a;
    }
    .header .subtitle {
      font-size: 11px;
      color: #64748b;
    }
    .verdict-box {
      border: 2px solid ${verdictColor};
      background: #f8fafc;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 20px;
    }
    .verdict-title {
      font-size: 18px;
      font-weight: bold;
      color: ${verdictColor};
      margin-bottom: 6px;
    }
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px dashed #cbd5e1;
    }
    .metric-card {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 8px;
      text-align: center;
    }
    .metric-value {
      font-size: 16px;
      font-weight: bold;
      color: #0f172a;
    }
    .metric-label {
      font-size: 10px;
      color: #64748b;
      text-transform: uppercase;
    }
    h2 {
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #334155;
      border-bottom: 1px solid #e2e8f0;
      padding-bottom: 4px;
      margin-top: 24px;
      margin-bottom: 10px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 16px;
      font-size: 12px;
    }
    th, td {
      border: 1px solid #e2e8f0;
      padding: 6px 10px;
      text-align: left;
    }
    th {
      background: #f1f5f9;
      color: #475569;
      font-weight: 600;
    }
    ul {
      margin: 0;
      padding-left: 20px;
    }
    li {
      margin-bottom: 4px;
    }
    .evidence-section {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }
    .disclaimer {
      margin-top: 30px;
      padding: 12px;
      background: #f8fafc;
      border-left: 3px solid #64748b;
      font-size: 11px;
      color: #475569;
    }
    @media print {
      body {
        padding: 0;
      }
      .no-print {
        display: none !important;
      }
    }
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1>${PRODUCT_NAME}</h1>
      <div class="subtitle">${RESEARCH_TITLE}</div>
    </div>
    <div style="text-align: right; font-size: 11px; color: #64748b;">
      <div>Thời gian tạo: ${new Date(result.timestamp).toLocaleString('vi-VN')}</div>
      <div>Phiên bản Schema: ${result.schemaVersion}</div>
    </div>
  </div>

  <div class="verdict-box">
    ${
      !hasModel
        ? `<div style="background: #fef3c7; border: 1px solid #f59e0b; color: #b45309; padding: 8px 12px; border-radius: 6px; font-weight: bold; margin-bottom: 10px;">
            ⚠️ Model not installed (Mô hình học sâu chưa được cài đặt) — Báo cáo giám định chạy ở chế độ khám phá sơ bộ.
           </div>`
        : ''
    }
    <div class="verdict-title">${verdictTitle}</div>
    <div>${escapeHtml(result.result.explanation)}</div>

    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-value">${hasModel && result.result.confidence !== null ? `${Math.round(result.result.confidence * 100)}%` : 'N/A'}</div>
        <div class="metric-label">Độ tin cậy hiệu chuẩn</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${hasModel && result.result.probabilities ? `${Math.round(result.result.probabilities.no_ai_evidence * 100)}%` : 'N/A'}</div>
        <div class="metric-label">Chưa có dấu vết AI</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${hasModel && result.result.probabilities ? `${Math.round(result.result.probabilities.fully_generated * 100)}%` : 'N/A'}</div>
        <div class="metric-label">Tạo sinh toàn phần</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${hasModel && result.result.probabilities ? `${Math.round(result.result.probabilities.ai_edited * 100)}%` : 'N/A'}</div>
        <div class="metric-label">Chỉnh sửa cục bộ</div>
      </div>
    </div>
  </div>

  <h2>1. Thông tin tập tin</h2>
  <table>
    <tr><th>Tên file</th><td>${escapeHtml(result.imageInfo.filename)}</td><th>MIME Type</th><td>${escapeHtml(result.imageInfo.mimeType)}</td></tr>
    <tr><th>Kích thước ảnh</th><td>${result.imageInfo.width} × ${result.imageInfo.height} px</td><th>Dung lượng</th><td>${(result.imageInfo.fileSizeBytes / 1024).toFixed(1)} KB</td></tr>
    <tr><th>Mã băm SHA-256</th><td colspan="3" style="font-family: monospace; font-size: 11px;">${result.imageInfo.sha256}</td></tr>
  </table>

  <h2>2. Bằng chứng tổng hợp</h2>
  <div class="evidence-section">
    <div>
      <h3 style="font-size: 12px; color: #15803d; margin-bottom: 6px;">Bằng chứng ủng hộ kết luận</h3>
      <ul>${supportingItems || '<li>Không có bằng chứng ủng hộ đáng kể.</li>'}</ul>
    </div>
    <div>
      <h3 style="font-size: 12px; color: #b91c1c; margin-bottom: 6px;">Bằng chứng phản đối / Cảnh báo</h3>
      <ul>${refutingItems || '<li>Không có điểm phản biện mâu thuẫn.</li>'}</ul>
    </div>
  </div>

  <h2>3. Tín hiệu xử lý tín hiệu số (DSP) & Tần số</h2>
  <table>
    <thead>
      <tr>
        <th style="width: 25%;">Tín hiệu</th>
        <th style="width: 15%;">Nhóm</th>
        <th style="width: 12%;">Chỉ số</th>
        <th>Phân tích & Ý nghĩa</th>
      </tr>
    </thead>
    <tbody>${signalsRows}</tbody>
  </table>

  <h2>4. Metadata & Provenance</h2>
  <table>
    <thead>
      <tr>
        <th style="width: 25%;">Thẻ</th>
        <th>Giá trị</th>
        <th style="width: 15%;">Phân loại</th>
        <th style="width: 12%;">Độ nghi vấn</th>
      </tr>
    </thead>
    <tbody>${metadataRows}</tbody>
  </table>

  <h2>5. Môi trường thực thi cục bộ</h2>
  <table>
    <tr><th>Backend</th><td>${escapeHtml(result.runtime.backend.toUpperCase())}</td><th>Thời gian xử lý</th><td>${result.runtime.durationMs} ms</td></tr>
    <tr><th>Mô hình</th><td>${hasModel ? `${escapeHtml(result.runtime.modelId)} (${escapeHtml(result.runtime.modelVersion)})` : 'Chưa cài đặt (Model not installed)'}</td><th>Thiết bị</th><td>Xử lý 100% trên trình duyệt người dùng (Zero-Egress)</td></tr>
  </table>

  <div class="disclaimer">
    <strong>Tuyên bố miễn trừ trách nhiệm & Giới hạn khoa học:</strong>
    <ul>${limitationsItems}</ul>
  </div>
</body>
</html>`;
  }

  /**
   * Opens print dialog in the browser for saving as PDF or direct printing.
   */
  public static printReport(result: AnalysisResult): void {
    const html = this.generateHtml(result);
    const printWindow = window.open('', '_blank', 'width=900,height=750');
    if (!printWindow) {
      alert('Vui lòng cho phép popup để mở báo cáo in PDF.');
      return;
    }
    printWindow.document.open();
    printWindow.document.write(html);
    printWindow.document.close();
    printWindow.focus();
    setTimeout(() => {
      printWindow.print();
    }, 400);
  }
}
