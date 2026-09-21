import { describe, it, expect } from 'vitest';
import { JsonReportExporter, PrintableReportGenerator } from '../index.js';
import { AnalysisResult } from '@forensics/shared';

describe('@forensics/report', () => {
  const sampleResult: AnalysisResult = {
    schemaVersion: '1.0.0',
    timestamp: new Date().toISOString(),
    imageInfo: {
      filename: 'audit_image.png',
      fileSizeBytes: 204800,
      mimeType: 'image/png',
      width: 1024,
      height: 768,
      sha256: 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    },
    result: {
      label: 'fully_generated',
      confidence: 0.88,
      probabilities: {
        no_ai_evidence: 0.05,
        fully_generated: 0.88,
        ai_edited: 0.07,
      },
      explanation: 'Ảnh thể hiện cấu trúc tạo sinh toàn phần từ mô hình AI.',
    },
    localization: {
      available: false,
      suspiciousAreaRatio: null,
      regions: [],
    },
    provenance: {
      c2paStatus: 'absent',
      metadataSummary: [
        {
          tag: 'Software',
          value: 'Midjourney v5.2',
          category: 'software',
          suspicionScore: 0.95,
        },
      ],
    },
    forensics: {
      signals: [
        {
          id: 'fft_radial_anomaly',
          name: '2D-FFT Radial Anomaly',
          category: 'frequency',
          score: 0.85,
          interpretation: 'Phát hiện gai tần số chu kỳ.',
          details: {},
        },
      ],
      supportingEvidence: ['Tìm thấy chữ ký Midjourney.'],
      refutingEvidence: [],
    },
    runtime: {
      modelId: 'global-local-v1',
      modelVersion: '0.1.0',
      backend: 'wasm',
      durationMs: 312,
    },
    limitations: ['Hệ thống là công cụ hỗ trợ điều tra...'],
  };

  it('serializes to valid JSON matching the schema', () => {
    const jsonStr = JsonReportExporter.exportToJson(sampleResult);
    const parsed = JSON.parse(jsonStr);

    expect(parsed.schemaVersion).toBe('1.0.0');
    expect(parsed.result.label).toBe('fully_generated');
    expect(parsed.imageInfo.sha256).toBe(sampleResult.imageInfo.sha256);
  });

  it('generates well-formed printable HTML with escaping', () => {
    const html = PrintableReportGenerator.generateHtml(sampleResult);

    expect(html).toContain('<!DOCTYPE html>');
    expect(html).toContain('audit_image.png');
    expect(html).toContain('Tạo hoàn toàn bằng AI (fully_generated)');
    expect(html).toContain('@media print');
  });
});
