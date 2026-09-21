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
    modelAvailable: true,
    modelStatus: 'installed',
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

  const noModelResult: AnalysisResult = {
    schemaVersion: '1.0.0',
    timestamp: new Date().toISOString(),
    imageInfo: {
      filename: 'no_model_sample.jpg',
      fileSizeBytes: 50000,
      mimeType: 'image/jpeg',
      width: 640,
      height: 480,
      sha256: '1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef',
    },
    modelAvailable: false,
    modelStatus: 'not-installed',
    result: {
      label: 'uncertain',
      confidence: null,
      probabilities: null,
      explanation: 'Mô hình học sâu chưa được cài đặt (Model not installed).',
    },
    localization: {
      available: false,
      suspiciousAreaRatio: null,
      regions: [],
    },
    provenance: {
      c2paStatus: 'unsupported',
      metadataSummary: [],
    },
    forensics: {
      signals: [],
      supportingEvidence: [],
      refutingEvidence: ['Model not installed'],
    },
    runtime: {
      modelId: 'none',
      modelVersion: 'none',
      backend: 'none',
      durationMs: 18,
    },
    limitations: ['Hệ thống là công cụ hỗ trợ điều tra...'],
  };

  it('serializes to valid JSON matching the schema for installed model', () => {
    const jsonStr = JsonReportExporter.exportToJson(sampleResult);
    const parsed = JSON.parse(jsonStr);

    expect(parsed.schemaVersion).toBe('1.0.0');
    expect(parsed.modelAvailable).toBe(true);
    expect(parsed.modelStatus).toBe('installed');
    expect(parsed.result.label).toBe('fully_generated');
    expect(parsed.imageInfo.sha256).toBe(sampleResult.imageInfo.sha256);
  });

  it('serializes honest no-model JSON report with null probabilities and modelAvailable=false', () => {
    const jsonStr = JsonReportExporter.exportToJson(noModelResult);
    const parsed = JSON.parse(jsonStr);

    expect(parsed.modelAvailable).toBe(false);
    expect(parsed.modelStatus).toBe('not-installed');
    expect(parsed.result.label).toBe('uncertain');
    expect(parsed.result.confidence).toBeNull();
    expect(parsed.result.probabilities).toBeNull();
  });

  it('generates well-formed printable HTML with escaping for installed model', () => {
    const html = PrintableReportGenerator.generateHtml(sampleResult);

    expect(html).toContain('<!DOCTYPE html>');
    expect(html).toContain('audit_image.png');
    expect(html).toContain('Tạo hoàn toàn bằng AI (fully_generated)');
    expect(html).toContain('@media print');
  });

  it('generates printable HTML with Model not installed banner and N/A values when model not installed', () => {
    const html = PrintableReportGenerator.generateHtml(noModelResult);

    expect(html).toContain('Model not installed');
    expect(html).toContain('N/A');
    expect(html).toContain('Chưa cài đặt (Model not installed)');
  });
});
