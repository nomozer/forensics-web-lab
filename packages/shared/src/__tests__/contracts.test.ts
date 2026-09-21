import { describe, it, expect } from 'vitest';
import { validateAnalysisResult, AnalysisResult, CALIBRATION_THRESHOLDS } from '../index.js';

describe('@forensics/shared contracts', () => {
  it('validates a valid AnalysisResult object', () => {
    const sampleResult: AnalysisResult = {
      schemaVersion: '1.0.0',
      timestamp: new Date().toISOString(),
      imageInfo: {
        filename: 'test_sample.jpg',
        fileSizeBytes: 102400,
        mimeType: 'image/jpeg',
        width: 800,
        height: 600,
        sha256: 'a'.repeat(64),
      },
      result: {
        label: 'no_ai_evidence',
        confidence: 0.92,
        probabilities: {
          no_ai_evidence: 0.92,
          fully_generated: 0.05,
          ai_edited: 0.03,
        },
        explanation: 'Không phát hiện thấy dấu vết AI trong các phổ tần số và mô hình.',
      },
      localization: {
        available: false,
        suspiciousAreaRatio: null,
        regions: [],
      },
      provenance: {
        c2paStatus: 'absent',
        metadataSummary: [],
      },
      forensics: {
        signals: [],
        supportingEvidence: [],
        refutingEvidence: [],
      },
      runtime: {
        modelId: 'global-local-v1',
        modelVersion: '0.1.0',
        backend: 'wasm',
        durationMs: 245,
      },
      limitations: ['Hệ thống là công cụ hỗ trợ điều tra...'],
    };

    expect(validateAnalysisResult(sampleResult)).toBe(true);
  });

  it('rejects invalid objects', () => {
    expect(validateAnalysisResult(null)).toBe(false);
    expect(validateAnalysisResult({})).toBe(false);
    expect(validateAnalysisResult({ schemaVersion: '2.0.0' })).toBe(false);
  });

  it('verifies calibration thresholds are bounded in [0, 1]', () => {
    expect(CALIBRATION_THRESHOLDS.UNCERTAIN_CONFIDENCE_THRESHOLD).toBeGreaterThan(0);
    expect(CALIBRATION_THRESHOLDS.UNCERTAIN_CONFIDENCE_THRESHOLD).toBeLessThan(1);
    expect(CALIBRATION_THRESHOLDS.HIGH_CONFIDENCE_THRESHOLD).toBeGreaterThan(
      CALIBRATION_THRESHOLDS.UNCERTAIN_CONFIDENCE_THRESHOLD
    );
  });
});
