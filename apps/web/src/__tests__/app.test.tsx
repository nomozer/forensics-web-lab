import { describe, it, expect } from 'vitest';
import React from 'react';
import { renderToString } from 'react-dom/server';
import { Header } from '../components/Header.js';
import { ResultVerdictCard } from '../components/ResultVerdictCard.js';

describe('Web UI components', () => {
  it('renders Header with title and zero-egress badge', () => {
    const html = renderToString(<Header />);
    expect(html).toContain('Forensics Web Lab');
    expect(html).toContain('Zero Server Egress');
    expect(html).toContain('ONNX WASM / WebGPU');
  });

  it('renders ResultVerdictCard with no_ai_evidence verdict and explanation', () => {
    const html = renderToString(
      <ResultVerdictCard
        verdict="no_ai_evidence"
        confidence={0.91}
        probabilities={{
          no_ai_evidence: 0.91,
          fully_generated: 0.05,
          ai_edited: 0.04,
        }}
        explanation="Chưa tìm thấy dấu hiệu AI can thiệp."
        modelAvailable={true}
        modelStatus="installed"
      />
    );

    expect(html).toContain('Chưa tìm thấy bằng chứng AI (no_ai_evidence)');
    expect(html).toContain('91%');
    expect(html).toContain('Chưa tìm thấy dấu hiệu AI can thiệp.');
  });

  it('renders ResultVerdictCard with honest no-model banner when model not installed', () => {
    const html = renderToString(
      <ResultVerdictCard
        verdict="uncertain"
        confidence={null}
        probabilities={null}
        explanation="Mô hình học sâu chưa được cài đặt (Model not installed)."
        modelAvailable={false}
        modelStatus="not-installed"
      />
    );

    expect(html).toContain('Model not installed');
    expect(html).toContain('Chưa có mô hình / Không chắc chắn (uncertain)');
    expect(html).toContain('N/A');
    expect(html).toContain('Xác suất dự đoán 3 lớp bị vô hiệu hóa vì chưa có checkpoint mô hình AI');
  });
});
