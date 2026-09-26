import fs from 'node:fs';
import path from 'node:path';
import { describe, it, expect } from 'vitest';
import {
  validateAnalysisResult,
  validateModelRegistry,
  validateEvidenceManifest,
  AnalysisResult,
  ModelRegistry,
  EvidenceManifest,
  CALIBRATION_THRESHOLDS,
} from '../index.js';

describe('@forensics/shared contracts', () => {
  it('validates a valid AnalysisResult object with installed model', () => {
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
      modelAvailable: true,
      modelStatus: 'installed',
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

  it('validates a valid honest no-model AnalysisResult object', () => {
    const noModelResult: AnalysisResult = {
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
        durationMs: 42,
      },
      limitations: ['Hệ thống là công cụ hỗ trợ điều tra...'],
    };

    expect(validateAnalysisResult(noModelResult)).toBe(true);
  });

  it('rejects an invalid no-model state that claims confident AI verdicts or fake probabilities', () => {
    const invalidNoModelResult = {
      schemaVersion: '1.0.0',
      timestamp: new Date().toISOString(),
      imageInfo: {
        filename: 'test.jpg',
        fileSizeBytes: 1000,
        mimeType: 'image/jpeg',
        width: 100,
        height: 100,
        sha256: 'b'.repeat(64),
      },
      modelAvailable: false,
      modelStatus: 'not-installed',
      result: {
        label: 'fully_generated', // VIOLATION: no model cannot conclude fully_generated
        confidence: 0.85,
        probabilities: { no_ai_evidence: 0.1, fully_generated: 0.8, ai_edited: 0.1 },
        explanation: 'Fake verdict',
      },
      localization: { available: false, suspiciousAreaRatio: null, regions: [] },
      provenance: { c2paStatus: 'absent', metadataSummary: [] },
      forensics: { signals: [], supportingEvidence: [], refutingEvidence: [] },
      runtime: { modelId: 'none', modelVersion: 'none', backend: 'none', durationMs: 10 },
      limitations: [],
    };

    expect(validateAnalysisResult(invalidNoModelResult)).toBe(false);
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

  describe('Model Registry Integrity Guard', () => {
    it('validates a correct not-trained registry entry', () => {
      const registry: ModelRegistry = {
        version: '1.0.0',
        models: [
          {
            id: 'global-local-v1',
            version: '0.1.0',
            path: '',
            sha256: '',
            sizeBytes: 0,
            inputShape: [1, 3, 224, 224],
            classes: ['authentic', 'fully_generated', 'ai_edited'],
            opset: 17,
            quantization: 'none',
            status: 'not-trained',
          },
        ],
      };

      const result = validateModelRegistry(registry);
      expect(result.valid).toBe(true);
      expect(result.errors.length).toBe(0);
    });

    it('rejects not-trained entry that declares a fake path or fake positive sizeBytes', () => {
      const registry: ModelRegistry = {
        version: '1.0.0',
        models: [
          {
            id: 'global-local-v1',
            version: '0.1.0',
            path: 'models/fake_weights.onnx',
            sha256: 'a'.repeat(64),
            sizeBytes: 1024,
            inputShape: [1, 3, 224, 224],
            classes: ['authentic', 'fully_generated', 'ai_edited'],
            opset: 17,
            quantization: 'none',
            status: 'not-trained',
          },
        ],
      };

      const result = validateModelRegistry(registry);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.includes('must not declare a fake or non-empty path'))).toBe(true);
    });

    it('rejects ready entry that has empty path, missing sha256, or zero sizeBytes', () => {
      const registry: ModelRegistry = {
        version: '1.0.0',
        models: [
          {
            id: 'ready-model',
            version: '1.0.0',
            path: '',
            sha256: '',
            sizeBytes: 0,
            inputShape: [1, 3, 224, 224],
            classes: ['authentic', 'fully_generated', 'ai_edited'],
            opset: 17,
            quantization: 'none',
            status: 'ready',
          },
        ],
      };

      const result = validateModelRegistry(registry);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.includes("requires a valid non-empty 'path'"))).toBe(true);
      expect(result.errors.some((e) => e.includes("requires a valid 64-character hex 'sha256'"))).toBe(true);
      expect(result.errors.some((e) => e.includes("requires 'sizeBytes' > 0"))).toBe(true);
    });

    it('rejects ready entry when model file does not exist on disk', () => {
      const registry: ModelRegistry = {
        version: '1.0.0',
        models: [
          {
            id: 'ready-model',
            version: '1.0.0',
            path: 'models/non_existent.onnx',
            sha256: 'c'.repeat(64),
            sizeBytes: 12345,
            inputShape: [1, 3, 224, 224],
            classes: ['authentic', 'fully_generated', 'ai_edited'],
            opset: 17,
            quantization: 'none',
            status: 'ready',
          },
        ],
      };

      const mockFsExists = (path: string) => path === 'models/real_file.onnx';
      const result = validateModelRegistry(registry, mockFsExists);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.includes('does not exist on disk'))).toBe(true);
    });

    it('rejects fixture or mock names in production model registry', () => {
      const registry: ModelRegistry = {
        version: '1.0.0',
        models: [
          {
            id: 'mock-detector-test',
            version: '0.1.0',
            path: '',
            sha256: '',
            sizeBytes: 0,
            inputShape: [1, 3, 224, 224],
            classes: ['authentic'],
            opset: 17,
            quantization: 'none',
            status: 'not-trained',
          },
        ],
      };

      const result = validateModelRegistry(registry);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.includes('Test fixtures or mock paths cannot be registered'))).toBe(true);
    });

    it('rejects ready entry with invalid task contract, runtime compatibility, license, or evaluation status', () => {
      const baseReadyModel = {
        id: 'real-model-v1',
        version: '1.0.0',
        path: 'models/real_model.onnx',
        sha256: 'e'.repeat(64),
        sizeBytes: 1048576,
        inputShape: [1, 3, 224, 224],
        classes: ['authentic', 'fully_generated', 'ai_edited'],
        opset: 17,
        quantization: 'none' as const,
        status: 'ready' as const,
        license: 'unverified', // VIOLATION
        runtimeCompatibility: [], // VIOLATION
        evaluationStatus: 'not-evaluated', // VIOLATION
      };

      const registry: ModelRegistry = {
        version: '1.0.0',
        models: [baseReadyModel],
      };

      const result = validateModelRegistry(registry, () => true);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.includes('runtimeCompatibility'))).toBe(true);
      expect(result.errors.some((e) => e.includes('license'))).toBe(true);
      expect(result.errors.some((e) => e.includes('evaluationStatus'))).toBe(true);
    });

    it('accepts ready entry when all 7 criteria are fully satisfied', () => {
      const validReadyRegistry: ModelRegistry = {
        version: '1.0.0',
        models: [
          {
            id: 'real-model-v1',
            version: '1.0.0',
            path: 'models/real_model.onnx',
            sha256: 'f'.repeat(64),
            sizeBytes: 2600000,
            inputShape: [1, 3, 224, 224],
            classes: ['authentic', 'fully_generated', 'ai_edited'],
            opset: 17,
            quantization: 'int8',
            status: 'ready',
            license: 'Apache-2.0',
            runtimeCompatibility: ['onnxruntime-web/wasm', 'onnxruntime-web/webgpu'],
            evaluationStatus: 'evaluated-macro-f1-0.88',
          },
        ],
      };

      const result = validateModelRegistry(validReadyRegistry, () => true);
      expect(result.valid).toBe(true);
      expect(result.errors.length).toBe(0);
    });
  });

  describe('Evidence Manifest Validator', () => {
    const validManifest: EvidenceManifest = {
      schemaVersion: '1.0.0',
      phase: 'Phase 3.6',
      commitAudited: 'cd59136',
      timestampUtc: '2026-09-21T12:00:00Z',
      items: [
        {
          evidenceId: 'EV-GIT-001',
          claim: 'Repository working tree is clean and on feat/production-ai-image-forensics branch',
          category: 'git-state',
          status: 'verified',
          artifact: 'research/evidence/phase-3.6/environment.json',
          reproductionCommand: 'git status --short && git branch --show-current',
          result: 'Working tree clean, branch feat/production-ai-image-forensics',
          limitations: 'Local workspace environment only',
        },
        {
          evidenceId: 'EV-NOMODEL-001',
          claim: 'System outputs uncertain verdict with null confidence and null probabilities when no model is installed',
          category: 'no-model',
          status: 'verified',
          artifact: 'packages/shared/src/__tests__/contracts.test.ts',
          reproductionCommand: 'pnpm test',
          result: 'Contracts enforce honest no-model state and reject fake predictions',
          limitations: 'Applies to UI and report export; does not evaluate trained model accuracy',
        },
      ],
    };

    it('accepts a fully conforming evidence manifest', () => {
      const result = validateEvidenceManifest(validManifest);
      expect(result.valid).toBe(true);
      expect(result.errors.length).toBe(0);
    });

    it('rejects manifest containing absolute machine paths', () => {
      const manifestWithAbsPath = {
        ...validManifest,
        items: [
          {
            ...validManifest.items[0],
            artifact: 'D:\\Documents\\forensics-web-lab\\research\\evidence\\phase-3.6\\environment.json',
          },
        ],
      };

      const result = validateEvidenceManifest(manifestWithAbsPath);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.includes('absolute machine paths prohibited'))).toBe(true);
    });

    it('rejects duplicate evidence IDs', () => {
      const manifestWithDupId = {
        ...validManifest,
        items: [validManifest.items[0], validManifest.items[0]],
      };

      const result = validateEvidenceManifest(manifestWithDupId);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.includes('Duplicate evidenceId'))).toBe(true);
    });

    it('rejects status verified with unmeasured or placeholder results', () => {
      const manifestWithUnmeasuredVerified = {
        ...validManifest,
        items: [
          {
            ...validManifest.items[0],
            status: 'verified' as const,
            result: 'not measured',
          },
        ],
      };

      const result = validateEvidenceManifest(manifestWithUnmeasuredVerified);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.includes('cannot have unmeasured or empty result'))).toBe(true);
    });

    it('validates the real research/evidence/phase-3.6/evidence-manifest.json on disk', () => {
      const manifestPath = path.resolve(__dirname, '../../../../research/evidence/phase-3.6/evidence-manifest.json');
      expect(fs.existsSync(manifestPath)).toBe(true);

      const rawContent = fs.readFileSync(manifestPath, 'utf-8');
      const manifest = JSON.parse(rawContent);
      const result = validateEvidenceManifest(manifest);
      if (!result.valid) {
        console.error('Validation errors:', result.errors);
      }
      expect(result.errors).toEqual([]);
      expect(result.valid).toBe(true);
      expect(manifest.items.length).toBeGreaterThanOrEqual(10);
    });

    it('validates the real research/evidence/phase-4a.0/evidence-manifest.json on disk', () => {
      const manifestPath = path.resolve(__dirname, '../../../../research/evidence/phase-4a.0/evidence-manifest.json');
      expect(fs.existsSync(manifestPath)).toBe(true);

      const rawContent = fs.readFileSync(manifestPath, 'utf-8');
      const manifest = JSON.parse(rawContent);
      const result = validateEvidenceManifest(manifest);
      if (!result.valid) {
        console.error('Validation errors for phase-4a.0:', result.errors);
      }
      expect(result.errors).toEqual([]);
      expect(result.valid).toBe(true);
      expect(manifest.items.length).toBeGreaterThanOrEqual(10);
    });

    it('validates the real research/evidence/phase-4a.1/evidence-manifest.json on disk', () => {
      const manifestPath = path.resolve(__dirname, '../../../../research/evidence/phase-4a.1/evidence-manifest.json');
      expect(fs.existsSync(manifestPath)).toBe(true);

      const rawContent = fs.readFileSync(manifestPath, 'utf-8');
      const manifest = JSON.parse(rawContent);
      const result = validateEvidenceManifest(manifest);
      if (!result.valid) {
        console.error('Validation errors for phase-4a.1:', result.errors);
      }
      expect(result.errors).toEqual([]);
      expect(result.valid).toBe(true);
      expect(manifest.items.length).toBeGreaterThanOrEqual(6);
    });

    it('validates the real research/evidence/phase-4a.2/evidence-manifest.json on disk', () => {
      const manifestPath = path.resolve(__dirname, '../../../../research/evidence/phase-4a.2/evidence-manifest.json');
      expect(fs.existsSync(manifestPath)).toBe(true);

      const rawContent = fs.readFileSync(manifestPath, 'utf-8');
      const manifest = JSON.parse(rawContent);
      const result = validateEvidenceManifest(manifest);
      if (!result.valid) {
        console.error('Validation errors for phase-4a.2:', result.errors);
      }
      expect(result.errors).toEqual([]);
      expect(result.valid).toBe(true);
      expect(manifest.items.length).toBeGreaterThanOrEqual(7);
    });
  });
});


