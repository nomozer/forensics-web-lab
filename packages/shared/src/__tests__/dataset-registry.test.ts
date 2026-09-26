import fs from 'node:fs';
import path from 'node:path';
import { describe, it, expect } from 'vitest';
import {
  validateDatasetRegistry,
  validateProductionModelLineage,
  DatasetRegistry,
} from '../index.js';

describe('Dataset Registry & Contamination Guards (ADR-0006)', () => {
  it('validates the official datasets/registry.json on disk', () => {
    const registryPath = path.resolve(__dirname, '../../../../datasets/registry.json');
    expect(fs.existsSync(registryPath)).toBe(true);

    const raw = fs.readFileSync(registryPath, 'utf-8');
    const registry = JSON.parse(raw);
    const result = validateDatasetRegistry(registry);

    expect(result.valid).toBe(true);
    expect(result.errors).toEqual([]);
    expect(registry.datasets.length).toBeGreaterThanOrEqual(4);
  });

  describe('Validation Rule Rejections', () => {
    const baseValidRegistry: DatasetRegistry = {
      schemaVersion: '1.0.0',
      datasets: [
        {
          id: 'test-dataset',
          name: 'Test Dataset',
          version: '1.0.0',
          track: 'research-only',
          purpose: 'scientific-benchmark',
          officialSource: 'https://example.org/test',
          downloadSource: 'https://example.org/test/download',
          codeLicense: 'MIT',
          datasetLicense: 'Research-Only',
          additionalTerms: ['Non-commercial'],
          commercialUse: 'prohibited',
          redistribution: 'unclear',
          derivativeWeights: 'prohibited',
          availability: 'available',
          expectedDownloadBytes: 1000,
          checksumAvailable: true,
          licenseEvidenceUrls: ['https://example.org/license'],
          reviewedAt: '2026-09-21T00:00:00Z',
          status: 'verified',
          notes: 'Test dataset',
        },
      ],
    };

    it('rejects product-eligible dataset when commercialUse is prohibited or unclear', () => {
      const invalid = {
        ...baseValidRegistry,
        datasets: [
          {
            ...baseValidRegistry.datasets[0],
            id: 'bad-product-data',
            track: 'product-eligible' as const,
            purpose: 'product-training' as const,
            commercialUse: 'prohibited' as const,
          },
        ],
      };

      const result = validateDatasetRegistry(invalid);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.includes("commercialUse: 'allowed'"))).toBe(true);
    });

    it('rejects product-eligible dataset when derivativeWeights is prohibited', () => {
      const invalid = {
        ...baseValidRegistry,
        datasets: [
          {
            ...baseValidRegistry.datasets[0],
            id: 'bad-product-weights',
            track: 'product-eligible' as const,
            purpose: 'product-training' as const,
            commercialUse: 'allowed' as const,
            derivativeWeights: 'prohibited' as const,
          },
        ],
      };

      const result = validateDatasetRegistry(invalid);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.includes("derivativeWeights: 'allowed'"))).toBe(true);
    });

    it('rejects product-eligible dataset when derivativeWeights is unclear', () => {
      const invalid = {
        ...baseValidRegistry,
        datasets: [
          {
            ...baseValidRegistry.datasets[0],
            id: 'bad-product-unclear-weights',
            track: 'product-eligible' as const,
            purpose: 'product-training' as const,
            commercialUse: 'allowed' as const,
            derivativeWeights: 'unclear' as const,
          },
        ],
      };

      const result = validateDatasetRegistry(invalid);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.includes("derivativeWeights: 'allowed'"))).toBe(true);
    });

    it('rejects product-eligible dataset with unverified license', () => {
      const invalid = {
        ...baseValidRegistry,
        datasets: [
          {
            ...baseValidRegistry.datasets[0],
            id: 'bad-product-unverified',
            track: 'product-eligible' as const,
            purpose: 'product-training' as const,
            commercialUse: 'allowed' as const,
            derivativeWeights: 'allowed' as const,
            datasetLicense: 'unverified',
          },
        ],
      };

      const result = validateDatasetRegistry(invalid);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.includes('unverified license'))).toBe(true);
    });

    it('rejects status verified when licenseEvidenceUrls is empty', () => {
      const invalid = {
        ...baseValidRegistry,
        datasets: [
          {
            ...baseValidRegistry.datasets[0],
            status: 'verified' as const,
            licenseEvidenceUrls: [],
          },
        ],
      };

      const result = validateDatasetRegistry(invalid);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.includes('requires at least one valid license evidence URL'))).toBe(true);
    });

    it('rejects dataset entry with empty officialSource', () => {
      const invalid = {
        ...baseValidRegistry,
        datasets: [
          {
            ...baseValidRegistry.datasets[0],
            officialSource: '  ',
          },
        ],
      };

      const result = validateDatasetRegistry(invalid);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.includes('Official source must be specified'))).toBe(true);
    });

    it('enforces purpose vs track alignment', () => {
      const mismatch = {
        ...baseValidRegistry,
        datasets: [
          {
            ...baseValidRegistry.datasets[0],
            id: 'mismatch-fixture',
            track: 'research-only' as const,
            purpose: 'fixture' as const,
          },
        ],
      };

      const result = validateDatasetRegistry(mismatch);
      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.includes("Purpose 'fixture' must be on 'fixture-only' track"))).toBe(true);
    });
  });

  describe('Model Lineage Contamination Guard (validateProductionModelLineage)', () => {
    const mockRegistry: DatasetRegistry = {
      schemaVersion: '1.0.0',
      datasets: [
        {
          id: 'genimage',
          name: 'GenImage',
          version: '1.0.0',
          track: 'research-only',
          purpose: 'scientific-benchmark',
          officialSource: 'https://github.com/GenImage-Dataset/GenImage',
          downloadSource: 'https://github.com/GenImage-Dataset/GenImage',
          codeLicense: 'Apache-2.0',
          datasetLicense: 'CC BY-NC-SA 4.0 with additional dataset terms',
          additionalTerms: [],
          commercialUse: 'prohibited',
          redistribution: 'prohibited',
          derivativeWeights: 'unclear',
          availability: 'available',
          expectedDownloadBytes: null,
          checksumAvailable: false,
          licenseEvidenceUrls: ['https://example.org'],
          reviewedAt: '2026-09-21T00:00:00Z',
          status: 'verified',
          notes: '',
        },
        {
          id: 'realhd',
          name: 'RealHD',
          version: '1.0.0',
          track: 'blocked',
          purpose: 'scientific-benchmark',
          officialSource: 'https://real-hd.github.io',
          downloadSource: 'https://github.com/Hanzhe-yu/RealHD',
          codeLicense: 'unverified',
          datasetLicense: 'unverified',
          additionalTerms: [],
          commercialUse: 'unclear',
          redistribution: 'unclear',
          derivativeWeights: 'unclear',
          availability: 'unavailable',
          expectedDownloadBytes: null,
          checksumAvailable: false,
          licenseEvidenceUrls: ['https://example.org'],
          reviewedAt: '2026-09-21T00:00:00Z',
          status: 'blocked',
          notes: '',
        },
        {
          id: 'synthetic-smoke',
          name: 'Smoke Fixture',
          version: '1.0.0',
          track: 'fixture-only',
          purpose: 'fixture',
          officialSource: 'internal://smoke',
          downloadSource: 'internal://smoke',
          codeLicense: 'Project-Internal',
          datasetLicense: 'Project-Internal',
          additionalTerms: [],
          commercialUse: 'internal-testing-only',
          redistribution: 'allowed',
          derivativeWeights: 'unclear',
          availability: 'available',
          expectedDownloadBytes: 0,
          checksumAvailable: true,
          licenseEvidenceUrls: ['https://example.org'],
          reviewedAt: '2026-09-21T00:00:00Z',
          status: 'verified',
          notes: '',
        },
        {
          id: 'clean-product-data',
          name: 'Clean Product Data',
          version: '1.0.0',
          track: 'product-eligible',
          purpose: 'product-training',
          officialSource: 'internal://clean',
          downloadSource: 'internal://clean',
          codeLicense: 'Apache-2.0',
          datasetLicense: 'Apache-2.0',
          additionalTerms: [],
          commercialUse: 'allowed',
          redistribution: 'allowed',
          derivativeWeights: 'allowed',
          availability: 'available',
          expectedDownloadBytes: 0,
          checksumAvailable: true,
          licenseEvidenceUrls: ['https://example.org'],
          reviewedAt: '2026-09-21T00:00:00Z',
          status: 'verified',
          notes: '',
        },
      ],
    };

    it('rejects model lineage containing research-only datasets', () => {
      const result = validateProductionModelLineage(['genimage'], mockRegistry);
      expect(result.compliant).toBe(false);
      expect(result.violations.some((v) => v.includes('Contamination violation'))).toBe(true);
    });

    it('rejects model lineage containing fixture-only datasets', () => {
      const result = validateProductionModelLineage(['synthetic-smoke'], mockRegistry);
      expect(result.compliant).toBe(false);
      expect(result.violations.some((v) => v.includes('Fixture violation'))).toBe(true);
    });

    it('rejects model lineage containing blocked datasets', () => {
      const result = validateProductionModelLineage(['realhd'], mockRegistry);
      expect(result.compliant).toBe(false);
      expect(result.violations.some((v) => v.includes('Blocked dataset violation'))).toBe(true);
    });

    it('rejects model lineage with unregistered datasets', () => {
      const result = validateProductionModelLineage(['unknown-data-xyz'], mockRegistry);
      expect(result.compliant).toBe(false);
      expect(result.violations.some((v) => v.includes('is not registered'))).toBe(true);
    });

    it('approves model lineage containing only product-eligible datasets', () => {
      const result = validateProductionModelLineage(['clean-product-data'], mockRegistry);
      expect(result.compliant).toBe(true);
      expect(result.violations).toEqual([]);
    });
  });

  describe('Repository Clean Link & Path Invariance (Section 9)', () => {
    it('verifies 0 machine-local links in repository documents', () => {
      const repoRoot = path.resolve(__dirname, '../../../../');
      const targetDirs = ['docs', 'datasets', 'models', 'research'];

      const checkFile = (filePath: string) => {
        const content = fs.readFileSync(filePath, 'utf-8');
        const lines = content.split('\n');

        lines.forEach((line, idx) => {
          // Check for actual markdown links e.g. [text](file:///...) or [text](D:\...)
          const linkMatch = line.match(/\]\((file:\/\/\/|[A-Za-z]:\\|\/Users\/|\/home\/)/);
          if (linkMatch) {
            throw new Error(
              `Found machine-local markdown link at ${path.relative(repoRoot, filePath)}:${idx + 1}: ${line}`
            );
          }
        });
      };

      const walk = (dir: string) => {
        if (!fs.existsSync(dir)) return;
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
          const fullPath = path.join(dir, entry.name);
          if (entry.isDirectory()) {
            if (entry.name !== 'node_modules' && entry.name !== '.venv' && entry.name !== '.git') {
              walk(fullPath);
            }
          } else if (entry.isFile() && (entry.name.endsWith('.md') || entry.name.endsWith('.json'))) {
            checkFile(fullPath);
          }
        }
      };

      for (const dir of targetDirs) {
        walk(path.join(repoRoot, dir));
      }
      checkFile(path.join(repoRoot, 'README.md'));
    });
  });
});
