import { describe, it, expect } from 'vitest';
import { ExifParser, analyzeProvenance, C2paProvenanceAdapter } from '../index.js';

describe('@forensics/provenance', () => {
  it('handles empty buffers defensively without throwing', () => {
    const emptyBuf = new ArrayBuffer(0);
    const parsed = ExifParser.parse(emptyBuf);
    expect(parsed.hasExif).toBe(false);
    expect(parsed.detectedGenerators).toEqual([]);
  });

  it('detects missing EXIF and produces polite summary note', async () => {
    const dummyBuf = new Uint8Array([0xff, 0xd8, 0xff, 0xd9]).buffer; // Minimal SOI + EOI
    const prov = await analyzeProvenance(dummyBuf);
    expect(prov.c2paStatus).toBe('unsupported');
    expect(prov.metadataSummary.length).toBeGreaterThan(0);
    expect(prov.metadataSummary[0].tag).toBe('ExifStatus');
    expect(prov.metadataSummary[0].suspicionScore).toBeLessThan(0.5);
  });

  it('verifies C2paProvenanceAdapter safely reports unsupported in standard web context', async () => {
    const adapter = new C2paProvenanceAdapter();
    expect(adapter.isSupported()).toBe(false);
    const result = await adapter.inspect(new ArrayBuffer(16));
    expect(result.status).toBe('unsupported');
    expect(result.details?.['reason']).toBeDefined();
  });
});
