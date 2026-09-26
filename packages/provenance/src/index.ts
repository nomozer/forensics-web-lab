import { ExifParser, ParsedExifData } from './exif-parser.js';
import { XmpParser, ParsedXmpData } from './xmp-parser.js';
import { C2paProvenanceAdapter, IC2paAdapter } from './c2pa-adapter.js';
import { ProvenanceResult, MetadataSummaryItem } from '@forensics/shared';

export * from './exif-parser.js';
export * from './xmp-parser.js';
export * from './software-signatures.js';
export * from './c2pa-adapter.js';

export async function analyzeProvenance(
  buffer: ArrayBuffer,
  c2paAdapter: IC2paAdapter = new C2paProvenanceAdapter()
): Promise<ProvenanceResult> {
  const exif = ExifParser.parse(buffer);
  const xmp = XmpParser.parse(buffer);
  const c2pa = await c2paAdapter.inspect(buffer);

  const summaryItems: MetadataSummaryItem[] = [...ExifParser.toSummaryItems(exif)];

  if (xmp.hasXmp) {
    if (xmp.detectedGenerators.length > 0) {
      summaryItems.push({
        tag: 'XMP:Generators',
        value: xmp.detectedGenerators.join(', '),
        category: 'software',
        suspicionScore: 0.95,
      });
    }
    if (xmp.prompt) {
      summaryItems.push({
        tag: 'XMP:Prompt',
        value: xmp.prompt,
        category: 'prompt',
        suspicionScore: 0.9,
      });
    }
  }

  return {
    c2paStatus: c2pa.status,
    c2paDetails: c2pa.details,
    metadataSummary: summaryItems,
  };
}
