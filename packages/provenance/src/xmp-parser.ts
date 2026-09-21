import { KNOWN_AI_SIGNATURES } from './software-signatures.js';

export interface ParsedXmpData {
  hasXmp: boolean;
  rawText?: string;
  detectedGenerators: string[];
  historyEntries: string[];
  prompt?: string;
}

export class XmpParser {
  /**
   * Scans an ArrayBuffer for embedded XMP data packet markers.
   */
  public static parse(buffer: ArrayBuffer): ParsedXmpData {
    const result: ParsedXmpData = {
      hasXmp: false,
      detectedGenerators: [],
      historyEntries: [],
    };

    try {
      // Decode a limited slice of the buffer looking for XMP markers
      // Most XMP packets are in the first 256 KB or last 64 KB
      const bytes = new Uint8Array(buffer);
      const scanLimit = Math.min(bytes.byteLength, 512 * 1024);
      const headerStr = new TextDecoder('utf-8', { fatal: false }).decode(
        bytes.subarray(0, scanLimit)
      );

      const xmpStart = headerStr.indexOf('<x:xmpmeta');
      const xmpEnd = headerStr.indexOf('</x:xmpmeta>');

      if (xmpStart !== -1 && xmpEnd !== -1 && xmpEnd > xmpStart) {
        result.hasXmp = true;
        const xmpContent = headerStr.substring(xmpStart, xmpEnd + 12);
        result.rawText = xmpContent.slice(0, 1024); // Cap length for security

        // Search for software signatures
        for (const sig of KNOWN_AI_SIGNATURES) {
          if (sig.pattern.test(xmpContent)) {
            result.detectedGenerators.push(sig.generatorName);
          }
        }

        // Search for prompt fields
        const promptMatch = xmpContent.match(/(?:prompt|parameters|sd-metadata)["'>:]([^<"'\n]{5,300})/i);
        if (promptMatch && promptMatch[1]) {
          result.prompt = promptMatch[1].trim();
        }
      }
    } catch {
      // Non-blocking fallback on corrupted binary
    }

    return result;
  }
}
