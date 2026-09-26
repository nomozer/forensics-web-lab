import { MetadataSummaryItem } from '@forensics/shared';
import { KNOWN_AI_SIGNATURES, TRADITIONAL_SOFTWARE_PATTERNS } from './software-signatures.js';

export interface ParsedExifData {
  hasExif: boolean;
  orientation?: number;
  make?: string;
  model?: string;
  software?: string;
  dateTime?: string;
  userComment?: string;
  detectedGenerators: string[];
  rawTags: Record<string, string>;
}

export class ExifParser {
  /**
   * Safely parses EXIF segments from a JPEG or raw ArrayBuffer.
   */
  public static parse(buffer: ArrayBuffer): ParsedExifData {
    const result: ParsedExifData = {
      hasExif: false,
      detectedGenerators: [],
      rawTags: {},
    };

    try {
      const view = new DataView(buffer);
      if (view.byteLength < 16) return result;

      // Check JPEG SOI (0xFFD8)
      if (view.getUint16(0, false) === 0xffd8) {
        this.parseJpeg(view, result);
      }
    } catch {
      // Defensive fallback on malformed headers
    }

    this.auditSignatures(result);
    return result;
  }

  private static parseJpeg(view: DataView, result: ParsedExifData): void {
    let offset = 2;
    const len = view.byteLength;

    while (offset < len - 4) {
      if (view.getUint8(offset) !== 0xff) {
        offset++;
        continue;
      }

      const marker = view.getUint8(offset + 1);
      // SOS (Start of Scan) or EOI (End of Image) -> stop reading headers
      if (marker === 0xda || marker === 0xd9) break;

      const markerLength = view.getUint16(offset + 2, false);
      // APP1 Marker for Exif (0xFFE1)
      if (marker === 0xe1 && offset + 10 <= len) {
        const isExif =
          view.getUint8(offset + 4) === 0x45 && // E
          view.getUint8(offset + 5) === 0x78 && // x
          view.getUint8(offset + 6) === 0x69 && // i
          view.getUint8(offset + 7) === 0x66 && // f
          view.getUint8(offset + 8) === 0x00 &&
          view.getUint8(offset + 9) === 0x00;

        if (isExif) {
          result.hasExif = true;
          this.parseTiff(view, offset + 10, markerLength - 8, result);
        }
      }

      offset += 2 + markerLength;
    }
  }

  private static parseTiff(
    view: DataView,
    tiffStart: number,
    tiffLength: number,
    result: ParsedExifData
  ): void {
    if (tiffStart + 8 > view.byteLength) return;

    const byteOrderMarker = view.getUint16(tiffStart, false);
    const littleEndian = byteOrderMarker === 0x4949; // 'II'

    const fortyTwo = view.getUint16(tiffStart + 2, littleEndian);
    if (fortyTwo !== 0x002a) return;

    const ifdOffset = view.getUint32(tiffStart + 4, littleEndian);
    if (ifdOffset >= tiffLength) return;

    this.parseIfd(view, tiffStart, tiffStart + ifdOffset, littleEndian, result);
  }

  private static parseIfd(
    view: DataView,
    tiffStart: number,
    ifdOffset: number,
    littleEndian: boolean,
    result: ParsedExifData
  ): void {
    if (ifdOffset + 2 > view.byteLength) return;
    const numEntries = view.getUint16(ifdOffset, littleEndian);

    let curr = ifdOffset + 2;
    for (let i = 0; i < numEntries && curr + 12 <= view.byteLength; i++, curr += 12) {
      const tag = view.getUint16(curr, littleEndian);
      const type = view.getUint16(curr + 2, littleEndian);
      const count = view.getUint32(curr + 4, littleEndian);

      let val = '';
      if (type === 2) {
        // ASCII string
        const valOffset = count > 4 ? tiffStart + view.getUint32(curr + 8, littleEndian) : curr + 8;
        val = this.readAscii(view, valOffset, count);
      } else if (type === 3) {
        // SHORT (uint16)
        val = String(view.getUint16(curr + 8, littleEndian));
      } else if (type === 4) {
        // LONG (uint32)
        val = String(view.getUint32(curr + 8, littleEndian));
      }

      switch (tag) {
        case 0x0112: // Orientation
          result.orientation = parseInt(val, 10);
          result.rawTags['Orientation'] = val;
          break;
        case 0x010f: // Make
          result.make = val;
          result.rawTags['Make'] = val;
          break;
        case 0x0110: // Model
          result.model = val;
          result.rawTags['Model'] = val;
          break;
        case 0x0131: // Software
          result.software = val;
          result.rawTags['Software'] = val;
          break;
        case 0x0132: // DateTime
          result.dateTime = val;
          result.rawTags['DateTime'] = val;
          break;
        case 0x9286: // UserComment
          result.userComment = val;
          result.rawTags['UserComment'] = val;
          break;
        case 0x8769: {
          // SubIFD (ExifIFD)
          const subIfdOffset = view.getUint32(curr + 8, littleEndian);
          if (subIfdOffset > 0 && tiffStart + subIfdOffset < view.byteLength) {
            this.parseIfd(view, tiffStart, tiffStart + subIfdOffset, littleEndian, result);
          }
          break;
        }
      }
    }
  }

  private static readAscii(view: DataView, offset: number, count: number): string {
    const chars: string[] = [];
    const maxLen = Math.min(count, 512); // Bound string read length
    for (let i = 0; i < maxLen && offset + i < view.byteLength; i++) {
      const code = view.getUint8(offset + i);
      if (code === 0) break;
      if (code >= 32 && code <= 126) {
        chars.push(String.fromCharCode(code));
      }
    }
    return chars.join('').trim();
  }

  private static auditSignatures(result: ParsedExifData): void {
    const concatenated = [
      result.software ?? '',
      result.userComment ?? '',
      result.make ?? '',
      result.model ?? '',
    ].join(' ');

    for (const sig of KNOWN_AI_SIGNATURES) {
      if (sig.pattern.test(concatenated)) {
        result.detectedGenerators.push(sig.generatorName);
      }
    }
  }

  /**
   * Translates parsed EXIF into standardized summary items.
   */
  public static toSummaryItems(data: ParsedExifData): MetadataSummaryItem[] {
    const items: MetadataSummaryItem[] = [];

    if (!data.hasExif) {
      items.push({
        tag: 'ExifStatus',
        value: 'Không phát hiện metadata EXIF (phổ biến với ảnh chia sẻ mạng xã hội)',
        category: 'provenance',
        suspicionScore: 0.1, // Missing metadata is NOT proof of AI
      });
      return items;
    }

    if (data.make || data.model) {
      items.push({
        tag: 'Camera',
        value: `${data.make ?? ''} ${data.model ?? ''}`.trim(),
        category: 'hardware',
        suspicionScore: 0.0,
      });
    }

    if (data.software) {
      const isAi = KNOWN_AI_SIGNATURES.some((s) => s.pattern.test(data.software ?? ''));
      const isTrad = TRADITIONAL_SOFTWARE_PATTERNS.some((p) => p.test(data.software ?? ''));
      items.push({
        tag: 'Software',
        value: data.software,
        category: 'software',
        suspicionScore: isAi ? 0.95 : isTrad ? 0.3 : 0.1,
      });
    }

    if (data.userComment) {
      const isAi = KNOWN_AI_SIGNATURES.some((s) => s.pattern.test(data.userComment ?? ''));
      items.push({
        tag: 'UserComment',
        value: data.userComment.slice(0, 120),
        category: 'notes',
        suspicionScore: isAi ? 0.9 : 0.05,
      });
    }

    return items;
  }
}
