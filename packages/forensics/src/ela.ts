export interface ElaAnalysisResult {
  meanError: number;
  maxError: number;
  errorVariance: number;
  elaScore: number;
}

/**
 * Computes Error Level Analysis (ELA) statistics between original image pixel data
 * and recompressed pixel data.
 */
export function computeElaStatistics(
  origRgba: Uint8ClampedArray | Uint8Array,
  recompRgba: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number,
  scaleFactor = 10
): { stats: ElaAnalysisResult; diffRgba: Uint8ClampedArray } {
  const totalPixels = width * height;
  const diffRgba = new Uint8ClampedArray(totalPixels * 4);

  let totalDiff = 0;
  let maxDiff = 0;
  const pixelErrors = new Float32Array(totalPixels);

  for (let i = 0; i < totalPixels; i++) {
    const idx = i * 4;
    const dr = Math.abs(origRgba[idx] - recompRgba[idx]);
    const dg = Math.abs(origRgba[idx + 1] - recompRgba[idx + 1]);
    const db = Math.abs(origRgba[idx + 2] - recompRgba[idx + 2]);

    const pixelDiff = (dr + dg + db) / 3.0;
    pixelErrors[i] = pixelDiff;
    totalDiff += pixelDiff;
    if (pixelDiff > maxDiff) maxDiff = pixelDiff;

    // Amplified visual output for ELA canvas
    diffRgba[idx] = Math.min(255, Math.round(dr * scaleFactor));
    diffRgba[idx + 1] = Math.min(255, Math.round(dg * scaleFactor));
    diffRgba[idx + 2] = Math.min(255, Math.round(db * scaleFactor));
    diffRgba[idx + 3] = 255; // Alpha
  }

  const meanError = totalDiff / (totalPixels || 1);

  let varianceSum = 0;
  for (let i = 0; i < totalPixels; i++) {
    const d = pixelErrors[i] - meanError;
    varianceSum += d * d;
  }
  const errorVariance = varianceSum / (totalPixels || 1);

  // High error variance across the image often flags localized spliced or composited segments
  const elaScore = Math.min(1.0, (Math.sqrt(errorVariance) / (meanError + 1.0)) * 0.5);

  return {
    stats: {
      meanError,
      maxError: maxDiff,
      errorVariance,
      elaScore,
    },
    diffRgba,
  };
}
