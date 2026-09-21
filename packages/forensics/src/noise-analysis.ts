export interface NoiseAnalysisResult {
  globalNoiseLevel: number;
  noiseInconsistencyScore: number;
  patchVariances: number[];
  maxToMinRatio: number;
}

/**
 * Computes noise residual extraction via 3x3 Laplacian filter and evaluates
 * localized spatial noise variance consistency.
 */
export function computeNoiseResidualAnalysis(
  grayscale: Float32Array,
  width: number,
  height: number,
  gridCols = 6,
  gridRows = 6
): NoiseAnalysisResult {
  const residual = new Float32Array(width * height);

  // 3x3 Laplacian kernel: [0, 1, 0; 1, -4, 1; 0, 1, 0]
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const val =
        grayscale[idx - width] +
        grayscale[idx + width] +
        grayscale[idx - 1] +
        grayscale[idx + 1] -
        4 * grayscale[idx];
      residual[idx] = val;
    }
  }

  // Calculate local patch variances
  const patchW = Math.floor(width / gridCols);
  const patchH = Math.floor(height / gridRows);
  const patchVariances: number[] = [];

  let sumAllVariances = 0.0;
  let minVar = Infinity;
  let maxVar = -Infinity;

  for (let gy = 0; gy < gridRows; gy++) {
    for (let gx = 0; gx < gridCols; gx++) {
      const startX = gx * patchW;
      const startY = gy * patchH;

      let sum = 0.0;
      let count = 0;

      for (let py = 1; py < patchH - 1; py++) {
        for (let px = 1; px < patchW - 1; px++) {
          const idx = (startY + py) * width + (startX + px);
          sum += Math.abs(residual[idx]);
          count++;
        }
      }

      const meanAbs = count > 0 ? sum / count : 0;
      patchVariances.push(meanAbs);
      sumAllVariances += meanAbs;

      if (meanAbs < minVar) minVar = meanAbs;
      if (meanAbs > maxVar) maxVar = meanAbs;
    }
  }

  const numPatches = gridCols * gridRows;
  const avgVar = numPatches > 0 ? sumAllVariances / numPatches : 0;

  // Inpainting or localized editing typically introduces unnatural discontinuities in noise variance
  const maxToMinRatio = maxVar > 0.001 ? maxVar / Math.max(minVar, 1e-4) : 1.0;

  // Standard deviation of patch variances
  let varianceOfVariances = 0.0;
  for (const v of patchVariances) {
    const diff = v - avgVar;
    varianceOfVariances += diff * diff;
  }
  const stdOfVariances = Math.sqrt(varianceOfVariances / (numPatches || 1));
  const coefVariation = avgVar > 0.001 ? stdOfVariances / avgVar : 0;

  // Normalized inconsistency score (0.0 = uniform noise, 1.0 = highly inconsistent localized noise)
  const noiseInconsistencyScore = Math.min(1.0, Math.max(0.0, (coefVariation - 0.25) / 0.75));

  return {
    globalNoiseLevel: avgVar,
    noiseInconsistencyScore,
    patchVariances,
    maxToMinRatio,
  };
}
