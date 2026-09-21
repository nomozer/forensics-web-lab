export interface DctAnalysisResult {
  meanHighFreqEnergy: number;
  acEnergyVariance: number;
  dctAnomalyScore: number;
}

/**
 * Computes 2D-DCT across 8x8 blocks of a grayscale image buffer.
 */
export function compute8x8DctAnalysis(
  grayscale: Float32Array,
  width: number,
  height: number
): DctAnalysisResult {
  const blocksX = Math.floor(width / 8);
  const blocksY = Math.floor(height / 8);
  const totalBlocks = blocksX * blocksY;

  if (totalBlocks === 0) {
    return { meanHighFreqEnergy: 0, acEnergyVariance: 0, dctAnomalyScore: 0 };
  }

  const blockHighEnergies = new Float32Array(totalBlocks);
  let totalHighFreq = 0.0;

  let blockIdx = 0;
  for (let by = 0; by < blocksY; by++) {
    for (let bx = 0; bx < blocksX; bx++) {
      let highFreqBlockEnergy = 0.0;

      // Extract 8x8 block and compute high-frequency AC coefficients
      for (let u = 0; u < 8; u++) {
        for (let v = 0; v < 8; v++) {
          if (u + v < 6) continue; // Skip DC and low AC components

          let sum = 0.0;
          for (let x = 0; x < 8; x++) {
            for (let y = 0; y < 8; y++) {
              const pixelVal = grayscale[(by * 8 + y) * width + (bx * 8 + x)];
              sum +=
                pixelVal *
                Math.cos(((2 * x + 1) * u * Math.PI) / 16) *
                Math.cos(((2 * y + 1) * v * Math.PI) / 16);
            }
          }
          const cu = u === 0 ? 1 / Math.SQRT2 : 1;
          const cv = v === 0 ? 1 / Math.SQRT2 : 1;
          const coeff = 0.25 * cu * cv * sum;
          highFreqBlockEnergy += Math.abs(coeff);
        }
      }

      blockHighEnergies[blockIdx] = highFreqBlockEnergy;
      totalHighFreq += highFreqBlockEnergy;
      blockIdx++;
    }
  }

  const meanHigh = totalHighFreq / totalBlocks;

  // Variance of high AC coefficients across blocks
  let varianceSum = 0.0;
  for (let i = 0; i < totalBlocks; i++) {
    const diff = blockHighEnergies[i] - meanHigh;
    varianceSum += diff * diff;
  }
  const variance = varianceSum / totalBlocks;

  // Normalize anomaly score: synthetic images often exhibit unusually low natural texture variance
  // or unnatural localized spikes
  const normalizedVariance = Math.min(1.0, Math.sqrt(variance) / (meanHigh + 1e-4));
  const dctAnomalyScore = Math.min(1.0, normalizedVariance * 0.8);

  return {
    meanHighFreqEnergy: meanHigh,
    acEnergyVariance: variance,
    dctAnomalyScore,
  };
}
