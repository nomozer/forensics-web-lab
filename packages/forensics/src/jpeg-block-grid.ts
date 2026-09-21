export interface JpegBlockGridResult {
  boundaryDifferenceRatio: number;
  periodicGridStrength: number;
  jpegArtifactScore: number;
}

/**
 * Evaluates 8x8 Block Artifact Grid (BAG) strength.
 * Measures gradient magnitude across 8-pixel block boundaries compared to interior gradients.
 */
export function computeJpegBlockGridAnalysis(
  grayscale: Float32Array,
  width: number,
  height: number
): JpegBlockGridResult {
  let boundaryGradSum = 0.0;
  let boundaryCount = 0;

  let interiorGradSum = 0.0;
  let interiorCount = 0;

  // Horizontal gradient differences
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width - 1; x++) {
      const idx = y * width + x;
      const diff = Math.abs(grayscale[idx + 1] - grayscale[idx]);

      if ((x + 1) % 8 === 0) {
        boundaryGradSum += diff;
        boundaryCount++;
      } else {
        interiorGradSum += diff;
        interiorCount++;
      }
    }
  }

  // Vertical gradient differences
  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const diff = Math.abs(grayscale[idx + width] - grayscale[idx]);

      if ((y + 1) % 8 === 0) {
        boundaryGradSum += diff;
        boundaryCount++;
      } else {
        interiorGradSum += diff;
        interiorCount++;
      }
    }
  }

  const avgBoundary = boundaryCount > 0 ? boundaryGradSum / boundaryCount : 0;
  const avgInterior = interiorCount > 0 ? interiorGradSum / interiorCount : 0;

  const ratio = avgInterior > 0.001 ? avgBoundary / avgInterior : 1.0;
  // A ratio significantly above 1.0 indicates pronounced JPEG 8x8 block artifacts.
  // A ratio near 1.0 indicates uncompressed or anti-aliased content.
  const periodicStrength = Math.max(0.0, ratio - 1.0);
  const jpegArtifactScore = Math.min(1.0, periodicStrength * 2.0);

  return {
    boundaryDifferenceRatio: ratio,
    periodicGridStrength: periodicStrength,
    jpegArtifactScore,
  };
}
