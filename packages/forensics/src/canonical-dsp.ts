/**
 * Canonical 16-D DSP Forensic Feature Extractor in TypeScript.
 *
 * Implements 100% numerical and algorithmic parity with Python ml/training/dsp_features.py:
 *  1. fft_high_freq_energy_ratio
 *  2. fft_spectral_peak_count (min(peak, 10) / 10)
 *  3. fft_radial_anomaly_score
 *  4. fft_spectral_decay_slope
 *  5. dct_mean_high_freq_energy
 *  6. dct_ac_energy_variance
 *  7. dct_anomaly_score
 *  8. noise_global_level
 *  9. noise_max_to_min_ratio (min(ratio, 50) / 50)
 * 10. noise_std_of_variances
 * 11. noise_coef_variation (min(coef, 5))
 * 12. noise_inconsistency_score
 * 13. jpeg_boundary_difference_ratio (min(ratio, 5))
 * 14. jpeg_periodic_grid_strength (min(periodic, 4))
 * 15. jpeg_artifact_score
 * 16. laplacian_variance
 */

export const CANONICAL_DSP_FEATURE_NAMES = [
  'fft_high_freq_energy_ratio',
  'fft_spectral_peak_count',
  'fft_radial_anomaly_score',
  'fft_spectral_decay_slope',
  'dct_mean_high_freq_energy',
  'dct_ac_energy_variance',
  'dct_anomaly_score',
  'noise_global_level',
  'noise_max_to_min_ratio',
  'noise_std_of_variances',
  'noise_coef_variation',
  'noise_inconsistency_score',
  'jpeg_boundary_difference_ratio',
  'jpeg_periodic_grid_strength',
  'jpeg_artifact_score',
  'laplacian_variance',
] as const;

export const CANONICAL_DSP_DIM = CANONICAL_DSP_FEATURE_NAMES.length; // 16

/**
 * Converts RGBA pixels to 2D grayscale Float32Array in [0.0, 1.0] using ITU-R BT.601.
 */
export function toGrayscaleFloat(
  rgba: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number
): Float32Array {
  const total = width * height;
  const gray = new Float32Array(total);
  for (let i = 0; i < total; i++) {
    const idx = i * 4;
    gray[i] = (0.299 * rgba[idx] + 0.587 * rgba[idx + 1] + 0.114 * rgba[idx + 2]) / 255.0;
  }
  return gray;
}

/**
 * 1D Radix-2 FFT
 */
function fft1d(real: Float32Array, imag: Float32Array): void {
  const n = real.length;
  if (n <= 1) return;

  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      const tr = real[i];
      const ti = imag[i];
      real[i] = real[j];
      imag[i] = imag[j];
      real[j] = tr;
      imag[j] = ti;
    }
    let k = n >> 1;
    while (k <= j) {
      j -= k;
      k >>= 1;
    }
    j += k;
  }

  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const wStepR = Math.cos(angle);
    const wStepI = Math.sin(angle);

    for (let i = 0; i < n; i += len) {
      let wr = 1.0;
      let wi = 0.0;
      for (let k = 0; k < halfLen; k++) {
        const idx = i + k;
        const targetIdx = idx + halfLen;

        const ur = real[idx];
        const ui = imag[idx];

        const vr = real[targetIdx] * wr - imag[targetIdx] * wi;
        const vi = real[targetIdx] * wi + imag[targetIdx] * wr;

        real[idx] = ur + vr;
        imag[idx] = ui + vi;
        real[targetIdx] = ur - vr;
        imag[targetIdx] = ui - vi;

        const nextWr = wr * wStepR - wi * wStepI;
        const nextWi = wr * wStepI + wi * wStepR;
        wr = nextWr;
        wi = nextWi;
      }
    }
  }
}

/**
 * Bilinear resize to fixed target_size x target_size
 */
function bilinearResize(
  src: Float32Array,
  w: number,
  h: number,
  targetSize: number
): Float32Array {
  const dst = new Float32Array(targetSize * targetSize);
  const scaleX = w / targetSize;
  const scaleY = h / targetSize;

  for (let y = 0; y < targetSize; y++) {
    const srcY = (y + 0.5) * scaleY - 0.5;
    const y0 = Math.max(0, Math.floor(srcY));
    const y1 = Math.min(h - 1, y0 + 1);
    const wy = srcY - y0;

    for (let x = 0; x < targetSize; x++) {
      const srcX = (x + 0.5) * scaleX - 0.5;
      const x0 = Math.max(0, Math.floor(srcX));
      const x1 = Math.min(w - 1, x0 + 1);
      const wx = srcX - x0;

      const p00 = src[y0 * w + x0];
      const p10 = src[y0 * w + x1];
      const p01 = src[y1 * w + x0];
      const p11 = src[y1 * w + x1];

      const val = (1 - wy) * ((1 - wx) * p00 + wx * p10) + wy * ((1 - wx) * p01 + wx * p11);
      dst[y * targetSize + x] = Math.max(0.0, Math.min(1.0, val));
    }
  }
  return dst;
}

/**
 * 2D FFT Analysis matching compute_2d_fft_analysis in dsp_features.py
 */
export function computeCanonicalFft(
  gray: Float32Array,
  width: number,
  height: number,
  targetSize = 128
): [number, number, number, number] {
  if (height < 8 || width < 8) {
    return [0.0, 0.0, 0.0, -2.0];
  }

  const res = (width === targetSize && height === targetSize)
    ? new Float32Array(gray)
    : bilinearResize(gray, width, height, targetSize);

  const real = new Float32Array(res);
  const imag = new Float32Array(targetSize * targetSize);

  const rowReal = new Float32Array(targetSize);
  const rowImag = new Float32Array(targetSize);

  for (let r = 0; r < targetSize; r++) {
    const offset = r * targetSize;
    for (let c = 0; c < targetSize; c++) {
      rowReal[c] = real[offset + c];
      rowImag[c] = imag[offset + c];
    }
    fft1d(rowReal, rowImag);
    for (let c = 0; c < targetSize; c++) {
      real[offset + c] = rowReal[c];
      imag[offset + c] = rowImag[c];
    }
  }

  const colReal = new Float32Array(targetSize);
  const colImag = new Float32Array(targetSize);

  for (let c = 0; c < targetSize; c++) {
    for (let r = 0; r < targetSize; r++) {
      colReal[r] = real[r * targetSize + c];
      colImag[r] = imag[r * targetSize + c];
    }
    fft1d(colReal, colImag);
    for (let r = 0; r < targetSize; r++) {
      real[r * targetSize + c] = colReal[r];
      imag[r * targetSize + c] = colImag[r];
    }
  }

  const half = targetSize >> 1;
  const numBins = half;
  const radialBins = new Float64Array(numBins);
  const radialCounts = new Int32Array(numBins);

  let totalEnergy = 0.0;
  let highFreqEnergy = 0.0;
  const highFreqCutoff = Math.floor(half * 0.6);

  for (let r = 0; r < targetSize; r++) {
    for (let c = 0; c < targetSize; c++) {
      const dy = (r + half) % targetSize - half;
      const dx = (c + half) % targetSize - half;
      const dist = Math.floor(Math.sqrt(dx * dx + dy * dy));

      const idx = r * targetSize + c;
      const mag = Math.sqrt(real[idx] * real[idx] + imag[idx] * imag[idx]);

      totalEnergy += mag;
      if (dist >= highFreqCutoff) {
        highFreqEnergy += mag;
      }
      if (dist < numBins) {
        radialBins[dist] += mag;
        radialCounts[dist]++;
      }
    }
  }

  for (let b = 0; b < numBins; b++) {
    if (radialCounts[b] > 0) {
      radialBins[b] /= radialCounts[b];
    }
  }

  let peakCount = 0.0;
  for (let b = 3; b < numBins - 2; b++) {
    const prev = radialBins[b - 1];
    const curr = radialBins[b];
    const next = radialBins[b + 1];
    if (curr > 1.4 * prev && curr > 1.4 * next) {
      peakCount += 1.0;
    }
  }

  const highFreqRatio = totalEnergy > 0 ? highFreqEnergy / totalEnergy : 0.0;
  const radialAnomalyScore = Math.min(
    1.0,
    peakCount * 0.3 + (highFreqRatio > 0.35 ? 0.4 : 0.0)
  );

  // Power-law decay slope via linear regression on log(r), log(P(r))
  const logR: number[] = [];
  const logP: number[] = [];
  for (let r = 2; r < numBins - 2; r++) {
    if (radialBins[r] > 1e-12) {
      logR.push(Math.log(r));
      logP.push(Math.log(radialBins[r]));
    }
  }

  let slope = -2.0;
  if (logR.length >= 5) {
    const n = logR.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    for (let i = 0; i < n; i++) {
      sumX += logR[i];
      sumY += logP[i];
      sumXY += logR[i] * logP[i];
      sumXX += logR[i] * logR[i];
    }
    const denom = n * sumXX - sumX * sumX;
    if (Math.abs(denom) > 1e-12) {
      slope = (n * sumXY - sumX * sumY) / denom;
    }
  }
  const clampedSlope = Math.max(-10.0, Math.min(2.0, slope));

  return [
    highFreqRatio,
    Math.min(peakCount, 10.0) / 10.0,
    radialAnomalyScore,
    clampedSlope,
  ];
}

/**
 * 2D DCT Analysis matching compute_8x8_dct_analysis in dsp_features.py
 */
export function computeCanonicalDct(
  gray: Float32Array,
  width: number,
  height: number
): [number, number, number] {
  const blocksX = Math.floor(width / 8);
  const blocksY = Math.floor(height / 8);
  const totalBlocks = blocksX * blocksY;

  if (totalBlocks === 0) {
    return [0.0, 0.0, 0.0];
  }

  const blockHighEnergies = new Float64Array(totalBlocks);
  let totalHighEnergy = 0.0;

  let blockIdx = 0;
  for (let by = 0; by < blocksY; by++) {
    for (let bx = 0; bx < blocksX; bx++) {
      let blockSum = 0.0;

      for (let u = 0; u < 8; u++) {
        for (let v = 0; v < 8; v++) {
          if (u + v < 6) continue;

          let sum = 0.0;
          for (let y = 0; y < 8; y++) {
            const cosY = Math.cos(((2 * y + 1) * v * Math.PI) / 16.0);
            for (let x = 0; x < 8; x++) {
              const cosX = Math.cos(((2 * x + 1) * u * Math.PI) / 16.0);
              const pixel = gray[(by * 8 + y) * width + (bx * 8 + x)];
              sum += pixel * cosX * cosY;
            }
          }

          const cu = u === 0 ? 1.0 / Math.SQRT2 : 1.0;
          const cv = v === 0 ? 1.0 / Math.SQRT2 : 1.0;
          const coeff = 0.25 * cu * cv * sum;
          blockSum += Math.abs(coeff);
        }
      }

      blockHighEnergies[blockIdx] = blockSum;
      totalHighEnergy += blockSum;
      blockIdx++;
    }
  }

  const meanHigh = totalHighEnergy / totalBlocks;

  let varianceSum = 0.0;
  for (let i = 0; i < totalBlocks; i++) {
    const diff = blockHighEnergies[i] - meanHigh;
    varianceSum += diff * diff;
  }
  const variance = varianceSum / totalBlocks;

  const normalizedVar = Math.sqrt(variance) / (meanHigh + 1e-4);
  const anomalyScore = Math.max(0.0, Math.min(1.0, normalizedVar * 0.8));

  return [meanHigh, variance, anomalyScore];
}

/**
 * Spatial Noise Residual Analysis matching compute_noise_residual_analysis in dsp_features.py
 */
export function computeCanonicalNoise(
  gray: Float32Array,
  width: number,
  height: number,
  gridCols = 6,
  gridRows = 6
): [number, number, number, number, number] {
  if (height < gridRows * 2 || width < gridCols * 2) {
    return [0.0, 1.0 / 50.0, 0.0, 0.0, 0.0];
  }

  // 3x3 Laplacian symmetric convolution
  const residual = new Float64Array(width * height);
  for (let y = 0; y < height; y++) {
    const yPrev = y === 0 ? 1 : y - 1;
    const yNext = y === height - 1 ? height - 2 : y + 1;
    for (let x = 0; x < width; x++) {
      const xPrev = x === 0 ? 1 : x - 1;
      const xNext = x === width - 1 ? width - 2 : x + 1;

      const top = gray[yPrev * width + x];
      const bottom = gray[yNext * width + x];
      const left = gray[y * width + xPrev];
      const right = gray[y * width + xNext];
      const center = gray[y * width + x];

      const val = top + bottom + left + right - 4.0 * center;
      residual[y * width + x] = Math.abs(val);
    }
  }

  const patchH = Math.floor(height / gridRows);
  const patchW = Math.floor(width / gridCols);
  const patchVariances = new Float64Array(gridRows * gridCols);
  let pIdx = 0;
  let sumAllVariances = 0.0;
  let minVar = Infinity;
  let maxVar = -Infinity;

  for (let gy = 0; gy < gridRows; gy++) {
    const y0 = gy * patchH;
    const y1 = gy < gridRows - 1 ? (gy + 1) * patchH : height;
    for (let gx = 0; gx < gridCols; gx++) {
      const x0 = gx * patchW;
      const x1 = gx < gridCols - 1 ? (gx + 1) * patchW : width;

      let sum = 0.0;
      let count = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          sum += residual[y * width + x];
          count++;
        }
      }
      const meanPatch = count > 0 ? sum / count : 0.0;
      patchVariances[pIdx++] = meanPatch;
      sumAllVariances += meanPatch;

      if (meanPatch < minVar) minVar = meanPatch;
      if (meanPatch > maxVar) maxVar = meanPatch;
    }
  }

  const numPatches = gridCols * gridRows;
  const avgVar = sumAllVariances / numPatches;
  const maxToMinRatio = maxVar > 1e-3 ? maxVar / Math.max(minVar, 1e-4) : 1.0;

  let varOfVars = 0.0;
  for (let i = 0; i < numPatches; i++) {
    const diff = patchVariances[i] - avgVar;
    varOfVars += diff * diff;
  }
  const stdVar = Math.sqrt(varOfVars / numPatches);
  const coefVar = stdVar / (avgVar + 1e-4);
  const inconsistencyScore = Math.max(0.0, Math.min(1.0, (coefVar - 0.25) / 0.75));

  return [
    avgVar,
    Math.min(maxToMinRatio, 50.0) / 50.0,
    stdVar,
    Math.min(coefVar, 5.0),
    inconsistencyScore,
  ];
}

/**
 * JPEG Block Grid Analysis matching compute_jpeg_block_grid_analysis in dsp_features.py
 */
export function computeCanonicalJpeg(
  gray: Float32Array,
  width: number,
  height: number
): [number, number, number] {
  if (height < 16 || width < 16) {
    return [1.0, 0.0, 0.0];
  }

  let boundaryGradSum = 0.0;
  let boundaryCount = 0;
  let interiorGradSum = 0.0;
  let interiorCount = 0;

  // Horizontal gradient differences
  for (let y = 0; y < height; y++) {
    const rowOffset = y * width;
    for (let x = 1; x < width; x++) {
      const diff = Math.abs(gray[rowOffset + x] - gray[rowOffset + x - 1]);
      if (x % 8 === 0) {
        boundaryGradSum += diff;
        boundaryCount++;
      } else {
        interiorGradSum += diff;
        interiorCount++;
      }
    }
  }

  // Vertical gradient differences
  for (let y = 1; y < height; y++) {
    const isBoundary = y % 8 === 0;
    const currOffset = y * width;
    const prevOffset = (y - 1) * width;
    for (let x = 0; x < width; x++) {
      const diff = Math.abs(gray[currOffset + x] - gray[prevOffset + x]);
      if (isBoundary) {
        boundaryGradSum += diff;
        boundaryCount++;
      } else {
        interiorGradSum += diff;
        interiorCount++;
      }
    }
  }

  const avgBoundary = boundaryCount > 0 ? boundaryGradSum / boundaryCount : 0.0;
  const avgInterior = interiorCount > 0 ? interiorGradSum / interiorCount : 0.0;

  const ratio = avgBoundary / Math.max(avgInterior, 1e-3);
  const periodicStrength = Math.max(0.0, ratio - 1.0);
  const artifactScore = Math.max(0.0, Math.min(1.0, periodicStrength * 2.0));

  return [
    Math.min(ratio, 5.0),
    Math.min(periodicStrength, 4.0),
    artifactScore,
  ];
}

/**
 * Laplacian Variance matching compute_laplacian_variance in dsp_features.py (valid convolution mode)
 */
export function computeCanonicalLaplacianVariance(
  gray: Float32Array,
  width: number,
  height: number
): number {
  if (height < 3 || width < 3) return 0.0;

  const validH = height - 2;
  const validW = width - 2;
  const total = validH * validW;
  const lap = new Float64Array(total);

  let sum = 0.0;
  let idx = 0;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const val =
        gray[(y - 1) * width + x] +
        gray[(y + 1) * width + x] +
        gray[y * width + (x - 1)] +
        gray[y * width + (x + 1)] -
        4.0 * gray[y * width + x];
      lap[idx++] = val;
      sum += val;
    }
  }

  const mean = sum / total;
  let varSum = 0.0;
  for (let i = 0; i < total; i++) {
    const diff = lap[i] - mean;
    varSum += diff * diff;
  }

  return varSum / total;
}

/**
 * Extracts all 16 canonical DSP features into a Float64Array.
 */
export function extractCanonicalDspFeatures(
  rgba: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number
): Float64Array {
  const gray = toGrayscaleFloat(rgba, width, height);

  const [fft0, fft1, fft2, fft3] = computeCanonicalFft(gray, width, height);
  const [dct0, dct1, dct2] = computeCanonicalDct(gray, width, height);
  const [noise0, noise1, noise2, noise3, noise4] = computeCanonicalNoise(gray, width, height);
  const [jpeg0, jpeg1, jpeg2] = computeCanonicalJpeg(gray, width, height);
  const lapVar = computeCanonicalLaplacianVariance(gray, width, height);

  const features = new Float64Array(16);
  features[0] = fft0;
  features[1] = fft1;
  features[2] = fft2;
  features[3] = fft3;
  features[4] = dct0;
  features[5] = dct1;
  features[6] = dct2;
  features[7] = noise0;
  features[8] = noise1;
  features[9] = noise2;
  features[10] = noise3;
  features[11] = noise4;
  features[12] = jpeg0;
  features[13] = jpeg1;
  features[14] = jpeg2;
  features[15] = lapVar;

  for (let i = 0; i < 16; i++) {
    if (!Number.isFinite(features[i])) {
      features[i] = 0.0;
    }
  }

  return features;
}
