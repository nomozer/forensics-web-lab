/**
 * 1D Cooley-Tukey Radix-2 Fast Fourier Transform
 */
function fft1d(real: Float32Array, imag: Float32Array): void {
  const n = real.length;
  if (n <= 1) return;

  // Bit-reversal permutation
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

  // Butterfly computations
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

export interface FftAnalysisResult {
  highFreqEnergyRatio: number;
  spectralPeakCount: number;
  radialAnomalyScore: number;
}

/**
 * Computes 2D FFT and radial power spectrum anomaly on a power-of-two 2D grayscale buffer.
 */
export function compute2DFftAnalysis(
  grayscale: Float32Array,
  size: number
): FftAnalysisResult {
  const real = new Float32Array(grayscale);
  const imag = new Float32Array(size * size);

  const rowReal = new Float32Array(size);
  const rowImag = new Float32Array(size);

  // Row-wise 1D FFT
  for (let r = 0; r < size; r++) {
    const offset = r * size;
    for (let c = 0; c < size; c++) {
      rowReal[c] = real[offset + c];
      rowImag[c] = imag[offset + c];
    }
    fft1d(rowReal, rowImag);
    for (let c = 0; c < size; c++) {
      real[offset + c] = rowReal[c];
      imag[offset + c] = rowImag[c];
    }
  }

  // Column-wise 1D FFT
  const colReal = new Float32Array(size);
  const colImag = new Float32Array(size);

  for (let c = 0; c < size; c++) {
    for (let r = 0; r < size; r++) {
      colReal[r] = real[r * size + c];
      colImag[r] = imag[r * size + c];
    }
    fft1d(colReal, colImag);
    for (let r = 0; r < size; r++) {
      real[r * size + c] = colReal[r];
      imag[r * size + c] = colImag[r];
    }
  }

  // Compute centered magnitude spectrum and radial profile
  const half = size >> 1;
  const numBins = half;
  const radialBins = new Float32Array(numBins);
  const radialCounts = new Uint32Array(numBins);

  let totalEnergy = 0.0;
  let highFreqEnergy = 0.0;
  const highFreqCutoff = Math.floor(half * 0.6);

  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      // Shift origin to center
      const dy = (r + half) % size - half;
      const dx = (c + half) % size - half;
      const dist = Math.floor(Math.sqrt(dx * dx + dy * dy));

      const idx = r * size + c;
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

  // Detect anomalous high-frequency spikes (common in GAN/Diffusion checkerboard artifacts)
  let peakCount = 0;
  for (let b = 3; b < numBins - 2; b++) {
    const prev = radialBins[b - 1];
    const curr = radialBins[b];
    const next = radialBins[b + 1];
    // A distinct spike higher than 1.4x surrounding neighborhood
    if (curr > 1.4 * prev && curr > 1.4 * next) {
      peakCount++;
    }
  }

  const highFreqRatio = totalEnergy > 0 ? highFreqEnergy / totalEnergy : 0;
  // Synthetic models often exhibit either abnormally high-frequency checkerboard peaks
  // or unnatural steep drop-offs from latent decoders.
  const radialAnomalyScore = Math.min(
    1.0,
    peakCount * 0.3 + (highFreqRatio > 0.35 ? 0.4 : 0.0)
  );

  return {
    highFreqEnergyRatio: highFreqRatio,
    spectralPeakCount: peakCount,
    radialAnomalyScore,
  };
}
