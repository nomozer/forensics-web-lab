import { MODEL_CONFIG, BoundingBox } from "@forensics/shared";

export interface PatchItem {
  box: BoundingBox; // Normalized [0, 1]
  pixelBox: { x: number; y: number; width: number; height: number };
  tensorData: Float32Array; // [3, 224, 224] NCHW normalized
}

export class PatchExtractor {
  /**
   * Extracts overlapping square patches from RGBA pixel data.
   * Dynamically adjusts stride if image dimensions produce > MAX_PATCHES patches.
   */
  public static extractPatches(
    rgba: Uint8ClampedArray | Uint8Array,
    imgWidth: number,
    imgHeight: number,
    targetPatchSize = MODEL_CONFIG.PATCH_SIZE,
  ): PatchItem[] {
    const patches: PatchItem[] = [];

    // Adaptive stride to guarantee bounding patch count <= MAX_PATCHES
    const estimatedCols = Math.max(1, Math.floor(imgWidth / targetPatchSize));
    const estimatedRows = Math.max(1, Math.floor(imgHeight / targetPatchSize));
    const baseGrid = estimatedCols * estimatedRows * 4;

    let strideX: number = MODEL_CONFIG.PATCH_STRIDE;
    let strideY: number = MODEL_CONFIG.PATCH_STRIDE;

    if (baseGrid > MODEL_CONFIG.MAX_PATCHES) {
      strideX = Math.max(strideX, Math.floor(imgWidth / 6));
      strideY = Math.max(strideY, Math.floor(imgHeight / 6));
    }

    const maxX = Math.max(0, imgWidth - targetPatchSize);
    const maxY = Math.max(0, imgHeight - targetPatchSize);

    const xCoords: number[] = [];
    for (let x = 0; x <= maxX; x += strideX) xCoords.push(x);
    if (xCoords[xCoords.length - 1] !== maxX) xCoords.push(maxX);

    const yCoords: number[] = [];
    for (let y = 0; y <= maxY; y += strideY) yCoords.push(y);
    if (yCoords[yCoords.length - 1] !== maxY) yCoords.push(maxY);

    for (const py of yCoords) {
      for (const px of xCoords) {
        const patchData = new Float32Array(
          3 * targetPatchSize * targetPatchSize,
        );
        const planeSize = targetPatchSize * targetPatchSize;

        // Extract and normalize pixels: (pixel / 255 - mean) / std
        for (let y = 0; y < targetPatchSize; y++) {
          const srcY = Math.min(imgHeight - 1, py + y);
          for (let x = 0; x < targetPatchSize; x++) {
            const srcX = Math.min(imgWidth - 1, px + x);
            const srcIdx = (srcY * imgWidth + srcX) * 4;
            const dstIdx = y * targetPatchSize + x;

            const r = rgba[srcIdx] / 255.0;
            const g = rgba[srcIdx + 1] / 255.0;
            const b = rgba[srcIdx + 2] / 255.0;

            patchData[dstIdx] =
              (r - MODEL_CONFIG.NORM_MEAN[0]) / MODEL_CONFIG.NORM_STD[0];
            patchData[planeSize + dstIdx] =
              (g - MODEL_CONFIG.NORM_MEAN[1]) / MODEL_CONFIG.NORM_STD[1];
            patchData[2 * planeSize + dstIdx] =
              (b - MODEL_CONFIG.NORM_MEAN[2]) / MODEL_CONFIG.NORM_STD[2];
          }
        }

        patches.push({
          box: {
            x: px / imgWidth,
            y: py / imgHeight,
            width: targetPatchSize / imgWidth,
            height: targetPatchSize / imgHeight,
          },
          pixelBox: {
            x: px,
            y: py,
            width: targetPatchSize,
            height: targetPatchSize,
          },
          tensorData: patchData,
        });

        if (patches.length >= MODEL_CONFIG.MAX_PATCHES) break;
      }
      if (patches.length >= MODEL_CONFIG.MAX_PATCHES) break;
    }

    return patches;
  }

  /**
   * Resamples whole image into a single [1, 3, 224, 224] Float32 tensor for global classification.
   */
  public static extractGlobalTensor(
    rgba: Uint8ClampedArray | Uint8Array,
    imgWidth: number,
    imgHeight: number,
    targetSize = MODEL_CONFIG.PATCH_SIZE,
  ): Float32Array {
    const tensor = new Float32Array(3 * targetSize * targetSize);
    const planeSize = targetSize * targetSize;
    const scaleX = imgWidth / targetSize;
    const scaleY = imgHeight / targetSize;

    for (let y = 0; y < targetSize; y++) {
      const sy = Math.min(imgHeight - 1, Math.floor(y * scaleY));
      for (let x = 0; x < targetSize; x++) {
        const sx = Math.min(imgWidth - 1, Math.floor(x * scaleX));
        const srcIdx = (sy * imgWidth + sx) * 4;
        const dstIdx = y * targetSize + x;

        const r = rgba[srcIdx] / 255.0;
        const g = rgba[srcIdx + 1] / 255.0;
        const b = rgba[srcIdx + 2] / 255.0;

        tensor[dstIdx] =
          (r - MODEL_CONFIG.NORM_MEAN[0]) / MODEL_CONFIG.NORM_STD[0];
        tensor[planeSize + dstIdx] =
          (g - MODEL_CONFIG.NORM_MEAN[1]) / MODEL_CONFIG.NORM_STD[1];
        tensor[2 * planeSize + dstIdx] =
          (b - MODEL_CONFIG.NORM_MEAN[2]) / MODEL_CONFIG.NORM_STD[2];
      }
    }

    return tensor;
  }
}
