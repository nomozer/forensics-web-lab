import { SuspiciousRegion, LocalizationResult } from "@forensics/shared";
import { PatchItem } from "./patch-extractor.js";

export interface ScoredPatch {
  patch: PatchItem;
  score: number; // Suspicion score [0.0, 1.0]
}

export class HeatmapAccumulator {
  /**
   * Accumulates patch scores into a normalized 2D suspicion matrix.
   */
  public static accumulate(
    scoredPatches: ScoredPatch[],
    gridRows = 32,
    gridCols = 32,
    threshold = 0.65,
  ): {
    localization: LocalizationResult;
    heatmapGrid: number[][];
  } {
    if (scoredPatches.length === 0) {
      return {
        localization: {
          available: false,
          suspiciousAreaRatio: null,
          regions: [],
        },
        heatmapGrid: [],
      };
    }

    const grid = Array.from(
      { length: gridRows },
      () => new Float32Array(gridCols),
    );
    const weightGrid = Array.from(
      { length: gridRows },
      () => new Float32Array(gridCols),
    );

    // Splat patch scores with Gaussian kernel
    for (const item of scoredPatches) {
      const b = item.patch.box;
      const cx = (b.x + b.width / 2) * gridCols;
      const cy = (b.y + b.height / 2) * gridRows;
      const rx = (b.width * gridCols) / 2;
      const ry = (b.height * gridRows) / 2;
      const sigma = Math.max(1.0, Math.min(rx, ry) * 0.6);

      const minGx = Math.max(0, Math.floor(cx - rx));
      const maxGx = Math.min(gridCols - 1, Math.ceil(cx + rx));
      const minGy = Math.max(0, Math.floor(cy - ry));
      const maxGy = Math.min(gridRows - 1, Math.ceil(cy + ry));

      for (let gy = minGy; gy <= maxGy; gy++) {
        for (let gx = minGx; gx <= maxGx; gx++) {
          const dx = gx - cx;
          const dy = gy - cy;
          const distSq = dx * dx + dy * dy;
          const w = Math.exp(-distSq / (2 * sigma * sigma));

          grid[gy][gx] += item.score * w;
          weightGrid[gy][gx] += w;
        }
      }
    }

    // Normalize
    let suspiciousCells = 0;
    const normalizedGrid: number[][] = [];

    for (let gy = 0; gy < gridRows; gy++) {
      const row: number[] = [];
      for (let gx = 0; gx < gridCols; gx++) {
        const totalW = weightGrid[gy][gx];
        const val = totalW > 1e-4 ? Math.min(1.0, grid[gy][gx] / totalW) : 0.0;
        row.push(val);
        if (val >= threshold) {
          suspiciousCells++;
        }
      }
      normalizedGrid.push(row);
    }

    const totalCells = gridRows * gridCols;
    const suspiciousAreaRatio = suspiciousCells / (totalCells || 1);

    // Extract bounding regions for clusters exceeding threshold
    const regions = this.extractRegions(
      normalizedGrid,
      gridRows,
      gridCols,
      threshold,
    );

    return {
      localization: {
        available: true,
        suspiciousAreaRatio,
        heatmapResolution: { rows: gridRows, cols: gridCols },
        regions,
        rawHeatmapGrid: normalizedGrid,
      },
      heatmapGrid: normalizedGrid,
    };
  }

  /**
   * Identifies rectangular bounding boxes enclosing high-suspicion clusters.
   */
  private static extractRegions(
    grid: number[][],
    rows: number,
    cols: number,
    threshold: number,
  ): SuspiciousRegion[] {
    const visited = Array.from({ length: rows }, () => new Uint8Array(cols));
    const regions: SuspiciousRegion[] = [];

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (grid[r][c] >= threshold && !visited[r][c]) {
          // Flood fill cluster
          let minR = r;
          let maxR = r;
          let minC = c;
          let maxC = c;
          let sumScore = 0;
          let count = 0;

          const queue: [number, number][] = [[r, c]];
          visited[r][c] = 1;

          while (queue.length > 0) {
            const [cr, cc] = queue.pop()!;
            minR = Math.min(minR, cr);
            maxR = Math.max(maxR, cr);
            minC = Math.min(minC, cc);
            maxC = Math.max(maxC, cc);
            sumScore += grid[cr][cc];
            count++;

            const neighbors: [number, number][] = [
              [cr - 1, cc],
              [cr + 1, cc],
              [cr, cc - 1],
              [cr, cc + 1],
            ];
            for (const [nr, nc] of neighbors) {
              if (
                nr >= 0 &&
                nr < rows &&
                nc >= 0 &&
                nc < cols &&
                !visited[nr][nc] &&
                grid[nr][nc] >= threshold
              ) {
                visited[nr][nc] = 1;
                queue.push([nr, nc]);
              }
            }
          }

          // Filter out tiny single-pixel noise clusters
          if (count >= 2) {
            const avgScore = sumScore / count;
            regions.push({
              box: {
                x: minC / cols,
                y: minR / rows,
                width: (maxC - minC + 1) / cols,
                height: (maxR - minR + 1) / rows,
              },
              score: Math.round(avgScore * 100) / 100,
              label: "Vùng mô hình nghi ngờ can thiệp",
            });
          }
        }
      }
    }

    return regions;
  }

  /**
   * Renders the 2D suspicion matrix to an RGBA pixel buffer using the Turbo colormap.
   */
  public static renderColormapRgba(
    grid: number[][],
    outWidth: number,
    outHeight: number,
  ): Uint8ClampedArray {
    const rows = grid.length;
    const cols = rows > 0 ? grid[0].length : 0;
    const rgba = new Uint8ClampedArray(outWidth * outHeight * 4);

    if (rows === 0 || cols === 0) return rgba;

    const scaleY = (rows - 1) / (outHeight - 1 || 1);
    const scaleX = (cols - 1) / (outWidth - 1 || 1);

    for (let y = 0; y < outHeight; y++) {
      const gy = y * scaleY;
      const y0 = Math.floor(gy);
      const y1 = Math.min(rows - 1, y0 + 1);
      const fy = gy - y0;

      for (let x = 0; x < outWidth; x++) {
        const gx = x * scaleX;
        const x0 = Math.floor(gx);
        const x1 = Math.min(cols - 1, x0 + 1);
        const fx = gx - x0;

        // Bilinear interpolation
        const v00 = grid[y0][x0];
        const v10 = grid[y0][x1];
        const v01 = grid[y1][x0];
        const v11 = grid[y1][x1];
        const val =
          (1 - fy) * ((1 - fx) * v00 + fx * v10) +
          fy * ((1 - fx) * v01 + fx * v11);

        // Turbo colormap approximation
        const [r, g, b] = this.turboColormap(val);
        const idx = (y * outWidth + x) * 4;

        rgba[idx] = r;
        rgba[idx + 1] = g;
        rgba[idx + 2] = b;
        rgba[idx + 3] = Math.round(val * 210); // Alpha scales with suspicion
      }
    }

    return rgba;
  }

  /**
   * Perceptually uniform Turbo colormap polynomial approximation.
   */
  private static turboColormap(t: number): [number, number, number] {
    const x = Math.max(0, Math.min(1, t));
    const r = Math.round(
      255 *
        Math.max(
          0,
          Math.min(
            1,
            0.1357 +
              x *
                (4.5974 -
                  x *
                    (42.3277 - x * (130.5887 - x * (150.5667 - x * 58.1375)))),
          ),
        ),
    );
    const g = Math.round(
      255 *
        Math.max(
          0,
          Math.min(
            1,
            0.0914 +
              x *
                (2.1856 +
                  x * (4.8052 - x * (14.0195 - x * (4.2109 + x * 2.7747)))),
          ),
        ),
    );
    const b = Math.round(
      255 *
        Math.max(
          0,
          Math.min(
            1,
            0.1067 +
              x *
                (12.5732 -
                  x * (83.812 - x * (280.983 - x * (365.176 - x * 151.2)))),
          ),
        ),
    );
    return [r, g, b];
  }
}
