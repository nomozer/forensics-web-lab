import { ExecutionBackend, ModelRegistryItem } from "@forensics/shared";
import * as ort from "onnxruntime-web";

export interface SessionInitResult {
  session: ort.InferenceSession | null;
  backend: ExecutionBackend;
  modelInfo: ModelRegistryItem | null;
  installed: boolean;
  error?: string;
}

export class OnnxSessionManager {
  private static activeSession: ort.InferenceSession | null = null;
  private static activeBackend: ExecutionBackend = "none";

  /**
   * Probes whether WebGPU is accessible in the current browser/worker context.
   */
  public static async isWebGpuAvailable(): Promise<boolean> {
    try {
      if (typeof navigator === "undefined" || !("gpu" in navigator))
        return false;
      const gpu = (
        navigator as unknown as {
          gpu?: { requestAdapter: () => Promise<unknown> };
        }
      ).gpu;
      if (!gpu) return false;
      const adapter = await gpu.requestAdapter();
      return !!adapter;
    } catch {
      return false;
    }
  }

  /**
   * Initializes the ONNX inference session with automatic WebGPU -> WASM fallback.
   */
  public static async initializeSession(
    modelItem: ModelRegistryItem | null,
    modelBuffer?: ArrayBuffer,
  ): Promise<SessionInitResult> {
    if (
      !modelItem ||
      modelItem.status === "not-trained" ||
      !modelBuffer ||
      modelBuffer.byteLength === 0
    ) {
      return {
        session: null,
        backend: "none",
        modelInfo: modelItem,
        installed: false,
        error: "Model not installed",
      };
    }

    // Attempt 1: WebGPU if available
    const canUseGpu = await this.isWebGpuAvailable();
    if (canUseGpu) {
      try {
        const session = await ort.InferenceSession.create(modelBuffer, {
          executionProviders: ["webgpu"],
          graphOptimizationLevel: "all",
        });
        this.activeSession = session;
        this.activeBackend = "webgpu";
        return {
          session,
          backend: "webgpu",
          modelInfo: modelItem,
          installed: true,
        };
      } catch (err) {
        console.warn(
          "WebGPU execution provider failed. Gracefully falling back to WASM...",
          err,
        );
      }
    }

    // Attempt 2: WASM with CPU threads
    try {
      // Configure WASM threads/SIMD
      ort.env.wasm.numThreads =
        typeof navigator !== "undefined"
          ? Math.min(4, navigator.hardwareConcurrency || 2)
          : 2;
      ort.env.wasm.simd = true;

      const session = await ort.InferenceSession.create(modelBuffer, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      this.activeSession = session;
      this.activeBackend = "wasm";
      return {
        session,
        backend: "wasm",
        modelInfo: modelItem,
        installed: true,
      };
    } catch (err) {
      console.error("WASM execution provider failed:", err);
      return {
        session: null,
        backend: "none",
        modelInfo: modelItem,
        installed: false,
        error: String(err),
      };
    }
  }

  public static getSession(): ort.InferenceSession | null {
    return this.activeSession;
  }

  public static getBackend(): ExecutionBackend {
    return this.activeBackend;
  }
}
