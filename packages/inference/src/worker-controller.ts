import { AnalysisProgressEvent, AnalysisResult } from '@forensics/shared';
import { WorkerAnalyzePayload } from './forensics.worker.js';

export interface AnalysisOptions {
  onProgress?: (event: AnalysisProgressEvent) => void;
  signal?: AbortSignal;
}

export class WorkerController {
  private worker: Worker | null = null;

  constructor(workerConstructor?: () => Worker) {
    if (typeof Worker !== 'undefined') {
      if (workerConstructor) {
        this.worker = workerConstructor();
      } else {
        // Vite standard worker URL constructor
        this.worker = new Worker(new URL('./forensics.worker.js', import.meta.url), {
          type: 'module',
        });
      }
    }
  }

  /**
   * Executes full forensic analysis inside the worker.
   */
  public async analyzeImage(
    payload: WorkerAnalyzePayload,
    options?: AnalysisOptions
  ): Promise<AnalysisResult> {
    if (!this.worker) {
      throw new Error('Web Worker is not supported or initialized in this environment.');
    }
    const worker = this.worker;

    return new Promise<AnalysisResult>((resolve, reject) => {
      const handleMessage = (e: MessageEvent) => {
        const { type, payload: msgPayload } = e.data;
        if (type === 'PROGRESS' && options?.onProgress) {
          options.onProgress(msgPayload as AnalysisProgressEvent);
        } else if (type === 'COMPLETE') {
          cleanup();
          resolve(msgPayload as AnalysisResult);
        } else if (type === 'ERROR') {
          cleanup();
          reject(new Error(msgPayload as string));
        }
      };

      const handleError = (err: ErrorEvent) => {
        cleanup();
        reject(err);
      };

      const handleAbort = () => {
        worker.postMessage({ type: 'CANCEL' });
        cleanup();
        reject(new Error('Phân tích đã bị hủy bỏ bởi người dùng.'));
      };

      if (options?.signal) {
        options.signal.addEventListener('abort', handleAbort);
      }

      const cleanup = () => {
        worker.removeEventListener('message', handleMessage);
        worker.removeEventListener('error', handleError);
        if (options?.signal) {
          options.signal.removeEventListener('abort', handleAbort);
        }
      };

      worker.addEventListener('message', handleMessage);
      worker.addEventListener('error', handleError);

      worker.postMessage({
        type: 'START_ANALYSIS',
        payload,
      });
    });
  }

  public terminate(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
  }
}
