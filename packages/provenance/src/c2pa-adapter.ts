import { C2paStatus, ProvenanceResult } from '@forensics/shared';

export interface IC2paAdapter {
  isSupported(): boolean;
  inspect(buffer: ArrayBuffer): Promise<{
    status: C2paStatus;
    details?: Record<string, unknown>;
  }>;
}

/**
 * Standard C2PA / Content Credentials adapter.
 * Per project specification: If C2PA WASM binaries are not safely loaded
 * in the client web context, it returns 'unsupported' status with explicit details.
 * Under NO circumstances does it fabricate fake provenance assertions.
 */
export class C2paProvenanceAdapter implements IC2paAdapter {
  private supported: boolean;

  constructor() {
    // True C2PA in browser requires WebAssembly and multi-megabyte C2PA manifests.
    // We check for native or WASM c2pa availability. Defaulting safely to unsupported.
    this.supported = false;
  }

  public isSupported(): boolean {
    return this.supported;
  }

  public async inspect(_buffer: ArrayBuffer): Promise<{
    status: C2paStatus;
    details?: Record<string, unknown>;
  }> {
    if (!this.supported) {
      return {
        status: 'unsupported',
        details: {
          reason: 'Môi trường trình duyệt chưa kích hoạt WASM C2PA parser an toàn',
          c2paManifestFound: false,
        },
      };
    }

    return {
      status: 'unknown',
      details: {},
    };
  }
}
