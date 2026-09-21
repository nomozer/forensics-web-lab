/**
 * Canonical 4-state verdict representing the forensic determination.
 * NEVER use "real" or "authentic" as an absolute assertion.
 */
export type AnalysisVerdict = 'no_ai_evidence' | 'fully_generated' | 'ai_edited' | 'uncertain';

/**
 * Internal dataset annotation label (mapped in UI to AnalysisVerdict).
 */
export type InternalDatasetLabel = 'authentic' | 'fully_generated' | 'ai_edited';

export interface ClassProbabilities {
  no_ai_evidence: number;
  fully_generated: number;
  ai_edited: number;
}

export interface ImageMetadataInfo {
  filename: string;
  fileSizeBytes: number;
  mimeType: string;
  width: number;
  height: number;
  sha256: string;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface SuspiciousRegion {
  box: BoundingBox;
  score: number;
  label: string;
}

export interface LocalizationResult {
  available: boolean;
  suspiciousAreaRatio: number | null;
  heatmapResolution?: {
    rows: number;
    cols: number;
  } | undefined;
  regions: SuspiciousRegion[];
  rawHeatmapGrid?: number[][] | undefined;
}

export type C2paStatus =
  | 'present_valid'
  | 'present_invalid'
  | 'absent'
  | 'unsupported'
  | 'unknown';

export interface MetadataSummaryItem {
  tag: string;
  value: string;
  category: string;
  suspicionScore: number;
}

export interface ProvenanceResult {
  c2paStatus: C2paStatus;
  c2paDetails?: Record<string, unknown> | undefined;
  metadataSummary: MetadataSummaryItem[];
}

export type ForensicSignalCategory = 'frequency' | 'compression' | 'noise' | 'metadata' | 'model';

export interface ForensicSignal {
  id: string;
  name: string;
  category: ForensicSignalCategory;
  score: number; // Normalized 0.0 to 1.0 (higher = more anomalous/indicative of AI)
  interpretation: string;
  details: Record<string, unknown>;
}

export interface ForensicSummary {
  signals: ForensicSignal[];
  supportingEvidence: string[];
  refutingEvidence: string[];
}

export type ExecutionBackend = 'wasm' | 'webgpu' | 'cpu-fallback' | 'none';

export interface RuntimeMetrics {
  modelId: string;
  modelVersion: string;
  backend: ExecutionBackend;
  durationMs: number;
  deviceInfo?: {
    userAgent?: string | undefined;
    hardwareConcurrency?: number | undefined;
    deviceMemoryGb?: number | undefined;
  } | undefined;
}

export interface AnalysisResult {
  schemaVersion: '1.0.0';
  timestamp: string;
  imageInfo: ImageMetadataInfo;
  modelAvailable: boolean;
  modelStatus: 'not-installed' | 'not-trained' | 'training' | 'installed' | 'ready';
  result: {
    label: AnalysisVerdict;
    confidence: number | null;
    probabilities: ClassProbabilities | null;
    explanation: string;
  };
  localization: LocalizationResult;
  provenance: ProvenanceResult;
  forensics: ForensicSummary;
  runtime: RuntimeMetrics;
  limitations: string[];
}

export type AnalysisProgressStage =
  | 'validating'
  | 'reading_metadata'
  | 'loading_model'
  | 'running_global_model'
  | 'scanning_patches'
  | 'calculating_forensic_signals'
  | 'calibrating_evidence'
  | 'building_report'
  | 'done'
  | 'error'
  | 'cancelled';

export interface AnalysisProgressEvent {
  stage: AnalysisProgressStage;
  progress: number; // 0.0 to 1.0
  message: string;
  currentPatch?: number | undefined;
  totalPatches?: number | undefined;
}

export type ModelLifecycleStatus =
  | 'not-trained'
  | 'not-installed'
  | 'training'
  | 'trained'
  | 'quantized'
  | 'ready'
  | 'deprecated';

export interface ModelRegistryItem {
  id: string;
  version: string;
  path: string;
  sha256: string;
  sizeBytes: number;
  inputShape: number[];
  classes: string[];
  opset: number;
  quantization: 'none' | 'int8' | 'fp16' | 'int4';
  status: ModelLifecycleStatus;
}

export interface ModelRegistry {
  version: string;
  models: ModelRegistryItem[];
}
