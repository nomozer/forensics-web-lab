import { AnalysisResult } from './types.js';

/**
 * Validates whether an object adheres to the AnalysisResult v1 contract.
 */
export function validateAnalysisResult(obj: unknown): obj is AnalysisResult {
  if (!obj || typeof obj !== 'object') return false;
  const cand = obj as Partial<AnalysisResult>;

  if (cand.schemaVersion !== '1.0.0') return false;
  if (!cand.timestamp || typeof cand.timestamp !== 'string') return false;
  if (!cand.imageInfo || typeof cand.imageInfo !== 'object') return false;
  if (typeof cand.modelAvailable !== 'boolean') return false;
  if (typeof cand.modelStatus !== 'string') return false;
  if (!cand.result || typeof cand.result !== 'object') return false;
  if (!['no_ai_evidence', 'fully_generated', 'ai_edited', 'uncertain'].includes(cand.result.label ?? '')) {
    return false;
  }
  if (cand.result.confidence !== null && typeof cand.result.confidence !== 'number') return false;
  if (cand.result.probabilities !== null && typeof cand.result.probabilities !== 'object') return false;

  // Strict honest no-model invariant
  if (!cand.modelAvailable || cand.modelStatus === 'not-installed') {
    if (cand.result.label !== 'uncertain') return false;
    if (cand.result.confidence !== null) return false;
    if (cand.result.probabilities !== null) return false;
  }

  if (!cand.localization || typeof cand.localization !== 'object') return false;
  if (!cand.provenance || typeof cand.provenance !== 'object') return false;
  if (!cand.forensics || typeof cand.forensics !== 'object') return false;
  if (!cand.runtime || typeof cand.runtime !== 'object') return false;
  if (!Array.isArray(cand.limitations)) return false;

  return true;
}
