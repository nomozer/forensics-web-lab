export type EvidenceCategory =
  | 'git-state'
  | 'test-ts'
  | 'test-py'
  | 'build'
  | 'no-model'
  | 'registry'
  | 'onnx-pipeline'
  | 'browser-runtime'
  | 'model-size'
  | 'license'
  | 'dataset'
  | 'science';

export type EvidenceStatus =
  | 'verified'
  | 'reported'
  | 'estimated'
  | 'architecture-only'
  | 'pipeline-only'
  | 'unverified'
  | 'blocked'
  | 'rejected';

export interface EvidenceItem {
  evidenceId: string;
  claim: string;
  category: EvidenceCategory;
  status: EvidenceStatus;
  artifact: string;
  reproductionCommand: string;
  result: string;
  limitations: string;
}

export interface EvidenceManifest {
  schemaVersion: '1.0.0';
  phase: string;
  commitAudited: string;
  timestampUtc: string;
  items: EvidenceItem[];
}

export interface EvidenceValidationResult {
  valid: boolean;
  errors: string[];
}

const VALID_CATEGORIES: EvidenceCategory[] = [
  'git-state',
  'test-ts',
  'test-py',
  'build',
  'no-model',
  'registry',
  'onnx-pipeline',
  'browser-runtime',
  'model-size',
  'license',
  'dataset',
  'science',
];

const VALID_STATUSES: EvidenceStatus[] = [
  'verified',
  'reported',
  'estimated',
  'architecture-only',
  'pipeline-only',
  'unverified',
  'blocked',
  'rejected',
];

const EVIDENCE_ID_REGEX = /^EV(?:-[A-Z0-9]+)+-[0-9]{3}$/;
const COMMIT_REGEX = /^[a-fA-F0-9]{7,40}$/;
const PHASE_REGEX = /^Phase [0-9A-Za-z.]+$/;
const ABSOLUTE_PATH_REGEX = /(?:^[a-zA-Z]:[\\/]|^\/Users\/|^\/home\/|^file:\/\/\/)/i;

/**
 * Validates an EvidenceManifest object against required rules.
 */
export function validateEvidenceManifest(data: unknown): EvidenceValidationResult {
  const errors: string[] = [];

  if (!data || typeof data !== 'object') {
    return { valid: false, errors: ['Manifest must be a non-null object.'] };
  }

  const manifest = data as Partial<EvidenceManifest>;

  if (manifest.schemaVersion !== '1.0.0') {
    errors.push(`Invalid schemaVersion: expected '1.0.0', got '${manifest.schemaVersion}'.`);
  }

  if (!manifest.phase || !PHASE_REGEX.test(manifest.phase)) {
    errors.push(`Invalid phase: '${manifest.phase}'. Must match format like 'Phase 3.6'.`);
  }

  if (!manifest.commitAudited || !COMMIT_REGEX.test(manifest.commitAudited)) {
    errors.push(`Invalid commitAudited: '${manifest.commitAudited}'. Must be 7 to 40 hex chars.`);
  }

  if (!manifest.timestampUtc || isNaN(Date.parse(manifest.timestampUtc))) {
    errors.push(`Invalid timestampUtc: '${manifest.timestampUtc}'. Must be a valid ISO date-time.`);
  }

  if (!Array.isArray(manifest.items) || manifest.items.length === 0) {
    errors.push('Manifest must contain a non-empty items array.');
    return { valid: errors.length === 0, errors };
  }

  const seenIds = new Set<string>();

  manifest.items.forEach((item, index) => {
    const prefix = `Item[${index}] (${item?.evidenceId || 'unnamed'}):`;

    if (!item || typeof item !== 'object') {
      errors.push(`${prefix} Must be an object.`);
      return;
    }

    if (!item.evidenceId || !EVIDENCE_ID_REGEX.test(item.evidenceId)) {
      errors.push(`${prefix} Invalid evidenceId format. Expected 'EV-CATEGORY-001'.`);
    } else {
      if (seenIds.has(item.evidenceId)) {
        errors.push(`${prefix} Duplicate evidenceId: '${item.evidenceId}'.`);
      }
      seenIds.add(item.evidenceId);
    }

    if (!item.claim || typeof item.claim !== 'string' || item.claim.trim() === '') {
      errors.push(`${prefix} Missing claim.`);
    }

    if (!VALID_CATEGORIES.includes(item.category)) {
      errors.push(`${prefix} Invalid category '${item.category}'.`);
    }

    if (!VALID_STATUSES.includes(item.status)) {
      errors.push(`${prefix} Invalid status '${item.status}'.`);
    }

    if (!item.artifact || typeof item.artifact !== 'string') {
      errors.push(`${prefix} Missing artifact path.`);
    } else if (ABSOLUTE_PATH_REGEX.test(item.artifact)) {
      errors.push(`${prefix} Artifact path must be relative to repository root; absolute machine paths prohibited: '${item.artifact}'.`);
    }

    if (!item.reproductionCommand || typeof item.reproductionCommand !== 'string') {
      errors.push(`${prefix} Missing reproductionCommand.`);
    }

    if (!item.result || typeof item.result !== 'string') {
      errors.push(`${prefix} Missing result.`);
    }

    if (!item.limitations || typeof item.limitations !== 'string') {
      errors.push(`${prefix} Missing limitations.`);
    }

    // Strict rule: cannot use 'verified' without concrete artifact, command, and measured result
    if (item.status === 'verified') {
      if (!item.artifact || item.artifact === 'none' || item.artifact === 'N/A') {
        errors.push(`${prefix} Status 'verified' requires a concrete artifact path.`);
      }
      if (!item.reproductionCommand || item.reproductionCommand === 'none' || item.reproductionCommand === 'N/A') {
        errors.push(`${prefix} Status 'verified' requires a valid reproduction command.`);
      }
      if (/(?:not measured|not evaluated|unverified|none)/i.test(item.result)) {
        errors.push(`${prefix} Status 'verified' cannot have unmeasured or empty result.`);
      }
    }
  });

  return {
    valid: errors.length === 0,
    errors,
  };
}
