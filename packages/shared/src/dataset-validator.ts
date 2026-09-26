export type DatasetTrack = 'research-only' | 'product-eligible' | 'blocked' | 'fixture-only';
export type LicensePermission = 'allowed' | 'prohibited' | 'unclear' | 'internal-testing-only';
export type DatasetAvailability = 'available' | 'pending' | 'unavailable';
export type DatasetRegistrationStatus = 'proposed' | 'verified' | 'blocked' | 'rejected';
export type DatasetPurpose =
  | 'fixture'
  | 'acquisition-smoke'
  | 'exploratory-pilot'
  | 'scientific-benchmark'
  | 'product-training';

export interface DatasetRegistryItem {
  id: string;
  name: string;
  version: string;
  track: DatasetTrack;
  purpose: DatasetPurpose;
  officialSource: string;
  downloadSource: string;
  codeLicense: string;
  datasetLicense: string;
  additionalTerms: string[];
  commercialUse: LicensePermission;
  redistribution: LicensePermission;
  derivativeWeights: LicensePermission;
  productionPromotion?: 'allowed' | 'prohibited-by-project-policy' | 'pending-evaluation';
  ownership?: 'project-generated' | 'third-party' | 'public-domain';
  licenseStatus?: 'verified' | 'unverified' | 'pending-project-license-decision';
  acquisitionEnabled?: boolean;
  approvalStatus?: 'approved' | 'pending-user-approval' | 'not-applicable';
  sourceTerms?: Record<string, unknown>;
  legalInterpretation?: Record<string, unknown>;
  projectPolicy?: Record<string, unknown>;
  metadataAccess?: string;
  metadataAccessReason?: string;
  remoteInventory?: unknown[];
  availability: DatasetAvailability;
  expectedDownloadBytes: number | null;
  checksumAvailable: boolean;
  licenseEvidenceUrls: string[];
  reviewedAt: string;
  status: DatasetRegistrationStatus;
  notes: string;
}

export interface DatasetRegistry {
  schemaVersion: '1.0.0';
  datasets: DatasetRegistryItem[];
}

export interface DatasetRegistryValidationResult {
  valid: boolean;
  errors: string[];
}

const VALID_TRACKS: DatasetTrack[] = ['research-only', 'product-eligible', 'blocked', 'fixture-only'];
const VALID_PURPOSES: DatasetPurpose[] = [
  'fixture',
  'acquisition-smoke',
  'exploratory-pilot',
  'scientific-benchmark',
  'product-training',
];
const VALID_PERMISSIONS: LicensePermission[] = ['allowed', 'prohibited', 'unclear', 'internal-testing-only'];
const VALID_AVAILABILITY: DatasetAvailability[] = ['available', 'pending', 'unavailable'];
const VALID_STATUSES: DatasetRegistrationStatus[] = ['proposed', 'verified', 'blocked', 'rejected'];

/**
 * Validates the structural and legal compliance of a DatasetRegistry configuration.
 *
 * Rules:
 * - 'product-eligible' must NOT have commercialUse: 'prohibited' or 'unclear' or 'internal-testing-only'.
 * - 'product-eligible' must NOT have derivativeWeights: 'prohibited' or 'unclear'.
 * - 'product-eligible' must NOT have datasetLicense: 'unverified'.
 * - 'product-eligible' must NOT have productionPromotion: 'prohibited-by-project-policy'.
 * - 'fixture-only' must have purpose: 'fixture'.
 * - 'fixture' purpose must have track: 'fixture-only'.
 * - 'product-training' purpose must have track: 'product-eligible'.
 * - Every item must declare a valid 'officialSource'.
 * - Status 'verified' requires at least one verified URL in 'licenseEvidenceUrls'.
 * - Status 'blocked' must not be marked 'product-eligible'.
 */
export function validateDatasetRegistry(data: unknown): DatasetRegistryValidationResult {
  const errors: string[] = [];

  if (!data || typeof data !== 'object') {
    return { valid: false, errors: ['Dataset registry must be a non-null object.'] };
  }

  const registry = data as Partial<DatasetRegistry>;

  if (registry.schemaVersion !== '1.0.0') {
    errors.push(`Invalid schemaVersion: expected '1.0.0', got '${registry.schemaVersion}'.`);
  }

  if (!Array.isArray(registry.datasets) || registry.datasets.length === 0) {
    errors.push('Dataset registry must contain a non-empty datasets array.');
    return { valid: errors.length === 0, errors };
  }

  const seenIds = new Set<string>();

  registry.datasets.forEach((item, index) => {
    const prefix = `Dataset[${index}] (${item?.id || 'unnamed'}):`;

    if (!item || typeof item !== 'object') {
      errors.push(`${prefix} Must be a valid object.`);
      return;
    }

    if (!item.id || typeof item.id !== 'string' || !/^[a-z0-9-]+$/.test(item.id)) {
      errors.push(`${prefix} Invalid or missing 'id'. Must match ^[a-z0-9-]+$.`);
    } else {
      if (seenIds.has(item.id)) {
        errors.push(`${prefix} Duplicate dataset ID: '${item.id}'.`);
      }
      seenIds.add(item.id);
    }

    if (!item.name || typeof item.name !== 'string') {
      errors.push(`${prefix} Missing 'name'.`);
    }

    if (!item.track || !VALID_TRACKS.includes(item.track)) {
      errors.push(`${prefix} Invalid 'track': '${item.track}'. Expected one of: ${VALID_TRACKS.join(', ')}.`);
    }

    if (!item.purpose || !VALID_PURPOSES.includes(item.purpose)) {
      errors.push(`${prefix} Invalid 'purpose': '${item.purpose}'. Expected one of: ${VALID_PURPOSES.join(', ')}.`);
    }

    // Purpose vs Track consistency
    if (item.track === 'fixture-only' && item.purpose !== 'fixture') {
      errors.push(`${prefix} Track 'fixture-only' must have purpose 'fixture', got '${item.purpose}'.`);
    }
    if (item.purpose === 'fixture' && item.track !== 'fixture-only') {
      errors.push(`${prefix} Purpose 'fixture' must be on 'fixture-only' track, got '${item.track}'.`);
    }
    if (item.purpose === 'product-training' && item.track !== 'product-eligible') {
      errors.push(`${prefix} Purpose 'product-training' must have track 'product-eligible', got '${item.track}'.`);
    }

    if (!item.officialSource || typeof item.officialSource !== 'string' || item.officialSource.trim() === '') {
      errors.push(`${prefix} Official source must be specified; unofficial/third-party aggregations prohibited.`);
    }

    if (!item.datasetLicense || typeof item.datasetLicense !== 'string') {
      errors.push(`${prefix} Missing 'datasetLicense'.`);
    }

    if (!item.commercialUse || !VALID_PERMISSIONS.includes(item.commercialUse)) {
      errors.push(`${prefix} Invalid 'commercialUse': '${item.commercialUse}'.`);
    }

    if (!item.redistribution || !VALID_PERMISSIONS.includes(item.redistribution)) {
      errors.push(`${prefix} Invalid 'redistribution'.`);
    }

    if (!item.derivativeWeights || !VALID_PERMISSIONS.includes(item.derivativeWeights)) {
      errors.push(`${prefix} Invalid 'derivativeWeights'.`);
    }

    if (!item.availability || !VALID_AVAILABILITY.includes(item.availability)) {
      errors.push(`${prefix} Invalid 'availability'.`);
    }

    if (!item.status || !VALID_STATUSES.includes(item.status)) {
      errors.push(`${prefix} Invalid 'status'.`);
    }

    // Strict Product-Eligible Gate
    if (item.track === 'product-eligible') {
      if (item.commercialUse !== 'allowed') {
        errors.push(`${prefix} Product-eligible dataset must have commercialUse: 'allowed', got '${item.commercialUse}'.`);
      }
      if (item.derivativeWeights !== 'allowed') {
        errors.push(`${prefix} Product-eligible dataset must have derivativeWeights: 'allowed', got '${item.derivativeWeights}'.`);
      }
      if (item.datasetLicense.toLowerCase() === 'unverified') {
        errors.push(`${prefix} Product-eligible dataset cannot have unverified license.`);
      }
      if (item.productionPromotion === 'prohibited-by-project-policy') {
        errors.push(`${prefix} Product-eligible dataset cannot have productionPromotion: 'prohibited-by-project-policy'.`);
      }
    }

    // Strict Verification Gate
    if (item.status === 'verified') {
      if (!Array.isArray(item.licenseEvidenceUrls) || item.licenseEvidenceUrls.length === 0) {
        errors.push(`${prefix} Status 'verified' requires at least one valid license evidence URL.`);
      }
    }

    // Blocked Consistency Gate
    if (item.track === 'blocked') {
      if (item.status !== 'blocked' && item.status !== 'proposed') {
        errors.push(`${prefix} Track 'blocked' must have status 'blocked' or 'proposed'.`);
      }
      if (item.acquisitionEnabled === true) {
        errors.push(`${prefix} Blocked dataset must not have acquisitionEnabled: true.`);
      }
    }
  });

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Validates that a list of dataset IDs used to train a model complies with production requirements.
 * Rejects any model whose lineage contains a 'research-only', 'fixture-only', or 'blocked' dataset,
 * or where derivative weights or commercial use are prohibited/unclear.
 */
export function validateProductionModelLineage(
  datasetIds: string[],
  registry: DatasetRegistry
): { compliant: boolean; violations: string[] } {
  const violations: string[] = [];

  if (!Array.isArray(datasetIds) || datasetIds.length === 0) {
    return {
      compliant: false,
      violations: ['Model must specify at least one training dataset ID in its lineage.'],
    };
  }

  const datasetMap = new Map<string, DatasetRegistryItem>(
    registry.datasets.map((d) => [d.id, d])
  );

  for (const id of datasetIds) {
    const dataset = datasetMap.get(id);
    if (!dataset) {
      violations.push(`Dataset '${id}' is not registered in datasets/registry.json.`);
      continue;
    }

    if (dataset.track === 'fixture-only') {
      violations.push(
        `Fixture violation: Model lineage contains fixture-only dataset '${id}'. Fixtures are for pipeline testing only and cannot be used in model training.`
      );
    } else if (dataset.track === 'research-only') {
      violations.push(
        `Contamination violation: Model lineage contains research-only dataset '${id}'. Research datasets cannot be promoted to production.`
      );
    } else if (dataset.track === 'blocked') {
      violations.push(`Blocked dataset violation: Model lineage contains blocked dataset '${id}'.`);
    } else if (dataset.commercialUse === 'prohibited') {
      violations.push(`Commercial prohibition: Dataset '${id}' prohibits commercial use and cannot be used in production.`);
    } else if (dataset.commercialUse === 'internal-testing-only') {
      violations.push(`Testing limitation: Dataset '${id}' is restricted to internal testing only.`);
    }

    if (dataset.derivativeWeights === 'unclear') {
      violations.push(
        `Derivative weights uncertainty: Dataset '${id}' has derivative weights 'unclear'. Production models require explicit permission.`
      );
    } else if (dataset.derivativeWeights === 'prohibited') {
      violations.push(
        `Derivative weights prohibition: Dataset '${id}' prohibits commercial derivative weights.`
      );
    }

    if (dataset.productionPromotion === 'prohibited-by-project-policy') {
      violations.push(
        `Project policy violation: Dataset '${id}' is marked 'prohibited-by-project-policy' for production promotion.`
      );
    }
  }

  return {
    compliant: violations.length === 0,
    violations,
  };
}
