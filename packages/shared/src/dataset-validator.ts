export type DatasetTrack = 'research-only' | 'product-eligible' | 'blocked';
export type LicensePermission = 'allowed' | 'prohibited' | 'unclear';
export type DatasetAvailability = 'available' | 'pending' | 'unavailable';
export type DatasetRegistrationStatus = 'proposed' | 'verified' | 'blocked' | 'rejected';

export interface DatasetRegistryItem {
  id: string;
  name: string;
  version: string;
  track: DatasetTrack;
  officialSource: string;
  downloadSource: string;
  codeLicense: string;
  datasetLicense: string;
  additionalTerms: string[];
  commercialUse: LicensePermission;
  redistribution: LicensePermission;
  derivativeWeights: LicensePermission;
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

const VALID_TRACKS: DatasetTrack[] = ['research-only', 'product-eligible', 'blocked'];
const VALID_PERMISSIONS: LicensePermission[] = ['allowed', 'prohibited', 'unclear'];
const VALID_AVAILABILITY: DatasetAvailability[] = ['available', 'pending', 'unavailable'];
const VALID_STATUSES: DatasetRegistrationStatus[] = ['proposed', 'verified', 'blocked', 'rejected'];

/**
 * Validates the structural and legal compliance of a DatasetRegistry configuration.
 *
 * Rules:
 * - 'product-eligible' must NOT have commercialUse: 'prohibited' or 'unclear'.
 * - 'product-eligible' must NOT have derivativeWeights: 'prohibited' or 'unclear'.
 * - 'product-eligible' must NOT have datasetLicense: 'unverified'.
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

    if (!item.officialSource || typeof item.officialSource !== 'string' || item.officialSource.trim() === '') {
      errors.push(`${prefix} Official source must be specified; unofficial/third-party aggregations prohibited.`);
    }

    if (!item.datasetLicense || typeof item.datasetLicense !== 'string') {
      errors.push(`${prefix} Missing 'datasetLicense'.`);
    }

    if (!item.commercialUse || !VALID_PERMISSIONS.includes(item.commercialUse)) {
      errors.push(`${prefix} Invalid 'commercialUse'.`);
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
    }

    // Strict Verification Gate
    if (item.status === 'verified') {
      if (!Array.isArray(item.licenseEvidenceUrls) || item.licenseEvidenceUrls.length === 0) {
        errors.push(`${prefix} Status 'verified' requires at least one valid license evidence URL.`);
      }
    }

    // Blocked Consistency Gate
    if (item.track === 'blocked' && item.status !== 'blocked') {
      errors.push(`${prefix} Track 'blocked' must have status 'blocked'.`);
    }
  });

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Validates that a list of dataset IDs used to train a model complies with production requirements.
 * Rejects any model whose lineage contains a 'research-only' or 'blocked' dataset.
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

    if (dataset.track === 'research-only') {
      violations.push(
        `Contamination violation: Model lineage contains research-only dataset '${id}'. Research datasets cannot be promoted to production.`
      );
    } else if (dataset.track === 'blocked') {
      violations.push(`Blocked dataset violation: Model lineage contains blocked dataset '${id}'.`);
    } else if (dataset.commercialUse === 'prohibited') {
      violations.push(`Commercial prohibition: Dataset '${id}' prohibits commercial use and cannot be used in production.`);
    }
  }

  return {
    compliant: violations.length === 0,
    violations,
  };
}
