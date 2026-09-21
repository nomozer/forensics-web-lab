import { ModelRegistry, ModelRegistryItem } from './types.js';

export interface RegistryValidationResult {
  valid: boolean;
  errors: string[];
}

/**
 * Validates the structural integrity of a ModelRegistry configuration.
 * Enforces strict rules:
 * - 'ready' / 'quantized' requires valid path, real sha256 hex string (64 chars), and sizeBytes > 0.
 * - 'not-trained' / 'not-installed' must NOT declare a fake non-empty path or fake positive size.
 * - Disallows test fixture / dummy / mock checkpoints in production registry entries.
 */
export function validateModelRegistry(
  registry: ModelRegistry,
  fsExistsSync?: (path: string) => boolean
): RegistryValidationResult {
  const errors: string[] = [];

  if (!registry || typeof registry !== 'object') {
    return { valid: false, errors: ['Registry must be a valid object.'] };
  }

  if (!Array.isArray(registry.models)) {
    return { valid: false, errors: ['Registry must contain a models array.'] };
  }

  const sha256Regex = /^[a-fA-F0-9]{64}$/;

  registry.models.forEach((item: ModelRegistryItem, index: number) => {
    const prefix = `Model[${index}] (${item.id || 'unnamed'}):`;

    if (!item.id || typeof item.id !== 'string') {
      errors.push(`${prefix} Missing or invalid 'id'.`);
    }

    // Guard against mock/fixture in production registry
    if (/(?:mock|fixture|dummy|fake)/i.test(item.id) || /(?:mock|fixture|dummy|fake)/i.test(item.path)) {
      errors.push(`${prefix} Test fixtures or mock paths cannot be registered in production model registry.`);
    }

    if (item.status === 'ready' || item.status === 'quantized') {
      if (!item.path || typeof item.path !== 'string' || item.path.trim() === '') {
        errors.push(`${prefix} Status '${item.status}' requires a valid non-empty 'path'.`);
      }
      if (!item.sha256 || !sha256Regex.test(item.sha256)) {
        errors.push(`${prefix} Status '${item.status}' requires a valid 64-character hex 'sha256' checksum.`);
      }
      if (typeof item.sizeBytes !== 'number' || item.sizeBytes <= 0) {
        errors.push(`${prefix} Status '${item.status}' requires 'sizeBytes' > 0.`);
      }
      if (fsExistsSync && item.path) {
        const exists = fsExistsSync(item.path);
        if (!exists) {
          errors.push(`${prefix} Referenced model file does not exist on disk: '${item.path}'.`);
        }
      }
    } else if (item.status === 'not-trained' || item.status === 'not-installed') {
      if (item.path && item.path.trim() !== '') {
        errors.push(`${prefix} Status '${item.status}' must not declare a fake or non-empty path.`);
      }
      if (typeof item.sizeBytes === 'number' && item.sizeBytes > 0) {
        errors.push(`${prefix} Status '${item.status}' must not declare a positive sizeBytes.`);
      }
      if (item.sha256 && item.sha256.trim() !== '') {
        errors.push(`${prefix} Status '${item.status}' must not declare a sha256 checksum.`);
      }
    }
  });

  return {
    valid: errors.length === 0,
    errors,
  };
}
