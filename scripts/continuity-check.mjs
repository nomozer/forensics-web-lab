#!/usr/bin/env node

/**
 * Continuity Checker for Forensics Web Lab
 * Cross-platform tool enforcing project continuity rules, integrity, and contracts.
 * 
 * Usage:
 *   node scripts/continuity-check.mjs
 *   node scripts/continuity-check.mjs --staged
 *   node scripts/continuity-check.mjs --base <commit> --head <commit>
 */

import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

// Parse command line arguments
export function parseArgs(args) {
  const options = {
    staged: false,
    base: null,
    head: null,
    cwd: process.cwd(),
    typo: false,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--staged') {
      options.staged = true;
    } else if (arg === '--base' && i + 1 < args.length) {
      options.base = args[++i];
    } else if (arg === '--head' && i + 1 < args.length) {
      options.head = args[++i];
    } else if (arg === '--cwd' && i + 1 < args.length) {
      options.cwd = path.resolve(args[++i]);
    } else if (arg === '--typo') {
      options.typo = true;
    }
  }

  return options;
}

// Find repository root from cwd
export function findRepoRoot(startDir) {
  let curr = path.resolve(startDir);
  while (true) {
    if (fs.existsSync(path.join(curr, '.git'))) {
      return curr;
    }
    const parent = path.dirname(curr);
    if (parent === curr) break;
    curr = parent;
  }
  return startDir;
}

// Retrieve changed files via git
export function getChangedFiles(cwd, options = {}) {
  const root = findRepoRoot(cwd);
  const changed = [];

  const runGit = (args) => {
    try {
      return execFileSync('git', args, { cwd: root, encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] }).trim();
    } catch {
      return '';
    }
  };

  if (options.base) {
    const head = options.head || 'HEAD';
    const output = runGit(['diff', '--name-status', options.base, head]);
    if (output) {
      for (const line of output.split('\n')) {
        const parts = line.trim().split(/\t+/);
        if (parts.length >= 2) {
          const status = parts[0].trim();
          const file = parts[parts.length - 1].trim().replace(/\\/g, '/');
          changed.push({ status, file });
        }
      }
    }
    return changed;
  }

  if (options.staged) {
    const output = runGit(['diff', '--cached', '--name-status']);
    if (output) {
      for (const line of output.split('\n')) {
        const parts = line.trim().split(/\t+/);
        if (parts.length >= 2) {
          const status = parts[0].trim();
          const file = parts[parts.length - 1].trim().replace(/\\/g, '/');
          changed.push({ status, file });
        }
      }
    }
    return changed;
  }

  // Default: check working tree changes vs HEAD + untracked files
  const diffOutput = runGit(['diff', '--name-status', 'HEAD']);
  if (diffOutput) {
    for (const line of diffOutput.split('\n')) {
      const parts = line.trim().split(/\t+/);
      if (parts.length >= 2) {
        const status = parts[0].trim();
        const file = parts[parts.length - 1].trim().replace(/\\/g, '/');
        changed.push({ status, file });
      }
    }
  }

  // Untracked files from status porcelain
  const statusOutput = runGit(['status', '--porcelain', '-uall']);
  if (statusOutput) {
    for (const line of statusOutput.split('\n')) {
      if (line.startsWith('?? ')) {
        const file = line.substring(3).trim().replace(/\\/g, '/');
        if (!changed.some(c => c.file === file)) {
          changed.push({ status: 'A', file });
        }
      }
    }
  }

  return changed;
}

// Parse phase tokens for sorting: e.g. "phase-4a.5" or "Phase 4A.5 — Title" -> { raw: "4a.5", tokens: [4, "a", 5] }
export function parsePhaseTokens(phaseStr) {
  if (!phaseStr) return { raw: '', tokens: [] };
  const cleaned = phaseStr
    .toLowerCase()
    .replace(/^phase[-_\s]*/, '')
    .trim()
    .split(/[\s—–-]+/)[0];
  const matches = cleaned.match(/(\d+|[a-z]+)/g) || [];
  return {
    raw: cleaned,
    tokens: matches.map(m => (/^\d+$/.test(m) ? parseInt(m, 10) : m)),
  };
}

export function comparePhaseTokens(aStr, bStr) {
  const a = parsePhaseTokens(aStr).tokens;
  const b = parsePhaseTokens(bStr).tokens;
  const len = Math.max(a.length, b.length);
  for (let i = 0; i < len; i++) {
    const valA = a[i];
    const valB = b[i];
    if (valA === undefined) return -1;
    if (valB === undefined) return 1;
    if (typeof valA === 'number' && typeof valB === 'number') {
      if (valA !== valB) return valA - valB;
    } else {
      const cmp = String(valA).localeCompare(String(valB));
      if (cmp !== 0) return cmp;
    }
  }
  return 0;
}

// Check diff against continuity rules
export function checkDiffRules(changedFiles, options = {}) {
  const errors = [];
  if (!changedFiles || changedFiles.length === 0) {
    return errors;
  }

  const hasCurrentState = changedFiles.some(f => f.file === 'docs/continuity/CURRENT_STATE.md');
  const hasCodeIndex = changedFiles.some(f => f.file === 'docs/continuity/CODE_INDEX.md');
  const hasStatusLedger = changedFiles.some(f => f.file === 'docs/continuity/STATUS_LEDGER.md');

  // Paths requiring CURRENT_STATE.md
  const statePrefixes = ['apps/', 'packages/', 'ml/', 'models/', 'datasets/'];
  const stateDocs = [
    'docs/RESEARCH_PLAN.md',
    'docs/EVALUATION.md',
    'docs/DATASETS.md',
    'docs/EVIDENCE_REGISTER.md',
    'docs/BACKLOG.md',
  ];

  const stateTriggerFiles = changedFiles.filter(f => {
    if (f.file.startsWith('docs/continuity/')) return false;
    return statePrefixes.some(prefix => f.file.startsWith(prefix)) || stateDocs.includes(f.file);
  });

  // Structural/module/contract changes requiring CODE_INDEX.md
  const codeIndexTriggerFiles = changedFiles.filter(f => {
    if (f.file.startsWith('docs/continuity/')) return false;
    // Added, deleted, or renamed code/module in packages, apps, ml, scripts
    const isCodeModule = ['packages/', 'apps/', 'ml/', 'scripts/'].some(p => f.file.startsWith(p));
    if (isCodeModule && (f.status.startsWith('A') || f.status.startsWith('D') || f.status.startsWith('R'))) {
      return true;
    }
    // Workspace or package changes
    if (f.file === 'pnpm-workspace.yaml' || f.file === 'package.json' || f.file.endsWith('/package.json')) {
      return true;
    }
    // Schema changes
    if (f.file.includes('schemas/') || f.file.endsWith('.schema.json')) {
      return true;
    }
    // Public contract changes
    if (f.file.includes('/contracts/') || f.file.includes('/types/') || f.file.includes('contract') || f.file.includes('types.ts')) {
      return true;
    }
    // CLI commands
    if (f.file === 'ml/datasets/acquire.py' || f.file === 'ml/configs/validator.py' || f.file.startsWith('scripts/')) {
      return true;
    }
    // Pipeline, config or registry
    if (f.file.startsWith('ml/configs/') || f.file === 'datasets/registry.json' || f.file.startsWith('datasets/acquisition-plans/')) {
      return true;
    }
    return false;
  });

  // Phase report trigger requiring STATUS_LEDGER.md and CURRENT_STATE.md
  const phaseReportFiles = changedFiles.filter(f => {
    return /^research\/evidence\/[^/]+\/PHASE_REPORT\.md$/.test(f.file);
  });

  // Rule 1: State changes require CURRENT_STATE.md (unless explicitly marked as typo)
  if (stateTriggerFiles.length > 0 && !hasCurrentState && !options.typo) {
    errors.push(
      `State-affecting changes detected in ${stateTriggerFiles.length} file(s) (e.g. ${stateTriggerFiles[0].file}), but 'docs/continuity/CURRENT_STATE.md' was not updated.`
    );
  }

  // Rule 2: Structural, module, contract, or CLI changes require CODE_INDEX.md
  if (codeIndexTriggerFiles.length > 0 && !hasCodeIndex && !options.typo) {
    errors.push(
      `Structural, module, contract, or CLI changes detected in ${codeIndexTriggerFiles.length} file(s) (e.g. ${codeIndexTriggerFiles[0].file}), but 'docs/continuity/CODE_INDEX.md' was not updated.`
    );
  }

  // Rule 3: Phase report changes require STATUS_LEDGER.md and CURRENT_STATE.md
  if (phaseReportFiles.length > 0) {
    if (!hasStatusLedger) {
      errors.push(
        `Phase report created or modified (${phaseReportFiles[0].file}), but 'docs/continuity/STATUS_LEDGER.md' was not updated.`
      );
    }
    if (!hasCurrentState) {
      errors.push(
        `Phase report created or modified (${phaseReportFiles[0].file}), but 'docs/continuity/CURRENT_STATE.md' was not updated.`
      );
    }
    // If structural changes also present, CODE_INDEX.md is required
    if (codeIndexTriggerFiles.length > 0 && !hasCodeIndex) {
      errors.push(
        `Phase report introduces structural/contract changes, but 'docs/continuity/CODE_INDEX.md' was not updated.`
      );
    }
  }

  return errors;
}

// Check repository integrity and canonical continuity invariants
export function checkRepositoryIntegrity(rootDir) {
  const errors = [];
  const continuityDir = path.join(rootDir, 'docs', 'continuity');

  // Invariant 1: Exactly three canonical continuity files exist
  const canonicalFiles = ['CURRENT_STATE.md', 'CODE_INDEX.md', 'STATUS_LEDGER.md'];
  for (const file of canonicalFiles) {
    const fullPath = path.join(continuityDir, file);
    if (!fs.existsSync(fullPath)) {
      errors.push(`Required continuity file missing: 'docs/continuity/${file}'.`);
    }
  }

  // Invariant 2: No extra files in docs/continuity/
  if (fs.existsSync(continuityDir)) {
    const files = fs.readdirSync(continuityDir);
    for (const f of files) {
      if (f.endsWith('.md') && !canonicalFiles.includes(f)) {
        errors.push(`Disallowed extra file in docs/continuity/: '${f}'.`);
      }
    }
  }

  // Invariant 3: No duplicate or numbered continuity files across the repository
  const scanDirs = ['docs', 'research', 'packages', 'apps', 'ml', 'scripts'];
  const duplicatePattern = /(CURRENT_STATE|CODE_INDEX|STATUS_LEDGER)[(_\s].*\.md$/i;
  for (const dir of scanDirs) {
    const fullDir = path.join(rootDir, dir);
    if (!fs.existsSync(fullDir)) continue;
    function walk(d) {
      const entries = fs.readdirSync(d, { withFileTypes: true });
      for (const ent of entries) {
        const p = path.join(d, ent.name);
        if (ent.isDirectory()) {
          if (!['node_modules', '.git', '.venv', 'dist', 'build', '.pytest_cache', '__pycache__'].includes(ent.name)) {
            walk(p);
          }
        } else if (duplicatePattern.test(ent.name)) {
          const rel = path.relative(rootDir, p).replace(/\\/g, '/');
          errors.push(`Duplicate/numbered continuity file detected: '${rel}'.`);
        }
      }
    }
    walk(fullDir);
  }

  // Read STATUS_LEDGER.md for latest phase and line count
  const ledgerPath = path.join(continuityDir, 'STATUS_LEDGER.md');
  let latestLedgerPhase = null;
  if (fs.existsSync(ledgerPath)) {
    const ledgerContent = fs.readFileSync(ledgerPath, 'utf-8');
    const ledgerLines = ledgerContent.split(/\r?\n/);
    let currentPhase = null;
    let lineCount = 0;

    for (let i = 0; i < ledgerLines.length; i++) {
      const line = ledgerLines[i];
      if (line.startsWith('## Phase ')) {
        if (!latestLedgerPhase) {
          const match = line.match(/^##\s+Phase\s+([^\n—–]+)/);
          if (match) latestLedgerPhase = match[1].trim();
        }
        if (currentPhase && lineCount > 20) {
          errors.push(
            `Phase section '${currentPhase}' in docs/continuity/STATUS_LEDGER.md has ${lineCount} lines (maximum allowed is 20 lines).`
          );
        }
        currentPhase = line.replace(/^##\s*/, '').trim();
        lineCount = 1;
      } else if (line.trim() === '---') {
        if (currentPhase) {
          if (lineCount > 20) {
            errors.push(
              `Phase section '${currentPhase}' in docs/continuity/STATUS_LEDGER.md has ${lineCount} lines (maximum allowed is 20 lines).`
            );
          }
          currentPhase = null;
          lineCount = 0;
        }
      } else if (currentPhase) {
        lineCount++;
      }
    }
    if (currentPhase && lineCount > 20) {
      errors.push(
        `Phase section '${currentPhase}' in docs/continuity/STATUS_LEDGER.md has ${lineCount} lines (maximum allowed is 20 lines).`
      );
    }
  }

  // Check research/evidence/ latest folder
  const evidenceDir = path.join(rootDir, 'research', 'evidence');
  let latestEvidencePhase = null;
  let latestEvidenceFolder = null;
  if (fs.existsSync(evidenceDir)) {
    const dirs = fs.readdirSync(evidenceDir, { withFileTypes: true })
      .filter(d => d.isDirectory() && d.name.startsWith('phase-'))
      .map(d => {
        const dirPath = path.join(evidenceDir, d.name);
        let timestamp = 0;
        try {
          for (const envFile of ['environment.json', 'analysis_environment.json', 'import_audit_summary.json', 'cross_phase_discrepancies.json']) {
            const envPath = path.join(dirPath, envFile);
            if (fs.existsSync(envPath)) {
              const envContent = fs.readFileSync(envPath, 'utf-8');
              const env = JSON.parse(envContent);
              const ts = env.timestamp || env.timestamp_utc;
              if (ts) {
                timestamp = new Date(ts).getTime();
                break;
              }
            }
          }
        } catch {}
        // Fallback: use git log for the directory
        if (timestamp === 0) {
          try {
            const gitLog = execFileSync('git', ['log', '-1', '--format=%ct', '--', dirPath], { cwd: rootDir, encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] }).trim();
            if (gitLog) timestamp = parseInt(gitLog, 10) * 1000;
          } catch {}
        }
        // Fallback: use filesystem stat mtime
        if (timestamp === 0) {
          try {
            const stat = fs.statSync(dirPath);
            timestamp = stat.mtimeMs;
          } catch {}
        }
        return { name: d.name, timestamp };
      })
      .filter(d => d.timestamp > 0)
      .sort((a, b) => {
        // Primary sort: timestamp (newer first)
        if (a.timestamp !== b.timestamp) return b.timestamp - a.timestamp;
        // Secondary sort: phase token comparison (higher version first)
        return comparePhaseTokens(b.name, a.name);
      });
    if (dirs.length > 0) {
      // Sorted descending (newest first), so first element is latest
      latestEvidenceFolder = dirs[0].name;
      latestEvidencePhase = latestEvidenceFolder.replace(/^phase-/, '');
    }
  }

  // Invariant 4: Latest phase in STATUS_LEDGER.md matches latest evidence directory
  if (latestLedgerPhase && latestEvidencePhase) {
    const parsedLedger = parsePhaseTokens(latestLedgerPhase).raw;
    const parsedEvidence = parsePhaseTokens(latestEvidencePhase).raw;
    if (parsedLedger !== parsedEvidence) {
      errors.push(
        `Latest phase in docs/continuity/STATUS_LEDGER.md ('${latestLedgerPhase}') does not match latest evidence directory ('${latestEvidenceFolder}').`
      );
    }
  }

  // Invariant 5: CURRENT_STATE.md references the latest phase
  const currentStatePath = path.join(continuityDir, 'CURRENT_STATE.md');
  if (fs.existsSync(currentStatePath) && latestLedgerPhase) {
    const currentContent = fs.readFileSync(currentStatePath, 'utf-8');
    const phaseToken = parsePhaseTokens(latestLedgerPhase).raw;
    if (!currentContent.toLowerCase().includes(phaseToken)) {
      errors.push(
        `'docs/continuity/CURRENT_STATE.md' does not reference the latest phase '${latestLedgerPhase}'.`
      );
    }
  }

  // Invariant 6: Latest PHASE_REPORT.md exists
  if (latestEvidenceFolder) {
    const reportPath = path.join(evidenceDir, latestEvidenceFolder, 'PHASE_REPORT.md');
    if (!fs.existsSync(reportPath)) {
      errors.push(`Latest phase report is missing: 'research/evidence/${latestEvidenceFolder}/PHASE_REPORT.md'.`);
    } else {
      const stat = fs.statSync(reportPath);
      if (stat.size < 50) {
        errors.push(`Latest phase report is suspiciously empty: 'research/evidence/${latestEvidenceFolder}/PHASE_REPORT.md'.`);
      }
    }
  }

  // Invariant 7: No machine-local links, 'See repository HEAD', or unfilled commit placeholders
  const filesToCheck = [
    'AGENTS.md',
    'README.md',
    'docs/continuity/CURRENT_STATE.md',
    'docs/continuity/CODE_INDEX.md',
    'docs/continuity/STATUS_LEDGER.md',
  ];

  if (latestEvidenceFolder) {
    filesToCheck.push(`research/evidence/${latestEvidenceFolder}/PHASE_REPORT.md`);
  }

  for (const rel of filesToCheck) {
    const full = path.join(rootDir, rel);
    if (!fs.existsSync(full)) continue;
    const content = fs.readFileSync(full, 'utf-8');
    const lines = content.split(/\r?\n/);

    for (let idx = 0; idx < lines.length; idx++) {
      const line = lines[idx];
      // Machine-local markdown link or path
      if (/\]\((file:\/\/\/|[A-Za-z]:[\\\/]|\/Users\/|\/home\/)/i.test(line)) {
        errors.push(`Machine-local link found in ${rel}:${idx + 1}: ${line.trim()}`);
      }
      // Commit field placeholders (e.g. commit: See repository HEAD, commit: <commit>, commit: Xác định sau commit báo cáo)
      const isCommitField = /(?:commit|snapshot|revision|hash)\s*[:=]/i.test(line);
      if (isCommitField) {
        if (line.includes('See repository HEAD')) {
          errors.push(`Placeholder 'See repository HEAD' found in field at ${rel}:${idx + 1}: ${line.trim()}`);
        }
        if (line.includes('<commit>') || line.includes('Xác định sau commit báo cáo') || /[:=]\s*TODO\b/i.test(line)) {
          errors.push(`Unfilled commit placeholder found in field at ${rel}:${idx + 1}: ${line.trim()}`);
        }
      }
    }
  }

  return errors;
}

// Master continuity check orchestrator
export function runContinuityCheck(options = {}) {
  const cwd = options.cwd || process.cwd();
  const root = findRepoRoot(cwd);

  const changedFiles = getChangedFiles(root, options);
  const diffErrors = checkDiffRules(changedFiles, options);
  const integrityErrors = checkRepositoryIntegrity(root);

  const allErrors = [...diffErrors, ...integrityErrors];
  return {
    pass: allErrors.length === 0,
    errors: allErrors,
    changedFiles,
  };
}

// CLI execution if executed directly
const isDirectExecution = Boolean(process.argv[1]) && (
  import.meta.url === `file://${process.argv[1].replace(/\\/g, '/')}` ||
  process.argv[1].endsWith('continuity-check.mjs')
);

if (isDirectExecution) {
  const options = parseArgs(process.argv.slice(2));
  const result = runContinuityCheck(options);

  if (result.pass) {
    console.log('CONTINUITY_CHECK: PASS');
    process.exit(0);
  } else {
    console.error('CONTINUITY_CHECK: FAIL');
    console.error('\nThe following continuity violations were detected:');
    for (const err of result.errors) {
      console.error(` - [!] ${err}`);
    }
    console.error('\nPlease update the required continuity documents according to AGENTS.md rules.\n');
    process.exit(1);
  }
}
