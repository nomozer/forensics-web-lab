import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { execFileSync } from 'node:child_process';
import {
  runContinuityCheck,
  checkDiffRules,
  checkRepositoryIntegrity,
  parsePhaseTokens,
  comparePhaseTokens,
} from '../continuity-check.mjs';

function createTempGitRepo() {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'continuity-test-'));

  const git = (args) => {
    return execFileSync('git', args, { cwd: tmpDir, encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] });
  };

  git(['init', '-b', 'main']);
  git(['config', 'user.email', 'test@example.com']);
  git(['config', 'user.name', 'Test Agent']);

  // Create baseline structure
  fs.mkdirSync(path.join(tmpDir, 'docs', 'continuity'), { recursive: true });
  fs.mkdirSync(path.join(tmpDir, 'research', 'evidence', 'phase-4a.4'), { recursive: true });
  fs.mkdirSync(path.join(tmpDir, 'packages', 'shared', 'src'), { recursive: true });
  fs.mkdirSync(path.join(tmpDir, 'ml'), { recursive: true });
  fs.mkdirSync(path.join(tmpDir, 'scripts'), { recursive: true });

  fs.writeFileSync(
    path.join(tmpDir, 'docs', 'continuity', 'CURRENT_STATE.md'),
    `# Current State\n\nPhase: 4A.4\nStatus: verified\nEnding commit: a480776\n`
  );

  fs.writeFileSync(
    path.join(tmpDir, 'docs', 'continuity', 'CODE_INDEX.md'),
    `# Code Index\n\n- packages/shared\n- ml\n`
  );

  fs.writeFileSync(
    path.join(tmpDir, 'docs', 'continuity', 'STATUS_LEDGER.md'),
    `# Status Ledger\n\n## Phase 4A.4 — Test\n- Starting commit: 9ab0e67\n- Ending commit: a480776\n- Status: PASS\n\n---\n`
  );

  fs.writeFileSync(
    path.join(tmpDir, 'research', 'evidence', 'phase-4a.4', 'PHASE_REPORT.md'),
    `# Phase 4A.4 Report\n\nVerdict: PASS\nEnding commit: a480776\n`
  );

  fs.writeFileSync(path.join(tmpDir, 'AGENTS.md'), `# AGENTS\n\nContinuity rules apply.\n`);
  fs.writeFileSync(path.join(tmpDir, 'README.md'), `# Forensics Web Lab\n`);
  fs.writeFileSync(path.join(tmpDir, 'package.json'), `{\n  "name": "root"\n}\n`);

  git(['add', '.']);
  git(['commit', '-m', 'Initial commit']);

  return {
    tmpDir,
    git,
    cleanup: () => {
      try {
        fs.rmSync(tmpDir, { recursive: true, force: true });
      } catch {
        // ignore cleanup error on Windows file locks
      }
    },
  };
}

describe('Continuity Checker Unit & Contract Suite', () => {
  describe('Phase Token Parsing and Comparison', () => {
    it('accurately parses and compares sequential phases', () => {
      assert.strictEqual(comparePhaseTokens('phase-0', 'phase-1'), -1);
      assert.strictEqual(comparePhaseTokens('phase-3.5', 'phase-3.6'), -1);
      assert.strictEqual(comparePhaseTokens('phase-3.6', 'phase-4a.0'), -1);
      assert.strictEqual(comparePhaseTokens('phase-4a.3', 'phase-4a.4'), -1);
      assert.strictEqual(comparePhaseTokens('phase-4a.4', 'Phase 4A.4 — Title'), 0);
      assert.strictEqual(comparePhaseTokens('phase-4a.5', 'phase-4a.4'), 1);
    });
  });

  describe('Diff Rules Evaluation', () => {
    it('1. Typo in non-state file (README.md) -> PASS without continuity file updates', () => {
      const changed = [{ status: 'M', file: 'README.md' }];
      const errors = checkDiffRules(changed);
      assert.strictEqual(errors.length, 0);
    });

    it('2. Model status change in ml/ -> requires CURRENT_STATE.md', () => {
      const changed = [{ status: 'M', file: 'ml/models/forensics.py' }];
      const errors = checkDiffRules(changed);
      assert.strictEqual(errors.length, 1);
      assert.match(errors[0], /CURRENT_STATE\.md/);
    });

    it('3. Add new module -> requires CURRENT_STATE.md and CODE_INDEX.md', () => {
      const changed = [{ status: 'A', file: 'packages/dsp/src/filter.ts' }];
      const errors = checkDiffRules(changed);
      assert.strictEqual(errors.length, 2);
      assert.ok(errors.some(e => e.includes('CURRENT_STATE.md')));
      assert.ok(errors.some(e => e.includes('CODE_INDEX.md')));
    });

    it('4. Create phase report -> requires CURRENT_STATE.md and STATUS_LEDGER.md', () => {
      const changed = [
        { status: 'A', file: 'research/evidence/phase-4a.5/PHASE_REPORT.md' },
      ];
      const errors = checkDiffRules(changed);
      assert.strictEqual(errors.length, 2);
      assert.ok(errors.some(e => e.includes('CURRENT_STATE.md')));
      assert.ok(errors.some(e => e.includes('STATUS_LEDGER.md')));
    });

    it('5. Create phase report with new module -> requires all three continuity files', () => {
      const changed = [
        { status: 'A', file: 'research/evidence/phase-4a.5/PHASE_REPORT.md' },
        { status: 'A', file: 'packages/new-module/index.ts' },
      ];
      const errors = checkDiffRules(changed);
      assert.ok(errors.some(e => e.includes('CURRENT_STATE.md')));
      assert.ok(errors.some(e => e.includes('STATUS_LEDGER.md')));
      assert.ok(errors.some(e => e.includes('CODE_INDEX.md')));
    });
  });

  describe('Isolated Git Repository Scenarios', () => {
    let repo;

    before(() => {
      repo = createTempGitRepo();
    });

    after(() => {
      repo.cleanup();
    });

    it('Clean baseline repository passes continuity check', () => {
      const result = runContinuityCheck({ cwd: repo.tmpDir });
      assert.strictEqual(result.pass, true, `Expected PASS but got: ${result.errors.join('; ')}`);
    });

    it('6. Missing latest phase report -> FAIL', () => {
      const testRepo = createTempGitRepo();
      try {
        // Add new phase in STATUS_LEDGER without creating folder
        fs.writeFileSync(
          path.join(testRepo.tmpDir, 'docs', 'continuity', 'STATUS_LEDGER.md'),
          `# Status Ledger\n\n## Phase 4A.6 — Nonexistent\n- Status: PASS\n\n---\n`
        );
        fs.writeFileSync(
          path.join(testRepo.tmpDir, 'docs', 'continuity', 'CURRENT_STATE.md'),
          `# Current State\n\nPhase 4A.6\n`
        );
        const result = runContinuityCheck({ cwd: testRepo.tmpDir });
        assert.strictEqual(result.pass, false);
        assert.ok(result.errors.some(e => e.includes('STATUS_LEDGER.md') || e.includes('phase report is missing')));
      } finally {
        testRepo.cleanup();
      }
    });

    it('7. Duplicate continuity file -> FAIL', () => {
      const testRepo = createTempGitRepo();
      try {
        fs.writeFileSync(
          path.join(testRepo.tmpDir, 'docs', 'CURRENT_STATE(1).md'),
          `# Duplicate\n`
        );
        const result = runContinuityCheck({ cwd: testRepo.tmpDir });
        assert.strictEqual(result.pass, false);
        assert.ok(result.errors.some(e => e.includes('Duplicate/numbered continuity file')));
      } finally {
        testRepo.cleanup();
      }
    });

    it('8. Unresolved placeholder See repository HEAD -> FAIL', () => {
      const testRepo = createTempGitRepo();
      try {
        fs.appendFileSync(
          path.join(testRepo.tmpDir, 'docs', 'continuity', 'CURRENT_STATE.md'),
          `Ending commit: See repository HEAD\n`
        );
        const result = runContinuityCheck({ cwd: testRepo.tmpDir });
        assert.strictEqual(result.pass, false);
        assert.ok(result.errors.some(e => e.includes('See repository HEAD')));
      } finally {
        testRepo.cleanup();
      }
    });

    it('9. Local machine link -> FAIL', () => {
      const testRepo = createTempGitRepo();
      try {
        fs.appendFileSync(
          path.join(testRepo.tmpDir, 'README.md'),
          `[Report](file:///C:/Users/Bunny/test.pdf)\n`
        );
        const result = runContinuityCheck({ cwd: testRepo.tmpDir });
        assert.strictEqual(result.pass, false);
        assert.ok(result.errors.some(e => e.includes('Machine-local link')));
      } finally {
        testRepo.cleanup();
      }
    });

    it('10. Properly updated continuity files -> PASS', () => {
      const testRepo = createTempGitRepo();
      try {
        // Create new phase 4a.5
        fs.mkdirSync(path.join(testRepo.tmpDir, 'research', 'evidence', 'phase-4a.5'), { recursive: true });
        fs.writeFileSync(
          path.join(testRepo.tmpDir, 'research', 'evidence', 'phase-4a.5', 'PHASE_REPORT.md'),
          `# Phase 4A.5 Report\n\nVerdict: PASS\nStarting commit: a480776\nEnding commit: b123456\n`
        );

        // Update STATUS_LEDGER.md with phase 4a.5 on top
        fs.writeFileSync(
          path.join(testRepo.tmpDir, 'docs', 'continuity', 'STATUS_LEDGER.md'),
          `# Status Ledger\n\n## Phase 4A.5 — Continuity Enforcement\n- Starting commit: a480776\n- Ending commit: b123456\n- Verdict: PASS\n\n---\n\n## Phase 4A.4 — Test\n- Ending commit: a480776\n\n---\n`
        );

        // Update CURRENT_STATE.md
        fs.writeFileSync(
          path.join(testRepo.tmpDir, 'docs', 'continuity', 'CURRENT_STATE.md'),
          `# Current State\n\nPhase 4A.5 — Continuity Enforcement\n`
        );

        // Update CODE_INDEX.md
        fs.writeFileSync(
          path.join(testRepo.tmpDir, 'docs', 'continuity', 'CODE_INDEX.md'),
          `# Code Index\n\n- scripts/continuity-check.mjs\n`
        );

        const result = runContinuityCheck({ cwd: testRepo.tmpDir });
        assert.strictEqual(result.pass, true, `Expected PASS but got: ${result.errors.join('; ')}`);
      } finally {
        testRepo.cleanup();
      }
    });

    it('Excessive lines per phase in STATUS_LEDGER (> 20 lines) -> FAIL', () => {
      const testRepo = createTempGitRepo();
      try {
        const longPhase = Array.from({ length: 25 }, (_, i) => `- Item line ${i + 1}`).join('\n');
        fs.writeFileSync(
          path.join(testRepo.tmpDir, 'docs', 'continuity', 'STATUS_LEDGER.md'),
          `# Status Ledger\n\n## Phase 4A.4 — Long\n${longPhase}\n\n---\n`
        );
        const result = runContinuityCheck({ cwd: testRepo.tmpDir });
        assert.strictEqual(result.pass, false);
        assert.ok(result.errors.some(e => e.includes('maximum allowed is 20 lines')));
      } finally {
        testRepo.cleanup();
      }
    });
  });
});
