<#
.SYNOPSIS
    Standalone Offline Runtime Verifier Runner for Phase 4C.2G.
.DESCRIPTION
    Runs scripts/research/verify_phase_4c2g_offline_runtime.py with exact resolved host paths.
    Does NOT toggle network adapters, does NOT call evaluator, does NOT mount locked-test,
    and does NOT create authorization.
#>

[CmdletBinding()]
param(
    [string]$ExecutionWorktree = "",
    [string]$CheckpointRoot = "",
    [string]$OutputDir = "",
    [string]$PlannedMountpoint = "",
    [string]$ReceiptPath = "",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

# 1. Resolve Repo Root and Script Directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path

# 2. Resolve Exact Host Paths (Defaulting to canonical host locations)
if (-not $PythonExe) {
    $VenvPython = Join-Path $RepoRoot "ml\.venv\Scripts\python.exe"
    if (Test-Path $VenvPython) {
        $PythonExe = (Resolve-Path $VenvPython).Path
    } else {
        $PythonExe = (Get-Command python).Source
    }
}

if (-not $ExecutionWorktree) {
    $DefaultWorktree = (Join-Path $RepoRoot "..\forensics-web-lab-offline-execution")
    if (Test-Path $DefaultWorktree) {
        $ExecutionWorktree = (Resolve-Path $DefaultWorktree).Path
    } else {
        $ExecutionWorktree = (Join-Path $RepoRoot "..\forensics-web-lab-offline-execution")
    }
}

if (-not $CheckpointRoot) {
    $DefaultCheckpointRoot = (Join-Path $RepoRoot "..\forensics-web-lab-local-artifacts\phase_4c1\runs\extracted_15_runs")
    if (Test-Path $DefaultCheckpointRoot) {
        $CheckpointRoot = (Resolve-Path $DefaultCheckpointRoot).Path
    } else {
        $CheckpointRoot = (Join-Path $RepoRoot "data\research\local-artifacts\phase-4c.1\extracted_15_runs")
    }
}

if (-not $OutputDir) {
    $OutputDir = (Join-Path $RepoRoot "data\research\evaluation-outputs\phase_4c2g")
}

if (-not $PlannedMountpoint) {
    $PlannedMountpoint = (Join-Path $RepoRoot "data\research\locked-test-mount")
}

if (-not $ReceiptPath) {
    $ReceiptPath = (Join-Path $RepoRoot "data\research\local-artifacts\phase-4c.2g\offline_verifier_execution_receipt.json")
}

$VerifierScript = Join-Path $RepoRoot "scripts\research\verify_phase_4c2g_offline_runtime.py"

# Ensure output and mount directories exist
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}
if (-not (Test-Path $PlannedMountpoint)) {
    New-Item -ItemType Directory -Path $PlannedMountpoint -Force | Out-Null
}
$ReceiptParent = Split-Path -Parent $ReceiptPath
if (-not (Test-Path $ReceiptParent)) {
    New-Item -ItemType Directory -Path $ReceiptParent -Force | Out-Null
}

# 3. Inspect Worktree HEAD
$WorktreeHead = "UNKNOWN"
if (Test-Path (Join-Path $ExecutionWorktree ".git")) {
    try {
        $WorktreeHead = (& git -C $ExecutionWorktree rev-parse HEAD 2>$null).Trim()
    } catch {
        $WorktreeHead = "ERROR_RETRIEVING_HEAD"
    }
}

# 4. Print Exact Runtime Information Before Execution
$CurrentUtc = [System.DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.ffffff+00:00")

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " PHASE 4C.2G STANDALONE OFFLINE RUNTIME VERIFIER" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "Current UTC Time       : $CurrentUtc"
Write-Host "Python Executable      : $PythonExe"
Write-Host "Execution Worktree     : $ExecutionWorktree"
Write-Host "Git Worktree HEAD      : $WorktreeHead"
Write-Host "Checkpoint Root        : $CheckpointRoot"
Write-Host "Output Directory       : $OutputDir"
Write-Host "Planned Mountpoint     : $PlannedMountpoint"
Write-Host "Expected Receipt Path  : $ReceiptPath"
Write-Host "Verifier Script        : $VerifierScript"
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "WARNING: DO NOT RUN WHILE NETWORK IS CONNECTED" -ForegroundColor Yellow
Write-Host "Ensure Ethernet, Wi-Fi, VPN, and Bluetooth PAN are physically disabled." -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Cyan

# 5. Invoke Offline Verifier Script
& $PythonExe $VerifierScript `
    --execution-worktree "$ExecutionWorktree" `
    --checkpoint-root "$CheckpointRoot" `
    --output-dir "$OutputDir" `
    --planned-locked-test-mountpoint "$PlannedMountpoint" `
    --receipt "$ReceiptPath"

$VerifierExitCode = $LASTEXITCODE

Write-Host "`nVerifier process exited with code: $VerifierExitCode"

# 6. Evaluate Exit Code & Output Verdict
if ($VerifierExitCode -eq 0) {
    Write-Host "[SUCCESS] Verdict: READY_FOR_HUMAN_AUTHORIZATION_REVIEW" -ForegroundColor Green
    Write-Host "Receipt preserved at: $ReceiptPath"
    Write-Host "NOTE: Do NOT connect to network before human authorization and test execution."
    exit 0
} elseif ($VerifierExitCode -eq 2) {
    Write-Host "[ACTION REQUIRED] Verdict: USER_PHYSICAL_ACTION_REQUIRED" -ForegroundColor Yellow
    Write-Host "Host still has an active default route or connected network adapter."
    Write-Host "Please physically disconnect network adapters and re-run."
    exit 2
} else {
    Write-Host "[FAILURE] Verifier failed with binding, component, or dependency error." -ForegroundColor Red
    Write-Host "Check receipt details at: $ReceiptPath"
    exit 1
}
