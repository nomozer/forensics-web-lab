#requires -Version 5.1
<#
.SYNOPSIS
Future one-time Windows orchestration for the sealed Phase 4C.2G evaluator.

.DESCRIPTION
This script is intentionally not a readiness launcher. Formal mode requires an
already elevated local console and a direct human-authored authorization bound
to the final package. ContractValidationOnly is non-mutating and is the only
mode used during Phase 4C.2G.0.5.
#>
[CmdletBinding()]
param(
    [switch]$ContractValidationOnly,
    [string]$PythonExe = "python",
    [string]$PackageArchive,
    [string]$AuthorizationFile,
    [string]$SourceBinding,
    [string]$AuthorizationSchema,
    [string]$CheckpointDir,
    [string]$LockedTestRoot,
    [string]$Manifest,
    [string]$ManifestSchema,
    [string]$OutputDir,
    [string]$SessionId,
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu",
    [ValidateRange(1, 512)]
    [int]$BatchSize = 32
)

$ErrorActionPreference = "Stop"
$WatchdogTaskName = "Phase4C2G_AuthorizedSession_NetworkRecovery"
$OrderedSteps = @(
    "verify_authorization_and_final_package",
    "create_network_recovery_watchdog",
    "isolate_active_egress_adapters",
    "verify_fresh_same_session_isolation_receipt",
    "verify_os_backed_read_only_storage",
    "run_exact_five_checkpoint_driver",
    "write_atomic_outputs",
    "restore_network",
    "delete_and_verify_watchdog",
    "seal_receipts_and_access_ledger"
)

if ($ContractValidationOnly) {
    [ordered]@{
        verdict = "AUTHORIZED_SESSION_CONTRACT_VALID"
        mutations_performed = 0
        uac_or_readiness_invoked = $false
        ordered_steps = $OrderedSteps
        read_only_policy = [ordered]@{
            accepted_evidence = @(
                "windows_disk_is_read_only",
                "windows_cdrom_volume",
                "windows_read_only_virtual_disk"
            )
            folder_readonly_attribute_accepted = $false
            canary_writes_performed = 0
            failure_code = "BLOCKED_LOCKED_TEST_STORAGE_NOT_PROVABLY_READ_ONLY"
        }
        output_protocol = ".part -> flush -> fsync -> os.replace"
        retry_after_reservation = "HUMAN_ADJUDICATION_REQUIRED"
    } | ConvertTo-Json -Depth 8 -Compress
    exit 0
}

function Write-AtomicJson {
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)]$Value)
    $target = [System.IO.Path]::GetFullPath($Path)
    $part = "$target.part"
    if (Test-Path -LiteralPath $target) { throw "Refusing to overwrite sealed output: $target" }
    if (Test-Path -LiteralPath $part) { throw "Interrupted .part output requires human adjudication: $part" }
    $parent = Split-Path -Parent $target
    [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    $json = $Value | ConvertTo-Json -Depth 20
    $encoding = New-Object System.Text.UTF8Encoding($false)
    $stream = New-Object System.IO.FileStream($part, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    try {
        $writer = New-Object System.IO.StreamWriter($stream, $encoding)
        try {
            $writer.Write($json)
            $writer.Write("`n")
            $writer.Flush()
            $stream.Flush($true)
        } finally {
            $writer.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
    [System.IO.File]::Move($part, $target)
}

function Get-ActiveEgressAdapters {
    $indices = @(
        Get-NetRoute -PolicyStore ActiveStore -ErrorAction Stop |
            Where-Object {
                ($_.DestinationPrefix -eq "0.0.0.0/0" -or $_.DestinationPrefix -eq "::/0") -and
                $_.State -ne "Invalid"
            } |
            Select-Object -ExpandProperty InterfaceIndex -Unique
    )
    $adapters = @()
    foreach ($index in $indices) {
        $adapter = Get-NetAdapter -InterfaceIndex $index -ErrorAction Stop
        if ($adapter.Name -match "Loopback") { continue }
        $adapters += $adapter
    }
    return @($adapters)
}

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-ProxyEnabled {
    $userProxy = (Get-ItemProperty -LiteralPath "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings" -Name ProxyEnable -ErrorAction SilentlyContinue).ProxyEnable
    $winHttp = (& netsh.exe winhttp show proxy 2>&1 | Out-String)
    return (($userProxy -eq 1) -or ($winHttp -notmatch "Direct access \(no proxy server\)"))
}

function Get-WindowsReadOnlyProof {
    param([Parameter(Mandatory)][string]$Root, [Parameter(Mandatory)][string]$CurrentSessionId)
    $resolvedRoot = (Resolve-Path -LiteralPath $Root -ErrorAction Stop).Path
    $volume = Get-Volume -FilePath $resolvedRoot -ErrorAction Stop
    $now = [DateTimeOffset]::UtcNow.ToString("o")
    if ($volume.DriveType -eq "CD-ROM") {
        return [ordered]@{
            session_id = $CurrentSessionId
            verified_at_utc = $now
            locked_test_root = $resolvedRoot
            status = "READ_ONLY_VOLUME_VERIFIED"
            os_enforced_read_only = $true
            evidence_kind = "windows_cdrom_volume"
            backing_volume_unique_id = [string]$volume.UniqueId
            backing_disk_number = $null
            canary_writes_performed = 0
        }
    }
    if (-not $volume.DriveLetter) {
        throw "BLOCKED_LOCKED_TEST_STORAGE_NOT_PROVABLY_READ_ONLY: volume has no resolvable drive letter."
    }
    $partition = Get-Partition -DriveLetter $volume.DriveLetter -ErrorAction Stop
    $disk = Get-Disk -Number $partition.DiskNumber -ErrorAction Stop
    if ($disk.IsReadOnly -ne $true) {
        throw "BLOCKED_LOCKED_TEST_STORAGE_NOT_PROVABLY_READ_ONLY: backing disk IsReadOnly is false."
    }
    return [ordered]@{
        session_id = $CurrentSessionId
        verified_at_utc = $now
        locked_test_root = $resolvedRoot
        status = "READ_ONLY_VOLUME_VERIFIED"
        os_enforced_read_only = $true
        evidence_kind = $(if ($disk.BusType -eq "File Backed Virtual") { "windows_read_only_virtual_disk" } else { "windows_disk_is_read_only" })
        backing_volume_unique_id = [string]$volume.UniqueId
        backing_disk_number = [int]$disk.Number
        canary_writes_performed = 0
    }
}

$required = [ordered]@{
    PackageArchive = $PackageArchive
    AuthorizationFile = $AuthorizationFile
    SourceBinding = $SourceBinding
    AuthorizationSchema = $AuthorizationSchema
    CheckpointDir = $CheckpointDir
    LockedTestRoot = $LockedTestRoot
    Manifest = $Manifest
    ManifestSchema = $ManifestSchema
    OutputDir = $OutputDir
    SessionId = $SessionId
}
foreach ($entry in $required.GetEnumerator()) {
    if ([string]::IsNullOrWhiteSpace([string]$entry.Value)) { throw "Missing required parameter: $($entry.Key)" }
}
if (-not (Test-Administrator)) { throw "Formal authorized session requires an already elevated local PowerShell console." }

$outputFull = [System.IO.Path]::GetFullPath($OutputDir)
if (Test-Path -LiteralPath $outputFull) {
    if (@(Get-ChildItem -LiteralPath $outputFull -Force).Count -ne 0) { throw "Output directory must be empty." }
} else {
    [System.IO.Directory]::CreateDirectory($outputFull) | Out-Null
}

# Step 1: verify authorization and its exact final archive binding before mutation.
$authorization = Get-Content -Raw -LiteralPath $AuthorizationFile | ConvertFrom-Json
$archiveItem = Get-Item -LiteralPath $PackageArchive
$archiveSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $PackageArchive).Hash.ToLowerInvariant()
if ($authorization.sealed_package_sha256 -ne $archiveSha -or [int64]$authorization.sealed_package_bytes -ne $archiveItem.Length) {
    throw "Authorization does not bind the exact final sealed package."
}
$preflightArgs = @(
    "-m", "ml.evaluation.run_phase_4c2g_confirmatory",
    "--authorization-preflight-only",
    "--authorization-file", $AuthorizationFile,
    "--package-archive", $PackageArchive,
    "--source-binding", $SourceBinding,
    "--authorization-schema", $AuthorizationSchema,
    "--checkpoint-dir", $CheckpointDir,
    "--locked-test-root", $LockedTestRoot,
    "--manifest", $Manifest,
    "--manifest-schema", $ManifestSchema,
    "--isolation-receipt", (Join-Path $outputFull "isolation_receipt.json"),
    "--read-only-receipt", (Join-Path $outputFull "read_only_receipt.json"),
    "--output-dir", $outputFull,
    "--session-id", $SessionId,
    "--device", $Device,
    "--batch-size", [string]$BatchSize
)
& $PythonExe @preflightArgs
if ($LASTEXITCODE -ne 0) { throw "Authorization/package preflight failed with exit code $LASTEXITCODE." }

# Step 2: create a recovery watchdog before changing any adapter.
$initialAdapters = @(Get-ActiveEgressAdapters)
if ($initialAdapters.Count -eq 0) { throw "No active egress adapters found to isolate." }
$recoveryScript = Join-Path $outputFull "RECOVER_NETWORK_AUTHORIZED_SESSION.ps1"
$recoveryLines = @("`$ErrorActionPreference = 'Continue'")
foreach ($adapter in $initialAdapters) {
    $recoveryLines += "Enable-NetAdapter -InterfaceIndex $($adapter.ifIndex) -Confirm:`$false -ErrorAction Continue"
}
[System.IO.File]::WriteAllLines($recoveryScript, $recoveryLines, (New-Object System.Text.UTF8Encoding($false)))
$null = [System.Management.Automation.Language.Parser]::ParseFile($recoveryScript, [ref]$null, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) { throw "Recovery script AST validation failed." }
$trigger = (Get-Date).AddMinutes(15).ToString("HH:mm")
$taskCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$recoveryScript`""
& schtasks.exe /create /tn $WatchdogTaskName /tr $taskCommand /sc once /st $trigger /f /rl HIGHEST | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Failed to create recovery watchdog." }
& schtasks.exe /query /tn $WatchdogTaskName | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Recovery watchdog read-back failed." }

$driverExit = $null
$restored = $false
try {
    # Step 3: isolate all active default-route owners.
    foreach ($adapter in $initialAdapters) {
        Disable-NetAdapter -InterfaceIndex $adapter.ifIndex -Confirm:$false -ErrorAction Stop
    }

    # Step 4: verify and atomically record fresh same-session isolation evidence.
    $remainingRoutes = @(Get-NetRoute -PolicyStore ActiveStore -ErrorAction Stop | Where-Object { $_.DestinationPrefix -eq "0.0.0.0/0" -or $_.DestinationPrefix -eq "::/0" })
    $remainingOwners = @(Get-ActiveEgressAdapters)
    $proxyEnabled = Get-ProxyEnabled
    if ($remainingRoutes.Count -ne 0 -or $remainingOwners.Count -ne 0 -or $proxyEnabled) {
        throw "Network isolation verification failed closed."
    }
    $isolationReceipt = [ordered]@{
        session_id = $SessionId
        verified_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
        isolation_verified = $true
        active_egress_adapters = @()
        active_vpn_route_owners = @()
        unidentified_route_owners = @()
        proxy_enabled = $false
        remaining_active_default_routes = 0
    }
    $isolationPath = Join-Path $outputFull "isolation_receipt.json"
    Write-AtomicJson -Path $isolationPath -Value $isolationReceipt

    # Step 5: accept only operating-system-backed volume/disk read-only evidence.
    $readOnlyReceipt = Get-WindowsReadOnlyProof -Root $LockedTestRoot -CurrentSessionId $SessionId
    $readOnlyPath = Join-Path $outputFull "read_only_receipt.json"
    Write-AtomicJson -Path $readOnlyPath -Value $readOnlyReceipt

    # Steps 6-7: the Python driver performs exactly five reservations/evaluations and atomic outputs.
    $driverArgs = @(
        "-m", "ml.evaluation.run_phase_4c2g_confirmatory",
        "--authorization-file", $AuthorizationFile,
        "--package-archive", $PackageArchive,
        "--source-binding", $SourceBinding,
        "--authorization-schema", $AuthorizationSchema,
        "--checkpoint-dir", $CheckpointDir,
        "--locked-test-root", $LockedTestRoot,
        "--manifest", $Manifest,
        "--manifest-schema", $ManifestSchema,
        "--isolation-receipt", $isolationPath,
        "--read-only-receipt", $readOnlyPath,
        "--output-dir", $outputFull,
        "--session-id", $SessionId,
        "--device", $Device,
        "--batch-size", [string]$BatchSize
    )
    & $PythonExe @driverArgs
    $driverExit = $LASTEXITCODE
    if ($driverExit -ne 0) { throw "Confirmatory driver failed with exit code $driverExit; automatic retry forbidden." }
} finally {
    # Step 8: always restore the exact adapters disabled by this session.
    foreach ($adapter in $initialAdapters) {
        Enable-NetAdapter -InterfaceIndex $adapter.ifIndex -Confirm:$false -ErrorAction Continue
    }
    $deadline = (Get-Date).AddSeconds(60)
    do {
        $notUp = @($initialAdapters | Where-Object {
            $current = Get-NetAdapter -InterfaceIndex $_.ifIndex -ErrorAction SilentlyContinue
            -not $current -or $current.AdminStatus -ne "Up" -or $current.Status -ne "Up"
        })
        if ($notUp.Count -eq 0) { $restored = $true; break }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)

    # Step 9: delete watchdog only after verified operational restoration.
    if ($restored) {
        & schtasks.exe /delete /tn $WatchdogTaskName /f | Out-Null
        & schtasks.exe /query /tn $WatchdogTaskName 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { throw "Watchdog deletion could not be verified." }
    }
}
if (-not $restored) { throw "NETWORK_RECOVERY_REQUIRED: watchdog retained." }

# Step 10: seal restoration state and the final ledger/output inventory.
$sessionReceipt = [ordered]@{
    session_id = $SessionId
    completed_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
    driver_exit_code = $driverExit
    network_restored = $true
    watchdog_deleted_and_verified_absent = $true
    automatic_retry_permitted = $false
}
Write-AtomicJson -Path (Join-Path $outputFull "authorized_session_receipt.json") -Value $sessionReceipt
$files = [ordered]@{}
Get-ChildItem -LiteralPath $outputFull -File | Where-Object { $_.Name -ne "session_checksums.json" -and -not $_.Name.EndsWith(".part") } | Sort-Object Name | ForEach-Object {
    $files[$_.Name] = [ordered]@{ bytes = $_.Length; sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant() }
}
Write-AtomicJson -Path (Join-Path $outputFull "session_checksums.json") -Value ([ordered]@{ algorithm = "sha256"; self_digest_excluded = $true; files = $files })
Write-Output "AUTHORIZED_SESSION_COMPLETED"
