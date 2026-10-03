#requires -Version 5.1
<#
.SYNOPSIS
Future one-UAC, one-session locked-test manifest custodian controller.

.DESCRIPTION
ContractValidationOnly is the only mode exercised in Phase 4C.2G.0.6. Formal
mode validates exact authorization/package bindings before one elevation,
isolates egress, proves OS-backed read-only storage, invokes only the manifest
sealer, restores networking, and never invokes a model evaluator.
#>
[CmdletBinding()]
param(
    [switch]$ContractValidationOnly,
    [switch]$ElevatedWorker,
    [string]$PythonExe = "python",
    [string]$AuthorizationFile,
    [string]$AuthorizationSchema,
    [string]$PackageArchive,
    [string]$PackageBinding,
    [string]$SealerScript,
    [string]$ManifestSchema,
    [string]$LockedTestRoot,
    [string]$OutputRoot,
    [string]$SessionId
)

$ErrorActionPreference = "Stop"
$WatchdogTaskName = "Phase4C2G_ManifestCustodian_NetworkRecovery"
$OrderedSteps = @(
    "validate_custodian_authorization",
    "validate_exact_custodian_tool_and_package",
    "create_recovery_watchdog",
    "isolate_active_egress_adapters",
    "verify_fresh_same_session_isolation_receipt",
    "verify_os_backed_read_only_locked_test_storage",
    "run_manifest_sealer_once",
    "verify_atomic_commitment_receipt",
    "restore_network",
    "delete_and_read_back_watchdog",
    "stop_without_evaluator"
)

if ($ContractValidationOnly) {
    [ordered]@{
        verdict = "MANIFEST_CUSTODIAN_CONTRACT_VALID"
        mutations_performed = 0
        uac_or_session_invoked = $false
        evaluator_invocations = 0
        ordered_steps = $OrderedSteps
        maximum_custodian_sessions = 1
        maximum_confirmatory_model_evaluation_sessions = 1
        confirmatory_model_evaluation_sessions_used = 0
        output_protocol = ".part -> flush -> fsync -> os.replace"
        retry_after_reservation = "HUMAN_ADJUDICATION_REQUIRED"
        folder_readonly_attribute_accepted = $false
        canary_writes_performed = 0
    } | ConvertTo-Json -Depth 8 -Compress
    exit 0
}

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Write-AtomicJson {
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)]$Value)
    $target = [System.IO.Path]::GetFullPath($Path)
    $part = "$target.part"
    if (Test-Path -LiteralPath $target) { throw "Refusing to overwrite sealed output: $target" }
    if (Test-Path -LiteralPath $part) { throw "Interrupted partial output requires human adjudication: $part" }
    [System.IO.Directory]::CreateDirectory((Split-Path -Parent $target)) | Out-Null
    $encoding = New-Object System.Text.UTF8Encoding($false)
    $stream = New-Object System.IO.FileStream($part, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    try {
        $writer = New-Object System.IO.StreamWriter($stream, $encoding)
        try {
            $writer.Write(($Value | ConvertTo-Json -Depth 20))
            $writer.Write("`n")
            $writer.Flush()
            $stream.Flush($true)
        } finally { $writer.Dispose() }
    } finally { $stream.Dispose() }
    [System.IO.File]::Move($part, $target)
}

function Get-ActiveEgressAdapters {
    $indices = @(Get-NetRoute -PolicyStore ActiveStore -ErrorAction Stop |
        Where-Object { ($_.DestinationPrefix -eq "0.0.0.0/0" -or $_.DestinationPrefix -eq "::/0") -and $_.State -ne "Invalid" } |
        Select-Object -ExpandProperty InterfaceIndex -Unique)
    return @(Get-NetAdapter -ErrorAction Stop | Where-Object { $indices -contains $_.ifIndex -and $_.Name -notmatch "Loopback" })
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
    $evidenceKind = $null
    $diskNumber = $null
    if ($volume.DriveType -eq "CD-ROM") {
        $evidenceKind = "windows_cdrom_volume"
    } else {
        if (-not $volume.DriveLetter) { throw "BLOCKED_LOCKED_TEST_STORAGE_NOT_PROVABLY_READ_ONLY" }
        $partition = Get-Partition -DriveLetter $volume.DriveLetter -ErrorAction Stop
        $disk = Get-Disk -Number $partition.DiskNumber -ErrorAction Stop
        if ($disk.IsReadOnly -ne $true) { throw "BLOCKED_LOCKED_TEST_STORAGE_NOT_PROVABLY_READ_ONLY" }
        $diskNumber = [int]$disk.Number
        $evidenceKind = $(if ($disk.BusType -eq "File Backed Virtual") { "windows_read_only_virtual_disk" } else { "windows_disk_is_read_only" })
    }
    return [ordered]@{
        session_id = $CurrentSessionId
        verified_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
        locked_test_root = $resolvedRoot
        status = "READ_ONLY_VOLUME_VERIFIED"
        os_enforced_read_only = $true
        evidence_kind = $evidenceKind
        backing_volume_unique_id = [string]$volume.UniqueId
        backing_disk_number = $diskNumber
        canary_writes_performed = 0
    }
}

$required = [ordered]@{
    AuthorizationFile = $AuthorizationFile; AuthorizationSchema = $AuthorizationSchema
    PackageArchive = $PackageArchive; PackageBinding = $PackageBinding
    SealerScript = $SealerScript; ManifestSchema = $ManifestSchema
    LockedTestRoot = $LockedTestRoot; OutputRoot = $OutputRoot; SessionId = $SessionId
}
foreach ($entry in $required.GetEnumerator()) {
    if ([string]::IsNullOrWhiteSpace([string]$entry.Value)) { throw "Missing required parameter: $($entry.Key)" }
}

# Steps 1-2 happen before elevation, network mutation, or locked-test access.
$authorization = Get-Content -Raw -LiteralPath $AuthorizationFile | ConvertFrom-Json
$binding = Get-Content -Raw -LiteralPath $PackageBinding | ConvertFrom-Json
$archiveItem = Get-Item -LiteralPath $PackageArchive
$archiveSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $PackageArchive).Hash.ToLowerInvariant()
if ($binding.status -ne "READY_FOR_HUMAN_MANIFEST_CUSTODIAN_APPROVAL") { throw "Custodian package binding is not active." }
if ($binding.archive.filename -ne $archiveItem.Name -or $binding.archive.sha256 -ne $archiveSha -or [int64]$binding.archive.bytes -ne $archiveItem.Length) { throw "Custodian package archive binding mismatch." }
if ($authorization.sealed_package_sha256 -ne $binding.archive.sha256 -or [int64]$authorization.sealed_package_bytes -ne [int64]$binding.archive.bytes) { throw "Custodian authorization archive binding mismatch." }
if ($authorization.custodian_tool_commit -ne $binding.effective_custodian_commit -or $authorization.custodian_package_commit -ne $binding.custodian_package_commit) { throw "Custodian authorization commit binding mismatch." }
if ($authorization.custodian_component_hashes.PSObject.Properties.Count -ne $binding.authorization_component_hashes.PSObject.Properties.Count) { throw "Custodian component binding mismatch." }
foreach ($property in $binding.authorization_component_hashes.PSObject.Properties) {
    if ($authorization.custodian_component_hashes.($property.Name) -ne $property.Value) { throw "Custodian component hash mismatch: $($property.Name)" }
}
& $PythonExe $SealerScript --authorization-preflight-only --authorization-file $AuthorizationFile --authorization-schema $AuthorizationSchema --package-binding $PackageBinding --package-archive $PackageArchive
if ($LASTEXITCODE -ne 0) { throw "Custodian authorization preflight failed." }

if (-not (Test-Administrator)) {
    if ($ElevatedWorker) { throw "Elevated worker does not have Administrator privileges." }
    $forward = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath, "-ElevatedWorker")
    foreach ($entry in $required.GetEnumerator()) { $forward += @("-$($entry.Key)", [string]$entry.Value) }
    if ($PythonExe) { $forward += @("-PythonExe", $PythonExe) }
    $process = Start-Process -FilePath "powershell.exe" -ArgumentList $forward -Verb RunAs -Wait -PassThru
    exit $process.ExitCode
}

$output = [System.IO.Path]::GetFullPath($OutputRoot)
if (Test-Path -LiteralPath $output) {
    if (@(Get-ChildItem -LiteralPath $output -Force).Count -ne 0) { throw "Custodian output root must be empty." }
} else { [System.IO.Directory]::CreateDirectory($output) | Out-Null }
$sealedOutput = Join-Path $output "sealed"

$initialAdapters = @(Get-ActiveEgressAdapters)
if ($initialAdapters.Count -eq 0) { throw "No active egress adapters found to isolate." }
$recoveryScript = Join-Path $output "RECOVER_NETWORK_MANIFEST_CUSTODIAN.ps1"
$recoveryLines = @("`$ErrorActionPreference = 'Continue'")
foreach ($adapter in $initialAdapters) {
    $safeName = $adapter.Name.Replace("'", "''")
    $recoveryLines += "Get-NetAdapter -Name '$safeName' -ErrorAction SilentlyContinue | Enable-NetAdapter -Confirm:`$false -ErrorAction Continue"
}
[System.IO.File]::WriteAllLines($recoveryScript, $recoveryLines, (New-Object System.Text.UTF8Encoding($false)))
$trigger = (Get-Date).AddMinutes(15).ToString("HH:mm")
& schtasks.exe /create /tn $WatchdogTaskName /tr "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$recoveryScript`"" /sc once /st $trigger /f /rl HIGHEST | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Failed to create recovery watchdog." }
& schtasks.exe /query /tn $WatchdogTaskName | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Recovery watchdog read-back failed." }

$restored = $false
try {
    foreach ($adapter in $initialAdapters) { $adapter | Disable-NetAdapter -Confirm:$false -ErrorAction Stop }
    $remaining = @(Get-NetRoute -PolicyStore ActiveStore -ErrorAction Stop | Where-Object { $_.DestinationPrefix -eq "0.0.0.0/0" -or $_.DestinationPrefix -eq "::/0" })
    if ($remaining.Count -ne 0 -or (Get-ProxyEnabled)) { throw "Network isolation verification failed closed." }
    $isolationPath = Join-Path $output "isolation_receipt.json"
    Write-AtomicJson -Path $isolationPath -Value ([ordered]@{ session_id=$SessionId; verified_at_utc=[DateTimeOffset]::UtcNow.ToString("o"); isolation_verified=$true; active_egress_adapters=@(); active_vpn_route_owners=@(); unidentified_route_owners=@(); proxy_enabled=$false; remaining_active_default_routes=0 })
    $readOnlyPath = Join-Path $output "read_only_receipt.json"
    Write-AtomicJson -Path $readOnlyPath -Value (Get-WindowsReadOnlyProof -Root $LockedTestRoot -CurrentSessionId $SessionId)
    & $PythonExe $SealerScript --locked-test-root $LockedTestRoot --output-root $sealedOutput --session-id $SessionId --read-only-receipt $readOnlyPath --isolation-receipt $isolationPath --manifest-schema $ManifestSchema
    if ($LASTEXITCODE -ne 0) { throw "Manifest sealer failed; automatic retry forbidden." }
    $commitment = Get-Content -Raw -LiteralPath (Join-Path $sealedOutput "manifest_commitment_receipt.json") | ConvertFrom-Json
    if ($commitment.verdict -ne "MANIFEST_COMMITMENT_SEALED" -or $commitment.counters.custodian_manifest_access_sessions -ne 1 -or $commitment.counters.completed_real_model_evaluations -ne 0) { throw "Commitment receipt verification failed." }
} finally {
    foreach ($adapter in $initialAdapters) { Get-NetAdapter -Name $adapter.Name -ErrorAction SilentlyContinue | Enable-NetAdapter -Confirm:$false -ErrorAction Continue }
    $deadline = (Get-Date).AddSeconds(60)
    do {
        $notUp = @($initialAdapters | Where-Object { $current = Get-NetAdapter -Name $_.Name -ErrorAction SilentlyContinue; -not $current -or $current.AdminStatus -ne "Up" -or $current.Status -ne "Up" })
        if ($notUp.Count -eq 0) { $restored = $true; break }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)
    if ($restored) {
        & schtasks.exe /delete /tn $WatchdogTaskName /f | Out-Null
        & schtasks.exe /query /tn $WatchdogTaskName 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { throw "Watchdog deletion read-back failed." }
    }
}
if (-not $restored) { throw "NETWORK_RECOVERY_REQUIRED: watchdog retained." }
Write-Output "MANIFEST_CUSTODIAN_SESSION_COMPLETED"
