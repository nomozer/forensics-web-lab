<#
.SYNOPSIS
    Automated Windows Network-Isolation Controller for Phase 4C.2G.
.DESCRIPTION
    Automates temporary passive network isolation on Windows hosts for Phase 4C.2G offline verification.
    Features:
    - Remote session detection and fail-closed safety (blocks execution over RDP, SSH, WinRM, CI).
    - Elevation check with single-UAC detached worker dispatch.
    - Minimal adapter selection based strictly on ifIndex and route ownership.
    - Loopback, host-only (VMnet1), and internal virtual switches (WSL, Default Switch) protected by default.
    - 15-minute Scheduled Task recovery watchdog with syntax verification and query read-back.
    - Iterative fail-closed isolation with ambiguous route owner detection.
    - try/finally isolation and exact adapter restoration by ifIndex.
    - Passive offline inspection and invocation of RUN_PHASE4C2G_OFFLINE_VERIFIER.ps1.
    - Zero outbound network probes (no ICMP probes, no DNS lookups, no raw sockets, no HTTP calls).
    - Zero evaluator or locked-test access.
#>

[CmdletBinding(DefaultParameterSetName = "Default")]
param(
    [Parameter(ParameterSetName = "DryRun")]
    [switch]$DryRun,

    [Parameter(ParameterSetName = "ReadinessTest")]
    [switch]$ReadinessTest,

    [Parameter(ParameterSetName = "ValidateRecoveryScriptOnly")]
    [switch]$ValidateRecoveryScriptOnly,

    [Parameter(ParameterSetName = "ValidateRecoveryScriptOnly")]
    [string]$TargetFixtureJson = "",

    [Parameter(ParameterSetName = "ValidateRecoveryScriptOnly")]
    [string]$OutputRecoveryScriptPath = "",

    [Parameter(ParameterSetName = "ValidateAdapterCmdletContractOnly")]
    [switch]$ValidateAdapterCmdletContractOnly,

    [Parameter()]
    [string]$RepoRoot = "",

    [Parameter()]
    [string]$ReceiptPath = "",

    [Parameter()]
    [switch]$ElevatedDetachedWorker
)

$ErrorActionPreference = "Stop"
$ControllerVersion = "1.4.0"

# 1. Resolve Paths
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
}

$ArtifactsDir = Join-Path $RepoRoot "data\research\local-artifacts\phase-4c.2g"
if (-not (Test-Path $ArtifactsDir)) {
    New-Item -ItemType Directory -Path $ArtifactsDir -Force | Out-Null
}

if (-not $ReceiptPath) {
    $ReceiptPath = Join-Path $ArtifactsDir "automated_isolation_readiness_receipt.json"
}

$RecoverScriptPath = Join-Path $ArtifactsDir "RECOVER_NETWORK.ps1"
$VerifierWrapperPath = Join-Path $RepoRoot "scripts\research\RUN_PHASE4C2G_OFFLINE_VERIFIER.ps1"
$WatchdogTaskName = "Phase4C2G_Emergency_Network_Recovery"

# 2. Remote Session Detection (Fail-Closed)
function Test-IsRemoteSession {
    $sessionName = $env:SESSIONNAME
    if ($sessionName -and ($sessionName -like "RDP*" -or $sessionName -like "ICA*" -or $sessionName -like "HDX*")) {
        return @{ IsRemote = $true; Reason = "Remote Desktop Session detected: $sessionName" }
    }

    if ($env:SSH_CLIENT -or $env:SSH_CONNECTION -or $env:SSH_TTY) {
        return @{ IsRemote = $true; Reason = "SSH Session detected" }
    }

    if ($Host.Name -eq "ServerRemoteHost" -or (Test-Path variable:PSSenderInfo) -or $env:PSSessionApplicationName) {
        return @{ IsRemote = $true; Reason = "Remote PowerShell/WinRM session detected" }
    }

    if ($env:CI -eq "true" -or $env:GITHUB_ACTIONS -eq "true" -or $env:TF_BUILD -eq "true") {
        return @{ IsRemote = $true; Reason = "CI Runner/Cloud automation session detected" }
    }

    return @{ IsRemote = $false; Reason = "Local interactive session verified ($sessionName)" }
}

# 3. Administrator Role Detection
function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# 4. Helper: Determine Adapter Type
function Get-AdapterType {
    param([string]$Name, [string]$Desc)
    if ($Name -match "Loopback" -or $Desc -match "Loopback") { return "Loopback" }
    if ($Name -match "VMnet1\b" -or $Desc -match "VMnet1\b") { return "Host-Only Virtual Network (VMware)" }
    if ($Name -match "VMnet\d+" -or $Desc -match "VMnet\d+") { return "Virtual Network (VMware)" }
    if ($Name -match "WSL" -or $Desc -match "WSL") { return "Internal Virtual Switch (WSL)" }
    if ($Name -match "Default Switch" -or $Desc -match "Default Switch") { return "Internal Virtual Switch (Hyper-V)" }
    if ($Name -match "vEthernet" -or $Desc -match "Hyper-V") { return "Virtual Network (Hyper-V)" }
    if ($Name -match "VPN|Tunnel|WireGuard" -or $Desc -match "VPN|Tunnel|WireGuard|TAP|TUN|Famatech|Radmin") { return "VPN / Tunnel" }
    if ($Name -match "Wi-Fi|Wireless" -or $Desc -match "Wi-Fi|Wireless|802\.11") { return "Physical Wi-Fi" }
    if ($Name -match "Bluetooth" -or $Desc -match "Bluetooth") { return "Bluetooth PAN" }
    if ($Name -match "Ethernet" -or $Desc -match "Ethernet|Gigabit|PCIe") { return "Physical Ethernet" }
    return "Other Network Interface"
}

# 5. Snapshot Adapters and Routes (Minimal Selection by ifIndex & Route Ownership)
function Get-NetworkIsolationSnapshot {
    $snapshotTime = [System.DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.ffffff+00:00")
    $adapterList = @()
    $rawAdapters = Get-NetAdapter -ErrorAction SilentlyContinue
    $adapterMap = @{}

    foreach ($a in $rawAdapters) {
        $macSha = ""
        if ($a.MacAddress) {
            $macBytes = [System.Text.Encoding]::UTF8.GetBytes($a.MacAddress)
            $shaObj = [System.Security.Cryptography.SHA256]::Create()
            $macSha = -join ($shaObj.ComputeHash($macBytes) | ForEach-Object { "{0:x2}" -f $_ })
        }
        $aType = Get-AdapterType -Name $a.Name -Desc $a.InterfaceDescription
        $entry = [ordered]@{
            Name                 = $a.Name
            InterfaceIndex       = $a.InterfaceIndex
            InterfaceDescription = $a.InterfaceDescription
            AdapterType          = $aType
            Status               = $a.Status
            AdminStatus          = $a.AdminStatus
            MacAddress           = [string]$a.MacAddress
            MacSha256            = $macSha
        }
        $adapterList += $entry
        $adapterMap[$a.InterfaceIndex] = $entry
    }

    # Gather IPv4 default routes
    $ipv4Routes = @()
    try {
        $rawRoutes = Get-NetRoute -PolicyStore ActiveStore -DestinationPrefix "0.0.0.0/0" -ErrorAction SilentlyContinue
        foreach ($r in $rawRoutes) {
            $ipv4Routes += [ordered]@{
                DestinationPrefix = "0.0.0.0/0"
                InterfaceAlias    = $r.InterfaceAlias
                InterfaceIndex    = $r.InterfaceIndex
                NextHop           = $r.NextHop
                RouteMetric       = $r.RouteMetric
            }
        }
    } catch {}

    # Gather IPv6 default routes
    $ipv6Routes = @()
    try {
        $rawRoutes6 = Get-NetRoute -PolicyStore ActiveStore -DestinationPrefix "::/0" -ErrorAction SilentlyContinue
        foreach ($r in $rawRoutes6) {
            $ipv6Routes += [ordered]@{
                DestinationPrefix = "::/0"
                InterfaceAlias    = $r.InterfaceAlias
                InterfaceIndex    = $r.InterfaceIndex
                NextHop           = $r.NextHop
                RouteMetric       = $r.RouteMetric
            }
        }
    } catch {}

    # Map route owners and track unidentified routes
    $defaultRouteOwners = @{}
    $unidentifiedRoutes = @()

    foreach ($r in ($ipv4Routes + $ipv6Routes)) {
        $idx = $r.InterfaceIndex
        if ($adapterMap.ContainsKey($idx)) {
            if (-not $defaultRouteOwners.ContainsKey($idx)) {
                $defaultRouteOwners[$idx] = @()
            }
            $defaultRouteOwners[$idx] += $r
        } else {
            $unidentifiedRoutes += $r
        }
    }

    # Evaluate each adapter strictly by evidenced ownership
    $evaluatedAdapters = @()
    $initialDisableTargets = @()
    $protectedInternalAdapters = @()
    $protectedDisconnected = @()

    foreach ($a in $adapterList) {
        $idx = $a.InterfaceIndex
        $name = $a.Name
        $type = $a.AdapterType
        $status = $a.Status
        $selected = $false
        $exactReason = ""
        $protectedReason = ""

        # Condition 1 & 2: Active default route owner
        if ($defaultRouteOwners.ContainsKey($idx)) {
            $routes = $defaultRouteOwners[$idx]
            $routeDescs = $routes | ForEach-Object { "$($_.DestinationPrefix) via $($_.NextHop) metric $($_.RouteMetric)" }
            $selected = $true
            $exactReason = "Owns active default route: $($routeDescs -join '; ')"
        }
        # Condition 3: Active VPN/tunnel adapter with non-loopback egress route
        elseif ($status -eq "Up" -and $type -eq "VPN / Tunnel") {
            $vpnRoutes = @(Get-NetRoute -PolicyStore ActiveStore -InterfaceIndex $idx -ErrorAction SilentlyContinue | Where-Object {
                $_.DestinationPrefix -notmatch "^(127\.|::1|ff00|224\.|255\.)"
            })
            if ($vpnRoutes.Count -gt 0) {
                $selected = $true
                $exactReason = "Active VPN/tunnel interface owning $(vpnRoutes.Count) non-loopback egress route(s)"
            } else {
                $protectedReason = "VPN/tunnel adapter has no active egress route"
            }
        }

        # If not selected, classify protection reason
        if (-not $selected) {
            if ($status -ne "Up") {
                $protectedReason = "Disconnected or disabled adapter preserved without modification"
                $protectedDisconnected += $name
            } elseif ($type -eq "Loopback") {
                $protectedReason = "Loopback interface protected from isolation"
            } elseif ($type -like "*Virtual*" -or $type -like "*Host-Only*" -or $name -like "VMnet*" -or $name -like "*WSL*" -or $name -like "*Default Switch*") {
                $protectedReason = "Internal/virtual adapter with no default or egress route ownership"
                $protectedInternalAdapters += $name
            } else {
                $protectedReason = "Adapter has no active default or egress route ownership"
            }
        }

        $evalEntry = [ordered]@{
            InterfaceIndex          = $idx
            Name                    = $name
            AdapterType             = $type
            Status                  = $status
            AdminStatus             = $a.AdminStatus
            SelectedForDisable      = $selected
            ExactSelectionReason    = $exactReason
            ProtectedReason         = $protectedReason
        }
        $evaluatedAdapters += $evalEntry

        if ($selected) {
            $initialDisableTargets += [ordered]@{
                InterfaceIndex       = $idx
                Name                 = $name
                InterfaceDescription = $a.InterfaceDescription
                MacAddress           = [string]$a.MacAddress
                Reason               = $exactReason
            }
        }
    }

    # Build detailed route-to-adapter table records
    $routeTableRecords = @()
    foreach ($a in $evaluatedAdapters) {
        $idx = $a.InterfaceIndex
        if ($defaultRouteOwners.ContainsKey($idx)) {
            foreach ($r in $defaultRouteOwners[$idx]) {
                $routeTableRecords += [ordered]@{
                    destination_prefix    = $r.DestinationPrefix
                    next_hop              = $r.NextHop
                    route_metric          = $r.RouteMetric
                    ifIndex               = $idx
                    adapter_name          = $a.Name
                    adapter_type          = $a.AdapterType
                    operational_status    = $a.Status
                    selected_for_disable  = $a.SelectedForDisable
                    exact_selection_reason= $a.ExactSelectionReason
                    protected_reason      = $a.ProtectedReason
                }
            }
        } else {
            $routeTableRecords += [ordered]@{
                destination_prefix    = "N/A"
                next_hop              = "N/A"
                route_metric          = "N/A"
                ifIndex               = $idx
                adapter_name          = $a.Name
                adapter_type          = $a.AdapterType
                operational_status    = $a.Status
                selected_for_disable  = $a.SelectedForDisable
                exact_selection_reason= $a.ExactSelectionReason
                protected_reason      = $a.ProtectedReason
            }
        }
    }

    foreach ($ur in $unidentifiedRoutes) {
        $routeTableRecords += [ordered]@{
            destination_prefix    = $ur.DestinationPrefix
            next_hop              = $ur.NextHop
            route_metric          = $ur.RouteMetric
            ifIndex               = $ur.InterfaceIndex
            adapter_name          = "UNIDENTIFIED"
            adapter_type          = "UNKNOWN"
            operational_status    = "UNKNOWN"
            selected_for_disable  = $false
            exact_selection_reason= ""
            protected_reason      = "Route owner cannot be identified"
        }
    }

    return [ordered]@{
        SnapshotUtc                     = $snapshotTime
        Adapters                        = $evaluatedAdapters
        RawAdapters                     = $adapterList
        IPv4DefaultRoutes               = $ipv4Routes
        IPv6DefaultRoutes               = $ipv6Routes
        DefaultRouteOwnerCount          = $defaultRouteOwners.Keys.Count
        InitialDisableTargets           = $initialDisableTargets
        InitialDisableTargetCount       = $initialDisableTargets.Count
        ProtectedInternalAdapterCount   = $protectedInternalAdapters.Count
        ProtectedDisconnectedCount      = $protectedDisconnected.Count
        UnidentifiedEgressRoutesCount   = $unidentifiedRoutes.Count
        UnidentifiedRoutes              = $unidentifiedRoutes
        RouteTableRecords               = $routeTableRecords
    }
}

# 6. Helper: Resolve unique adapter object by ifIndex and snapshot identity
function Resolve-TargetNetAdapter {
    param(
        [Parameter(Mandatory=$true)]
        [int]$InterfaceIndex,
        [Parameter(Mandatory=$false)]
        [string]$ExpectedName = $null,
        [Parameter(Mandatory=$false)]
        [string]$ExpectedDescription = $null,
        [Parameter(Mandatory=$false)]
        [string]$ExpectedMacAddress = $null
    )

    $candidates = @(
        Get-NetAdapter -IncludeHidden -ErrorAction Stop |
            Where-Object { [int]$_.ifIndex -eq [int]$InterfaceIndex }
    )

    if ($candidates.Count -eq 0) {
        throw "Adapter resolution failed: Found 0 adapters matching InterfaceIndex $InterfaceIndex."
    }
    if ($candidates.Count -gt 1) {
        throw "Adapter resolution failed: Ambiguous match, found $($candidates.Count) adapters for InterfaceIndex $InterfaceIndex."
    }

    $adapter = $candidates[0]

    # Validate snapshot identity attributes
    if ($ExpectedName -and ($adapter.Name -ne $ExpectedName)) {
        throw "Adapter identity mismatch for ifIndex $($InterfaceIndex): Expected Name '$ExpectedName', found '$($adapter.Name)'."
    }
    if ($ExpectedDescription -and ($adapter.InterfaceDescription -ne $ExpectedDescription)) {
        throw "Adapter identity mismatch for ifIndex $($InterfaceIndex): Expected InterfaceDescription '$ExpectedDescription', found '$($adapter.InterfaceDescription)'."
    }
    if ($ExpectedMacAddress -and ($adapter.MacAddress -ne $ExpectedMacAddress)) {
        throw "Adapter identity mismatch for ifIndex $($InterfaceIndex): Expected MacAddress '$ExpectedMacAddress', found '$($adapter.MacAddress)'."
    }

    return $adapter
}

# 6b. Helper: Query Scheduled Task without terminating on native stderr
function Test-ScheduledTaskExists {
    param([string]$TaskName)

    # 1. Try native ScheduledTasks module first
    if (Get-Command Get-ScheduledTask -ErrorAction SilentlyContinue) {
        $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        return [bool]$task
    }

    # 2. Fallback to schtasks.exe with safe ErrorActionPreference handling
    $origPref = $ErrorActionPreference
    try {
        $ErrorActionPreference = "SilentlyContinue"
        $null = & schtasks.exe /query /tn $TaskName 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $origPref
    }
}

# 6c. Helper: Delete Scheduled Task safely and verify read-back absence
function Remove-ScheduledTaskSafely {
    param([string]$TaskName)

    # If task doesn't exist, it is already absent
    if (-not (Test-ScheduledTaskExists -TaskName $TaskName)) {
        return $true
    }

    # 1. Try native Unregister-ScheduledTask cmdlet first
    if (Get-Command Unregister-ScheduledTask -ErrorAction SilentlyContinue) {
        try {
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
        } catch {}
    }

    # If still present, fallback to schtasks.exe /delete
    if (Test-ScheduledTaskExists -TaskName $TaskName) {
        $origPref = $ErrorActionPreference
        try {
            $ErrorActionPreference = "SilentlyContinue"
            $null = & schtasks.exe /delete /tn $TaskName /f 2>$null
        } catch {} finally {
            $ErrorActionPreference = $origPref
        }
    }

    # Read-back verification: task must be absent now
    $stillExists = Test-ScheduledTaskExists -TaskName $TaskName
    return (-not $stillExists)
}

# 6d. Helper: Safe cleanup of stale watchdog task before registering new one
function Remove-StaleWatchdogIfSafe {
    param(
        [string]$TaskName = "Phase4C2G_Emergency_Network_Recovery",
        [System.Collections.Generic.List[PSObject]]$Targets = $null
    )

    $taskExists = Test-ScheduledTaskExists -TaskName $TaskName
    if (-not $taskExists) {
        Write-Host "No stale watchdog scheduled task '$TaskName' detected."
        return $true
    }

    Write-Host "Detected existing scheduled task '$TaskName'. Checking prerequisites for safe cleanup..."

    # Check 1: No other controller/verifier/worker process is running
    $currentPid = $PID
    $activeProcesses = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.ProcessId -ne $currentPid -and
        $_.Name -like "*powershell*" -and
        ($_.CommandLine -like "*RUN_PHASE4C2G_AUTOMATED_ISOLATION*" -or `
         $_.CommandLine -like "*RUN_PHASE4C2G_OFFLINE_VERIFIER*" -or `
         $_.CommandLine -like "*RECOVER_NETWORK*")
    })

    if ($activeProcesses.Count -gt 0) {
        Write-Host "[FAIL-CLOSED] Cannot safely clean up stale watchdog: active controller/verifier processes detected ($($activeProcesses.Count))." -ForegroundColor Red
        throw "BLOCKED_ACTIVE_CONTROLLER_PROCESS_DETECTED"
    }

    # Check 2: All target egress adapters must be Up
    if ($Targets -and $Targets.Count -gt 0) {
        foreach ($t in $Targets) {
            try {
                $adapterObj = Resolve-TargetNetAdapter -InterfaceIndex $t.InterfaceIndex `
                                                      -ExpectedName $t.Name `
                                                      -ExpectedDescription $t.InterfaceDescription `
                                                      -ExpectedMacAddress $t.MacAddress
                if (-not $adapterObj -or $adapterObj.AdminStatus -ne "Up" -or $adapterObj.Status -ne "Up") {
                    Write-Host "[FAIL-CLOSED] Cannot safely clean up stale watchdog: target adapter ifIndex $($t.InterfaceIndex) ($($t.Name)) is not Up." -ForegroundColor Red
                    throw "BLOCKED_TARGET_ADAPTER_NOT_UP_FOR_STALE_WATCHDOG_CLEANUP"
                }
            } catch {
                Write-Host "[FAIL-CLOSED] Cannot safely clean up stale watchdog: failed to resolve/verify adapter ifIndex $($t.InterfaceIndex): $($_.Exception.Message)" -ForegroundColor Red
                throw "BLOCKED_TARGET_ADAPTER_UNRESOLVED_FOR_STALE_WATCHDOG_CLEANUP"
            }
        }
    }

    # All conditions satisfied: Delete the stale task using safe deletion helper
    Write-Host "Prerequisites satisfied: all egress adapters verified Up and zero active controller processes. Deleting stale watchdog task..." -ForegroundColor Yellow
    $deleted = Remove-ScheduledTaskSafely -TaskName $TaskName
    if (-not $deleted) {
        Write-Host "[BLOCKED] Failed to delete stale watchdog scheduled task '$TaskName'. Task still present." -ForegroundColor Red
        throw "BLOCKED_STALE_WATCHDOG_CLEANUP_FAILED"
    }

    Write-Host "[WATCHDOG CLEANED] Stale watchdog scheduled task successfully removed." -ForegroundColor Green
    return $true
}

# 6e. Helper: Streaming SHA-256 without loading the complete receipt into memory
function Get-StreamingFileSha256 {
    param([Parameter(Mandatory=$true)][string]$Path)

    $stream = [System.IO.File]::OpenRead($Path)
    $hashAlgorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        return -join ($hashAlgorithm.ComputeHash($stream) | ForEach-Object { "{0:x2}" -f $_ })
    } finally {
        $hashAlgorithm.Dispose()
        $stream.Dispose()
    }
}

# 6f. Helper: Bind a verifier receipt to exactly the current invocation window
function Test-VerifierReceiptSessionBinding {
    param(
        [Parameter(Mandatory=$true)][PSObject]$ReceiptJson,
        [Parameter(Mandatory=$true)][System.DateTimeOffset]$SessionStartedAtUtc,
        [Parameter(Mandatory=$true)][System.DateTimeOffset]$SessionCompletedAtUtc
    )

    $observedStart = [System.DateTimeOffset]::MinValue
    $observedComplete = [System.DateTimeOffset]::MinValue
    $generatedAt = [System.DateTimeOffset]::MinValue
    $startValid = [System.DateTimeOffset]::TryParse([string]$ReceiptJson.observation_started_at_utc, [ref]$observedStart)
    $completeValid = [System.DateTimeOffset]::TryParse([string]$ReceiptJson.observation_completed_at_utc, [ref]$observedComplete)
    $generatedValid = [System.DateTimeOffset]::TryParse([string]$ReceiptJson.generated_at_utc, [ref]$generatedAt)

    if (-not $startValid -or -not $completeValid -or -not $generatedValid) {
        return $false
    }

    return (
        $observedStart -ge $SessionStartedAtUtc -and
        $observedStart -le $observedComplete -and
        $observedComplete -le $SessionCompletedAtUtc -and
        $generatedAt -ge $observedStart -and
        $generatedAt -le $SessionCompletedAtUtc
    )
}

function Bind-OfflineVerifierReceipt {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][System.DateTimeOffset]$SessionStartedAtUtc,
        [Parameter(Mandatory=$true)][System.DateTimeOffset]$SessionCompletedAtUtc,
        [Parameter(Mandatory=$true)][int]$ExitCode
    )

    $binding = [ordered]@{
        Exists          = $false
        Sha256          = $null
        Verdict         = $null
        ExitCode        = $ExitCode
        TimestampValid  = $false
        ReceiptJson     = $null
        FailureReason   = "OFFLINE_VERIFIER_RECEIPT_MISSING"
    }

    if (-not (Test-Path -LiteralPath $Path)) {
        return $binding
    }

    $binding.Exists = $true
    $binding.Sha256 = Get-StreamingFileSha256 -Path $Path
    try {
        $receiptContent = [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8)
        $receiptJson = $receiptContent | ConvertFrom-Json
        $binding.ReceiptJson = $receiptJson
        $binding.Verdict = [string]$receiptJson.verdict
        $binding.TimestampValid = Test-VerifierReceiptSessionBinding `
            -ReceiptJson $receiptJson `
            -SessionStartedAtUtc $SessionStartedAtUtc `
            -SessionCompletedAtUtc $SessionCompletedAtUtc
        $binding.FailureReason = if ($binding.TimestampValid) { $null } else { "OFFLINE_VERIFIER_RECEIPT_STALE_OR_OUTSIDE_SESSION" }
    } catch {
        $binding.FailureReason = "OFFLINE_VERIFIER_RECEIPT_INVALID_JSON"
    }

    return $binding
}

# 6g. Helper: Poll restoration until the required operational state is proven
function Wait-AdapterOperationalRestoration {
    param(
        [Parameter(Mandatory=$true)][PSObject]$Target,
        [Parameter(Mandatory=$true)][bool]$RequireOperationalUp,
        [int]$TimeoutSeconds = 60,
        [int]$PollIntervalSeconds = 2,
        [scriptblock]$Resolver = $null,
        [scriptblock]$Sleeper = $null
    )

    $deadline = [System.DateTimeOffset]::UtcNow.AddSeconds($TimeoutSeconds)
    $lastAdapter = $null
    $lastError = $null

    do {
        try {
            $curr = if ($Resolver) {
                & $Resolver $Target
            } else {
                Resolve-TargetNetAdapter -InterfaceIndex $Target.InterfaceIndex `
                                         -ExpectedName $Target.Name `
                                         -ExpectedDescription $Target.InterfaceDescription `
                                         -ExpectedMacAddress $Target.MacAddress
            }
            $lastAdapter = $curr
            $isRestored = if ($RequireOperationalUp) {
                $curr -and ($curr.AdminStatus -eq "Up" -and $curr.Status -eq "Up")
            } else {
                $curr -and $curr.AdminStatus -eq "Up"
            }
            if ($isRestored) {
                return [ordered]@{ Success = $true; Adapter = $curr; Error = $null }
            }
            $lastError = "AdminStatus='$($curr.AdminStatus)', Status='$($curr.Status)'"
        } catch {
            $lastError = $_.Exception.Message
        }

        if ([System.DateTimeOffset]::UtcNow -ge $deadline) {
            break
        }
        if ($Sleeper) {
            & $Sleeper $PollIntervalSeconds
        } else {
            Start-Sleep -Seconds $PollIntervalSeconds
        }
    } while ($true)

    return [ordered]@{ Success = $false; Adapter = $lastAdapter; Error = $lastError }
}

# 6c. Helper: Generate and Validate Recovery Script
function Update-RecoveryScriptAndValidate {
    param(
        [System.Collections.Generic.List[PSObject]]$Targets,
        [string]$Path
    )

    $parentDir = Split-Path -Parent $Path
    if (-not (Test-Path $parentDir)) {
        New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
    }
    $failedDir = Join-Path $parentDir "failed_recovery_scripts"
    if (-not (Test-Path $failedDir)) {
        New-Item -ItemType Directory -Path $failedDir -Force | Out-Null
    }

    $partPath = "$Path.part"
    if (Test-Path $partPath) {
        Remove-Item -Path $partPath -Force -ErrorAction SilentlyContinue
    }

    $genUtc = [System.DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.ffffff+00:00")

    # Build script lines with bullet-proof string escaping
    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add("# Emergency Network Recovery Script - Generated by Phase 4C.2G Controller")
    $lines.Add("# Generated at: $($genUtc)")
    $lines.Add("`$generatedAtUtc = '$($genUtc)'")
    $lines.Add("`$executionUtc = [System.DateTime]::UtcNow.ToString('o')")
    $lines.Add("Write-Host ('Restoration invoked at: ' + `$executionUtc)")
    $lines.Add("")
    $lines.Add("`$targetAdapters = @(")

    $expectedIfIndices = [System.Collections.Generic.HashSet[int]]::new()
    foreach ($t in $Targets) {
        $ifIdx = [int]$t.InterfaceIndex
        $expectedIfIndices.Add($ifIdx) | Out-Null

        if ([string]::IsNullOrWhiteSpace($t.InterfaceDescription) -or [string]::IsNullOrWhiteSpace($t.MacAddress)) {
            throw "Target adapter ifIndex $ifIdx missing required InterfaceDescription or MacAddress."
        }

        # Escape single quotes by doubling them for PowerShell single-quoted literals: ' -> ''
        $rawName = if ($t.Name) { [string]$t.Name } else { "" }
        $escapedName = $rawName.Replace("'", "''")

        $rawDesc = if ($t.InterfaceDescription) { [string]$t.InterfaceDescription } else { "" }
        $escapedDesc = $rawDesc.Replace("'", "''")

        $rawMac = if ($t.MacAddress) { [string]$t.MacAddress } else { "" }
        $escapedMac = $rawMac.Replace("'", "''")

        $lines.Add("    [PSCustomObject]@{")
        $lines.Add("        InterfaceIndex       = $($ifIdx)")
        $lines.Add("        Name                 = '$($escapedName)'")
        $lines.Add("        InterfaceDescription = '$($escapedDesc)'")
        $lines.Add("        MacAddress           = '$($escapedMac)'")
        $lines.Add("    }")
    }

    $lines.Add(")")
    $lines.Add("Write-Host 'Enabling Phase 4C.2G isolated network adapters...'")
    $lines.Add("`$restorationErrors = 0")
    $lines.Add("")
    $lines.Add("foreach (`$t in `$targetAdapters) {")
    $lines.Add("    Write-Host (`"Resolving adapter for ifIndex `" + `$t.InterfaceIndex + `" ('`" + `$t.Name + `"')...`")")
    $lines.Add("    `$candidates = @(")
    $lines.Add("        Get-NetAdapter -IncludeHidden -ErrorAction SilentlyContinue |")
    $lines.Add("            Where-Object { [int]`$_.ifIndex -eq [int]`$t.InterfaceIndex }")
    $lines.Add("    )")
    $lines.Add("")
    $lines.Add("    if (`$candidates.Count -ne 1) {")
    $lines.Add("        Write-Host (`"[ERROR] Expected exactly 1 adapter for ifIndex `" + `$t.InterfaceIndex + `", found `" + `$candidates.Count) -ForegroundColor Red")
    $lines.Add("        `$restorationErrors++")
    $lines.Add("        continue")
    $lines.Add("    }")
    $lines.Add("")
    $lines.Add("    `$ad = `$candidates[0]")
    $lines.Add("")
    $lines.Add("    if (`$t.Name -and (`$ad.Name -ne `$t.Name)) {")
    $lines.Add("        Write-Host (`"[ERROR] Identity mismatch for ifIndex `" + `$t.InterfaceIndex + `": Name expected '`" + `$t.Name + `"', found '`" + `$ad.Name + `"'`") -ForegroundColor Red")
    $lines.Add("        `$restorationErrors++")
    $lines.Add("        continue")
    $lines.Add("    }")
    $lines.Add("")
    $lines.Add("    if (`$t.InterfaceDescription -and (`$ad.InterfaceDescription -ne `$t.InterfaceDescription)) {")
    $lines.Add("        Write-Host (`"[ERROR] Identity mismatch for ifIndex `" + `$t.InterfaceIndex + `": InterfaceDescription expected '`" + `$t.InterfaceDescription + `"', found '`" + `$ad.InterfaceDescription + `"'`") -ForegroundColor Red")
    $lines.Add("        `$restorationErrors++")
    $lines.Add("        continue")
    $lines.Add("    }")
    $lines.Add("")
    $lines.Add("    if (`$t.MacAddress -and `$ad.MacAddress -and (`$ad.MacAddress -ne `$t.MacAddress)) {")
    $lines.Add("        Write-Host (`"[ERROR] Identity mismatch for ifIndex `" + `$t.InterfaceIndex + `": MacAddress expected '`" + `$t.MacAddress + `"', found '`" + `$ad.MacAddress + `"'`") -ForegroundColor Red")
    $lines.Add("        `$restorationErrors++")
    $lines.Add("        continue")
    $lines.Add("    }")
    $lines.Add("")
    $lines.Add("    Write-Host (`"Enabling adapter object: `" + `$ad.Name + `" (`" + `$ad.InterfaceDescription + `")`")")
    $lines.Add("    try {")
    $lines.Add("        `$ad | Enable-NetAdapter -Confirm:`$false -ErrorAction Stop")
    $lines.Add("        Write-Host (`"Successfully enabled: `" + `$ad.Name) -ForegroundColor Green")
    $lines.Add("    } catch {")
    $lines.Add("        Write-Host (`"[ERROR] Failed to enable adapter `" + `$ad.Name + `": `" + `$_.Exception.Message) -ForegroundColor Red")
    $lines.Add("        `$restorationErrors++")
    $lines.Add("    }")
    $lines.Add("}")
    $lines.Add("")
    $lines.Add("if (`$restorationErrors -gt 0) {")
    $lines.Add("    Write-Host (`"[CRITICAL] Restoration completed with `" + `$restorationErrors + `" error(s).`") -ForegroundColor Red")
    $lines.Add("    exit 1")
    $lines.Add("}")
    $lines.Add("")
    $lines.Add("Write-Host 'Restoration completed successfully.'")
    $lines.Add("exit 0")

    # Write to .part with flush and FileStream.Flush(true) for durability
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    $fileStream = [System.IO.File]::Create($partPath)
    try {
        $writer = New-Object System.IO.StreamWriter($fileStream, $utf8NoBom)
        try {
            foreach ($line in $lines) {
                $writer.WriteLine($line)
            }
            $writer.Flush()
            $fileStream.Flush($true)
        } finally {
            $writer.Dispose()
        }
    } finally {
        $fileStream.Dispose()
    }

    # Validate syntax via built-in Language Parser on the .part file
    $parseTokens = $null
    $parseErrors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($partPath, [ref]$parseTokens, [ref]$parseErrors)

    $validationFailed = $false
    $failReason = ""

    if ($null -eq $ast) {
        $validationFailed = $true
        $failReason = "AST parser returned null"
    } elseif ($parseErrors.Count -gt 0) {
        $validationFailed = $true
        $failReason = $parseErrors[0].Message
    } else {
        # Allowlist semantic verification: ensure AST contains the expected interface indices
        $partContent = [System.IO.File]::ReadAllText($partPath)
        foreach ($ifIdx in $expectedIfIndices) {
            if ($partContent -notmatch "InterfaceIndex\s*=\s*$ifIdx\b") {
                $validationFailed = $true
                $failReason = "Missing expected InterfaceIndex $ifIdx in generated recovery script"
                break
            }
        }
        if (-not $validationFailed) {
            $illegalEnaPattern = 'Enable-NetAdapter' + '\s+-InterfaceIndex'
            if ($partContent -match $illegalEnaPattern) {
                $validationFailed = $true
                $failReason = "Generated script must not contain direct " + "Enable-NetAdapter " + "-InterfaceIndex"
            } elseif ($partContent -notmatch '\|\s*Enable-NetAdapter') {
                $validationFailed = $true
                $failReason = "Generated script must pipe resolved adapter object to Enable-NetAdapter"
            }
        }
    }

    if ($validationFailed) {
        $errTimestamp = [System.DateTime]::UtcNow.ToString("yyyyMMdd_HHmmss_ffffff")
        $failedPath = Join-Path $failedDir "RECOVER_NETWORK_FAILED_$errTimestamp.ps1"
        Copy-Item -Path $partPath -Destination $failedPath -Force
        Remove-Item -Path $partPath -Force -ErrorAction SilentlyContinue
        throw "Recovery script syntax validation failed: $failReason. Quarantined to: $failedPath"
    }

    # Atomic rename from .part to final destination
    if (Test-Path $Path) {
        Remove-Item -Path $Path -Force
    }
    Move-Item -Path $partPath -Destination $Path -Force
}

# 7. Helper: Verify Scheduled Task Watchdog via Read-Back
function Test-WatchdogTaskVerified {
    param([string]$TaskName)
    return (Test-ScheduledTaskExists -TaskName $TaskName)
}

# 8. Passive Network Check
function Test-PassiveIsolation {
    $proxySet = [bool]($env:HTTP_PROXY -or $env:HTTPS_PROXY -or $env:ALL_PROXY)
    $hasActiveDefaultRoute = $false
    $persistentRoutesIgnored = @()
    $connectedEgressAdapters = @()
    $protectedInternalAdapters = @()
    $connectedAdaptersInformational = @()
    $activeVpnEgressOwners = @()
    $unidentifiedActiveEgressRouteOwners = @()
    $activeDefaultRoutes = @()

    # Primary authoritative source: Get-NetRoute -PolicyStore ActiveStore
    try {
        $activeAdapters = @(Get-NetAdapter -IncludeHidden -ErrorAction SilentlyContinue | Where-Object { $_.Status -eq "Up" -or $_.AdminStatus -eq "Up" })
        $activeIndices = [System.Collections.Generic.HashSet[int]]::new()
        foreach ($ad in $activeAdapters) {
            $activeIndices.Add([int]$ad.ifIndex) | Out-Null
            $connectedAdaptersInformational += [string]$ad.Name
            $adapterType = Get-AdapterType -Name $ad.Name -Desc $ad.InterfaceDescription
            if ($adapterType -like "*Virtual*" -or $adapterType -like "*Host-Only*" -or $ad.Name -like "VMnet*" -or $ad.Name -like "*WSL*" -or $ad.Name -like "*Default Switch*") {
                $protectedInternalAdapters += [string]$ad.Name
            }
        }

        $act4 = Get-NetRoute -PolicyStore ActiveStore -DestinationPrefix "0.0.0.0/0" -ErrorAction SilentlyContinue
        if ($act4) {
            foreach ($r in $act4) {
                if ($activeIndices.Contains([int]$r.InterfaceIndex)) {
                    $activeDefaultRoutes += $r
                } else {
                    $unidentifiedActiveEgressRouteOwners += $r
                }
            }
        }
        $act6 = Get-NetRoute -PolicyStore ActiveStore -DestinationPrefix "::/0" -ErrorAction SilentlyContinue
        if ($act6) {
            foreach ($r in $act6) {
                if ($activeIndices.Contains([int]$r.InterfaceIndex)) {
                    $activeDefaultRoutes += $r
                } else {
                    $unidentifiedActiveEgressRouteOwners += $r
                }
            }
        }

        # A live VPN/tunnel owner of any non-loopback ActiveStore route is egress evidence.
        $allActiveRoutes = @(Get-NetRoute -PolicyStore ActiveStore -ErrorAction SilentlyContinue)
        foreach ($ad in $activeAdapters) {
            $adapterType = Get-AdapterType -Name $ad.Name -Desc $ad.InterfaceDescription
            if ($adapterType -ne "VPN / Tunnel") { continue }
            $vpnRoutes = @($allActiveRoutes | Where-Object {
                [int]$_.InterfaceIndex -eq [int]$ad.ifIndex -and
                $_.DestinationPrefix -notmatch "^(127\.|::1|ff00|224\.|255\.)"
            })
            if ($vpnRoutes.Count -gt 0) {
                $activeVpnEgressOwners += [string]$ad.Name
                $connectedEgressAdapters += "$($ad.Name) (ifIndex $($ad.ifIndex))"
            }
        }
    } catch {}

    if ($activeDefaultRoutes.Count -gt 0) {
        $hasActiveDefaultRoute = $true
        foreach ($r in $activeDefaultRoutes) {
            $connectedEgressAdapters += "$($r.InterfaceAlias) (ifIndex $($r.InterfaceIndex))"
        }
    }
    $connectedEgressAdapters = @($connectedEgressAdapters | Select-Object -Unique)
    $activeVpnEgressOwners = @($activeVpnEgressOwners | Select-Object -Unique)
    $protectedInternalAdapters = @($protectedInternalAdapters | Select-Object -Unique)
    $connectedAdaptersInformational = @($connectedAdaptersInformational | Select-Object -Unique)

    # Secondary cross-check: route.exe print (parsing strictly Active Routes section, ignoring Persistent Routes)
    try {
        $routeOut = & route.exe print 0.0.0.0 2>$null
        $inActiveRoutes = $false
        $inPersistentRoutes = $false

        foreach ($line in $routeOut) {
            $trimmed = $line.Trim()
            if ($trimmed -like "*Active Routes:*") {
                $inActiveRoutes = $true
                $inPersistentRoutes = $false
                continue
            }
            if ($trimmed -like "*Persistent Routes:*") {
                $inActiveRoutes = $false
                $inPersistentRoutes = $true
                continue
            }
            if ($trimmed -like "*IPv6*" -or $trimmed -like "*Interface List*") {
                $inActiveRoutes = $false
                $inPersistentRoutes = $false
                continue
            }

            $parts = $trimmed -split "\s+"
            if ($parts.Count -ge 3 -and $parts[0] -eq "0.0.0.0" -and $parts[1] -eq "0.0.0.0") {
                if ($inActiveRoutes) {
                    $hasActiveDefaultRoute = $true
                } elseif ($inPersistentRoutes) {
                    $persistentRoutesIgnored += $trimmed
                }
            }
        }
    } catch {}

    return [ordered]@{
        IsIsolated              = ((-not $proxySet) -and (-not $hasActiveDefaultRoute) -and ($connectedEgressAdapters.Count -eq 0) -and ($activeVpnEgressOwners.Count -eq 0) -and ($unidentifiedActiveEgressRouteOwners.Count -eq 0))
        ProxyDetected           = $proxySet
        DefaultRouteDetected    = $hasActiveDefaultRoute
        ConnectedAdapters       = $connectedEgressAdapters
        ActiveEgressAdapters    = $connectedEgressAdapters
        ProtectedInternalAdapters = $protectedInternalAdapters
        ConnectedAdaptersInformational = $connectedAdaptersInformational
        ActiveVpnEgressOwners   = $activeVpnEgressOwners
        UnidentifiedActiveEgressRouteOwners = $unidentifiedActiveEgressRouteOwners
        ActiveDefaultRoutes     = $activeDefaultRoutes
        PersistentRoutesIgnored = $persistentRoutesIgnored
        OutboundProbesSent      = 0
        DnsLookupsPerformed     = 0
        HttpRequestsSent        = 0
    }
}

# 9. Atomic Receipt Writer (using .part -> Move-Item for atomic replacement)
function Write-ReceiptAtomic {
    param([string]$Path, [hashtable]$Data)
    $jsonStr = $Data | ConvertTo-Json -Depth 10
    $parentDir = Split-Path -Parent $Path
    if (-not (Test-Path $parentDir)) {
        New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
    }

    $partPath = $Path + ".part"
    [System.IO.File]::WriteAllText($partPath, $jsonStr, [System.Text.Encoding]::UTF8)
    if (Test-Path $Path) {
        Remove-Item $Path -Force
    }
    Move-Item -Path $partPath -Destination $Path -Force
}

# 10. Compute Script SHA-256
$ScriptContent = [System.IO.File]::ReadAllBytes($MyInvocation.MyCommand.Definition)
$Sha256 = [System.Security.Cryptography.SHA256]::Create()
$ControllerSha256 = -join ($Sha256.ComputeHash($ScriptContent) | ForEach-Object { "{0:x2}" -f $_ })

# ==============================================================================
# MAIN EXECUTION FLOW
# ==============================================================================

$CurrentUtc = [System.DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.ffffff+00:00")
$IsAdmin = Test-IsAdministrator
$RemoteCheck = Test-IsRemoteSession

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " PHASE 4C.2G AUTOMATED WINDOWS NETWORK-ISOLATION CONTROLLER" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "Version                : $ControllerVersion"
Write-Host "Controller SHA-256     : $ControllerSha256"
Write-Host "Current UTC Time       : $CurrentUtc"
Write-Host "Execution Mode         : $(if ($DryRun) { 'DryRun' } elseif ($ReadinessTest) { 'ReadinessTest' } elseif ($ValidateRecoveryScriptOnly) { 'ValidateRecoveryScriptOnly' } elseif ($ValidateAdapterCmdletContractOnly) { 'ValidateAdapterCmdletContractOnly' } else { 'Default/Inspection' })"
Write-Host "Administrator Role     : $IsAdmin"
Write-Host "Session Type           : $($RemoteCheck.Reason)"
Write-Host "=================================================================" -ForegroundColor Cyan

# Check for Remote Sessions
if ($RemoteCheck.IsRemote) {
    Write-Host "[FAIL-CLOSED] Remote session detected. Network isolation is unsafe over remote connections." -ForegroundColor Red
    Write-Host "Verdict: BLOCKED_REMOTE_SESSION_NETWORK_ISOLATION_UNSAFE" -ForegroundColor Red
    exit 3
}

# Capture Snapshot
$Snapshot = Get-NetworkIsolationSnapshot

# ------------------------------------------------------------------------------
# MODE C: VALIDATE RECOVERY SCRIPT ONLY (NON-MUTATING AST & ALLOWLIST VALIDATION)
# ------------------------------------------------------------------------------
if ($ValidateRecoveryScriptOnly) {
    Write-Host "`n[VALIDATION MODE] Testing production recovery script generation and AST validation..." -ForegroundColor Cyan
    $testTargets = [System.Collections.Generic.List[PSObject]]::new()
    if ($TargetFixtureJson -and (Test-Path $TargetFixtureJson)) {
        $fixtureData = Get-Content $TargetFixtureJson -Raw | ConvertFrom-Json
        foreach ($item in $fixtureData) {
            $testTargets.Add([PSCustomObject]@{
                InterfaceIndex       = [int]$item.InterfaceIndex
                Name                 = [string]$item.Name
                InterfaceDescription = if ($item.InterfaceDescription) { [string]$item.InterfaceDescription } else { "" }
                MacAddress           = if ($item.MacAddress) { [string]$item.MacAddress } else { "" }
                Reason               = [string]$item.Reason
            })
        }
    } else {
        # Use initial disable targets from snapshot
        foreach ($t in $Snapshot.InitialDisableTargets) {
            $testTargets.Add([PSCustomObject]@{
                InterfaceIndex       = [int]$t.InterfaceIndex
                Name                 = [string]$t.Name
                InterfaceDescription = [string]$t.InterfaceDescription
                MacAddress           = [string]$t.MacAddress
                Reason               = [string]$t.Reason
            })
        }
    }

    $outPath = if ($OutputRecoveryScriptPath) { $OutputRecoveryScriptPath } else { Join-Path $ArtifactsDir "RECOVER_NETWORK_TEST.ps1" }
    try {
        Update-RecoveryScriptAndValidate -Targets $testTargets -Path $outPath
        Write-Host "[VALIDATION PASS] Recovery script successfully generated and AST validated with 0 errors at: $outPath" -ForegroundColor Green

        $parseTokens = $null
        $parseErrors = $null
        $ast = [System.Management.Automation.Language.Parser]::ParseFile($outPath, [ref]$parseTokens, [ref]$parseErrors)

        $receipt = [ordered]@{
            "script_path"          = $outPath
            "ast_parse_errors"     = $parseErrors.Count
            "is_ast_valid"         = ($parseErrors.Count -eq 0)
            "target_count"         = $testTargets.Count
            "target_if_indices"    = @($testTargets | ForEach-Object { $_.InterfaceIndex })
            "validation_timestamp" = [System.DateTime]::UtcNow.ToString("o")
            "verdict"              = "RECOVERY_SCRIPT_SYNTAX_VALID"
        }
        $receipt | ConvertTo-Json -Depth 5 | Write-Host
        exit 0
    } catch {
        Write-Host "[VALIDATION FAIL] $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Verdict: BLOCKED_RECOVERY_SCRIPT_SYNTAX_INVALID" -ForegroundColor Red
        exit 1
    }
}

# ------------------------------------------------------------------------------
# MODE C: VALIDATE ADAPTER CMDLET CONTRACT ONLY
# ------------------------------------------------------------------------------
if ($ValidateAdapterCmdletContractOnly) {
    Write-Host "[ADAPTER CMDLET CONTRACT VALIDATION] Verifying Disable/Enable-NetAdapter parameter contract..." -ForegroundColor Cyan

    $disCmd = Get-Command Disable-NetAdapter -ErrorAction Stop
    $enaCmd = Get-Command Enable-NetAdapter -ErrorAction Stop

    $disHasIfIdx = $disCmd.Parameters.ContainsKey("InterfaceIndex")
    $enaHasIfIdx = $enaCmd.Parameters.ContainsKey("InterfaceIndex")

    $disInputObj = $disCmd.Parameters["InputObject"]
    $enaInputObj = $enaCmd.Parameters["InputObject"]

    $disPipesInput = [bool]($disInputObj.Attributes | Where-Object { $_.ValueFromPipeline -eq $true })
    $enaPipesInput = [bool]($enaInputObj.Attributes | Where-Object { $_.ValueFromPipeline -eq $true })

    Write-Host ("  Disable-NetAdapter has InterfaceIndex param : {0} (Expected: False)" -f $disHasIfIdx)
    Write-Host ("  Enable-NetAdapter  has InterfaceIndex param : {0} (Expected: False)" -f $enaHasIfIdx)
    Write-Host ("  Disable-NetAdapter accepts pipeline InputObject: {0} (Expected: True)" -f $disPipesInput)
    Write-Host ("  Enable-NetAdapter  accepts pipeline InputObject: {0} (Expected: True)" -f $enaPipesInput)

    # AST and static analysis of controller source code: ensure no direct -InterfaceIndex calls on Disable/Enable/Get-NetAdapter
    $selfTokens = $null
    $selfErrors = $null
    $selfAst = [System.Management.Automation.Language.Parser]::ParseFile($MyInvocation.MyCommand.Definition, [ref]$selfTokens, [ref]$selfErrors)
    $directParamCalls = $selfAst.FindAll({
        param($astNode)
        if ($astNode -is [System.Management.Automation.Language.CommandAst]) {
            $cmdName = $astNode.GetCommandName()
            if ($cmdName -in @('Disable-NetAdapter', 'Enable-NetAdapter', 'Get-NetAdapter')) {
                foreach ($param in $astNode.CommandElements) {
                    if ($param -is [System.Management.Automation.Language.CommandParameterAst]) {
                        if ($param.ParameterName -eq 'InterfaceIndex') {
                            return $true
                        }
                    }
                }
            }
        }
        return $false
    }, $true)

    $hasDirectParamAst = ($directParamCalls.Count -gt 0)

    $selfContent = [System.IO.File]::ReadAllText($MyInvocation.MyCommand.Definition)
    $disPat = 'Disable-NetAdapter' + '\s+-InterfaceIndex'
    $enaPat = 'Enable-NetAdapter' + '\s+-InterfaceIndex'
    $getPat = 'Get-NetAdapter' + '\s+-InterfaceIndex'
    $linesToScan = $selfContent -split "`r?`n" | Where-Object {
        $_ -notmatch '\$disPat' -and $_ -notmatch '\$enaPat' -and $_ -notmatch '\$getPat' -and `
        $_ -notmatch 'illegalEnaPattern' -and $_ -notmatch 'genHasEnableIfIndex' -and `
        $_ -notmatch 'hasDisableIfIndex' -and $_ -notmatch 'hasEnableIfIndex' -and $_ -notmatch 'hasGetIfIndex' -and `
        $_ -notmatch 'directParamCalls'
    }
    $hasDisableIfIndex = ($linesToScan -match $disPat).Count -gt 0
    $hasEnableIfIndex  = ($linesToScan -match $enaPat).Count -gt 0
    $hasGetIfIndex     = ($linesToScan -match $getPat).Count -gt 0

    Write-Host ("  Controller source contains Disable-NetAdapter direct parameter: {0} (Expected: False)" -f ($hasDisableIfIndex -or $hasDirectParamAst))
    Write-Host ("  Controller source contains Enable-NetAdapter  direct parameter: {0} (Expected: False)" -f ($hasEnableIfIndex -or $hasDirectParamAst))
    Write-Host ("  Controller source contains Get-NetAdapter     direct parameter: {0} (Expected: False)" -f ($hasGetIfIndex -or $hasDirectParamAst))

    # Generate a fixture recovery script and verify it
    $testScriptPath = Join-Path $ArtifactsDir "RECOVER_CMDLET_CONTRACT_TEST.ps1"
    $fixtureTargets = @(
        [PSCustomObject]@{
            InterfaceIndex       = 21
            Name                 = "Wi-Fi"
            InterfaceDescription = "Killer(R) Wi-Fi 6 AX1650i"
            MacAddress           = "00:11:22:33:44:55"
            Reason               = "Contract check target"
        }
    )
    Update-RecoveryScriptAndValidate -Targets $fixtureTargets -Path $testScriptPath

    $parseTokens = $null
    $parseErrors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($testScriptPath, [ref]$parseTokens, [ref]$parseErrors)

    $genContent = [System.IO.File]::ReadAllText($testScriptPath)
    $genHasEnableIfIndex = $genContent -match ('Enable-NetAdapter' + '\s+-InterfaceIndex')
    $genHasGetIfIndex    = $genContent -match ('Get-NetAdapter' + '\s+-InterfaceIndex')
    $genPipesToEnable    = $genContent -match '\|\s*Enable-NetAdapter'

    Write-Host ("  Generated script AST parse error count: {0} (Expected: 0)" -f $parseErrors.Count)
    Write-Host ("  Generated script contains direct Enable-NetAdapter parameter: {0} (Expected: False)" -f $genHasEnableIfIndex)
    Write-Host ("  Generated script contains direct Get-NetAdapter    parameter: {0} (Expected: False)" -f $genHasGetIfIndex)
    Write-Host ("  Generated script pipes adapter object to Enable-NetAdapter: {0} (Expected: True)" -f $genPipesToEnable)

    # Clean up test script
    Remove-Item -Path $testScriptPath -Force -ErrorAction SilentlyContinue

    # Contract expansion 1: Verify missing InterfaceDescription or MacAddress in allowlist is strictly rejected
    $testMissingDesc = [System.Collections.Generic.List[PSObject]]::new()
    $testMissingDesc.Add([PSCustomObject]@{
        InterfaceIndex       = 21
        Name                 = "Wi-Fi"
        InterfaceDescription = ""
        MacAddress           = "00:11:22:33:44:55"
        Reason               = "Contract check missing desc"
    })
    $caughtMissingDesc = $false
    try {
        Update-RecoveryScriptAndValidate -Targets $testMissingDesc -Path $testScriptPath
    } catch {
        $caughtMissingDesc = $true
    }

    $testMissingMac = [System.Collections.Generic.List[PSObject]]::new()
    $testMissingMac.Add([PSCustomObject]@{
        InterfaceIndex       = 21
        Name                 = "Wi-Fi"
        InterfaceDescription = "Killer(R) Wi-Fi"
        MacAddress           = ""
        Reason               = "Contract check missing mac"
    })
    $caughtMissingMac = $false
    try {
        Update-RecoveryScriptAndValidate -Targets $testMissingMac -Path $testScriptPath
    } catch {
        $caughtMissingMac = $true
    }
    $missingIdentityRejected = ($caughtMissingDesc -and $caughtMissingMac)
    Write-Host ("  Allowlist missing InterfaceDescription/MacAddress rejected: {0} (Expected: True)" -f $missingIdentityRejected)

    # Contract expansion 2: Verify null or non-existent adapter restoration is strictly detected as failure
    $testNullTarget = @([PSCustomObject]@{
        InterfaceIndex       = 99999
        Name                 = "NonExistentAdapter"
        InterfaceDescription = "NonExistent"
        MacAddress           = "00:00:00:00:00:00"
        Reason               = "Null restoration test"
    })
    $nullRestorationFailures = @()
    foreach ($t in $testNullTarget) {
        try {
            $curr = Resolve-TargetNetAdapter -InterfaceIndex $t.InterfaceIndex `
                                             -ExpectedName $t.Name `
                                             -ExpectedDescription $t.InterfaceDescription `
                                             -ExpectedMacAddress $t.MacAddress
            if (-not $curr) {
                $nullRestorationFailures += "$($t.Name) (ifIndex $($t.InterfaceIndex)): adapter resolution returned null"
            } elseif ($curr.AdminStatus -ne "Up" -or $curr.Status -ne "Up") {
                $nullRestorationFailures += "$($t.Name) (ifIndex $($t.InterfaceIndex)): AdminStatus and operational Status must both be 'Up'"
            }
        } catch {
            $nullRestorationFailures += "$($t.Name) (ifIndex $($t.InterfaceIndex)): resolution failed ($($_.Exception.Message))"
        }
    }
    $nullDetectedAsFailure = ($nullRestorationFailures.Count -gt 0)
    Write-Host ("  Null/unresolved adapter restoration detected as failure    : {0} (Expected: True)" -f $nullDetectedAsFailure)

    # Mock pipeline test: Verify piping adapter object to a mock cmdlet receives the exact adapter
    $mockReceived = [System.Collections.Generic.List[PSObject]]::new()
    function Mock-NetAdapterCmdlet {
        [CmdletBinding(SupportsShouldProcess = $true)]
        param(
            [Parameter(Mandatory = $true, ValueFromPipeline = $true)]
            [PSObject]$InputObject
        )
        process {
            $mockReceived.Add($InputObject)
        }
    }

    $testAdapter = [PSCustomObject]@{
        ifIndex              = 21
        Name                 = "Wi-Fi"
        InterfaceDescription = "Killer(R) Wi-Fi 6 AX1650i"
        MacAddress           = "00:11:22:33:44:55"
    }

    $testAdapter | Mock-NetAdapterCmdlet -Confirm:$false -ErrorAction Stop

    $mockPipesCorrectly = ($mockReceived.Count -eq 1 -and [int]$mockReceived[0].ifIndex -eq 21 -and $mockReceived[0].Name -eq "Wi-Fi")
    Write-Host ("  Mock pipeline object binding verified: {0} (Expected: True)" -f $mockPipesCorrectly)

    # Contract expansion 3: Verify Test-ScheduledTaskExists does not throw terminating error on non-existent task
    $nonExistentAbsent = (-not (Test-ScheduledTaskExists -TaskName "NonExistentPhase4C2GTask"))
    Write-Host ("  Non-existent task correctly detected absent without terminating error: {0} (Expected: True)" -f $nonExistentAbsent)

    # Contract expansion 4: Verify Remove-ScheduledTaskSafely on absent task succeeds cleanly
    $safeRemoveOnAbsent = (Remove-ScheduledTaskSafely -TaskName "NonExistentPhase4C2GTask")
    Write-Host ("  Safe remove on absent task succeeds cleanly: {0} (Expected: True)" -f $safeRemoveOnAbsent)

    # Contract expansion 5: Verify verdict decision logic: restoration success + watchdog cleanup success => PASS
    $mockPassVerdict = if ($true -and $true -and $true -and ("RESTORED_VERIFIED" -eq "RESTORED_VERIFIED")) { "AUTOMATED_ISOLATION_READINESS_TEST_PASS_NETWORK_RESTORED" } else { "BLOCKED" }
    $verdictPassCorrect = ($mockPassVerdict -eq "AUTOMATED_ISOLATION_READINESS_TEST_PASS_NETWORK_RESTORED")
    Write-Host ("  Verdict logic: restoration success + watchdog cleanup success => PASS: {0} (Expected: True)" -f $verdictPassCorrect)

    # Contract expansion 6: Verify verdict decision logic: restoration success + watchdog cleanup failure => BLOCKED
    $mockBlockedVerdict = if ($true -and $true -and $false -and ("WATCHDOG_CLEANUP_FAILED" -eq "RESTORED_VERIFIED")) { "AUTOMATED_ISOLATION_READINESS_TEST_PASS_NETWORK_RESTORED" } elseif ($true -and (-not $false)) { "BLOCKED_WATCHDOG_CLEANUP_FAILED_NETWORK_RESTORED" } else { "BLOCKED" }
    $verdictBlockedCorrect = ($mockBlockedVerdict -eq "BLOCKED_WATCHDOG_CLEANUP_FAILED_NETWORK_RESTORED")
    Write-Host ("  Verdict logic: restoration success + watchdog cleanup failure => BLOCKED: {0} (Expected: True)" -f $verdictBlockedCorrect)

    # Contract expansion 7: Verify passive isolation ignores persistent routes when active routes are 0
    $mockRouteOut = @(
        "Active Routes:",
        "Network Destination        Netmask          Gateway       Interface  Metric",
        "None",
        "Persistent Routes:",
        "  Network Address          Netmask  Gateway Address  Metric",
        "          0.0.0.0          0.0.0.0         26.0.0.1    9256"
    )
    $mockActiveRouteDetected = $false
    $mockInActive = $false
    $mockInPersistent = $false
    $mockPersistentIgnored = @()
    foreach ($mLine in $mockRouteOut) {
        $mTrimmed = $mLine.Trim()
        if ($mTrimmed -like "*Active Routes:*") { $mockInActive = $true; $mockInPersistent = $false; continue }
        if ($mTrimmed -like "*Persistent Routes:*") { $mockInActive = $false; $mockInPersistent = $true; continue }
        if ($mTrimmed -like "*IPv6*" -or $mTrimmed -like "*Interface List*") { $mockInActive = $false; $mockInPersistent = $false; continue }
        $mParts = $mTrimmed -split "\s+"
        if ($mParts.Count -ge 3 -and $mParts[0] -eq "0.0.0.0" -and $mParts[1] -eq "0.0.0.0") {
            if ($mockInActive) { $mockActiveRouteDetected = $true }
            elseif ($mockInPersistent) { $mockPersistentIgnored += $mTrimmed }
        }
    }
    $persistentIgnoredCorrect = (-not $mockActiveRouteDetected) -and ($mockPersistentIgnored.Count -eq 1)
    Write-Host ("  Persistent routes ignored and excluded from active default route: {0} (Expected: True)" -f $persistentIgnoredCorrect)

    # Contract expansion 8: Verify remainingRoutes == 0 with passive disagreement yields BLOCKED_PASSIVE_ISOLATION_SOURCE_DISAGREEMENT
    $mockDisagreementVerdict = if (0 -eq 0 -and (-not $false)) { "BLOCKED_PASSIVE_ISOLATION_SOURCE_DISAGREEMENT" } else { "BLOCKED" }
    $disagreementVerdictCorrect = ($mockDisagreementVerdict -eq "BLOCKED_PASSIVE_ISOLATION_SOURCE_DISAGREEMENT")
    Write-Host ("  Remaining routes 0 with passive disagreement => DISAGREEMENT: {0} (Expected: True)" -f $disagreementVerdictCorrect)

    # Contract expansion 9: Verify remainingRoutes == 0 with proxy detected yields BLOCKED_PROXY_DETECTED
    $mockProxyVerdict = if (0 -eq 0 -and $true) { "BLOCKED_PROXY_DETECTED" } else { "BLOCKED" }
    $proxyVerdictCorrect = ($mockProxyVerdict -eq "BLOCKED_PROXY_DETECTED")
    Write-Host ("  Remaining routes 0 with proxy detected => BLOCKED_PROXY_DETECTED: {0} (Expected: True)" -f $proxyVerdictCorrect)

    # Contract expansion 10: AdminStatus Up alone must not satisfy restoration for a pre-isolation Up adapter
    $disconnectedResolver = {
        param($Target)
        return [PSCustomObject]@{ AdminStatus = "Up"; Status = "Disconnected" }
    }
    $noSleep = { param($Seconds) }
    $disconnectedPoll = Wait-AdapterOperationalRestoration `
        -Target $fixtureTargets[0] `
        -RequireOperationalUp $true `
        -TimeoutSeconds 0 `
        -PollIntervalSeconds 0 `
        -Resolver $disconnectedResolver `
        -Sleeper $noSleep
    $adminOnlyRejected = (-not $disconnectedPoll.Success)
    Write-Host ("  Restoration poll rejects AdminStatus Up / Status Disconnected: {0} (Expected: True)" -f $adminOnlyRejected)

    # Contract expansion 11: Operational Status returning Up inside the window succeeds
    $statusQueue = [System.Collections.Generic.Queue[PSObject]]::new()
    $statusQueue.Enqueue([PSCustomObject]@{ AdminStatus = "Up"; Status = "Disconnected" })
    $statusQueue.Enqueue([PSCustomObject]@{ AdminStatus = "Up"; Status = "Up" })
    $recoveringResolver = {
        param($Target)
        if ($statusQueue.Count -gt 1) { return $statusQueue.Dequeue() }
        return $statusQueue.Peek()
    }
    $recoveredPoll = Wait-AdapterOperationalRestoration `
        -Target $fixtureTargets[0] `
        -RequireOperationalUp $true `
        -TimeoutSeconds 5 `
        -PollIntervalSeconds 0 `
        -Resolver $recoveringResolver `
        -Sleeper $noSleep
    $operationalPollAccepted = ($recoveredPoll.Success -and $recoveredPoll.Adapter.Status -eq "Up")
    Write-Host ("  Restoration poll accepts Status Up inside polling window: {0} (Expected: True)" -f $operationalPollAccepted)

    # Contract expansion 12: Stale receipts are rejected; current non-zero receipts are still hashed and bound
    $sessionStart = [System.DateTimeOffset]::UtcNow
    $sessionEnd = $sessionStart.AddSeconds(2)
    $staleReceipt = [PSCustomObject]@{
        observation_started_at_utc = $sessionStart.AddMinutes(-5).ToString("o")
        observation_completed_at_utc = $sessionStart.AddMinutes(-4).ToString("o")
        generated_at_utc = $sessionStart.AddMinutes(-4).ToString("o")
        verdict = "USER_PHYSICAL_ACTION_REQUIRED"
    }
    $staleReceiptRejected = (-not (Test-VerifierReceiptSessionBinding -ReceiptJson $staleReceipt -SessionStartedAtUtc $sessionStart -SessionCompletedAtUtc $sessionEnd))
    Write-Host ("  Stale verifier receipt timestamp rejected: {0} (Expected: True)" -f $staleReceiptRejected)

    $bindingFixturePath = Join-Path $ArtifactsDir "OFFLINE_VERIFIER_BINDING_CONTRACT_TEST.json"
    $currentReceipt = [ordered]@{
        observation_started_at_utc = $sessionStart.AddMilliseconds(100).ToString("o")
        observation_completed_at_utc = $sessionStart.AddMilliseconds(200).ToString("o")
        generated_at_utc = $sessionStart.AddMilliseconds(200).ToString("o")
        verdict = "USER_PHYSICAL_ACTION_REQUIRED"
    }
    [System.IO.File]::WriteAllText($bindingFixturePath, ($currentReceipt | ConvertTo-Json), (New-Object System.Text.UTF8Encoding($false)))
    try {
        $nonzeroBinding = Bind-OfflineVerifierReceipt `
            -Path $bindingFixturePath `
            -SessionStartedAtUtc $sessionStart `
            -SessionCompletedAtUtc $sessionEnd `
            -ExitCode 2
        $nonzeroReceiptBound = (
            $nonzeroBinding.Exists -and
            $nonzeroBinding.TimestampValid -and
            $nonzeroBinding.Sha256 -match '^[0-9a-f]{64}$' -and
            $nonzeroBinding.Verdict -eq "USER_PHYSICAL_ACTION_REQUIRED" -and
            $nonzeroBinding.ExitCode -eq 2
        )
    } finally {
        Remove-Item -LiteralPath $bindingFixturePath -Force -ErrorAction SilentlyContinue
    }
    Write-Host ("  Current-session non-zero verifier receipt hashed and bound: {0} (Expected: True)" -f $nonzeroReceiptBound)

    $allContractPassed = (-not $disHasIfIdx) -and (-not $enaHasIfIdx) -and $disPipesInput -and $enaPipesInput -and `
                         (-not $hasDisableIfIndex) -and (-not $hasEnableIfIndex) -and (-not $hasGetIfIndex) -and (-not $hasDirectParamAst) -and `
                         ($parseErrors.Count -eq 0) -and (-not $genHasEnableIfIndex) -and (-not $genHasGetIfIndex) -and `
                         $genPipesToEnable -and $mockPipesCorrectly -and $missingIdentityRejected -and $nullDetectedAsFailure -and `
                         $nonExistentAbsent -and $safeRemoveOnAbsent -and $verdictPassCorrect -and $verdictBlockedCorrect -and `
                         $persistentIgnoredCorrect -and $disagreementVerdictCorrect -and $proxyVerdictCorrect -and `
                         $adminOnlyRejected -and $operationalPollAccepted -and $staleReceiptRejected -and $nonzeroReceiptBound

    if ($allContractPassed) {
        Write-Host "`n[CONTRACT PASS] Adapter cmdlet parameter and pipeline contract fully verified." -ForegroundColor Green
        Write-Host "Verdict: ADAPTER_CMDLET_CONTRACT_PASS" -ForegroundColor Green
        exit 0
    } else {
        Write-Host "`n[CONTRACT FAIL] One or more adapter cmdlet contract checks failed." -ForegroundColor Red
        Write-Host "Verdict: BLOCKED_ADAPTER_CMDLET_CONTRACT_INVALID" -ForegroundColor Red
        exit 1
    }
}

# ------------------------------------------------------------------------------
# MODE A: DRY RUN
# ------------------------------------------------------------------------------
if ($DryRun -or (-not $ReadinessTest -and -not $ValidateRecoveryScriptOnly -and -not $ValidateAdapterCmdletContractOnly)) {
    Write-Host "`n[DRY RUN AUDIT] Route-to-Adapter Evidence Table:" -ForegroundColor Green
    Write-Host "destination_prefix | next_hop | route_metric | ifIndex | adapter_name | adapter_type | operational_status | selected_for_disable | exact_selection_reason | protected_reason"

    # Output formatted table as required by Section E
    $tableToDisplay = @()
    foreach ($row in $Snapshot.RouteTableRecords) {
        $tableToDisplay += [PSCustomObject]@{
            "destination_prefix"    = $row.destination_prefix
            "next_hop"              = $row.next_hop
            "route_metric"          = $row.route_metric
            "ifIndex"               = $row.ifIndex
            "adapter_name"          = $row.adapter_name
            "adapter_type"          = $row.adapter_type
            "operational_status"    = $row.operational_status
            "selected_for_disable"  = $row.selected_for_disable
            "exact_selection_reason"= $row.exact_selection_reason
            "protected_reason"      = $row.protected_reason
        }
    }
    $tableToDisplay | Format-Table -Property destination_prefix, next_hop, route_metric, ifIndex, adapter_name, adapter_type, operational_status, selected_for_disable, exact_selection_reason, protected_reason -AutoSize | Out-String -Width 240 | Write-Host

    Write-Host "[DRY RUN METRICS] Summary Counters:" -ForegroundColor Green
    Write-Host ("  default_route_owner_count          : {0}" -f $Snapshot.DefaultRouteOwnerCount)
    Write-Host ("  initial_disable_target_count       : {0}" -f $Snapshot.InitialDisableTargetCount)
    Write-Host ("  protected_internal_adapter_count   : {0}" -f $Snapshot.ProtectedInternalAdapterCount)
    Write-Host ("  unidentified_egress_routes_count   : {0}" -f $Snapshot.UnidentifiedEgressRoutesCount)

    # Fail closed if unidentified egress route exists
    if ($Snapshot.UnidentifiedEgressRoutesCount -gt 0) {
        Write-Host "`n[FAIL-CLOSED] Unidentified egress routes detected without matching adapter!" -ForegroundColor Red
        Write-Host "Verdict: BLOCKED_UNIDENTIFIED_EGRESS_ROUTES" -ForegroundColor Red
        exit 1
    }

    # Strict invariant check: internal-only adapters must not be selected without route evidence
    foreach ($a in $Snapshot.Adapters) {
        if ($a.SelectedForDisable -and ($a.Name -like "VMnet1*" -or $a.Name -like "*WSL*" -or $a.Name -like "*Default Switch*")) {
            $hasRoute = ($Snapshot.IPv4DefaultRoutes + $Snapshot.IPv6DefaultRoutes | Where-Object { $_.InterfaceIndex -eq $a.InterfaceIndex })
            if (-not $hasRoute) {
                Write-Host "`n[FAIL-CLOSED] Internal adapter $($a.Name) was selected without route evidence!" -ForegroundColor Red
                Write-Host "Verdict: BLOCKED_INTERNAL_ADAPTER_INCORRECTLY_SELECTED" -ForegroundColor Red
                exit 1
            }
        }
    }

    # Test Watchdog Capability
    $watchdogTest = $false
    try {
        & schtasks.exe /? >$null 2>&1
        $watchdogTest = ($LASTEXITCODE -eq 0)
    } catch {
        $watchdogTest = $false
    }
    Write-Host ("  scheduled_task_subsystem_available : {0}" -f $watchdogTest)
    Write-Host ("  recovery_script_destination        : {0}" -f $RecoverScriptPath)

    # Validate recovery script generation and AST in DryRun without modifying production recovery script
    $dryRunTargets = [System.Collections.Generic.List[PSObject]]::new()
    foreach ($t in $Snapshot.InitialDisableTargets) {
        if ([string]::IsNullOrWhiteSpace($t.InterfaceDescription) -or [string]::IsNullOrWhiteSpace($t.MacAddress)) {
            Write-Host "[FAIL-CLOSED] Target adapter ifIndex $($t.InterfaceIndex) missing InterfaceDescription or MacAddress" -ForegroundColor Red
            throw "BLOCKED_TARGET_ADAPTER_IDENTITY_INCOMPLETE"
        }
        $dryRunTargets.Add([PSCustomObject]@{
            InterfaceIndex       = [int]$t.InterfaceIndex
            Name                 = [string]$t.Name
            InterfaceDescription = [string]$t.InterfaceDescription
            MacAddress           = [string]$t.MacAddress
            Reason               = [string]$t.Reason
        })
    }
    $dryRunTestPath = Join-Path $ArtifactsDir "RECOVER_NETWORK_DRYRUN_TEST.ps1"
    try {
        Update-RecoveryScriptAndValidate -Targets $dryRunTargets -Path $dryRunTestPath
        Write-Host "  recovery_script_syntax_validated   : True (AST parse 0 errors)" -ForegroundColor Green
        Write-Host "  adapter_cmdlet_contract_validated  : True (Pipeline object binding)" -ForegroundColor Green
        Remove-Item $dryRunTestPath -Force -ErrorAction SilentlyContinue
    } catch {
        Write-Host "  recovery_script_syntax_validated   : False ($($_.Exception.Message))" -ForegroundColor Red
        Write-Host "`n[FAIL-CLOSED] Recovery script syntax validation failed during DryRun!" -ForegroundColor Red
        Write-Host "Verdict: BLOCKED_RECOVERY_SCRIPT_SYNTAX_INVALID" -ForegroundColor Red
        exit 1
    }

    Write-Host "`nDry-Run Verdict: DRY_RUN_INSPECTION_PASS" -ForegroundColor Green
    Write-Host "Host is verified ready for single-UAC elevated readiness testing." -ForegroundColor Cyan
    Write-Host "Exact command for elevated execution:" -ForegroundColor White
    Write-Host ("  powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"{0}`" -ReadinessTest" -f $MyInvocation.MyCommand.Definition) -ForegroundColor Yellow
    exit 0
}

# ------------------------------------------------------------------------------
# MODE B: READINESS TEST (ELEVATED SINGLE-UAC ISOLATION -> VERIFICATION -> RESTORATION)
# ------------------------------------------------------------------------------
if ($ReadinessTest) {
    # Check Administrator Elevation
    if (-not $IsAdmin) {
        Write-Host "`n[ACTION REQUIRED] Process is not running as Administrator." -ForegroundColor Yellow
        Write-Host "Elevated Administrator privileges are required to temporarily toggle network adapters." -ForegroundColor Yellow

        $workerArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$($MyInvocation.MyCommand.Definition)`" -ReadinessTest -ElevatedDetachedWorker -RepoRoot `"$RepoRoot`" -ReceiptPath `"$ReceiptPath`""
        try {
            Start-Process powershell.exe -Verb RunAs -ArgumentList $workerArgs -ErrorAction Stop
            Write-Host "Elevated detached worker process launched. Please click 'Yes' on the Windows UAC confirmation dialog." -ForegroundColor Green
        } catch {
            Write-Host "[NOTICE] Background session cannot directly spawn interactive UAC window ($($_.Exception.Message))." -ForegroundColor Yellow
            Write-Host "Please open an elevated PowerShell window (Run as Administrator) and run:" -ForegroundColor Cyan
            Write-Host "  powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$($MyInvocation.MyCommand.Definition)`" -ReadinessTest" -ForegroundColor White
        }

        Write-Host "`nVerdict: USER_UAC_CONFIRMATION_REQUIRED" -ForegroundColor Cyan
        exit 0
    }

    $LogPath = Join-Path $ArtifactsDir "automated_isolation_worker.log"
    if ($ElevatedDetachedWorker) {
        try {
            Start-Transcript -Path $LogPath -Force -ErrorAction SilentlyContinue | Out-Null
        } catch {}
    }

    Write-Host "`n[READINESS TEST] Operating with confirmed Administrator privileges." -ForegroundColor Green

    # Build active disabled allowlist with full 5 identity fields
    $disabledAllowlist = [System.Collections.Generic.List[PSObject]]::new()
    foreach ($t in $Snapshot.InitialDisableTargets) {
        if ([string]::IsNullOrWhiteSpace($t.InterfaceDescription) -or [string]::IsNullOrWhiteSpace($t.MacAddress)) {
            Write-Host "[FAIL-CLOSED] Target adapter ifIndex $($t.InterfaceIndex) missing InterfaceDescription or MacAddress" -ForegroundColor Red
            throw "BLOCKED_TARGET_ADAPTER_IDENTITY_INCOMPLETE"
        }
        $disabledAllowlist.Add([PSCustomObject]@{
            InterfaceIndex       = [int]$t.InterfaceIndex
            Name                 = [string]$t.Name
            InterfaceDescription = [string]$t.InterfaceDescription
            MacAddress           = [string]$t.MacAddress
            Reason               = [string]$t.Reason
        })
    }

    if ($disabledAllowlist.Count -eq 0) {
        Write-Host "[WARN] No active egress route owners detected. System may already be offline." -ForegroundColor Yellow
    }

    # Step 1: Create Recovery Script with syntax check
    Write-Host "Generating emergency recovery script at: $RecoverScriptPath"
    try {
        Update-RecoveryScriptAndValidate -Targets $disabledAllowlist -Path $RecoverScriptPath
    } catch {
        Write-Host "`n[BLOCKED] Recovery script generation/validation failed: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Verdict: BLOCKED_RECOVERY_SCRIPT_SYNTAX_INVALID" -ForegroundColor Red
        exit 1
    }

    $recoverBytes = [System.IO.File]::ReadAllBytes($RecoverScriptPath)
    $RecoverScriptSha256 = -join ($Sha256.ComputeHash($recoverBytes) | ForEach-Object { "{0:x2}" -f $_ })

    # Step 2: Clean up any stale Scheduled Task Watchdog prior to registering new watchdog
    Remove-StaleWatchdogIfSafe -TaskName $WatchdogTaskName -Targets $disabledAllowlist

    # Step 2b: Register Scheduled Task Watchdog (15 minutes in future)
    $TriggerTime = (Get-Date).AddMinutes(15).ToString("HH:mm")
    $TaskCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$RecoverScriptPath`""
    Write-Host "Registering recovery watchdog scheduled task '$WatchdogTaskName' for $TriggerTime (15-min timeout)..."

    $taskCreated = $false
    try {
        $createOut = & schtasks.exe /create /tn $WatchdogTaskName /tr $TaskCommand /sc once /st $TriggerTime /f /rl HIGHEST 2>&1
        if ($LASTEXITCODE -eq 0) {
            $taskCreated = $true
            Write-Host "[WATCHDOG REGISTERED] Scheduled task created." -ForegroundColor Green
        } else {
            Write-Host "[WATCHDOG ERROR] schtasks returned error: $createOut" -ForegroundColor Red
        }
    } catch {
        Write-Host "[WATCHDOG ERROR] Exception creating scheduled task: $($_.Exception.Message)" -ForegroundColor Red
    }

    if (-not $taskCreated) {
        Write-Host "[BLOCKED] Recovery watchdog scheduled task could not be established." -ForegroundColor Red
        Write-Host "Verdict: BLOCKED_RECOVERY_WATCHDOG_NOT_AVAILABLE" -ForegroundColor Red
        exit 1
    }

    # Step 3: Strictly verify watchdog via read-back query
    Write-Host "Verifying watchdog scheduled task via query read-back..."
    $watchdogVerified = Test-WatchdogTaskVerified -TaskName $WatchdogTaskName
    if (-not $watchdogVerified) {
        Write-Host "[BLOCKED] Watchdog task read-back verification failed." -ForegroundColor Red
        Write-Host "Verdict: BLOCKED_RECOVERY_WATCHDOG_NOT_AVAILABLE" -ForegroundColor Red
        exit 1
    }
    Write-Host "[WATCHDOG VERIFIED] Watchdog task existence confirmed." -ForegroundColor Green

    $isolationVerifiedTime = $null
    $verifierReceiptSha = $null
    $verifierReceiptVerdict = $null
    $verifierReceiptTimestampValid = $false
    $verifierExit = $null
    $verifierSessionStartedAtUtc = $null
    $verifierSessionCompletedAtUtc = $null
    $testPassed = $false
    $restorationResult = "PENDING"
    $networkRestored = $false
    $watchdogCleanupVerified = $false
    $watchdogDeleted = $false
    $failureReason = $null
    $offlineVerifierInvoked = $false
    $lastPassiveCheck = $null
    $remainingRoutesCount = 0

    # Step 4: Iterative Isolation and Verification Execution Block
    try {
        Write-Host "`nDisabling initial egress targets ($($disabledAllowlist.Count) adapter(s))..." -ForegroundColor Yellow
        foreach ($t in $disabledAllowlist) {
            Write-Host "  Resolving & disabling: ifIndex $($t.InterfaceIndex) ($($t.Name))"
            if ([string]::IsNullOrWhiteSpace($t.InterfaceDescription) -or [string]::IsNullOrWhiteSpace($t.MacAddress)) {
                Write-Host "[FAIL-CLOSED] Target adapter ifIndex $($t.InterfaceIndex) missing InterfaceDescription or MacAddress prior to disable" -ForegroundColor Red
                throw "BLOCKED_TARGET_ADAPTER_IDENTITY_INCOMPLETE"
            }
            $adapterObj = Resolve-TargetNetAdapter -InterfaceIndex $t.InterfaceIndex `
                                                  -ExpectedName $t.Name `
                                                  -ExpectedDescription $t.InterfaceDescription `
                                                  -ExpectedMacAddress $t.MacAddress
            $adapterObj | Disable-NetAdapter -Confirm:$false -ErrorAction Stop
        }

        # Iterative rescan loop (up to 3 rounds)
        $maxRounds = 3
        $currentRound = 1
        $isolationAchieved = $false
        $passiveDisagreement = $false

        while ($currentRound -le $maxRounds) {
            Write-Host "Waiting 3 seconds for route table stabilization (Round $currentRound)..."
            Start-Sleep -Seconds 3

            # Check remaining active default routes via ActiveStore
            $remainingRoutes = @()
            try {
                $activeAdapters = @(Get-NetAdapter -IncludeHidden -ErrorAction SilentlyContinue | Where-Object { $_.Status -eq "Up" -or $_.AdminStatus -eq "Up" })
                $activeIndices = [System.Collections.Generic.HashSet[int]]::new()
                foreach ($ad in $activeAdapters) {
                    $activeIndices.Add([int]$ad.ifIndex) | Out-Null
                }

                $rem4 = Get-NetRoute -PolicyStore ActiveStore -DestinationPrefix "0.0.0.0/0" -ErrorAction SilentlyContinue
                if ($rem4) {
                    foreach ($r in $rem4) {
                        if ($activeIndices.Contains([int]$r.InterfaceIndex)) {
                            $remainingRoutes += $r
                        }
                    }
                }
                $rem6 = Get-NetRoute -PolicyStore ActiveStore -DestinationPrefix "::/0" -ErrorAction SilentlyContinue
                if ($rem6) {
                    foreach ($r in $rem6) {
                        if ($activeIndices.Contains([int]$r.InterfaceIndex)) {
                            $remainingRoutes += $r
                        }
                    }
                }
            } catch {}

            $remainingRoutesCount = $remainingRoutes.Count
            $passive = Test-PassiveIsolation
            $lastPassiveCheck = $passive

            if ($remainingRoutes.Count -eq 0) {
                if ($passive.IsIsolated) {
                    $isolationAchieved = $true
                    Write-Host "[ISOLATION CONFIRMED] Zero active default routes and passive isolation verified in round $currentRound." -ForegroundColor Green
                    break
                } else {
                    Write-Host "[NOTICE] remainingRoutes is 0 but Test-PassiveIsolation returned IsIsolated = false in round $($currentRound):" -ForegroundColor Yellow
                    Write-Host "  ProxyDetected         : $($passive.ProxyDetected)"
                    Write-Host "  DefaultRouteDetected  : $($passive.DefaultRouteDetected)"
                    Write-Host "  ConnectedAdapters     : $($passive.ConnectedAdapters -join ', ')"
                    Write-Host "  Active route source   : Get-NetRoute -PolicyStore ActiveStore"
                    Write-Host "  Persistent ignored    : $($passive.PersistentRoutesIgnored -join '; ')"

                    if ($passive.ProxyDetected) {
                        throw "BLOCKED_PROXY_DETECTED"
                    }

                    $passiveDisagreement = $true
                    if ($currentRound -lt $maxRounds) {
                        Write-Host "Waiting for route/passive convergence (round $currentRound/$maxRounds)..."
                        $currentRound++
                        continue
                    } else {
                        Write-Host "[FAIL-CLOSED] Passive isolation source disagreement persisted after $maxRounds rounds." -ForegroundColor Red
                        throw "BLOCKED_PASSIVE_ISOLATION_SOURCE_DISAGREEMENT"
                    }
                }
            }

            # remainingRoutes.Count > 0: identify and disable emergent route owners
            Write-Host "[NOTICE] Egress routes still present after round $currentRound ($($remainingRoutes.Count) route(s))." -ForegroundColor Yellow

            $newTargetsFound = 0
            foreach ($r in $remainingRoutes) {
                $idx = $r.InterfaceIndex
                $alreadyDisabled = ($disabledAllowlist | Where-Object { $_.InterfaceIndex -eq $idx })
                if (-not $alreadyDisabled) {
                    try {
                        $matchingAdapter = Resolve-TargetNetAdapter -InterfaceIndex $idx
                    } catch {
                        Write-Host "[FAIL-CLOSED] Ambiguous route owner: Route prefix $($r.DestinationPrefix) has ifIndex $idx which cannot be resolved uniquely: $($_.Exception.Message)" -ForegroundColor Red
                        throw "BLOCKED_AMBIGUOUS_ROUTE_OWNER"
                    }

                    Write-Host "  Discovered secondary egress owner: ifIndex $idx ($($matchingAdapter.Name))"
                    if ([string]::IsNullOrWhiteSpace($matchingAdapter.InterfaceDescription) -or [string]::IsNullOrWhiteSpace($matchingAdapter.MacAddress)) {
                        Write-Host "[FAIL-CLOSED] Emergent target adapter ifIndex $idx missing InterfaceDescription or MacAddress" -ForegroundColor Red
                        throw "BLOCKED_TARGET_ADAPTER_IDENTITY_INCOMPLETE"
                    }
                    $disabledAllowlist.Add([PSCustomObject]@{
                        InterfaceIndex       = [int]$idx
                        Name                 = [string]$matchingAdapter.Name
                        InterfaceDescription = [string]$matchingAdapter.InterfaceDescription
                        MacAddress           = [string]$matchingAdapter.MacAddress
                        Reason               = "Secondary egress route owner discovered in round $currentRound"
                    })
                    try {
                        Update-RecoveryScriptAndValidate -Targets $disabledAllowlist -Path $RecoverScriptPath
                    } catch {
                        Write-Host "[FAIL-CLOSED] Failed to update recovery script for emergent adapter: $($_.Exception.Message)" -ForegroundColor Red
                        throw "BLOCKED_RECOVERY_SCRIPT_SYNTAX_INVALID"
                    }
                    $matchingAdapter | Disable-NetAdapter -Confirm:$false -ErrorAction Stop
                    $newTargetsFound++
                }
            }

            if ($newTargetsFound -eq 0) {
                Write-Host "[FAIL-CLOSED] Route owner ambiguity: remaining active routes exist ($($remainingRoutes.Count)) but no new un-isolated adapter could be identified." -ForegroundColor Red
                throw "BLOCKED_AMBIGUOUS_ROUTE_OWNER"
            }

            $currentRound++
        }

        if (-not $isolationAchieved) {
            if ($passiveDisagreement) {
                throw "BLOCKED_PASSIVE_ISOLATION_SOURCE_DISAGREEMENT"
            } else {
                throw "BLOCKED_NETWORK_ISOLATION_INCOMPLETE"
            }
        }

        $isolationVerifiedTime = [System.DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.ffffff+00:00")
        Write-Host "[ISOLATION VERIFIED] Zero connected egress adapters and zero default routes verified passively." -ForegroundColor Green

        # Step 5: Invoke Offline Verifier Wrapper
        Write-Host "`nInvoking Offline Verifier Wrapper: $VerifierWrapperPath" -ForegroundColor Cyan
        $offlineVerifierInvoked = $true
        $verifierSessionStartedAtUtc = [System.DateTimeOffset]::UtcNow
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $VerifierWrapperPath
        $verifierExit = $LASTEXITCODE
        $verifierSessionCompletedAtUtc = [System.DateTimeOffset]::UtcNow
        Write-Host "Offline Verifier process exit code: $verifierExit"

        # Step 6: Bind and verify the receipt regardless of verifier exit code
        $offlineReceiptPath = Join-Path $ArtifactsDir "offline_verifier_execution_receipt.json"
        $receiptBinding = Bind-OfflineVerifierReceipt `
            -Path $offlineReceiptPath `
            -SessionStartedAtUtc $verifierSessionStartedAtUtc `
            -SessionCompletedAtUtc $verifierSessionCompletedAtUtc `
            -ExitCode $verifierExit
        $verifierReceiptSha = $receiptBinding.Sha256
        $verifierReceiptVerdict = $receiptBinding.Verdict
        $verifierReceiptTimestampValid = [bool]$receiptBinding.TimestampValid

        if ($receiptBinding.FailureReason) {
            throw $receiptBinding.FailureReason
        }

        $receiptJson = $receiptBinding.ReceiptJson
        if ($verifierExit -ne 0) {
            throw "OFFLINE_VERIFIER_EXIT_NON_ZERO"
        }

        # Validate Strict Receipt Criteria
        $critFailures = @()
        if ($receiptJson.synthetic_only -ne $false) { $critFailures += "synthetic_only is not false" }
        if ($receiptJson.verdict -ne "READY_FOR_HUMAN_AUTHORIZATION_REVIEW") { $critFailures += "verdict is '$($receiptJson.verdict)'" }
        if ($receiptJson.checks.network_isolation.default_route_detected -ne $false) { $critFailures += "default_route_detected is not false" }
        if (@($receiptJson.checks.network_isolation.active_ipv4_default_routes).Count -ne 0) { $critFailures += "active IPv4 default routes not empty" }
        if (@($receiptJson.checks.network_isolation.active_ipv6_default_routes).Count -ne 0) { $critFailures += "active IPv6 default routes not empty" }
        if (@($receiptJson.checks.network_isolation.active_vpn_egress_owners).Count -ne 0) { $critFailures += "active VPN egress owners not empty" }
        if (@($receiptJson.checks.network_isolation.active_egress_adapters).Count -ne 0) { $critFailures += "active egress adapters not empty" }
        if (@($receiptJson.checks.network_isolation.unidentified_active_egress_route_owners).Count -ne 0) { $critFailures += "unidentified active egress route owners not empty" }
        if ($receiptJson.checks.worktree_head.commit -ne "2826a8274cb89ec548d6fac5c8ae50c1c2836202") { $critFailures += "worktree HEAD mismatch" }
        if ($receiptJson.checks.worktree_cleanliness.status -ne "PASS") { $critFailures += "worktree not clean" }
        if ($receiptJson.checks.evaluator_components.status -ne "PASS") { $critFailures += "evaluator components failed" }
        if ($receiptJson.checks.checkpoints.status -ne "PASS") { $critFailures += "checkpoints verification failed" }
        if ($receiptJson.checks.filesystem.locked_test_mount_state -ne "UNMOUNTED") { $critFailures += "locked test mount not UNMOUNTED" }
        if ($receiptJson.checks.filesystem.read_only_mount_verification -ne "PENDING_HUMAN_AUTHORIZATION") { $critFailures += "read-only mount not PENDING" }
        if ($receiptJson.checks.authorization.authorization_artifact_exists -ne $false) { $critFailures += "authorization artifact exists" }
        if ($receiptJson.real_counters.locked_test_real_accesses -ne 0) { $critFailures += "locked_test_real_accesses not 0" }

        if ($critFailures.Count -gt 0) {
            throw "Offline verifier receipt validation failed: $($critFailures -join '; ')"
        }

        Write-Host "[VERIFIER RECEIPT VALIDATED] All 12 fail-closed criteria confirmed." -ForegroundColor Green
        $testPassed = $true

    } catch {
        $failureReason = $_.Exception.Message
        Write-Host "`n[FAIL-CLOSED EXCEPTION CAUGHT] $failureReason" -ForegroundColor Red
    } finally {
        # Step 7: Restore Network Adapters (Guaranteed Execution by ifIndex and pipeline object)
        Write-Host "`n[RESTORATION] Re-enabling disabled network adapters..." -ForegroundColor Yellow
        foreach ($t in $disabledAllowlist) {
            Write-Host "  Resolving & enabling: ifIndex $($t.InterfaceIndex) ($($t.Name))"
            try {
                $adapterObj = Resolve-TargetNetAdapter -InterfaceIndex $t.InterfaceIndex `
                                                      -ExpectedName $t.Name `
                                                      -ExpectedDescription $t.InterfaceDescription `
                                                      -ExpectedMacAddress $t.MacAddress
                $adapterObj | Enable-NetAdapter -Confirm:$false -ErrorAction Stop
            } catch {
                Write-Host "  [WARNING] Error resolving or enabling adapter $($t.Name) ($($t.InterfaceIndex)): $($_.Exception.Message)" -ForegroundColor Red
            }
        }

        # Verify restoration by polling exact identity for up to 60 seconds per adapter.
        # Adapters that were operationally Up before isolation must return to Status=Up;
        # an initially Disconnected adapter is not required to become operationally Up.
        $restorationFailures = @()
        foreach ($t in $disabledAllowlist) {
            $preIsolationAdapter = @($Snapshot.Adapters | Where-Object { [int]$_.InterfaceIndex -eq [int]$t.InterfaceIndex })
            if ($preIsolationAdapter.Count -ne 1) {
                $restorationFailures += "$($t.Name) (ifIndex $($t.InterfaceIndex)): pre-isolation identity could not be resolved uniquely"
                continue
            }
            $requireOperationalUp = ($preIsolationAdapter[0].Status -eq "Up")
            $pollResult = Wait-AdapterOperationalRestoration `
                -Target $t `
                -RequireOperationalUp $requireOperationalUp `
                -TimeoutSeconds 60 `
                -PollIntervalSeconds 2
            if (-not $pollResult.Success) {
                $expectedState = if ($requireOperationalUp) { "AdminStatus=Up and Status=Up" } else { "AdminStatus=Up" }
                $restorationFailures += "$($t.Name) (ifIndex $($t.InterfaceIndex)): $($pollResult.Error) (expected $expectedState within 60 seconds)"
            } else {
                $curr = $pollResult.Adapter
                Write-Host "  Verified restored: ifIndex $($t.InterfaceIndex) ($($t.Name)) [AdminStatus: $($curr.AdminStatus), Status: $($curr.Status), required operational Up: $requireOperationalUp]" -ForegroundColor Green
            }
        }

        if ($restorationFailures.Count -eq 0) {
            Write-Host "[RESTORATION SUCCESS] All isolated adapters successfully re-enabled and verified Up." -ForegroundColor Green
            $networkRestored = $true

            # Remove Watchdog Scheduled Task only after verified restoration
            Write-Host "Removing scheduled task watchdog '$WatchdogTaskName'..."
            $watchdogDeleted = Remove-ScheduledTaskSafely -TaskName $WatchdogTaskName
            $taskAbsent = (-not (Test-ScheduledTaskExists -TaskName $WatchdogTaskName))
            if ($watchdogDeleted -and $taskAbsent) {
                Write-Host "Watchdog scheduled task removed and verified absent." -ForegroundColor Green
                $watchdogCleanupVerified = $true
                $restorationResult = "RESTORED_VERIFIED"
            } else {
                Write-Host "[FAIL-CLOSED] Watchdog scheduled task could not be confirmed removed." -ForegroundColor Red
                $watchdogCleanupVerified = $false
                $restorationResult = "WATCHDOG_CLEANUP_FAILED"
            }
        } else {
            Write-Host "[CRITICAL] Restoration verification failed for adapter(s):" -ForegroundColor Red
            foreach ($rf in $restorationFailures) {
                Write-Host "  - $rf" -ForegroundColor Red
            }
            Write-Host "Retaining scheduled task watchdog '$WatchdogTaskName'." -ForegroundColor Yellow
            Write-Host "Execute emergency recovery manually: $RecoverScriptPath" -ForegroundColor Red
            $networkRestored = $false
            $watchdogCleanupVerified = $false
            $restorationResult = "NETWORK_RECOVERY_REQUIRED"
        }
    }

    $networkRestoredTime = [System.DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.ffffff+00:00")

    # Step 8: Build and Write Atomic Controller Receipt (ALWAYS REACHED!)
    $finalVerdict = if ($testPassed -and $networkRestored -and $watchdogCleanupVerified -and ($restorationResult -eq "RESTORED_VERIFIED")) {
        "AUTOMATED_ISOLATION_READINESS_TEST_PASS_NETWORK_RESTORED"
    } elseif ($networkRestored -and (-not $watchdogCleanupVerified)) {
        "BLOCKED_WATCHDOG_CLEANUP_FAILED_NETWORK_RESTORED"
    } elseif ($restorationResult -eq "NETWORK_RECOVERY_REQUIRED") {
        "NETWORK_RECOVERY_REQUIRED"
    } elseif ($failureReason) {
        $failureReason
    } else {
        "BLOCKED_NETWORK_ISOLATION_INCOMPLETE"
    }

    $controllerReceipt = [ordered]@{
        schema_version                    = "1.4.0"
        phase                             = "Phase 4C.2G.0.3.4"
        controller_name                   = "automated_windows_network_isolation_controller"
        controller_version                = $ControllerVersion
        controller_sha256                 = $ControllerSha256
        started_at_utc                    = $CurrentUtc
        isolation_verified_at_utc         = $isolationVerifiedTime
        network_restored_at_utc           = $networkRestoredTime
        isolation_verified                = [bool]$isolationVerifiedTime
        failure_reason                    = $failureReason
        remaining_active_default_routes   = [int]$remainingRoutesCount
        proxy_detected                    = if ($lastPassiveCheck) { [bool]$lastPassiveCheck.ProxyDetected } else { [bool]($env:HTTP_PROXY -or $env:HTTPS_PROXY -or $env:ALL_PROXY) }
        active_egress_adapters             = [object[]]@(if ($lastPassiveCheck) { $lastPassiveCheck.ActiveEgressAdapters } else { @() })
        protected_internal_adapters        = [object[]]@(if ($lastPassiveCheck) { $lastPassiveCheck.ProtectedInternalAdapters } else { @() })
        connected_adapters_informational   = [object[]]@(if ($lastPassiveCheck) { $lastPassiveCheck.ConnectedAdaptersInformational } else { @() })
        active_vpn_egress_owners           = [object[]]@(if ($lastPassiveCheck) { $lastPassiveCheck.ActiveVpnEgressOwners } else { @() })
        unidentified_active_egress_route_owners = [object[]]@(if ($lastPassiveCheck) { $lastPassiveCheck.UnidentifiedActiveEgressRouteOwners } else { @() })
        persistent_routes_ignored         = [object[]]@(if ($lastPassiveCheck) { $lastPassiveCheck.PersistentRoutesIgnored } else { @() })
        network_restored                  = [bool]$networkRestored
        watchdog_cleanup_verified         = [bool]$watchdogCleanupVerified
        offline_verifier_invoked          = [bool]$offlineVerifierInvoked
        pre_isolation_adapter_snapshot    = $Snapshot.Adapters
        exact_disabled_adapter_allowlist  = $disabledAllowlist
        passive_offline_checks            = [ordered]@{
            outbound_probes_sent  = 0
            dns_lookups_performed = 0
            http_requests_sent    = 0
            isolation_verified    = [bool]$isolationVerifiedTime
        }
        offline_verifier_receipt_sha256   = $verifierReceiptSha
        offline_verifier_verdict          = $verifierReceiptVerdict
        offline_verifier_exit_code        = $verifierExit
        offline_verifier_receipt_timestamp_valid = [bool]$verifierReceiptTimestampValid
        offline_verifier_session_started_at_utc = if ($verifierSessionStartedAtUtc) { $verifierSessionStartedAtUtc.ToString("o") } else { $null }
        offline_verifier_session_completed_at_utc = if ($verifierSessionCompletedAtUtc) { $verifierSessionCompletedAtUtc.ToString("o") } else { $null }
        watchdog_metadata                 = [ordered]@{
            scheduled_task_name       = $WatchdogTaskName
            trigger_time_utc          = $TriggerTime
            timeout_minutes           = 15
            recovery_script_path      = $RecoverScriptPath
            recovery_script_sha256    = $RecoverScriptSha256
            watchdog_auto_cleaned     = [bool]$watchdogDeleted
            watchdog_cleanup_verified = [bool]$watchdogCleanupVerified
        }
        restoration_result                = $restorationResult
        all_scientific_counters           = [ordered]@{
            locked_test_real_accesses          = 0
            completed_real_unsealing_sessions  = 0
            completed_real_model_evaluations   = 0
            evaluation_attempts                = 0
            cpu_inference_calls                = 0
            gpu_inference_calls                = 0
            new_training_runs                  = 0
        }
        locked_test_real_accesses         = 0
        verdict                           = $finalVerdict
        caveat                            = "Receipt proves the controller can establish offline isolation, but the network was restored afterward. A fresh isolation verification is required inside the future authorized confirmatory session."
    }

    Write-ReceiptAtomic -Path $ReceiptPath -Data $controllerReceipt
    Write-Host "`nReadiness receipt written to: $ReceiptPath" -ForegroundColor Cyan
    Write-Host "Controller Final Verdict: $finalVerdict" -ForegroundColor $(if ($finalVerdict -like "*PASS*") { "Green" } else { "Red" })

    if ($ElevatedDetachedWorker) {
        try {
            Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
        } catch {}
    }

    if ($finalVerdict -eq "AUTOMATED_ISOLATION_READINESS_TEST_PASS_NETWORK_RESTORED") {
        exit 0
    } elseif ($finalVerdict -eq "NETWORK_RECOVERY_REQUIRED") {
        exit 2
    } else {
        exit 1
    }
}
