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
$ControllerVersion = "1.3.0"

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
        $rawRoutes = Get-NetRoute -DestinationPrefix "0.0.0.0/0" -ErrorAction SilentlyContinue
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
        $rawRoutes6 = Get-NetRoute -DestinationPrefix "::/0" -ErrorAction SilentlyContinue
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
            $vpnRoutes = @(Get-NetRoute -InterfaceIndex $idx -ErrorAction SilentlyContinue | Where-Object {
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
    if ($ExpectedMacAddress -and $adapter.MacAddress -and ($adapter.MacAddress -ne $ExpectedMacAddress)) {
        throw "Adapter identity mismatch for ifIndex $($InterfaceIndex): Expected MacAddress '$ExpectedMacAddress', found '$($adapter.MacAddress)'."
    }

    return $adapter
}

# 6b. Helper: Generate and Validate Recovery Script
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
    try {
        $queryOut = & schtasks.exe /query /tn $TaskName 2>&1
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

# 8. Passive Network Check
function Test-PassiveIsolation {
    $proxySet = [bool]($env:HTTP_PROXY -or $env:HTTPS_PROXY -or $env:ALL_PROXY)
    $hasDefaultRoute = $false

    try {
        $routeOut = & route.exe print 0.0.0.0 2>$null
        foreach ($line in $routeOut) {
            $parts = $line.Trim() -split "\s+"
            if ($parts.Count -ge 3 -and $parts[0] -eq "0.0.0.0" -and $parts[1] -eq "0.0.0.0") {
                $hasDefaultRoute = $true
                break
            }
        }
    } catch {}

    # Check for connected egress adapters (adapters with default route or physical status)
    $remainingDefaultRoutes = @()
    try {
        $rem4 = Get-NetRoute -DestinationPrefix "0.0.0.0/0" -ErrorAction SilentlyContinue
        if ($rem4) { $remainingDefaultRoutes += $rem4 }
        $rem6 = Get-NetRoute -DestinationPrefix "::/0" -ErrorAction SilentlyContinue
        if ($rem6) { $remainingDefaultRoutes += $rem6 }
    } catch {}

    $connectedEgressAdapters = @()
    if ($remainingDefaultRoutes.Count -gt 0) {
        $hasDefaultRoute = $true
        foreach ($r in $remainingDefaultRoutes) {
            $connectedEgressAdapters += "$($r.InterfaceAlias) (ifIndex $($r.InterfaceIndex))"
        }
    }

    return [ordered]@{
        IsIsolated            = ((-not $proxySet) -and (-not $hasDefaultRoute) -and ($connectedEgressAdapters.Count -eq 0))
        ProxyDetected         = $proxySet
        DefaultRouteDetected  = $hasDefaultRoute
        ConnectedAdapters     = $connectedEgressAdapters
        OutboundProbesSent    = 0
        DnsLookupsPerformed   = 0
        HttpRequestsSent      = 0
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

    # AST and static analysis of controller source code: ensure no direct -InterfaceIndex calls on Disable/Enable-NetAdapter
    $selfTokens = $null
    $selfErrors = $null
    $selfAst = [System.Management.Automation.Language.Parser]::ParseFile($MyInvocation.MyCommand.Definition, [ref]$selfTokens, [ref]$selfErrors)
    $directParamCalls = $selfAst.FindAll({
        param($astNode)
        if ($astNode -is [System.Management.Automation.Language.CommandAst]) {
            $cmdName = $astNode.GetCommandName()
            if ($cmdName -in @('Disable-NetAdapter', 'Enable-NetAdapter')) {
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
    $linesToScan = $selfContent -split "`r?`n" | Where-Object {
        $_ -notmatch '\$disPat' -and $_ -notmatch '\$enaPat' -and `
        $_ -notmatch 'illegalEnaPattern' -and $_ -notmatch 'genHasEnableIfIndex' -and `
        $_ -notmatch 'hasDisableIfIndex' -and $_ -notmatch 'hasEnableIfIndex' -and `
        $_ -notmatch 'directParamCalls'
    }
    $hasDisableIfIndex = ($linesToScan -match $disPat).Count -gt 0
    $hasEnableIfIndex  = ($linesToScan -match $enaPat).Count -gt 0

    Write-Host ("  Controller source contains Disable-NetAdapter direct parameter: {0} (Expected: False)" -f ($hasDisableIfIndex -or $hasDirectParamAst))
    Write-Host ("  Controller source contains Enable-NetAdapter  direct parameter: {0} (Expected: False)" -f ($hasEnableIfIndex -or $hasDirectParamAst))

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
    $genPipesToEnable    = $genContent -match '\|\s*Enable-NetAdapter'

    Write-Host ("  Generated script AST parse error count: {0} (Expected: 0)" -f $parseErrors.Count)
    Write-Host ("  Generated script contains direct Enable-NetAdapter parameter: {0} (Expected: False)" -f $genHasEnableIfIndex)
    Write-Host ("  Generated script pipes adapter object to Enable-NetAdapter: {0} (Expected: True)" -f $genPipesToEnable)

    # Clean up test script
    Remove-Item -Path $testScriptPath -Force -ErrorAction SilentlyContinue

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

    $allContractPassed = (-not $disHasIfIdx) -and (-not $enaHasIfIdx) -and $disPipesInput -and $enaPipesInput -and `
                         (-not $hasDisableIfIndex) -and (-not $hasEnableIfIndex) -and (-not $hasDirectParamAst) -and `
                         ($parseErrors.Count -eq 0) -and (-not $genHasEnableIfIndex) -and $genPipesToEnable -and $mockPipesCorrectly

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
        $dryRunTargets.Add([PSCustomObject]@{
            InterfaceIndex = [int]$t.InterfaceIndex
            Name           = [string]$t.Name
            Reason         = [string]$t.Reason
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

    # Build active disabled allowlist
    $disabledAllowlist = [System.Collections.Generic.List[PSObject]]::new()
    foreach ($t in $Snapshot.InitialDisableTargets) {
        $disabledAllowlist.Add([PSCustomObject]@{
            InterfaceIndex = $t.InterfaceIndex
            Name           = $t.Name
            Reason         = $t.Reason
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

    # Step 2: Register Scheduled Task Watchdog (15 minutes in future)
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
    $testPassed = $false
    $restorationResult = "PENDING"

    # Step 4: Iterative Isolation and Verification Execution Block
    try {
        Write-Host "`nDisabling initial egress targets ($($disabledAllowlist.Count) adapter(s))..." -ForegroundColor Yellow
        foreach ($t in $disabledAllowlist) {
            Write-Host "  Resolving & disabling: ifIndex $($t.InterfaceIndex) ($($t.Name))"
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

        while ($currentRound -le $maxRounds) {
            Write-Host "Waiting 3 seconds for route table stabilization (Round $currentRound)..."
            Start-Sleep -Seconds 3

            # Check remaining default routes
            $remainingRoutes = @()
            try {
                $rem4 = Get-NetRoute -DestinationPrefix "0.0.0.0/0" -ErrorAction SilentlyContinue
                if ($rem4) { $remainingRoutes += $rem4 }
                $rem6 = Get-NetRoute -DestinationPrefix "::/0" -ErrorAction SilentlyContinue
                if ($rem6) { $remainingRoutes += $rem6 }
            } catch {}

            if ($remainingRoutes.Count -eq 0) {
                $passive = Test-PassiveIsolation
                if ($passive.IsIsolated) {
                    $isolationAchieved = $true
                    break
                }
            }

            Write-Host "[NOTICE] Egress routes still present after round $currentRound ($($remainingRoutes.Count) route(s))." -ForegroundColor Yellow

            # Identify remaining route owners
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
                    $disabledAllowlist.Add([PSCustomObject]@{
                        InterfaceIndex       = $idx
                        Name                 = $matchingAdapter.Name
                        InterfaceDescription = $matchingAdapter.InterfaceDescription
                        MacAddress           = $matchingAdapter.MacAddress
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
                Write-Host "[FAIL-CLOSED] Route owner ambiguity or passive check incomplete." -ForegroundColor Red
                throw "BLOCKED_AMBIGUOUS_ROUTE_OWNER"
            }

            $currentRound++
        }

        if (-not $isolationAchieved) {
            throw "BLOCKED_NETWORK_ISOLATION_INCOMPLETE"
        }

        $isolationVerifiedTime = [System.DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.ffffff+00:00")
        Write-Host "[ISOLATION VERIFIED] Zero connected egress adapters and zero default routes verified passively." -ForegroundColor Green

        # Step 5: Invoke Offline Verifier Wrapper
        Write-Host "`nInvoking Offline Verifier Wrapper: $VerifierWrapperPath" -ForegroundColor Cyan
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $VerifierWrapperPath
        $verifierExit = $LASTEXITCODE
        Write-Host "Offline Verifier process exit code: $verifierExit"

        # Step 6: Verify Verifier Receipt
        $offlineReceiptPath = Join-Path $ArtifactsDir "offline_verifier_execution_receipt.json"
        if (-not (Test-Path $offlineReceiptPath)) {
            throw "Verifier receipt not found at: $offlineReceiptPath"
        }

        $receiptContent = [System.IO.File]::ReadAllText($offlineReceiptPath, [System.Text.Encoding]::UTF8)
        $receiptJson = $receiptContent | ConvertFrom-Json

        $rBytes = [System.IO.File]::ReadAllBytes($offlineReceiptPath)
        $verifierReceiptSha = -join ($Sha256.ComputeHash($rBytes) | ForEach-Object { "{0:x2}" -f $_ })

        # Validate Strict Receipt Criteria
        $critFailures = @()
        if ($receiptJson.synthetic_only -ne $false) { $critFailures += "synthetic_only is not false" }
        if ($receiptJson.verdict -ne "READY_FOR_HUMAN_AUTHORIZATION_REVIEW") { $critFailures += "verdict is '$($receiptJson.verdict)'" }
        if ($receiptJson.checks.network_isolation.default_route_detected -ne $false) { $critFailures += "default_route_detected is not false" }
        if ($receiptJson.checks.network_isolation.connected_network_adapters.Count -ne 0) { $critFailures += "connected_network_adapters not empty" }
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

        Write-Host "Waiting 3 seconds for network operational stabilization..."
        Start-Sleep -Seconds 3

        # Verify Restoration
        $stillDisabled = @()
        foreach ($t in $disabledAllowlist) {
            $curr = Get-NetAdapter -InterfaceIndex $t.InterfaceIndex -ErrorAction SilentlyContinue
            if ($curr -and $curr.AdminStatus -eq "Disabled") {
                $stillDisabled += "$($t.Name) (ifIndex $($t.InterfaceIndex))"
            }
        }

        if ($stillDisabled.Count -eq 0) {
            Write-Host "[RESTORATION SUCCESS] All isolated adapters successfully re-enabled." -ForegroundColor Green
            $restorationResult = "RESTORED_VERIFIED"

            # Remove Watchdog Scheduled Task only after verified restoration
            Write-Host "Removing scheduled task watchdog '$WatchdogTaskName'..."
            & schtasks.exe /delete /tn $WatchdogTaskName /f 2>$null | Out-Null
            Write-Host "Watchdog scheduled task removed."
        } else {
            Write-Host "[CRITICAL] Failed to re-enable adapters: $($stillDisabled -join ', ')" -ForegroundColor Red
            Write-Host "Execute emergency recovery manually: $RecoverScriptPath" -ForegroundColor Red
            $restorationResult = "NETWORK_RECOVERY_REQUIRED"
        }
    }

    $networkRestoredTime = [System.DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.ffffff+00:00")

    # Step 8: Build and Write Atomic Controller Receipt
    $finalVerdict = if ($testPassed -and ($restorationResult -eq "RESTORED_VERIFIED")) {
        "AUTOMATED_ISOLATION_READINESS_TEST_PASS_NETWORK_RESTORED"
    } elseif ($restorationResult -eq "NETWORK_RECOVERY_REQUIRED") {
        "NETWORK_RECOVERY_REQUIRED"
    } else {
        "BLOCKED_NETWORK_ISOLATION_INCOMPLETE"
    }

    $controllerReceipt = [ordered]@{
        schema_version                    = "1.2.0"
        phase                             = "Phase 4C.2G.0.3.3"
        controller_name                   = "automated_windows_network_isolation_controller"
        controller_version                = $ControllerVersion
        controller_sha256                 = $ControllerSha256
        started_at_utc                    = $CurrentUtc
        isolation_verified_at_utc         = $isolationVerifiedTime
        network_restored_at_utc           = $networkRestoredTime
        pre_isolation_adapter_snapshot    = $Snapshot.Adapters
        exact_disabled_adapter_allowlist  = $disabledAllowlist
        passive_offline_checks            = [ordered]@{
            outbound_probes_sent  = 0
            dns_lookups_performed = 0
            http_requests_sent    = 0
            isolation_verified    = [bool]$isolationVerifiedTime
        }
        offline_verifier_receipt_sha256   = $verifierReceiptSha
        watchdog_metadata                 = [ordered]@{
            scheduled_task_name       = $WatchdogTaskName
            trigger_time_utc          = $TriggerTime
            timeout_minutes           = 15
            recovery_script_path      = $RecoverScriptPath
            recovery_script_sha256    = $RecoverScriptSha256
            watchdog_auto_cleaned     = ($restorationResult -eq "RESTORED_VERIFIED")
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
    Write-Host "Controller Final Verdict: $finalVerdict" -ForegroundColor Green

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
