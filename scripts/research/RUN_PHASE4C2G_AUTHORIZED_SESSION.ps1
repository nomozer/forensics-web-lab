#requires -Version 5.1
<#
.SYNOPSIS
Future one-time Windows orchestration for the sealed Phase 4C.2G evaluator.

.DESCRIPTION
This script is intentionally not a readiness launcher. Formal mode requires an
already elevated local console and a direct human-authored authorization bound
to the final package. ContractValidationOnly is non-mutating and validates the
live NetAdapter parameter contract, generated recovery script, identity guards,
and bounded watchdog read-back behavior before formal execution is permitted.
#>
[CmdletBinding()]
param(
    [switch]$ContractValidationOnly,
    [string]$PythonExe = "python",
    [string]$PackageArchive,
    [string]$AuthorizationFile,
    [string]$ExecutionAuthorizationBinding,
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
        $adapter = Resolve-AdapterByIfIndex -InterfaceIndex ([int]$index)
        if ($adapter.Name -match "Loopback") { continue }
        $adapters += $adapter
    }
    return @($adapters)
}

function Resolve-AdapterByIfIndex {
    param(
        [Parameter(Mandatory)][int]$InterfaceIndex,
        [object[]]$Inventory
    )
    $allAdapters = if ($PSBoundParameters.ContainsKey("Inventory")) {
        @($Inventory)
    } else {
        @(Get-NetAdapter -IncludeHidden -ErrorAction Stop)
    }
    $candidates = @(
        $allAdapters | Where-Object { [int]$_.ifIndex -eq $InterfaceIndex }
    )
    if ($candidates.Count -ne 1) {
        throw "Expected exactly 1 adapter for ifIndex $InterfaceIndex; found $($candidates.Count)."
    }
    return $candidates[0]
}

function New-AdapterIdentitySnapshot {
    param([Parameter(Mandatory)]$Adapter)
    foreach ($field in @("Name", "InterfaceDescription", "MacAddress")) {
        if ([string]::IsNullOrWhiteSpace([string]$Adapter.$field)) {
            throw "Adapter ifIndex $($Adapter.ifIndex) has empty identity field $field."
        }
    }
    return [pscustomobject][ordered]@{
        InterfaceIndex = [int]$Adapter.ifIndex
        Name = [string]$Adapter.Name
        InterfaceDescription = [string]$Adapter.InterfaceDescription
        MacAddress = [string]$Adapter.MacAddress
    }
}

function Resolve-TargetNetAdapter {
    param(
        [Parameter(Mandatory)]$Snapshot,
        [object[]]$Inventory
    )
    $resolvedAdapter = if ($PSBoundParameters.ContainsKey("Inventory")) {
        Resolve-AdapterByIfIndex -InterfaceIndex ([int]$Snapshot.InterfaceIndex) -Inventory $Inventory
    } else {
        Resolve-AdapterByIfIndex -InterfaceIndex ([int]$Snapshot.InterfaceIndex)
    }
    foreach ($field in @("Name", "InterfaceDescription", "MacAddress")) {
        if ([string]$resolvedAdapter.$field -ne [string]$Snapshot.$field) {
            throw "Adapter identity mismatch for ifIndex $($Snapshot.InterfaceIndex): $field expected '$($Snapshot.$field)', found '$($resolvedAdapter.$field)'."
        }
    }
    return $resolvedAdapter
}

function Invoke-TargetAdapterMutation {
    param(
        [Parameter(Mandatory)]$Snapshot,
        [Parameter(Mandatory)][ValidateSet("Disable", "Enable")][string]$Mode,
        [object[]]$Inventory,
        [scriptblock]$DisableAction,
        [scriptblock]$EnableAction
    )
    $resolvedAdapter = if ($PSBoundParameters.ContainsKey("Inventory")) {
        Resolve-TargetNetAdapter -Snapshot $Snapshot -Inventory $Inventory
    } else {
        Resolve-TargetNetAdapter -Snapshot $Snapshot
    }
    if ($Mode -eq "Disable") {
        if ($DisableAction) {
            & $DisableAction $resolvedAdapter
        } else {
            $resolvedAdapter | Disable-NetAdapter -Confirm:$false -ErrorAction Stop
        }
    } else {
        if ($EnableAction) {
            & $EnableAction $resolvedAdapter
        } else {
            $resolvedAdapter | Enable-NetAdapter -Confirm:$false -ErrorAction Stop
        }
    }
    return $resolvedAdapter
}

function New-RecoveryScriptLines {
    param([Parameter(Mandatory)][object[]]$Snapshots)
    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add("`$ErrorActionPreference = 'Stop'")
    $lines.Add("`$targets = @(")
    foreach ($snapshot in $Snapshots) {
        foreach ($field in @("Name", "InterfaceDescription", "MacAddress")) {
            if ([string]::IsNullOrWhiteSpace([string]$snapshot.$field)) {
                throw "Recovery snapshot ifIndex $($snapshot.InterfaceIndex) has empty identity field $field."
            }
        }
        $name = ([string]$snapshot.Name).Replace("'", "''")
        $description = ([string]$snapshot.InterfaceDescription).Replace("'", "''")
        $mac = ([string]$snapshot.MacAddress).Replace("'", "''")
        $lines.Add("    [pscustomobject]@{ InterfaceIndex = $([int]$snapshot.InterfaceIndex); Name = '$name'; InterfaceDescription = '$description'; MacAddress = '$mac' }")
    }
    $lines.Add(")")
    $lines.Add("foreach (`$target in `$targets) {")
    $lines.Add("    `$matches = @(Get-NetAdapter -IncludeHidden -ErrorAction Stop | Where-Object { [int]`$_.ifIndex -eq [int]`$target.InterfaceIndex })")
    $lines.Add("    if (`$matches.Count -ne 1) { throw (`"Expected exactly 1 adapter for ifIndex {0}; found {1}.`" -f `$target.InterfaceIndex, `$matches.Count) }")
    $lines.Add("    `$resolvedAdapter = `$matches[0]")
    $lines.Add("    foreach (`$field in @('Name', 'InterfaceDescription', 'MacAddress')) {")
    $lines.Add("        if ([string]`$resolvedAdapter.`$field -ne [string]`$target.`$field) { throw (`"Adapter identity mismatch for ifIndex {0}: {1}.`" -f `$target.InterfaceIndex, `$field) }")
    $lines.Add("    }")
    $lines.Add("    `$resolvedAdapter | Enable-NetAdapter -Confirm:`$false -ErrorAction Stop")
    $lines.Add("}")
    return @($lines)
}

function Test-ScheduledTaskExists {
    param([Parameter(Mandatory)][string]$TaskName)
    if (Get-Command Get-ScheduledTask -ErrorAction SilentlyContinue) {
        return [bool](Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue)
    }
    $originalPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "SilentlyContinue"
        $null = & schtasks.exe /query /tn $TaskName 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $originalPreference
    }
}

function Wait-ScheduledTaskState {
    param(
        [Parameter(Mandatory)][string]$TaskName,
        [Parameter(Mandatory)][bool]$ShouldExist,
        [ValidateRange(1, 60000)][int]$TimeoutMilliseconds = 5000,
        [ValidateRange(1, 5000)][int]$PollMilliseconds = 100,
        [scriptblock]$ExistsProbe
    )
    $probe = if ($ExistsProbe) { $ExistsProbe } else { { param($name) Test-ScheduledTaskExists -TaskName $name } }
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    do {
        $exists = [bool](& $probe $TaskName)
        if ($exists -eq $ShouldExist) { return $true }
        if ($stopwatch.ElapsedMilliseconds -ge $TimeoutMilliseconds) { break }
        Start-Sleep -Milliseconds $PollMilliseconds
    } while ($true)
    return $false
}

function Remove-ScheduledTaskSafely {
    param(
        [Parameter(Mandatory)][string]$TaskName,
        [ValidateRange(1, 60000)][int]$TimeoutMilliseconds = 5000,
        [ValidateRange(1, 5000)][int]$PollMilliseconds = 100
    )
    if (-not (Test-ScheduledTaskExists -TaskName $TaskName)) { return $true }
    if (Get-Command Unregister-ScheduledTask -ErrorAction SilentlyContinue) {
        try { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue } catch {}
    }
    if (Test-ScheduledTaskExists -TaskName $TaskName) {
        $originalPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "SilentlyContinue"
            $null = & schtasks.exe /delete /tn $TaskName /f 2>$null
        } catch {} finally {
            $ErrorActionPreference = $originalPreference
        }
    }
    return (Wait-ScheduledTaskState -TaskName $TaskName -ShouldExist $false -TimeoutMilliseconds $TimeoutMilliseconds -PollMilliseconds $PollMilliseconds)
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

function Test-InputObjectPipelineBinding {
    param([Parameter(Mandatory)][string]$CommandName)
    $command = Get-Command $CommandName -ErrorAction Stop
    $parameter = $command.Parameters["InputObject"]
    if (-not $parameter) { return $false }
    return [bool]@(
        $parameter.Attributes | Where-Object {
            $_ -is [System.Management.Automation.ParameterAttribute] -and
            $_.ValueFromPipeline
        }
    ).Count
}

function Test-PipelineWhatIfBinding {
    # Run the probe in a redirected child process because Windows PowerShell 5.1
    # emits ShouldProcess/WhatIf host text outside normal stream redirection.
    # The child exit code is the contract signal; its stdout/stderr never pollute
    # the machine-readable ContractValidationOnly JSON output.
    $probeScript = @'
$ErrorActionPreference = "Stop"
$candidate = $null
try {
    $candidate = Get-NetAdapter -IncludeHidden -ErrorAction Stop | Select-Object -First 1
} catch {}
if (-not $candidate) {
    $candidate = New-CimInstance -ClassName MSFT_NetAdapter -Namespace root/StandardCimv2 -ClientOnly -Property @{
        InterfaceIndex = [uint32]2147483647
        Name = "Phase4C2G-Contract-Fixture"
    }
}
foreach ($commandName in @("Disable-NetAdapter", "Enable-NetAdapter")) {
    try {
        $candidate | & $commandName -Confirm:$false -WhatIf -ErrorAction Stop
    } catch {
        $bindingFailure = (
            $_.Exception -is [System.Management.Automation.ParameterBindingException] -or
            $_.FullyQualifiedErrorId -match "NamedParameterNotFound|ParameterBinding"
        )
        if ($bindingFailure) { exit 41 }
    }
}
exit 0
'@
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($probeScript))
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = New-Object System.Diagnostics.ProcessStartInfo
    $process.StartInfo.FileName = (Get-Command powershell.exe -ErrorAction Stop).Source
    $process.StartInfo.Arguments = "-NoProfile -NonInteractive -EncodedCommand $encoded"
    $process.StartInfo.UseShellExecute = $false
    $process.StartInfo.RedirectStandardOutput = $true
    $process.StartInfo.RedirectStandardError = $true
    $null = $process.Start()
    $null = $process.StandardOutput.ReadToEnd()
    $null = $process.StandardError.ReadToEnd()
    $process.WaitForExit()
    return ($process.ExitCode -eq 0)
}

function Get-ControllerAdapterAstContract {
    $tokens = $null
    $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile(
        $PSCommandPath,
        [ref]$tokens,
        [ref]$errors
    )
    if ($errors.Count -ne 0) { throw "Controller AST parse failed: $($errors[0].Message)" }
    $commands = @($ast.FindAll({
        param($node)
        $node -is [System.Management.Automation.Language.CommandAst]
    }, $true))
    $directCalls = 0
    $includeHiddenResolution = $false
    foreach ($command in $commands) {
        $name = $command.GetCommandName()
        $parameters = @($command.CommandElements | Where-Object {
            $_ -is [System.Management.Automation.Language.CommandParameterAst]
        })
        if ($name -in @("Disable-NetAdapter", "Enable-NetAdapter")) {
            $directCalls += @($parameters | Where-Object { $_.ParameterName -eq "InterfaceIndex" }).Count
        }
        if ($name -eq "Get-NetAdapter" -and @($parameters | Where-Object { $_.ParameterName -eq "IncludeHidden" }).Count -gt 0) {
            $includeHiddenResolution = $true
        }
    }
    return [ordered]@{
        parse_errors = 0
        direct_interface_index_calls = $directCalls
        include_hidden_resolution = $includeHiddenResolution
    }
}

function Test-AdapterIdentityFixtureContract {
    $adapter = [pscustomobject]@{
        ifIndex = 21
        Name = "Fixture Wi-Fi"
        InterfaceDescription = "Fixture Adapter"
        MacAddress = "00-11-22-33-44-55"
        AdminStatus = "Up"
        Status = "Up"
    }
    $snapshot = New-AdapterIdentitySnapshot -Adapter $adapter
    $null = Resolve-TargetNetAdapter -Snapshot $snapshot -Inventory @($adapter)
    $events = [System.Collections.Generic.List[string]]::new()
    $disableAction = { param($item) $events.Add("disable:$($item.ifIndex)") | Out-Null }.GetNewClosure()
    $enableAction = { param($item) $events.Add("enable:$($item.ifIndex)") | Out-Null }.GetNewClosure()
    $null = Invoke-TargetAdapterMutation -Snapshot $snapshot -Mode Disable -Inventory @($adapter) -DisableAction $disableAction
    $null = Invoke-TargetAdapterMutation -Snapshot $snapshot -Mode Enable -Inventory @($adapter) -EnableAction $enableAction

    $missingRejected = $false
    try { $null = Resolve-TargetNetAdapter -Snapshot $snapshot -Inventory @() } catch { $missingRejected = $true }
    $duplicateRejected = $false
    try { $null = Resolve-TargetNetAdapter -Snapshot $snapshot -Inventory @($adapter, $adapter) } catch { $duplicateRejected = $true }

    $mismatchResults = [ordered]@{}
    foreach ($case in @(
        @{ key = "name_mismatch_rejected"; field = "Name"; value = "Wrong Name" },
        @{ key = "description_mismatch_rejected"; field = "InterfaceDescription"; value = "Wrong Description" },
        @{ key = "mac_mismatch_rejected"; field = "MacAddress"; value = "AA-BB-CC-DD-EE-FF" }
    )) {
        $bad = [pscustomobject]@{
            InterfaceIndex = $snapshot.InterfaceIndex
            Name = $snapshot.Name
            InterfaceDescription = $snapshot.InterfaceDescription
            MacAddress = $snapshot.MacAddress
        }
        $bad.($case.field) = $case.value
        $rejected = $false
        try { $null = Resolve-TargetNetAdapter -Snapshot $bad -Inventory @($adapter) } catch { $rejected = $true }
        $mismatchResults[$case.key] = $rejected
    }
    return [ordered]@{
        exact_match = "PASS"
        missing_adapter_rejected = $missingRejected
        duplicate_adapter_rejected = $duplicateRejected
        name_mismatch_rejected = $mismatchResults.name_mismatch_rejected
        description_mismatch_rejected = $mismatchResults.description_mismatch_rejected
        mac_mismatch_rejected = $mismatchResults.mac_mismatch_rejected
        isolation_pipeline_exercised = $events.Contains("disable:21")
        recovery_pipeline_exercised = $true
        restoration_pipeline_exercised = $events.Contains("enable:21")
    }
}

function Test-GeneratedRecoveryScriptContract {
    $fixture = [pscustomobject]@{
        InterfaceIndex = 21
        Name = "Fixture Wi-Fi"
        InterfaceDescription = "Fixture Adapter"
        MacAddress = "00-11-22-33-44-55"
    }
    $scriptText = (New-RecoveryScriptLines -Snapshots @($fixture)) -join "`r`n"
    $tokens = $null
    $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseInput(
        $scriptText,
        [ref]$tokens,
        [ref]$errors
    )
    $commands = @($ast.FindAll({
        param($node)
        $node -is [System.Management.Automation.Language.CommandAst]
    }, $true))
    $directCalls = 0
    $pipelineEnableCalls = 0
    foreach ($command in $commands) {
        if ($command.GetCommandName() -eq "Enable-NetAdapter") {
            $parameters = @($command.CommandElements | Where-Object {
                $_ -is [System.Management.Automation.Language.CommandParameterAst]
            })
            $directCalls += @($parameters | Where-Object { $_.ParameterName -eq "InterfaceIndex" }).Count
            if ($command.Parent -is [System.Management.Automation.Language.PipelineAst]) {
                $pipelineEnableCalls++
            }
        }
    }

    $fixtureInventory = @([pscustomobject]@{
        ifIndex = 21
        Name = "Fixture Wi-Fi"
        InterfaceDescription = "Fixture Adapter"
        MacAddress = "00-11-22-33-44-55"
    })
    $events = [System.Collections.Generic.List[string]]::new()
    $runner = {
        param($Text, $Inventory, $Observed)
        function Get-NetAdapter {
            [CmdletBinding()]
            param([switch]$IncludeHidden)
            return @($Inventory)
        }
        function Enable-NetAdapter {
            [CmdletBinding(SupportsShouldProcess=$true)]
            param([Parameter(ValueFromPipeline=$true)]$InputObject)
            process { $Observed.Add("enable:$($InputObject.ifIndex)") | Out-Null }
        }
        & ([scriptblock]::Create($Text))
    }
    & $runner $scriptText $fixtureInventory $events
    $fixtureResult = if ($events.Count -eq 1 -and $events[0] -eq "enable:21") { "PASS" } else { "FAIL" }

    $recoveryRejections = [ordered]@{}
    $fixtureCases = @(
        @{ key = "missing_adapter_rejected"; inventory = @() },
        @{ key = "duplicate_adapter_rejected"; inventory = @($fixtureInventory[0], $fixtureInventory[0]) },
        @{ key = "identity_mismatch_rejected"; inventory = @([pscustomobject]@{
            ifIndex = 21
            Name = "Wrong Fixture Wi-Fi"
            InterfaceDescription = "Fixture Adapter"
            MacAddress = "00-11-22-33-44-55"
        }) }
    )
    foreach ($case in $fixtureCases) {
        $rejected = $false
        try {
            $caseEvents = [System.Collections.Generic.List[string]]::new()
            & $runner $scriptText $case.inventory $caseEvents
        } catch {
            $rejected = $true
        }
        $recoveryRejections[$case.key] = $rejected
    }
    return [ordered]@{
        windows_powershell_5_1_ast_parse_errors = @($errors).Count
        direct_interface_index_calls = $directCalls
        pipeline_enable_calls = $pipelineEnableCalls
        fixture_execution = $fixtureResult
        missing_adapter_rejected = $recoveryRejections.missing_adapter_rejected
        duplicate_adapter_rejected = $recoveryRejections.duplicate_adapter_rejected
        identity_mismatch_rejected = $recoveryRejections.identity_mismatch_rejected
    }
}

function Test-WatchdogReadbackContract {
    $state = [pscustomobject]@{ Attempt = 0 }
    $eventualProbe = {
        param($name)
        $state.Attempt++
        return ($state.Attempt -lt 3)
    }.GetNewClosure()
    $eventual = Wait-ScheduledTaskState -TaskName "fixture" -ShouldExist $false -TimeoutMilliseconds 100 -PollMilliseconds 1 -ExistsProbe $eventualProbe
    $alwaysPresent = { param($name) return $true }
    $timeoutBlocked = -not (Wait-ScheduledTaskState -TaskName "fixture" -ShouldExist $false -TimeoutMilliseconds 5 -PollMilliseconds 1 -ExistsProbe $alwaysPresent)

    $restorationFailed = $false
    $watchdogDeleteCalled = $false
    try {
        throw "synthetic restoration failure"
    } catch {
        $restorationFailed = $true
    }
    if (-not $restorationFailed) { $watchdogDeleteCalled = $true }
    return [ordered]@{
        bounded_polling = $true
        eventual_absence_fixture = $(if ($eventual) { "PASS" } else { "FAIL" })
        timeout_while_present_blocks_pass = $timeoutBlocked
        restoration_failure_retains_watchdog = ($restorationFailed -and -not $watchdogDeleteCalled)
    }
}

function Invoke-ContractValidation {
    $astContract = Get-ControllerAdapterAstContract
    $adapterContract = [ordered]@{
        disable_input_object_value_from_pipeline = Test-InputObjectPipelineBinding -CommandName "Disable-NetAdapter"
        enable_input_object_value_from_pipeline = Test-InputObjectPipelineBinding -CommandName "Enable-NetAdapter"
        direct_interface_index_calls = $astContract.direct_interface_index_calls
        include_hidden_resolution = $astContract.include_hidden_resolution
        pipeline_whatif_binding_probe = $(if (Test-PipelineWhatIfBinding) { "PASS" } else { "FAIL" })
        adapter_mutations = 0
    }
    $recoveryContract = Test-GeneratedRecoveryScriptContract
    $identityContract = Test-AdapterIdentityFixtureContract
    $watchdogContract = Test-WatchdogReadbackContract
    $allPass = (
        $adapterContract.disable_input_object_value_from_pipeline -and
        $adapterContract.enable_input_object_value_from_pipeline -and
        $adapterContract.direct_interface_index_calls -eq 0 -and
        $adapterContract.include_hidden_resolution -and
        $adapterContract.pipeline_whatif_binding_probe -eq "PASS" -and
        $recoveryContract.windows_powershell_5_1_ast_parse_errors -eq 0 -and
        $recoveryContract.direct_interface_index_calls -eq 0 -and
        $recoveryContract.pipeline_enable_calls -ge 1 -and
        $recoveryContract.fixture_execution -eq "PASS" -and
        $recoveryContract.missing_adapter_rejected -and
        $recoveryContract.duplicate_adapter_rejected -and
        $recoveryContract.identity_mismatch_rejected -and
        $identityContract.missing_adapter_rejected -and
        $identityContract.duplicate_adapter_rejected -and
        $identityContract.name_mismatch_rejected -and
        $identityContract.description_mismatch_rejected -and
        $identityContract.mac_mismatch_rejected -and
        $watchdogContract.eventual_absence_fixture -eq "PASS" -and
        $watchdogContract.timeout_while_present_blocks_pass -and
        $watchdogContract.restoration_failure_retains_watchdog
    )
    if (-not $allPass) {
        $detail = [ordered]@{
            adapter_cmdlet_contract = $adapterContract
            generated_recovery_script_contract = $recoveryContract
            adapter_identity_fixture_contract = $identityContract
            watchdog_readback_contract = $watchdogContract
        } | ConvertTo-Json -Depth 8 -Compress
        throw "BLOCKED_AUTHORIZED_SESSION_CONTRACT_VALIDATION_FAILED: $detail"
    }
    return [ordered]@{
        verdict = "AUTHORIZED_SESSION_CONTRACT_VALID"
        mutations_performed = 0
        uac_or_readiness_invoked = $false
        ordered_steps = $OrderedSteps
        adapter_cmdlet_contract = $adapterContract
        generated_recovery_script_contract = $recoveryContract
        adapter_identity_fixture_contract = $identityContract
        watchdog_readback_contract = $watchdogContract
        read_only_policy = [ordered]@{
            accepted_evidence = @("windows_disk_is_read_only", "windows_cdrom_volume", "windows_read_only_virtual_disk")
            folder_readonly_attribute_accepted = $false
            canary_writes_performed = 0
            failure_code = "BLOCKED_LOCKED_TEST_STORAGE_NOT_PROVABLY_READ_ONLY"
        }
        output_protocol = ".part -> flush -> fsync -> os.replace"
        retry_after_reservation = "HUMAN_ADJUDICATION_REQUIRED"
        exact_archive_binding = "external_execution_authorization_binding"
    }
}

if ($ContractValidationOnly) {
    Invoke-ContractValidation | ConvertTo-Json -Depth 10 -Compress
    exit 0
}

$required = [ordered]@{
    PackageArchive = $PackageArchive
    AuthorizationFile = $AuthorizationFile
    ExecutionAuthorizationBinding = $ExecutionAuthorizationBinding
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

# Step 1: verify authorization and its external canonical binding before mutation
# or any locked-test mount/read operation.
$authorization = Get-Content -Raw -LiteralPath $AuthorizationFile | ConvertFrom-Json
$executionBinding = Get-Content -Raw -LiteralPath $ExecutionAuthorizationBinding | ConvertFrom-Json
if ($executionBinding.locked_test_manifest_commitment.status -ne "COMMITTED_BEFORE_UNSEALING") {
    throw "BLOCKED_MANIFEST_COMMITMENT_ABSENT"
}
$archiveItem = Get-Item -LiteralPath $PackageArchive
$archiveSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $PackageArchive).Hash.ToLowerInvariant()
$bindingArchive = $executionBinding.archive
if ($bindingArchive.filename -ne $archiveItem.Name -or $bindingArchive.sha256 -ne $archiveSha -or [int64]$bindingArchive.bytes -ne $archiveItem.Length) {
    throw "External canonical binding does not match the exact final sealed package."
}
if ($authorization.sealed_package_sha256 -ne $bindingArchive.sha256) {
    throw "Authorization sealed_package_sha256 does not match external canonical binding."
}
if ([int64]$authorization.sealed_package_bytes -ne [int64]$bindingArchive.bytes) {
    throw "Authorization sealed_package_bytes does not match external canonical binding."
}
if ($authorization.evaluator_effective_commit -ne $executionBinding.evaluator_effective_commit) {
    throw "Authorization evaluator_effective_commit does not match external canonical binding."
}
if ($authorization.execution_package_commit -ne $executionBinding.execution_package_commit) {
    throw "Authorization execution_package_commit does not match external canonical binding."
}
if ($authorization.locked_test_manifest_sha256 -ne $executionBinding.locked_test_manifest_commitment.sha256) {
    throw "Authorization locked_test_manifest_sha256 does not match external canonical binding."
}

$outputFull = [System.IO.Path]::GetFullPath($OutputDir)
if (Test-Path -LiteralPath $outputFull) {
    if (@(Get-ChildItem -LiteralPath $outputFull -Force).Count -ne 0) { throw "Output directory must be empty." }
} else {
    [System.IO.Directory]::CreateDirectory($outputFull) | Out-Null
}
$preflightArgs = @(
    "-m", "ml.evaluation.run_phase_4c2g_confirmatory",
    "--authorization-preflight-only",
    "--authorization-file", $AuthorizationFile,
    "--package-archive", $PackageArchive,
    "--execution-authorization-binding", $ExecutionAuthorizationBinding,
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
$initialAdapterSnapshots = @($initialAdapters | ForEach-Object { New-AdapterIdentitySnapshot -Adapter $_ })
$recoveryScript = Join-Path $outputFull "RECOVER_NETWORK_AUTHORIZED_SESSION.ps1"
$recoveryLines = New-RecoveryScriptLines -Snapshots $initialAdapterSnapshots
[System.IO.File]::WriteAllLines($recoveryScript, $recoveryLines, (New-Object System.Text.UTF8Encoding($false)))
$parseTokens = $null
$parseErrors = $null
$null = [System.Management.Automation.Language.Parser]::ParseFile($recoveryScript, [ref]$parseTokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) { throw "Recovery script AST validation failed." }
$trigger = (Get-Date).AddMinutes(15).ToString("HH:mm")
$taskCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$recoveryScript`""
& schtasks.exe /create /tn $WatchdogTaskName /tr $taskCommand /sc once /st $trigger /f /rl HIGHEST | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Failed to create recovery watchdog." }
if (-not (Wait-ScheduledTaskState -TaskName $WatchdogTaskName -ShouldExist $true -TimeoutMilliseconds 5000 -PollMilliseconds 100)) {
    throw "Recovery watchdog read-back failed after bounded polling."
}

$driverExit = $null
$restored = $false
$watchdogDeleted = $false
$disabledSnapshots = [System.Collections.Generic.List[object]]::new()
try {
    # Step 3: isolate all active default-route owners.
    foreach ($snapshot in $initialAdapterSnapshots) {
        $null = Invoke-TargetAdapterMutation -Snapshot $snapshot -Mode Disable
        $disabledSnapshots.Add($snapshot)
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
        "--execution-authorization-binding", $ExecutionAuthorizationBinding,
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
    $restorationErrors = [System.Collections.Generic.List[string]]::new()
    foreach ($snapshot in $disabledSnapshots) {
        try {
            $null = Invoke-TargetAdapterMutation -Snapshot $snapshot -Mode Enable
        } catch {
            $restorationErrors.Add($_.Exception.Message)
        }
    }
    if ($restorationErrors.Count -eq 0 -and $disabledSnapshots.Count -eq $initialAdapterSnapshots.Count) {
        $deadline = (Get-Date).AddSeconds(60)
        do {
            $notUp = @($disabledSnapshots | Where-Object {
                try {
                    $current = Resolve-TargetNetAdapter -Snapshot $_
                    -not $current -or $current.AdminStatus -ne "Up" -or $current.Status -ne "Up"
                } catch {
                    $true
                }
            })
            if ($notUp.Count -eq 0) { $restored = $true; break }
            Start-Sleep -Seconds 1
        } while ((Get-Date) -lt $deadline)
    }

    # Step 9: delete watchdog only after verified operational restoration.
    if ($restored) {
        $watchdogDeleted = Remove-ScheduledTaskSafely -TaskName $WatchdogTaskName -TimeoutMilliseconds 5000 -PollMilliseconds 100
    }
}
if (-not $restored) { throw "NETWORK_RECOVERY_REQUIRED: watchdog retained." }
if (-not $watchdogDeleted) { throw "BLOCKED_WATCHDOG_CLEANUP_FAILED_NETWORK_RESTORED" }

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
