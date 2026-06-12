param(
    [string]$TaskName = "CashSnapWinMemoryCleaner",
    [string]$MemoryAreas = "/StandbyList /WorkingSet",
    [string]$ExePath = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
if ([string]::IsNullOrWhiteSpace($ExePath)) {
    $toolRoot = Join-Path $repoRoot ".cache_runtime\tools\winmemorycleaner"
    $candidate = Get-ChildItem -LiteralPath $toolRoot -Filter "WinMemoryCleaner.exe" -Recurse |
        Sort-Object -Property FullName -Descending |
        Select-Object -First 1
    if ($null -eq $candidate) {
        throw "No WinMemoryCleaner.exe found under $toolRoot"
    }
    $ExePath = $candidate.FullName
}

if (-not (Test-Path -LiteralPath $ExePath)) {
    throw "WinMemoryCleaner executable not found: $ExePath"
}

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this script from an elevated PowerShell session so the task can use RunLevel Highest."
}

$action = New-ScheduledTaskAction -Execute $ExePath -Argument $MemoryAreas
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 2)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Settings $settings `
    -RunLevel Highest `
    -Description "Silent emergency RAM cleanup for CashSnap RunLong headroom guard using WinMemoryCleaner." `
    -Force | Out-Null

Write-Output "registered $TaskName -> $ExePath $MemoryAreas"
