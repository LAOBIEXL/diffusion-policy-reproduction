param(
    [string]$OutputPath = "$PSScriptRoot/../records/environment/windows.json"
)

$os = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor
$snapshot = [ordered]@{
    recorded_at = [DateTimeOffset]::Now.ToString('o')
    os = $os | Select-Object Caption, Version, OSArchitecture, TotalVisibleMemorySize, FreePhysicalMemory
    cpu = $cpu | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
    gpu = @(nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv)
    disks = @(Get-PSDrive -PSProvider FileSystem | Select-Object Name, Used, Free)
    commands = @(Get-Command git, python, docker, wsl, nvidia-smi | Select-Object Name, Source)
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputPath) | Out-Null
$snapshot | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $OutputPath -Encoding utf8
Write-Output $OutputPath
