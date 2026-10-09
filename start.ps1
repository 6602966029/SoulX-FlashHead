param([switch]$Streaming)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonPath = Join-Path $projectRoot 'tools\python-official\python.exe'
$launcherPath = Join-Path $projectRoot 'launch_windows.py'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Project Python is missing. Install the runtime first.'
}
$port = if ($Streaming) { 7861 } else { 7860 }
$existing = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Output "Service already listening at http://127.0.0.1:$port"
    exit 0
}
$logDirectory = Join-Path $projectRoot 'logs'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$arguments = @('-u', ('"{0}"' -f $launcherPath))
if ($Streaming) { $arguments += '--streaming' }
$process = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDirectory "gradio-$port.log") -RedirectStandardError (Join-Path $logDirectory "gradio-$port.err.log")
$process.Id | Set-Content (Join-Path $logDirectory "gradio-$port.pid")
Write-Output "Starting http://127.0.0.1:$port (PID $($process.Id)). Logs: $logDirectory"
