param([switch]$WithOBS)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonPath = Join-Path $projectRoot 'tools\python-official\python.exe'
$logDirectory = Join-Path $projectRoot 'logs'
New-Item -ItemType Directory $logDirectory -Force | Out-Null
& (Join-Path $projectRoot 'start.ps1')
$listener = Get-NetTCPConnection -LocalPort 7862 -State Listen -ErrorAction SilentlyContinue
if (-not $listener) {
    $arguments = @('-X', 'utf8', '-u', ('"{0}"' -f (Join-Path $projectRoot 'live_setup.py')))
    $process = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDirectory 'live-setup.log') -RedirectStandardError (Join-Path $logDirectory 'live-setup.err.log')
    $process.Id | Set-Content (Join-Path $logDirectory 'live-setup.pid')
}
if ($WithOBS) {
    $obsPath = 'D:\obs\obs-studio\bin\64bit\obs64.exe'
    if (-not (Get-Process obs64 -ErrorAction SilentlyContinue)) {
        & $pythonPath -X utf8 (Join-Path $projectRoot 'configure_live.py')
        if ($LASTEXITCODE -ne 0) { throw 'OBS configuration could not be applied. See the message above.' }
        Start-Process -FilePath $obsPath -ArgumentList '--collection SoulXLive --profile SoulXLive' -WorkingDirectory (Split-Path $obsPath) -WindowStyle Hidden
    } else {
        Write-Output 'OBS is already running. Save and close it, then rerun with -WithOBS to apply SoulXLive settings.'
    }
}
Write-Output 'Setup: http://127.0.0.1:7862/'
Write-Output 'OBS player: http://127.0.0.1:7862/player'
Write-Output 'No public stream is started by this script.'
