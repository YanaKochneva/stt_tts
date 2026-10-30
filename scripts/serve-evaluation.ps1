$ErrorActionPreference = 'Stop'
$taskRoot = 'D:\Games\STT_TTS_project'
$taskFolder = Join-Path $taskRoot 'evaluation\2026-10-29'
$taskPython = Join-Path $taskRoot 'envs\py312\Scripts\python.exe'
$taskLog = Join-Path $taskFolder 'http.log'
$taskProcess = Start-Process -FilePath $taskPython `
    -ArgumentList '-m','http.server','8091','--bind','127.0.0.1','--directory',('"' + $taskFolder + '"') `
    -WorkingDirectory $taskFolder -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $taskLog -RedirectStandardError (Join-Path $taskFolder 'http-error.log')
Set-Content -LiteralPath (Join-Path $taskFolder 'http.pid') -Value $taskProcess.Id
Write-Output "Оценка доступна: http://127.0.0.1:8091/"
