$ErrorActionPreference = 'Stop'
$taskPidFile = 'D:\Games\STT_TTS_project\data\server.pid'
if (-not (Test-Path -LiteralPath $taskPidFile)) {
    Write-Host 'Фоновый сервер проекта не зарегистрирован.'
    exit 0
}
$taskProcessId = [int](Get-Content -LiteralPath $taskPidFile -Raw)
$taskProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$taskProcessId"
if ($taskProcess) {
    $taskExpectedPython = 'D:\Games\STT_TTS_project\envs\py312\Scripts\python.exe'
    if ($taskProcess.ExecutablePath -ne $taskExpectedPython -or
        $taskProcess.CommandLine -notmatch '\bstt_tts\.server\b') {
        throw 'PID относится к другому процессу. Остановка отменена.'
    }
    Stop-Process -Id $taskProcessId
    Write-Host 'Фоновый сервер остановлен.'
}
Remove-Item -LiteralPath $taskPidFile
