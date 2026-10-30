$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$taskRoot = 'D:\Games\STT_TTS_project'
$env:TEMP = Join-Path $taskRoot 'tmp'
$env:TMP = $env:TEMP
$env:PYTHONDONTWRITEBYTECODE = '1'
$taskPython = Join-Path $taskRoot 'envs\tts\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Сначала запусти scripts/setup-tts.ps1 для установки окружения Qwen TTS'
}
& $taskPython -m stt_tts.prepare_qwen_clone
if ($LASTEXITCODE -ne 0) { throw 'Загрузка Qwen Base не удалась' }
