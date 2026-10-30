$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskRoot = 'D:\Games\STT_TTS_project'
$env:STT_TTS_DATA_ROOT = $taskRoot
$env:TEMP = Join-Path $taskRoot 'tmp'
$env:TMP = $env:TEMP
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
$taskPython = Join-Path $taskRoot 'envs\py312\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Окружение не найдено. См. README.md.' }
& $taskPython -m stt_tts.api.app
