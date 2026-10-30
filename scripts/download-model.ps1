$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'D:\Games\STT_TTS_project\envs\py312\Scripts\python.exe' -m stt_tts.prepare
