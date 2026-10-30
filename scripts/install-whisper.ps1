$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$taskRoot = 'D:\Games\STT_TTS_project'
$env:STT_TTS_DATA_ROOT = $taskRoot
$env:UV_CACHE_DIR = Join-Path $taskRoot 'cache\uv'
$env:TEMP = Join-Path $taskRoot 'tmp'
$env:TMP = $env:TEMP
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$taskPython = Join-Path $taskRoot 'envs\py312\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Основное Python-окружение не найдено' }
uv pip install --python $taskPython 'faster-whisper==1.2.1'
if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить faster-whisper' }
& $taskPython -m stt_tts.prepare_whisper
if ($LASTEXITCODE -ne 0) { throw 'Не удалось скачать модель Whisper' }
