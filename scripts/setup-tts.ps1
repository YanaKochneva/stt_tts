$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$taskRoot = 'D:\Games\STT_TTS_project'
$env:TEMP = Join-Path $taskRoot 'tmp'
$env:TMP = $env:TEMP
$env:UV_CACHE_DIR = Join-Path $taskRoot 'cache\uv'
$env:PYTHONDONTWRITEBYTECODE = '1'
$taskPython = Join-Path $taskRoot 'envs\tts\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    uv venv --python 3.12 (Join-Path $taskRoot 'envs\tts')
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось создать окружение TTS' }
}
uv pip install --python $taskPython torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw 'Установка CUDA PyTorch не удалась' }
uv pip install --python $taskPython -r requirements/qwen-tts.txt
if ($LASTEXITCODE -ne 0) { throw 'Установка TTS не удалась' }
uv pip install --python $taskPython --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'Установка пакета проекта в окружение TTS не удалась' }
& $taskPython -m stt_tts.prepare_tts
if ($LASTEXITCODE -ne 0) { throw 'Загрузка весов TTS не удалась' }
