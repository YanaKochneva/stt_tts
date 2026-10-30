$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$taskRoot = 'D:\Games\STT_TTS_project'
$env:TEMP = Join-Path $taskRoot 'tmp'
$env:TMP = $env:TEMP
$env:UV_CACHE_DIR = Join-Path $taskRoot 'cache\uv'
$env:PYTHONDONTWRITEBYTECODE = '1'
New-Item -ItemType Directory -Force -Path $env:TEMP,$env:UV_CACHE_DIR | Out-Null
$taskPython = Join-Path $taskRoot 'envs\py312\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    uv venv --python 3.12 (Join-Path $taskRoot 'envs\py312')
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось создать окружение' }
}
uv pip install --python $taskPython torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw 'Установка PyTorch не удалась' }
uv pip install --python $taskPython -e .
if ($LASTEXITCODE -ne 0) { throw 'Установка проекта не удалась' }
