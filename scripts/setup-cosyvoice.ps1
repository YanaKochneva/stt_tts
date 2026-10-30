$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$taskRoot = 'D:\Games\STT_TTS_project'
$env:TEMP = Join-Path $taskRoot 'tmp'
$env:TMP = $env:TEMP
$env:UV_CACHE_DIR = Join-Path $taskRoot 'cache\uv'
$env:PYTHONDONTWRITEBYTECODE = '1'
$taskSource = Join-Path $taskRoot 'tools\cosyvoice3'
$taskRevision = '074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc'
if (-not (Test-Path -LiteralPath $taskSource)) {
    git clone https://github.com/QwenAudio/CosyVoice.git $taskSource
    if ($LASTEXITCODE -ne 0) { throw 'Ошибка клонирования CosyVoice' }
    git -C $taskSource checkout $taskRevision
    git -C $taskSource submodule update --init --recursive
    if ($LASTEXITCODE -ne 0) { throw 'Ошибка подготовки подмодулей' }
}
if ((git -C $taskSource rev-parse HEAD) -ne $taskRevision) { throw 'Неожиданная версия runtime CosyVoice' }
$taskPython = Join-Path $taskRoot 'envs\cosyvoice3\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    uv venv --python 3.12 (Join-Path $taskRoot 'envs\cosyvoice3')
    if ($LASTEXITCODE -ne 0) { throw 'Ошибка создания окружения' }
}
uv pip install --python $taskPython torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw 'Ошибка установки CUDA PyTorch' }
uv pip install --python $taskPython -r requirements/cosyvoice.txt -c requirements/cosyvoice.lock.txt --build-constraint requirements/cosyvoice-build.txt
if ($LASTEXITCODE -ne 0) { throw 'Ошибка установки зависимостей' }
uv pip install --python $taskPython --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'Ошибка установки пакета проекта в окружение CosyVoice' }
& $taskPython -m stt_tts.prepare_cosyvoice
if ($LASTEXITCODE -ne 0) { throw 'Ошибка загрузки весов CosyVoice' }
