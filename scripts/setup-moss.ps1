$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$taskRoot = 'D:\Games\STT_TTS_project'
$env:TEMP = Join-Path $taskRoot 'tmp'
$env:TMP = $env:TEMP
$env:UV_CACHE_DIR = Join-Path $taskRoot 'cache\uv'
$env:PYTHONDONTWRITEBYTECODE = '1'
$taskPython = Join-Path $taskRoot 'envs\py312\Scripts\python.exe'
$taskRepo = Join-Path $taskRoot 'tools\moss-tts-nano'
$taskRevision = '8b7bcc9341b3b4ef3a3a58ba1338a7d85ff133eb'
if (-not (Test-Path -LiteralPath $taskRepo)) {
    git init $taskRepo
    if ($LASTEXITCODE -ne 0) { throw 'git init failed' }
    git -C $taskRepo remote add origin https://github.com/OpenMOSS/MOSS-TTS-Nano.git
    git -C $taskRepo fetch --depth 1 origin $taskRevision
    if ($LASTEXITCODE -ne 0) { throw 'git fetch failed' }
    git -C $taskRepo checkout --detach FETCH_HEAD
    if ($LASTEXITCODE -ne 0) { throw 'git checkout failed' }
}
$taskActual = git -C $taskRepo rev-parse HEAD
if ($taskActual -ne $taskRevision) { throw 'Runtime MOSS отличается от проверенной версии' }
uv pip install --python $taskPython -e .
if ($LASTEXITCODE -ne 0) { throw 'Установка ONNX-зависимостей не удалась' }
& $taskPython -m stt_tts.prepare_moss
if ($LASTEXITCODE -ne 0) { throw 'Загрузка MOSS не удалась' }
