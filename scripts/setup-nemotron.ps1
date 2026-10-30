$ErrorActionPreference = 'Stop'
$taskRoot = 'D:\Games\STT_TTS_project'
$toolsRoot = Join-Path $taskRoot 'tools'
$modelRoot = Join-Path $taskRoot 'models\nemotron'
$installer = Join-Path $toolsRoot 'nemo-install.ps1'
$runtime = Join-Path $toolsRoot 'nemo-speech\bin\nemo-speech.exe'
New-Item -ItemType Directory -Force -Path $toolsRoot, $modelRoot | Out-Null

if (-not (Test-Path -LiteralPath $runtime)) {
    Invoke-WebRequest 'https://raw.githubusercontent.com/NVIDIA/NeMo-Speech.cpp/main/scripts/install.ps1' -OutFile $installer
    & $installer -Channel nightly -Prefix (Join-Path $toolsRoot 'nemo-speech') -Backend cuda -NoModifyPath -BinaryOnly
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить NeMo-Speech.cpp' }
}

$env:NEMO_SPEECH_MODEL_DIR = $modelRoot
& $runtime pull nemotron-3.5
if ($LASTEXITCODE -ne 0) { throw 'Не удалось скачать Nemotron 3.5 ASR' }
$weights = Get-ChildItem -LiteralPath $modelRoot -Recurse -File -Filter 'nemotron-3.5-asr-streaming-0.6b.q8_0.gguf' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $weights) { throw 'Веса GGUF не найдены после скачивания' }
$manifest = @{model = 'nvidia/nemotron-3.5-asr-streaming-0.6b'; model_path = $weights.FullName;
              runtime = $runtime; license = 'OpenMDW-1.1'}
$manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRoot 'models\nemotron-asr.json') -Encoding utf8
Write-Host 'Nemotron установлен. Перезапусти start.ps1 и выбери Nemotron в интерфейсе.'
