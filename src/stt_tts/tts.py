"""SPDX-License-Identifier: Apache-2.0. Isolated Qwen TTS runtime."""
from .paths import ROOT, configure
configure()

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
CLONE_MODEL_ID = 'Qwen/Qwen3-TTS-12Hz-0.6B-Base'
SPEAKERS = ("Ryan", "Aiden", "Serena", "Vivian", "Uncle_Fu", "Dylan", "Eric", "Ono_Anna", "Sohee")
DEFAULT_SPEAKER = "Serena"
MAX_TEXT = 2000


def validate(text, speaker, speakers=SPEAKERS):
    if not isinstance(text, str):
        raise ValueError("В поле text нужен текст")
    text = text.lstrip("\ufeff").strip()
    if not text:
        raise ValueError("Введи текст или загрузи непустой TXT-файл")
    if len(text) > MAX_TEXT:
        raise ValueError(f"Максимум {MAX_TEXT} символов за один синтез")
    if any(ord(c) < 32 and c not in "\n\r\t" for c in text):
        raise ValueError("Текст содержит недопустимые управляющие символы")
    if not isinstance(speaker, str) or speaker not in speakers:
        raise ValueError("Выбери голос из списка")
    return text


class TTSEngine:
    environment = 'tts'
    worker_module = 'stt_tts.tts_worker'
    worker_log = 'tts-worker.log'
    def synthesize(self, text, speaker=DEFAULT_SPEAKER):
        text = validate(text, speaker)
        return self._run({'text': text, 'speaker': speaker}, 'tts.json')

    def _run(self, payload, manifest_name):
        python = ROOT / 'envs' / self.environment / 'Scripts/python.exe'
        if not python.exists() or not (ROOT / 'models' / manifest_name).exists():
            raise RuntimeError('Модель TTS не подготовлена. Проверь scripts/setup-tts.ps1 и scripts/setup-qwen-clone.ps1.')
        begin = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="tts-", dir=ROOT / "tmp") as directory:
            folder = Path(directory)
            request = folder / "input.json"
            request.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
            try:
                run = subprocess.run(
                    [str(python), '-m', self.worker_module, str(request)],
                    cwd=ROOT / 'tmp', env=env,
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                    timeout=900, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("Синтез превысил 15 минут. Попробуй более короткий текст.") from exc
            if run.returncode:
                exit_code = f'0x{run.returncode & 0xffffffff:08X}' if os.name == 'nt' else str(run.returncode)
                diagnostic = f'Процесс завершился с кодом {exit_code}\n\n{run.stderr[-16000:]}'
                (ROOT / 'logs' / self.worker_log).write_text(diagnostic, encoding="utf-8")
                if "out of memory" in run.stderr.lower():
                    raise RuntimeError("Недостаточно VRAM для TTS. Закрой другие GPU-программы или установи TTS_DEVICE=cpu и перезапусти сервер.")
                raise RuntimeError(f'Ошибка синтеза (код {exit_code}). Диагностика: '+str(ROOT / 'logs' / self.worker_log))
            metadata = json.loads((folder / "result.json").read_text(encoding="utf-8"))
            metadata["processing_seconds"] = round(time.perf_counter() - begin, 3)
            audio = (folder / "output.wav").read_bytes()
            return audio, metadata


class QwenCloneEngine(TTSEngine):
    clone_manifest = 'qwen-clone.json'
    def __init__(self, voices):
        self.voices = voices

    def synthesize(self, text, speaker, ref_text=''):
        voice = self.voices.get(speaker)
        text = validate(text, speaker, speakers=(speaker,))
        ref_text = validate_reference_text(ref_text)
        reference = self.voices.path(speaker, '.wav')
        if not reference.is_file():
            raise ValueError('Не найден образец голоса. Загрузи его заново.')
        audio, metadata = self._run(dict(text=text, speaker=speaker, mode='clone',
                                         reference_audio=str(reference), ref_text=ref_text), self.clone_manifest)
        metadata.update(voice_name=voice['name'], cloned_voice=True)
        return audio, metadata


def validate_reference_text(value):
    if not isinstance(value, str) or len(value) > 2000:
        raise ValueError('Расшифровка образца должна быть текстом до 2000 символов')
    if any(ord(c) < 32 and c not in '\n\r\t' for c in value):
        raise ValueError('Расшифровка содержит недопустимые символы')
    return value.strip()
