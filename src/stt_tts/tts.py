"""SPDX-License-Identifier: Apache-2.0. Isolated Qwen TTS runtime."""
from .paths import ROOT, configure
configure()

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import queue
import threading
import atexit

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
CLONE_MODEL_ID = 'Qwen/Qwen3-TTS-12Hz-0.6B-Base'
SPEAKERS = ("Ryan", "Aiden", "Serena", "Vivian", "Uncle_Fu", "Dylan", "Eric", "Ono_Anna", "Sohee")
DEFAULT_SPEAKER = "Serena"
MAX_TEXT = 2000


class _Worker:
    def __init__(self):
        self.lock = threading.RLock()
        self.process = None
        self.responses = None
        self.log = None

    def stop(self):
        with self.lock:
            if self.process is not None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                self.process = None
            if self.log is not None:
                self.log.close()
                self.log = None

    def _start(self, python, module, env, log_path):
        self.log = log_path.open('a', encoding='utf-8')
        self.process = subprocess.Popen(
            [str(python), '-m', module, '--serve'], cwd=ROOT / 'tmp', env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, encoding='utf-8', bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )
        self.responses = queue.Queue()
        process = self.process
        responses = self.responses
        def read_responses():
            for line in process.stdout:
                responses.put(line)
            responses.put(None)
        threading.Thread(target=read_responses, daemon=True).start()

    def run(self, python, module, env, log_path, request):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self.stop()
                self._start(python, module, env, log_path)
            try:
                self.process.stdin.write(json.dumps({'request': str(request)}) + '\n')
                self.process.stdin.flush()
                response = self.responses.get(timeout=900)
                if response is None:
                    raise RuntimeError('Процесс Qwen TTS завершился. Диагностика: ' + str(log_path))
                result = json.loads(response)
                if not result.get('ok'):
                    raise RuntimeError('Ошибка синтеза: ' + result.get('error', 'неизвестная ошибка') +
                                       '. Диагностика: ' + str(log_path))
            except (BrokenPipeError, queue.Empty, json.JSONDecodeError):
                self.stop()
                raise RuntimeError('Синтез не завершился. Диагностика: ' + str(log_path))


_worker = _Worker()
atexit.register(_worker.stop)


def stop_tts_worker():
    """Release GPU memory before loading ASR."""
    _worker.stop()


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
            _worker.run(python, self.worker_module, env, ROOT / 'logs' / self.worker_log, request)
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
