"""Optional local Nemotron ASR via the NeMo-Speech.cpp HTTP runtime.

The application code is Apache-2.0; NVIDIA model weights are OpenMDW 1.1.
"""
from .paths import ROOT, configure
configure()

import io
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.error
import urllib.request
import wave

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
MANIFEST = ROOT / "models/nemotron-asr.json"
RUNTIME = ROOT / "tools/nemo-speech/bin/nemo-speech.exe"


class NemotronEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.process = None
        self.port = None
        self.state = "not_loaded"
        self.device = None
        self._log = None

    def unload(self):
        with self.lock:
            if self.process is not None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
                self.process = None
            if self._log is not None:
                self._log.close()
                self._log = None
            self.port = None
            self.state = "not_loaded"

    def load(self):
        if self.process is not None and self.process.poll() is None:
            return
        if not RUNTIME.is_file() or not MANIFEST.is_file():
            raise RuntimeError("Nemotron не установлен. Запусти scripts/setup-nemotron.ps1")
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        model = Path(manifest["model_path"])
        if not model.is_file():
            raise RuntimeError("Не найдены веса Nemotron. Запусти scripts/setup-nemotron.ps1")
        import socket
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.device = os.getenv("NEMOTRON_DEVICE", "cuda:0")
        self.state = "loading"
        self._log = (ROOT / "logs/nemotron-server.log").open("ab")
        command = [str(RUNTIME), "serve", "--host", "127.0.0.1", "--port", str(self.port),
                   "--asr-model", str(model), "--device", self.device, "--no-ui"]
        self.process = subprocess.Popen(command, cwd=str(ROOT / "tmp"), stdout=self._log,
                                        stderr=subprocess.STDOUT,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.state = "error"
                raise RuntimeError("Nemotron не запустился. См. D:\\Games\\STT_TTS_project\\logs\\nemotron-server.log")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/ready", timeout=2) as response:
                    if json.load(response).get("ready"):
                        self.state = "ready"
                        return
            except (OSError, ValueError):
                pass
            time.sleep(0.5)
        self.process.terminate()
        self.process.wait(timeout=10)
        self.process = None
        self._log.close()
        self._log = None
        self.state = "error"
        raise RuntimeError("Nemotron не успел загрузиться за 90 секунд")

    def transcribe(self, audio_path: Path, context=""):
        from .engine import decode_audio
        return self.transcribe_pcm(decode_audio(audio_path), context=context)

    def transcribe_pcm(self, samples, context=""):
        if context:
            raise ValueError("Контекстные подсказки для Nemotron пока не поддерживаются")
        import numpy as np
        with self.lock:
            begin = time.perf_counter()
            self.load()
            pcm = np.clip(samples, -1, 1)
            pcm = (pcm * 32767).astype("<i2", copy=False)
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(pcm.tobytes())
            boundary = "stt-tts-nemotron-boundary"
            fields = (("language", "auto"), ("response_format", "verbose_json"))
            body = bytearray()
            for key, value in fields:
                body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode())
            body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode())
            body.extend(buffer.getvalue())
            body.extend(f"\r\n--{boundary}--\r\n".encode())
            request = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/audio/transcriptions",
                                             data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    result = json.load(response)
            except urllib.error.HTTPError as exc:
                raise RuntimeError(f"Nemotron HTTP {exc.code}: {exc.read(1000).decode('utf-8', 'replace')}") from exc
            text = result.get("text", "").strip()
            language = result.get("language", "")
            return {"text": text, "language": language, "model": MODEL_ID,
                    "device": self.device, "language_mode": "auto",
                    "duration_seconds": round(len(samples) / 16000, 3),
                    "processing_seconds": round(time.perf_counter() - begin, 3),
                    "segments": [{"start_seconds": 0, "text": text, "language": language}],
                    "timestamp_type": "whole_utterance", "scope": "audio_only", "simulated": False}
