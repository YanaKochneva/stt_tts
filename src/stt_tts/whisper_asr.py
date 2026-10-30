"""SPDX-License-Identifier: Apache-2.0. Optional Russian Whisper ASR."""
from .paths import ROOT, configure
configure()

import gc
import json
import os
import threading
import time
from pathlib import Path

MODEL_ID = "coriollon/whisper-large-v3-turbo-russian"
MANIFEST = ROOT / "models/whisper-asr.json"


class WhisperEngine:
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()
        self.state = "not_loaded"
        self.device = None

    def unload(self):
        with self.lock:
            self.model = None
            gc.collect()
            self.state = "not_loaded"

    def load(self):
        if self.model is not None:
            return
        if not MANIFEST.is_file():
            raise RuntimeError("Whisper не установлен. Запусти scripts/install-whisper.ps1")
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        path = Path(manifest["snapshot"]) / "ct2_int8_float16"
        if not (path / "model.bin").is_file():
            raise RuntimeError("Файлы Whisper не найдены. Запусти scripts/install-whisper.ps1")
        from faster_whisper import WhisperModel
        import torch
        requested = os.getenv("ASR_DEVICE", "auto")
        self.device = "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested.split(":")[0])
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна; установи ASR_DEVICE=cpu")
        self.state = "loading"
        try:
            self.model = WhisperModel(str(path), device=self.device,
                                      compute_type="int8_float16" if self.device == "cuda" else "int8",
                                      local_files_only=True)
            self.state = "ready"
        except Exception:
            self.state = "error"
            raise

    def transcribe(self, audio_path: Path, context=""):
        if context:
            raise ValueError("Контекстные подсказки для Whisper пока не поддерживаются")
        from .engine import decode_audio
        with self.lock:
            self.load()
            begin = time.perf_counter()
            audio = decode_audio(audio_path)
            # Russian-specific checkpoint: transcribe source speech, never translate.
            pieces, info = self.model.transcribe(audio, language="ru", task="transcribe",
                                                   beam_size=10, patience=1.2, vad_filter=False)
            segments = []
            for part in pieces:
                segments.append({"start_seconds": round(part.start, 3),
                                 "end_seconds": round(part.end, 3),
                                 "text": part.text.strip(), "language": "Russian"})
            return {"text": " ".join(s["text"] for s in segments).strip(),
                    "language": "Russian", "model": MODEL_ID, "device": self.device,
                    "language_mode": "ru_source_transcription",
                    "duration_seconds": round(len(audio) / 16000, 3),
                    "processing_seconds": round(time.perf_counter() - begin, 3),
                    "segments": segments, "timestamp_type": "whisper_segments",
                    "scope": "audio_only", "simulated": False}
