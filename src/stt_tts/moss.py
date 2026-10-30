"""SPDX-License-Identifier: Apache-2.0. CPU-only adapter for official MOSS ONNX graphs."""
from .paths import ROOT, configure
configure()

import hashlib
import importlib.util
import io
import json
import os
import re
from pathlib import Path
import sys
import threading
import time

from .text_normalization import normalize_for_tts, VERSION
from .tts import validate
from .tts_worker import split_text
from .voices import VoiceStore, decode_reference

MODEL_ID = "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX"
SPEAKERS = ("Bella", "Ava", "Adam", "Nathan")
DEFAULT_SPEAKER = "Bella"


class MossEngine:
    def __init__(self):
        self.runtime = None
        self.tokenizer = None
        self.lock = threading.Lock()
        self.state = "not_loaded"
        self.voices = VoiceStore()

    def add_voice(self, path, name):
        waveform, audio, duration = decode_reference(path)
        with self.lock:
            self.load()
            import numpy as np
            session = self.runtime.sessions['codec_encode']
            outputs = session.run(None, {'waveform': waveform[None, ...],
                                        'input_lengths': np.asarray([waveform.shape[-1]], dtype=np.int32)})
            named = dict(zip([o.name for o in session.get_outputs()], outputs, strict=True))
            length = int(named['audio_code_lengths'].reshape(-1)[0])
            codes = named['audio_codes'][0, :length].astype(int).tolist()
            if not codes:
                raise RuntimeError('Не удалось подготовить образец голоса')
            return self.voices.save(name, codes, audio, duration)

    def load(self):
        if self.runtime is not None:
            return
        manifest_path = ROOT / "models/moss.json"
        if not manifest_path.exists():
            raise RuntimeError("MOSS не подготовлен. Запусти scripts/setup-moss.ps1")
        config = json.loads(manifest_path.read_text(encoding="utf-8"))
        provenance = config.get("runtime")
        if not provenance:
            raise RuntimeError("Повтори подготовку MOSS: python -m stt_tts.prepare_moss")
        source = Path(provenance["source"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != provenance["sha256"]:
            raise RuntimeError("Исходный runtime MOSS изменился относительно подготовленной версии")
        tts_dir, codec_dir = Path(config["tts"]["snapshot"]), Path(config["codec"]["snapshot"])
        runtime_manifest = json.loads((tts_dir / "browser_poc_manifest.json").read_text(encoding="utf-8"))
        runtime_manifest["model_files"] = {
            "tts_meta": str(tts_dir / "tts_browser_onnx_meta.json"),
            "codec_meta": str(codec_dir / "codec_browser_onnx_meta.json"),
            "tokenizer_model": str(tts_dir / "tokenizer.model"),
        }
        runtime_dir = ROOT / "models/moss-runtime"
        runtime_dir.mkdir(exist_ok=True)
        (runtime_dir / "browser_poc_manifest.json").write_text(json.dumps(runtime_manifest), encoding="utf-8")
        self.state = "loading"
        try:
            spec = importlib.util.spec_from_file_location("_stt_moss_ort_cpu", source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            import sentencepiece as spm
            runtime = module.OrtCpuRuntime(runtime_dir, thread_count=int(os.getenv("MOSS_CPU_THREADS", "4")),
                                           max_new_frames=500, sample_mode="fixed", execution_provider="cpu")
            tokenizer = spm.SentencePieceProcessor(model_file=str(tts_dir / "tokenizer.model"))
            self.runtime, self.tokenizer = runtime, tokenizer
            self.state = "ready"
        except Exception:
            self.state = "error"
            raise

    def _pieces(self, text):
        # Keep the official 75-token budget as well as sentence boundaries.
        pieces = []
        sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
        for sentence in (piece for s in sentences for piece in split_text(s, limit=150)):
            current = ""
            for word in sentence.split():
                candidate = (current + " " + word).strip()
                if current and len(self.tokenizer.encode(candidate)) > 75:
                    pieces.append(current)
                    current = word
                else:
                    current = candidate
            if current:
                pieces.append(current)
        return pieces

    def synthesize(self, text, speaker=DEFAULT_SPEAKER):
        custom = self.voices.get(speaker) if isinstance(speaker, str) and speaker.startswith('clone-') else None
        text = validate(text, speaker, speakers=(*SPEAKERS, speaker) if custom else SPEAKERS)
        prepared = normalize_for_tts(text)
        with self.lock:
            begin = time.perf_counter()
            cold = self.runtime is None
            self.load()
            import numpy as np
            import soundfile as sf
            # Repeatable comparisons across cold/warm runs, matching upstream's seed.
            self.runtime.rng = np.random.default_rng(1234)
            codes = custom['codes'] if custom else next(v for v in self.runtime.list_builtin_voices() if v["voice"] == speaker)['prompt_audio_codes']
            rate = int(self.runtime.codec_meta["codec_config"]["sample_rate"])
            channels = int(self.runtime.codec_meta["codec_config"]["channels"])
            pieces = self._pieces(prepared)
            waves = []
            for piece in pieces:
                if time.perf_counter() - begin > 900:
                    raise RuntimeError("MOSS превысил 15 минут. Сократи текст.")
                ids = self.tokenizer.encode(piece, out_type=int)
                rows = self.runtime.build_voice_clone_request_rows(codes, ids)
                frames = self.runtime.generate_audio_frames(rows)
                if not frames or len(frames) >= 500:
                    raise RuntimeError("MOSS не завершил фрагмент речи. Попробуй более короткий текст или другой голос.")
                decoded, length = self.runtime.decode_full_audio(frames)
                audio = np.stack(decoded, axis=1).astype(np.float32, copy=False)
                if not length or not np.isfinite(audio).all():
                    raise RuntimeError("MOSS вернул некорректное аудио")
                if waves:
                    waves.append(np.zeros((int(rate * .2), channels), dtype=np.float32))
                waves.append(audio)
            audio = np.concatenate(waves, axis=0)
            output = io.BytesIO()
            sf.write(output, audio, rate, format="WAV", subtype="PCM_16")
            metadata = {"model": MODEL_ID, "speaker": speaker, "device": "cpu",
                        "sample_rate": rate, "channels": channels, "chunks": len(pieces),
                        "duration_seconds": round(len(audio) / rate, 3),
                        "processing_seconds": round(time.perf_counter() - begin, 3),
                        "cold_start": cold, "text_normalization": VERSION,
                        "normalization_applied": prepared != text, "simulated": False}
            metadata['voice_name'] = custom['name'] if custom else speaker
            metadata['cloned_voice'] = custom is not None
            return output.getvalue(), metadata
