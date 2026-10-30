"""SPDX-License-Identifier: Apache-2.0"""
from ..config import ROOT, configure
configure()

import json
import os
import threading
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-ASR-0.6B"


def decode_audio(audio_path: Path):
    """Decode uploaded audio/video to mono 16 kHz with the shared 10-minute limit."""
    import av
    import numpy as np
    chunks, count = [], 0
    try:
        with av.open(str(audio_path), options={"protocol_whitelist": "file,pipe"}) as container:
            streams = [s for s in container.streams if s.type == "audio"]
            if not streams:
                raise ValueError("В файле нет звуковой дорожки")
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
            for frame in container.decode(streams[0]):
                for converted in resampler.resample(frame):
                    samples = converted.to_ndarray().reshape(-1)
                    count += samples.size
                    if count > 600 * 16000:
                        raise ValueError("Максимум 10 минут аудио в одном запросе")
                    chunks.append(samples)
            for converted in resampler.resample(None):
                samples = converted.to_ndarray().reshape(-1)
                count += samples.size
                if count > 600 * 16000:
                    raise ValueError("Максимум 10 минут аудио в одном запросе")
                chunks.append(samples)
    except av.FFmpegError as exc:
        raise ValueError("Не удалось декодировать аудио/видео") from exc
    if not count:
        raise ValueError("Пустое аудио")
    return np.concatenate(chunks).astype(np.float32, copy=False)


class Engine:
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()
        self.device = None
        self.last_error = None
        self.state = "not_loaded"

    def unload(self):
        """Release ASR weights before running TTS on a small GPU."""
        with self.lock:
            if self.model is None:
                return
            import gc
            import torch
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.state = "not_loaded"

    def warmup(self):
        """Load weights ahead of the first utterance without racing inference."""
        with self.lock:
            self.load()

    def load(self):
        if self.model is not None:
            return
        import torch
        from qwen_asr import Qwen3ASRModel
        manifest_path = ROOT / "models/asr.json"
        if not manifest_path.exists():
            raise RuntimeError("Model not prepared. Run .\\scripts\\download-model.ps1 first.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_path = Path(manifest["snapshot"])
        if not model_path.is_dir():
            raise RuntimeError("Model snapshot does not exist; run scripts/download-model.ps1")
        requested = os.getenv("ASR_DEVICE", "auto")
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Set ASR_DEVICE=cpu, or install CUDA PyTorch.")
        self.state = "loading"
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        try:
            self.model = Qwen3ASRModel.from_pretrained(
                str(model_path), dtype=dtype, device_map=self.device,
                attn_implementation="sdpa", max_inference_batch_size=1, max_new_tokens=512,
                local_files_only=True,
            )
            self.state = "ready"
            self.last_error = None
        except Exception as exc:
            self.state = "error"
            self.last_error = str(exc)
            raise

    def transcribe(self, audio_path: Path, context=""):
        return self.transcribe_pcm(decode_audio(audio_path), context=context)

    def transcribe_pcm(self, samples, context=""):
        import torch

        # Single GPU worker. Lock must stay held until the actual thread ends.
        with self.lock:
            self.load()
            begin = time.perf_counter()
            count = samples.size
            from qwen_asr.inference.utils import split_audio_into_chunks
            pieces = split_audio_into_chunks(wav=samples, sr=16000, max_chunk_sec=25)
            results, segments = [], []
            with torch.inference_mode():
                for part, offset in pieces:
                    # Preserve the source language. Never force a target language.
                    part_results = self.model.transcribe(audio=(part, 16000), language=None, context=context)
                    results.extend(part_results)
                    segments.append({"start_seconds": round(offset, 3),
                                     "text": "\n".join(r.text for r in part_results),
                                     "language": ", ".join(dict.fromkeys(r.language for r in part_results))})
            elapsed = time.perf_counter() - begin
            return {
                "text": "\n".join(r.text for r in results),
                "language": ", ".join(dict.fromkeys(r.language for r in results)),
                "model": MODEL_ID, "device": self.device, "language_mode": "auto",
                "duration_seconds": round(count / 16000, 3),
                "processing_seconds": round(elapsed, 3),
                "segments": segments, "timestamp_type": "chunk_boundaries_not_word_alignment",
                "scope": "audio_only", "simulated": False,
            }
