"""SPDX-License-Identifier: Apache-2.0. Runs only in the separate TTS environment."""
from .paths import ROOT, configure
configure()

import json
import os
from pathlib import Path
import re
import sys
import textwrap


def split_text(text, limit=180):
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?。！？])\s+|\n+", text):
        for part in textwrap.wrap(sentence, width=limit, break_long_words=True, break_on_hyphens=False):
            if current and len(current) + len(part) + 1 > limit:
                pieces.append(current)
                current = ""
            current = (current + " " + part).strip()
    if current:
        pieces.append(current)
    return pieces


def main():
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    from .tts import MODEL_ID, CLONE_MODEL_ID, validate, validate_reference_text
    from .text_normalization import normalize_for_tts, VERSION

    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    clone = request.get('mode') == 'clone'
    text = validate(request['text'], request['speaker'], **({'speakers': (request['speaker'],)} if clone else {}))
    prepared_text = normalize_for_tts(text)
    manifest = json.loads((ROOT / 'models' / ('qwen-clone.json' if clone else 'tts.json')).read_text(encoding='utf-8'))
    requested = os.getenv("TTS_DEVICE", "auto")
    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.startswith("cuda") else torch.float32
    model = Qwen3TTSModel.from_pretrained(manifest["snapshot"], device_map=device, dtype=dtype,
                                          attn_implementation="sdpa", local_files_only=True)
    chunks, rate = [], None
    pieces = split_text(prepared_text)
    with torch.inference_mode():
        prompt = None
        if clone:
            ref_text = validate_reference_text(request.get('ref_text', ''))
            prompt = model.create_voice_clone_prompt(ref_audio=request['reference_audio'],
                                                     ref_text=ref_text or None,
                                                     x_vector_only_mode=not bool(ref_text))
        for piece in pieces:
            if clone:
                waves, sample_rate = model.generate_voice_clone(
                    text=piece, language='Auto', voice_clone_prompt=prompt,
                    non_streaming_mode=True, max_new_tokens=768)
            else:
                waves, sample_rate = model.generate_custom_voice(
                    text=piece, language='Auto', speaker=request['speaker'],
                    non_streaming_mode=True, max_new_tokens=768)
            wave = np.asarray(waves[0], dtype=np.float32).reshape(-1)
            if not wave.size or not np.isfinite(wave).all() or len(wave) / sample_rate >= 63:
                raise RuntimeError("TTS returned invalid or excessively long audio")
            if rate is not None and rate != sample_rate:
                raise RuntimeError("Inconsistent sample rates")
            if chunks:
                chunks.append(np.zeros(int(sample_rate * 0.15), dtype=np.float32))
            chunks.append(wave)
            rate = sample_rate
    if not chunks:
        raise ValueError("No text to synthesize")
    audio = np.concatenate(chunks)
    sf.write(str(request_path.parent / "output.wav"), audio, rate, subtype="PCM_16")
    metadata = {"model": CLONE_MODEL_ID if clone else MODEL_ID, "speaker": request["speaker"], "device": device,
                "duration_seconds": round(len(audio) / rate, 3), "sample_rate": rate,
                "chunks": len(pieces), "simulated": False,
                "text_normalization": VERSION, "normalization_applied": prepared_text != text,
                "cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device.startswith("cuda") else None}
    if clone:
        metadata['clone_mode'] = 'reference_text' if ref_text else 'speaker_embedding'
    (request_path.parent / "result.json").write_text(json.dumps(metadata), encoding="utf-8")


if __name__ == "__main__":
    main()
