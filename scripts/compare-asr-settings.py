"""SPDX-License-Identifier: Apache-2.0. Compare decoding on one authorized sample."""
import json
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

from stt_tts.engine import decode_audio, Engine
from stt_tts.whisper_asr import WhisperEngine
from stt_tts.paths import ROOT

REFERENCE = ("Информационная справка по транспортному происшествию от двадцать "
             "седьмого точка десятого точка две тысячи двадцать шесть. Краткое "
             "описание обстоятельств и последствий: водитель автомобиля Ютонг "
             "при заезде в ремонтный бокс задним ходом не выдержал боковой "
             "интервал, допустил наезд на арку въездных ворот. Механические "
             "повреждения ТС - повреждено лкп заднего правого угла, трещины "
             "корпуса правого стоп сигнала.")


def words(text):
    return re.findall(r"[\w]+", text.lower().replace("ё", "е"), re.UNICODE)


def score(hypothesis):
    expected, actual = words(REFERENCE), words(hypothesis)
    previous = list(range(len(actual) + 1))
    for i, token in enumerate(expected, 1):
        current = [i]
        for j, other in enumerate(actual, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (token != other)))
        previous = current
    operations = SequenceMatcher(None, expected, actual, autojunk=False).get_opcodes()
    return round(previous[-1] / len(expected), 4), [
        {"expected": " ".join(expected[a1:a2]), "actual": " ".join(actual[b1:b2])}
        for op,a1,a2,b1,b2 in operations if op != "equal"]


def main():
    path = Path(sys.argv[1])
    audio = decode_audio(path)
    model = WhisperEngine()
    model.load()
    variants = {
        "current": dict(beam_size=5, vad_filter=False),
        "greedy": dict(beam_size=1, best_of=1, vad_filter=False),
        "beam10": dict(beam_size=10, patience=1.2, vad_filter=False),
        "beam15": dict(beam_size=15, patience=1.2, vad_filter=False),
        "beam10_hotwords": dict(beam_size=10, patience=1.2, hotwords="Ютонг, въездные ворота, ЛКП, стоп-сигнал"),
        "beam10_prompt": dict(beam_size=10, patience=1.2, initial_prompt="Ютонг. Въездные ворота. ЛКП. Стоп-сигнал."),
        "vad": dict(beam_size=5, vad_filter=True, vad_parameters={"min_silence_duration_ms": 500}),
        "no_previous": dict(beam_size=5, condition_on_previous_text=False),
        "hotwords": dict(beam_size=5, hotwords="Ютонг, ремонтный бокс, боковой интервал, ЛКП, стоп-сигнал"),
        "prompt": dict(beam_size=5, initial_prompt="Ютонг. ЛКП. ТС. Стоп-сигнал. Ремонтный бокс."),
    }
    output = {"source": str(path), "reference": REFERENCE, "duration_seconds": round(len(audio)/16000, 3), "variants": {}}
    for name, options in variants.items():
        start = time.perf_counter()
        segments, _ = model.model.transcribe(audio, language="ru", task="transcribe", **options)
        text = " ".join(s.text.strip() for s in segments).strip()
        wer, changes = score(text)
        output["variants"][name] = {"settings": options, "text": text, "word_error_rate": wer,
                                     "differences": changes, "seconds": round(time.perf_counter()-start, 3)}
        print(name, wer, round(time.perf_counter()-start, 3), flush=True)
    model.unload()
    qwen = Engine().transcribe(path)
    wer, changes = score(qwen["text"])
    output["qwen"] = {"text": qwen["text"], "word_error_rate": wer, "differences": changes,
                      "seconds": qwen["processing_seconds"]}
    destination = ROOT / "results/asr-settings-academy-20261029.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("qwen", wer, qwen["processing_seconds"])
    print(destination)


if __name__ == "__main__":
    main()
