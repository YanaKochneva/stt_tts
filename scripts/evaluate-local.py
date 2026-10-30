"""Resumable local speech evaluation. All generated data stays on D:.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import asyncio
import html
import json
import os
import shutil
import subprocess
import time
import wave
from pathlib import Path

import requests
import aiohttp
import numpy as np

from stt_tts.evaluation import adjacent_repetitions, error_rates, words
from stt_tts.engine import decode_audio
from stt_tts.paths import ROOT
from stt_tts.text_normalization import normalize_for_tts

BASE = ROOT / "evaluation/2026-10-29"
API = "http://127.0.0.1:8090"
VOICE_ID = "clone-c196b954a6474f1ea019de96cbd7a59e"
REFERENCE_VOICE = ROOT / "voices" / f"{VOICE_ID}.wav"
PROMPTS = {
    "neutral": "Здравствуйте. Я расскажу о работе нашей команды и отвечу на ваши вопросы.",
    "stress": "Встреча состоится 24 и 25 октября в 7:00. Температура +10−12 °С.",
    "long": "Сегодня мы проверяем качество синтеза речи. Система должна произнести каждое слово ровно один раз. После короткой паузы она продолжит фразу без повторов, пропусков и перестановок.",
}
VARIANTS = {
    "qwen_custom": {"label": "Qwen CustomVoice · Serena", "engine": "qwen", "speaker": "Serena", "clone": False},
    "moss_builtin": {"label": "MOSS · Bella", "engine": "moss", "speaker": "Bella", "clone": False},
    "moss_clone": {"label": "MOSS · Миша 2", "engine": "moss", "speaker": VOICE_ID, "clone": True},
    "qwen_clone": {"label": "Qwen Base · Миша 2", "engine": "qwen_clone", "speaker": VOICE_ID, "clone": True},
    "cosyvoice": {"label": "CosyVoice 3 · Миша 2", "engine": "cosyvoice", "speaker": VOICE_ID, "clone": True},
    "firered": {"label": "FireRedTTS3 · Миша 2 (эксперимент)", "engine": "firered", "speaker": VOICE_ID, "clone": True},
}
SOURCE_RU = "Привет. Это проверка распознавания русской речи. Сегодня мы создаем локальный проект. Все временные файлы хранятся на диске д."


def load(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def duration(path):
    with wave.open(str(path), "rb") as wav:
        return round(wav.getnframes() / wav.getframerate(), 3)


def reference_text():
    draft = BASE / "misha2-reference-draft.json"
    if not draft.is_file():
        raise RuntimeError("Сначала сохрани черновик расшифровки Миша 2")
    return load(draft, {})["text"].replace("\n", " ")


def asr_cases():
    academy = load(ROOT / "results/asr-settings-academy-20261029.json", {})
    return {
        "spoken_prompt": {"path": str(ROOT / "data/test-ru.wav"), "reference": SOURCE_RU,
                          "reference_status": "known_synthetic_source", "engines": ["qwen", "whisper"]},
        "video_audio": {"path": str(ROOT / "data/test-video.mp4"), "reference": " ".join([SOURCE_RU] * 3),
                        "reference_status": "known_synthetic_source", "engines": ["qwen", "whisper"]},
        "user_recording": {"path": r"C:\Users\andre\Downloads\Академия Печати 3.m4a",
                           "reference": academy.get("reference", ""),
                           "reference_status": "provisional_user_decoding", "engines": ["qwen", "whisper"]},
        "english_audio": {"path": str(ROOT / "data/test-en-source-language.wav"),
                          "reference": "", "reference_status": "missing", "engines": ["qwen"]},
    }


def post_asr(path, engine):
    with Path(path).open("rb") as source:
        start = time.perf_counter()
        response = requests.post(API + "/api/transcribe", files={"file": (Path(path).name, source)},
                                 data={"engine": engine}, timeout=300)
        elapsed = time.perf_counter() - start
    response.raise_for_status()
    result = response.json()
    result["response_latency_seconds"] = round(elapsed, 3)
    result["real_time_factor"] = round(elapsed / result["duration_seconds"], 3)
    return result


def run_asr():
    path = BASE / "asr-results.json"
    data = load(path, {"cases": {}, "results": {}})
    data["cases"] = asr_cases()
    for case_id, case in data["cases"].items():
        for engine in case["engines"]:
            key = f"{case_id}:{engine}"
            if key in data["results"] and "error" not in data["results"][key]:
                continue
            print("ASR", key, flush=True)
            try:
                result = post_asr(case["path"], engine)
                if case["reference"]:
                    result["scores"] = error_rates(case["reference"], result["text"])
                data["results"][key] = result
            except Exception as exc:
                data["results"][key] = {"error": str(exc)}
            save(path, data)


async def run_live_asr():
    path = BASE / "asr-results.json"
    data = load(path, {"cases": {}, "results": {}})
    data["cases"] = asr_cases()
    key = "spoken_prompt:qwen_stream"
    if key in data["results"] and "error" not in data["results"][key]:
        return
    samples = decode_audio(Path(data["cases"]["spoken_prompt"]["path"]))
    print("ASR", key, flush=True)
    try:
        async with aiohttp.ClientSession() as client:
            async with client.ws_connect(API + "/api/stream", heartbeat=15) as ws:
                await ws.send_json({"type": "start", "sample_rate": 16000})
                ready = await ws.receive_json()
                if ready.get("type") != "ready":
                    raise RuntimeError(ready)
                started = time.perf_counter()
                result = {"model": ready["model"], "device": "server", "duration_seconds": round(len(samples)/16000,3)}

                async def producer():
                    for offset in range(0, len(samples), 3200):
                        part = np.clip(samples[offset:offset+3200], -1, 1)
                        await ws.send_bytes((part * 32767).astype("<i2").tobytes())
                        await asyncio.sleep(len(part)/16000)
                    result["audio_sent_seconds"] = round(time.perf_counter()-started,3)
                    await ws.send_json({"type": "stop"})

                task = asyncio.create_task(producer())
                async for message in ws:
                    if message.type != aiohttp.WSMsgType.TEXT:
                        break
                    item = json.loads(message.data)
                    if item["type"] == "partial" and "first_partial_seconds" not in result:
                        result["first_partial_seconds"] = round(time.perf_counter()-started,3)
                    if item["type"] == "final":
                        result["text"] = item["text"]
                        result["response_latency_seconds"] = round(time.perf_counter()-started,3)
                        result["processing_seconds"] = item["processing_seconds"]
                    if item["type"] == "error":
                        raise RuntimeError(item["error"])
                    if item["type"] == "stopped":
                        break
                await task
        result["final_after_audio_seconds"] = round(result["response_latency_seconds"] - result["audio_sent_seconds"],3)
        result["real_time_factor"] = round(result["response_latency_seconds"] / result["duration_seconds"],3)
        result["scores"] = error_rates(SOURCE_RU, result["text"])
        data["results"][key] = result
    except Exception as exc:
        data["results"][key] = {"error": str(exc)}
    save(path, data)


def run_firered(prompt, output, ref_text):
    root = ROOT
    binary = root / "third_party/crispasr/crispasr-windows-x86_64-cpu/crispasr.exe"
    model = root / "models/firered-gguf/fireredtts3-base-q4_k.gguf"
    codec = root / "models/firered-gguf/fireredtts3-redae-f16.gguf"
    command = [str(binary), "--backend", "fireredtts3", "-m", str(model), "--codec-model", str(codec),
               "--voice", str(REFERENCE_VOICE), "--ref-text", ref_text, "--i-have-rights",
               "--tts", normalize_for_tts(prompt), "--tts-output", str(output)]
    start = time.perf_counter()
    run = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                         timeout=1200, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if run.returncode:
        raise RuntimeError(f"FireRed exit {run.returncode}: {run.stderr[-1800:]}")
    return {"response_latency_seconds": round(time.perf_counter()-start, 3),
            "model": "FireRedTeam/FireRedTTS3-Base GGUF Q4", "device": "cpu",
            "duration_seconds": duration(output), "disclosure_seconds": 4.0,
            "runtime_license": "MIT (CrispASR); model Apache-2.0"}


def run_tts(only=None):
    path = BASE / "tts-results.json"
    data = load(path, {"prompts": PROMPTS, "variants": VARIANTS, "results": {},
                       "reference_voice": str(REFERENCE_VOICE),
                       "reference_text_status": "qwen_draft_unverified"})
    data.update(prompts=PROMPTS, variants=VARIANTS)
    audio_dir = BASE / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REFERENCE_VOICE, audio_dir / "misha2-reference.wav")
    ref_text = reference_text()
    for variant_id, variant in VARIANTS.items():
        if only and variant_id not in only:
            continue
        for prompt_id, prompt in PROMPTS.items():
            key = f"{variant_id}:{prompt_id}"
            output = audio_dir / f"{variant_id}-{prompt_id}.wav"
            if output.is_file() and key in data["results"] and "error" not in data["results"][key]:
                continue
            print("TTS", key, flush=True)
            try:
                if variant["engine"] == "firered":
                    meta = run_firered(prompt, output, ref_text)
                else:
                    payload = {"engine": variant["engine"], "speaker": variant["speaker"], "text": prompt}
                    if variant["engine"] in {"qwen_clone", "cosyvoice"}:
                        payload["ref_text"] = ref_text
                    start = time.perf_counter()
                    response = requests.post(API + "/api/tts", json=payload, timeout=950)
                    elapsed = time.perf_counter() - start
                    if not response.ok:
                        try:
                            reason = response.json().get("error", response.text[:500])
                        except ValueError:
                            reason = response.text[:500]
                        raise RuntimeError(f"HTTP {response.status_code}: {reason}")
                    output.write_bytes(response.content)
                    meta = json.loads(response.headers.get("X-TTS-Metadata", "{}"))
                    meta["response_latency_seconds"] = round(elapsed, 3)
                meta["audio_file"] = str(output.relative_to(BASE)).replace("\\", "/")
                meta["spoken_reference"] = normalize_for_tts(prompt)
                meta["real_time_factor"] = round(meta["response_latency_seconds"] / meta["duration_seconds"], 3)
                effective = max(0.1, meta["duration_seconds"] - meta.get("disclosure_seconds", 0))
                meta["expected_words_per_minute"] = round(len(words(meta["spoken_reference"])) * 60 / effective, 1)
                data["results"][key] = meta
            except Exception as exc:
                data["results"][key] = {"error": str(exc)}
            save(path, data)


def score_tts(only=None):
    path = BASE / "tts-results.json"
    data = load(path, {})
    for key, result in data.get("results", {}).items():
        if "audio_file" not in result or (only and key.split(":")[0] not in only):
            continue
        audio_path = BASE / result["audio_file"]
        scoring_path = audio_path
        if result.get("disclosure_seconds"):
            scoring_dir = BASE / "scoring"
            scoring_dir.mkdir(exist_ok=True)
            scoring_path = scoring_dir / audio_path.name
            if not scoring_path.is_file():
                with wave.open(str(audio_path), "rb") as source:
                    params = source.getparams()
                    source.setpos(min(source.getnframes(), round(result["disclosure_seconds"] * source.getframerate())))
                    speech = source.readframes(source.getnframes() - source.tell())
                with wave.open(str(scoring_path), "wb") as target:
                    target.setparams(params)
                    target.writeframes(speech)
        for judge in ("qwen", "whisper"):
            if judge in result.get("roundtrip", {}):
                continue
            print("ROUNDTRIP", key, judge, flush=True)
            try:
                transcript = post_asr(scoring_path, judge)
                heard = transcript["text"]
                transcript["scores"] = error_rates(result["spoken_reference"], heard)
                expected = adjacent_repetitions(result["spoken_reference"])
                observed = adjacent_repetitions(heard)
                transcript["adjacent_repetitions"] = observed
                transcript["new_repetition_events"] = max(0, len(observed)-len(expected))
                result.setdefault("roundtrip", {})[judge] = transcript
            except Exception as exc:
                result.setdefault("roundtrip", {})[judge] = {"error": str(exc)}
            save(path, data)


def report():
    asr = load(BASE / "asr-results.json", {"cases": {}, "results": {}})
    tts = load(BASE / "tts-results.json", {"prompts": PROMPTS, "variants": VARIANTS, "results": {}})
    h = html.escape
    rows_asr = []
    for key, result in asr.get("results", {}).items():
        case_id, engine = key.split(":")
        case = asr["cases"][case_id]
        score = result.get("scores", {})
        rows_asr.append(f"<tr><td>{h(case_id)}</td><td>{h(engine)}</td><td>{h(case['reference_status'])}</td>"
                        f"<td>{result.get('response_latency_seconds','—')}</td>"
                        f"<td>{score.get('cer','—')}</td><td>{score.get('wer','—')}</td>"
                        f"<td>{h(result.get('error',''))}</td></tr>")
    sections = []
    for variant_id, variant in VARIANTS.items():
        parts = [f"<section><h2>{h(variant['label'])}</h2>"]
        for prompt_id, prompt in PROMPTS.items():
            key = f"{variant_id}:{prompt_id}"
            result = tts.get("results", {}).get(key, {})
            parts.append(f"<article><h3>{h(prompt_id)}</h3><p class='prompt'>{h(prompt)}</p>")
            if "audio_file" in result:
                src = h(result["audio_file"])
                judges = result.get("roundtrip", {})
                scores = " · ".join(f"{judge}: CER {r.get('scores',{}).get('cer','—')}, WER {r.get('scores',{}).get('wer','—')}, повторы {r.get('new_repetition_events','—')}"
                                    for judge, r in judges.items())
                parts.append(f"<audio controls preload='none' src='{src}'></audio>"
                             f"<p class='metric'>Задержка {result['response_latency_seconds']} с · длительность {result['duration_seconds']} с · "
                             f"{result['expected_words_per_minute']} слов/мин · RTF {result['real_time_factor']}</p>"
                             f"<p class='metric'>{h(scores or 'Обратное ASR ещё не выполнено')}</p>")
                for label, field, opts in (("Сходство с образцом", "similarity", ["—", "1", "2", "3", "4", "5"]),
                                          ("Естественность", "naturalness", ["—", "1", "2", "3", "4", "5"]),
                                          ("Акцент", "accent", ["—", "нет", "слабый", "сильный"]),
                                          ("Темп", "speed", ["—", "медленно", "нормально", "быстро"]),
                                          ("Повторы на слух", "repetition", ["—", "нет", "есть"])):
                    if field == "similarity" and not variant["clone"]:
                        continue
                    options = "".join(f"<option>{h(option)}</option>" for option in opts)
                    parts.append(f"<label>{label}<select data-key='{h(key)}' data-field='{field}'>{options}</select></label>")
                parts.append(f"<label class='note'>Заметки<input data-key='{h(key)}' data-field='notes' placeholder='Что услышали?'></label>")
            else:
                parts.append(f"<p class='error'>{h(result.get('error','Ещё не сгенерировано'))}</p>")
            parts.append("</article>")
        parts.append("</section>")
        sections.append("".join(parts))
    page = f"""<!doctype html><html lang='ru'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Оценка речи — 29.10.2026</title><style>
body{{font:16px system-ui,sans-serif;max-width:1150px;margin:auto;padding:28px;background:#f6f9ff;color:#18334c}}
h1{{font-size:34px}}h2{{border-bottom:2px solid #bed4f0;padding-bottom:8px}}section{{margin:40px 0}}
article{{background:white;border:1px solid #cfdbec;border-radius:16px;padding:20px;margin:18px 0;box-shadow:0 4px 16px #dce6f477}}
audio{{display:block;width:min(100%,550px);margin:12px 0}}label{{display:inline-flex;flex-direction:column;gap:4px;margin:8px 12px 8px 0}}
select,input,button{{font:inherit;padding:8px;border:1px solid #a9bad2;border-radius:8px;background:white}}
button{{background:#bde3f9;cursor:pointer}}.note{{display:flex}}.note input{{width:min(90vw,650px)}}
.metric{{color:#4d6680;font-size:14px}}.prompt{{font-weight:600}}.error{{color:#9b2c39}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #c8d8e9;text-align:left;vertical-align:top}}
.scroll{{overflow:auto}}.warning{{background:#fff2cf;padding:14px;border-radius:10px}}
</style><h1>Оценка распознавания, синтеза и клонирования</h1>
<p class='warning'>CER/WER у синтеза — обратное распознавание другими моделями, а не прямая оценка качества аудио. Источник «Миша 2» расшифрован Qwen предварительно; проверьте образец на слух. FireRed работает вне веб-API через MIT runtime.</p>
<h2>Образец «Миша 2»</h2><audio controls src='audio/misha2-reference.wav'></audio>
<button id='save'>Скачать мои оценки JSON</button>
<h2>Распознавание: задержка и ошибки</h2><div class='scroll'><table><thead><tr><th>Сценарий</th><th>Модель</th><th>Эталон</th><th>Ответ, с</th><th>CER</th><th>WER</th><th>Ошибка</th></tr></thead><tbody>{''.join(rows_asr)}</tbody></table></div>
<p class='metric'>known_synthetic_source — известный текст синтетической записи; provisional_user_decoding — присланный ранее текст, не подтверждённый как дословный; missing — CER/WER не вычисляются.</p>
<h2>Синтез и клонирование: прослушивание</h2>{''.join(sections)}
<script>
let values={{}};
try{{values=JSON.parse(localStorage.getItem('speech-ratings-2026-10-29')||'{{}}')}}catch{{}}
document.querySelectorAll('[data-key]').forEach(el=>{{
 const saved=values[el.dataset.key]?.[el.dataset.field];if(saved!==undefined)el.value=saved;
 el.addEventListener('change',()=>{{
  const key=el.dataset.key;values[key]??={{}};values[key][el.dataset.field]=el.value;
  try{{localStorage.setItem('speech-ratings-2026-10-29',JSON.stringify(values))}}catch{{}}
 }});
}});
document.getElementById('save').onclick=()=>{{
 const blob=new Blob([JSON.stringify({{date:'2026-10-29',ratings:values}},null,2)],{{type:'application/json'}});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='my-speech-ratings.json';a.click();
 setTimeout(()=>URL.revokeObjectURL(a.href),1000);
}};
</script></html>"""
    target = BASE / "index.html"
    target.write_text(page, encoding="utf-8")
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["asr", "live", "tts", "score", "report", "all"])
    parser.add_argument("--only", nargs="*", choices=VARIANTS.keys())
    args = parser.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    if args.phase in {"asr", "all"}:
        run_asr()
    if args.phase in {"live", "all"}:
        asyncio.run(run_live_asr())
    if args.phase in {"tts", "all"}:
        run_tts(set(args.only) if args.only else None)
    if args.phase in {"score", "all"}:
        score_tts(set(args.only) if args.only else None)
    report()
