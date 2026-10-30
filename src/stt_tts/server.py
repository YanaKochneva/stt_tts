"""SPDX-License-Identifier: Apache-2.0"""
from .paths import ROOT, configure
configure()

import asyncio
import contextlib
import hmac
import json
import logging
import os
import tempfile
from pathlib import Path

from aiohttp import web
from .engine import Engine, MODEL_ID
from .tts import TTSEngine, MODEL_ID as TTS_MODEL_ID, SPEAKERS, DEFAULT_SPEAKER, MAX_TEXT, validate
from .voices import VoiceStore, decode_reference
from .tts import QwenCloneEngine, CLONE_MODEL_ID, validate_reference_text
from .streaming import AudioBuffer, MIN_SECONDS, SAMPLE_RATE

LOG = logging.getLogger("stt_tts")
WEB = Path(__file__).parent / "web"
ENGINE = web.AppKey("engine", Engine)
TTS = web.AppKey("tts", TTSEngine)
STATE = web.AppKey("state", dict)
TOKEN = web.AppKey("token", str)
VOICES = web.AppKey("voices", VoiceStore)
QWEN_CLONE = web.AppKey('qwen_clone', QwenCloneEngine)
WARMUP = web.AppKey('warmup', asyncio.Task)
LIMIT = 64 * 1024 * 1024


@web.middleware
async def errors(request, handler):
    if request.path.startswith("/api/"):
        origin = request.headers.get("Origin")
        if origin and origin not in {f"http://{request.host}", f"https://{request.host}"}:
            return web.json_response({"error": "Запрос с постороннего сайта запрещён"}, status=403)
        token = request.app[TOKEN]
        given = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if token and request.path != "/api/stream" and not hmac.compare_digest(token, given):
            return web.json_response({"error": "Нужен API-токен"}, status=401)
    try:
        return await handler(request)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except RuntimeError as exc:
        LOG.exception("Speech runtime error")
        message = str(exc)
        if "out of memory" in message.lower():
            message = "Недостаточно VRAM. Закрой другие GPU-программы или перезапусти с ASR_DEVICE=cpu / TTS_DEVICE=cpu."
        return web.json_response({"error": message}, status=503)


async def index(request):
    return web.FileResponse(WEB / "index.html")


async def app_js(request):
    return web.FileResponse(WEB / "app.js")


async def pcm_worklet(request):
    return web.FileResponse(WEB / "pcm-worklet.js")


async def health(request):
    engine = request.app[ENGINE]
    return web.json_response({"service": "stt_tts", "model": MODEL_ID,
                             "model_state": engine.state, "device": engine.device,
                             "asr_engines": {
                                 "qwen": {"model": MODEL_ID, "installed": (ROOT / "models/asr.json").exists(), "state": engine.state}},
                             "busy": request.app[STATE]["busy"], "data_root": str(ROOT),
                             "operation": request.app[STATE].get("operation"),
                             "tts": {"model": TTS_MODEL_ID, "speakers": SPEAKERS, "default_speaker": DEFAULT_SPEAKER, "max_text": MAX_TEXT,
                                     "installed": (ROOT / "models/tts.json").exists() and (ROOT / "envs/tts/Scripts/python.exe").exists(),
                                     "engines": {
                                         'qwen_clone': {'label': 'Qwen Base · клонирование · GPU', 'model': CLONE_MODEL_ID,
                                                        'speakers': [], 'default_speaker': '',
                                                        'installed': (ROOT / 'models/qwen-clone.json').exists() and (ROOT / 'envs/tts/Scripts/python.exe').exists()},
                                         "qwen": {"label": "Qwen3-TTS · GPU", "model": TTS_MODEL_ID,
                                                  "speakers": SPEAKERS, "default_speaker": DEFAULT_SPEAKER,
                                                  "installed": (ROOT / "models/tts.json").exists()},
                                     }},
                             "features": {"stt": True, "streaming_stt": True, "tts": True, "voice_cloning": True, "live_dialog": False},
                             "simulated": False})


async def transcribe(request):
    if request.app[STATE]["busy"]:
        return web.json_response({"error": "Модель занята. Повтори запрос после завершения текущего."}, status=429)
    request.app[STATE]["busy"] = True
    request.app[STATE]["operation"] = "stt"
    try:
        with tempfile.TemporaryDirectory(prefix="request-", dir=ROOT / "tmp") as directory:
            if request.content_type != "multipart/form-data":
                raise ValueError("Ожидается multipart/form-data с полем file")
            reader = await request.multipart()
            path, context, total, asr_engine = None, "", 0, "qwen"
            while (part := await reader.next()) is not None:
                if part.name == "file":
                    if path:
                        raise ValueError("Разрешён один файл")
                    path = Path(directory) / "input.media"
                    with path.open("wb") as output:
                        while chunk := await part.read_chunk():
                            total += len(chunk)
                            if total > LIMIT:
                                raise web.HTTPRequestEntityTooLarge(max_size=LIMIT, actual_size=total)
                            output.write(chunk)
                else:
                    raw = bytearray()
                    while chunk := await part.read_chunk():
                        raw.extend(chunk)
                        if len(raw) > 8192:
                            raise ValueError("Слишком длинное текстовое поле")
                    value = raw.decode("utf-8")
                    if part.name == "language":
                        if value.strip():
                            raise ValueError("Принудительный язык отключён: используется язык исходной речи. Обнови страницу и отправь файл заново.")
                    elif part.name == "context":
                        context = value
                    elif part.name == "engine":
                        asr_engine = value
            if not path or not total:
                raise ValueError("Выбери непустой аудио- или видеофайл")
            if asr_engine != "qwen":
                raise ValueError("Доступен только Qwen3-ASR")
            def run_asr():
                return request.app[ENGINE].transcribe(path, context=context)
            task = asyncio.create_task(asyncio.to_thread(run_asr))
            try:
                result = await asyncio.shield(task)
            except asyncio.CancelledError:
                # Do not delete temporary files while worker is reading them.
                with contextlib.suppress(Exception):
                    await task
                raise
            return web.json_response(result)
    finally:
        request.app[STATE]["busy"] = False
        request.app[STATE]["operation"] = None


async def stream_transcribe(request):
    """Receive 16 kHz mono PCM16 and emit interim/final utterance updates."""
    if request.app[STATE]["busy"]:
        return web.json_response({"error": "Сервис занят. Повтори запрос после завершения текущего."}, status=429)
    ws = web.WebSocketResponse(max_msg_size=SAMPLE_RATE * 2 + 1024, heartbeat=15)
    await ws.prepare(request)
    request.app[STATE].update(busy=True, operation="streaming_stt")
    pending = None
    buffer = AudioBuffer()
    last_requested = 0
    utterance_id = 0
    total_samples = 0
    try:
        try:
            first = await asyncio.wait_for(ws.receive(), timeout=5)
            if first.type != web.WSMsgType.TEXT:
                raise ValueError("Сначала отправь настройки потока")
            config = json.loads(first.data)
            if not isinstance(config, dict) or config.get("type") != "start":
                raise ValueError("Ожидаются настройки потока")
            token = request.app[TOKEN]
            if token and not hmac.compare_digest(token, str(config.get("token", ""))):
                await ws.send_json({"type": "error", "error": "Нужен API-токен"})
                return ws
            if config.get("sample_rate") != SAMPLE_RATE:
                raise ValueError("Ожидается моно PCM16 с частотой 16000 Гц")
            context = config.get("context", "")
            if not isinstance(context, str) or len(context) > 2000:
                raise ValueError("Слишком длинная контекстная подсказка")
            if config.get("engine", "qwen") != "qwen":
                raise ValueError("Живое распознавание поддерживает только Qwen")
            await ws.send_json({"type": "ready", "model": MODEL_ID, "sample_rate": SAMPLE_RATE})
        except (ValueError, json.JSONDecodeError, asyncio.TimeoutError) as exc:
            await ws.send_json({"type": "error", "error": str(exc)})
            return ws

        async def recognize(samples, final, phrase):
            result = await asyncio.to_thread(request.app[ENGINE].transcribe_pcm, samples, context)
            if not ws.closed:
                await ws.send_json({"type": "final" if final else "partial", "utterance_id": phrase,
                                    "text": result["text"], "language": result["language"],
                                    "audio_seconds": result["duration_seconds"],
                                    "processing_seconds": result["processing_seconds"]})

        while not ws.closed:
            try:
                message = await asyncio.wait_for(ws.receive(), timeout=45)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "error", "error": "Поток закрыт после 45 секунд без звука"})
                break
            if message.type == web.WSMsgType.BINARY:
                if len(message.data) > SAMPLE_RATE * 2:
                    raise ValueError("Аудиопорция должна быть не длиннее одной секунды")
                buffer.append_pcm16(message.data)
                total_samples += len(message.data) // 2
                if total_samples > 600 * SAMPLE_RATE:
                    raise ValueError("Максимум 10 минут аудио за сеанс")
                if pending is not None and pending.done():
                    await pending
                    pending = None
                if pending is None and buffer.seconds >= 2 and buffer.samples - last_requested >= 2 * SAMPLE_RATE:
                    last_requested = buffer.samples
                    pending = asyncio.create_task(recognize(buffer.snapshot(), False, utterance_id))
            elif message.type == web.WSMsgType.TEXT:
                command = json.loads(message.data)
                if not isinstance(command, dict) or command.get("type") not in {"flush", "stop"}:
                    raise ValueError("Неизвестная команда потока")
                if pending is not None:
                    await pending
                    pending = None
                if buffer.seconds >= MIN_SECONDS:
                    await recognize(buffer.snapshot(), True, utterance_id)
                    utterance_id += 1
                buffer.clear()
                last_requested = 0
                if command["type"] == "stop":
                    await ws.send_json({"type": "stopped"})
                    break
            elif message.type in {web.WSMsgType.CLOSE, web.WSMsgType.CLOSED, web.WSMsgType.ERROR}:
                break
    except (ValueError, json.JSONDecodeError, RuntimeError) as exc:
        if not ws.closed:
            await ws.send_json({"type": "error", "error": str(exc)})
    finally:
        if pending is not None:
            with contextlib.suppress(Exception):
                await pending
        request.app[STATE].update(busy=False, operation=None)
        await ws.close()
    return ws


async def synthesize(request):
    if request.app[STATE]["busy"]:
        return web.json_response({"error": "Сервис занят распознаванием или синтезом. Дождись завершения."}, status=429)
    request.app[STATE]["busy"] = True
    request.app[STATE]["operation"] = "tts"
    try:
        if request.content_type != "application/json":
            raise ValueError("Ожидается JSON с полями text и speaker")
        body = bytearray()
        async for chunk in request.content.iter_chunked(8192):
            body.extend(chunk)
            if len(body) > 65536:
                raise ValueError("Слишком большой запрос синтеза")
        data = json.loads(body)
        if not isinstance(data, dict):
            raise ValueError("Ожидается JSON-объект")
        engine_name = data.get("engine", "qwen")
        if not isinstance(engine_name, str) or engine_name not in {'qwen', 'qwen_clone'}:
            raise ValueError('Выбери Qwen с готовым или пользовательским голосом')
        speaker = data.get("speaker", DEFAULT_SPEAKER)
        allowed = () if engine_name == 'qwen_clone' else SPEAKERS
        if engine_name == 'qwen_clone' and isinstance(speaker, str) and speaker.startswith('clone-'):
            request.app[VOICES].get(speaker)
            allowed = (*allowed, speaker)
        text = validate(data.get("text"), speaker, speakers=allowed)
        ref_text = validate_reference_text(data.get('ref_text', '')) if engine_name == 'qwen_clone' else ''
        if engine_name == 'qwen_clone' and not ref_text:
            ref_text = validate_reference_text(request.app[VOICES].get(speaker).get('ref_text', ''))

        def run():
            request.app[ENGINE].unload()
            if engine_name == 'qwen_clone':
                return request.app[QWEN_CLONE].synthesize(text, speaker, ref_text=ref_text)
            return request.app[TTS].synthesize(text, speaker)

        task = asyncio.create_task(asyncio.to_thread(run))
        try:
            audio, metadata = await asyncio.shield(task)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await task
            raise
        metadata["engine"] = engine_name
        return web.Response(body=audio, content_type="audio/wav", headers={
            "Content-Disposition": 'attachment; filename="speech.wav"',
            "Cache-Control": "no-store", "X-TTS-Metadata": json.dumps(metadata),
        })
    finally:
        request.app[STATE]["busy"] = False
        request.app[STATE]["operation"] = None


async def voice_list(request):
    return web.json_response({'voices': request.app[VOICES].list()}, headers={'Cache-Control': 'no-store'})


async def voice_audio(request):
    key = request.match_info['key']
    request.app[VOICES].get(key)
    return web.FileResponse(request.app[VOICES].path(key, '.wav'), headers={'Cache-Control': 'no-store'})


async def voice_delete(request):
    if request.app[STATE]['busy']:
        return web.json_response({'error': 'Сервис занят. Дождись завершения.'}, status=429)
    request.app[VOICES].delete(request.match_info['key'])
    return web.json_response({'deleted': True})


async def voice_reference_text(request):
    if request.app[STATE]['busy']:
        return web.json_response({'error': 'Сервис занят. Дождись завершения.'}, status=429)
    if request.content_type != 'application/json':
        raise ValueError('Ожидается JSON с полем ref_text')
    data = await request.json()
    if not isinstance(data, dict):
        raise ValueError('Ожидается JSON-объект')
    text = validate_reference_text(data.get('ref_text', ''))
    return web.json_response(request.app[VOICES].set_reference_text(request.match_info['key'], text))


async def voice_add(request):
    if request.app[STATE]['busy']:
        return web.json_response({'error': 'Сервис занят. Дождись завершения.'}, status=429)
    request.app[STATE].update(busy=True, operation='voice')
    try:
        if request.content_type != 'multipart/form-data':
            raise ValueError('Ожидается файл образца и имя голоса')
        with tempfile.TemporaryDirectory(prefix='voice-', dir=ROOT / 'tmp') as directory:
            reader = await request.multipart()
            path, total, name = None, 0, ''
            while (part := await reader.next()) is not None:
                if part.name == 'file':
                    if path is not None:
                        raise ValueError('Разрешён один образец')
                    path = Path(directory) / 'reference.media'
                    with path.open('wb') as output:
                        while chunk := await part.read_chunk():
                            total += len(chunk)
                            if total > 16 * 1024 * 1024:
                                raise ValueError('Образец должен быть меньше 16 МиБ')
                            output.write(chunk)
                else:
                    raw = bytearray()
                    while chunk := await part.read_chunk():
                        raw.extend(chunk)
                        if len(raw) > 1024:
                            raise ValueError('Слишком длинное имя голоса')
                    if part.name == 'name':
                        name = raw.decode('utf-8').strip()
            if not name or len(name) > 60 or any(ord(c) < 32 for c in name):
                raise ValueError('Введи имя голоса от 1 до 60 символов')
            if path is None or not total:
                raise ValueError('Выбери непустой образец голоса')
            def save_reference():
                _, audio, duration = decode_reference(path)
                return request.app[VOICES].save(name, None, audio, duration)
            task = asyncio.create_task(asyncio.to_thread(save_reference))
            try:
                result = await asyncio.shield(task)
            except asyncio.CancelledError:
                with contextlib.suppress(Exception):
                    await task
                raise
            return web.json_response(result, status=201)
    finally:
        request.app[STATE].update(busy=False, operation=None)


def create_app(engine=None, tts=None, voices=None, qwen_clone=None):
    app = web.Application(middlewares=[errors], client_max_size=LIMIT)
    app[ENGINE] = engine or Engine()
    app[TTS] = tts or TTSEngine()
    app[VOICES] = voices or VoiceStore()
    app[QWEN_CLONE] = qwen_clone or QwenCloneEngine(app[VOICES])
    app[STATE] = {"busy": False}
    app[TOKEN] = os.getenv("API_TOKEN", "")
    app.router.add_get("/", index)
    app.router.add_get("/app.js", app_js)
    app.router.add_get("/pcm-worklet.js", pcm_worklet)
    app.router.add_get("/health", health)
    app.router.add_post("/api/transcribe", transcribe)
    app.router.add_get("/api/stream", stream_transcribe)
    app.router.add_post("/api/tts", synthesize)
    app.router.add_get('/api/voices', voice_list)
    app.router.add_post('/api/voices', voice_add)
    app.router.add_get('/api/voices/{key}/audio', voice_audio)
    app.router.add_delete('/api/voices/{key}', voice_delete)
    app.router.add_put('/api/voices/{key}/transcript', voice_reference_text)
    if isinstance(app[ENGINE], Engine) and os.getenv("ASR_PRELOAD", "1") != "0" and (ROOT / "models/asr.json").exists():
        async def preload(application):
            async def run():
                try:
                    await asyncio.to_thread(application[ENGINE].warmup)
                    LOG.info("Qwen ASR preloaded")
                except Exception:
                    LOG.exception("Qwen ASR preload failed")
            application[WARMUP] = asyncio.create_task(run())

        async def wait_for_preload(application):
            await application[WARMUP]

        app.on_startup.append(preload)
        app.on_cleanup.append(wait_for_preload)
    return app


def main():
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                        logging.FileHandler(ROOT / "logs/server.log", encoding="utf-8")])
    host = os.getenv("HOST", "127.0.0.1")
    if host not in {"127.0.0.1", "localhost", "::1"} and not os.getenv("API_TOKEN"):
        raise SystemExit("Для сетевого доступа установи API_TOKEN и HTTPS-прокси")
    web.run_app(create_app(), host=host, port=int(os.getenv("PORT", "8090")))


if __name__ == "__main__":
    main()
