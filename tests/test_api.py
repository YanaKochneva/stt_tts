"""SPDX-License-Identifier: Apache-2.0. Transport tests; model inference tested separately."""
import asyncio
import io
import json
import os
import threading
import wave
from pathlib import Path
import unittest
from unittest.mock import patch

from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer
from stt_tts.paths import ROOT
from stt_tts.server import create_app, STATE


class FixtureEngine:
    state = "ready"
    device = "test-fixture"

    def unload(self):
        self.unloaded = True

    def transcribe(self, path, context):
        self.last_path = path
        assert path.is_file()
        assert path.is_relative_to(ROOT / "tmp")
        return {"text": "fixture", "language": "English", "context": context}

    def transcribe_pcm(self, samples, context=""):
        return {"text": "fixture", "language": "English", "duration_seconds": round(len(samples)/16000, 3),
                "processing_seconds": 0.01, "context": context}


class FixtureWhisper(FixtureEngine):
    state = "ready"

    def transcribe(self, path, context):
        self.last_path = path
        return {"text": "пример", "language": "Russian", "model": "whisper-fixture"}


class FixtureTTS:
    def synthesize(self, text, speaker):
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(8000)
            wav.writeframes(b"\0\0" * 800)
        return output.getvalue(), {"speaker": speaker, "duration_seconds": 0.1}


class FixtureMoss(FixtureTTS):
    state = "ready"


class APITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = FixtureEngine()
        with patch.dict(os.environ, {"API_TOKEN": "test-secret"}):
            self.client = TestClient(TestServer(create_app(self.engine, FixtureTTS(), FixtureMoss(), whisper=FixtureWhisper())))
        await self.client.start_server()
        self.headers = {"Authorization": "Bearer test-secret"}

    async def asyncTearDown(self):
        await self.client.close()

    async def test_authorization(self):
        r = await self.client.post("/api/transcribe")
        self.assertEqual(r.status, 401)

    async def test_stream_returns_partial_and_final_without_file_upload(self):
        ws = await self.client.ws_connect("/api/stream")
        await ws.send_json({"type": "start", "token": "test-secret", "sample_rate": 16000})
        self.assertEqual((await asyncio.wait_for(ws.receive_json(), 5))["type"], "ready")
        for _ in range(2):
            await ws.send_bytes(b"\0\0" * 16000)
        partial = await asyncio.wait_for(ws.receive_json(), 5)
        self.assertEqual((partial["type"], partial["text"], partial["audio_seconds"]),
                         ("partial", "fixture", 2.0))
        await ws.send_json({"type": "flush"})
        final = await asyncio.wait_for(ws.receive_json(), 5)
        self.assertEqual((final["type"], final["utterance_id"]), ("final", 0))
        await ws.send_bytes(b"\0\0" * 8000)
        await ws.send_json({"type": "stop"})
        self.assertEqual((await asyncio.wait_for(ws.receive_json(), 5))["utterance_id"], 1)
        self.assertEqual((await asyncio.wait_for(ws.receive_json(), 5))["type"], "stopped")
        await ws.close()
        await asyncio.sleep(0)
        self.assertFalse(self.client.server.app[STATE]["busy"])

    async def test_stream_requires_token_before_audio(self):
        ws = await self.client.ws_connect("/api/stream")
        await ws.send_json({"type": "start", "token": "wrong", "sample_rate": 16000})
        message = await ws.receive_json()
        self.assertEqual(message["type"], "error")
        self.assertIn("API-токен", message["error"])
        await ws.close()
        await asyncio.sleep(0)
        self.assertFalse(self.client.server.app[STATE]["busy"])

    async def test_cross_origin_rejected(self):
        r = await self.client.post("/api/transcribe", headers={**self.headers, "Origin": "https://other.example"})
        self.assertEqual(r.status, 403)

    async def test_upload_on_d_and_deleted_after_response(self):
        form = FormData()
        form.add_field("file", b"test audio", filename="../../../outside.wav")
        r = await self.client.post("/api/transcribe", headers=self.headers, data=form)
        self.assertEqual(r.status, 200)
        self.assertEqual((await r.json())["language"], "English")
        self.assertFalse(self.engine.last_path.exists())
        self.assertFalse(self.engine.last_path.parent.exists())

    async def test_whisper_selection_unloads_qwen(self):
        form = FormData()
        form.add_field("file", b"test audio", filename="sample.wav")
        form.add_field("engine", "whisper")
        r = await self.client.post("/api/transcribe", headers=self.headers, data=form)
        self.assertEqual(r.status, 200)
        self.assertEqual((await r.json())["model"], "whisper-fixture")
        self.assertTrue(self.engine.unloaded)

    async def test_busy_returns_429(self):
        self.client.server.app[STATE]["busy"] = True
        r = await self.client.post("/api/transcribe", headers=self.headers)
        self.assertEqual(r.status, 429)

    async def test_invalid_upload_releases_busy_state(self):
        r = await self.client.post("/api/transcribe", headers=self.headers, json={})
        self.assertEqual(r.status, 400)
        self.assertFalse(self.client.server.app[STATE]["busy"])

    async def test_forced_language_rejected(self):
        form = FormData()
        form.add_field("file", b"test", filename="test.wav")
        form.add_field("language", "Russian")
        r = await self.client.post("/api/transcribe", headers=self.headers, data=form)
        self.assertEqual(r.status, 400)
        self.assertIn("Принудительный язык отключён", (await r.json())["error"])

    async def test_tts_returns_playable_download_and_unloads_asr(self):
        r = await self.client.post("/api/tts", headers=self.headers,
                                   json={"text": "Hello world", "speaker": "Ryan"})
        self.assertEqual(r.status, 200)
        self.assertEqual(r.content_type, "audio/wav")
        self.assertIn('filename="speech.wav"', r.headers["Content-Disposition"])
        self.assertEqual(json.loads(r.headers["X-TTS-Metadata"])["speaker"], "Ryan")
        with wave.open(io.BytesIO(await r.read()), "rb") as audio:
            self.assertEqual(audio.getnframes(), 800)
        self.assertTrue(self.engine.unloaded)
        self.assertFalse(self.client.server.app[STATE]["busy"])

    async def test_tts_validation_before_unloading_asr(self):
        for payload in ({"text": ""}, {"text": "a" * 2001}, {"text": "hello", "speaker": "unknown"}, ["hello"]):
            r = await self.client.post("/api/tts", headers=self.headers, json=payload)
            self.assertEqual(r.status, 400)
        self.assertFalse(hasattr(self.engine, "unloaded"))
        self.assertFalse(self.client.server.app[STATE]["busy"])

    async def test_tts_requires_authorization(self):
        r = await self.client.post("/api/tts", json={"text": "hello"})
        self.assertEqual(r.status, 401)

    async def test_moss_routing_keeps_asr_loaded(self):
        r = await self.client.post("/api/tts", headers=self.headers,
                                   json={"text": "Привет", "engine": "moss"})
        self.assertEqual(r.status, 200)
        metadata = json.loads(r.headers["X-TTS-Metadata"])
        self.assertEqual(metadata["engine"], "moss")
        self.assertEqual(metadata["speaker"], "Bella")
        self.assertFalse(hasattr(self.engine, "unloaded"))
        await r.read()

    async def test_engine_and_voice_validation(self):
        for payload in ({"engine": "unknown"}, {"engine": []}, {"engine": "moss", "speaker": "Serena"}):
            r = await self.client.post("/api/tts", headers=self.headers,
                                       json={"text": "Привет", **payload})
            self.assertEqual(r.status, 400)
        self.assertFalse(self.client.server.app[STATE]["busy"])

    async def test_tts_holds_shared_slot_until_worker_finishes(self):
        started, release = threading.Event(), threading.Event()

        class SlowTTS(FixtureTTS):
            def synthesize(self, text, speaker):
                started.set()
                if not release.wait(5):
                    raise RuntimeError("test worker timeout")
                return super().synthesize(text, speaker)

        from stt_tts.server import TTS
        # Change only fixture behavior; keep the running application's objects intact.
        self.client.server.app[TTS].synthesize = SlowTTS().synthesize
        pending = asyncio.create_task(self.client.post("/api/tts", headers=self.headers, json={"text": "hello"}))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            r = await self.client.post("/api/transcribe", headers=self.headers)
            self.assertEqual(r.status, 429)
            r = await self.client.post("/api/tts", headers=self.headers, json={"text": "another"})
            self.assertEqual(r.status, 429)
        finally:
            release.set()
            response = await pending
            await response.read()
        self.assertEqual(response.status, 200)
        self.assertFalse(self.client.server.app[STATE]["busy"])


if __name__ == "__main__":
    unittest.main()
