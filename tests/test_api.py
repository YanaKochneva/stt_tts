"""Transport and routing checks for the Qwen-only service."""
import asyncio
import io
import json
import os
import threading
import unittest
import wave
from unittest.mock import patch

from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer
from stt_tts.config import ROOT
from stt_tts.api.app import create_app, STATE, QWEN_CLONE


class FixtureEngine:
    state = "ready"
    device = "fixture"

    def unload(self):
        self.unloaded = True

    def transcribe(self, path, context=""):
        self.last_path = path
        assert path.is_file() and path.is_relative_to(ROOT / "tmp")
        return {"text": "fixture", "language": "Russian", "context": context}

    def transcribe_pcm(self, samples, context=""):
        self.calls = getattr(self, "calls", 0) + 1
        return {"text": "fixture", "language": "Russian", "duration_seconds": round(len(samples)/16000, 3),
                "processing_seconds": .01, "context": context}


class FixtureTTS:
    def synthesize(self, text, speaker):
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(8000)
            wav.writeframes(b"\0\0" * 800)
        return output.getvalue(), {"speaker": speaker, "duration_seconds": .1}


class APITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = FixtureEngine()
        with patch.dict(os.environ, {"API_TOKEN": "test-secret"}):
            self.client = TestClient(TestServer(create_app(engine=self.engine, tts=FixtureTTS())))
        await self.client.start_server()
        self.headers = {"Authorization": "Bearer test-secret"}

    async def asyncTearDown(self):
        await self.client.close()

    async def test_health_exposes_only_qwen(self):
        response = await self.client.get("/health")
        data = await response.json()
        self.assertEqual(set(data["asr_engines"]), {"qwen"})
        self.assertEqual(set(data["tts"]["engines"]), {"qwen", "qwen_clone"})

    async def test_authorization_and_upload_cleanup(self):
        self.assertEqual((await self.client.post("/api/transcribe")).status, 401)
        form = FormData()
        form.add_field("file", b"fixture audio", filename="../../outside.wav")
        response = await self.client.post("/api/transcribe", headers=self.headers, data=form)
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["text"], "fixture")
        self.assertFalse(self.engine.last_path.exists())

    async def test_non_qwen_engines_rejected(self):
        for name in ("whisper", "nemotron"):
            form = FormData()
            form.add_field("file", b"fixture", filename="a.wav")
            form.add_field("engine", name)
            response = await self.client.post("/api/transcribe", headers=self.headers, data=form)
            self.assertEqual(response.status, 400)
        for name in ("moss", "cosyvoice"):
            response = await self.client.post("/api/tts", headers=self.headers,
                                              json={"text": "Привет", "engine": name})
            self.assertEqual(response.status, 400)

    async def test_stream_partial_and_final(self):
        ws = await self.client.ws_connect("/api/stream")
        await ws.send_json({"type": "start", "token": "test-secret", "sample_rate": 16000, "engine": "qwen"})
        self.assertEqual((await asyncio.wait_for(ws.receive_json(), 5))["type"], "ready")
        for _ in range(2):
            await ws.send_bytes(b"\0\0" * 16000)
        partial = await asyncio.wait_for(ws.receive_json(), 5)
        self.assertEqual((partial["type"], partial["text"]), ("partial", "fixture"))
        await ws.send_json({"type": "flush"})
        final = await asyncio.wait_for(ws.receive_json(), 5)
        self.assertEqual((final["type"], final["utterance_id"]), ("final", 0))
        self.assertTrue(final["reused_partial"])
        self.assertEqual(self.engine.calls, 1)
        await ws.send_json({"type": "stop"})
        self.assertEqual((await asyncio.wait_for(ws.receive_json(), 5))["type"], "stopped")
        await ws.close()

    async def test_tts_builtin_and_busy_slot(self):
        response = await self.client.post("/api/tts", headers=self.headers,
                                          json={"text": "Привет", "engine": "qwen", "speaker": "Serena"})
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.headers["X-TTS-Metadata"])["speaker"], "Serena")
        with wave.open(io.BytesIO(await response.read()), "rb") as wav:
            self.assertEqual(wav.getnframes(), 800)
        self.assertTrue(self.engine.unloaded)
        self.client.server.app[STATE]["busy"] = True
        self.assertEqual((await self.client.post("/api/tts", headers=self.headers)).status, 429)

    async def test_one_off_clone_accepts_text_and_sample_without_saving_voice(self):
        class CloneFixture(FixtureTTS):
            def synthesize_reference(self, text, reference_audio, ref_text):
                self.received = (text, reference_audio.read_bytes(), ref_text)
                return super().synthesize(text, "one-off")
        clone = CloneFixture()
        before = (await (await self.client.get("/api/voices", headers=self.headers)).json())["voices"]
        form = FormData()
        form.add_field("text", "Привет")
        form.add_field("ref_text", "Образец")
        form.add_field("file", b"fixture audio", filename="sample.wav")
        with patch("stt_tts.api.app.decode_reference", return_value=(None, b"RIFF-reference", 4.0)), \
             patch.object(self.client.server.app[QWEN_CLONE], "synthesize_reference", clone.synthesize_reference):
            response = await self.client.post("/api/tts/clone", headers=self.headers, data=form)
        self.assertEqual(response.status, 200)
        self.assertEqual(clone.received[0::2], ("Привет", "Образец"))
        self.assertEqual(clone.received[1], b"RIFF-reference")
        self.assertEqual((await (await self.client.get("/api/voices", headers=self.headers)).json())["voices"], before)

    async def test_tts_holds_slot_until_worker_finishes(self):
        started, release = threading.Event(), threading.Event()

        class SlowTTS(FixtureTTS):
            def synthesize(self, text, speaker):
                started.set()
                if not release.wait(5):
                    raise RuntimeError("worker timed out")
                return super().synthesize(text, speaker)

        from stt_tts.api.app import TTS
        self.client.server.app[TTS].synthesize = SlowTTS().synthesize
        task = asyncio.create_task(self.client.post("/api/tts", headers=self.headers, json={"text": "hello"}))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            self.assertEqual((await self.client.post("/api/transcribe", headers=self.headers)).status, 429)
        finally:
            release.set()
            response = await task
            await response.read()
        self.assertEqual(response.status, 200)


if __name__ == "__main__":
    unittest.main()
