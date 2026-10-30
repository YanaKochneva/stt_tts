"""Saved voice compatibility and Qwen cloning checks."""
import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer
from stt_tts.paths import ROOT
from stt_tts.voices import VoiceStore, decode_reference
from stt_tts.server import create_app, STATE, ENGINE
from stt_tts.tts import QwenCloneEngine
from tests.test_api import FixtureEngine, FixtureTTS


class CloneFixture(FixtureTTS):
    def synthesize(self, text, speaker, ref_text=""):
        self.reference_text = ref_text
        return super().synthesize(text, speaker)


class VoiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.store = VoiceStore(Path(self.tmp.name))
        self.clone = CloneFixture()
        self.client = TestClient(TestServer(create_app(engine=FixtureEngine(), tts=FixtureTTS(),
                                                       voices=self.store, qwen_clone=self.clone)))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.tmp.cleanup()

    async def test_upload_stores_wav_without_moss_codes(self):
        form = FormData()
        form.add_field("name", "Мой голос")
        form.add_field("file", b"fixture", filename="../../outside.wav")
        with patch("stt_tts.server.decode_reference", return_value=(None, b"RIFF-fixture", 4.0)):
            response = await self.client.post("/api/voices", data=form)
        self.assertEqual(response.status, 201)
        profile = await response.json()
        self.assertNotIn("codes", self.store.get(profile["id"]))
        self.assertEqual(self.store.path(profile["id"], ".wav").read_bytes(), b"RIFF-fixture")
        self.assertEqual((await (await self.client.get("/api/voices")).json())["voices"][0]["name"], "Мой голос")
        audio = await self.client.get(f"/api/voices/{profile['id']}/audio")
        self.assertEqual(await audio.read(), b"RIFF-fixture")
        speech = await self.client.post("/api/tts", json={"engine": "qwen_clone", "speaker": profile["id"],
                                                         "text": "Привет"})
        self.assertEqual(speech.status, 200)
        self.assertTrue(self.client.server.app[ENGINE].unloaded)
        self.assertEqual((await self.client.post("/api/tts", json={"engine": "qwen", "speaker": profile["id"],
                                                                  "text": "Привет"})).status, 400)
        self.assertEqual((await self.client.delete(f"/api/voices/{profile['id']}")).status, 200)
        self.assertEqual(list(self.store.root.iterdir()), [])

    async def test_old_moss_profile_keeps_wav_for_qwen(self):
        profile = self.store.save("Старый голос", [[1, 2]], b"RIFF-old", 4)
        reopened = VoiceStore(self.store.root)
        self.assertEqual(reopened.get(profile["id"])["codes"], [[1, 2]])
        self.assertNotIn("codes", reopened.list()[0])
        adapter = QwenCloneEngine(reopened)
        with patch.object(adapter, "_run", return_value=(b"WAV", {})) as run:
            _, metadata = adapter.synthesize("Привет", profile["id"], "Образец")
        payload, manifest = run.call_args.args
        self.assertEqual(manifest, "qwen-clone.json")
        self.assertEqual(Path(payload["reference_audio"]).read_bytes(), b"RIFF-old")
        self.assertEqual(metadata["voice_name"], "Старый голос")

    async def test_reference_text_persists_and_is_used(self):
        profile = self.store.save("Мой голос", None, b"RIFF-fixture", 4)
        key = profile["id"]
        response = await self.client.put(f"/api/voices/{key}/transcript",
                                         json={"ref_text": "Точная расшифровка."})
        self.assertEqual(response.status, 200)
        self.assertEqual(VoiceStore(self.store.root).get(key)["ref_text"], "Точная расшифровка.")
        response = await self.client.post("/api/tts", json={"engine": "qwen_clone", "speaker": key,
                                                            "text": "Новая фраза"})
        self.assertEqual(response.status, 200)
        self.assertEqual(self.clone.reference_text, "Точная расшифровка.")

    async def test_invalid_voice_and_busy_delete(self):
        for key in ("../outside", "clone-evil", "Serena"):
            with self.assertRaises(ValueError):
                self.store.get(key)
        profile = self.store.save("Sample", None, b"RIFF-fixture", 4)
        self.client.server.app[STATE]["busy"] = True
        self.assertEqual((await self.client.delete(f"/api/voices/{profile['id']}")).status, 429)
        self.client.server.app[STATE]["busy"] = False
        self.assertEqual((await self.client.post("/api/voices", json={})).status, 400)

    async def test_reference_limits_and_conversion(self):
        import numpy as np
        import soundfile as sf
        path = self.store.root / "test.wav"
        for seconds in (1, 31):
            sf.write(path, np.full(16000 * seconds, .1), 16000)
            with self.assertRaises(ValueError):
                decode_reference(path)
        sf.write(path, np.zeros(16000 * 3), 16000)
        with self.assertRaises(ValueError):
            decode_reference(path)
        sf.write(path, np.sin(np.arange(16000 * 4) * .1) * .1, 16000)
        waveform, audio, duration = decode_reference(path)
        self.assertEqual(waveform.shape, (2, 48000 * 4))
        self.assertEqual(duration, 4)
        self.assertEqual(sf.info(io.BytesIO(audio)).samplerate, 48000)


if __name__ == "__main__":
    unittest.main()
