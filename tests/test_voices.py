"""SPDX-License-Identifier: Apache-2.0. Voice lifecycle and hostile input checks."""
import io
import tempfile
from pathlib import Path
import unittest
from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer
from stt_tts.paths import ROOT
from stt_tts.voices import VoiceStore, decode_reference
from stt_tts.server import create_app, STATE, TOKEN, ENGINE
from tests.test_api import FixtureEngine, FixtureTTS


class VoiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / 'tmp')
        self.store = VoiceStore(Path(self.tmp.name))
        store = self.store
        class Moss(FixtureTTS):
            state = 'ready'
            def add_voice(self, path, name):
                assert path.exists() and path.is_relative_to(ROOT / 'tmp')
                self.path = path
                return store.save(name, [[1, 2]], b'RIFF-fixture', 4)
        self.moss = Moss()
        class Clone(FixtureTTS):
            def synthesize(self, text, speaker, ref_text=''):
                self.reference_text = ref_text
                return super().synthesize(text, speaker)
        self.clone = Clone()
        self.cosy = Clone()
        self.client = TestClient(TestServer(create_app(FixtureEngine(), FixtureTTS(), self.moss, self.store, self.clone, self.cosy)))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.tmp.cleanup()

    async def test_lifecycle_and_engine_isolation(self):
        form = FormData()
        form.add_field('name', 'Мой голос')
        form.add_field('file', b'fixture', filename='../../outside.wav')
        r = await self.client.post('/api/voices', data=form)
        self.assertEqual(r.status, 201)
        key = (await r.json())['id']
        self.assertFalse(self.moss.path.exists())
        r = await self.client.get('/api/voices')
        item = (await r.json())['voices'][0]
        self.assertNotIn('codes', item)
        r = await self.client.get(f'/api/voices/{key}/audio')
        self.assertEqual(await r.read(), b'RIFF-fixture')
        r = await self.client.post('/api/tts', json=dict(engine='moss', speaker=key, text='Привет'))
        self.assertEqual(r.status, 200)
        r = await self.client.post('/api/tts', json=dict(engine='qwen', speaker=key, text='Привет'))
        self.assertEqual(r.status, 400)
        self.client.server.app[STATE]['busy'] = True
        r = await self.client.delete(f'/api/voices/{key}')
        self.assertEqual(r.status, 429)
        self.client.server.app[STATE]['busy'] = False
        r = await self.client.delete(f'/api/voices/{key}')
        self.assertEqual(r.status, 200)
        self.assertEqual(list(self.store.root.iterdir()), [])
        r = await self.client.post('/api/tts', json=dict(engine='moss', speaker=key, text='Привет'))
        self.assertEqual(r.status, 400)

    async def test_path_traversal_and_invalid_upload(self):
        for key in ('../outside', 'clone-evil', 'Bella'):
            with self.assertRaises(ValueError):
                self.store.get(key)
        r = await self.client.post('/api/voices', json={})
        self.assertEqual(r.status, 400)
        self.assertFalse(self.client.server.app[STATE]['busy'])

    async def test_voice_api_authorization(self):
        await self.client.close()
        app = create_app(FixtureEngine(), FixtureTTS(), self.moss, self.store, self.clone)
        app[TOKEN] = 'secret'
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        for method, url in [('get', '/api/voices'), ('post', '/api/voices'),
                            ('get', '/api/voices/clone-' + 'a' * 32 + '/audio'),
                            ('delete', '/api/voices/clone-' + 'a' * 32)]:
            response = await getattr(self.client, method)(url)
            self.assertEqual(response.status, 401)
        response = await self.client.get('/api/voices', headers={'Authorization': 'Bearer secret'})
        self.assertEqual(response.status, 200)

    async def test_profiles_survive_store_reopening(self):
        profile = self.store.save('Сохранённый', [[1, 2]], b'RIFF-fixture', 4)
        reopened = VoiceStore(self.store.root)
        self.assertEqual(reopened.get(profile['id'])['codes'], [[1, 2]])
        self.assertEqual(reopened.list(), [profile])

    async def test_reference_text_is_saved_with_voice(self):
        profile = self.store.save('Мой голос', [[1, 2]], b'RIFF-fixture', 4)
        key = profile['id']
        response = await self.client.put(f'/api/voices/{key}/transcript',
                                         json={'ref_text': 'Точная расшифровка.'})
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())['ref_text'], 'Точная расшифровка.')
        reopened = VoiceStore(self.store.root)
        self.assertEqual(reopened.list()[0]['ref_text'], 'Точная расшифровка.')
        response = await self.client.post('/api/tts', json={
            'engine': 'cosyvoice', 'speaker': key, 'text': 'Новая фраза'})
        self.assertEqual(response.status, 200)
        self.assertEqual(self.cosy.reference_text, 'Точная расшифровка.')
        response = await self.client.put(f'/api/voices/{key}/transcript',
                                         json={'ref_text': 'a' * 2001})
        self.assertEqual(response.status, 400)

    async def test_qwen_clone_modes_and_validation_before_gpu_unload(self):
        profile = self.store.save('Qwen sample', [[1, 2]], b'RIFF-fixture', 4)
        for ref_text in ('', 'Это точная расшифровка.'):
            response = await self.client.post('/api/tts', json=dict(engine='qwen_clone', speaker=profile['id'], text='Привет', ref_text=ref_text))
            self.assertEqual(response.status, 200)
            self.assertEqual(self.clone.reference_text, ref_text)
            self.assertTrue(self.client.server.app[ENGINE].unloaded)
        self.client.server.app[ENGINE].unloaded = False
        for speaker, ref_text in [('Bella', ''), ('../outside', ''), (profile['id'], None), (profile['id'], 'x'*2001)]:
            response = await self.client.post('/api/tts', json=dict(engine='qwen_clone', speaker=speaker, text='Привет', ref_text=ref_text))
            self.assertEqual(response.status, 400)
            self.assertFalse(self.client.server.app[ENGINE].unloaded)

    async def test_qwen_adapter_uses_original_wav_and_leaves_moss_codes(self):
        from unittest.mock import patch
        from stt_tts.tts import QwenCloneEngine
        profile = self.store.save('Sample', [[1, 2]], b'RIFF-fixture', 4)
        adapter = QwenCloneEngine(self.store)
        with patch.object(adapter, '_run', return_value=(b'WAV', {})) as run:
            _, meta = adapter.synthesize('Привет', profile['id'], 'Образец.')
            payload, manifest = run.call_args.args
            self.assertEqual(manifest, 'qwen-clone.json')
            self.assertEqual(Path(payload['reference_audio']).read_bytes(), b'RIFF-fixture')
            self.assertEqual(payload['ref_text'], 'Образец.')
            self.assertEqual(meta['voice_name'], 'Sample')
        self.assertEqual(self.store.get(profile['id'])['codes'], [[1, 2]])

    async def test_cosyvoice_requires_transcript_before_gpu_unload(self):
        profile = self.store.save('Cosy sample', [[1, 2]], b'RIFF-fixture', 4)
        for ref_text in ('', '   ', None, 'x'*2001):
            response = await self.client.post('/api/tts', json=dict(engine='cosyvoice', speaker=profile['id'], text='Привет', ref_text=ref_text))
            self.assertEqual(response.status, 400)
            self.assertFalse(getattr(self.client.server.app[ENGINE], 'unloaded', False))
        response = await self.client.post('/api/tts', json=dict(engine='cosyvoice', speaker=profile['id'], text='Привет', ref_text='Точная расшифровка.'))
        self.assertEqual(response.status, 200)
        self.assertEqual(self.cosy.reference_text, 'Точная расшифровка.')
        self.assertTrue(self.client.server.app[ENGINE].unloaded)

    async def test_cosyvoice_adapter_uses_separate_runtime(self):
        from unittest.mock import patch
        from stt_tts.cosyvoice import CosyVoiceEngine
        profile = self.store.save('Sample', [[1, 2]], b'RIFF-fixture', 4)
        adapter = CosyVoiceEngine(self.store)
        with patch.object(adapter, '_run', return_value=(b'WAV', {})) as run:
            adapter.synthesize('Привет', profile['id'], 'Образец.')
            payload, manifest = run.call_args.args
            self.assertEqual(manifest, 'cosyvoice.json')
            self.assertEqual(adapter.environment, 'cosyvoice3')
            self.assertEqual(adapter.worker_module, 'stt_tts.cosyvoice_worker')
            self.assertEqual(payload['ref_text'], 'Образец.')

    async def test_native_crash_exit_code_is_reported(self):
        from unittest.mock import patch, Mock
        from stt_tts.tts import TTSEngine
        root = Path(self.tmp.name)
        for directory in ('tmp', 'logs', 'models', 'envs/tts/Scripts'):
            (root / directory).mkdir(parents=True, exist_ok=True)
        (root / 'models/tts.json').write_text('{}')
        (root / 'envs/tts/Scripts/python.exe').touch()
        failed = Mock(returncode=0xc0000005, stderr='native failure')
        with patch('stt_tts.tts.ROOT', root), patch('stt_tts.tts.subprocess.run', return_value=failed):
            with self.assertRaisesRegex(RuntimeError, '0xC0000005'):
                TTSEngine().synthesize('Привет', 'Serena')
        log = (root / 'logs/tts-worker.log').read_text(encoding='utf-8')
        self.assertIn('0xC0000005', log)
        self.assertIn('native failure', log)

    async def test_reference_limits_and_conversion(self):
        import numpy as np
        import soundfile as sf
        path = self.store.root / 'test.wav'
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
