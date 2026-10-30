"""SPDX-License-Identifier: Apache-2.0. Local reference audio and cached codec tokens."""
import io
import json
import re
import uuid
import os
from .paths import ROOT


class VoiceStore:
    def __init__(self, root=None):
        self.root = root or ROOT / 'voices'
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key, suffix='.json'):
        if not isinstance(key, str) or not re.fullmatch(r'clone-[0-9a-f]{32}', key):
            raise ValueError('Некорректный идентификатор голоса')
        return self.root / (key + suffix)

    def get(self, key):
        path = self.path(key)
        if not path.exists():
            raise ValueError('Сохранённый голос не найден')
        return json.loads(path.read_text(encoding='utf-8'))

    def list(self):
        return [{k: v for k, v in self.get(p.stem).items() if k != 'codes'}
                for p in sorted(self.root.glob('clone-*.json'))]

    def save(self, name, codes, audio, duration):
        key = 'clone-' + uuid.uuid4().hex
        data = dict(id=key, name=name, codes=codes, duration_seconds=round(duration, 3))
        wav = self.path(key, '.wav')
        temporary = self.path(key, '.tmp')
        try:
            wav.write_bytes(audio)
            temporary.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
            temporary.replace(self.path(key))
        except Exception:
            wav.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)
            raise
        return {k: v for k, v in data.items() if k != 'codes'}

    def delete(self, key):
        self.get(key)
        self.path(key, '.wav').unlink(missing_ok=True)
        self.path(key).unlink()

    def set_reference_text(self, key, text):
        data = self.get(key)
        data['ref_text'] = text
        temporary = self.path(key, '.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        os.replace(temporary, self.path(key))
        return {k: v for k, v in data.items() if k != 'codes'}


def decode_reference(path):
    import av
    import numpy as np
    import soundfile as sf
    chunks, count = [], 0
    try:
        with av.open(str(path)) as container:
            stream = next((s for s in container.streams if s.type == 'audio'), None)
            if stream is None:
                raise ValueError('В файле нет аудиодорожки')
            resampler = av.AudioResampler(format='fltp', layout='stereo', rate=48000)
            def append(frame):
                nonlocal count
                values = frame.to_ndarray()
                count += values.shape[1]
                if count > 30 * 48000:
                    raise ValueError('Образец должен быть от 3 до 30 секунд. Выбери короткий фрагмент.')
                chunks.append(values)
            for frame in container.decode(stream):
                for converted in resampler.resample(frame):
                    append(converted)
            for converted in resampler.resample(None):
                append(converted)
    except av.error.FFmpegError as exc:
        raise ValueError('Не удалось прочитать аудио. Попробуй WAV или MP3.') from exc
    if count < 3 * 48000:
        raise ValueError('Для образца нужно минимум 3 секунды речи')
    waveform = np.concatenate(chunks, axis=1).astype(np.float32)
    if not np.isfinite(waveform).all() or np.max(np.abs(waveform)) < .001:
        raise ValueError('Образец пустой или слишком тихий')
    output = io.BytesIO()
    sf.write(output, waveform.T, 48000, format='WAV', subtype='PCM_16')
    return waveform, output.getvalue(), count / 48000
