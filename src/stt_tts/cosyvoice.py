"""SPDX-License-Identifier: Apache-2.0. Isolated CosyVoice 3 adapter."""
from .tts import QwenCloneEngine, validate_reference_text

MODEL_ID = 'FunAudioLLM/Fun-CosyVoice3-0.5B-2512'

class CosyVoiceEngine(QwenCloneEngine):
    environment = 'cosyvoice3'
    worker_module = 'stt_tts.cosyvoice_worker'
    worker_log = 'cosyvoice-worker.log'
    clone_manifest = 'cosyvoice.json'

    def synthesize(self, text, speaker, ref_text=''):
        if not validate_reference_text(ref_text):
            raise ValueError('Для CosyVoice 3 нужна точная расшифровка образца. Нажми «Распознать выбранный образец» и проверь текст.')
        return super().synthesize(text, speaker, ref_text=ref_text)
