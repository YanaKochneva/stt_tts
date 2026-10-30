"""Low-latency utterance buffering for the local ASR websocket.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
MAX_SECONDS = 30
MIN_SECONDS = 0.35


class AudioBuffer:
    def __init__(self):
        self.parts: list[np.ndarray] = []
        self.samples = 0

    def append_pcm16(self, data: bytes) -> None:
        if not data or len(data) % 2:
            raise ValueError("Ожидается непустой PCM16 с целым числом отсчётов")
        part = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        if self.samples + len(part) > MAX_SECONDS * SAMPLE_RATE:
            raise ValueError("Одна фраза не должна быть длиннее 30 секунд")
        self.parts.append(part)
        self.samples += len(part)

    @property
    def seconds(self) -> float:
        return self.samples / SAMPLE_RATE

    def snapshot(self) -> np.ndarray:
        return np.concatenate(self.parts) if self.parts else np.empty(0, dtype=np.float32)

    def clear(self) -> None:
        self.parts.clear()
        self.samples = 0
