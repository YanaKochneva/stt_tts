"""Measure first/final text latency against the running local streaming ASR.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp
import numpy as np
from stt_tts.engine import decode_audio


async def benchmark(path: Path, seconds: float, url: str):
    samples = decode_audio(path)[: int(seconds * 16000)]
    async with aiohttp.ClientSession() as client:
        async with client.ws_connect(url, heartbeat=15) as ws:
            await ws.send_json({"type": "start", "sample_rate": 16000})
            ready = await ws.receive_json()
            if ready.get("type") != "ready":
                raise RuntimeError(ready)
            started = time.perf_counter()

            async def send_audio():
                for offset in range(0, len(samples), 3200):
                    chunk = np.clip(samples[offset:offset+3200], -1, 1)
                    await ws.send_bytes((chunk * 32767).astype("<i2").tobytes())
                    await asyncio.sleep(len(chunk) / 16000)
                await ws.send_json({"type": "stop"})

            producer = asyncio.create_task(send_audio())
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    break
                item = json.loads(msg.data)
                if item["type"] in {"partial", "final"}:
                    print(f"{item['type']:7} t={time.perf_counter()-started:.2f}s "
                          f"audio={item['audio_seconds']}s compute={item['processing_seconds']}s: {item['text']}", flush=True)
                elif item["type"] == "stopped":
                    break
                elif item["type"] == "error":
                    raise RuntimeError(item["error"])
            await producer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--url", default="http://127.0.0.1:8090/api/stream")
    args = parser.parse_args()
    asyncio.run(benchmark(args.file, args.seconds, args.url))
