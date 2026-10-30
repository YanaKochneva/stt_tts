"""SPDX-License-Identifier: Apache-2.0. Download only the CTranslate2 checkpoint."""
from .paths import ROOT, configure
configure()

import json
import os
import hashlib
import requests
from huggingface_hub import HfApi
from .whisper_asr import MODEL_ID, MANIFEST


def main():
    info = HfApi().model_info(MODEL_ID, files_metadata=True)
    if info.cardData and info.cardData.get("license") != "apache-2.0":
        raise RuntimeError("Неожиданная лицензия весов Whisper")
    snapshot = ROOT / "models/whisper-stage-aw"
    target = snapshot / "ct2_int8_float16"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "model.bin", "preprocessor_config.json",
                 "tokenizer.json", "vocabulary.json"):
        sibling = next(s for s in info.siblings if s.rfilename == f"ct2_int8_float16/{name}")
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{info.sha}/ct2_int8_float16/{name}"
        with requests.get(url, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            expected = sibling.size
            path = target / name
            if expected and path.is_file() and path.stat().st_size == expected:
                continue
            partial = target / (name + ".partial")
            with partial.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
            if expected and partial.stat().st_size != expected:
                raise RuntimeError(f"Неполная загрузка {name}")
            os.replace(partial, path)
    model_file = target / "model.bin"
    with model_file.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    expected_digest = next(s.lfs.sha256 for s in info.siblings
                           if s.rfilename == "ct2_int8_float16/model.bin")
    if digest != expected_digest:
        raise RuntimeError("Контрольная сумма Whisper не совпадает с опубликованной")
    manifest = {"model": MODEL_ID, "revision": info.sha, "snapshot": str(snapshot),
                "license": "apache-2.0", "format": "ct2_int8_float16"}
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
