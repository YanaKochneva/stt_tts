"""SPDX-License-Identifier: Apache-2.0. Download pinned TTS weights to D."""
from .paths import ROOT, configure
configure()

import json
from huggingface_hub import HfApi, snapshot_download

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"


def main(model_id=MODEL_ID, manifest_name='tts.json'):
    info = HfApi().model_info(model_id)
    if info.card_data.get("license") != "apache-2.0":
        raise RuntimeError("Unexpected TTS model license")
    snapshot = snapshot_download(model_id, revision=info.sha, max_workers=1)
    manifest = {"model": model_id, "revision": info.sha, "snapshot": snapshot,
                "license": "apache-2.0"}
    (ROOT / 'models' / manifest_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
