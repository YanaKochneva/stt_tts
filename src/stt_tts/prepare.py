"""SPDX-License-Identifier: Apache-2.0"""
from .config import ROOT, configure
configure()

import json
from huggingface_hub import HfApi, snapshot_download
from .engines.asr import MODEL_ID


def main():
    info = HfApi().model_info(MODEL_ID)
    revision = info.sha
    # Sequential download avoids the Hub 0.x symlink-capability probe race on
    # Windows without Developer Mode; existing cached weights are reused.
    snapshot = snapshot_download(MODEL_ID, revision=revision, max_workers=1)
    manifest = {"model": MODEL_ID, "revision": revision, "snapshot": snapshot, "license": "apache-2.0"}
    (ROOT / "models/asr.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
