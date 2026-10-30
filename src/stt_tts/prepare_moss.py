"""SPDX-License-Identifier: Apache-2.0. Prepare pinned MOSS ONNX assets on D."""
from .paths import ROOT, configure
configure()

import json
import hashlib
import subprocess
from huggingface_hub import HfApi, snapshot_download

REPOS = {"tts": "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX",
         "codec": "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"}


def main():
    manifest = {}
    for role, model in REPOS.items():
        info = HfApi().model_info(model)
        if info.card_data.get("license") != "apache-2.0":
            raise RuntimeError(f"Unexpected license: {model}")
        snapshot = snapshot_download(model, revision=info.sha, max_workers=1)
        manifest[role] = {"model": model, "revision": info.sha, "snapshot": snapshot, "license": "apache-2.0"}
    repo = ROOT / "tools/moss-tts-nano"
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    source = repo / "ort_cpu_runtime.py"
    manifest["runtime"] = {"source": str(source), "revision": revision,
                           "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                           "repository": "https://github.com/OpenMOSS/MOSS-TTS-Nano", "license": "apache-2.0"}
    (ROOT / "models/moss.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
