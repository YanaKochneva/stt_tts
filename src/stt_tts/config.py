"""SPDX-License-Identifier: Apache-2.0. Configure paths BEFORE model imports."""
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.getenv("STT_TTS_DATA_ROOT", r"D:\Games\STT_TTS_project"))


@dataclass(frozen=True)
class ApiConfig:
    """HTTP settings. Override with HOST, PORT and API_TOKEN environment variables."""
    host: str
    port: int
    token: str

    @classmethod
    def from_env(cls):
        return cls(host=os.getenv("HOST", "127.0.0.1"),
                   port=int(os.getenv("PORT", "8090")),
                   token=os.getenv("API_TOKEN", ""))


def configure():
    sys.dont_write_bytecode = True
    locations = {
        "TEMP": ROOT / "tmp",
        "TMP": ROOT / "tmp",
        "TMPDIR": ROOT / "tmp",
        "HF_HOME": ROOT / "cache/huggingface",
        "HF_HUB_CACHE": ROOT / "cache/huggingface/hub",
        "HF_ASSETS_CACHE": ROOT / "cache/huggingface/assets",
        "TORCH_HOME": ROOT / "cache/torch",
        "TORCHINDUCTOR_CACHE_DIR": ROOT / "cache/torchinductor",
        "TRITON_CACHE_DIR": ROOT / "cache/triton",
        "XDG_CACHE_HOME": ROOT / "cache",
        "NUMBA_CACHE_DIR": ROOT / "cache/numba",
        "PIP_CACHE_DIR": ROOT / "cache/pip",
        "UV_CACHE_DIR": ROOT / "cache/uv",
        "MODELSCOPE_CACHE": ROOT / "cache/modelscope",
        "MPLCONFIGDIR": ROOT / "cache/matplotlib",
    }
    for key, path in locations.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    for part in ("data", "logs", "models", "results"):
        (ROOT / part).mkdir(exist_ok=True)
    tempfile.tempdir = str(ROOT / "tmp")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    return ROOT
