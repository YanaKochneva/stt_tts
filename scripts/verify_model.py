"""SPDX-License-Identifier: Apache-2.0. Actual inference, not a mock."""
import json
import sys
from pathlib import Path

from stt_tts.paths import ROOT, configure
configure()
from stt_tts.engine import Engine


def main():
    engine = Engine()
    result = engine.transcribe(Path(sys.argv[1]))
    import torch
    result["cuda_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated()/1024**2, 1) if torch.cuda.is_available() else None
    result["cuda_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved()/1024**2, 1) if torch.cuda.is_available() else None
    target = ROOT / "results/verification.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
