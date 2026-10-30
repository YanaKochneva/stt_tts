"""SPDX-License-Identifier: Apache-2.0. Download pinned official CV3 with verified resumable ranges."""
from .paths import ROOT, configure
configure()
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
import requests
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

MODEL_ID = 'FunAudioLLM/Fun-CosyVoice3-0.5B-2512'
FILES = {'campplus.onnx', 'cosyvoice3.yaml', 'llm.pt', 'flow.pt', 'hift.pt', 'speech_tokenizer_v3.onnx', 'README.md'}

def main():
    info = HfApi().model_info(MODEL_ID, files_metadata=True)
    if info.card_data.get('license') != 'apache-2.0':
        raise RuntimeError('Unexpected CosyVoice license')
    source = ROOT / 'tools/cosyvoice3'
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    folder = ROOT / 'models/cosyvoice3' / info.sha
    folder.mkdir(parents=True, exist_ok=True)
    block, jobs, large = 10 * 1024 * 1024, [], []
    selected = [f for f in info.siblings if f.rfilename in FILES or f.rfilename.startswith('CosyVoice-BlankEN/')]
    for item in selected:
        dest = folder / item.rfilename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if item.lfs:
            digest = item.lfs.sha256
            if dest.exists():
                with dest.open('rb') as f:
                    if hashlib.file_digest(f, 'sha256').hexdigest() == digest:
                        continue
            parts = ROOT / 'tmp/cosyvoice-download' / digest
            parts.mkdir(parents=True, exist_ok=True)
            large.append((item, dest, parts))
            for start in range(0, item.size, block):
                length = min(block, item.size-start)
                part = parts / str(start)
                if not part.exists() or part.stat().st_size != length:
                    jobs.append((item, start, length, part))
        else:
            cached = hf_hub_download(MODEL_ID, item.rfilename, revision=info.sha)
            shutil.copyfile(cached, dest)
    print('Download MiB:', round(sum(j[2] for j in jobs)/1024**2), 'ranges:', len(jobs), flush=True)
    def fetch(job):
        item, start, length, part = job
        end = start+length-1
        for attempt in range(6):
            try:
                url = hf_hub_url(MODEL_ID, item.rfilename, revision=info.sha)
                with requests.get(url+'?download=true&range='+str(start)+'&fresh='+str(time.time_ns()),
                                  headers={'Range': f'bytes={start}-{end}'}, stream=True, timeout=(20,90)) as r:
                    r.raise_for_status()
                    if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{item.size}':
                        raise RuntimeError('Unexpected byte range')
                    data = r.content
                if len(data) != length:
                    raise RuntimeError('Truncated download')
                temporary = part.with_suffix('.tmp')
                temporary.write_bytes(data)
                temporary.replace(part)
                return
            except Exception as exc:
                print('Retry:', item.rfilename, start, type(exc).__name__, flush=True)
                if attempt == 5:
                    raise RuntimeError('Could not download '+item.rfilename) from None
                time.sleep(min(2**attempt,15))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch, job) for job in jobs]
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if count % 10 == 0 or count == len(jobs):
                print('Downloaded ranges:', count, '/', len(jobs), flush=True)
    for item, dest, parts in large:
        temporary = dest.with_suffix(dest.suffix+'.download')
        digest = hashlib.sha256()
        with temporary.open('wb') as output:
            for start in range(0,item.size,block):
                data = (parts / str(start)).read_bytes()
                output.write(data)
                digest.update(data)
        if digest.hexdigest() != item.lfs.sha256:
            raise RuntimeError('SHA-256 mismatch: '+item.rfilename)
        temporary.replace(dest)
        for start in range(0,item.size,block):
            (parts / str(start)).unlink()
        parts.rmdir()
        print('Verified:', item.rfilename, flush=True)
    manifest = dict(model=MODEL_ID, revision=info.sha, snapshot=str(folder), license='apache-2.0',
                    source=str(source), source_revision=revision)
    (ROOT / 'models/cosyvoice.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest), flush=True)

if __name__ == '__main__':
    main()
