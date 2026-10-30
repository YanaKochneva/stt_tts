"""SPDX-License-Identifier: Apache-2.0. Official CV3 inference with memory-conscious weight loading."""
from .paths import ROOT, configure
configure()
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback
import faulthandler

def main():
    faulthandler.enable(all_threads=True)
    def stage(message):
        print('CosyVoice этап: ' + message, file=sys.stderr, flush=True)
    def thread_failed(args):
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
        sys.stderr.flush()
        os._exit(1)
    threading.excepthook = thread_failed
    stage('подготовка окружения')
    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text(encoding='utf-8'))
    config = json.loads((ROOT / 'models/cosyvoice.json').read_text(encoding='utf-8'))
    source = Path(config['source'])
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != config['source_revision']:
        raise RuntimeError('Версия runtime CosyVoice изменилась. Повтори подготовку.')
    requested = os.getenv('COSYVOICE_DEVICE', os.getenv('TTS_DEVICE', 'auto'))
    if requested == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif requested not in {'auto', 'cuda', 'cuda:0'}:
        raise ValueError('COSYVOICE_DEVICE: auto, cpu или cuda:0')
    sys.path[:0] = [str(source), str(source / 'third_party/Matcha-TTS')]
    import numpy as np
    import soundfile as sf
    import torch
    from cosyvoice.cli.cosyvoice import CosyVoice3
    from cosyvoice.cli.model import CosyVoice3Model
    from .cosyvoice import MODEL_ID
    from .tts import validate, validate_reference_text
    from .tts_worker import split_text
    from .text_normalization import normalize_for_tts, VERSION

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if requested.startswith('cuda') and device == 'cpu':
        raise RuntimeError('CUDA недоступна для CosyVoice')
    torch.set_num_threads(int(os.getenv('COSYVOICE_CPU_THREADS', '4')))
    # Upstream loads each state_dict directly to GPU while keeping fp32 weights.
    # Load through CPU and keep the LLM in fp16; preserve flow/vocoder fp32 weights.
    def load_low_memory(self, llm_model, flow_model, hift_model):
        for label, module, path, is_hift in (('языковая модель',self.llm,llm_model,False), ('flow',self.flow,flow_model,False), ('вокодер',self.hift,hift_model,True)):
            stage('загрузка весов: '+label)
            weights = torch.load(path, map_location='cpu', weights_only=True)
            if is_hift:
                weights = {k.replace('generator.', ''):v for k,v in weights.items()}
            module.load_state_dict(weights, strict=True)
            del weights
            if module is self.llm and device.startswith('cuda'):
                module.half()
            module.to(self.device).eval()
            stage('веса загружены: '+label)
    CosyVoice3Model.load = load_low_memory
    text = validate(request['text'], request['speaker'], speakers=(request['speaker'],))
    ref_text = validate_reference_text(request.get('ref_text', ''))
    if not ref_text:
        raise ValueError('Нужна расшифровка образца')
    stage('инициализация модели на '+device)
    model = CosyVoice3(config['snapshot'], fp16=device.startswith('cuda'), load_trt=False, load_vllm=False)
    stage('модель загружена')
    prepared = normalize_for_tts(text)
    pieces = split_text(prepared, limit=120)
    chunks = []
    prompt = 'You are a helpful assistant.<|endofprompt|>' + ref_text
    with torch.inference_mode():
        stage('подготовка образца голоса')
        model.add_zero_shot_spk(prompt, request['reference_audio'], 'reference')
        for piece in pieces:
            stage('синтез фрагмента')
            waves = [part['tts_speech'].detach().float().cpu().numpy().reshape(-1)
                     for part in model.inference_zero_shot(piece, prompt, request['reference_audio'],
                                                          zero_shot_spk_id='reference', stream=False, text_frontend=False)]
            wave = np.concatenate(waves) if waves else np.array([], dtype=np.float32)
            if not wave.size or not np.isfinite(wave).all() or len(wave)/model.sample_rate > 90:
                raise RuntimeError('CosyVoice вернул некорректное или слишком длинное аудио')
            if chunks:
                chunks.append(np.zeros(int(.15*model.sample_rate),dtype=np.float32))
            chunks.append(wave)
    audio = np.concatenate(chunks)
    sf.write(request_path.parent / 'output.wav', audio, model.sample_rate, subtype='PCM_16')
    metadata = dict(model=MODEL_ID, speaker=request['speaker'], device=device,
                    duration_seconds=round(len(audio)/model.sample_rate,3), sample_rate=model.sample_rate,
                    chunks=len(pieces), simulated=False, clone_mode='reference_text',
                    text_normalization=VERSION, normalization_applied=prepared!=text,
                    weight_loading='cpu-state-dict; '+('fp16-llm' if device.startswith('cuda') else 'fp32-llm')+'; fp32-flow-vocoder',
                    cuda_peak_allocated_mib=round(torch.cuda.max_memory_allocated()/1024**2,1) if device.startswith('cuda') else None)
    (request_path.parent/'result.json').write_text(json.dumps(metadata),encoding='utf-8')

if __name__ == '__main__':
    main()
