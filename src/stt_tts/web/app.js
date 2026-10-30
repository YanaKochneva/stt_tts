// SPDX-License-Identifier: Apache-2.0
const $=id=>document.getElementById(id);
let recorder,stream,recording,previewURL,ttsURL,ttsCatalog={},voiceEngine=null,ttsGenerating=false;
let savedVoices=[],voicePreviewURL,voiceSaving=false;
let referenceVoice=null,referenceTranscribing=false;
let liveSocket=null,liveAudio=null,liveNode=null,liveSource=null,liveStream=null;
let liveFinal='',livePartial='',liveSpeaking=false,liveSilenceMs=0,liveSpeechMs=0,livePreRoll=null,liveStopping=false;
const authHeaders=()=>$('token').value?{Authorization:'Bearer '+$('token').value}:{};
function updateTTSEngine(){
 const name=$('ttsEngine').value,info=ttsCatalog[name];
 if(!info)return;
 if(voiceEngine!==name){
  const female=new Set(['Bella','Ava','Serena','Vivian','Ono_Anna','Sohee']);
  $('ttsSpeaker').replaceChildren(...info.speakers.map(s=>new Option(`${s} — ${female.has(s)?'женский':'мужской'}`,s)),...(['moss','qwen_clone','cosyvoice'].includes(name)?savedVoices.map(v=>new Option(`${v.name} — мой голос`,v.id)):[]));
  $('ttsSpeaker').value=info.default_speaker||(savedVoices[0]?.id||'');voiceEngine=name;
  if(!$('ttsSpeaker').options.length)$('ttsSpeaker').add(new Option('Сначала сохраните образец голоса',''));
 }
 $('ttsSpeaker').disabled=ttsGenerating||referenceTranscribing||(!info.installed&&!['qwen_clone','cosyvoice'].includes(name));
 $('ttsGenerate').disabled=ttsGenerating||referenceTranscribing||!info.installed||!$('ttsSpeaker').value;
 $('qwenCloneOptions').hidden=!['qwen_clone','cosyvoice'].includes(name);
 $('referenceHint').textContent=name==='cosyvoice'?'CosyVoice 3: нужна точная расшифровка выбранного образца. Нажмите кнопку распознавания и проверьте текст.':'Qwen Base: без расшифровки используется только тембр; с точным текстом — речь и текст образца.';
 $('referenceLabel').textContent=name==='cosyvoice'?'Что сказано в образце (обязательно для CosyVoice)':'Что сказано в образце (необязательно)';
 $('qwenReferenceText').placeholder=name==='cosyvoice'?'Точная расшифровка выбранного образца. Обязательна для CosyVoice 3.':'Точная расшифровка выбранного образца. Можно оставить пустым для клонирования только по тембру.';
 $('qwenTranscribeReference').disabled=ttsGenerating||referenceTranscribing||!$('ttsSpeaker').value;
 syncReferenceVoice();
}
function syncReferenceVoice(){
 const key=$('ttsSpeaker').value;
 if(referenceVoice!==key){$('qwenReferenceText').value=savedVoices.find(v=>v.id===key)?.ref_text||'';$('qwenReferenceMessage').textContent='';referenceVoice=key;}
}
$('ttsSpeaker').onchange=()=>{syncReferenceVoice();updateTTSEngine();};
async function refresh(){try{const s=await(await fetch('/health')).json();ttsCatalog=s.tts?.engines||{};updateTTSEngine();const selected=$('asrEngine').value,asr=s.asr_engines?.[selected];$('status').textContent=`ASR ${selected}: ${asr?(asr.installed?asr.state:'не установлен'):s.model_state} · MOSS: ${s.tts?.engines?.moss?.state||'не установлен'} · ${s.busy?({tts:'идёт синтез',voice:'подготавливается голос',stt:'идёт распознавание'}[s.operation]||'сервис занят'):'сервис свободен'}`;}catch{$('status').textContent='Сервер недоступен';}}
$('asrEngine').onchange=refresh;
refresh();setInterval(refresh,5000);
async function refreshVoices(selected){
 const response=await fetch('/api/voices',{headers:authHeaders()});
 const data=await response.json();if(!response.ok)throw new Error(data.error||'Не удалось загрузить голоса');
 const old=selected||$('ttsSpeaker').value,oldSaved=$('savedVoice').value;
 savedVoices=data.voices;voiceEngine=null;updateTTSEngine();
 if([...$('ttsSpeaker').options].some(o=>o.value===old))$('ttsSpeaker').value=old;
 syncReferenceVoice();updateTTSEngine();
 $('savedVoice').replaceChildren(...(savedVoices.length?savedVoices.map(v=>new Option(v.name,v.id)):[new Option('Пока нет голосов','')]));
 if(savedVoices.some(v=>v.id===(selected||oldSaved)))$('savedVoice').value=selected||oldSaved;
 $('voiceListen').disabled=$('voiceDelete').disabled=!savedVoices.length;
}
refreshVoices().catch(e=>{$('voiceMessage').textContent=e.message;});
$('token').onchange=()=>refreshVoices().catch(e=>{$('voiceMessage').textContent=e.message;});
function showVoicePreview(blob){
 if(voicePreviewURL)URL.revokeObjectURL(voicePreviewURL);
 voicePreviewURL=URL.createObjectURL(blob);$('voicePreview').src=voicePreviewURL;$('voicePreview').hidden=false;
}
$('voiceFile').onchange=()=>{const file=$('voiceFile').files[0];if(file)showVoicePreview(file);};
$('voiceSave').onclick=async()=>{
 const file=$('voiceFile').files[0],name=$('voiceName').value.trim();
 if(!file||!name){$('voiceMessage').textContent='Укажи имя и выбери образец голоса';return;}
 if(file.size>16*1024*1024){$('voiceMessage').textContent='Максимум 16 МиБ';return;}
 if(voiceSaving)return;voiceSaving=true;$('voiceSave').disabled=true;
 $('voiceMessage').textContent='Подготавливаем голос на CPU… Первая загрузка может занять время.';
 try{
  const form=new FormData();form.append('name',name);form.append('file',file);
  const response=await fetch('/api/voices',{method:'POST',headers:authHeaders(),body:form});
  const data=await response.json();if(!response.ok)throw new Error(data.error||'Ошибка сохранения');
  if(!['qwen_clone','cosyvoice'].includes($('ttsEngine').value))$('ttsEngine').value='moss';await refreshVoices(data.id);
  $('voiceMessage').textContent=`Голос «${data.name}» сохранён и выбран для озвучивания. Образец: ${data.duration_seconds} с.`;
 }catch(e){$('voiceMessage').textContent=e.message;}
 finally{voiceSaving=false;$('voiceSave').disabled=false;refresh();}
};
$('voiceListen').onclick=async()=>{try{
 const key=$('savedVoice').value;if(!key)return;
 const response=await fetch(`/api/voices/${encodeURIComponent(key)}/audio`,{headers:authHeaders()});
 if(!response.ok)throw new Error((await response.json()).error||'Не удалось загрузить образец');
 showVoicePreview(await response.blob());await $('voicePreview').play();
}catch(e){$('voiceMessage').textContent=e.message;}};
$('voiceDelete').onclick=async()=>{try{
 const key=$('savedVoice').value;if(!key)return;
 const response=await fetch(`/api/voices/${encodeURIComponent(key)}`,{method:'DELETE',headers:authHeaders()});
 if(!response.ok)throw new Error((await response.json()).error||'Не удалось удалить голос');
 $('voicePreview').pause();$('voicePreview').removeAttribute('src');$('voicePreview').hidden=true;
 if(voicePreviewURL)URL.revokeObjectURL(voicePreviewURL);voicePreviewURL=null;
 await refreshVoices();$('voiceMessage').textContent='Голос и сохранённый образец удалены.';
}catch(e){$('voiceMessage').textContent=e.message;}};
async function transcribe(file){
 if(!file){$('message').textContent='Сначала выбери файл или запиши голос';return;}
 const form=new FormData();form.append('file',file);form.append('engine',$('asrEngine').value);
 $('send').disabled=$('sendRecording').disabled=true;$('message').textContent='Распознавание… Первая загрузка модели может занять время.';
 try{const response=await fetch('/api/transcribe',{method:'POST',headers:$('token').value?{Authorization:'Bearer '+$('token').value}:{},body:form});
 const data=await response.json();if(!response.ok)throw new Error(data.error||JSON.stringify(data));
 $('result').value=data.text;$('message').textContent=`${data.model} · язык: ${data.language||'не определён'}. Аудио: ${data.duration_seconds} с; обработка: ${data.processing_seconds} с; устройство: ${data.device}.`;
 }catch(e){$('message').textContent=e.message;}finally{$('send').disabled=false;$('sendRecording').disabled=!recording;refresh();}
}
$('send').onclick=()=>transcribe($('file').files[0]);
$('record').onclick=async()=>{try{
 stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true}});
 const type=['audio/webm;codecs=opus','audio/mp4'].find(t=>MediaRecorder.isTypeSupported(t));
 recorder=new MediaRecorder(stream,type?{mimeType:type}:undefined);const parts=[];
 recorder.ondataavailable=e=>{if(e.data.size)parts.push(e.data);};
 recorder.onstop=()=>{recording=new File(parts,'recording',{type:recorder.mimeType});if(previewURL)URL.revokeObjectURL(previewURL);previewURL=URL.createObjectURL(recording);$('preview').src=previewURL;$('preview').hidden=false;$('sendRecording').disabled=false;stream.getTracks().forEach(t=>t.stop());};
 recorder.start();$('record').disabled=true;$('stop').disabled=false;$('message').textContent='Идёт запись…';
 }catch(e){stream?.getTracks().forEach(t=>t.stop());$('message').textContent=e.message;}};
$('stop').onclick=()=>{if(recorder?.state==='recording')recorder.stop();$('record').disabled=false;$('stop').disabled=true;};
$('sendRecording').onclick=()=>transcribe(recording);
function liveResult(){ $('result').value=liveFinal+(livePartial?(liveFinal?'\n':'')+livePartial:''); }
async function stopLiveAudio(){
 liveNode?.disconnect();liveSource?.disconnect();
 liveStream?.getTracks().forEach(t=>t.stop());
 if(liveAudio)await liveAudio.close().catch(()=>{});
 liveNode=liveSource=liveStream=liveAudio=null;
}
function finishLive(){
 stopLiveAudio();liveSocket=null;liveStopping=false;
 $('liveStart').disabled=false;$('liveStop').disabled=true;
 $('record').disabled=false;$('send').disabled=false;
}
function sendLiveChunk(event){
 if(!liveSocket||liveSocket.readyState!==WebSocket.OPEN||liveStopping)return;
 const {pcm,rms}=event.data;
 const voiced=rms>=0.012;
 if(!liveSpeaking){
  if(!voiced){livePreRoll=pcm;return;}
  liveSpeaking=true;liveSpeechMs=0;liveSilenceMs=0;
  if(livePreRoll)liveSocket.send(livePreRoll);
  livePreRoll=null;
 }
 liveSocket.send(pcm);
 liveSpeechMs+=200;
 liveSilenceMs=voiced?0:liveSilenceMs+200;
 if((liveSilenceMs>=800&&liveSpeechMs>=600)||liveSpeechMs>=25000){
  liveSocket.send(JSON.stringify({type:'flush'}));
  liveSpeaking=false;liveSilenceMs=liveSpeechMs=0;
 }
}
$('liveStart').onclick=async()=>{
 if(liveSocket)return;
 $('liveStart').disabled=true;$('record').disabled=true;$('send').disabled=true;
 liveFinal=livePartial='';liveResult();liveSpeaking=false;liveSilenceMs=liveSpeechMs=0;livePreRoll=null;liveStopping=false;
 try{
  const url=new URL('/api/stream',location.href);url.protocol=location.protocol==='https:'?'wss:':'ws:';
  const socket=new WebSocket(url);socket.binaryType='arraybuffer';liveSocket=socket;
  const ready=new Promise((resolve,reject)=>{
   const timer=setTimeout(()=>reject(new Error('Сервер не ответил на подключение')),15000);
   socket.onopen=()=>socket.send(JSON.stringify({type:'start',sample_rate:16000,token:$('token').value,engine:$('asrEngine').value}));
   socket.onmessage=e=>{try{
    const data=JSON.parse(e.data);
    if(data.type==='ready'){clearTimeout(timer);resolve();return;}
    if(data.type==='error'){clearTimeout(timer);$('message').textContent=data.error;reject(new Error(data.error));socket.close();return;}
    if(data.type==='partial'){livePartial=data.text||'';liveResult();$('message').textContent='Предварительный текст · '+data.audio_seconds+' с речи';}
    if(data.type==='final'){liveFinal+=(liveFinal?'\n':'')+(data.text||'');livePartial='';liveResult();$('message').textContent='Фраза распознана · обработка '+data.processing_seconds+' с';}
    if(data.type==='stopped'){socket.close();$('message').textContent='Живое распознавание завершено';}
   }catch(error){reject(error);}};
   socket.onerror=()=>{clearTimeout(timer);reject(new Error('Ошибка соединения с сервером'));};
   socket.onclose=()=>{clearTimeout(timer);if(liveSocket===socket){if(!liveStopping)$('message').textContent='Соединение прервано';finishLive();}};
  });
  await ready;
  liveStream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,channelCount:1}});
  liveAudio=new AudioContext();
  await liveAudio.audioWorklet.addModule('/pcm-worklet.js');
  liveSource=liveAudio.createMediaStreamSource(liveStream);
  liveNode=new AudioWorkletNode(liveAudio,'pcm-worklet');
  liveNode.port.onmessage=sendLiveChunk;
  const mute=liveAudio.createGain();mute.gain.value=0;
  liveSource.connect(liveNode);liveNode.connect(mute);mute.connect(liveAudio.destination);
  await liveAudio.resume();
  $('liveStop').disabled=false;$('message').textContent='Слушаю. Говорите — текст появится во время речи.';
 }catch(error){$('message').textContent=error.message;liveSocket?.close();finishLive();}
};
$('liveStop').onclick=async()=>{
 if(!liveSocket||liveStopping)return;
 liveStopping=true;$('liveStop').disabled=true;
 await stopLiveAudio();
 if(liveSocket.readyState===WebSocket.OPEN){
  liveSocket.send(JSON.stringify({type:'stop'}));
  $('message').textContent='Завершаю последнюю фразу…';
 }else finishLive();
};
$('copy').onclick=()=>navigator.clipboard.writeText($('result').value).catch(e=>{$('message').textContent=e.message;});
$('save').onclick=()=>{const url=URL.createObjectURL(new Blob([$ ('result').value],{type:'text/plain;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='transcript.txt';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
window.addEventListener('beforeunload',()=>{stream?.getTracks().forEach(t=>t.stop());liveStream?.getTracks().forEach(t=>t.stop());liveSocket?.close();});

function updateTTSCount(){$('ttsCount').textContent=`${[...$('ttsText').value].length} / 2000 символов`;}
$('ttsEngine').onchange=()=>{updateTTSEngine();$('ttsMessage').textContent='Движок изменён. Нажмите «Озвучить текст», чтобы получить новое аудио.';};
$('ttsText').oninput=updateTTSCount;
$('ttsFile').onchange=async()=>{
 const file=$('ttsFile').files[0];if(!file)return;
 try{
  if(file.size>16000)throw new Error('Файл слишком большой: максимум 2000 символов текста');
  const text=new TextDecoder('utf-8',{fatal:true}).decode(await file.arrayBuffer());
  if([...text].length>2000)throw new Error('В файле больше 2000 символов. Сократите текст.');
  $('ttsText').value=text;updateTTSCount();$('ttsMessage').textContent='Текст загружен. Нажмите «Озвучить текст».';
 }catch(e){$('ttsMessage').textContent=e instanceof TypeError?'Не удалось прочитать текст. Сохраните TXT в кодировке UTF-8.':e.message;}
};
$('ttsUseTranscript').onclick=()=>{
 const text=$('result').value;
 if([...text].length>2000){$('ttsMessage').textContent='Расшифровка длиннее 2000 символов. Скопируйте нужную часть.';return;}
 $('ttsText').value=text;updateTTSCount();
};
$('qwenTranscribeReference').onclick=async()=>{
 const key=$('ttsSpeaker').value;if(!key||referenceTranscribing)return;
 referenceTranscribing=true;updateTTSEngine();$('ttsEngine').disabled=true;
 $('qwenReferenceMessage').textContent='Распознаём образец… После завершения проверьте расшифровку.';
 try{
  const sample=await fetch(`/api/voices/${encodeURIComponent(key)}/audio`,{headers:authHeaders()});
  if(!sample.ok)throw new Error((await sample.json()).error||'Не удалось загрузить образец');
  const form=new FormData();form.append('file',await sample.blob(),'reference.wav');form.append('engine',$('asrEngine').value);
  const response=await fetch('/api/transcribe',{method:'POST',headers:authHeaders(),body:form});
  const data=await response.json();if(!response.ok)throw new Error(data.error||'Ошибка распознавания');
  if([...data.text].length>2000)throw new Error('Расшифровка длиннее 2000 символов. Введите её вручную.');
  if($('ttsSpeaker').value!==key)throw new Error('Голос изменился. Повторите распознавание.');
  $('qwenReferenceText').value=data.text;
  $('qwenReferenceMessage').textContent='Готово. Проверьте, что текст точно соответствует выбранному образцу.';
 }catch(e){$('qwenReferenceMessage').textContent=e.message;}
 finally{referenceTranscribing=false;$('ttsEngine').disabled=false;updateTTSEngine();refresh();}
};
$('saveReferenceText').onclick=async()=>{try{
 const key=$('ttsSpeaker').value;if(!key)throw new Error('Сначала выберите сохранённый голос');
 const response=await fetch(`/api/voices/${encodeURIComponent(key)}/transcript`,{method:'PUT',headers:{...authHeaders(),'Content-Type':'application/json'},body:JSON.stringify({ref_text:$('qwenReferenceText').value})});
 const data=await response.json();if(!response.ok)throw new Error(data.error||'Не удалось сохранить текст');
 const voice=savedVoices.find(v=>v.id===key);if(voice)voice.ref_text=data.ref_text;
 $('qwenReferenceMessage').textContent='Текст образца сохранён. При следующем выборе голоса он появится автоматически.';
}catch(e){$('qwenReferenceMessage').textContent=e.message;}};
$('ttsGenerate').onclick=async()=>{
 const text=$('ttsText').value.trim();
 if(!text){$('ttsMessage').textContent='Введите текст или загрузите TXT-файл';return;}
 if([...text].length>2000){$('ttsMessage').textContent='Максимум 2000 символов';return;}
 if($('ttsEngine').value==='cosyvoice'&&!$('qwenReferenceText').value.trim()){$('ttsMessage').textContent='Для CosyVoice распознайте выбранный образец и проверьте его расшифровку.';return;}
 ttsGenerating=true;updateTTSEngine();$('ttsEngine').disabled=true;$('ttsOutput').hidden=true;$('ttsAudio').pause();
 const engine=$('ttsEngine').value;
 $('ttsMessage').textContent=engine==='moss'?'MOSS синтезирует речь на CPU… Первый запрос включает загрузку модели.':engine==='cosyvoice'?'CosyVoice 3 готовит образец и синтезирует речь… Загрузка модели входит во время ожидания.':'Qwen готовит голос и синтезирует речь… Загрузка модели входит во время ожидания.';
 try{
  const headers={'Content-Type':'application/json'};
  if($('token').value)headers.Authorization='Bearer '+$('token').value;
  const response=await fetch('/api/tts',{method:'POST',headers,body:JSON.stringify({text,engine,speaker:$('ttsSpeaker').value,...(['qwen_clone','cosyvoice'].includes(engine)?{ref_text:$('qwenReferenceText').value}: {})})});
  if(!response.ok){const error=await response.json();throw new Error(error.error||`Ошибка ${response.status}`);}
  const meta=JSON.parse(response.headers.get('X-TTS-Metadata')||'{}');
  const blob=await response.blob();
  if(ttsURL)URL.revokeObjectURL(ttsURL);
  ttsURL=URL.createObjectURL(blob);$('ttsAudio').src=ttsURL;$('ttsDownload').href=ttsURL;
  $('ttsOutput').hidden=false;
  const mode=['qwen_clone','cosyvoice'].includes(meta.engine)?(meta.clone_mode==='reference_text'?' · образец + расшифровка':' · только тембр'):'';
  $('ttsMessage').textContent=`Готово · ${meta.engine==='moss'?'MOSS':meta.engine==='cosyvoice'?'CosyVoice 3':meta.engine==='qwen_clone'?'Qwen Base':'Qwen'}${mode} · ${meta.voice_name||meta.speaker}. Аудио: ${meta.duration_seconds} с; обработка: ${meta.processing_seconds} с. Можно прослушать и скачать WAV.`;
 }catch(e){$('ttsMessage').textContent=e.message;}
 finally{ttsGenerating=false;$('ttsEngine').disabled=false;updateTTSEngine();refresh();}
};
window.addEventListener('beforeunload',()=>{if(ttsURL)URL.revokeObjectURL(ttsURL);if(voicePreviewURL)URL.revokeObjectURL(voicePreviewURL);});
