# API речевого сервиса

Базовый адрес локального сервера: `http://127.0.0.1:8090`.
Если задана переменная `API_TOKEN`, передавайте заголовок
`Authorization: Bearer <токен>` для всех HTTP-запросов `/api/*`.
Одновременно выполняется одна операция; при занятости сервер возвращает
`429` и JSON `{"error":"..."}`. Ошибки входных данных — `400`, ошибки
модели — `503`.

## 1. Распознавание: аудио → текст

`POST /api/transcribe`, `multipart/form-data`:

| Поле | Обязательное | Значение |
|---|---|---|
| `file` | да | Аудиофайл или видео со звуковой дорожкой, до 64 МиБ и 10 минут |
| `engine` | нет | `qwen` |
| `context` | нет | Подсказка по терминологии, если нужна |

Ответ `200 application/json`: `text`, `language`, `model`, `device`,
`duration_seconds`, `processing_seconds`, `segments`. Текст сохраняет
язык исходной речи. Для микрофона есть отдельный WebSocket `/api/stream`:
после JSON `{"type":"start","sample_rate":16000,"engine":"qwen","token":"..."}`
передавайте бинарные порции mono PCM16 16 кГц длительностью до 1 с;
`{"type":"flush"}` завершает фразу, `{"type":"stop"}` завершает сеанс.
Сервер отправляет `ready`, `partial`, `final`, `stopped` или `error`.

## 2. Синтез встроенным голосом: текст → WAV

`POST /api/tts`, `application/json`:

```json
{"text":"Привет!","engine":"qwen","speaker":"Serena"}
```

Доступные `speaker`: Ryan, Aiden, Serena, Vivian, Uncle_Fu, Dylan, Eric,
Ono_Anna, Sohee. Максимум 2000 символов. Ответ `200 audio/wav`; заголовок
`X-TTS-Metadata` содержит JSON с моделью, временем обработки и длительностью
аудио.

## 3. Клонирование: текст + образец голоса → WAV

`POST /api/tts/clone`, `multipart/form-data`:

| Поле | Обязательное | Значение |
|---|---|---|
| `text` | да | Текст для озвучивания, до 2000 символов |
| `file` | да | Образец голоса 3–30 с, до 16 МиБ |
| `ref_text` | нет | Точная расшифровка образца, до 2000 символов; повышает точность клонирования |

Ответ `200 audio/wav` с `X-TTS-Metadata`. Образец обрабатывается в рамках
запроса и не добавляется в список сохранённых голосов.

Для частого использования одного голоса его можно сохранить один раз через
`POST /api/voices` (`multipart/form-data`: `name`, `file`). Ответ содержит `id`
вида `clone-...`. Затем вызывайте `POST /api/tts`:

```json
{"text":"Привет!","engine":"qwen_clone","speaker":"clone-..."}
```

Список голосов: `GET /api/voices`; прослушивание образца:
`GET /api/voices/{id}/audio`; удаление: `DELETE /api/voices/{id}`;
сохранение расшифровки образца:
`PUT /api/voices/{id}/transcript` с JSON `{"ref_text":"..."}`.

## Примеры PowerShell

```powershell
curl.exe -F "file=@C:\audio\speech.wav" http://127.0.0.1:8090/api/transcribe
curl.exe -H "Content-Type: application/json" -d '{"text":"Привет!","engine":"qwen","speaker":"Serena"}' -o speech.wav http://127.0.0.1:8090/api/tts
curl.exe -F "text=Привет!" -F "file=@C:\audio\voice.wav" -o cloned.wav http://127.0.0.1:8090/api/tts/clone
```

При включённом токене добавьте к `curl.exe` `-H "Authorization: Bearer <токен>"`.
