"""Render a light, readable process map for the Qwen speech service."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "docs" / "architecture-processes.png"
W, H = 2600, 4200
BG = "#F7F9FD"
INK = "#203047"
MUTED = "#617088"
BORDER = "#DCE4EF"
WHITE = "#FFFFFF"
FONT = r"C:\Windows\Fonts\segoeui.ttf"
BOLD = r"C:\Windows\Fonts\segoeuib.ttf"


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


im = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(im)


def label(x, y, value, size=28, color=INK, bold=False):
    d.text((x, y), value, font=font(size, bold), fill=color)


def wrap(value, max_width, size, bold=False):
    words = value.split()
    lines, line = [], ""
    f = font(size, bold)
    for word in words:
        candidate = (line + " " + word).strip()
        if line and d.textbbox((0, 0), candidate, font=f)[2] > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def card(x, y, width, height, title, detail, accent):
    d.rounded_rectangle((x, y, x + width, y + height), radius=22, fill=WHITE, outline=BORDER, width=2)
    d.rounded_rectangle((x, y, x + 10, y + height), radius=5, fill=accent)
    label(x + 31, y + 17, title, 29, INK, True)
    for i, line in enumerate(wrap(detail, width - 65, 22)):
        label(x + 31, y + 59 + 29 * i, line, 22, MUTED)


def panel(x, y, width, height, number, title, subtitle, steps, accent):
    d.rounded_rectangle((x, y, x + width, y + height), radius=30, fill="#EEF2F8")
    d.rounded_rectangle((x + 22, y + 22, x + 88, y + 88), radius=18, fill=accent)
    label(x + 43, y + 34, str(number), 32, WHITE, True)
    label(x + 109, y + 29, title, 39, INK, True)
    label(x + 34, y + 101, subtitle, 24, MUTED)
    top = y + 158
    gap = 28
    ch = (height - 198 - gap * (len(steps) - 1)) // len(steps)
    for i, (name, detail) in enumerate(steps):
        cy = top + i * (ch + gap)
        card(x + 34, cy, width - 68, ch, name, detail, accent)
        if i + 1 < len(steps):
            cx = x + width // 2
            arrow_y = cy + ch + 3
            d.line((cx, arrow_y, cx, arrow_y + 15), fill=accent, width=5)
            d.polygon(((cx - 9, arrow_y + 11), (cx + 9, arrow_y + 11),
                       (cx, arrow_y + 22)), fill=accent)


label(72, 53, "Как работает речевой сервис", 65, INK, True)
label(74, 137, "Qwen · распознавание · синтез · клонирование", 32, MUTED)
d.line((72, 202, W - 72, 202), fill=BORDER, width=3)

margin, gap, pw, ph = 72, 42, 1207, 1150
panels = [
    (1, "Распознавание файла", "Аудио или звуковая дорожка видео → текст", [
        ("Пользователь / клиент", "Отправляет файл в POST /api/transcribe"),
        ("API", "Проверяет запрос и временно сохраняет файл на D:"),
        ("Декодирование", "Извлекает звук и приводит к mono 16 кГц"),
        ("Qwen3-ASR 0.6B", "Распознаёт фрагменты на языке исходной речи"),
        ("Ответ JSON", "Текст, язык, сегменты и время обработки"),
    ], "#2979BC"),
    (2, "Живая речь", "Микрофон → предварительный и итоговый текст", [
        ("Браузер", "Отсекает тишину и передаёт PCM16 через WebSocket"),
        ("API /api/stream", "Собирает звук текущей фразы в буфер"),
        ("Qwen3-ASR 0.6B", "Периодически распознаёт накопленную речь"),
        ("Предварительный текст", "Сервер отправляет событие partial"),
        ("Конец фразы", "Событие final; готовый partial может быть повторно использован"),
    ], "#26A69A"),
    (3, "Встроенный голос", "Текст + голос Qwen → WAV", [
        ("Пользователь / клиент", "POST /api/tts: текст и имя голоса"),
        ("API", "Проверяет текст и освобождает GPU от ASR"),
        ("Нормализация", "Подготавливает даты, время и температуры"),
        ("Qwen CustomVoice", "TTS worker повторно использует загруженную модель"),
        ("Ответ WAV", "Аудио для прослушивания или скачивания"),
    ], "#8D62C8"),
    (4, "Разовое клонирование", "Текст + аудиообразец → WAV", [
        ("Пользователь / клиент", "POST /api/tts/clone: text + file + необязательный ref_text"),
        ("API", "Проверяет текст, длительность и размер образца"),
        ("Подготовка образца", "Декодирует запись во временный WAV на D:"),
        ("Qwen Base", "Создаёт голосовой prompt и произносит новый текст"),
        ("Ответ WAV", "Образец удаляется из временной папки после запроса"),
    ], "#E69640"),
    (5, "Сохранение голоса", "Один раз загрузить образец → получить ID", [
        ("Пользователь / клиент", "POST /api/voices: имя и образец 3–30 секунд"),
        ("API", "Проверяет файл и декодирует его в WAV"),
        ("VoiceStore", "Сохраняет WAV и JSON-профиль на диске D:"),
        ("Текст образца", "При желании сохраняется через /transcript"),
        ("Ответ JSON", "Возвращает ID вида clone-... для следующих запросов"),
    ], "#D67A99"),
    (6, "Сохранённый голос", "Текст + ID голоса → WAV", [
        ("Пользователь / клиент", "POST /api/tts: qwen_clone, текст и ID голоса"),
        ("VoiceStore", "Находит WAV и сохранённую расшифровку образца"),
        ("Нормализация", "Подготавливает текст для произношения"),
        ("Qwen Base", "Повторно использует модель и prompt при том же образце"),
        ("Ответ WAV", "Готовую речь можно прослушать или скачать"),
    ], "#5B89CF"),
]

for i, (num, title, subtitle, steps, accent) in enumerate(panels):
    x = margin + (i % 2) * (pw + gap)
    y = 250 + (i // 2) * (ph + 46)
    panel(x, y, pw, ph, num, title, subtitle, steps, accent)

footer_y = 3860
d.rounded_rectangle((72, footer_y, W - 72, footer_y + 235), radius=28, fill="#203047")
label(110, footer_y + 31, "Общий GPU · RTX 3050, 4 ГБ", 38, WHITE, True)
label(110, footer_y + 93, "Перед TTS выгружается ASR. Перед ASR останавливается TTS worker.", 28, "#DCE8F5")
label(110, footer_y + 141, "При смене CustomVoice ↔ Base модель загружается заново.", 28, "#DCE8F5")
label(76, H - 55, "Исходный код: src/stt_tts/api, engines, config.py, voices.py · Данные моделей: D:\\Games\\STT_TTS_project", 23, MUTED)

OUT.parent.mkdir(parents=True, exist_ok=True)
im.save(OUT, optimize=True)
print(OUT)
