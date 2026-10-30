"""SPDX-License-Identifier: Apache-2.0. Language and grammar regression cases."""
import unittest

from stt_tts.text_normalization import normalize_for_tts, year_words
from stt_tts.engines.tts_worker import split_text


class TextNormalizationTests(unittest.TestCase):
    def test_calendar_and_event_dates_have_different_cases(self):
        cases = {
            "26 октября": "двадцать шестое октября",
            "Сегодня 26 октября.": "Сегодня двадцать шестое октября.",
            "Дата: 3 мая": "Дата: третье мая",
            "Встреча состоится 26 октября.": "Встреча состоится двадцать шестого октября.",
            "26 октября состоится встреча.": "двадцать шестого октября состоится встреча.",
            "с 26 октября до 31 октября": "с двадцать шестого октября до тридцать первого октября",
            "к 3 мая": "к третьему мая",
            "на 26 октября": "на двадцать шестое октября",
            "после 26-го октября": "после двадцать шестого октября",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(normalize_for_tts(source), expected)

    def test_numeric_dates_year_and_leap_day(self):
        cases = {
            "Встреча 26.10.2026 в 7:00.": "Встреча двадцать шестого октября две тысячи двадцать шестого года в семь часов.",
            "Дата: 2026-10-26": "Дата: двадцать шестое октября две тысячи двадцать шестого года",
            "Сегодня 26 октября 2026 года.": "Сегодня двадцать шестое октября две тысячи двадцать шестого года.",
            "Родился 3 мая 1990 г.": "Родился третьего мая тысяча девятьсот девяностого года.",
            "29 февраля 2024 года": "двадцать девятое февраля две тысячи двадцать четвёртого года",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(normalize_for_tts(source), expected)
        self.assertEqual(year_words(2000), "двухтысячного")
        self.assertEqual(year_words(1900), "тысяча девятисотого")

    def test_clock_hours_minutes_and_cases(self):
        cases = {
            "7:00": "семь часов",
            "Встреча в 7:00.": "Встреча в семь часов.",
            "Сейчас 7:01.": "Сейчас семь часов одна минута.",
            "в 7:01": "в семь часов одну минуту",
            "в 21:21": "в двадцать один час двадцать одну минуту",
            "в 2:02": "в два часа две минуты",
            "с 7:00 до 8:30": "с семи часов до восьми часов тридцати минут",
            "до 1:01": "до одного часа одной минуты",
            "к 7:05": "к семи часам пяти минутам",
            "перед 7:00": "перед семью часами",
            "в 0:00": "в ноль часов",
            "в 7:00:05": "в семь часов ноль минут пять секунд",
            "в 11:11": "в одиннадцать часов одиннадцать минут",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(normalize_for_tts(source), expected)

    def test_foreign_sentences_and_protected_tokens_unchanged(self):
        cases = [
            "The meeting is at 7:00 on 26.10.2026.",
            "Rendez-vous à 7:00.",
            "会议在7:00开始。",
            "Ссылка https://example.org/26.10.2026/7:00",
            "Код `7:00`, дата `26 октября`.",
            "Счёт 7:00. Соотношение 2:00.",
            "31 апреля, 29 февраля 2025 года, 32 октября, 25:00, 7:99, 7:00:99.",
            "Цена 26.10. Версия 26.10.2026.7. Адрес 127.0.0.1:80.",
        ]
        for source in cases:
            with self.subTest(source=source):
                self.assertEqual(normalize_for_tts(source), source)

    def test_mixed_languages_are_processed_per_sentence(self):
        self.assertEqual(normalize_for_tts("Meet at 7:00. Встреча в 8:00."),
                         "Meet at 7:00. Встреча в восемь часов.")

    def test_date_lists_share_month_and_grammatical_case(self):
        cases = {
            "24 и 25 октября": "двадцать четвёртое и двадцать пятое октября",
            "Встречи 24 и 25 октября.": "Встречи двадцать четвёртого и двадцать пятого октября.",
            "24 и 25 октября ожидается дождь.": "двадцать четвёртого и двадцать пятого октября ожидается дождь.",
            "На 24 и 25 октября": "На двадцать четвёртое и двадцать пятое октября",
            "К 24 и 25 октября": "К двадцать четвёртому и двадцать пятому октября",
            "После 24-го и 25-го октября": "После двадцать четвёртого и двадцать пятого октября",
            "24, 25 и 26 октября 2026 года": "двадцать четвёртое, двадцать пятое и двадцать шестое октября две тысячи двадцать шестого года",
            "Встречи 24 или 25 октября": "Встречи двадцать четвёртого или двадцать пятого октября",
            "31 и 25 апреля": "31 и 25 апреля",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(normalize_for_tts(source), expected)

    def test_temperature_signs_ranges_and_scale(self):
        cases = {
            "+10−12 °С": "от плюс десяти до плюс двенадцати градусов Цельсия",
            "Прогноз: +10–12 °C.": "Прогноз: от плюс десяти до плюс двенадцати градусов Цельсия.",
            "−10−12 °С": "от минус десяти до минус двенадцати градусов Цельсия",
            "−12…−10 °C": "от минус двенадцати до минус десяти градусов Цельсия",
            "−10…+12 °C": "от минус десяти до плюс двенадцати градусов Цельсия",
            "От +10 до +12 °С": "От плюс десяти до плюс двенадцати градусов Цельсия",
            "+1 °C": "плюс один градус Цельсия",
            "+2 °C": "плюс два градуса Цельсия",
            "−21 °С": "минус двадцать один градус Цельсия",
            "0 ℃": "ноль градусов Цельсия",
            "+100 °C": "плюс сто градусов Цельсия",
            "Температура 72 °F": "Температура семьдесят два градуса Фаренгейта",
            "При −10 °С": "При минус десяти градусах Цельсия",
            "До +21 °C": "До плюс двадцати одного градуса Цельсия",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(normalize_for_tts(source), expected)

    def test_temperature_rules_do_not_rewrite_foreign_text_urls_or_decimals(self):
        for source in ("Temperature +10−12 °C.", "Il fait +10 °C.",
                       "Код `+10−12 °С`.", "Ссылка https://example.org/+10°C",
                       "+10,5 °C", "−10.5 °C", "+1000 °C"):
            with self.subTest(source=source):
                self.assertEqual(normalize_for_tts(source), source)
        self.assertEqual(normalize_for_tts("Weather: +10°C. Прогноз: +10°C."),
                         "Weather: +10°C. Прогноз: плюс десять градусов Цельсия.")

    def test_temperatures_and_date_lists_are_idempotent(self):
        source = "24 и 25 октября ожидается +10−12 °С. Встречи 24, 25 и 26 октября в 7:00."
        prepared = normalize_for_tts(source)
        self.assertEqual(normalize_for_tts(prepared), prepared)
        self.assertNotRegex(prepared, r"[0-9°]")

    def test_normalization_is_idempotent_and_precedes_chunking(self):
        text = "Встреча состоится 26 октября 2026 года в 7:00. " * 5
        prepared = normalize_for_tts(text)
        self.assertEqual(normalize_for_tts(prepared), prepared)
        self.assertGreater(len(split_text(prepared)), 1)
        self.assertEqual(" ".join(split_text(prepared)), prepared.strip())


if __name__ == "__main__":
    unittest.main()
