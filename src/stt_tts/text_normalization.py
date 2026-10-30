"""SPDX-License-Identifier: Apache-2.0. Russian date, clock and temperature verbalization.

This is a deterministic preprocessor, not a general Russian morphological parser.
Foreign-language sentences, invalid dates, URLs and inline code are preserved.
"""
import datetime as dt
import re

VERSION = "ru-dates-time-temperature-v2"
MONTHS = ("января", "февраля", "марта", "апреля", "мая", "июня", "июля",
          "августа", "сентября", "октября", "ноября", "декабря")
ONES = "ноль один два три четыре пять шесть семь восемь девять десять одиннадцать двенадцать тринадцать четырнадцать пятнадцать шестнадцать семнадцать восемнадцать девятнадцать".split()
GEN = "нуля одного двух трёх четырёх пяти шести семи восьми девяти десяти одиннадцати двенадцати тринадцати четырнадцати пятнадцати шестнадцати семнадцати восемнадцати девятнадцати".split()
DAT = GEN.copy()
DAT[:5] = "нулю одному двум трём четырём".split()
INS = "нулём одним двумя тремя четырьмя пятью шестью семью восемью девятью десятью одиннадцатью двенадцатью тринадцатью четырнадцатью пятнадцатью шестнадцатью семнадцатью восемнадцатью девятнадцатью".split()
TENS = {
    "nom": {20: "двадцать", 30: "тридцать", 40: "сорок", 50: "пятьдесят"},
    "gen": {20: "двадцати", 30: "тридцати", 40: "сорока", 50: "пятидесяти"},
    "dat": {20: "двадцати", 30: "тридцати", 40: "сорока", 50: "пятидесяти"},
    "ins": {20: "двадцатью", 30: "тридцатью", 40: "сорока", 50: "пятьюдесятью"},
}
for case in ("nom", "gen", "dat", "ins"):
    TENS[case].update(dict(zip((60, 70, 80, 90), {
        "nom": "шестьдесят семьдесят восемьдесят девяносто",
        "gen": "шестидесяти семидесяти восьмидесяти девяноста",
        "dat": "шестидесяти семидесяти восьмидесяти девяноста",
        "ins": "шестьюдесятью семьюдесятью восемьюдесятью девяноста",
    }[case].split())))
TENS["prep"] = TENS["gen"].copy()
PREPOSITIONAL = GEN.copy()
PREPOSITIONAL[:2] = ["нуле", "одном"]
HUNDREDS = {
    "nom": "сто двести триста четыреста пятьсот шестьсот семьсот восемьсот девятьсот".split(),
    "gen": "ста двухсот трёхсот четырёхсот пятисот шестисот семисот восьмисот девятисот".split(),
    "dat": "ста двумстам трёмстам четырёмстам пятистам шестистам семистам восьмистам девятистам".split(),
    "ins": "ста двумястами тремястами четырьмястами пятьюстами шестьюстами семьюстами восемьюстами девятьюстами".split(),
    "prep": "ста двухстах трёхстах четырёхстах пятистах шестистах семистах восьмистах девятистах".split(),
}
ORDINALS = dict(enumerate("первое второе третье четвёртое пятое шестое седьмое восьмое девятое десятое одиннадцатое двенадцатое тринадцатое четырнадцатое пятнадцатое шестнадцатое семнадцатое восемнадцатое девятнадцатое".split(), 1))
ORDINALS.update({20: "двадцатое", 30: "тридцатое"})
PROTECTED = re.compile(r"```[\s\S]*?```|`[^`\n]*`|\b[a-z][a-z0-9+.-]*://\S+|\bwww\.\S+", re.I)
BOUNDARY = re.compile(r"(?<!\d)[.!?;\n]|[.!?;\n](?!\d)")
PREP = re.compile(r"\b(с|со|до|после|от|около|к|ко|между|перед|в|во|на|по|за|о|об)\s*$", re.I)
DATE_YEAR = r"(?:[ \t]+(?P<year>[1-9][0-9]{3})(?:[ \t]*(?P<mark>года\b|году\b|год\b|г\.(?!\w)|г\b))?)?"
DATE_LIST_SEPARATOR = r"(?:[ \t]*,[ \t]*(?:(?:и|или|либо)[ \t]+)?|[ \t]+(?:и|или|либо)[ \t]+)"
NAMED_DATE = re.compile(r"(?<![\w-])(?P<day>[0-9]{1,2})(?:-(?P<suffix>го|му|е))?(?P<more>(?:" + DATE_LIST_SEPARATOR + r"[0-9]{1,2}(?:-(?:го|му|е))?)*?)[ \t]+(?P<month>" + "|".join(MONTHS) + r")\b" + DATE_YEAR, re.I)
NUMERIC_DATE = re.compile(r"(?<![\w./-])(?P<day>[0-9]{1,2})(?P<sep>[./])(?P<month>[0-9]{1,2})(?P=sep)(?P<year>[1-9][0-9]{3})(?:[ \t]*(?P<mark>года\b|году\b|год\b|г\.(?!\w)|г\b))?(?![\w/]|\.[0-9])", re.I)
ISO_DATE = re.compile(r"(?<![\w/-])(?P<year>[1-9][0-9]{3})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})(?![\w/-])")
CLOCK = re.compile(r"(?<![\w:/\.])(?P<hour>[0-9]{1,2}):(?P<minute>[0-5][0-9])(?::(?P<second>[0-5][0-9]))?(?![\w:]|\.[0-9])")
TEMPERATURE = re.compile(
    r"(?<![\w.,+−-])(?P<first>[+−-]?[ \t]*[0-9]{1,3})"
    r"(?:[ \t]*(?:[−–—-]|…|\.{3}|до\b)[ \t]*(?P<second>[+−-]?[ \t]*[0-9]{1,3}))?"
    r"[ \t]*(?:°[ \t]*(?P<scale>[CСFФ])|(?P<symbol>[℃℉]))(?!\w)", re.I)


def cardinal(number, case="nom", feminine=False):
    """Numbers 0..999 for clocks and temperatures."""
    if number >= 100:
        hundred = HUNDREDS["nom" if case == "acc" else case][number // 100 - 1]
        return hundred + (" " + cardinal(number % 100, case, feminine) if number % 100 else "")
    if number >= 20:
        tens = TENS["nom" if case == "acc" else case][number // 10 * 10]
        return tens + (" " + cardinal(number % 10, case, feminine) if number % 10 else "")
    if feminine and number == 1:
        return {"nom": "одна", "acc": "одну", "gen": "одной", "dat": "одной", "ins": "одной", "prep": "одной"}[case]
    if feminine and number == 2 and case in {"nom", "acc"}:
        return "две"
    return {"nom": ONES, "acc": ONES, "gen": GEN, "dat": DAT, "ins": INS, "prep": PREPOSITIONAL}[case][number]


def ordinal(number, case="nom"):
    if number not in ORDINALS:
        return TENS["nom"][number // 10 * 10] + " " + ordinal(number % 10, case)
    word = ORDINALS[number]
    if case in {"nom", "acc"}:
        return word
    if word == "третье":
        return {"gen": "третьего", "dat": "третьему", "ins": "третьим", "prep": "третьем"}[case]
    return word[:-2] + {"gen": "ого", "dat": "ому", "ins": "ым", "prep": "ом"}[case]


def year_words(year):
    thousand, rest = divmod(year, 1000)
    if not rest:
        return {1: "тысячного", 2: "двухтысячного", 3: "трёхтысячного", 4: "четырёхтысячного",
                5: "пятитысячного", 6: "шеститысячного", 7: "семитысячного", 8: "восьмитысячного", 9: "девятитысячного"}[thousand]
    prefix = "тысяча" if thousand == 1 else cardinal(thousand, feminine=True) + (" тысячи" if thousand <= 4 else " тысяч")
    hundred, tail = divmod(rest, 100)
    if not tail:
        last = {1: "сотого", 2: "двухсотого", 3: "трёхсотого", 4: "четырёхсотого", 5: "пятисотого",
                6: "шестисотого", 7: "семисотого", 8: "восьмисотого", 9: "девятисотого"}[hundred]
        return prefix + " " + last
    if hundred:
        prefix += " " + {1: "сто", 2: "двести", 3: "триста", 4: "четыреста", 5: "пятьсот",
                           6: "шестьсот", 7: "семьсот", 8: "восемьсот", 9: "девятьсот"}[hundred]
    # Ordinal tens above the day range occur in years such as 1990.
    decade = tail // 10 * 10
    if tail > 31:
        tens = {30: "тридцать", 40: "сорок", 50: "пятьдесят", 60: "шестьдесят", 70: "семьдесят", 80: "восемьдесят", 90: "девяносто"}
        last = (tens[decade] + " " + ordinal(tail % 10, "gen")) if tail % 10 else {
            40: "сорокового", 50: "пятидесятого", 60: "шестидесятого", 70: "семидесятого", 80: "восьмидесятого", 90: "девяностого"}[decade]
    else:
        last = ordinal(tail, "gen")
    return prefix + " " + last


def _russian_context(text, start, end):
    left = max([0] + [m.end() for m in BOUNDARY.finditer(text, 0, start)])
    next_boundary = BOUNDARY.search(text, end)
    right = next_boundary.start() if next_boundary else len(text)
    context = text[max(left, start - 100):min(right, end + 100)]
    ru = len(re.findall(r"[А-Яа-яЁё]", context))
    other = sum(c.isalpha() for c in context) - ru
    return ru > other or (not ru and not other)


def _case(text, start, date=False, suffix=None, end=None):
    if suffix:
        return {"го": "gen", "му": "dat", "е": "nom"}[suffix.lower()]
    before = text[max(0, start - 100):start]
    prep = PREP.search(before)
    if prep:
        word = prep[1].lower()
        if word in {"с", "со", "до", "после", "от", "около"}:
            return "gen"
        if word in {"к", "ко"}:
            return "dat"
        if word in {"между", "перед"}:
            return "ins"
        if date and word in {"о", "об"}:
            return "prep"
        return "nom" if date else "acc"
    if not date:
        return "nom"
    if re.search(r"\b(?:сегодня|завтра|вчера|дата|число)(?:\s+(?:у нас|будет|было|это))*\s*[:—–-]?\s*$", before, re.I):
        return "nom"
    # A standalone date/title is nominative; an event date defaults to genitive.
    if not before.strip(" \t\n([\"«") and not re.match(r"\s+(?:состоится|состоятся|будет|будут|пройдёт|пройдет|пройдут|ожидается|ожидаются|начнётся|начнется|начнутся|мы|я|он|она|встреча)\b", text[end or start:], re.I):
        return "nom"
    return "gen"


def _date(match, text, numeric=False):
    if numeric and not _russian_context(text, match.start(), match.end()):
        return match[0]
    day = int(match["day"])
    month = int(match["month"]) if numeric else MONTHS.index(match["month"].lower()) + 1
    year = int(match["year"]) if match["year"] else None
    more = match.groupdict().get("more") or ""
    extra_days = list(re.finditer(r"([0-9]{1,2})(?:-(го|му|е))?", more, re.I))
    try:
        for candidate in [day] + [int(m[1]) for m in extra_days]:
            dt.date(year or 2024, month, candidate)
    except ValueError:
        return match[0]
    case = _case(text, match.start(), date=True, suffix=match.groupdict().get("suffix"), end=match.end())
    def list_day(m):
        item_case = {"го": "gen", "му": "dat", "е": "nom"}.get((m[2] or "").lower(), case)
        return ordinal(int(m[1]), item_case)

    result = ordinal(day, case) + re.sub(r"([0-9]{1,2})(?:-(го|му|е))?", list_day, more, flags=re.I) + " " + MONTHS[month - 1]
    if year:
        result += " " + year_words(year) + " года"
        mark = match.groupdict().get("mark")
        if mark and mark.lower() == "г." and (not text[match.end():].strip() or re.match(r"\s+[А-ЯЁA-Z]", text[match.end():])):
            result += "."
    return result


def _quantity(number, unit, case):
    singular = number % 10 == 1 and number % 100 != 11
    few = number % 10 in {2, 3, 4} and number % 100 not in {12, 13, 14}
    forms = {
        "hour": {"nom": ("час", "часа", "часов"), "acc": ("час", "часа", "часов"), "gen": ("часа", "часов"), "dat": ("часу", "часам"), "ins": ("часом", "часами")},
        "minute": {"nom": ("минута", "минуты", "минут"), "acc": ("минуту", "минуты", "минут"), "gen": ("минуты", "минут"), "dat": ("минуте", "минутам"), "ins": ("минутой", "минутами")},
        "second": {"nom": ("секунда", "секунды", "секунд"), "acc": ("секунду", "секунды", "секунд"), "gen": ("секунды", "секунд"), "dat": ("секунде", "секундам"), "ins": ("секундой", "секундами")},
        "degree": {"nom": ("градус", "градуса", "градусов"), "acc": ("градус", "градуса", "градусов"), "gen": ("градуса", "градусов"), "dat": ("градусу", "градусам"), "ins": ("градусом", "градусами"), "prep": ("градусе", "градусах")},
    }[unit][case]
    noun = forms[0] if singular else forms[1 if few or len(forms) == 2 else 2]
    return cardinal(number, case, feminine=unit in {"minute", "second"}) + " " + noun


def _temperature(match, text):
    # A Latin C/F inside the token must not make a standalone temperature English.
    context = text[:match.start()] + " " * len(match[0]) + text[match.end():]
    if not _russian_context(context, match.start(), match.end()):
        return match[0]
    first = re.sub(r"\s", "", match["first"])
    second = re.sub(r"\s", "", match["second"]) if match["second"] else None
    scale = "Фаренгейта" if (match["scale"] or "").upper() in {"F", "Ф"} or match["symbol"] == "℉" else "Цельсия"

    def sign(value):
        return "плюс " if value.startswith("+") else "минус " if value.startswith(("-", "−")) else ""

    def number(value):
        return int(value.lstrip("+−-"))

    if second is not None:
        # Weather shorthand +10−12 means +10..+12, −10−12 means −10..−12.
        if second[0].isdigit() and first.startswith(("+", "-", "−")):
            second = first[0] + second
        prefix = "" if re.search(r"\bот\s*$", text[:match.start()], re.I) else "от "
        return (prefix + sign(first) + cardinal(number(first), "gen") + " до "
                + sign(second) + _quantity(number(second), "degree", "gen") + " " + scale)
    before = text[max(0, match.start()-50):match.start()]
    case = "prep" if re.search(r"\bпри\s*$", before, re.I) else _case(text, match.start())
    return sign(first) + _quantity(number(first), "degree", case) + " " + scale


def _clock(match, text):
    if not _russian_context(text, match.start(), match.end()) or int(match["hour"]) > 23:
        return match[0]
    if re.search(r"\b(?:счёт|счет|соотношение|пропорция)\s*[:—-]?\s*$", text[max(0, match.start()-40):match.start()], re.I):
        return match[0]
    case = _case(text, match.start())
    hour, minute = int(match["hour"]), int(match["minute"])
    second = int(match["second"]) if match["second"] is not None else None
    parts = [_quantity(hour, "hour", case)]
    if minute or second is not None:
        parts.append(_quantity(minute, "minute", case))
    if second is not None:
        parts.append(_quantity(second, "second", case))
    return " ".join(parts)


def normalize_for_tts(text):
    """Expand Russian dates, clocks and temperatures, preserving the UI source."""
    for pattern, transform in ((NAMED_DATE, lambda m, s: _date(m, s)),
                               (NUMERIC_DATE, lambda m, s: _date(m, s, True)),
                               (ISO_DATE, lambda m, s: _date(m, s, True)),
                               (CLOCK, _clock), (TEMPERATURE, _temperature)):
        protected = [(m.start(), m.end()) for m in PROTECTED.finditer(text)]
        source = text

        def replace(match):
            if any(match.start() < end and match.end() > start for start, end in protected):
                return match[0]
            return transform(match, source)

        text = pattern.sub(replace, source)
    return text
