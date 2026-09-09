"""Deterministic Chinese-numeral -> Arabic normalization (ITN-lite).

Uses the pure-Python `cn2an` package as the conversion engine, but shields
things cn2an would wrongly rewrite (weekday names, clock-time readings and
fixed expressions / function words that merely *contain* numerals) behind
placeholders first. When cn2an is unavailable the text is returned unchanged,
so this module never breaks the pipeline by itself.

Examples:
    现在是早上三点五十分 -> 现在是早上3点50分
    下午两点半开会 -> 下午2点半开会
    三十五份材料 -> 35份材料
    二千五百块钱 -> 2500块钱
    百分之三十到百分之五十 -> 30%到50%
    二〇二五年三月五号 -> 2025年3月5号
    我一共有五十六个 -> 我一共有56个   (一共 保持中文)
"""

from __future__ import annotations

import re

try:
    import cn2an as _cn2an
except Exception:  # pragma: no cover - environment dependent
    _cn2an = None

# Weekday spellings that must never be digitized (星期五 -> 星期5 is wrong).
_WEEKDAY_RE = re.compile(r"(?:星期|礼拜|周)[一二三四五六日天]")

# Fixed expressions / function words that merely *contain* numerals and must
# keep their Chinese spelling (一共 -> 1共 is wrong). Combined with the idiom
# list below; sorted longest-first so longer phrases win over prefixes.
_FUNCTION_WORDS = (
    "一点儿",
    "一会儿",
    "一下子",
    "一大早",
    "一样一样",
    "一共",
    "一切",
    "一直",
    "一起",
    "一样",
    "一般",
    "一些",
    "一定",
    "一律",
    "一再",
    "一度",
    "一贯",
    "一并",
    "一道",
    "一向",
    "一连",
    "一经",
    "一举",
    "一路",
    "一同",
    "一味",
    "一气",
    "一色",
    "一瞬",
    "一丝",
    "一早",
    "一点",
    "两样",
    "两可",
    "两难",
    "两全",
    "两面",
    "再三",
    "三思",
)
_IDIOMS = (
    "不三不四",
    "说三道四",
    "丢三落四",
    "三心二意",
    "四面八方",
    "五颜六色",
    "七上八下",
    "乱七八糟",
    "七嘴八舌",
    "十全十美",
    "一清二楚",
    "二话不说",
    "三番五次",
    "五花八门",
    "一心一意",
    "一模一样",
    "一来二去",
    "三五成群",
    "五湖四海",
    "一五一十",
    "三三两两",
    "半斤八两",
    "五光十色",
    "三言两语",
    "一举两得",
    "三长两短",
    "独一无二",
    "三生有幸",
    "十有八九",
    "八九不离十",
    "一心二用",
    "三更半夜",
    "一石二鸟",
    "三顾茅庐",
    "九牛一毛",
    "八仙过海",
    "两面三刀",
    "一波三折",
    "一字千金",
    "一心一德",
    "一时半会",
    "一波未平一波又起",
    "一不做二不休",
    "不管三七二十一",
    "一半",
    "另一半",
    "一大半",
    "一小半",
    "多一半",
    "少一半",
)
_PROTECT_WORDS = tuple(sorted(set(_FUNCTION_WORDS + _IDIOMS), key=len, reverse=True))
_PROTECT_RE = re.compile("|".join(map(re.escape, _PROTECT_WORDS)))

# Clock-time reading: 三点五十 / 九点四十 / 三点半 / 三点九十分.
_TIME_HOUR = "[零一二两三四五六七八九十]{1,3}"
_TIME_MIN = "[零一二两三四五六七八九十]{1,3}"
_TIME_RE = re.compile(
    rf"(?P<h>{_TIME_HOUR})点"
    rf"(?P<tail>(?P<half>半)|(?P<min>{_TIME_MIN})(?P<minf>分)?)?"
)

# Characters that make a bare "X点" unambiguous clock time rather than the
# function word 一点: 凌晨/早上/上午/中午/下午/傍晚/晚上/夜里/昨晚/今天...
_TIME_CTX = set("早上午晚傍凌晨昨今明当夜")
# 十 inside the minute part means a clock minute (三点五十 / 两点四十), never a
# decimal fraction; decimals read their digits one by one (一点八五 = 1.85).
_TEN = "十"

# Placeholders: guard token -> original text.
_PH = "\ue000"


class _Protector:
    def __init__(self, prefix: str = _PH) -> None:
        self._prefix = prefix
        self._map: list[str] = []

    def _guard(self, match: re.Match[str]) -> str:
        token = f"{self._prefix}{len(self._map)}{self._prefix}"
        self._map.append(match.group(0))
        return token

    def protect(self, text: str) -> str:
        self._map = []
        text = _WEEKDAY_RE.sub(self._guard, text)
        return _PROTECT_RE.sub(self._guard, text)

    def restore(self, text: str) -> str:
        for i, original in enumerate(self._map):
            text = text.replace(f"{self._prefix}{i}{self._prefix}", original)
        return text.replace(self._prefix, "")


def _to_digits(chinese_num: str) -> str:
    """Convert a pure Chinese numeral to its Arabic digit string."""
    converter = _cn2an.Cn2An()  # type: ignore[union-attr]
    for mode in ("normal", "strict", "smart", "direct"):
        try:
            value = converter.cn2an(chinese_num, mode=mode)
            break
        except Exception:
            continue
    else:
        # Standalone place units (百分之百 / 百分之千) fall back to a table.
        value = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}.get(
            chinese_num, chinese_num
        )
        if isinstance(value, str):
            return value
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def _is_bare_clock_hour(hour: str, prev: str) -> bool:
    """Whether "X点" without 半/分/分钟 reads as clock time.

    一点 is ambiguous with the function word (有一点累), so it only counts as
    a clock time next to a time-word or at the start of a sentence.
    """
    if hour != "一":
        return True
    return not prev or prev in _TIME_CTX or prev in "。？！，；,.!? "


def _clock_minutes(min_text: str, has_分: bool) -> bool:
    """Whether "X点<min>" is a clock minute reading rather than a decimal.

    Decimals read their digits individually (一点八五 = 1.85). Clock minutes
    are signalled by 半 or by an explicit, non-trivial minute part:
      - contains 十              三点五十 / 三点五十分   (never a fraction)
      - starts with 零            三点零五分
      - two+ digits               十一点十五分
    A single bare digit + 分 (九点八分) reads as a decimal score 9.8分, so it
    is treated as decimal; without 分 a single digit is also a decimal
    (一点八 = 1.8).
    """
    if has_分:
        return _TEN in min_text or min_text.startswith("零") or len(min_text) >= 2
    return _TEN in min_text


def _convert_clock_times(text: str) -> str:
    def sub(match: re.Match[str]) -> str:
        hour = match.group("h")
        tail = match.group("tail") or ""
        prev = text[match.start() - 1] if match.start() > 0 else ""
        if not tail:
            if not _is_bare_clock_hour(hour, prev):
                return match.group(0)  # 一点 as a word - shield from cn2an below
            return f"{_to_digits(hour)}点"
        if match.group("half"):
            return f"{_to_digits(hour)}点半"
        minutes = match.group("min")
        has_分 = match.group("minf") is not None
        if not _clock_minutes(minutes, has_分):
            # Decimal reading: 一点八秒 = 1.8秒, 十点五元 = 10.5元,
            # 九点八分 (score) = 9.8分. Converted here (not via cn2an) so the
            # 一点 prefix isn't shielded as a word.
            frac = _to_digits(minutes)
            if minutes.startswith("零") and len(minutes) >= 2 and len(frac) < len(minutes):
                frac = frac.zfill(len(minutes))
            suffix = "分" if has_分 else ""
            return f"{_to_digits(hour)}.{frac}{suffix}"
        digits = _to_digits(minutes)
        if minutes.startswith("零") and len(digits) < 2:
            digits = f"0{digits}"
        suffix = "分" if has_分 else ""
        return f"{_to_digits(hour)}点{digits}{suffix}"

    return _TIME_RE.sub(sub, text)


def _transform(text: str) -> str:
    """Run cn2an over the whole text; fall back to the input on any failure."""
    if not text.strip():
        return text
    try:
        return _cn2an.transform(text, "cn2an")  # type: ignore[union-attr]
    except Exception:
        return text


# Policy: a *single* Chinese numeral character (一个 / 三个人 / 看一下) stays in
# Chinese; only well-formed numeric expressions become Arabic. The specialised
# passes below (percent, date 年月日号, money units) already consumed the strong
# numeric positions, so what is left for cn2an is composite numbers (五十六).
# Single characters that survive all passes are shielded so cn2an never sees them.
_NUM = "零〇一二两三四五六七八九十百千万亿"
_SINGLE = "零〇一二两三四五六七八九十"

# "百分之X到百分之Y" / "百分之X" -> X%到Y% / X%
_PERCENT_RE = re.compile(rf"百分之([{_NUM}]+)([到至])百分之([{_NUM}]+)")
_PERCENT_SINGLE_RE = re.compile(rf"百分之([{_NUM}]+)")
# Calendar / money units that make even a single digit a real value: 三月 -> 3月,
# 五块钱 -> 5块钱. Multi-character numbers before these units convert too.
_UNIT_PASSES = (
    re.compile(rf"([{_NUM}]+?)(?=年)"),
    re.compile(rf"([{_NUM}]+?)(?=月)"),
    re.compile(rf"([{_NUM}]+?)(?=(?:日|号))"),
    re.compile(rf"([{_NUM}]+?)(?=(?:块|块钱|元|角|毛))"),
)
# An isolated single digit (a 一/三/五... not glued to another numeral or to an
# already-digitised context). These stay Chinese. 一…九 plus 十 included.
_ISOLATED_SINGLE_RE = re.compile(
    rf"(?<![{_NUM}点])([{_SINGLE}])(?![{_NUM}点%])"
)
_PH2 = "\ue001"


def _pass_percent(text: str) -> str:
    def ranged(m: re.Match[str]) -> str:
        return f"{_to_digits(m.group(1))}%{m.group(2)}{_to_digits(m.group(3))}%"

    text = _PERCENT_RE.sub(ranged, text)
    text = _PERCENT_SINGLE_RE.sub(lambda m: f"{_to_digits(m.group(1))}%", text)
    return text


def _pass_units(text: str) -> str:
    for rx in _UNIT_PASSES:
        text = rx.sub(lambda m: _to_digits(m.group(1)), text)
    return text


def _shield_isolated_singles(text: str, holder: _Protector) -> str:
    return _ISOLATED_SINGLE_RE.sub(holder._guard, text)  # noqa: SLF001


def normalize_chinese_digits(text: str) -> str:
    """Convert Chinese numerals in `text` to Arabic, preserving the rest.

    Multi-character numbers, decimals, percentages, dates and money values
    become Arabic; lone single digits inside ordinary words (一个 / 三个人 /
    看一下) keep their Chinese spelling.
    """
    if _cn2an is None or not text:
        return text
    # 1. Clock & decimal readings first (点 handling is unambiguous here).
    timed = _convert_clock_times(text)
    # 2. Percentages and calendar/money units.
    timed = _pass_percent(timed)
    timed = _pass_units(timed)
    # 3. Shield fixed words (一共/一点/星期/成语), then lone single digits.
    words = _Protector()
    shielded = words.protect(timed)
    singles = _Protector(_PH2)
    shielded = _shield_isolated_singles(shielded, singles)
    # 4. Composite numbers (三十五 -> 35) via cn2an, then restore.
    converted = _transform(shielded)
    converted = singles.restore(converted)
    return words.restore(converted)


__all__ = ["normalize_chinese_digits"]
