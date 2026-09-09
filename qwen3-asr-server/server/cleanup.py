"""Deterministic text cleanup for the Qwen3-ASR service.

Engine (llama.cpp Qwen3-ASR) returns raw transcripts, optionally with a
language marker prefix (`language Chinese<asr_text>...`) and possible /sil
placeholders. Everything downstream is deterministic, model-free cleanup:

  - strip_markers  - drop the leading `language ...<asr_text>` marker
  - strip_sil      - drop /sil noise placeholders (defensive)
  - organize_by_rule - line-initial 第X点 / 一、 -> "N. " numbered lines
  - normalize_chinese_digits - Chinese numerals -> Arabic (ITN-lite, cn2an)
  - finalize_transcription - the whole per-utterance pipeline used by the server

Per design this project has NO alias/dictionary mapping (Qwen3-ASR-1.7B is
strong enough on mixed zh/en that mis-heard-word tables are not needed), but it
DOES keep deterministic numeral normalization like the old project.
"""

from __future__ import annotations

import re

try:
    from filler_filter import filter_fillers
except ImportError:  # pragma: no cover - co-located module
    def filter_fillers(text: str) -> str:  # type: ignore[misc]
        return text

try:
    from itn_digits import normalize_chinese_digits
except ImportError:  # pragma: no cover - co-located module
    def normalize_chinese_digits(text: str) -> str:  # type: ignore[misc]
        return text


# Qwen3-ASR prefixes its output with a detected-language marker when enabled:
#   language English<asr_text>Hello world.
#   language Chinese<asr_text>你好。
# Drop the marker (and the leading language name) entirely.
_MARKER_RE = re.compile(r"^\s*language\s+[A-Za-z()\- ]+<asr_text>", re.IGNORECASE)


def strip_markers(text: str) -> str:
    return _MARKER_RE.sub("", text, count=1)


def strip_sil(text: str) -> str:
    return text.replace("/sil", "")


# Line-initial Chinese ordinals turned into numbered lines (rule organizer).
_CN_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_HEAD_CN_ORDINAL_RE = re.compile(r"^第\s*([一二三四五六七八九十]|[0-9]+)\s*[点条项部分]")
_HEAD_CN_PUNCT_RE = re.compile(r"^([一二三四五六七八九十])\s*[、.．]")


def _to_arabic(num: str) -> str:
    if num.isdigit():
        return str(int(num))
    if num in _CN_DIGITS:
        return str(_CN_DIGITS[num])
    return num


def organize_by_rule(text: str) -> str:
    """Rewrite clean line-initial enumeration markers into numbered lines
    (第X点/一、... -> "N. ..."); everything else is left untouched."""
    if not text.strip():
        return text
    lines_out = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        match = _HEAD_CN_ORDINAL_RE.match(line)
        if match:
            head = _to_arabic(match.group(1))
            line = f"{head}. {line[match.end():]}".rstrip()
        else:
            match = _HEAD_CN_PUNCT_RE.match(line)
            if match:
                head = _to_arabic(match.group(1))
                line = f"{head}. {line[match.end():]}".rstrip()
        lines_out.append(line)
    return "\n".join(lines_out)


def has_spoken_content(text: str) -> bool:
    """Whether `text` carries real speech (alphanumerics or CJK ideographs).
    Pure punctuation-only transcripts count as not spoken."""
    return any(ch.isalnum() for ch in text)


def finalize_transcription(raw: str, organizer: str = "rule") -> str:
    """Raw engine output -> text returned by /v1/audio/transcriptions.

    none - only marker/sil removal (keep raw words otherwise)
    rule - marker/sil removal + spoken-filler cleanup + enumeration numbering
    A trailing blank line is always appended (unless empty) so consecutive
    voice inserts read as separate paragraphs.
    """
    if not raw:
        return ""
    text = strip_sil(strip_markers(raw))
    if organizer == "rule":
        text = filter_fillers(text)
        text = organize_by_rule(text)
    # Deterministic Chinese-numeral -> Arabic (ITN-lite) on every organizer,
    # mirroring the old project: 三十五 -> 35, 百分之三十 -> 30%, 三点半 -> 3点半.
    # No-op when the optional cn2an package is not installed.
    text = normalize_chinese_digits(text)
    if not has_spoken_content(text):
        return ""
    return text.rstrip("\n") + "\n\n"


__all__ = [
    "strip_markers",
    "strip_sil",
    "organize_by_rule",
    "has_spoken_content",
    "finalize_transcription",
]
