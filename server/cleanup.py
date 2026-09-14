"""Deterministic text cleanup for the Qwen3-ASR service.

Engine (llama.cpp Qwen3-ASR) returns raw transcripts, optionally with a
language marker prefix (`language Chinese<asr_text>...`) and possible /sil
placeholders. Everything downstream is deterministic, model-free cleanup:

  - strip_markers  - drop the leading `language ...<asr_text>` marker
  - strip_sil      - drop /sil noise placeholders (defensive)
  - organize_by_rule - line-initial 第X点 / 一、 -> "N. " numbered lines
  - normalize_chinese_digits - Chinese numerals -> Arabic (ITN-lite, cn2an)
  - normalize_commands - 发送/clear 指令词字典映射(误识 -> 精确指令)
  - finalize_transcription - the whole per-utterance pipeline used by the server

General vocabulary (org/emacs/funasr...) is intentionally NOT mapped - only the
two voice commands 发送 and clear get the deterministic alias table, so they are
always recognized unambiguously (config: server/command_aliases.json).
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

try:
    from command_canon import normalize_commands
except ImportError:  # pragma: no cover - co-located module
    def normalize_commands(text: str) -> str:  # type: ignore[misc]
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


# Prompt-echo guard. The engine gets the instruction prompt as decode bias; when
# the clip carries no usable speech (silence/noise) the model occasionally
# "answers" by echoing that instruction back instead of transcribing. Such text
# is not speech, so drop the whole result when it clearly mirrors the prompt.
_ECHO_MIN_RUN = 6        # a shared run this long is not a coincidence
_ECHO_MIN_COVERAGE = 0.5  # and it must cover at least half the output


def _longest_common_substring_len(a: str, b: str) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def is_prompt_echo(text: str, prompt: str) -> bool:
    """Whether `text` is (mostly) the instruction `prompt` echoed back."""
    if not text or not prompt:
        return False
    text_compact = re.sub(r"\s+", "", text)
    prompt_compact = re.sub(r"\s+", "", prompt)
    if not text_compact:
        return False
    run = _longest_common_substring_len(text_compact, prompt_compact)
    return run >= _ECHO_MIN_RUN and run >= _ECHO_MIN_COVERAGE * len(text_compact)


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


# The engine starts a new line at every pause, so one spoken sentence can be
# split across several lines. Only line-initial enumeration markers are real
# line starts; every other line is a continuation and is joined back, keeping
# an utterance as one paragraph while numbered lists still survive.
_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_CJK_PUNCT = set("，。！？；：、…（）【】「」《》〈〉“”‘’—～·")
_NO_SPACE_AFTER = "([{"
_NO_SPACE_BEFORE = ")]}%,.;:!?"


def _needs_space(left: str, right: str) -> bool:
    """Whether joining `left` + `right` needs a separating space.

    Mixed CJK/ASCII text reads naturally with a space at the seam (用 API
    做转写); two Han characters are glued directly, and CJK punctuation never
    takes a space.
    """
    if not left or not right:
        return False
    a, b = left[-1], right[0]
    if a.isspace() or b.isspace():
        return False
    if a in _CJK_PUNCT or b in _CJK_PUNCT:
        return False
    a_han = bool(_HAN_RE.match(a))
    b_han = bool(_HAN_RE.match(b))
    if a_han != b_han:
        return True
    if a_han and b_han:
        return False
    if a in _NO_SPACE_AFTER or b in _NO_SPACE_BEFORE:
        return False
    return True


def organize_by_rule(text: str) -> str:
    """Rewrite clean line-initial enumeration markers into numbered lines
    (第X点/一、... -> "N. ...").

    Pause-induced line breaks are dropped: a line that does not start a new
    enumeration item is a wrapped continuation and is joined onto the previous
    line, so an utterance comes back as one paragraph instead of one line per
    breath."""
    if not text.strip():
        return text
    lines_out: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        numbered = False
        match = _HEAD_CN_ORDINAL_RE.match(line)
        if match:
            head = _to_arabic(match.group(1))
            line = f"{head}. {line[match.end():]}".rstrip()
            numbered = True
        else:
            match = _HEAD_CN_PUNCT_RE.match(line)
            if match:
                head = _to_arabic(match.group(1))
                line = f"{head}. {line[match.end():]}".rstrip()
                numbered = True
        if numbered or not lines_out:
            lines_out.append(line)
        elif _needs_space(lines_out[-1], line):
            lines_out[-1] = f"{lines_out[-1]} {line}"
        else:
            lines_out[-1] = lines_out[-1] + line
    return "\n".join(lines_out)


def has_spoken_content(text: str) -> bool:
    """Whether `text` carries real speech (alphanumerics or CJK ideographs).
    Pure punctuation-only transcripts count as not spoken."""
    return any(ch.isalnum() for ch in text)


def finalize_transcription(
    raw: str, organizer: str = "rule", prompt: str = "", trailing: str = "\n\n"
) -> str:
    """Raw engine output -> text returned by /v1/audio/transcriptions.

    none - only marker/sil removal (keep raw words otherwise)
    rule - marker/sil removal + spoken-filler cleanup + enumeration numbering
    `prompt` is the instruction bias (if any) sent to the engine; output that
    merely echoes it is discarded rather than returned as speech.
    `trailing` is appended to a non-empty result (default one blank line) so
    callers that insert each transcript on its own get separate paragraphs; pass
    "" when the caller concatenates segments (e.g. OpenChamber dictation joins
    segments with a space, where a trailing newline would show as a line break).
    """
    if not raw:
        return ""
    text = strip_sil(strip_markers(raw))
    if is_prompt_echo(text, prompt):
        return ""
    if organizer == "rule":
        text = filter_fillers(text)
        text = organize_by_rule(text)
    # Deterministic Chinese-numeral -> Arabic (ITN-lite) on every organizer,
    # mirroring the old project: 三十五 -> 35, 百分之三十 -> 30%, 三点半 -> 3点半.
    # No-op when the optional cn2an package is not installed.
    text = normalize_chinese_digits(text)
    # Command-word dictionary mapping (发送/clear): even if the model hears a
    # command as 发松/法送/可丽儿/claer/C L E A R..., map it back to the exact
    # canonical command so 发送 / clear always come out unambiguous. Config in
    # server/command_aliases.json. General vocabulary (org/emacs/funasr...) is
    # intentionally NOT mapped.
    text = normalize_commands(text)
    if not has_spoken_content(text):
        return ""
    return text.rstrip("\n") + trailing


__all__ = [
    "strip_markers",
    "strip_sil",
    "is_prompt_echo",
    "organize_by_rule",
    "has_spoken_content",
    "finalize_transcription",
]
