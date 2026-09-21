"""Deterministic text cleanup for the Qwen3-ASR service.

Engine (llama.cpp Qwen3-ASR) returns raw transcripts, optionally with a
language marker prefix (`language Chinese<asr_text>...`) and possible /sil
placeholders. Everything downstream is deterministic, model-free cleanup; line
breaks are passed through untouched - paragraph layout is left to the caller
(or a downstream LLM):

  - strip_markers  - drop the leading `language ...<asr_text>` marker
  - strip_sil      - drop /sil noise placeholders (defensive)
  - filter_fillers - remove spoken filler words (嗯/呃/那个/就是说...)
  - normalize_chinese_digits - Chinese numerals -> Arabic (ITN-lite, cn2an)
  - normalize_commands - 发送/clear 指令词字典映射(误识 -> 精确指令)
  - strip_trailing_punctuation - drop the sentence-ender ASR put at the very end
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


def has_spoken_content(text: str) -> bool:
    """Whether `text` carries real speech (alphanumerics or CJK ideographs).
    Pure punctuation-only transcripts count as not spoken."""
    return any(ch.isalnum() for ch in text)


# Sentence-enders the ASR model appends because the clip ended, not because the
# speaker finished a sentence. Weak pauses are included too: a segment flushed
# on a silence timeout may end with a comma the caller has no use for.
_TRAILING_PUNCT = "。．.！!？?；;，,、：:…"
# Closing quotes/brackets that may wrap the ender (「你好。」/ “你好。”). They are
# not punctuation the ASR invented, so they stay; only the ender inside is cut.
_TRAILING_CLOSERS = "”’\"'」』）)》】〕]})"


def strip_trailing_punctuation(text: str) -> str:
    """Drop the trailing sentence-ender the ASR added at the end of a segment.

    Each request is flushed on a pause (silence timeout), so the punctuation the
    model puts at the very end reflects that pause - not necessarily a real
    sentence boundary. Removing it lets the downstream LLM decide the final
    punctuation. Only the final run is touched; in-segment punctuation and all
    line breaks stay exactly as transcribed. A wrapping closing quote/bracket is
    preserved (「你好。」 -> 「你好」).
    """
    if not text:
        return text
    end = len(text)
    while end > 0 and text[end - 1] in _TRAILING_CLOSERS:
        end -= 1
    start = end
    while start > 0 and text[start - 1] in _TRAILING_PUNCT:
        start -= 1
    if start == end:
        return text
    return text[:start] + text[end:]


def finalize_transcription(
    raw: str,
    organizer: str = "rule",
    prompt: str = "",
    strip_trailing_punct: bool = True,
) -> str:
    """Raw engine output -> text returned by /v1/audio/transcriptions.

    none - only marker/sil removal (keep raw words otherwise)
    rule - marker/sil removal + spoken-filler cleanup
    `prompt` is the instruction bias (if any) sent to the engine; output that
    merely echoes it is discarded rather than returned as speech. Line breaks
    are left exactly as the engine produced them - paragraph layout is the
    caller's (or a downstream LLM's) job, so none are added or merged here.
    `strip_trailing_punct` (default on) drops the sentence-ender the ASR put at
    the end of this pause-flushed segment, so the downstream LLM owns the final
    punctuation. Set it off to return the engine's trailing mark verbatim.
    """
    if not raw:
        return ""
    text = strip_sil(strip_markers(raw))
    if is_prompt_echo(text, prompt):
        return ""
    if organizer == "rule":
        text = filter_fillers(text)
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
    if strip_trailing_punct:
        text = strip_trailing_punctuation(text)
    return text


__all__ = [
    "strip_markers",
    "strip_sil",
    "is_prompt_echo",
    "has_spoken_content",
    "strip_trailing_punctuation",
    "finalize_transcription",
]
