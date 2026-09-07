"""Server-side command canonicalization (deterministic, model-free).

The two voice commands are "clear" (清空) and "发送" (发送). The ASR model
occasionally hears them as a mis-spelled English word or a Chinese homophone
(clear -> claer / 可丽儿..., 发送 -> 发松 / 发宋...). This module rewrites such
near-misses back to the canonical command spelling after transcription, so the
caller can reliably detect the commands.

Extend the alias tables either by editing the defaults below or by setting the
environment variable FUNASR_COMMAND_ALIASES to a JSON object mapping each
canonical command to a list of extra aliases, e.g.:

    FUNASR_COMMAND_ALIASES='{"clear": ["克丽尔"], "发送": ["法宋"]}'
"""

from __future__ import annotations

import json
import os
import re

_ALPHABETIC = re.compile(r"[A-Za-z]+")

# Common mis-hearings of "clear" / "发送" the ASR tends to emit. Chinese
# entries are transliterations or near-homophones; English entries are also
# caught generically by edit distance, so only distinctive Chinese ones are
# listed here.
_DEFAULT_ALIASES: dict[str, list[str]] = {
    "clear": ["可丽儿", "可丽尔", "可利尔", "克丽尔", "克里尔", "科里尔", "克莱儿", "可莱儿"],
    "发送": ["发松", "发宋", "法送", "法松", "发耸", "发颂", "发诵"],
}

_CANONICAL_CLEAR = "clear"


def _load_aliases() -> dict[str, list[str]]:
    aliases = {canon: list(variants) for canon, variants in _DEFAULT_ALIASES.items()}
    raw = os.environ.get("FUNASR_COMMAND_ALIASES", "").strip()
    if not raw:
        return aliases
    try:
        extra = json.loads(raw)
    except Exception:
        return aliases
    if not isinstance(extra, dict):
        return aliases
    for canon, variants in extra.items():
        if not isinstance(canon, str) or not isinstance(variants, list):
            continue
        bucket = aliases.setdefault(canon, [])
        for v in variants:
            if isinstance(v, str) and v not in bucket:
                bucket.append(v)
    return aliases


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _fix_english_clear(text: str) -> str:
    """Rewrite any alphabetic token close to "clear" into canonical "clear"."""

    def sub(match: re.Match[str]) -> str:
        token = match.group(0)
        lowered = token.lower()
        if lowered == _CANONICAL_CLEAR:
            return _CANONICAL_CLEAR
        if 3 <= len(lowered) <= 8 and _levenshtein(lowered, _CANONICAL_CLEAR) <= 2:
            return _CANONICAL_CLEAR
        return token

    return _ALPHABETIC.sub(sub, text)


def _fix_chinese_aliases(text: str, aliases: dict[str, list[str]]) -> str:
    for canon, variants in aliases.items():
        if not variants:
            continue
        pattern = "|".join(map(re.escape, sorted(variants, key=len, reverse=True)))
        text = re.sub(pattern, canon, text)
    return text


def normalize_commands(text: str) -> str:
    """Canonicalize near-miss spellings of the voice commands in `text`."""
    if not text:
        return text
    text = _fix_english_clear(text)
    return _fix_chinese_aliases(text, _load_aliases())


__all__ = ["normalize_commands"]
