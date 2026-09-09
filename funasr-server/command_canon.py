"""Server-side command / special-vocabulary canonicalization (deterministic,
model-free, config-driven).

The 0.6B Fun-ASR model hears a fixed command in several different ways - both
across speakers and across repetitions by the same speaker (发送 -> 发松/发宋,
clear -> claer/可丽儿, "org model" -> 奥格猫抖 / o r g model ...). Instead of
depending on the model hearing the canonical spelling, this module rewrites any
*observed* spelling listed under a canonical command/vocabulary word back to
that canonical form after transcription.

Two sources of configured spellings, both optional (nothing matches = text
unchanged):

  - A JSON file, edited by hand, next to this module (default
    `command_aliases.json`; override with FUNASR_COMMAND_ALIASES_FILE):
        {
          "clear":      ["可丽儿", "可丽尔", "克丽尔"],
          "发送":        ["发松", "发宋", "法送", "发耸"],
          "org model":  ["org model", "奥格猫抖", "o r g model", "og model"],
          "qwen":       {"forms": ["quen", "kwen", "Q W E N"], "fuzzy": true}
        }
    Each key is the canonical output; its list is every recognition result you
    have personally observed for it. A dict form additionally enables `fuzzy`
    (see below). For a key listed in the file its list REPLACES the built-in
    default; unlisted keys keep their default.

  - The environment variable FUNASR_COMMAND_ALIASES adds extra spellings onto
    whatever is effective after the file, e.g.
        FUNASR_COMMAND_ALIASES='{"clear": ["克丽尔"], "发送": ["法宋"]}'

Matching is robust to how the ASR writes English:
  - case-insensitive, and "C L E A R" / "c l e a r" (spelled one letter at a
    time) are joined to "CLEAR" before matching, so an alias that is the normal
    word also catches the spelled-out version;
  - spaces inside an alias are loose: "org model" also matches "org  model";
  - `fuzzy: true` (on by default for the built-in "clear") also accepts any
    nearby ASCII spelling within edit distance 2 (claer -> clear).

Chinese aliases keep a simple rule: the exact observed spelling is replaced
anywhere it appears (发松 -> 发送), so only list spellings you would actually
want rewritten.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

_ALPHABETIC = re.compile(r"[A-Za-z]+")
_CJK = "[\u3400-\u4dbf\u4e00-\u9fff]"
_CJK_RE = re.compile(_CJK)

# Common mis-hearings observed with the stock model. Kept as the built-in
# baseline; a `command_aliases.json` key replaces its entry wholesale.
_DEFAULT_ALIASES: dict[str, list[str]] = {
    "clear": ["可丽儿", "可丽尔", "可利尔", "克丽尔", "克里尔", "科里尔", "克莱儿", "可莱儿"],
    "发送": ["发松", "发宋", "法送", "法松", "发耸", "发颂", "发诵"],
}

_DEFAULT_FILE = "command_aliases.json"
_CANONICAL_CLEAR = "clear"

# Canonical commands that also accept any *nearby* English spelling of the same
# word (edit distance <= 2) so the fuzzy sweep covers the whole bucket, not a
# hand-listed guess (claer/clea/celar -> clear).
_FUZZY_BY_DEFAULT = {_CANONICAL_CLEAR}


def _default_alias_path() -> Path:
    override = os.environ.get("FUNASR_COMMAND_ALIASES_FILE", "").strip()
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent / _DEFAULT_FILE


def _is_ascii_alias(value: str) -> bool:
    return bool(value) and not re.search(_CJK, value)


def _load_file_profiles() -> dict[str, dict]:
    """Profiles from the JSON file (path may be configured)."""

    path = _default_alias_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}

    profiles: dict[str, dict] = {}
    for canon, value in raw.items():
        if not isinstance(canon, str):
            continue
        forms: list[str] = []
        fuzzy: bool = canon in _FUZZY_BY_DEFAULT
        if isinstance(value, list):
            forms = [v for v in value if isinstance(v, str)]
        elif isinstance(value, dict):
            forms = [v for v in value.get("forms", []) if isinstance(v, str)]
            if isinstance(value.get("fuzzy"), bool):
                fuzzy = value["fuzzy"]
        else:
            continue
        if canon.strip():
            profiles[canon] = {"forms": forms, "fuzzy": fuzzy}
    return profiles


def _load_profiles() -> dict[str, dict]:
    """Effective profiles: defaults, then file (replaces per key), then env
    (appends extra spellings on top)."""

    profiles: dict[str, dict] = {}
    for canon, variants in _DEFAULT_ALIASES.items():
        profiles[canon] = {
            "forms": list(variants),
            "fuzzy": canon in _FUZZY_BY_DEFAULT,
        }
    for canon, profile in _load_file_profiles().items():
        profiles[canon] = profile

    raw_env = os.environ.get("FUNASR_COMMAND_ALIASES", "").strip()
    if raw_env:
        try:
            extra = json.loads(raw_env)
        except Exception:
            extra = None
        if isinstance(extra, dict):
            for canon, variants in extra.items():
                if not isinstance(canon, str) or not isinstance(variants, list):
                    continue
                bucket = profiles.setdefault(
                    canon, {"forms": [], "fuzzy": canon in _FUZZY_BY_DEFAULT}
                )
                for v in variants:
                    if isinstance(v, str) and v not in bucket["forms"]:
                        bucket["forms"].append(v)

    # A profile never knows its own canonical form by coincidence of aliasing;
    # strip no-op spellings that equal the canonical text.
    for canon, profile in profiles.items():
        profile["forms"] = [v for v in profile["forms"] if v.strip() and v != canon]
    return profiles


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


# Tokens that make up the ASR's "spelled it out" mode: consecutive ASCII-letter
# words where EVERY token is a single letter and only whitespace separates them
# (C L E A R / o r g). Joined into one word before alias matching so an alias
# that is the normal spelling also catches the spelled-out version. A run of 3+
# single letters qualifies (I'm/ok are left alone), and a longer word glued at
# the end (o r g model) stops the run, leaving "org model".
_ASCII_WORD_RE = re.compile(r"[A-Za-z]+")
_WS_ONLY_RE = re.compile(r"[ \t]*")


def _join_spelled_letters(text: str) -> str:
    words = [(m.group(0), m.start(), m.end()) for m in _ASCII_WORD_RE.finditer(text)]

    # Greedily extend a run while every token is a single letter AND only
    # whitespace sits between consecutive letters. A normal word (token of
    # length > 1) ends the run, so "o r g model" joins to "org model", not
    # "orgmodel".
    runs: list[tuple[int, int, str]] = []
    i = 0
    while i < len(words):
        token, w_start, w_end = words[i]
        if len(token) != 1:
            i += 1
            continue
        chain = [(w_start, w_end, token)]
        j = i + 1
        while j < len(words):
            nxt, n_start, n_end = words[j]
            if len(nxt) == 1 and _WS_ONLY_RE.fullmatch(text[chain[-1][1]:n_start]):
                chain.append((n_start, n_end, nxt))
                j += 1
            else:
                break
        if len(chain) >= 3:
            runs.append((chain[0][0], chain[-1][1], "".join(c[2] for c in chain)))
        i = j
    if not runs:
        return text
    out = []
    prev = 0
    for s, e, letters in runs:
        out.append(text[prev:s])
        out.append(letters)
        prev = e
    out.append(text[prev:])
    return "".join(out)


def _alias_to_regex(alias: str) -> re.Pattern[str]:
    """Tolerant matcher for an ASCII alias: case-insensitive, whitespace inside
    the alias is loose, and the match must not sit inside a longer word."""

    parts = re.split(r"\s+", alias.strip())
    core = r"[ \t]*".join(map(re.escape, parts))
    if parts and parts[0] and parts[0][0].isalnum():
        core = r"(?<![A-Za-z0-9])" + core
    if parts and parts[-1] and parts[-1][-1].isalnum():
        core = core + r"(?![A-Za-z0-9])"
    return re.compile(core, re.IGNORECASE)


def _pad_ascii(text: str, match: re.Match[str], canonical: str) -> str:
    """Keep one space between an ASCII canonical and neighbouring CJK text so
    mixed-language reads naturally: 奥格猫抖有新版 -> "org model 有新版"."""
    if not canonical or not (canonical[0].isascii() and canonical[-1].isascii()):
        return canonical
    start, end = match.span()
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    pad_l = bool(before) and bool(_CJK_RE.fullmatch(before)) and before != " "
    pad_r = bool(after) and bool(_CJK_RE.fullmatch(after)) and after != " "
    return (" " if pad_l else "") + canonical + (" " if pad_r else "")


def _replace_ascii_aliases(text: str, profiles: dict[str, dict]) -> str:
    """Rewrite spelled-out/close English forms for the configured commands.

    Applies the fuzzy sweep (edit distance <= 2 against a single-word ASCII
    canonical) first so both 可丽儿-free English near-misses and exact aliases
    collapse onto the canonical token."""

    def canon_for(token: str) -> str:
        lowered = token.lower()
        if lowered == _CANONICAL_CLEAR:
            return _CANONICAL_CLEAR
        # The canonical spelling itself (any case, incl. letters spelled out and
        # then joined) always normalizes to the canonical form - no fuzzy needed.
        for canon, profile in profiles.items():
            if not _is_ascii_alias(canon) or " " in canon:
                continue
            if canon.lower() == lowered:
                return canon
        for canon, profile in profiles.items():
            if not profile["fuzzy"]:
                continue
            if not _is_ascii_alias(canon) or " " in canon:
                continue
            if 3 <= len(lowered) <= 8 and _levenshtein(lowered, canon.lower()) <= 2:
                return canon
        for canon, profile in profiles.items():
            for alias in profile["forms"]:
                if alias.lower() == lowered and alias != canon:
                    return canon
        return token

    out: list[str] = []
    cursor = 0
    for match in _ALPHABETIC.finditer(text):
        out.append(text[cursor:match.start()])
        token = match.group(0)
        canon = canon_for(token)
        out.append(_pad_ascii(text, match, canon))
        cursor = match.end()
    out.append(text[cursor:])
    text = "".join(out)

    # Exact multi-word / spelled ASCII aliases that survived as full phrases
    # (org model -> org model etc.) are rewritten by their tolerant pattern.
    for canon, profile in profiles.items():
        for alias in profile["forms"]:
            if not _is_ascii_alias(alias):
                continue
            text = _alias_to_regex(alias).sub(
                lambda m: _pad_ascii(text, m, canon), text
            )
    return text


def _replace_cjk_aliases(text: str, profiles: dict[str, dict]) -> str:
    """Rewrite observed Chinese (or CJK-containing) spellings of the commands."""

    for canon, profile in profiles.items():
        forms = sorted(
            (v for v in profile["forms"] if not _is_ascii_alias(v)),
            key=len,
            reverse=True,
        )
        if not forms:
            continue
        pattern = "|".join(map(re.escape, forms))
        # Case-insensitive so ASCII letters embedded in a CJK alias still match
        # regardless of how the ASR capitalised them (ASR引擎 vs asr引擎).
        text = re.sub(
            pattern,
            lambda m: _pad_ascii(text, m, canon),
            text,
            flags=re.IGNORECASE,
        )
    return text


def normalize_commands(text: str) -> str:
    """Canonicalize observed near-miss spellings of configured voice commands /
    vocabulary in `text`. No configured match => text returned unchanged."""
    if not text:
        return text
    profiles = _load_profiles()
    text = _join_spelled_letters(text)
    text = _replace_ascii_aliases(text, profiles)
    text = _replace_cjk_aliases(text, profiles)
    return text


__all__ = ["normalize_commands"]
