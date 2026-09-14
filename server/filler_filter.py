"""Deterministic spoken-filler cleanup (model-free).

Removes colloquial filler words (嗯/呃/啊/那个/就是说...) that the ASR keeps
verbatim, without an LLM. Same design language as itn_digits.py /
command_canon.py: static word tables plus positional guards, applied as one
step of the deterministic transcription pipeline.

The ASR often transcribes speech without inserting pauses, so fillers arrive
GLUED to their neighbours (啊那官方的呃飞猪云 / 版本啊嗯而且...). Blanket string
matching would corrupt real words, so removal is decided per interjection
character by its role:

  - Pure stall-only characters (呃/嗯/诶) are almost never grammatical - they
    only mean "er/um". They are dropped everywhere, glued or not:
      官方的呃飞猪云 -> 官方的飞猪云    嗯，我觉得可以 -> 我觉得可以

  - Ambiguous characters (啊/呀/哦/唉/哎...) are ALSO particles and
    exclamations (好的啊 / 是呀 / 对哦). They survive when they carry meaning:
      - a single one glued between real words:   好的啊 / 我说啊，这个不行 -> kept
      - a single one trailing at a sentence end: 三点了啊 -> kept
    They are dropped when they only stall:
      - run-on utterance start glued to content: 啊那... -> 那...  (啊+那)
      - a cluster of two+ interjection characters between two clauses
        (版本啊嗯而且): the cluster is a pause -> replaced by "，"
      - before a clause connector such as 而且/然后/因为
        (贡献了啊然后 -> 贡献了，然后)

  Whole-word fillers keep their own rules:
  - Demonstratives 这个/那个 only leave when reduplicated (这个这个...) or
    segment-initial before a pause; 就选那个。/ 拿这个来 keep the pronoun. A
    trailing echo after an interjection is also dropped (火啊这个 -> 火啊).
  - Phrase interjections (哎呀/哎哟/哎呦) and discourse stalls (就是说/
    也就是说/怎么说呢) only leave in isolation; glued 哎哟好疼 survives.

A final tidy pass cleans up the punctuation/spacing the removals leave behind
(，，-> ，; 。，-> 。; leading "，" dropped) without ever crossing a newline.
"""

from __future__ import annotations

import re

# Characters that count as real content when glued to a candidate: CJK
# ideographs, CJK numerals fall in that range, full-width + ASCII letters and
# digits. Anything else (punctuation, spaces, line edges) is a boundary.
_WORD = "\u3400-\u4dbf\u4e00-\u9fff\uff21-\uff3a\uff41-\uff5a\uff10-\uff19A-Za-z0-9"
_WORD_CHAR_RE = re.compile(rf"[{_WORD}]")

# Weak clause punctuation (a pause), strong sentence-enders.
_WEAK = "，,、：:"
_STRONG = "。.！!？?；;"

# Interjection characters, split by how safe they are to delete:
#   - _PURE_STALL: only ever a stall sound (呃=er, 嗯=um/uh-huh, 诶=hey) -
#     never a grammatical particle, so deletable everywhere.
#   - _AMBIGUOUS: 啊/呀/哦/... double as sentence-final particles and
#     exclamations (好的啊, 是呀, 对哦); they must only go when clearly stalling.
# Whole-word exclamations (哎呀/哎哟/哎呦) sit on top of ambiguous characters.
_PURE_STALL = "嗯呃诶"
_AMBIGUOUS = "啊呀哦噢唉哎喔哟"
_INTERJ_CHARS = _PURE_STALL + _AMBIGUOUS

# ASR stutter: the engine occasionally emits one character twice, glued to its
# neighbours (这这 / 有有) or separated by a pause the model punctuated away
# (这、这、，我觉得). Only characters that are essentially never a legitimate 叠词
# (reduplication) may be collapsed; content words that really double (妈妈 / 看看
# / 刚刚) and emphatic or onomatopoeic runs (哈哈哈) are deliberately left alone.
_NEVER_DOUBLE = (
    "我你您他她它这那哪的了吗呢吧嘛呀"
    "是在有就都也还又才很不没别再已"
    "被给让从向到与及或和而但因由且并则若虽"
    "之于以其所该各每此咱俺怎可"
)
# Pause characters/whitespace a stutter may hide behind (never a sentence ender:
# 这。这 stays two sentences). Horizontal whitespace only - a newline is a line
# boundary and must not be crossed. A trailing pause run is captured so a
# collapsed utterance-initial stutter (这、这、，我觉得) can drop the whole thing.
_STUTTER_GAP = r"[，,、：:；;… \t\f\v　]*"
_STUTTER_RE = re.compile(rf"([{_NEVER_DOUBLE}])(?:{_STUTTER_GAP}\1)+(?:([{_WEAK}]+))?")
# The model sometimes punctuates a stutter with a sentence ender (可。可能 ->
# 可能). A strong ender is crossed only when the echoed character starts a
# longer word, so a genuine sentence pair (这。这。) is never joined.
_STRONG_STUTTER_RE = re.compile(rf"([{_NEVER_DOUBLE}])[{_STRONG}…]+(?=\1[{_WORD}])")
# Legitimate words that merely contain a doubled character the stutter pass
# would otherwise collapse (可可西里 -> 可西里). 可可 is itself a real word
# (cocoa), so bare 可可 stays ambiguous and is NOT listed; only fixed names are
# shielded. Longest first so 可可西里 wins over shorter prefixes.
_PROTECTED_REDUPLICATIONS = (
    "可可西里", "可可托海", "可可豆", "可可粉", "可可脂", "可可树", "可可碱", "可可油",
)
_PROTECT_WORDS_RE = re.compile(
    "|".join(map(re.escape, sorted(_PROTECTED_REDUPLICATIONS, key=len, reverse=True)))
)
_PROTECT_PH = "\ue010"


def _protect_legit_words(text: str) -> tuple[str, list[str]]:
    saved: list[str] = []

    def guard(match: re.Match[str]) -> str:
        saved.append(match.group(0))
        return f"{_PROTECT_PH}{len(saved) - 1}{_PROTECT_PH}"

    return _PROTECT_WORDS_RE.sub(guard, text), saved


def _restore_legit_words(text: str, saved: list[str]) -> str:
    for i, word in enumerate(saved):
        text = text.replace(f"{_PROTECT_PH}{i}{_PROTECT_PH}", word)
    return text.replace(_PROTECT_PH, "")

# Whole-word interjections; dropped in isolation, kept when glued to content
# (哎哟好疼 still means "ouch").
_PHRASE_INTERJECTIONS = ("哎呀", "哎哟", "哎呦")

# Discourse connectors used purely to stall; removed only in isolation. Bare
# 然后/就是 are deliberately absent - they usually carry real meaning.
_DISCOURSE_MARKERS = ("也就是说", "就是说", "怎么说呢", "怎么说")

# Reduplicated demonstratives are stall fillers; single ones are pronouns and
# are handled separately (see module docstring).
_DEICTIC_FILLERS = ("这个", "那个")

# Connectors that START a new clause. An interjection glued right in front of
# one is a stall, and the dropped slot becomes a clause comma: 贡献了啊然后 ->
# 贡献了，然后. Sorted longest-first so 然后呢 wins over 然后.
_CLAUSE_CONNECTORS = (
    "而且", "然后呢", "然后", "但是呢", "但是", "可是", "因为", "所以", "于是",
    "那么", "接着", "还有呢", "还有", "此外", "另外呢", "另外", "因而", "因此",
    "不过呢", "不过", "然而", "只是", "只要", "如果", "既然", "并且", "甚至",
    "就是说",
)

# Common English/non-Chinese stall words, for bilingual speech. Case-insensitive
# and only matched as whole tokens, so real words are never clipped.
_ASCII_FILLERS = (
    "um", "umm", "uuh", "uh", "uhm", "mm", "mhm", "hm", "hmm", "eh", "ah", "oh", "er",
)

_WORD_LB = rf"(?<![{_WORD}])"
_PAUSE_OR_END = rf"(?=[{_WEAK}{_STRONG}\s…—]|$)"

_DEICTIC_RUN_RE = re.compile(
    rf"{_WORD_LB}(?:{'|'.join(_DEICTIC_FILLERS)}){{2,}}(?![{_WORD}])"
)
# Single demonstrative only when segment-initial (left neighbour is a boundary)
# AND followed by a pause/end: 那个，我们先走 -> remove; 就选那个。/ 拿那个来 -> keep.
_DEICTIC_SINGLE_RE = re.compile(rf"{_WORD_LB}(?:{'|'.join(_DEICTIC_FILLERS)}){_PAUSE_OR_END}")
# Phrase interjections / discourse markers, isolation-only.
_PHRASE_RE = re.compile(
    rf"{_WORD_LB}(?:{'|'.join(sorted(_PHRASE_INTERJECTIONS + _DISCOURSE_MARKERS, key=len, reverse=True))}){_PAUSE_OR_END}",
)
_ASCII_RE = re.compile(
    rf"(?<![A-Za-z])(?:{'|'.join(sorted(_ASCII_FILLERS, key=len, reverse=True))})(?![A-Za-z])",
    re.IGNORECASE,
)

# Maximal runs of consecutive interjection characters, the unit the stall
# detector works on (版本啊嗯而且 -> the whole 啊嗯 is one cluster).
_RUN_RE = re.compile(rf"[{_INTERJ_CHARS}]+")

# A trailing demonstrative glued to an interjection echo: 火啊这个 -> 火啊.
_TOPIC_TAIL_RE = re.compile(
    rf"(?<=[{_INTERJ_CHARS}])(?:{'|'.join(_DEICTIC_FILLERS)})(?=[{_STRONG}\s…]|$)"
)


def _starts_with_connector(text: str) -> bool:
    return any(text.startswith(c) for c in _CLAUSE_CONNECTORS)


def _glued_cluster_replacement(text: str, i: int, j: int) -> str:
    """Decide what an interjection-char run becomes (itself / "" / "，").

    `text[i:j]` is a maximal run of interjection characters; look at its
    neighbours in the ORIGINAL text to classify the stall:
      - never delete whole-word exclamations glued to real content (哎哟好疼);
      - a run standing between two real words and stalling a clause boundary
        (版本啊嗯而且 / GPL3.0啊嗯目前) becomes "，";
      - an ambiguous single trapped inside a phrase is a real particle and stays
        (好啊 / 我说啊) unless a clause connector follows (贡献了啊然后 -> 贡献了，);
      - pure stall characters (呃/嗯) and any leading/trailing filler go away.
    """
    token = text[i:j]
    prev = text[i - 1] if i > 0 else ""
    nxt = text[j] if j < len(text) else ""
    left_word = bool(prev) and bool(_WORD_CHAR_RE.fullmatch(prev))
    right_word = bool(nxt) and bool(_WORD_CHAR_RE.fullmatch(nxt))
    conn = right_word and _starts_with_connector(text[j:])
    # Whole-word exclamations keep their meaning when glued to content.
    if token in _PHRASE_INTERJECTIONS and (left_word or right_word):
        return token
    single = len(token) == 1
    pure = any(ch in token for ch in _PURE_STALL)
    if left_word and right_word:
        # Deep inside real words - only a clear clause stall may touch it.
        if single:
            if pure:
                return "，" if conn else ""  # 官方的呃飞猪云 -> delete 呃
            return "，" if conn else token  # 好的啊 -> keep 啊
        # A multi-char cluster between two clauses is a spoken pause.
        return "，" if (pure or conn) else ""
    if right_word:
        return ""  # clause/utterance start glued to content: 啊那... -> 那...
    if left_word and single and not pure:
        return token  # trailing particle: 三点了啊 / 好吧呀
    return ""  # anything trailing or isolated (incl. 嗯，/ 走啊哦哦。)


def _strip_interjection_clusters(text: str) -> str:
    out: list[str] = []
    cursor = 0
    for match in _RUN_RE.finditer(text):
        out.append(text[cursor:match.start()])
        cursor = match.end()
        out.append(_glued_cluster_replacement(text, match.start(), match.end()))
    out.append(text[cursor:])
    return "".join(out)


_HSPACE = r"[ \t\f\v]"
_HSPACES = rf"{_HSPACE}+"

# Full-width (CJK) punctuation. A space may follow one only when joining mixed
# text, so trailing-space cleanup applies here but never to ASCII "..." / ". "
# which keep their normal English spacing.
_FULL_PUNCT = "，。！？；：、…"

# Same-char duplication is only ever a deletion artifact for weak pauses
# (，， / ,,); ASCII "..." and emphatic ?? / !! must survive untouched. Full-width
# 。 alone is deduped too ("好。嗯。走。" -> "好。走。" after the 嗯 is dropped).
_WEAK_DUP = "，,、：:"
_STRONG_DUP = "。"


def _collapse_space(text: str) -> str:
    """Collapse horizontal whitespace only - a newline is a line boundary and
    must survive so later passes (rule numbering, paragraphing) still see it."""
    lines = []
    for raw in text.split("\n"):
        line = re.sub(_HSPACES, " ", raw)
        # No space before any punctuation (CJK or ASCII).
        line = re.sub(rf"{_HSPACES}([{_WEAK}{_STRONG}])", r"\1", line)
        # After a full-width pause no run-on space is wanted.
        line = re.sub(rf"([{_FULL_PUNCT}]){_HSPACES}", r"\1", line)
        lines.append(line)
    return "\n".join(lines)


def _drop_duplicate_punct(text: str) -> str:
    return re.sub(rf"([{_WEAK_DUP}{_STRONG_DUP}])(?:{_HSPACE}*\1)+", r"\1", text)


def _resolve_punct_runs(text: str) -> str:
    """When a strong ender sits next to a weak pause (，。 / 。， / ；，...),
    keep the strong one and drop the weak one."""
    for _ in range(3):
        updated = re.sub(rf"([{_STRONG}]){_HSPACE}*([{_WEAK}])", r"\1", text)
        updated = re.sub(rf"([{_WEAK}]){_HSPACE}*([{_STRONG}])", r"\2", updated)
        if updated == text:
            break
        text = updated
    return text


def _tidy(text: str) -> str:
    text = _collapse_space(text)
    text = _drop_duplicate_punct(text)
    text = _resolve_punct_runs(text)
    lines = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        line = re.sub(rf"^[{_WEAK}]+", "", line)
        line = re.sub(rf"[{_WEAK}]+$", "", line)
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def collapse_stutter(text: str) -> str:
    """Collapse repeated function-word characters from ASR stutter.

    Handles both glued repeats (这这 -> 这) and the same character split by a
    pause the model punctuated (这、这、，我觉得 -> 我觉得, 他说这、这、对 -> 他说这，对).
    Only characters that are never a valid bare 叠词 are touched, so real
    reduplication (妈妈 / 看看 / 刚刚) survives untouched.
    """
    if not text:
        return text
    # Shield fixed names (可可西里) so the stutter rules cannot eat their 可可.
    text, protected = _protect_legit_words(text)
    # A stutter punctuated with a sentence ender (可。可能) is dropped first:
    # the echoed character must start a longer word, so 这。这。 stays intact.
    text = _STRONG_STUTTER_RE.sub("", text)

    def repl(match: re.Match[str]) -> str:
        char = match.group(1)
        trailing = match.group(2)
        if trailing:
            line_start = text.rfind("\n", 0, match.start()) + 1
            if not text[line_start:match.start()].strip():
                return ""  # utterance-initial stutter + pause = a filler
            return f"{char}，"  # mid-sentence: keep one char and one pause
        return char

    return _restore_legit_words(_STUTTER_RE.sub(repl, text), protected)


def filter_fillers(text: str) -> str:
    """Drop spoken fillers from `text`; leave meaningful words untouched.

    Stall-only characters (嗯/呃/诶) leave everywhere. 啊/呀/哦 and friends stay
    as real particles (好的啊 / 我说啊) but leave as utterance-start stalls,
    multi-char stall clusters between clauses (replaced by a comma) or filler
    before a clause connector. Whole words keep their guards (这个那个/哎呀...).
    """
    if not text:
        return text
    text = collapse_stutter(text)
    for pattern in (
        _PHRASE_RE,  # isolated 也就是说/怎么说呢/哎呀 first.
        _DEICTIC_RUN_RE,
        _DEICTIC_SINGLE_RE,
    ):
        text = pattern.sub("", text)
    text = _strip_interjection_clusters(text)
    text = _TOPIC_TAIL_RE.sub("", text)
    text = _ASCII_RE.sub("", text)
    return _tidy(text).strip()


__all__ = ["filter_fillers", "collapse_stutter"]
