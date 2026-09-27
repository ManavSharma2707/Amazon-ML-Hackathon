"""Step V7.1a -- script detection + learned transliteration dictionary.

Basis (`reports/diag.md` D3/D7): ~half of sampled blocking misses are
cross-script (Devanagari/Bengali S2/S3 name vs Latin S1 name). `normalize.py`
already romanises Brahmic scripts character-by-character (Unicode-name based,
stdlib only), but that is a *phonetic* transliteration, not the spelling an
English loanword actually uses in the data (e.g. a business-name token spelled
"future" on the Latin side may not char-romanise to "future" from its
Devanagari counterpart). This module learns a token-level correction *from
train positives only* (never test, never any external corpus) and applies it
on top of the existing normalisation.

Compliance: no external data or lookup. The dictionary is built purely from
counting token co-occurrence inside TRAIN ground-truth positive pairs. The
fallback transliterator for tokens with no dictionary entry is `anyascii`
(ISC licence; see memory.md SS4), a rule-based Unicode-to-ASCII table with no
external corpus dependency (unlike `unidecode`, GPL, or `libpostal`, trained
on external OSM/OpenAddresses data -- both are excluded by CLAUDE.md SS3).
"""

from __future__ import annotations

import logging
import time
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import Iterable

from . import normalize

log = logging.getLogger(__name__)

try:
    from anyascii import anyascii as _anyascii
except ImportError:  # pragma: no cover - resolved lazily so bundled kernels
    # that install wheels after module import time still work (memory.md
    # pitfall: "never bind an optional import at module import time").
    _anyascii = None


def _ascii_fallback(token: str) -> str:
    """Lazily-resolved `anyascii` call (see the import-time note above)."""
    global _anyascii
    if _anyascii is None:
        from anyascii import anyascii as _anyascii_fn

        _anyascii = _anyascii_fn
    return _anyascii(token)


_NAME_FIRST_WORD_RE_CACHE: dict[str, str] = {}


@lru_cache(maxsize=200_000)
def script_of_char(ch: str) -> str:
    """Return a character's Unicode script family, or "LATIN"/"OTHER".

    Inputs: ch - a single character.
    Outputs: "LATIN" for ASCII letters/digits/punctuation, "OTHER" for
    characters whose Unicode name doesn't start with a recognisable script
    word (rare symbols), else the leading word of its Unicode name (e.g.
    "DEVANAGARI", "BENGALI", "HAN", "CYRILLIC") -- this generalises to any
    script without a hard-coded list, the same trick `normalize.py` uses for
    Brahmic detection.
    """
    if ch.isascii():
        return "LATIN"
    name = unicodedata.name(ch, "")
    if not name:
        return "OTHER"
    return name.split()[0]


def script_of_token(token: str) -> str:
    """Return the dominant non-Latin script of a token, or "LATIN".

    A token counts as non-Latin if any letter in it resolves to a non-Latin,
    non-"OTHER" script; ties broken by the most frequent script among its
    characters. Digits/punctuation don't count either way.

    Inputs: token - one whitespace-free string (already casefolded).
    Outputs: script name, e.g. "LATIN", "DEVANAGARI", "BENGALI".
    """
    if token.isascii():
        return "LATIN"
    counts: Counter = Counter()
    for ch in token:
        if ch.isalpha():
            s = script_of_char(ch)
            if s not in ("LATIN", "OTHER"):
                counts[s] += 1
    if not counts:
        return "LATIN"
    return counts.most_common(1)[0][0]


def has_non_latin(text: str) -> bool:
    """True if any token of `text` has a non-Latin dominant script."""
    return any(script_of_token(t) != "LATIN" for t in text.split())


def transliterate_token(token: str, dictionary: dict[str, str]) -> str:
    """Map one token to Latin: dictionary hit first, else `anyascii`.

    Inputs: token - a single token (any script); dictionary - learned
    native_token -> latin_token map (see `build_dictionary`).
    Outputs: Latin-script token. Latin input is returned unchanged.
    """
    if token.isascii():
        return token
    hit = dictionary.get(token)
    if hit is not None:
        return hit
    return _ascii_fallback(token).strip().lower() or token


def transliterate_text(text: str, dictionary: dict[str, str]) -> str:
    """Apply `transliterate_token` to every token of `text`, space-joined."""
    return " ".join(transliterate_token(t, dictionary) for t in text.split())


def build_dictionary(
    pairs: Iterable[tuple[list[str], list[str]]],
    min_count: int = 3,
    min_dice: float = 0.5,
) -> dict[str, str]:
    """Learn native_token -> latin_token mappings from TRAIN positive pairs.

    For each ground-truth positive pair where one side has at least one
    non-Latin token, count how often each native token co-occurs (within the
    same pair, either field) with each Latin token, plus each token's total
    pair count. Keep (native, latin) with co-occurrence count >= `min_count`
    and Dice coefficient >= `min_dice`; each native token keeps only its best
    (highest Dice, then count) Latin partner.

    Inputs: pairs - iterable of (native_tokens, latin_tokens): the non-Latin
        side's tokens and the Latin side's tokens of one positive pair
        (name + address tokens pooled together by the caller). min_count,
        min_dice - thresholds (tuned on Half A, checked on Half B per the
        prompt). Outputs: dict native_token -> latin_token.
    """
    t0 = time.time()
    native_pair_count: Counter = Counter()
    latin_pair_count: Counter = Counter()
    co_count: Counter = Counter()
    n_pairs = 0
    for native_tokens, latin_tokens in pairs:
        n_pairs += 1
        natives = {t for t in native_tokens if script_of_token(t) != "LATIN"}
        latins = {t for t in latin_tokens if t.isascii() and t}
        if not natives or not latins:
            continue
        for n in natives:
            native_pair_count[n] += 1
        for l in latins:
            latin_pair_count[l] += 1
        for n in natives:
            for l in latins:
                co_count[(n, l)] += 1
        if n_pairs % 500_000 == 0:
            print(f"    translit.build_dictionary: {n_pairs:,} pairs scanned ({time.time() - t0:.0f}s)", flush=True)

    best: dict[str, tuple[float, int, str]] = {}
    for (n, l), c in co_count.items():
        if c < min_count:
            continue
        dice = 2.0 * c / (native_pair_count[n] + latin_pair_count[l])
        if dice < min_dice:
            continue
        prev = best.get(n)
        if prev is None or (dice, c) > (prev[0], prev[1]):
            best[n] = (dice, c, l)
    out = {n: l for n, (_, _, l) in best.items()}
    print(
        f"    translit.build_dictionary: {n_pairs:,} pairs, {len(native_pair_count):,} native tokens, "
        f"{len(out):,} kept (min_count={min_count}, min_dice={min_dice}), {time.time() - t0:.0f}s",
        flush=True,
    )
    return out


def _split_any_script(text: str) -> list[str]:
    """Split into word-ish tokens, keeping combining marks attached to their base letter.

    `normalize._NON_WORD_RE` (`[^\\w&]|_`) relies on Python's `\\w`, which
    excludes Unicode category Mn (combining marks, e.g. Devanagari virama /
    vowel signs) -- fine for `normalize.py` because it only ever runs on
    already-romanised (pure-ASCII) text, but wrong here since `clean_tokens`
    deliberately works on raw, un-romanised script. Splitting on `\\w` alone
    would shatter "फ्यूचर" into several pieces
    at each combining mark. Categories L (letter), M (mark), N (number) and
    "&" are kept as token characters; everything else (punctuation,
    separators, symbols) splits.
    """
    tokens: list[str] = []
    cur: list[str] = []
    for ch in text:
        if ch == "&" or unicodedata.category(ch)[0] in ("L", "M", "N"):
            cur.append(ch)
        elif cur:
            tokens.append("".join(cur))
            cur = []
    if cur:
        tokens.append("".join(cur))
    return tokens


def clean_tokens(raw_text: str) -> list[str]:
    """Tokenise raw text WITHOUT romanising it (unlike `normalize.basic_clean`).

    NFKC + casefold, domain-label reduction ("abc.com" -> "abc", "www." strip),
    punctuation -> space, null-token drop. Kept deliberately close to
    `normalize.basic_clean`'s pipeline minus the romanisation step, so the
    native-script tokens this function returns are what a learned dictionary
    should be keyed on (the un-romanised original spelling).

    Inputs: raw_text - a raw field value (name or address).
    Outputs: list of tokens, any script, lowercase.
    """
    text = unicodedata.normalize("NFKC", raw_text).casefold()
    text = normalize._URL_PREFIX_RE.sub("", text)
    text = " ".join(normalize._reduce_domain(t) for t in text.split())
    return [t for t in _split_any_script(text) if t not in normalize._NULL_TOKENS]


def translit_field(raw_text: str, dictionary: dict[str, str]) -> str:
    """Transliterate + fold one raw field into a Latin `translit_*` field.

    Latin input passes through clean_tokens + fold unchanged (so the S1 side,
    which E10 found is 100% Latin, gets a directly comparable field for
    free). Non-Latin tokens use the learned dictionary, falling back to
    `anyascii` (see module docstring).

    Inputs: raw_text - raw name/address; dictionary - from `build_dictionary`.
    Outputs: space-joined Latin text, phonetically folded (`normalize.fold`).
    """
    tokens = [transliterate_token(t, dictionary) for t in clean_tokens(raw_text)]
    return normalize.fold(" ".join(tokens))


def add_translit_columns(df, dictionary: dict[str, str], n_jobs: int = 1, chunk: int = 50_000, log_every: int = 20):
    """Add `translit_name` / `translit_addr` columns to a records DataFrame, chunked + parallel.

    Mirrors `normalize.normalize_df`'s chunking pattern (same reason: millions
    of rows, Python-level per-record work).

    Inputs: df - records frame with raw_name, raw_addr; dictionary; n_jobs;
            chunk; log_every. Outputs: df with the two new columns added
            (mutates and returns the same object).
    """
    from multiprocessing import Pool

    t0 = time.time()
    pairs = list(zip(df["raw_name"].tolist(), df["raw_addr"].tolist()))
    chunks = [pairs[i : i + chunk] for i in range(0, len(pairs), chunk)]

    def _work(part):
        return (
            [translit_field(n, dictionary) for n, _ in part],
            [translit_field(a, dictionary) for _, a in part],
        )

    names, addrs = [], []

    def _collect(i, part):
        n, a = part
        names.extend(n)
        addrs.extend(a)
        if (i + 1) % log_every == 0 or i + 1 == len(chunks):
            print(f"    translit: {min((i + 1) * chunk, len(pairs)):,}/{len(pairs):,} ({time.time() - t0:.0f}s)", flush=True)

    if n_jobs > 1 and len(chunks) > 1:
        with Pool(n_jobs) as pool:
            for i, part in enumerate(pool.imap(_work, chunks)):
                _collect(i, part)
    else:
        for i, c in enumerate(chunks):
            _collect(i, _work(c))
    df["translit_name"] = names
    df["translit_addr"] = addrs
    return df


def build_dictionary_from_pairs(s1: "pd.DataFrame", pool: "pd.DataFrame", pairs: "pd.DataFrame", s1_id_mask: set,
                                 min_count: int = 3, min_dice: float = 0.5) -> dict[str, str]:
    """Build the translit dictionary from TRAIN ground-truth positives, restricted to `s1_id_mask` (Half A).

    For each (s1_id, match_id) positive pair with s1_id in `s1_id_mask`: the
    Latin side's tokens come from the S1 row's already-normalised norm_name +
    norm_addr (clean and, per E10, always Latin); the native side's tokens
    come from the match row's RAW name + address via `clean_tokens` (i.e.
    un-romanised, so genuinely native-script tokens survive as dictionary
    keys).

    Inputs: s1 - S1 records frame (entity_id, norm_name, norm_addr); pool -
            S2+S3 records frame (entity_id, raw_name, raw_addr); pairs -
            gt.parquet (s1_id, match_id); s1_id_mask - set of Half-A s1_ids;
            min_count, min_dice - thresholds passed to `build_dictionary`.
    Outputs: dict native_token -> latin_token.
    """
    pr = pairs[pairs["s1_id"].isin(s1_id_mask)]
    s1_small = s1[["entity_id", "norm_name", "norm_addr"]].rename(columns={"entity_id": "s1_id"})
    pool_small = pool[["entity_id", "raw_name", "raw_addr"]].rename(columns={"entity_id": "match_id"})
    merged = pr.merge(s1_small, on="s1_id", how="inner").merge(pool_small, on="match_id", how="inner")
    latin_texts = (merged["norm_name"] + " " + merged["norm_addr"]).tolist()
    raw_names = merged["raw_name"].tolist()
    raw_addrs = merged["raw_addr"].tolist()

    def gen():
        for lt, rn, ra in zip(latin_texts, raw_names, raw_addrs):
            yield clean_tokens(rn) + clean_tokens(ra), lt.split()

    return build_dictionary(gen(), min_count=min_count, min_dice=min_dice)


def save_dictionary(dictionary: dict[str, str], path: str) -> None:
    """Save the dictionary as a two-column TSV (native_token, latin_token)."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("native_token\tlatin_token\n")
        for n, l in dictionary.items():
            f.write(f"{n}\t{l}\n")


def load_dictionary(path: str) -> dict[str, str]:
    """Load a dictionary saved by `save_dictionary`."""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            n, l = line.rstrip("\n").split("\t", 1)
            out[n] = l
    return out


def main() -> None:
    """CLI smoke test on a tiny synthetic example."""
    pairs = [
        (["फ्यूचर", "टेक्नोलॉजी"], ["future", "technology", "pvt", "ltd"]),
    ]
    d = build_dictionary(pairs * 5, min_count=3, min_dice=0.1)
    print({k.encode("unicode_escape").decode("ascii"): v for k, v in d.items()})


if __name__ == "__main__":
    main()
