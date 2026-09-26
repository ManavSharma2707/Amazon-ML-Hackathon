"""Step 2 — universal normalisation (master plan SS8 / architecture.md SS4).

Applied identically to every record from every source and country. No
country value, word list per country, or external resource is used: the only
lists are tiny, language-neutral trigger sets (connectors, landmark
triggers, number suffixes) named in the master plan. Everything else is
driven by Unicode properties from the stdlib `unicodedata` module.

Output fields per record (see `normalize_df`):
- `norm_name`, `norm_addr`: casefolded, accent-stripped, Brahmic scripts
  romanised, punctuation -> spaces, dotted initials joined, domains reduced
  to their label, landmark phrase removed from the address.
- `fold_name`, `fold_addr`: phonetic/transliteration fold of the norm text
  (vowel-run reduction, digraphs, doubled consonants). Used *alongside*
  norm features, never instead of them.
- `name_numbers`, `addr_numbers`: space-joined digit-bearing tokens.
- `house_number`: first digit-bearing token of the address.
- `postcodes`: shape-based postcode candidates (4-6 digit runs, or mixed
  letter/digit tokens, in the last two comma segments, never the house
  number). EDA E11 showed a naive "last digit run" mostly finds house
  numbers, hence the position rule.
- `landmark`: trigger + up to 4 tokens of a landmark phrase ("near X").
- `name_romanized`: True if the name contained a Brahmic script that was
  romanised (a later feature, and a bug-detection aid).
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from functools import lru_cache
from multiprocessing import Pool

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Small language-neutral trigger sets (master plan SS8.1). Not country lists.
# ---------------------------------------------------------------------------
_CONNECTORS = {"and", "et", "und", "&"}
# Multi-word triggers are matched as token tuples, longest first.
_LANDMARK_TRIGGERS = sorted(
    [
        ("near",), ("nr",), ("opp",), ("opposite",), ("behind",), ("beside",), ("besides",),
        ("adjacent",), ("next", "to"), ("in", "front", "of"), ("pres", "de"), ("en", "face", "de"),
        ("derriere",), ("a", "cote", "de"),
    ],
    key=len,
    reverse=True,
)
_LANDMARK_MAX_TOKENS = 4
_NULL_TOKENS = {"null", "nan"}  # literal missing-value markers seen inside S2 addresses

_LIGATURES = str.maketrans({"œ": "oe", "æ": "ae", "ß": "ss", "ø": "o", "đ": "d", "ł": "l", "ı": "i"})
_QUOTES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'"})

_COMBINING_LATIN_RE = re.compile(r"[̀-ͯ]+")  # diacritics block (Latin/Greek/Cyrillic only)
_DOMAIN_RE = re.compile(r"^(?:https?://)?(?:www\.)?([^\W_][\w-]*?)(?:\.[a-z]{2,6}){1,2}/?$")
_URL_PREFIX_RE = re.compile(r"(?:https?://)?www\.")
_INITIALS_RE = re.compile(r"(?<![^\W\d_])((?:[^\W\d_]\.){2,})")
_NUM_SUFFIX_RE = re.compile(r"(\d+)\s*[- ]?\s*(bis|ter|quater)(?![^\W\d_])")
_NON_WORD_RE = re.compile(r"[^\w&]|_")
_WS_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d")
_POSTCODE_DIGITS_RE = re.compile(r"^\d{4,6}$")
# Mixed postcodes start with a letter ("sw1a", "h3z"); "3rd"/"12b" are house numbers or ordinals.
_POSTCODE_MIXED_RE = re.compile(r"^[a-z]{1,2}\d[a-z0-9]{0,2}$")

# Phonetic fold rules, applied in order (master plan SS8.1 step 8).
_FOLD_RULES = [
    (re.compile(r"ph"), "f"),
    (re.compile(r"sh"), "s"),
    (re.compile(r"th"), "t"),
    (re.compile(r"kh"), "k"),
    (re.compile(r"gh"), "g"),
    (re.compile(r"ck"), "k"),
    (re.compile(r"qu"), "k"),
    (re.compile(r"w"), "v"),
    (re.compile(r"z"), "s"),
    (re.compile(r"ee"), "i"),
    (re.compile(r"oo"), "u"),
    (re.compile(r"([aiu])\1+"), r"\1"),  # aa->a, ii->i, uu->u
    (re.compile(r"([b-df-hj-np-tv-xz])\1+"), r"\1"),  # doubled consonants
]

# ---------------------------------------------------------------------------
# Brahmic-script romanisation from Unicode character names (stdlib only).
# A script counts as Brahmic if it has a "<SCRIPT> SIGN VIRAMA" character, so
# Devanagari, Kannada, Tamil, Bengali, ... all work without listing them.
# ---------------------------------------------------------------------------
_VOWEL_NAMES = {
    "A", "AA", "I", "II", "U", "UU", "E", "EE", "AI", "O", "OO", "AU",
    "VOCALIC R", "VOCALIC RR", "VOCALIC L", "VOCALIC LL", "SHORT E", "SHORT O",
    "CANDRA E", "CANDRA O",
}
_NAME_PARTS_RE = re.compile(r"^(.*?) (LETTER|VOWEL SIGN|SIGN|DIGIT) (.+)$")


@lru_cache(maxsize=None)
def _is_brahmic_script(script: str) -> bool:
    """True if Unicode defines `<script> SIGN VIRAMA` (the Brahmic-family marker).

    Inputs: script - script prefix of a character name, e.g. "DEVANAGARI".
    Outputs: bool.
    """
    try:
        unicodedata.lookup(f"{script} SIGN VIRAMA")
        return True
    except KeyError:
        return False


def _vowel_latin(vname: str) -> str:
    """Map a vowel name ("AA", "VOCALIC R", "CANDRA E") to a Latin string."""
    parts = vname.split()
    if parts[0] == "VOCALIC":
        return "ri" if parts[-1].startswith("R") else "li"
    v = parts[-1].lower()
    # Long vowels map to one letter: Latin spellings rarely double them.
    return v[0] if len(v) == 2 and v[0] == v[1] else v


@lru_cache(maxsize=None)
def _char_role(ch: str) -> tuple[str, str]:
    """Classify one character for romanisation.

    Inputs: ch - a single character.
    Outputs: (role, latin) where role is one of "cons" (consonant with an
    inherent vowel), "vowel" (independent vowel), "sign" (dependent vowel
    sign), "virama", "nasal", "visarga", "digit", "skip" (nukta etc.), or
    "other" (not a Brahmic character; kept unchanged).
    """
    if ch.isascii():
        return ("other", ch)
    name = unicodedata.name(ch, "")
    m = _NAME_PARTS_RE.match(name)
    if not m or not _is_brahmic_script(m.group(1)):
        return ("other", ch)
    kind, rest = m.group(2), m.group(3)
    if kind == "DIGIT":
        return ("digit", str(unicodedata.digit(ch, 0)))
    if kind == "VOWEL SIGN":
        return ("sign", _vowel_latin(rest))
    if kind == "SIGN":
        if rest == "VIRAMA":
            return ("virama", "")
        if rest in ("ANUSVARA", "CANDRABINDU"):
            return ("nasal", "n")
        if rest == "VISARGA":
            return ("visarga", "h")
        return ("skip", "")
    # LETTER
    if rest in _VOWEL_NAMES:
        return ("vowel", _vowel_latin(rest))
    last = rest.split()[-1].lower()
    if last.endswith("a"):
        base = last[:-1] or last
        # Retroflex consonants are named with a doubled letter (TTA, DDA,
        # NNA); Latin spellings use a single one.
        if len(base) == 2 and base[0] == base[1]:
            base = base[0]
        return ("cons", base)
    return ("other", ch)


@lru_cache(maxsize=200_000)
def romanize_word(word: str) -> str:
    """Romanise one word written in a Brahmic script; other characters pass through.

    Consonants carry an inherent "a" that a vowel sign replaces and a virama
    removes; a word-final inherent "a" is dropped (schwa deletion), which
    matches common Latin spellings ("राम" -> "ram"). The final "a" is kept
    after a conjunct cluster ("राष्ट्र" -> "rashtra"), again as in common
    spellings.

    Inputs: word - one whitespace-free token.
    Outputs: Latin approximation (lowercase), or the word unchanged if it has
    no Brahmic characters.
    """
    out: list[str] = []
    pending_a = False  # a consonant was just emitted and still owns its inherent 'a'
    prev_virama = False  # the previous character was a virama
    in_conjunct = False  # the pending consonant closes a conjunct cluster
    for ch in word:
        role, lat = _char_role(ch)
        if role == "cons":
            if pending_a:
                out.append("a")
            out.append(lat)
            in_conjunct = prev_virama
            pending_a = True
        elif role == "sign":
            out.append(lat)
            pending_a = False
        elif role in ("vowel", "digit", "other", "nasal", "visarga"):
            if pending_a:
                out.append("a")
            out.append(lat)
            pending_a = False
        # "skip": nukta and similar marks change nothing
        prev_virama = role == "virama"
        if prev_virama:
            pending_a = False
    if pending_a and in_conjunct:
        out.append("a")
    return "".join(out)


def _romanize_text(text: str) -> tuple[str, bool]:
    """Romanise every Brahmic-script word in a string.

    Inputs: text - casefolded text.
    Outputs: (romanised text, True if anything was romanised).
    """
    words = text.split()
    changed = False
    for i, w in enumerate(words):
        if not w.isascii():
            r = romanize_word(w)
            if r != w:
                words[i] = r
                changed = True
    return " ".join(words), changed


# ---------------------------------------------------------------------------
# Core text normalisation
# ---------------------------------------------------------------------------
def _unicode_clean(text: str) -> str:
    """NFKC + casefold + ligatures + accent stripping (Latin diacritics only).

    Combining marks in the U+0300-U+036F block are removed after NFKD; marks
    of other scripts (e.g. Devanagari vowel signs) are kept, because deleting
    them would destroy the word (master plan SS8.3: never delete what we
    can't fold).

    Inputs: text - raw string. Outputs: cleaned lowercase string.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    if text.isascii():
        return text
    text = text.translate(_LIGATURES).translate(_QUOTES)
    text = unicodedata.normalize("NFKD", text)
    text = _COMBINING_LATIN_RE.sub("", text)
    return unicodedata.normalize("NFC", text)


def _join_initials(text: str) -> str:
    """Join dotted single-letter initials: "s.b.i." -> "sbi", "o.d." -> "od".

    Inputs: text - lowercase text. Outputs: text with initials joined.
    """
    return _INITIALS_RE.sub(lambda m: m.group(1).replace(".", "") + " ", text)


def _reduce_domain(token: str) -> str:
    """Reduce a web-domain-looking token to its main label ("abc.com" -> "abc").

    Inputs: token - one whitespace-free token.
    Outputs: the label, or the token unchanged if it isn't a domain.
    """
    if "." not in token:
        return token
    m = _DOMAIN_RE.match(token)
    return m.group(1) if m else token


def basic_clean(text: str) -> tuple[str, bool]:
    """Full text normalisation shared by names and addresses (plan SS8.1 steps 1-5, 9).

    Inputs: text - raw field value.
    Outputs: (normalised text with single spaces, romanised flag).
    """
    text = _unicode_clean(text)
    text, romanized = _romanize_text(text) if not text.isascii() else (text, False)
    text = text.replace("&", " & ")
    text = _URL_PREFIX_RE.sub("", text)
    text = " ".join(_reduce_domain(t) for t in text.split())
    text = _join_initials(text)
    text = text.replace("'", "")  # "macy's" -> "macys"
    text = _NUM_SUFFIX_RE.sub(r"\1\2", text)  # "12 bis" -> "12bis"
    text = _NON_WORD_RE.sub(" ", text)
    tokens = []
    for t in text.split():
        if t in _NULL_TOKENS:
            continue
        tokens.append("&" if t in _CONNECTORS else t)
    return " ".join(tokens), romanized


def fold(text: str) -> str:
    """Phonetic/transliteration fold of already-normalised text (plan SS8.1 step 8).

    Digit-bearing tokens are left unchanged.

    Inputs: text - output of `basic_clean`. Outputs: folded text.
    """
    out = []
    for t in text.split():
        if not _DIGIT_RE.search(t):
            for pat, rep in _FOLD_RULES:
                t = pat.sub(rep, t)
        out.append(t)
    return " ".join(out)


def extract_landmark(segments: list[str]) -> tuple[list[str], str]:
    """Move landmark phrases ("near X Y") out of normalised address segments.

    The trigger plus up to 4 following tokens of the same comma segment are
    removed. A trigger right after a house number is kept (it is probably a
    street name such as "12 Near Road"; plan SS8.3).

    Inputs: segments - list of normalised comma segments of one address.
    Outputs: (segments without landmark phrases, landmark text).
    """
    landmark: list[str] = []
    out_segments = []
    for seg in segments:
        toks = seg.split()
        i = 0
        kept: list[str] = []
        while i < len(toks):
            hit = None
            for trig in _LANDMARK_TRIGGERS:
                if tuple(toks[i : i + len(trig)]) == trig:
                    hit = trig
                    break
            after_number = i > 0 and _DIGIT_RE.search(toks[i - 1]) is not None
            if hit and not after_number and i + len(hit) < len(toks):
                end = min(len(toks), i + len(hit) + _LANDMARK_MAX_TOKENS)
                landmark.extend(toks[i:end])
                i = end
            else:
                kept.append(toks[i])
                i += 1
        out_segments.append(" ".join(kept))
    return out_segments, " ".join(landmark)


def number_tokens(text: str) -> list[str]:
    """Return the digit-bearing tokens of normalised text, in order."""
    return [t for t in text.split() if _DIGIT_RE.search(t)]


def postcode_candidates(segments: list[str], house_number: str) -> list[str]:
    """Find postcode-shaped tokens by shape and position, never by country.

    Candidates: 4-6 digit runs, or 2-4-char mixed letter+digit tokens, that
    sit in the last two comma segments and are not the house number.

    Inputs: segments - normalised comma segments; house_number - to exclude.
    Outputs: list of candidate tokens (possibly empty).
    """
    cands = []
    for seg in segments[-2:] if len(segments) > 1 else []:
        for t in seg.split():
            if t == house_number:
                continue
            if _POSTCODE_DIGITS_RE.match(t) or _POSTCODE_MIXED_RE.match(t):
                cands.append(t)
    return cands


def normalize_record(name: str, address: str) -> dict:
    """Normalise one record's name and address into all derived fields.

    Inputs: name, address - raw strings (may be empty).
    Outputs: dict with keys norm_name, norm_addr, fold_name, fold_addr,
    name_numbers, addr_numbers, house_number, postcodes, landmark,
    name_romanized.
    """
    norm_name, name_rom = basic_clean(name)
    # Addresses are normalised per comma segment so landmark and postcode
    # rules can use segment boundaries.
    segs = []
    for seg in address.split(","):
        s, _ = basic_clean(seg)
        if s:
            segs.append(s)
    segs, landmark = extract_landmark(segs)
    segs = [s for s in segs if s]
    norm_addr = " ".join(segs)
    addr_nums = number_tokens(norm_addr)
    house = addr_nums[0] if addr_nums else ""
    return {
        "norm_name": norm_name,
        "norm_addr": norm_addr,
        "fold_name": fold(norm_name),
        "fold_addr": fold(norm_addr),
        "name_numbers": " ".join(number_tokens(norm_name)),
        "addr_numbers": " ".join(addr_nums),
        "house_number": house,
        "postcodes": " ".join(postcode_candidates(segs, house)),
        "landmark": landmark,
        "name_romanized": name_rom,
    }


_OUT_COLS = [
    "norm_name", "norm_addr", "fold_name", "fold_addr", "name_numbers",
    "addr_numbers", "house_number", "postcodes", "landmark", "name_romanized",
]


def _normalize_chunk(pairs: list[tuple[str, str]]) -> dict[str, list]:
    """Worker: normalise a chunk of (name, address) pairs into column lists."""
    cols: dict[str, list] = {c: [] for c in _OUT_COLS}
    for name, addr in pairs:
        rec = normalize_record(name, addr)
        for c in _OUT_COLS:
            cols[c].append(rec[c])
    return cols


def normalize_df(
    df: pd.DataFrame, source: str, n_jobs: int = 1, chunk: int = 50_000, log_every: int = 20
) -> pd.DataFrame:
    """Normalise a raw source DataFrame (S1/S2/S3) into the NB02 record schema.

    Inputs: df - raw source frame (entity_id, business_name, business_address,
            country); source - "S1"/"S2"/"S3"; n_jobs - worker processes;
            chunk - records per worker task; log_every - log each N chunks.
    Outputs: new DataFrame with entity_id, source, country, raw_name,
    raw_addr, the `normalize_record` fields, and name_empty/addr_empty flags.
    """
    t0 = time.time()
    pairs = list(zip(df["business_name"].tolist(), df["business_address"].tolist()))
    chunks = [pairs[i : i + chunk] for i in range(0, len(pairs), chunk)]
    cols: dict[str, list] = {c: [] for c in _OUT_COLS}

    def _collect(i: int, part: dict[str, list]) -> None:
        for c in _OUT_COLS:
            cols[c].extend(part[c])
        if (i + 1) % log_every == 0 or i + 1 == len(chunks):
            print(
                f"    normalize {source}: {min((i + 1) * chunk, len(pairs)):,}/{len(pairs):,} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    if n_jobs > 1 and len(chunks) > 1:
        with Pool(n_jobs) as pool:
            for i, part in enumerate(pool.imap(_normalize_chunk, chunks)):
                _collect(i, part)
    else:
        for i, c in enumerate(chunks):
            _collect(i, _normalize_chunk(c))

    out = pd.DataFrame(
        {
            "entity_id": df["entity_id"].to_numpy(),
            "source": source,
            "country": df["country"].to_numpy(),
            "raw_name": df["business_name"].to_numpy(),
            "raw_addr": df["business_address"].to_numpy(),
            **cols,
        }
    )
    out["name_empty"] = out["norm_name"].eq("")
    out["addr_empty"] = out["norm_addr"].eq("")
    return out


def main() -> None:
    """CLI smoke test: normalise a few strings given on the command line."""
    import sys

    for s in sys.argv[1:] or ["S.B.I. Café & Co., near City Mall, 12 bis Rue X"]:
        print(s, "->", normalize_record(s, s))


if __name__ == "__main__":
    main()
