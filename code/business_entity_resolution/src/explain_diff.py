"""Explain-the-difference features (master plan SS13).

For one field (name or address) of a record pair, every token is either
*explained* by a token (or token span) on the other side through a typed
relation (exact, fold, typo, abbreviation, split/join, initialism, number
equality), or *unexplained*. Rare unexplained content words are the main
negative signal; the relation counts tell the model what kind of noise
separates the two records. Everything is language-free: relations are tests
on letters and positions, never word lists.

Pure Python (rapidfuzz is used for edit distances when installed, with a
stdlib fallback so the module also runs without it). Token relations are
cached per process because the vocabulary is small relative to the number of
pairs (memory.md E13).
"""

from __future__ import annotations

import re

from . import blocking, normalize

def _py_dl(a: str, b: str) -> int:
    """Optimal-string-alignment Damerau-Levenshtein distance (stdlib fallback)."""
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[len(b)]


def _py_jw(a: str, b: str) -> float:
    """Jaro-Winkler similarity (stdlib fallback, prefix scale 0.1, max prefix 4)."""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return 0.0
    win = max(max(la, lb) // 2 - 1, 0)
    ma, mb = [False] * la, [False] * lb
    m = 0
    for i, ch in enumerate(a):
        for j in range(max(0, i - win), min(lb, i + win + 1)):
            if not mb[j] and b[j] == ch:
                ma[i] = mb[j] = True
                m += 1
                break
    if not m:
        return 0.0
    sa = [a[i] for i in range(la) if ma[i]]
    sb = [b[j] for j in range(lb) if mb[j]]
    t = sum(x != y for x, y in zip(sa, sb)) / 2
    jaro = (m / la + m / lb + (m - t) / m) / 3
    p = 0
    while p < min(4, la, lb) and a[p] == b[p]:
        p += 1
    return jaro + p * 0.1 * (1 - jaro)


_DIST: tuple | None = None


def _distances() -> tuple:
    """(damerau_levenshtein, jaro_winkler) functions: rapidfuzz (C++) if importable, else the stdlib fallbacks.

    Resolved on first use, not at import: Kaggle kernels install rapidfuzz from
    offline wheels after the bundled modules are loaded. Both give the same answers.
    """
    global _DIST
    if _DIST is None:
        try:
            from rapidfuzz.distance import DamerauLevenshtein, JaroWinkler

            _DIST = (DamerauLevenshtein.distance, JaroWinkler.similarity)
        except ImportError:
            _DIST = (_py_dl, _py_jw)
    return _DIST


# Prior strengths used only to order links in the alignment (plan SS13.1);
# the model sees counts per relation type and learns their real value.
STRENGTH = {
    "exact": 1.00,
    "fold_exact": 0.95,
    "numeric_equal": 0.95,
    "split_join": 0.90,
    "typo": 0.85,
    "prefix_abbrev": 0.80,
    "initialism": 0.80,
    "skeleton_abbrev": 0.70,
    "short_abbrev": 0.45,  # 2-letter abbreviations ("st", "bd"): ambiguous, own counter (plan SS13.5)
    "none": 0.0,
}
_ABBREV = ("prefix_abbrev", "skeleton_abbrev")
_NUM_TOKEN = re.compile(r"^(\d+)([a-z]*)$")
_CACHE: dict[tuple[str, str], str] = {}
_CACHE_MAX = 4_000_000
_JOIN_MAX = 3  # split_join: one token vs up to 3 consecutive tokens
_INIT_MAX_SPAN = 6  # initialism: letters matched over at most 6 consecutive tokens
_INIT_SKIP_LEN = 3  # tokens this short may be skipped inside an initialism ("of", "&", "de")


def _num_base(tok: str) -> str | None:
    """Digit base of a number-like token ("12bis" -> "12", "03" -> "3"); None if not number-like."""
    m = _NUM_TOKEN.match(tok)
    if not m:
        return None
    return m.group(1).lstrip("0") or "0"


_DIGIT = re.compile(r"\d")


def _has_digit(tok: str) -> bool:
    """True if the token contains any digit."""
    return _DIGIT.search(tok) is not None


def _is_subsequence(short: str, long: str) -> bool:
    """True if `short`'s letters appear in `long` in order (plan SS28.4)."""
    it = iter(long)
    return all(ch in it for ch in short)


_FOLD_CACHE: dict[str, str] = {}


def _fold_token(tok: str) -> str:
    """Look-alike digits back to letters, then the phonetic fold of normalize.py (cached)."""
    f = _FOLD_CACHE.get(tok)
    if f is None:
        if len(_FOLD_CACHE) >= _CACHE_MAX:
            _FOLD_CACHE.clear()
        f = _FOLD_CACHE[tok] = normalize.fold(blocking.deleet(tok))
    return f


def _relation(a: str, b: str) -> str:
    """Uncached relation between two single tokens (see `token_relation`)."""
    if a == b:
        return "exact"
    da, db = blocking.deleet(a), blocking.deleet(b)
    na, nb = _has_digit(da), _has_digit(db)
    if na or nb:
        # Numbers are never typos or abbreviations of each other: 12 vs 13 is a conflict.
        ba, bb = _num_base(da), _num_base(db)
        if ba is not None and ba == bb:
            return "numeric_equal"
        return "exact" if da == db else "none"
    if da == db:
        return "fold_exact"
    fa, fb = _fold_token(a), _fold_token(b)
    if fa == fb:
        return "fold_exact"
    s, l = (da, db) if len(da) <= len(db) else (db, da)
    if len(s) >= 3:
        lim = 1 if len(l) <= 5 else 2
        dl, jw = _distances()
        if dl(da, db) <= lim or dl(fa, fb) <= lim or jw(da, db) >= 0.92:
            return "typo"
    if len(s) < 2 or len(s) >= len(l):
        return "none"
    if l.startswith(s):
        return "short_abbrev" if len(s) == 2 else "prefix_abbrev"
    if s[0] == l[0] and len(s) <= 0.7 * len(l) and _is_subsequence(s, l):
        return "short_abbrev" if len(s) == 2 else "skeleton_abbrev"
    return "none"


def token_relation(a: str, b: str) -> str:
    """Relation type between two normalised tokens (symmetric, cached).

    Tests, strongest first (plan SS13.1): exact; fold_exact (same after
    look-alike-digit and phonetic fold); numeric_equal (same digit base,
    e.g. 12 vs 12bis); typo (Damerau-Levenshtein <= 1 for length <= 5, <= 2
    above, or Jaro-Winkler >= 0.92; both tokens >= 3 letters); prefix_abbrev
    and skeleton_abbrev (same first letter, ordered subsequence, length ratio
    <= 0.7); abbreviations of only 2 letters become the weak `short_abbrev`.
    Multi-token relations (split_join, initialism) are found in `align`.

    Inputs: a, b - tokens. Outputs: relation name (a key of STRENGTH).
    """
    key = (a, b) if a <= b else (b, a)
    r = _CACHE.get(key)
    if r is None:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.clear()
        r = _CACHE[key] = _relation(*key)
    return r


def _initialism_span(acr: str, toks: list[str], free: list[bool]) -> list[int] | None:
    """Token positions whose initials spell `acr`, over consecutive free tokens.

    Short tokens (<= 3 chars, e.g. "of", "&") may be skipped inside the span,
    so "sbi" matches "state bank of india" without any word list.
    Inputs: acr - candidate initialism (2-5 letters); toks; free - usable flags.
    Outputs: list of positions (first letters consumed), or None.
    """
    n = len(toks)
    for start in range(n):
        if not free[start] or not toks[start] or toks[start][0] != acr[0]:
            continue
        used, k, j = [], 0, start
        while j < n and j - start < _INIT_MAX_SPAN and k < len(acr) and free[j]:
            t = toks[j]
            if t and t[0] == acr[k]:
                used.append(j)
                k += 1
            elif len(t) > _INIT_SKIP_LEN or k == 0:
                break
            j += 1
        if k == len(acr) and len(used) >= 2:
            return list(range(start, used[-1] + 1))
    return None


def _join_span(tok: str, toks: list[str], free: list[bool]) -> list[int] | None:
    """Positions of 2-3 consecutive free tokens whose concatenation equals `tok` (exact or folded)."""
    ft = None
    n = len(tok)
    for start in range(len(toks)):
        cat = ""
        for j in range(start, min(len(toks), start + _JOIN_MAX)):
            if not free[j]:
                break
            cat += toks[j]
            if j > start and abs(len(cat) - n) <= 2:
                if cat == tok:
                    return list(range(start, j + 1))
                if ft is None:
                    ft = _fold_token(tok)
                if _fold_token(cat) == ft:
                    return list(range(start, j + 1))
            if len(cat) >= n + 2:
                break
    return None


def align(t1: list[str], t2: list[str]) -> dict:
    """Align two token lists and label every token explained/unexplained (plan SS13.2).

    Steps: (1) exact matches; (2) multi-token links, split_join and
    initialism, in both directions among the remaining tokens; (3) greedy
    maximum-strength 1-to-1 links among the rest (tokens are few, so greedy
    by strength gives the same result as the Hungarian step in practice and
    is much faster).

    Inputs: t1, t2 - token lists of the same field of two records.
    Outputs: dict with
      links: list of (positions_1, positions_2, relation);
      rel1/rel2: relation per token (None = unexplained).
    """
    n1, n2 = len(t1), len(t2)
    rel1: list[str | None] = [None] * n1
    rel2: list[str | None] = [None] * n2
    links: list[tuple[list[int], list[int], str]] = []

    # (1) exact matches first, left to right
    pos2: dict[str, list[int]] = {}
    for j, t in enumerate(t2):
        pos2.setdefault(t, []).append(j)
    for i, t in enumerate(t1):
        js = pos2.get(t)
        if js:
            j = js.pop(0)
            rel1[i] = rel2[j] = "exact"
            links.append(([i], [j], "exact"))
    if all(rel1) and all(rel2):
        return {"links": links, "rel1": rel1, "rel2": rel2}

    # (2) multi-token relations among unexplained tokens
    for src, dst, rs, rd, flip in ((t1, t2, rel1, rel2, False), (t2, t1, rel2, rel1, True)):
        for i, tok in enumerate(src):
            if rs[i] is not None or _has_digit(tok):
                continue
            free = [r is None for r in rd]
            span = _join_span(tok, dst, free) if len(tok) >= 4 else None
            rel = "split_join"
            if span is None and 2 <= len(tok) <= 5:
                span = _initialism_span(tok, dst, free)
                rel = "initialism"
            if span is not None:
                rs[i] = rel
                for j in span:
                    rd[j] = rel
                links.append((span, [i], rel) if flip else ([i], span, rel))

    # (3) greedy 1-to-1 links among the rest
    free1 = [i for i in range(n1) if rel1[i] is None]
    free2 = [j for j in range(n2) if rel2[j] is None]
    if free1 and free2:
        cand = []
        for i in free1:
            a = t1[i]
            for j in free2:
                r = token_relation(a, t2[j])
                if r != "none":
                    # tie-break: prefer links between close positions (keeps order)
                    cand.append((-STRENGTH[r], abs(i / max(n1, 1) - j / max(n2, 1)), i, j, r))
        cand.sort()
        for _, _, i, j, r in cand:
            if rel1[i] is None and rel2[j] is None:
                rel1[i] = rel2[j] = r
                links.append(([i], [j], r))
    return {"links": links, "rel1": rel1, "rel2": rel2}


def _kendall_tau(pairs: list[tuple[int, int]]) -> float:
    """Kendall tau between aligned positions (1.0 = same order); NaN if < 2 links."""
    n = len(pairs)
    if n < 2:
        return float("nan")
    conc = disc = 0
    for x in range(n):
        for y in range(x + 1, n):
            s = (pairs[x][0] - pairs[y][0]) * (pairs[x][1] - pairs[y][1])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    tot = conc + disc
    return (conc - disc) / tot if tot else float("nan")


FIELD_FEATURES = [
    "expl_frac_idf", "expl_frac_idf_min", "unexpl_max_idf_1", "unexpl_max_idf_2", "unexpl_sum_idf_1",
    "unexpl_sum_idf_2", "n_exact", "n_fold", "n_typo", "n_abbrev", "n_short_abbrev", "n_initialism",
    "n_splitjoin", "n_numeq", "n_dropped_generic", "n_unexpl", "n_unexpl_1", "n_unexpl_2", "order_tau",
    "core_equal", "first_explained", "ntok_1", "ntok_2",
]


def explain_features(
    t1: list[str], t2: list[str], idf1: list[float], idf2: list[float], generic1: list[bool], generic2: list[bool]
) -> tuple[list[float], dict]:
    """Features for one field of one pair (plan SS13.3), plus the alignment.

    Unexplained tokens flagged generic (suffix-/street-type-like, decided by
    corpus statistics, not word lists) count as `dropped_generic`; the others
    are unexplained content, the strong negative evidence.

    Inputs: t1, t2 - tokens; idf1, idf2 - IDF per token; generic1/2 - flags.
    Outputs: (values in FIELD_FEATURES order, alignment dict from `align`).
    """
    al = align(t1, t2)
    rel1, rel2 = al["rel1"], al["rel2"]
    tot1, tot2 = sum(idf1), sum(idf2)
    ex1 = sum(w for w, r in zip(idf1, rel1) if r is not None)
    ex2 = sum(w for w, r in zip(idf2, rel2) if r is not None)
    nan = float("nan")
    frac = (ex1 + ex2) / (tot1 + tot2) if tot1 + tot2 > 0 else nan
    frac_min = min(ex1 / tot1 if tot1 > 0 else 1.0, ex2 / tot2 if tot2 > 0 else 1.0) if t1 and t2 else nan
    u1 = [w for w, r, g in zip(idf1, rel1, generic1) if r is None and not g]
    u2 = [w for w, r, g in zip(idf2, rel2, generic2) if r is None and not g]
    n_gen = sum(1 for r, g in zip(rel1, generic1) if r is None and g) + sum(
        1 for r, g in zip(rel2, generic2) if r is None and g
    )
    cnt = {"exact": 0, "fold_exact": 0, "typo": 0, "abbrev": 0, "short_abbrev": 0, "initialism": 0,
           "split_join": 0, "numeric_equal": 0}
    one_to_one = []
    for p1, p2, r in al["links"]:
        cnt["abbrev" if r in _ABBREV else r] += 1
        if len(p1) == 1 and len(p2) == 1:
            one_to_one.append((p1[0], p2[0]))
    first = float(bool(t1) and bool(t2) and rel1[0] is not None and rel2[0] is not None)
    vals = [
        frac, frac_min,
        max(u1) if u1 else 0.0, max(u2) if u2 else 0.0, sum(u1), sum(u2),
        cnt["exact"], cnt["fold_exact"], cnt["typo"], cnt["abbrev"], cnt["short_abbrev"], cnt["initialism"],
        cnt["split_join"], cnt["numeric_equal"], n_gen, len(u1) + len(u2), len(u1), len(u2),
        _kendall_tau(one_to_one),
        float(not u1 and not u2), first, len(t1), len(t2),
    ]
    return vals, al


# ---------------------------------------------------------------------------
# Number agreement (plan SS13.4)
# ---------------------------------------------------------------------------
# Relation codes (ordinal on purpose: missing < equal-ish < conflict is not
# implied; LightGBM splits on the codes freely).
REL_BOTH_MISSING, REL_ONE_MISSING, REL_EQUAL, REL_BASE_EQUAL, REL_PREFIX_EQUAL, REL_CONFLICT = 0, 1, 2, 3, 4, 5

NUMBER_FEATURES = [
    "house_no_rel", "postcode_rel", "unit_rel", "name_number_rel", "any_number_conflict", "house_in_other",
    "n_shared_numbers", "jaccard_numbers", "n_numbers_1", "n_numbers_2",
]


def _bases(tokens: list[str]) -> set[str]:
    """Digit bases of the number-like tokens in a list."""
    out = set()
    for t in tokens:
        b = _num_base(t)
        if b is not None:
            out.add(b)
    return out


def _set_rel(a: set[str], b: set[str]) -> int:
    """equal (share any) / conflict (both present, disjoint) / missing codes for two number sets."""
    if not a and not b:
        return REL_BOTH_MISSING
    if not a or not b:
        return REL_ONE_MISSING
    return REL_EQUAL if a & b else REL_CONFLICT


def house_relation(h1: str, h2: str) -> int:
    """House-number relation: equal / base-equal (12 vs 12bis, 12 vs 12b) / conflict / missing."""
    if not h1 and not h2:
        return REL_BOTH_MISSING
    if not h1 or not h2:
        return REL_ONE_MISSING
    if h1 == h2:
        return REL_EQUAL
    b1, b2 = _num_base(h1), _num_base(h2)
    if b1 is not None and b1 == b2:
        return REL_BASE_EQUAL
    return REL_CONFLICT


def postcode_relation(p1: list[str], p2: list[str]) -> int:
    """Postcode relation: equal / first-3-characters equal / conflict / missing."""
    if not p1 and not p2:
        return REL_BOTH_MISSING
    if not p1 or not p2:
        return REL_ONE_MISSING
    if set(p1) & set(p2):
        return REL_EQUAL
    if {x[:3] for x in p1 if len(x) >= 4} & {x[:3] for x in p2 if len(x) >= 4}:
        return REL_PREFIX_EQUAL
    return REL_CONFLICT


def number_features(r1: dict, r2: dict) -> list[float]:
    """Number agreement features of a record pair (plan SS13.4).

    Inputs: r1, r2 - dicts with house_number (str), postcodes, addr_numbers,
            name_numbers (space-separated strings, as written by normalize.py).
    Outputs: values in NUMBER_FEATURES order.
    """
    h1, h2 = r1["house_number"], r2["house_number"]
    pc1, pc2 = r1["postcodes"].split(), r2["postcodes"].split()
    a1, a2 = r1["addr_numbers"].split(), r2["addr_numbers"].split()
    nn1, nn2 = _bases(r1["name_numbers"].split()), _bases(r2["name_numbers"].split())
    # unit/other numbers = address numbers that are neither the house number nor a postcode
    u1 = _bases([t for t in a1 if t != h1 and t not in pc1])
    u2 = _bases([t for t in a2 if t != h2 and t not in pc2])
    hr = house_relation(h1, h2)
    pr = postcode_relation(pc1, pc2)
    ur = _set_rel(u1, u2)
    nr = _set_rel(nn1, nn2)
    all1 = _bases(a1) | nn1
    all2 = _bases(a2) | nn2
    shared = all1 & all2
    union = all1 | all2
    hb1, hb2 = _num_base(h1) if h1 else None, _num_base(h2) if h2 else None
    house_in_other = float((hb1 is not None and hb1 in all2) or (hb2 is not None and hb2 in all1))
    conflict = float(
        (hr == REL_CONFLICT and not house_in_other) or pr == REL_CONFLICT or nr == REL_CONFLICT
    )
    return [
        hr, pr, ur, nr, conflict, house_in_other,
        len(shared), len(shared) / len(union) if union else float("nan"), len(all1), len(all2),
    ]


def main() -> None:
    """Print the relation of a few example token pairs (smoke test)."""
    for a, b in [("corp", "corporation"), ("pvt", "private"), ("bd", "boulevard"), ("st", "south"),
                 ("12", "12bis"), ("glypheus", "glypheos")]:
        print(f"{a!r} ~ {b!r}: {token_relation(a, b)}")
    print(align("state bank of india".split(), ["sbi"])["links"])
    print(align(["walmart"], "wal mart".split())["links"])


if __name__ == "__main__":
    main()
