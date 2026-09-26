"""Evidence-augmented prompts for the LLM judge (master plan SS16.3-16.5).

Each prompt shows the two raw records plus automatically computed evidence
(token alignment with relation types, unexplained words, number agreement,
landmarks, string similarities), so the judge weighs structured evidence that
transfers across languages. The country field is deliberately NOT shown: all
candidate pairs are within one country, and the only country signal the
pipeline may use is the equality flag (CLAUDE.md SS3 rule 5).

Training data (Half A): stage-1 OOF probability in the band [0.05, 0.95] plus
a random 10% of confident pairs, about 1 positive : 2 negatives, 12k-20k
examples, with augmentations (A/B order swap 50%, address dropped on one side
5%, letter scramble 7%). Inference data (Half B, test): stage-1 probability in
[0.10, 0.90], most uncertain first, capped by a pair budget.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import explain_diff, features, normalize, scramble

SYSTEM = (
    "You decide whether two business records describe the same real-world business "
    "(same legal entity at the same location). Differences in abbreviations, legal "
    "suffixes, word order, typos, transliteration and missing address parts are normal. "
    "Different branches of the same chain at different addresses are NOT the same business. "
    'Answer only "Yes" or "No".'
)
_REL_LABEL = {"fold_exact": "spelling variant", "numeric_equal": "same number", "typo": "typo",
              "prefix_abbrev": "abbreviation", "skeleton_abbrev": "abbreviation", "short_abbrev": "short abbreviation",
              "split_join": "split/joined", "initialism": "initials"}
_HOUSE = {0: "both missing", 1: "one missing", 2: "equal", 3: "same number, different suffix", 5: "CONFLICT"}
_POST = {0: "both missing", 1: "one missing", 2: "equal", 4: "same first 3 digits", 5: "CONFLICT"}
_UNIT = {0: "none", 1: "one side only", 2: "equal", 5: "CONFLICT"}
MAX_RAW = 160  # characters of raw text shown per field (prompt budget: 384 tokens)
REC_FIELDS = ["raw_name", "raw_addr"] + features.REC_COLS[1:]


def _clip(s: str, n: int = MAX_RAW) -> str:
    """Raw text for the prompt: single-line, quotes neutralised, clipped to n characters."""
    s = " ".join(str(s).replace('"', "'").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _field_evidence(t1: list[str], t2: list[str], generic: set) -> tuple[str, str]:
    """(alignment line, unexplained-words line) for one field."""
    al = explain_diff.align(t1, t2)
    n_exact = sum(1 for _, _, r in al["links"] if r == "exact")
    parts = [f"{n_exact} exact"] if n_exact else []
    for p1, p2, r in al["links"]:
        if r != "exact":
            parts.append(f"'{' '.join(t1[i] for i in p1)}'~'{' '.join(t2[j] for j in p2)}' ({_REL_LABEL.get(r, r)})")
    un1 = [t + ("(generic)" if t in generic else "") for t, r in zip(t1, al["rel1"]) if r is None]
    un2 = [t + ("(generic)" if t in generic else "") for t, r in zip(t2, al["rel2"]) if r is None]
    return "; ".join(parts[:8]) or "none", f"A: {' '.join(un1[:8]) or '-'} | B: {' '.join(un2[:8]) or '-'}"


def evidence(r1: dict, r2: dict, lookups: dict) -> str:
    """Evidence block (plan SS16.3) for two normalised records (dicts with REC_FIELDS)."""
    from rapidfuzz import fuzz

    n_al, n_un = _field_evidence(r1["norm_name"].split(), r2["norm_name"].split(), lookups["generic_name"])
    a_al, a_un = _field_evidence(r1["norm_addr"].split(), r2["norm_addr"].split(), lookups["generic_addr"])
    num = dict(zip(explain_diff.NUMBER_FEATURES, explain_diff.number_features(r1, r2)))
    h1, h2 = r1["house_number"] or "-", r2["house_number"] or "-"
    p1, p2 = r1["postcodes"] or "-", r2["postcodes"] or "-"
    lines = [
        f"- Name alignment: {n_al}",
        f"- Unexplained name words: {n_un}",
        f"- Address alignment: {a_al}",
        f"- Unexplained address words: {a_un}",
        f"- Numbers: house {h1} vs {h2}: {_HOUSE.get(int(num['house_no_rel']), '?')}; postcode {p1} vs {p2}: "
        f"{_POST.get(int(num['postcode_rel']), '?')}; other numbers: {_UNIT.get(int(num['unit_rel']), '?')}",
        f"- Landmarks: A: {r1['landmark'] or 'none'} | B: {r2['landmark'] or 'none'}",
        f"- Similarity: name {fuzz.token_set_ratio(r1['norm_name'], r2['norm_name']) / 100:.2f}, "
        f"address {fuzz.token_set_ratio(r1['norm_addr'], r2['norm_addr']) / 100:.2f}",
    ]
    return "\n".join(lines)


def user_text(r1: dict, r2: dict, lookups: dict) -> str:
    """User turn: both raw records + evidence + the question."""
    return (f'Record A: name = "{_clip(r1["raw_name"])}" | address = "{_clip(r1["raw_addr"])}"\n'
            f'Record B: name = "{_clip(r2["raw_name"])}" | address = "{_clip(r2["raw_addr"])}"\n\n'
            f"Evidence (automatically computed):\n{evidence(r1, r2, lookups)}\nSame business?")


def _renorm(raw_name: str, raw_addr: str) -> dict:
    """Normalise raw text again (after an augmentation changed it)."""
    rec = normalize.normalize_record(raw_name, raw_addr)
    rec["raw_name"], rec["raw_addr"] = raw_name, raw_addr
    return rec


def select_train_band(oof: pd.DataFrame, lo: float = 0.05, hi: float = 0.95, confident_frac: float = 0.10,
                      neg_per_pos: float = 2.0, max_n: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """Half-A training pairs for the judge (plan SS16.4).

    Inputs: oof - s1_id, cand_id, label, p1 (stage-1 OOF on A); band limits;
            confident_frac - share of out-of-band pairs added at random;
            neg_per_pos - negatives per positive; max_n - cap; seed.
    Outputs: selected rows with a `band` flag.
    """
    rng = np.random.default_rng(seed)
    inb = oof[(oof["p1"] >= lo) & (oof["p1"] <= hi)].assign(band=1)
    out = oof[(oof["p1"] < lo) | (oof["p1"] > hi)]
    n_conf = int(confident_frac * len(inb))
    conf = out.iloc[rng.choice(len(out), min(n_conf, len(out)), replace=False)].assign(band=0)
    sel = pd.concat([inb, conf], ignore_index=True)
    pos, neg = sel[sel["label"] == 1], sel[sel["label"] == 0]
    n_pos = min(len(pos), int(max_n / (1 + neg_per_pos)))
    n_neg = min(len(neg), int(n_pos * neg_per_pos), max_n - n_pos)
    pos = pos.iloc[rng.choice(len(pos), n_pos, replace=False)]
    neg = neg.iloc[rng.choice(len(neg), n_neg, replace=False)]
    return pd.concat([pos, neg], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def select_infer_band(p1: pd.DataFrame, lo: float = 0.10, hi: float = 0.90, max_n: int | None = None) -> pd.DataFrame:
    """Band pairs for inference, most uncertain first (|p - 0.5| ascending), capped at max_n (plan SS16.5)."""
    b = p1[(p1["p1"] >= lo) & (p1["p1"] <= hi)].copy()
    b = b.iloc[np.argsort(np.abs(b["p1"].to_numpy() - 0.5), kind="stable")]
    return (b.iloc[:max_n] if max_n else b).reset_index(drop=True)


def build_dataset(pairs: pd.DataFrame, s1: pd.DataFrame, pool: pd.DataFrame, lookups: dict, augment: bool = False,
                  seed: int = 42, p_swap: float = 0.5, p_drop_addr: float = 0.05, p_scramble: float = 0.07,
                  log_every: int = 20_000, log=print) -> pd.DataFrame:
    """Prompt table for a set of pairs.

    Inputs: pairs - s1_id, cand_id (+ label, p1); s1, pool - record frames
            with REC_FIELDS; lookups - features.token_lookups output;
            augment - apply the training augmentations (plan SS16.4).
    Outputs: DataFrame s1_id, cand_id, [label, p1,] system, user, aug.
    """
    rng = np.random.default_rng(seed)
    s1i = s1.set_index("entity_id")
    pi = pool.set_index("entity_id")
    table = scramble.make_scrambler(seed + 1)
    users, augs = [], []
    for n, (a, b) in enumerate(zip(pairs["s1_id"].tolist(), pairs["cand_id"].tolist())):
        r1 = {k: s1i.at[a, k] for k in REC_FIELDS}
        r2 = {k: pi.at[b, k] for k in REC_FIELDS}
        tag = []
        if augment:
            if rng.random() < p_drop_addr:  # address missing on one side
                if rng.random() < 0.5:
                    r1 = _renorm(r1["raw_name"], "")
                else:
                    r2 = _renorm(r2["raw_name"], "")
                tag.append("drop_addr")
            if rng.random() < p_scramble:  # same letter permutation on both records (text and evidence)
                r1 = _renorm(r1["raw_name"].translate(table), r1["raw_addr"].translate(table))
                r2 = _renorm(r2["raw_name"].translate(table), r2["raw_addr"].translate(table))
                tag.append("scramble")
            if rng.random() < p_swap:
                r1, r2 = r2, r1
                tag.append("swap")
        users.append(user_text(r1, r2, lookups))
        augs.append(",".join(tag))
        if log_every and (n + 1) % log_every == 0:
            log(f"    prompts {n + 1:,}/{len(pairs):,}")
    keep = [c for c in ("s1_id", "cand_id", "label", "p1", "band") if c in pairs]
    out = pairs[keep].reset_index(drop=True).copy()
    out["system"] = SYSTEM
    out["user"] = users
    out["aug"] = augs
    return out


def main() -> None:
    """Print one example prompt (smoke test)."""
    r1 = _renorm("State Bank of India Pvt Ltd", "12 MG Road, Near City Mall, Pune 411001")
    r2 = _renorm("SBI Private Limited", "12bis M.G. Rd, Pune")
    lk = {"generic_name": {"ltd", "limited", "pvt", "private"}, "generic_addr": {"road", "rd"}}
    print(SYSTEM + "\n\n" + user_text(r1, r2, lk))


if __name__ == "__main__":
    main()
