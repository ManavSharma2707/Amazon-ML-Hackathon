"""Collective features from stage-1 probabilities (master plan SS17), for the combiner.

Implemented:
- within-entity context (SS17.2): rank_in_entity, gap_to_top, n_above_05,
  entropy of the S1's normalised probabilities, top1-top2 margin;
- sibling support (SS17.3): for candidate r of S1 e, the confident siblings
  Conf(e) = e's other candidates with stage-1 p >= 0.7; max name / address
  similarity of r to them, number agreement, count.

Not used on purpose: cross-entity competition counts (SS17.1). Half B only
queries a sample of train S1s (~11-18% of them), while on test every S1
competes, so claimant counts and margins would shift between training and
test in a way LOCO cannot detect. Cross-entity conflicts are handled by the
one-owner rule (exclusivity.py) instead. `competition_features` is kept for
diagnostics only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ENTITY_FEATURES = ["rank_in_entity", "gap_to_top", "n_above_05", "entropy", "top_margin", "p_share"]
SIBLING_FEATURES = ["sib_count", "sib_max_name_sim", "sib_max_addr_sim", "sib_number_agree"]
COLLECTIVE_FEATURES = ENTITY_FEATURES + SIBLING_FEATURES
SIB_P = 0.7


def _groups(s1: np.ndarray):
    """Start index and size of runs of equal S1 ids (input must be grouped by S1)."""
    starts = np.flatnonzero(np.r_[True, s1[1:] != s1[:-1]]) if len(s1) else np.empty(0, int)
    return starts, np.diff(np.r_[starts, len(s1)])


def entity_features(s1: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Within-entity context features, rows grouped by S1 (any order inside a group).

    Outputs: float32 (n, len(ENTITY_FEATURES)).
    """
    n = len(p)
    starts, sizes = _groups(s1)
    gid = np.repeat(np.arange(len(starts)), sizes)
    order = np.lexsort((-p, gid))
    rank = np.empty(n, np.float32)
    rank[order] = np.arange(n) - np.repeat(starts, sizes)
    top = np.maximum.reduceat(p, starts) if n else np.empty(0)
    second = np.zeros(len(starts))
    ps = p[order]
    has2 = sizes >= 2
    second[has2] = ps[starts[has2] + 1]
    tot = np.add.reduceat(p, starts) if n else np.empty(0)
    q = p / np.maximum(np.repeat(tot, sizes), 1e-12)
    ent = -np.add.reduceat(np.where(q > 0, q * np.log(np.maximum(q, 1e-12)), 0.0), starts) if n else np.empty(0)
    n05 = np.add.reduceat((p > 0.5).astype(np.float64), starts) if n else np.empty(0)
    out = np.column_stack([rank, p - np.repeat(top, sizes), np.repeat(n05, sizes), np.repeat(ent, sizes),
                           np.repeat(top - second, sizes), q])
    return out.astype(np.float32)


def sibling_features(s1: np.ndarray, p: np.ndarray, name: np.ndarray, addr: np.ndarray, house: np.ndarray,
                     post: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    """Sibling-support features (plan SS17.3), rows grouped by S1.

    Inputs: s1 ids, stage-1 p, and per-row candidate record fields (norm name,
            norm address, house number, postcodes as strings).
    Outputs: float32 (n, len(SIBLING_FEATURES)); similarities NaN when there is no confident sibling.
    """
    from rapidfuzz import fuzz, process

    n = len(p)
    out = np.full((n, len(SIBLING_FEATURES)), np.nan, dtype=np.float32)
    starts, sizes = _groups(s1)
    conf = p >= SIB_P
    gid = np.repeat(np.arange(len(starts)), sizes)
    n_conf = np.add.reduceat(conf.astype(np.int64), starts) if n else np.empty(0, np.int64)
    out[:, 0] = np.repeat(n_conf, sizes) - conf  # siblings exclude the row itself
    # all (row, sibling) index pairs inside each S1: rows x confident rows of the same S1, minus self
    conf_idx = np.flatnonzero(conf)
    conf_g = gid[conf_idx]
    cstart = np.searchsorted(conf_g, np.arange(len(starts)))  # first confident row of each group
    ccount = n_conf[gid]
    rows = np.repeat(np.arange(n), ccount)
    off = np.arange(len(rows)) - np.repeat(np.cumsum(ccount) - ccount, ccount)
    sib = conf_idx[np.repeat(cstart[gid], ccount) + off]
    keep = rows != sib
    rows, sib = rows[keep], sib[keep]
    if not len(rows):
        return out
    nm = np.full(n, -1.0)
    ad = np.full(n, -1.0)
    na = np.zeros(n)
    for s in range(0, len(rows), chunk):
        r, b = rows[s : s + chunk], sib[s : s + chunk]
        sn = process.cpdist(name[r].tolist(), name[b].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
        sa = process.cpdist(addr[r].tolist(), addr[b].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
        hn = (house[r] == house[b]) & (house[r] != "")
        pc = (post[r] == post[b]) & (post[r] != "")
        np.maximum.at(nm, r, sn)
        np.maximum.at(ad, r, sa)
        np.maximum.at(na, r, (hn | pc).astype(np.float64))
    has = nm >= 0
    out[has, 1], out[has, 2], out[has, 3] = nm[has], ad[has], na[has]
    return out


def competition_features(s1: np.ndarray, cand: np.ndarray, p: np.ndarray, p_min: float = 0.05) -> pd.DataFrame:
    """Diagnostics only (see module docstring): claimants per record and each pair's margin to the best other claimant."""
    d = pd.DataFrame({"s1": s1, "cand": cand, "p": p})
    act = d["p"] > p_min
    g = d[act].groupby("cand")["p"]
    n_cl = d["cand"].map(g.size()).fillna(0).to_numpy()
    best = d["cand"].map(g.max()).fillna(0).to_numpy()
    return pd.DataFrame({"n_claimants": n_cl, "margin_to_best": d["p"].to_numpy() - best})


def main() -> None:
    """Tiny example (smoke test)."""
    s1 = np.array(["a", "a", "a", "b"])
    p = np.array([0.9, 0.8, 0.1, 0.4])
    print(entity_features(s1, p))


if __name__ == "__main__":
    main()
