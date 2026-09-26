"""Official metric scorer and validation harnesses.

Implements the exact macro F0.5 metric from the problem statement (master
plan SS28.1), plus bootstrap significance testing for the component gates
(CLAUDE.md SS4.3). LOCO and blocking-report helpers are stubs for now; they
are filled in once blocking/combiner code exists (Prompt 2+).
"""

from __future__ import annotations

import random
from typing import Callable


def f05_entity(pred: set, true: set) -> float:
    """Score one S1 entity's predicted match set against its true match set.

    Inputs: pred - predicted set of matched IDs; true - ground-truth set.
    Outputs: 1.0 if both empty, 0.0 if exactly one is empty, else the F0.5 of
    the overlap (a false positive costs 4x a false negative).
    """
    if not true and not pred:
        return 1.0
    if not true or not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def f05_macro(pred_map: dict[str, set], true_map: dict[str, set]) -> float:
    """Macro-average `f05_entity` over every S1 entity in `true_map`.

    Inputs: pred_map - {s1_id: predicted set}; true_map - {s1_id: true set}
            (this is the authoritative list of entities to score).
    Outputs: mean F0.5 across all entities in `true_map`.
    """
    ids = list(true_map)
    return sum(f05_entity(pred_map.get(i, set()), true_map[i]) for i in ids) / len(ids)


def per_entity_scores(pred_map: dict[str, set], true_map: dict[str, set]) -> dict[str, float]:
    """Return the per-entity F0.5 scores (used for bootstrap and error buckets).

    Inputs/outputs: same maps as `f05_macro`; returns {s1_id: score}.
    """
    return {i: f05_entity(pred_map.get(i, set()), true_map[i]) for i in true_map}


def bootstrap_diff(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap the mean-score difference (b - a) over shared S1 entities.

    Used for the component gates (CLAUDE.md SS4.3): a component is kept only
    if the 90% CI of (new - old) excludes 0, or the point gain is >= +0.003.

    Inputs: scores_a, scores_b - {s1_id: per-entity F0.5} for the baseline and
            the candidate variant (must share the same entity IDs);
            n_resamples - bootstrap resample count; seed - RNG seed.
    Outputs: dict with `mean_diff`, `ci90` (tuple), and `excludes_zero` (bool).
    """
    import numpy as np

    ids = sorted(set(scores_a) & set(scores_b))
    assert ids, "scores_a and scores_b share no S1 entities"
    diffs = np.array([scores_b[i] - scores_a[i] for i in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(diffs)
    # vectorised resampling in blocks (1000 x 250k entities would be slow in pure Python)
    means = np.concatenate([diffs[rng.integers(0, n, size=(min(100, n_resamples - k), n))].mean(axis=1)
                            for k in range(0, n_resamples, 100)])
    lo, hi = np.quantile(means, [0.05, 0.95])
    return {
        "mean_diff": float(diffs.mean()),
        "ci90": (float(lo), float(hi)),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def loco_eval(
    train_fn: Callable,
    eval_fn: Callable,
    entities_by_country: dict[str, list[str]],
) -> dict:
    """Leave-one-country-out evaluation: train on country X, score on country Y, every ordered pair.

    Inputs: train_fn(ids) -> fitted artifact; eval_fn(artifact, ids) -> mean
            F0.5 on those ids; entities_by_country - {country: [s1_id, ...]}
            (Half B). Country labels are only grouping keys here.
    Outputs: {"X->Y": score, ..., "mean": mean over directions}.
    """
    out: dict = {}
    for x, ids_x in entities_by_country.items():
        art = train_fn(ids_x)
        for y, ids_y in entities_by_country.items():
            if x != y:
                out[f"{x}->{y}"] = float(eval_fn(art, ids_y))
    vals = [v for v in out.values()]
    out["mean"] = sum(vals) / len(vals) if vals else float("nan")
    return out


def blocking_metrics(
    candidates: dict[str, set[str]],
    true_map: dict[str, set[str]],
) -> dict:
    """Blocking-quality report: pair recall, entity-complete recall, mean/p95 candidates.

    Stub — filled in with NB05 (Prompt 2).

    Inputs: candidates - {s1_id: set(candidate ids)}; true_map - {s1_id: set(true ids)}.
    Outputs: dict of blocking metrics. Raises NotImplementedError for now.
    """
    raise NotImplementedError("blocking_metrics: implement with blocking.py (Prompt 2)")


BUCKETS = ["false_match_on_singleton", "blocking_miss", "multi_claim_conflict", "chain_branch", "heavy_noise", "other"]


def error_buckets(pred, truth, cands, s1_ids, name_sim=None, number_conflict=None) -> "pd.DataFrame":
    """Bucket every wrong S1 entity (F0.5 < 1) by its most likely failure cause (CLAUDE.md SS4.4).

    Priority (first match wins):
      false_match_on_singleton - truly empty, something predicted;
      blocking_miss            - a true match is missing from the scored candidates;
      multi_claim_conflict     - a wrong/missed record is predicted for another S1;
      chain_branch             - a false match with a similar name but conflicting numbers;
      heavy_noise              - a missed true match (in candidates) with low name similarity;
      other.
    Inputs: pred (s1_id, cand_id), truth (s1_id, match_id), cands (s1_id,
            cand_id [+ name_sim, number_conflict columns]), s1_ids evaluated;
            name_sim / number_conflict - column names in cands (optional).
    Outputs: DataFrame s1_id, bucket, n_fp, n_fn, n_miss_blocking.
    """
    import numpy as np
    import pandas as pd

    ids = pd.Index(pd.unique(np.asarray(list(s1_ids))))
    t = truth[truth["s1_id"].isin(ids)].rename(columns={"match_id": "cand_id"})
    pr = pred[pred["s1_id"].isin(ids)][["s1_id", "cand_id"]]
    key = lambda d: d["s1_id"].astype(str) + "|" + d["cand_id"].astype(str)
    tk, pk, ck = set(key(t)), set(key(pr)), set(key(cands))
    fp = pr[~key(pr).isin(tk)]
    fn = t[~key(t).isin(pk)]
    fn_block = fn[~key(fn).isin(ck)]
    n_true = t.groupby("s1_id").size().reindex(ids, fill_value=0)
    n_fp = fp.groupby("s1_id").size().reindex(ids, fill_value=0)
    n_fn = fn.groupby("s1_id").size().reindex(ids, fill_value=0)
    n_blk = fn_block.groupby("s1_id").size().reindex(ids, fill_value=0)
    wrong = (n_fp > 0) | (n_fn > 0)
    # records predicted for a different S1 than the one being judged
    owner = pr.groupby("cand_id")["s1_id"].agg(lambda s: set(s))
    def claimed_elsewhere(d):
        o = d["cand_id"].map(owner)
        return np.array([isinstance(x, set) and bool(x - {s}) for x, s in zip(o, d["s1_id"])], dtype=bool)
    mc = set(fp.loc[claimed_elsewhere(fp), "s1_id"]) | set(fn.loc[claimed_elsewhere(fn), "s1_id"]) if len(fp) + len(fn) else set()
    cb, hn = set(), set()
    if name_sim and number_conflict and len(fp):
        f = fp.merge(cands[["s1_id", "cand_id", name_sim, number_conflict]], on=["s1_id", "cand_id"], how="left")
        cb = set(f.loc[(f[name_sim] >= 85) & (f[number_conflict] > 0), "s1_id"])
    if name_sim and len(fn):
        f = fn.merge(cands[["s1_id", "cand_id", name_sim]], on=["s1_id", "cand_id"], how="inner")
        hn = set(f.loc[f[name_sim] < 70, "s1_id"])
    rows = []
    for s in ids[wrong.to_numpy()]:
        if n_true[s] == 0:
            b = "false_match_on_singleton"
        elif n_blk[s] > 0:
            b = "blocking_miss"
        elif s in mc:
            b = "multi_claim_conflict"
        elif s in cb:
            b = "chain_branch"
        elif s in hn:
            b = "heavy_noise"
        else:
            b = "other"
        rows.append((s, b, int(n_fp[s]), int(n_fn[s]), int(n_blk[s])))
    return pd.DataFrame(rows, columns=["s1_id", "bucket", "n_fp", "n_fn", "n_miss_blocking"])


def main() -> None:
    """Smoke-test: run the official worked example from the problem statement."""
    score = f05_entity({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    print(f"worked example f05_entity = {score:.3f} (expected 0.714)")
    assert abs(score - 0.714) < 1e-3


if __name__ == "__main__":
    main()
