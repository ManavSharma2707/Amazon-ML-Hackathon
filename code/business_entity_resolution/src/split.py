"""Step 4 — Half A / Half B split of training S1 entities (master plan SS10).

S1 entities are split 55/45 into A and B, stratified by country x
match-count bucket (0, 1, 2-3, 4+) so both halves keep the same singleton
rate and country mix. S2/S3 records follow the half of the S1 they match;
unmatched S2/S3 records are "shared" (only ever negatives, so no leak).

Determinism: IDs are sorted inside each stratum before a seeded shuffle,
so the split doesn't depend on file row order (CLAUDE.md SS3 rule 7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def match_bucket(n: pd.Series) -> pd.Series:
    """Bucket match counts into "0", "1", "2-3", "4+" (plan SS10.2).

    Inputs: n - integer Series of match counts. Outputs: string Series.
    """
    return pd.Series(
        np.select([n == 0, n == 1, n <= 3], ["0", "1", "2-3"], default="4+"), index=n.index
    )


def gt_long(gt: dict[str, set[str]]) -> pd.DataFrame:
    """Convert `{s1: set(matches)}` into a long (s1_id, match_id) pair table.

    Inputs: gt - output of `io_utils.load_ground_truth`.
    Outputs: DataFrame with one row per true pair (singletons have no row).
    """
    s1s, ms = [], []
    for s1, matches in gt.items():
        for m in matches:
            s1s.append(s1)
            ms.append(m)
    return pd.DataFrame({"s1_id": s1s, "match_id": ms})


def make_ab_split(s1: pd.DataFrame, gt: dict[str, set[str]], a_frac: float = 0.55, seed: int = 42) -> pd.DataFrame:
    """Assign every training S1 entity to Half A or Half B, stratified.

    Inputs: s1 - frame with `entity_id` and `country` for all train S1
            records; gt - ground truth dict; a_frac - share of A; seed.
    Outputs: DataFrame (entity_id, country, n_matches, bucket, split) with
    split in {"A", "B"}.
    """
    out = s1[["entity_id", "country"]].copy()
    out["n_matches"] = out["entity_id"].map(lambda i: len(gt.get(i, ()))).astype(np.int32)
    out["bucket"] = match_bucket(out["n_matches"])
    out = out.sort_values(["country", "bucket", "entity_id"], kind="mergesort").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    split = np.empty(len(out), dtype=object)
    for _, idx in out.groupby(["country", "bucket"], sort=True).indices.items():
        perm = rng.permutation(idx)
        n_a = int(round(a_frac * len(idx)))
        split[perm[:n_a]] = "A"
        split[perm[n_a:]] = "B"
    out["split"] = split
    return out


def pool_split(s1_split: pd.DataFrame, pairs: pd.DataFrame, pool_ids: pd.Series) -> pd.DataFrame:
    """Give every S2/S3 record the half of the S1 it matches, else "shared".

    Inputs: s1_split - `make_ab_split` output; pairs - `gt_long` output;
            pool_ids - all train S2/S3 entity IDs.
    Outputs: DataFrame (entity_id, split) for the pool records.
    Raises: AssertionError if one record would belong to both halves (would
    contradict the one-owner finding E4).
    """
    owner = pairs.merge(s1_split[["entity_id", "split"]], left_on="s1_id", right_on="entity_id")
    owner = owner.groupby("match_id")["split"].agg(lambda s: "".join(sorted(set(s))))
    assert owner.isin(["A", "B"]).all(), "a pool record matches S1s in both halves"
    out = pd.DataFrame({"entity_id": pool_ids.to_numpy()})
    out["split"] = out["entity_id"].map(owner).fillna("shared")
    return out


def split_report(s1_split: pd.DataFrame) -> dict:
    """Summarise a split: sizes and singleton rates per half and per half x country.

    Inputs: s1_split - `make_ab_split` output.
    Outputs: JSON-serialisable dict.
    """
    rep: dict = {}
    for half, g in s1_split.groupby("split"):
        rep[half] = {
            "n_s1": int(len(g)),
            "singleton_rate": round(float((g["n_matches"] == 0).mean()), 5),
            "mean_matches": round(float(g["n_matches"].mean()), 4),
            "per_country": {
                c: {"n_s1": int(len(gc)), "singleton_rate": round(float((gc["n_matches"] == 0).mean()), 5)}
                for c, gc in g.groupby("country")
            },
        }
    return rep


def main() -> None:
    """Smoke test on a tiny synthetic ground truth."""
    s1 = pd.DataFrame({"entity_id": [f"S1-{i}" for i in range(20)], "country": ["X"] * 10 + ["Y"] * 10})
    gt = {f"S1-{i}": ({f"S2-{i}"} if i % 3 else set()) for i in range(20)}
    sp = make_ab_split(s1, gt)
    print(split_report(sp))


if __name__ == "__main__":
    main()
