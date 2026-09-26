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
    ids = sorted(set(scores_a) & set(scores_b))
    assert ids, "scores_a and scores_b share no S1 entities"
    diffs = [scores_b[i] - scores_a[i] for i in ids]
    mean_diff = sum(diffs) / len(diffs)

    rng = random.Random(seed)
    n = len(diffs)
    resample_means = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        resample_means.append(sum(sample) / n)
    resample_means.sort()
    lo = resample_means[int(0.05 * n_resamples)]
    hi = resample_means[int(0.95 * n_resamples) - 1]
    return {
        "mean_diff": mean_diff,
        "ci90": (lo, hi),
        "excludes_zero": lo > 0 or hi < 0,
    }


def loco_eval(
    train_fn: Callable,
    eval_fn: Callable,
    entities_by_country: dict[str, list[str]],
) -> dict:
    """Leave-one-country-out evaluation: train on country X, score on country Y, both ways.

    Stub — filled in once the combiner/decoder exist (Prompt 4+). `train_fn`
    and `eval_fn` are expected to take a list of S1 entity IDs and return a
    fitted artifact / a {s1_id: score} map respectively.

    Inputs: train_fn, eval_fn - pipeline callables; entities_by_country -
            {country: [s1_id, ...]} for exactly two countries (Half B).
    Outputs: dict with per-direction and mean LOCO F0.5. Raises NotImplementedError for now.
    """
    raise NotImplementedError("loco_eval: implement once the combiner exists (Prompt 4)")


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


def error_buckets(
    pred_map: dict[str, set],
    true_map: dict[str, set],
    records: dict | None = None,
) -> dict:
    """Bucket wrong entities by failure cause for the error-analysis loop (CLAUDE.md SS4.4).

    Stub — buckets (false match on singleton / chain branch / multi-claim
    conflict / blocking miss / heavy noise / other) need feature/record access
    not yet available before Prompt 3.

    Inputs: pred_map, true_map - as above; records - optional record lookup for examples.
    Outputs: dict of bucket -> list of s1_ids. Raises NotImplementedError for now.
    """
    raise NotImplementedError("error_buckets: implement once features/records are available")


def main() -> None:
    """Smoke-test: run the official worked example from the problem statement."""
    score = f05_entity({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    print(f"worked example f05_entity = {score:.3f} (expected 0.714)")
    assert abs(score - 0.714) < 1e-3


if __name__ == "__main__":
    main()
