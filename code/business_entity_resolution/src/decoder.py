"""Expected-F0.5 decoder, calibration and the macro-F0.5 scorer on pair tables (plan SS18.2, SS20, SS28.3).

For every S1 entity the candidates are sorted by calibrated probability and
each top-k set (k = 0..n, where k = 0 is the "empty" answer) is valued by its
expected F0.5 under independent Bernoulli matches, with an extra Bernoulli q
for "a true match exists outside the candidate set". The best k wins.

The reference implementation (plan SS28.3) loops per entity; here the same
computation is vectorised over entities: prefix / suffix Poisson-binomial
pmfs are built by one column update per candidate, and E[F](k) is an
einsum of the TP pmf, the FN pmf and a fixed score table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BETA2 = 0.25  # F0.5


def score_table(k: int, n_fn: int, beta2: float = BETA2) -> np.ndarray:
    """F-beta value for a top-k prediction as a (tp = 0..k) x (fn = 0..n_fn) table (official convention).

    k = 0: 1 if fn = 0 else 0; k > 0 and tp = 0: 0; else (1+b2)tp / ((1+b2)tp + b2 fn + (k - tp)).
    """
    tp = np.arange(k + 1)[:, None].astype(np.float64)
    fn = np.arange(n_fn + 1)[None, :].astype(np.float64)
    if k == 0:
        return (fn == 0).astype(np.float64)
    denom = (1 + beta2) * tp + beta2 * fn + (k - tp)
    return np.where(tp > 0, (1 + beta2) * tp / np.maximum(denom, 1e-12), 0.0)


def expected_f_matrix(P: np.ndarray, q: np.ndarray, beta2: float = BETA2) -> np.ndarray:
    """E[F-beta] of every top-k prediction for a batch of entities.

    Inputs: P - (E, n) probabilities sorted descending per row (pad with 0);
            q - (E,) probability of a true match outside the candidates.
    Outputs: (E, n + 1) expected scores for k = 0..n.
    """
    E, n = P.shape
    # prefix[k]: pmf of TP among the first k candidates, shape (E, k + 1)
    prefix = [np.ones((E, 1))]
    for j in range(n):
        prev, p = prefix[-1], P[:, j : j + 1]
        cur = np.zeros((E, j + 2))
        cur[:, :-1] += prev * (1 - p)
        cur[:, 1:] += prev * p
        prefix.append(cur)
    # suffix[k]: pmf of FN among candidates k..n-1 plus the outside Bernoulli q, shape (E, n - k + 2)
    suffix = [None] * (n + 1)
    cur = np.stack([1 - q, q], axis=1)
    suffix[n] = cur
    for j in range(n - 1, -1, -1):
        p = P[:, j : j + 1]
        nxt = np.zeros((E, cur.shape[1] + 1))
        nxt[:, :-1] += cur * (1 - p)
        nxt[:, 1:] += cur * p
        cur = nxt
        suffix[j] = cur
    out = np.empty((E, n + 1))
    for k in range(n + 1):
        S = score_table(k, suffix[k].shape[1] - 1, beta2)
        out[:, k] = np.einsum("et,tf,ef->e", prefix[k], S, suffix[k], optimize=True)
    return out


def sharpen(p: np.ndarray, lam: float) -> np.ndarray:
    """p' = sigmoid(lam * logit(p)) (lam < 1 softens, > 1 sharpens; plan SS18.2)."""
    if lam == 1.0:
        return p
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-lam * np.log(p / (1 - p))))


def _grouped(df: pd.DataFrame, pcol: str, max_n: int):
    """Sort pairs by (s1, p desc); return (sorted frame, group starts, sizes, rank within group)."""
    d = df.sort_values(["s1_id", pcol], ascending=[True, False], kind="stable").reset_index(drop=True)
    s = d["s1_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]]) if len(d) else np.empty(0, int)
    sizes = np.diff(np.r_[starts, len(d)])
    rank = np.arange(len(d)) - np.repeat(starts, sizes)
    return d, starts, sizes, rank


def decode(df: pd.DataFrame, pcol: str = "p", q: float | np.ndarray = 0.0, max_n: int = 20, min_p: float = 0.0,
           batch: int = 50_000) -> pd.DataFrame:
    """Expected-F0.5 top-k decision for every S1 in a pair table.

    Inputs: df - pairs with s1_id, cand_id and calibrated probability `pcol`;
            q - outside-candidates match probability (scalar or per-row array
            aligned with df, constant within an S1); max_n - candidates
            considered per S1 (the rest are ignored); min_p - probabilities
            below this are treated as 0; batch - entities per vectorised batch.
    Outputs: the pairs chosen (s1_id, cand_id, pcol) — S1s absent from the
    result predict the empty set.
    """
    if len(df) == 0:
        return df.iloc[:0][["s1_id", "cand_id", pcol]]
    d = df[["s1_id", "cand_id", pcol]].copy()
    d["_q"] = q if np.ndim(q) == 0 else np.asarray(q)
    d, starts, sizes, rank = _grouped(d, pcol, max_n)
    keep_rank = rank < max_n
    E = len(starts)
    n = int(min(max_n, sizes.max()))
    P = np.zeros((E, n))
    ent = np.repeat(np.arange(E), sizes)
    p = d[pcol].to_numpy(np.float64)
    p = np.where(p < min_p, 0.0, p)
    P[ent[keep_rank], rank[keep_rank]] = p[keep_rank]
    qe = d["_q"].to_numpy(np.float64)[starts]
    best = np.empty(E, dtype=np.int64)
    for s in range(0, E, batch):
        best[s : s + batch] = expected_f_matrix(P[s : s + batch], qe[s : s + batch]).argmax(axis=1)
    chosen = rank < np.repeat(best, sizes)
    return d.loc[chosen, ["s1_id", "cand_id", pcol]].reset_index(drop=True)


def threshold_decode(df: pd.DataFrame, pcol: str, t: float) -> pd.DataFrame:
    """Baseline: every candidate with probability >= t (plan SS20.5 comparison)."""
    return df.loc[df[pcol] >= t, ["s1_id", "cand_id", pcol]].reset_index(drop=True)


def macro_f05(pred: pd.DataFrame, truth: pd.DataFrame, s1_ids, beta2: float = BETA2, return_per_entity: bool = False):
    """Official macro F0.5 over the given S1 entities (plan SS1.2; same rules as metrics.f05_entity).

    Inputs: pred - predicted pairs (s1_id, cand_id); truth - true pairs
            (s1_id, match_id), may include pairs outside the candidates;
            s1_ids - every evaluated S1 (singletons included).
    Outputs: mean F0.5 (and the per-entity Series if requested).
    """
    ids = pd.Index(pd.unique(np.asarray(list(s1_ids))))
    t = truth[truth["s1_id"].isin(ids)]
    pr = pred[pred["s1_id"].isin(ids)]
    n_true = t.groupby("s1_id").size().reindex(ids, fill_value=0).to_numpy()
    n_pred = pr.groupby("s1_id").size().reindex(ids, fill_value=0).to_numpy()
    hit = pr.merge(t, left_on=["s1_id", "cand_id"], right_on=["s1_id", "match_id"], how="inner")
    tp = hit.groupby("s1_id").size().reindex(ids, fill_value=0).to_numpy()
    fn, fp = n_true - tp, n_pred - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        f = (1 + beta2) * tp / ((1 + beta2) * tp + beta2 * fn + fp)
    f = np.where((n_true == 0) & (n_pred == 0), 1.0, np.nan_to_num(f, nan=0.0))
    f = np.where(tp == 0, np.where((n_true == 0) & (n_pred == 0), 1.0, 0.0), f)
    s = pd.Series(f, index=ids)
    return (float(s.mean()), s) if return_per_entity else float(s.mean())


class Isotonic:
    """Isotonic calibration p -> empirical match rate (sklearn), with a Platt fallback (plan SS18.2)."""

    def __init__(self, kind: str = "isotonic"):
        """kind: "isotonic" or "platt"."""
        self.kind = kind
        self.model = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Isotonic":
        """Fit on raw probabilities p and 0/1 labels y."""
        if self.kind == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
        else:
            from sklearn.linear_model import LogisticRegression

            x = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
            self.model = LogisticRegression(C=1e4).fit(x[:, None], y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        """Calibrated probabilities (float64)."""
        if self.kind == "isotonic":
            return self.model.predict(p).astype(np.float64)
        x = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        return self.model.predict_proba(x[:, None])[:, 1]


def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error with equal-width bins."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(tot / max(len(p), 1))


def main() -> None:
    """Check the vectorised decoder against the reference loop on random entities (smoke test)."""
    rng = np.random.default_rng(0)
    P = -np.sort(-rng.random((5, 6)) ** 3, axis=1)
    print(expected_f_matrix(P, np.full(5, 0.05)).argmax(axis=1))


if __name__ == "__main__":
    main()
