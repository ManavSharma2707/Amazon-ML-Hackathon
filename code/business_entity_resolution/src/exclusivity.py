"""One-owner rule: each S2/S3 record belongs to at most one S1 (plan SS19).

EDA E4 found 0 of 7,638,365 matched IDs shared between two S1s, so the
default mode is `hard` (memory.md SS5); `soft` and `none` stay available and
are compared on Half B.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hard_one_owner(df: pd.DataFrame, pcol: str = "p", p_min: float = 0.05) -> np.ndarray:
    """Greedy global assignment (plan SS19.1).

    Among pairs with p >= p_min, sorted by p descending, a record goes to the
    first S1 that claims it; the other claimants' probabilities become 0.
    Pairs below p_min are left unchanged.

    Inputs: df - pairs (s1_id, cand_id, pcol); pcol; p_min.
    Outputs: adjusted probability array aligned with df's rows.
    """
    p = df[pcol].to_numpy(np.float64).copy()
    idx = np.flatnonzero(p >= p_min)
    if len(idx) == 0:
        return p
    sub = pd.DataFrame({"cand_id": df["cand_id"].to_numpy()[idx], "p": p[idx], "i": idx})
    sub = sub.sort_values(["p", "i"], ascending=[False, True], kind="stable")
    lost = sub["i"].to_numpy()[sub["cand_id"].duplicated(keep="first").to_numpy()]
    p[lost] = 0.0
    return p


def soft_one_owner(df: pd.DataFrame, pcol: str = "p", gamma: float = 1.0) -> np.ndarray:
    """p'(e, r) = p(e, r) * (p(e, r) / sum_e' p(e', r)) ** gamma (plan SS19.2)."""
    p = df[pcol].to_numpy(np.float64)
    tot = df.assign(_p=p).groupby("cand_id")["_p"].transform("sum").to_numpy()
    share = np.divide(p, tot, out=np.zeros_like(p), where=tot > 0)
    return p * share ** gamma


def apply(df: pd.DataFrame, mode: str, pcol: str = "p", p_min: float = 0.05, gamma: float = 1.0) -> np.ndarray:
    """Dispatch on mode: "none" | "soft" | "hard". Outputs: adjusted probabilities aligned with df."""
    if mode == "none":
        return df[pcol].to_numpy(np.float64).copy()
    if mode == "soft":
        return soft_one_owner(df, pcol, gamma)
    if mode == "hard":
        return hard_one_owner(df, pcol, p_min)
    raise ValueError(f"unknown exclusivity mode {mode!r}")


def main() -> None:
    """Tiny example (smoke test)."""
    df = pd.DataFrame({"s1_id": ["a", "b", "b"], "cand_id": ["x", "x", "y"], "p": [0.9, 0.6, 0.7]})
    print(hard_one_owner(df), soft_one_owner(df))


if __name__ == "__main__":
    main()
