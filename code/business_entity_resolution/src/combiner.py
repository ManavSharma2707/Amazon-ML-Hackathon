"""Combiner (master plan SS18.1) and the shared post-processing chain (SS18.2, SS19, SS20).

Combiner: LightGBM on Half-B candidate pairs with every stage-1 feature
(groups A-H), the stage-1 probability, the collective features (collective.py)
and, when available, the judge score. 5-fold group-k-fold by S1 gives OOF
probabilities on all of B; one model on all of B scores test.

Post-processing chain (used for B evaluation and for test):
raw probability -> isotonic calibration -> sharpening lambda -> one-owner
rule (none | soft | hard) -> expected-F0.5 decoder (or a global threshold).
Its parameters are chosen on B only, by a coarse grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import collective, decoder, exclusivity, stage1

JUDGE_FEATURES = ["judge_p", "judge_scored", "judge_minus_p1"]


def extra_features(p1: np.ndarray) -> np.ndarray:
    """p1 and its logit as combiner inputs."""
    p = np.clip(p1.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.column_stack([p1, np.log(p / (1 - p))]).astype(np.float32)


def matrix(feats: pd.DataFrame, p1: np.ndarray, coll: np.ndarray, judge: pd.DataFrame | None = None,
           base_names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Combiner design matrix: stage-1 features + p1/logit + collective (+ judge).

    Inputs: feats - stage-1 feature frame; p1; coll - collective features;
            judge - optional frame with judge_p (NaN = not scored).
    Outputs: (float32 matrix, feature names).
    """
    base_names = base_names or list(feats.columns)
    parts = [feats[base_names].to_numpy(np.float32), extra_features(p1), coll.astype(np.float32)]
    names = base_names + ["p1", "logit_p1"] + collective.COLLECTIVE_FEATURES
    if judge is not None:
        jp = judge["judge_p"].to_numpy(np.float32)
        parts.append(np.column_stack([jp, np.isfinite(jp).astype(np.float32), jp - p1.astype(np.float32)]))
        names += JUDGE_FEATURES
    return np.hstack(parts), names


def params(cfg: dict, seed: int) -> dict:
    """LightGBM parameters: stage-1 settings with num_leaves 31 (plan SS18.1)."""
    p = stage1.lgb_params({"lr": cfg.get("lr", 0.08), "num_leaves": cfg.get("num_leaves", 31),
                           "min_data_in_leaf": cfg.get("min_data_in_leaf", 100), "max_bin": cfg.get("max_bin", 63)}, seed)
    p["feature_fraction"] = cfg.get("feature_fraction", 0.7)
    return p


# ---------------------------------------------------------------------------
# Post-processing chain
# ---------------------------------------------------------------------------


class Fixed:
    """Calibrator stand-in returning precomputed calibrated values (cross-fitted B evaluation)."""

    def __init__(self, values: np.ndarray):
        """values: calibrated probabilities aligned with the frame passed to `postprocess`."""
        self.values = values

    def predict(self, p):
        """Ignore p; return the stored values."""
        return self.values


def crossfit_calibrated(p: np.ndarray, y: np.ndarray, s1: np.ndarray, folds: int = 2, kind: str = "isotonic") -> np.ndarray:
    """Calibrated probabilities where each S1 group is calibrated by a model fitted on the other fold(s)."""
    f = pd.util.hash_array(np.asarray(s1, dtype=object)) % folds
    out = np.empty(len(p))
    for k in range(folds):
        tr = f != k
        out[~tr] = decoder.Isotonic(kind).fit(p[tr], y[tr]).predict(p[~tr])
    return out


def postprocess(df: pd.DataFrame, cal, lam: float, mode: str, method: str, q: float = 0.0, t: float = 0.5,
                pcol: str = "p", max_n: int = 20, min_p: float = 0.0, gamma: float = 1.0) -> pd.DataFrame:
    """calibrate -> sharpen -> exclusivity -> decode. Inputs: df with s1_id, cand_id, `pcol`. Outputs: chosen pairs."""
    pc = decoder.sharpen(cal.predict(df[pcol].to_numpy(np.float64)), lam)
    d = df[["s1_id", "cand_id"]].assign(p=pc)
    d["p"] = exclusivity.apply(d, mode, "p", gamma=gamma)
    if method == "threshold":
        return decoder.threshold_decode(d, "p", t)
    return decoder.decode(d, "p", q=q, max_n=max_n, min_p=min_p)


def run_config(df, cal, cfg: dict, pcol: str = "p", max_n: int = 20, min_p: float = 0.0) -> pd.DataFrame:
    """postprocess with a config dict from `grid_search`."""
    return postprocess(df, cal, cfg["lam"], cfg["mode"], cfg["method"], cfg.get("q", 0.0), cfg.get("t") or 0.5, pcol,
                       max_n, min_p)


def grid_search(df, cal, truth, ids, qs, lambdas, modes, thresholds, pcol: str = "p", max_n: int = 20,
                min_p: float = 0.0, log=None) -> tuple[dict, list]:
    """Coarse grid over decoder (mode x lambda x q) and threshold (mode x t) configs; best by macro F0.5 on `ids`."""
    rows = []
    for mode in modes:
        for lam in lambdas:
            for q in qs:
                cfg = {"method": "decoder", "mode": mode, "lam": lam, "q": q, "t": None}
                cfg["f05"] = round(decoder.macro_f05(run_config(df, cal, cfg, pcol, max_n, min_p), truth, ids), 6)
                rows.append(cfg)
                if log:
                    log(f"    {cfg}")
        for t in thresholds:
            cfg = {"method": "threshold", "mode": mode, "lam": 1.0, "q": 0.0, "t": float(t)}
            cfg["f05"] = round(decoder.macro_f05(run_config(df, cal, cfg, pcol, max_n, min_p), truth, ids), 6)
            rows.append(cfg)
    return max(rows, key=lambda r: r["f05"]), rows


def main() -> None:
    """Print the combiner's extra feature names (smoke test)."""
    print(["p1", "logit_p1"] + collective.COLLECTIVE_FEATURES + JUDGE_FEATURES)


if __name__ == "__main__":
    main()
