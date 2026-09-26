"""Stage-1 LightGBM pair classifier (master plan SS15).

Trained on Half A candidate pairs only: 5-fold group-k-fold (groups = S1 id)
gives out-of-fold probabilities for A (used to pick the judge's training
band), and one model on all of A scores Half B and test. No class
re-weighting (probabilities are calibrated later on B).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def lgb_params(cfg: dict, seed: int) -> dict:
    """LightGBM parameters from the `stage1` config block (plan SS15.1).

    Inputs: cfg - config["stage1"]; seed. Outputs: params dict.
    """
    return {
        "objective": "binary", "learning_rate": cfg.get("lr", 0.05), "num_leaves": cfg.get("num_leaves", 63),
        "min_data_in_leaf": cfg.get("min_data_in_leaf", 50), "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": cfg.get("max_bin", 127), "seed": seed, "bagging_seed": seed,
        "feature_fraction_seed": seed, "deterministic": True, "verbose": -1, "num_threads": cfg.get("num_threads", 0),
    }


def group_folds(groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Fold id per row such that all rows of one group share a fold (seeded, balanced by group count).

    Inputs: groups - group key per row (S1 id); k; seed. Outputs: int array of fold ids.
    """
    uniq, inv = np.unique(groups, return_inverse=True)
    perm = np.random.default_rng(seed).permutation(len(uniq))
    fold_of_group = np.empty(len(uniq), dtype=np.int32)
    fold_of_group[perm] = np.arange(len(uniq)) % k
    return fold_of_group[inv]


def train_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray, params: dict, folds: int = 5, seed: int = 42,
              max_rounds: int = 2000, early_stop: int = 50, feature_names: list[str] | None = None, log=print) -> dict:
    """Group-k-fold OOF training with early stopping on each held-out fold.

    Asserts that no group appears in both the training and validation part.
    Inputs: X, y, groups (S1 id per row); params; folds; seed; max_rounds;
            early_stop; feature_names; log.
    Outputs: {"oof": probabilities, "best_iters": [...], "fold_auc": [...],
              "models": [boosters]}.
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    fid = group_folds(groups, folds, seed)
    oof = np.zeros(len(y), dtype=np.float32)
    best, aucs, models = [], [], []
    for f in range(folds):
        va = fid == f
        tr = ~va
        assert not set(np.unique(groups[va]).tolist()) & set(np.unique(groups[tr]).tolist()), "group leak between folds"
        dtr = lgb.Dataset(X[tr], label=y[tr], feature_name=feature_names or "auto", free_raw_data=True)
        dva = lgb.Dataset(X[va], label=y[va], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=max_rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(early_stop, verbose=False)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        best.append(int(m.best_iteration))
        aucs.append(float(roc_auc_score(y[va], oof[va])) if 0 < y[va].sum() < va.sum() else float("nan"))
        models.append(m)
        log(f"    fold {f}: best_iter {best[-1]}, AUC {aucs[-1]:.5f}")
    return {"oof": oof, "best_iters": best, "fold_auc": aucs, "models": models}


def train_full(X: np.ndarray, y: np.ndarray, params: dict, num_rounds: int, feature_names: list[str] | None = None):
    """One model on all rows with a fixed number of rounds (from the OOF best iterations).

    Outputs: lightgbm.Booster.
    """
    import lightgbm as lgb

    return lgb.train(params, lgb.Dataset(X, label=y, feature_name=feature_names or "auto"), num_boost_round=num_rounds)


def predict(model, X: np.ndarray, num_threads: int = 0) -> np.ndarray:
    """Probabilities from a trained booster (float32)."""
    return model.predict(X, num_threads=num_threads).astype(np.float32)


def eval_by_group(y: np.ndarray, p: np.ndarray, country: np.ndarray) -> dict:
    """AUC and log-loss overall and per country label (reporting only, never a feature).

    Inputs: labels, probabilities, country label per row.
    Outputs: {"ALL": {...}, "<country>": {...}}.
    """
    from sklearn.metrics import log_loss, roc_auc_score

    out = {}
    for key, m in [("ALL", np.ones(len(y), bool))] + [(c, country == c) for c in sorted(set(country.tolist()))]:
        yy, pp = y[m], np.clip(p[m], 1e-7, 1 - 1e-7)
        ok = 0 < yy.sum() < len(yy)
        out[key] = {"n": int(m.sum()), "pos_rate": round(float(yy.mean()), 5) if len(yy) else None,
                    "auc": round(float(roc_auc_score(yy, pp)), 6) if ok else None,
                    "logloss": round(float(log_loss(yy, pp, labels=[0, 1])), 6) if len(yy) else None}
    return out


def main() -> None:
    """Tiny synthetic run (smoke test)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 5)).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.normal(size=4000) > 0).astype(np.int8)
    g = np.repeat(np.arange(400), 10)
    r = train_oof(X, y, g, lgb_params({"lr": 0.1}, 0), folds=3, max_rounds=100)
    print(pd.Series(r["fold_auc"]).round(4).tolist())


if __name__ == "__main__":
    main()
