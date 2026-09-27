"""NB06v7-delta (A/B only) driver -- fast gate check for V7's retrained stage-1.

Purpose: answer "does the retrained stage-1 model (translit-extended
candidates + 2 new meta features) actually beat sub-01 on B/LOCO?" as cheaply
as possible, BEFORE paying for the hours-long full test-set feature pass.
G1v7 (blocking recall) already came back essentially flat vs v6 (memory.md,
2026-09-27 19:45 IST), so this is the real remaining question; if it's also
flat or negative here, there is no reason to run the expensive test pass at
all. No test-set computation happens in this notebook by design.

Feature computation uses `features_v7.build_features_delta`: META_FEATURES are
always computed fresh from the current (v7) candidate table (cheap,
vectorised, and blocking-config-dependent so it must be fresh); the expensive
RF+PY block is reused verbatim from NB06 v1's saved features (`feats_A.parquet`
/ `feats_B.parquet`, the shipped sub-01 feature set) for any (s1_id, cand_id)
pair already scored there, and computed fresh only for genuinely new pairs.
Correctness verified locally first (test_features_v7.py: delta output is
byte-identical to a full compute).

Bundled with kaggle_env/io_utils/normalize/corpus_stats/blocking/explain_diff/
features_v7/stage1/decoder/exclusivity into `nb06v7_delta_ab.py`
(CLAUDE.md SS6.3). CPU kernel, internet OFF.

Inputs (globbed under /kaggle/input):
- NB02 output: records_train_S{1,2,3}.parquet, gt.parquet, stats_train.parquet
- NB05v7-partial dataset (or a successful NB05v7 output): cands_A.parquet, cands_B.parquet
- NB06 v1 output (shipped, er-nb06-features-stage1): feats_A.parquet, feats_B.parquet

Outputs in /kaggle/working:
- feats_A.parquet, feats_B.parquet   v7 feature tables (delta-computed)
- stage1_model.txt, p1_A_oof.parquet, p1_B.parquet
- metrics.json    delta reuse rate, stage-1 AUC, B F0.5 and LOCO-lite using
  sub-01's exact shipped decoder config (lam=1.3, hard one-owner), so the
  comparison against sub-01 (B F0.5 0.9648, LOCO-mean 0.9635) is apples to
  apples -- no new grid search / re-tuning on B here.
"""

import gc
import os
import time

import numpy as np
import pandas as pd

WORK = kaggle_env.WORK_DIR
N_JOBS = os.cpu_count() or 1
# Forced single-process for the residual RF+PY feature computation (see
# build_features_delta below): pair_features uses a fork-based
# multiprocessing.Pool (blocking._fork_pool). This driver reads two large
# parquet files via pyarrow (v7 cands + the prior-run feats) right before
# that call; nb05v7_translit hit two other fork-related failures on this same
# Kaggle image today (an inherited LightGBM Booster segfaulting post-fork,
# and OOM from cross-country memory accumulation), and this run hung with
# zero progress logged for over an hour at exactly the point pair_features's
# pool would spin up -- consistent with pyarrow's internal thread pool still
# holding a lock at fork time, which a forked worker can then block on
# forever. Trading parallelism for reliability here on the same reasoning.
FEATURE_N_JOBS = 1
SC = CONFIG["stage1"]
SEED = CONFIG["seed"]
REC_COLS = features_v7.REC_COLS
CAND_COLS = ["s1_id", "cand_id", "bitmask", "n_channels", "cheap_score"] + [f"{c}_score" for c in features_v7.SPARSE_CHANNELS]
TAIL_COLS = features_v7.RF_FEATURES + features_v7.PY_FEATURES
# sub-01's shipped decoder config (progress.md Submissions log) -- reused as-is,
# only q (the estimated blocking-miss rate) is recomputed since it is a property
# of the (new) candidate set, not a tuned hyperparameter.
LAM, MODE = 1.3, "hard"


def load_records(recs_dir):
    """Train S1 and pool (S2 then S3) record frames with the feature columns."""
    s1 = io_utils.read_parquet_compact(recs_dir / "records_train_S1.parquet", REC_COLS)
    pool = pd.concat(
        [io_utils.read_parquet_compact(recs_dir / f"records_train_S{s}.parquet", REC_COLS) for s in (2, 3)],
        ignore_index=True,
    )
    return s1, pool


def read_cands(path) -> pd.DataFrame:
    """v7 candidate file; missing channel-score columns (older schema) filled with NaN."""
    import pyarrow.parquet as pq

    names = set(pq.ParquetFile(path).schema_arrow.names)
    cols = [c for c in CAND_COLS if c in names]
    c = io_utils.read_parquet_compact(path, cols)
    for col in CAND_COLS:
        if col not in c:
            c[col] = np.float32(np.nan)
    return c


def read_prior_tail(path) -> pd.DataFrame:
    """NB06 v1's saved feats file, reduced to s1_id, cand_id + the reusable RF+PY tail."""
    return io_utils.read_parquet_compact(path, ["s1_id", "cand_id"] + TAIL_COLS)


def add_labels(c: pd.DataFrame, gt: pd.DataFrame) -> np.ndarray:
    """0/1 label per candidate row from the ground-truth pairs (s1_id, match_id)."""
    key = c["s1_id"].astype(str) + "|" + c["cand_id"].astype(str)
    true = set((gt["s1_id"].astype(str) + "|" + gt["match_id"].astype(str)).tolist())
    return np.asarray(key.isin(true), dtype=bool).astype(np.int8)


def decode_b(p1b: pd.DataFrame, truth: pd.DataFrame, ids: list, q: float) -> pd.DataFrame:
    """Isotonic-calibrate -> sharpen (LAM) -> hard one-owner -> expected-F0.5 decode."""
    cal = decoder.Isotonic().fit(p1b["p1"].to_numpy(), p1b["label"].to_numpy())
    p = decoder.sharpen(cal.predict(p1b["p1"].to_numpy(np.float64)), LAM)
    d = p1b[["s1_id", "cand_id"]].assign(p=p)
    d["p"] = exclusivity.apply(d, MODE, "p")
    return decoder.decode(d, "p", q=q, max_n=CONFIG["decoder"]["max_n"], min_p=CONFIG["decoder"].get("min_p", 0.0))


def main() -> None:
    """Delta-compute v7 features for A/B only, retrain stage-1, evaluate against sub-01's config."""
    t0 = time.time()
    io_utils.set_seeds(SEED)
    WORK.mkdir(parents=True, exist_ok=True)
    kaggle_env.ensure_rapidfuzz()
    recs_dir = kaggle_env.find_input("records_train_S1.parquet").parent
    cand_dir = kaggle_env.find_input("cands_A.parquet").parent
    prior_dir = kaggle_env.find_input("feats_A.parquet").parent
    kaggle_env.log(f"records {recs_dir}; v7 cands {cand_dir}; prior (v4) feats {prior_dir}; workers {N_JOBS}")
    metrics: dict = {"features": {"n": len(features_v7.FEATURES)}, "decoder_config": {"lam": LAM, "mode": MODE}}

    s1, pool = load_records(recs_dir)
    gt = pd.read_parquet(recs_dir / "gt.parquet")
    lookups = features_v7.token_lookups(pd.read_parquet(recs_dir / "stats_train.parquet"))
    cty = pd.Series(s1["country"].to_numpy(), index=s1["entity_id"].to_numpy())
    kaggle_env.log(f"records S1 {len(s1):,} pool {len(pool):,}")

    feats, labels, cands_out = {}, {}, {}
    for h in ("A", "B"):
        t_h = time.time()
        c = read_cands(cand_dir / f"cands_{h}.parquet")
        prior_tail = read_prior_tail(prior_dir / f"feats_{h}.parquet")
        kaggle_env.log(f"{h}: {len(c):,} v7 candidate pairs, {c['s1_id'].nunique():,} S1; prior feats {len(prior_tail):,} rows")
        f = features_v7.build_features_delta(c, s1, pool, lookups, prior_tail, n_jobs=FEATURE_N_JOBS, log=kaggle_env.log)
        y = add_labels(c, gt)
        feats[h], labels[h], cands_out[h] = f, y, c
        out = f.copy()
        out.insert(0, "label", y)
        out.insert(0, "cand_id", c["cand_id"].to_numpy())
        out.insert(0, "s1_id", c["s1_id"].to_numpy())
        out.to_parquet(WORK / f"feats_{h}.parquet", index=False, compression="zstd")
        metrics[f"{h}_pos_rate"] = round(float(y.mean()), 5)
        kaggle_env.log(f"  {h} done ({time.time() - t_h:.0f}s)")
        del prior_tail
        gc.collect()
    kaggle_env.write_json(metrics, WORK / "metrics.json")

    # ---------------- stage-1: OOF on A, full-A model -> B ----------------
    params = stage1.lgb_params(SC, SEED)
    XA = feats["A"].to_numpy(np.float32)
    kaggle_env.log(f"stage-1 OOF on A: {XA.shape}")
    r = stage1.train_oof(XA, labels["A"], cands_out["A"]["s1_id"].to_numpy(), params, folds=SC.get("folds", 5), seed=SEED,
                         max_rounds=SC.get("max_rounds", 1500), feature_names=features_v7.FEATURES, log=kaggle_env.log)
    n_rounds = int(np.mean(r["best_iters"]) * 1.1) + 1
    model = stage1.train_full(XA, labels["A"], params, n_rounds, features_v7.FEATURES)
    model.save_model(str(WORK / "stage1_model.txt"))
    cA = cands_out["A"]["s1_id"].map(cty).to_numpy()
    pd.DataFrame({"s1_id": cands_out["A"]["s1_id"], "cand_id": cands_out["A"]["cand_id"], "label": labels["A"],
                 "p1": r["oof"]}).to_parquet(WORK / "p1_A_oof.parquet", index=False)
    pB = stage1.predict(model, feats["B"].to_numpy(np.float32), N_JOBS)
    cB = cands_out["B"]["s1_id"].map(cty).to_numpy()
    p1b = pd.DataFrame({"s1_id": cands_out["B"]["s1_id"], "cand_id": cands_out["B"]["cand_id"], "label": labels["B"], "p1": pB})
    p1b.to_parquet(WORK / "p1_B.parquet", index=False)
    metrics["stage1"] = {"best_iters": r["best_iters"], "fold_auc": r["fold_auc"], "n_rounds_full": n_rounds,
                         "A_oof": stage1.eval_by_group(labels["A"], r["oof"], cA),
                         "B": stage1.eval_by_group(labels["B"], pB, cB)}
    kaggle_env.log(f"stage-1 A OOF {metrics['stage1']['A_oof']['ALL']} | B {metrics['stage1']['B']['ALL']}")
    kaggle_env.write_json(metrics, WORK / "metrics.json")

    # ---------------- decode B with sub-01's exact config; compare ----------------
    ids = sorted(p1b["s1_id"].unique().tolist())
    truth = gt[gt["s1_id"].isin(set(ids))][["s1_id", "match_id"]]
    in_c = truth.merge(p1b[["s1_id", "cand_id"]], left_on=["s1_id", "match_id"], right_on=["s1_id", "cand_id"], how="left")
    miss_s1 = in_c[in_c["cand_id"].isna()]["s1_id"].nunique()
    q_est = round(miss_s1 / len(ids), 4)
    pred_b = decode_b(p1b, truth, ids, q_est)
    b_f05 = decoder.macro_f05(pred_b, truth, ids)
    metrics["B_eval"] = {"n_s1": len(ids), "pair_recall": round(float(in_c["cand_id"].notna().mean()), 5),
                         "q_est": q_est, "f05": round(b_f05, 6),
                         "vs_sub01_f05": round(b_f05 - 0.964763, 6)}
    kaggle_env.log(f"B (sub-01 decoder config): f05={b_f05:.6f} (sub-01 was 0.964763, delta {metrics['B_eval']['vs_sub01_f05']:+.6f})")

    # ---------------- LOCO-lite (same config, cross-country) ----------------
    cb = p1b["s1_id"].map(cty).to_numpy()
    countries = sorted(set(cb.tolist()))
    loco = {}
    for c_tr in countries:
        for c_te in countries:
            if c_tr == c_te:
                continue
            te = p1b[cb == c_te].reset_index(drop=True)
            ids_te = sorted(te["s1_id"].unique().tolist())
            f = decoder.macro_f05(decode_b(te, truth, ids_te, q_est), truth, ids_te)
            loco[f"{c_tr}->{c_te}"] = round(f, 6)
    metrics["loco_lite"] = loco
    metrics["loco_lite_mean"] = round(float(np.mean(list(loco.values()))), 6) if loco else None
    metrics["loco_lite_vs_sub01"] = round(metrics["loco_lite_mean"] - 0.9635, 6) if loco else None
    kaggle_env.log(f"LOCO-lite: {metrics['loco_lite_mean']} (sub-01 was 0.9635, delta {metrics['loco_lite_vs_sub01']:+.6f})")

    metrics["runtime_s"] = round(time.time() - t0, 1)
    kaggle_env.write_json(metrics, WORK / "metrics.json")
    kaggle_env.log(f"done in {metrics['runtime_s']}s")


if __name__ == "__main__":
    main()
