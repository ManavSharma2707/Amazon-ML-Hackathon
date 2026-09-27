"""NB09a driver — calibration + one-owner rule + expected-F0.5 decoder -> submission files.

Safety submission path (architecture.md SS6.11 without the combiner/judge):
stage-1 probabilities from NB06 -> isotonic calibration fitted on Half B ->
sharpening lambda -> exclusivity (none | hard) -> decoder (expected F0.5, or
the best global threshold if that scores higher on B).

Bundled with kaggle_env/io_utils/normalize/corpus_stats/blocking/explain_diff/
features/stage1/decoder/exclusivity/scramble/check_outputs into
`nb09a_decode_submit.py`. CPU kernel, internet OFF.

Inputs (globbed under /kaggle/input):
- NB02 output: records_*.parquet, stats_train.parquet, gt.parquet
- NB06 output: p1_B.parquet, p1_test.parquet, feats_B.parquet, stage1_model.txt
- er-data: test TSVs + utils/validate_submission.py
- NB00 output: wheels/ (rapidfuzz, for the scrambled-letter features)

Outputs in /kaggle/working:
- output/matching_results.tsv, output/candidate_pairs.tsv (candidates = exactly the scored test set)
- artifacts/calibrator.npz, final_config.json, stage1_model.txt, prerank_half{0,1}.txt (the trained/fitted
  pieces needed to reproduce this exact submission from saved features; see src/predict.py)
- metrics.json (B F0.5 overall / per country vs floor, grid, LOCO-lite, scrambled drop, validator results)
"""

import gc
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

WORK = kaggle_env.WORK_DIR
OUT = WORK / "output"
N_JOBS = os.cpu_count() or 1
SEED = CONFIG["seed"]
DC = CONFIG["decoder"]
LAMBDAS = DC["lambda_grid"]
MAX_N = DC["max_n"]
MODES = ["none", "hard"]
THRESHOLDS = np.round(np.arange(0.10, 0.951, 0.05), 3)
SCR_S1 = int(os.environ.get("ER_SCR_S1", 30_000))  # B S1s in the scrambled-letter run
SCR_FIELDS = ["norm_name", "norm_addr", "landmark", "postcodes", "house_number", "addr_numbers", "name_numbers"]


# ---------------------------------------------------------------------------
# Post-processing chain
# ---------------------------------------------------------------------------


def chain(df: pd.DataFrame, cal, lam: float, mode: str, method: str, q: float, t: float = 0.5) -> pd.DataFrame:
    """calibrate -> sharpen -> exclusivity -> decode. df: s1_id, cand_id, p1. Outputs: chosen pairs."""
    p = decoder.sharpen(cal.predict(df["p1"].to_numpy(np.float64)), lam)
    d = df[["s1_id", "cand_id"]].assign(p=p)
    d["p"] = exclusivity.apply(d, mode, "p")
    if method == "threshold":
        return decoder.threshold_decode(d, "p", t)
    return decoder.decode(d, "p", q=q, max_n=MAX_N, min_p=DC.get("min_p", 0.0))


def crossfit_calibrated(df: pd.DataFrame, folds: int = 2):
    """Isotonic p1 -> label, fitted on the other fold of S1s (honest B calibration). Returns the calibrated p."""
    f = pd.util.hash_array(df["s1_id"].to_numpy(dtype=object)) % folds
    out = np.empty(len(df))
    for k in range(folds):
        tr = f != k
        cal = decoder.Isotonic().fit(df["p1"].to_numpy()[tr], df["label"].to_numpy()[tr])
        out[~tr] = cal.predict(df["p1"].to_numpy()[~tr])
    return out


class Fixed:
    """Calibrator stand-in that returns precomputed calibrated values (for cross-fitted B evaluation)."""

    def __init__(self, values: np.ndarray):
        """values: calibrated probabilities aligned with the frame passed to `chain`."""
        self.values = values

    def predict(self, p):
        """Ignore p; return the stored values."""
        return self.values


def grid_search(df, cal, truth, ids, qs) -> tuple[dict, list]:
    """Evaluate decoder (lambda x mode x q) and threshold (t x mode) configs on B; return (best, all rows).

    Coarse grids only (plan SS20.4) to avoid overfitting B.
    """
    rows = []
    for mode in MODES:
        for lam in LAMBDAS:
            for q in qs:
                f = decoder.macro_f05(chain(df, cal, lam, mode, "decoder", q), truth, ids)
                rows.append({"method": "decoder", "mode": mode, "lam": lam, "q": q, "t": None, "f05": round(f, 6)})
                kaggle_env.log(f"    decoder mode={mode} lam={lam} q={q}: {f:.5f}")
        for t in THRESHOLDS:
            f = decoder.macro_f05(chain(df, cal, 1.0, mode, "threshold", 0.0, t), truth, ids)
            rows.append({"method": "threshold", "mode": mode, "lam": 1.0, "q": 0.0, "t": float(t), "f05": round(f, 6)})
    best = max(rows, key=lambda r: r["f05"])
    return best, rows


def per_country(pred, truth, ids, cty) -> dict:
    """Macro F0.5 per country label (reporting only) + all-empty floor per country."""
    out = {}
    empty = pred.iloc[:0]
    for c in sorted(set(cty.reindex(ids).tolist())):
        sub = [i for i in ids if cty.get(i) == c]
        out[c] = {"n_s1": len(sub), "f05": round(decoder.macro_f05(pred, truth, sub), 6),
                  "floor": round(decoder.macro_f05(empty, truth, sub), 6)}
    return out


# ---------------------------------------------------------------------------
# Scrambled-letter test (plan SS21.4)
# ---------------------------------------------------------------------------


def scramble_records(df: pd.DataFrame, table: dict) -> pd.DataFrame:
    """Apply the a-z permutation to the normalised text fields; recompute the folds on the scrambled text.

    Scrambling after normalisation keeps romanised native-script names consistent
    with their Latin counterparts (both sides are scrambled identically).
    """
    out = df.copy()
    for c in SCR_FIELDS:
        out[c] = [str(x).translate(table) for x in df[c].tolist()]
    out["fold_name"] = [normalize.fold(x) for x in out["norm_name"].tolist()]
    out["fold_addr"] = [normalize.fold(x) for x in out["norm_addr"].tolist()]
    return out


def scramble_lookups(stats: pd.DataFrame, table: dict) -> dict:
    """Token statistics with every token permuted the same way (IDF / generic sets follow the scramble)."""
    s = stats[stats["field"].isin(["norm_name", "norm_addr"])].copy()
    s["token"] = [str(t).translate(table) for t in s["token"].tolist()]
    return features.token_lookups(s)


def scrambled_run(recs_dir, p1b, cal, best, truth, ids_all, model, rng) -> dict:
    """F0.5 on a B subset with and without the letter scramble (same model, calibrator and decoder)."""
    ids = sorted(rng.choice(np.array(ids_all, dtype=object), min(SCR_S1, len(ids_all)), replace=False).tolist())
    idset = set(ids)
    fpath = kaggle_env.find_input("feats_B.parquet")
    import pyarrow.parquet as pq

    _, meta = features.names_from_columns(pq.ParquetFile(fpath).schema_arrow.names)
    fb = io_utils.read_parquet_compact(fpath, ["s1_id", "cand_id"] + meta)
    fb = fb[fb["s1_id"].isin(idset)].reset_index(drop=True)
    need = set(fb["cand_id"].tolist())
    s1 = io_utils.read_parquet_compact(recs_dir / "records_train_S1.parquet", features.REC_COLS)
    s1 = s1[s1["entity_id"].isin(idset)].reset_index(drop=True)
    pool = pd.concat([io_utils.read_parquet_compact(recs_dir / f"records_train_S{s}.parquet", features.REC_COLS)
                      for s in (2, 3)], ignore_index=True)
    pool = pool[pool["entity_id"].isin(need)].reset_index(drop=True)
    table = scramble.make_scrambler(SEED)
    s1x, poolx = scramble_records(s1, table), scramble_records(pool, table)
    lk = scramble_lookups(pd.read_parquet(recs_dir / "stats_train.parquet"), table)
    kaggle_env.log(f"  scrambled: {len(ids):,} S1, {len(fb):,} pairs; example {s1['norm_name'].iloc[0]!r} -> {s1x['norm_name'].iloc[0]!r}")
    q, p = features.rows_for(fb, s1x, poolx)
    features.set_context(s1x, poolx, lk)
    order = np.argsort(q, kind="stable")
    rest = features.pair_features(q[order], p[order], n_jobs=N_JOBS, log=kaggle_env.log)
    back = np.empty_like(order)
    back[order] = np.arange(len(order))
    X = np.hstack([fb[meta].to_numpy(np.float32), rest[back]])
    pscr = stage1.predict(model, X, N_JOBS)
    scr = fb[["s1_id", "cand_id"]].assign(p1=pscr)
    orig = p1b[p1b["s1_id"].isin(idset)][["s1_id", "cand_id", "p1"]]
    f_orig = decoder.macro_f05(chain(orig, cal, best["lam"], best["mode"], best["method"], best["q"], best["t"] or 0.5), truth, ids)
    f_scr = decoder.macro_f05(chain(scr, cal, best["lam"], best["mode"], best["method"], best["q"], best["t"] or 0.5), truth, ids)
    both = orig.merge(scr, on=["s1_id", "cand_id"], suffixes=("_orig", "_scr"))
    return {"n_s1": len(ids), "n_pairs": int(len(fb)), "f05_orig": round(f_orig, 6), "f05_scrambled": round(f_scr, 6),
            "drop": round(f_orig - f_scr, 6),
            "p1_corr_orig_vs_scr": round(float(np.corrcoef(both["p1_orig"], both["p1_scr"])[0, 1]), 4)}


# ---------------------------------------------------------------------------
# Submission files
# ---------------------------------------------------------------------------


def write_outputs(test_s1_ids: list, cands: pd.DataFrame, pred: pd.DataFrame) -> None:
    """matching_results.tsv (decoder picks) + candidate_pairs.tsv (every scored pair), one row per test S1."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame, col, header in (("candidate_pairs.tsv", cands, "cand_id", ("source1_entity_id", "candidate_entity_ids")),
                                     ("matching_results.tsv", pred, "cand_id", ("source1_entity_id", "matched_entity_ids"))):
        lists = io_utils.pairs_to_lists(frame["s1_id"].tolist(), frame[col].tolist())
        io_utils.write_id_list_tsv(OUT / name, ((s, lists.get(s, [])) for s in test_s1_ids), header=header)


def run_validators(test_dir) -> dict:
    """check_outputs (our pre-flight) + the official validator with --check-ids; results into metrics."""
    res = {}
    problems = check_outputs.check_outputs(OUT, test_dir)
    res["check_outputs"] = "PASS" if not problems else problems[:20]
    val = kaggle_env.find_input("utils/validate_submission.py")
    r = subprocess.run([sys.executable, str(val), "--matching", str(OUT / "matching_results.tsv"), "--candidate",
                        str(OUT / "candidate_pairs.tsv"), "--test-dir", str(test_dir), "--check-ids"],
                       capture_output=True, text=True)
    res["official_validator_returncode"] = r.returncode
    res["official_validator_tail"] = (r.stdout + r.stderr).strip().splitlines()[-15:]
    return res


def main() -> None:
    """Run NB09a end to end."""
    t0 = time.time()
    io_utils.set_seeds(SEED)
    WORK.mkdir(parents=True, exist_ok=True)
    kaggle_env.ensure_rapidfuzz()
    recs_dir = kaggle_env.find_input("records_train_S1.parquet").parent
    s1_dir = kaggle_env.find_input("p1_test.parquet").parent
    test_dir = kaggle_env.find_input_dir("test/test_source1.tsv") / "test"
    kaggle_env.log(f"records {recs_dir}; stage-1 {s1_dir}; test {test_dir}")
    metrics: dict = {"decoder_config": DC, "modes": MODES}
    rng = np.random.default_rng(SEED)

    # ---------------- Half B: calibration, grid, per country ----------------
    p1b = io_utils.read_parquet_compact(s1_dir / "p1_B.parquet")
    ids = sorted(p1b["s1_id"].unique().tolist())
    gt = pd.read_parquet(recs_dir / "gt.parquet")
    truth = gt[gt["s1_id"].isin(set(ids))][["s1_id", "match_id"]]
    s1_train = io_utils.read_parquet_compact(recs_dir / "records_train_S1.parquet", ["entity_id", "country"])
    cty = pd.Series(s1_train["country"].to_numpy(), index=s1_train["entity_id"].to_numpy())
    # blocking-miss rate: share of B S1s with a true match outside the scored candidates
    in_c = truth.merge(p1b[["s1_id", "cand_id"]], left_on=["s1_id", "match_id"], right_on=["s1_id", "cand_id"], how="left")
    miss_s1 = in_c[in_c["cand_id"].isna()]["s1_id"].nunique()
    q_est = round(miss_s1 / len(ids), 4)
    qs = [0.0, q_est]
    metrics["B"] = {"n_s1": len(ids), "n_pairs": int(len(p1b)), "n_true_pairs": int(len(truth)),
                    "pair_recall": round(float(in_c["cand_id"].notna().mean()), 5), "q_est": q_est,
                    "floor_all_empty": round(decoder.macro_f05(p1b.iloc[:0], truth, ids), 6)}
    kaggle_env.log(f"B: {metrics['B']}")
    p_cf = crossfit_calibrated(p1b)
    metrics["B"]["ece_raw"] = round(decoder.ece(p1b["p1"].to_numpy(), p1b["label"].to_numpy()), 5)
    metrics["B"]["ece_crossfit_isotonic"] = round(decoder.ece(p_cf, p1b["label"].to_numpy()), 5)
    best, rows = grid_search(p1b, Fixed(p_cf), truth, ids, qs)
    best_dec = max((r for r in rows if r["method"] == "decoder"), key=lambda r: r["f05"])
    best_thr = max((r for r in rows if r["method"] == "threshold"), key=lambda r: r["f05"])
    metrics["grid"] = rows
    metrics["best"] = best
    metrics["decoder_vs_threshold"] = {"decoder": best_dec, "threshold": best_thr,
                                       "kept": best["method"]}
    kaggle_env.log(f"best on B: {best}")
    pred_b = chain(p1b, Fixed(p_cf), best["lam"], best["mode"], best["method"], best["q"], best["t"] or 0.5)
    metrics["B"]["f05"] = best["f05"]
    metrics["B"]["per_country"] = per_country(pred_b, truth, ids, cty)
    fb, per_b = decoder.macro_f05(pred_b, truth, ids, return_per_entity=True)
    metrics["B"]["n_entities_below_1"] = int((per_b < 1).sum())
    metrics["B"]["pred_empty_rate"] = round(1 - pred_b["s1_id"].nunique() / len(ids), 5)
    metrics["B"]["true_empty_rate"] = round(1 - truth["s1_id"].nunique() / len(ids), 5)

    # ---------------- LOCO-lite: calibration + decoder params fitted on one country, scored on the other ----------------
    cb = p1b["s1_id"].map(cty).to_numpy()
    countries = sorted(set(cb.tolist()))
    loco = {}
    for c_tr in countries:
        for c_te in countries:
            if c_tr == c_te:
                continue
            tr, te = p1b[cb == c_tr].reset_index(drop=True), p1b[cb == c_te].reset_index(drop=True)
            ids_tr, ids_te = sorted(tr["s1_id"].unique().tolist()), sorted(te["s1_id"].unique().tolist())
            cal = decoder.Isotonic().fit(tr["p1"].to_numpy(), tr["label"].to_numpy())
            b_tr, _ = grid_search(tr, cal, truth, ids_tr, qs)
            f = decoder.macro_f05(chain(te, cal, b_tr["lam"], b_tr["mode"], b_tr["method"], b_tr["q"], b_tr["t"] or 0.5),
                                  truth, ids_te)
            loco[f"{c_tr}->{c_te}"] = {"f05": round(f, 6), "config": b_tr}
    metrics["loco_lite"] = loco
    metrics["loco_lite_mean"] = round(float(np.mean([v["f05"] for v in loco.values()])), 6) if loco else None
    kaggle_env.log(f"LOCO-lite: {metrics['loco_lite_mean']}")
    kaggle_env.write_json(metrics, WORK / "metrics.json")

    # ---------------- final calibrator on all of B ----------------
    cal = decoder.Isotonic().fit(p1b["p1"].to_numpy(), p1b["label"].to_numpy())

    # ---------------- persist the final artifacts (reproducibility: predict.py / NB11) ----------------
    art = WORK / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    cal.save(art / "calibrator.npz")
    kaggle_env.write_json({"decoder": {k: best[k] for k in ("method", "mode", "lam", "q", "t")},
                           "max_n": MAX_N, "min_p": DC.get("min_p", 0.0), "seed": SEED}, art / "final_config.json")
    (art / "stage1_model.txt").write_bytes((s1_dir / "stage1_model.txt").read_bytes())
    if (s1_dir / "prerank_half0.txt").exists():  # older NB06 outputs may lack the cross-fitted pre-ranker
        for f in ("prerank_half0.txt", "prerank_half1.txt"):
            (art / f).write_bytes((s1_dir / f).read_bytes())
    kaggle_env.log(f"artifacts saved: {[p.name for p in art.iterdir()]}")

    # ---------------- scrambled-letter run ----------------
    import lightgbm as lgb

    model = lgb.Booster(model_file=str(s1_dir / "stage1_model.txt"))
    try:
        metrics["scrambled"] = scrambled_run(recs_dir, p1b, cal, best, truth, ids, model, rng)
    except Exception as e:  # the submission must not depend on this diagnostic
        metrics["scrambled"] = {"error": repr(e)}
    kaggle_env.log(f"scrambled: {metrics['scrambled']}")
    kaggle_env.write_json(metrics, WORK / "metrics.json")
    del p1b, pred_b
    gc.collect()

    # ---------------- test: decode + write + validate ----------------
    p1t = io_utils.read_parquet_compact(s1_dir / "p1_test.parquet", ["s1_id", "cand_id", "p1"])
    test_s1 = io_utils.load_tsv(test_dir / "test_source1.tsv")[["entity_id", "country"]]
    test_ids = test_s1["entity_id"].tolist()
    assert set(p1t["s1_id"].unique().tolist()) <= set(test_ids), "scored S1 not in test_source1"
    pred_t = chain(p1t, cal, best["lam"], best["mode"], best["method"], best["q"], best["t"] or 0.5)
    tc = pd.Series(test_s1["country"].to_numpy(), index=test_ids)
    n_pred = pred_t.groupby("s1_id").size()
    dist = {}
    for c, g in test_s1.groupby("country"):
        k = n_pred.reindex(g["entity_id"].tolist(), fill_value=0)
        dist[c] = {"n_s1": int(len(g)), "pred_empty_rate": round(float((k == 0).mean()), 5),
                   "mean_pred_size": round(float(k.mean()), 3),
                   "mean_cands": round(float(p1t[p1t["s1_id"].map(tc) == c].shape[0] / len(g)), 2)}
    metrics["test"] = {"n_s1": len(test_ids), "n_scored_pairs": int(len(p1t)), "n_pred_pairs": int(len(pred_t)),
                       "per_country": dist, "train_singleton_rate": 0.0558}
    kaggle_env.log(f"test: {metrics['test']}")
    write_outputs(test_ids, p1t, pred_t)
    del p1t, pred_t
    gc.collect()
    metrics["validation"] = run_validators(test_dir)
    kaggle_env.log(f"validation: {metrics['validation']}")
    metrics["runtime_s"] = round(time.time() - t0, 1)
    kaggle_env.write_json(metrics, WORK / "metrics.json")
    kaggle_env.log(f"done in {metrics['runtime_s']}s")


if __name__ == "__main__":
    main()
