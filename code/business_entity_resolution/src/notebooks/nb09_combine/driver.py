"""NB09 driver — collective features + combiner + calibration + one-owner + decoder, full validation, test outputs.

architecture.md SS6.11 / plan SS17-21. CPU kernel, internet OFF. Bundled with
kaggle_env/io_utils/normalize/corpus_stats/blocking/explain_diff/features/stage1/
decoder/exclusivity/scramble/collective/combiner/metrics/check_outputs.

Inputs (globbed under /kaggle/input):
- NB02 output: records_*.parquet, stats_train.parquet, gt.parquet
- NB06 output: feats_B.parquet, p1_B.parquet, feats_test/part-*.parquet, p1_test.parquet, stage1_model.txt
- optional er-judge-scores: judge_B.parquet, judge_test.parquet (s1_id, cand_id, judge_p)
- er-data (test TSVs + utils/validate_submission.py); NB00 output (wheels/, rapidfuzz)

Validation on Half B (plan SS21): OOF combiner, cross-fitted isotonic
calibration, coarse grid over (one-owner mode x lambda x q) and a global
threshold; LOCO (combiner + calibration + decoder trained on B-country X,
scored on B-country Y); bootstrap CI vs the stage-1 pipeline (sub-01);
scrambled-letter run; ablation rows; error buckets.

Outputs in /kaggle/working: output/{matching_results,candidate_pairs}.tsv,
combiner_model.txt, reports/{ablation.md, errors.md}, b_oof.parquet, metrics.json.
"""

import gc
import os
import time

import numpy as np
import pandas as pd

WORK = kaggle_env.WORK_DIR
OUT = WORK / "output"
REP = WORK / "reports"
N_JOBS = os.cpu_count() or 1
SEED = CONFIG["seed"]
CC = CONFIG["combiner"]
DC = CONFIG["decoder"]
P_FLOOR = float(CC.get("p_floor", 0.001))  # pairs below keep p1 (certain negatives; same rule on test)
LAMBDAS, MAX_N, MIN_P = DC["lambda_grid"], DC["max_n"], DC.get("min_p", 0.0)
MODES = ["none", "hard"]
THRESHOLDS = np.round(np.arange(0.10, 0.951, 0.05), 3)
SCR_S1 = int(os.environ.get("ER_SCR_S1", 30_000))
SIB_COLS = ["entity_id", "norm_name", "norm_addr", "house_number", "postcodes"]
USE_JUDGE = os.environ.get("ER_USE_JUDGE", "auto")


def log(m):
    """Shorthand for the flushed, timestamped kernel log."""
    kaggle_env.log(m)


def collective_for(df: pd.DataFrame, recs: pd.DataFrame) -> np.ndarray:
    """Within-entity + sibling features for a pair frame grouped by S1 (s1_id, cand_id, p1)."""
    ent = collective.entity_features(df["s1_id"].to_numpy(dtype=object), df["p1"].to_numpy(np.float64))
    r = recs.reindex(df["cand_id"].to_numpy())
    sib = collective.sibling_features(df["s1_id"].to_numpy(dtype=object), df["p1"].to_numpy(np.float64),
                                      r["norm_name"].fillna("").to_numpy(dtype=object),
                                      r["norm_addr"].fillna("").to_numpy(dtype=object),
                                      r["house_number"].fillna("").to_numpy(dtype=object),
                                      r["postcodes"].fillna("").to_numpy(dtype=object))
    return np.hstack([ent, sib])


def pool_records(recs_dir, split: str, ids: set) -> pd.DataFrame:
    """Sibling-feature columns of the pool records referenced by `ids`, indexed by entity_id."""
    parts = []
    for s in (2, 3):
        d = io_utils.read_parquet_compact(recs_dir / f"records_{split}_S{s}.parquet", SIB_COLS)
        parts.append(d[d["entity_id"].isin(ids)])
    out = pd.concat(parts, ignore_index=True)
    out = out.astype({c: object for c in SIB_COLS})
    return out.set_index("entity_id")


def load_judge(name: str, df: pd.DataFrame) -> pd.DataFrame | None:
    """judge_p aligned with df rows (NaN where not scored), or None when no judge scores are attached."""
    if USE_JUDGE == "off":
        return None
    try:
        path = kaggle_env.find_input(f"judge_{name}.parquet")
    except FileNotFoundError:
        return None
    j = pd.read_parquet(path, columns=["s1_id", "cand_id", "judge_p"])
    m = df[["s1_id", "cand_id"]].astype(str).merge(j.astype({"s1_id": str, "cand_id": str}), on=["s1_id", "cand_id"], how="left")
    return m[["judge_p"]].reset_index(drop=True)


def combine_p(p1: np.ndarray, mask: np.ndarray, pc: np.ndarray) -> np.ndarray:
    """Final raw probability: combiner output where p1 >= P_FLOOR, else p1."""
    out = p1.astype(np.float64).copy()
    out[mask] = pc
    return out


def train_combiner(X, y, groups, folds: int, log_fn=log):
    """OOF + full combiner. Outputs (oof, fold model per row id array, models, full model, n_rounds)."""
    prm = combiner.params(CC, SEED)
    r = stage1.train_oof(X, y, groups, prm, folds=folds, seed=SEED, max_rounds=CC.get("max_rounds", 1500),
                         early_stop=50, log=log_fn)
    n_rounds = int(np.mean(r["best_iters"]) * 1.1) + 1
    full = stage1.train_full(X, y, prm, n_rounds)
    return r, full, n_rounds


def evaluate(df, pcol, truth, ids, qs, modes=MODES, lambdas=LAMBDAS, thresholds=THRESHOLDS, cal=None):
    """Grid on B with cross-fitted isotonic calibration (unless `cal` given). Outputs (best, rows, pred)."""
    if cal is None:
        cal = combiner.Fixed(combiner.crossfit_calibrated(df[pcol].to_numpy(np.float64), df["label"].to_numpy(),
                                                          df["s1_id"].to_numpy(dtype=object)))
    best, rows = combiner.grid_search(df, cal, truth, ids, qs, lambdas, modes, thresholds, pcol, MAX_N, MIN_P)
    return best, rows, combiner.run_config(df, cal, best, pcol, MAX_N, MIN_P)


def per_country(pred, truth, ids, cty) -> dict:
    """Macro F0.5, singleton accuracy and non-singleton F0.5 per country (+ ALL), with the floor."""
    out = {}
    tn = truth.groupby("s1_id").size()
    groups = {"ALL": list(ids)}
    for c in sorted(set(cty.reindex(ids).tolist())):
        groups[c] = [i for i in ids if cty.get(i) == c]
    for c, sub in groups.items():
        f, per = decoder.macro_f05(pred, truth, sub, return_per_entity=True)
        single = tn.reindex(sub, fill_value=0).eq(0).to_numpy()
        out[c] = {"n_s1": len(sub), "f05": round(f, 6),
                  "floor": round(decoder.macro_f05(pred.iloc[:0], truth, sub), 6),
                  "singleton_acc": round(float(per.to_numpy()[single].mean()), 6) if single.any() else None,
                  "non_singleton_f05": round(float(per.to_numpy()[~single].mean()), 6) if (~single).any() else None}
    return out


def loco(df, Xm, mask, truth, cty, qs, feat_names) -> dict:
    """LOCO (plan SS21.3): combiner (3-fold OOF + full) + calibration + decoder params from B-country X, scored on Y."""
    cb = df["s1_id"].map(cty).to_numpy()
    out = {}
    countries = sorted(set(cb.tolist()))
    for x in countries:
        tr = cb == x
        trm = tr & mask
        r, full, _ = train_combiner(Xm[trm[mask]], df["label"].to_numpy()[trm], df["s1_id"].to_numpy(dtype=object)[trm], 3,
                                    log_fn=lambda m: None)
        dtr = df[tr].reset_index(drop=True)
        dtr["pl"] = combine_p(dtr["p1"].to_numpy(), mask[tr], r["oof"])
        ids_tr = sorted(dtr["s1_id"].unique().tolist())
        cal = decoder.Isotonic().fit(dtr["pl"].to_numpy(), dtr["label"].to_numpy())
        best, _ = combiner.grid_search(dtr, combiner.Fixed(combiner.crossfit_calibrated(
            dtr["pl"].to_numpy(), dtr["label"].to_numpy(), dtr["s1_id"].to_numpy(dtype=object))), truth, ids_tr, qs,
            LAMBDAS, MODES, THRESHOLDS, "pl", MAX_N, MIN_P)
        for y in countries:
            if y == x:
                continue
            te = cb == y
            dte = df[te].reset_index(drop=True)
            dte["pl"] = combine_p(dte["p1"].to_numpy(), mask[te], stage1.predict(full, Xm[(te & mask)[mask]], N_JOBS))
            ids_te = sorted(dte["s1_id"].unique().tolist())
            f = decoder.macro_f05(combiner.run_config(dte, cal, best, "pl", MAX_N, MIN_P), truth, ids_te)
            out[f"{x}->{y}"] = {"f05": round(f, 6), "config": best}
            log(f"  LOCO {x}->{y}: {f:.5f}")
    out["mean"] = round(float(np.mean([v["f05"] for k, v in out.items()])), 6) if out else None
    return out


def loco_stage1(df, truth, cty, qs) -> dict:
    """LOCO of the stage-1 pipeline (sub-01): calibration + decoder from country X, scored on Y."""
    cb = df["s1_id"].map(cty).to_numpy()
    out = {}
    for x in sorted(set(cb.tolist())):
        tr = df[cb == x].reset_index(drop=True)
        cal = decoder.Isotonic().fit(tr["p1"].to_numpy(), tr["label"].to_numpy())
        ids_tr = sorted(tr["s1_id"].unique().tolist())
        best, _ = combiner.grid_search(tr, cal, truth, ids_tr, qs, LAMBDAS, MODES, THRESHOLDS, "p1", MAX_N, MIN_P)
        for y in sorted(set(cb.tolist())):
            if y != x:
                te = df[cb == y].reset_index(drop=True)
                f = decoder.macro_f05(combiner.run_config(te, cal, best, "p1", MAX_N, MIN_P), truth,
                                      sorted(te["s1_id"].unique().tolist()))
                out[f"{x}->{y}"] = round(f, 6)
    out["mean"] = round(float(np.mean(list(out.values()))), 6) if out else None
    return out


def scrambled_run(recs_dir, s1_dir, df, fold_models, fold_of_row, mask, cal, best, truth, rng, feat_names, meta_names, judge_cols):
    """Full pipeline on a scrambled B subset (plan SS21.4): stage-1 features, stage-1, collective, combiner (fold models)."""
    import lightgbm as lgb

    ids = sorted(rng.choice(np.array(sorted(df["s1_id"].unique().tolist()), dtype=object),
                            min(SCR_S1, df["s1_id"].nunique()), replace=False).tolist())
    idset = set(ids)
    rows = np.flatnonzero(df["s1_id"].isin(idset).to_numpy())
    sub = df.iloc[rows].reset_index(drop=True)
    s1 = io_utils.read_parquet_compact(recs_dir / "records_train_S1.parquet", features.REC_COLS)
    s1 = s1[s1["entity_id"].isin(idset)].reset_index(drop=True)
    need = set(sub["cand_id"].tolist())
    pool = pd.concat([io_utils.read_parquet_compact(recs_dir / f"records_train_S{s}.parquet", features.REC_COLS)
                      for s in (2, 3)], ignore_index=True)
    pool = pool[pool["entity_id"].isin(need)].reset_index(drop=True)
    table = scramble.make_scrambler(SEED)
    s1x, poolx = scramble.scramble_normalized(s1, table), scramble.scramble_normalized(pool, table)
    st = pd.read_parquet(recs_dir / "stats_train.parquet")
    st = st[st["field"].isin(["norm_name", "norm_addr"])].copy()
    st["token"] = [str(t).translate(table) for t in st["token"].tolist()]
    lk = features.token_lookups(st)
    q, p = features.rows_for(sub, s1x, poolx)
    features.set_context(s1x, poolx, lk)
    order = np.argsort(q, kind="stable")
    rest = features.pair_features(q[order], p[order], n_jobs=N_JOBS, log=log)
    back = np.empty_like(order)
    back[order] = np.arange(len(order))
    fx = pd.DataFrame(np.hstack([sub[meta_names].to_numpy(np.float32), rest[back]]), columns=feat_names)
    m1 = lgb.Booster(model_file=str(s1_dir / "stage1_model.txt"))
    p1x = stage1.predict(m1, fx.to_numpy(np.float32), N_JOBS)
    subx = sub[["s1_id", "cand_id", "label"]].assign(p1=p1x)
    recx = poolx[SIB_COLS].astype({c: object for c in SIB_COLS}).set_index("entity_id")
    coll = collective_for(subx, recx)
    jx = sub[judge_cols].reset_index(drop=True) if judge_cols else None
    X, _ = combiner.matrix(fx, p1x, coll, jx.rename(columns={judge_cols[0]: "judge_p"}) if jx is not None else None,
                           base_names=feat_names)
    mk = p1x >= P_FLOOR
    # fold per S1 (rows under the floor originally have no fold of their own; S1s never trained on get fold 0)
    fold_by_s1 = dict(zip(df["s1_id"].to_numpy()[fold_of_row >= 0], fold_of_row[fold_of_row >= 0]))
    fo = np.array([fold_by_s1.get(s, 0) for s in sub["s1_id"].tolist()])
    pc = np.empty(int(mk.sum()))
    Xm, fom = X[mk], fo[mk]
    for f, m in enumerate(fold_models):  # each row scored by the fold model that did not see its S1
        sel = fom == f
        if sel.any():
            pc[sel] = stage1.predict(m, Xm[sel], N_JOBS)
    subx["pl"] = combine_p(p1x, mk, pc)
    orig = sub[["s1_id", "cand_id", "pl"]]
    f_o = decoder.macro_f05(combiner.run_config(orig, cal, best, "pl", MAX_N, MIN_P), truth, ids)
    f_s = decoder.macro_f05(combiner.run_config(subx, cal, best, "pl", MAX_N, MIN_P), truth, ids)
    return {"n_s1": len(ids), "n_pairs": int(len(sub)), "f05_orig": round(f_o, 6), "f05_scrambled": round(f_s, 6),
            "drop": round(f_o - f_s, 6), "p1_corr": round(float(np.corrcoef(sub["p1"], p1x)[0, 1]), 4)}


def error_report(df, pred, truth, ids, recs_dir, path) -> dict:
    """Error buckets (CLAUDE.md SS4.4) -> counts + 3 raw examples per bucket in a markdown file."""
    eb = metrics.error_buckets(pred, truth, df, ids, name_sim="norm_name_token_set_ratio", number_conflict="any_number_conflict")
    counts = eb["bucket"].value_counts().to_dict()
    lines = [f"# Error buckets on Half B ({len(eb):,} wrong of {len(ids):,} S1)", ""]
    lines += [f"- {b}: {counts.get(b, 0):,}" for b in metrics.BUCKETS] + [""]
    ex = eb.groupby("bucket").head(3)
    want_s1 = set(ex["s1_id"])
    t = truth[truth["s1_id"].isin(want_s1)]
    pr = pred[pred["s1_id"].isin(want_s1)]
    rid = set(t["match_id"]) | set(pr["cand_id"]) | want_s1
    cols = ["entity_id", "raw_name", "raw_addr"]
    raw = pd.concat([io_utils.read_parquet_compact(recs_dir / f"records_train_S{s}.parquet", cols) for s in (1, 2, 3)])
    raw = raw[raw["entity_id"].isin(rid)].astype(object).set_index("entity_id")
    txt = lambda i: f"{i}: {raw.at[i, 'raw_name']} | {raw.at[i, 'raw_addr']}" if i in raw.index else i
    pmap = df.set_index(["s1_id", "cand_id"])["pl"]
    for b in metrics.BUCKETS:
        sub = ex[ex["bucket"] == b]
        if sub.empty:
            continue
        lines += [f"## {b} ({counts.get(b, 0):,})", ""]
        for s in sub["s1_id"]:
            tt, pp = set(t.loc[t["s1_id"] == s, "match_id"]), set(pr.loc[pr["s1_id"] == s, "cand_id"])
            lines.append(f"- S1 {txt(s)}")
            for i in sorted(tt | pp):
                tag = "TP" if i in tt and i in pp else ("FN" if i in tt else "FP")
                pv = pmap.get((s, i), float("nan"))
                lines.append(f"  - {tag} p={pv:.3f} {txt(i)}")
        lines.append("")
    path.write_text("\n".join(lines[:200]) + "\n", encoding="utf-8")
    return counts


def write_outputs(test_ids, cands, pred):
    """matching_results.tsv + candidate_pairs.tsv, one row per test S1 (candidates = exactly the scored set)."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame, header in (("candidate_pairs.tsv", cands, ("source1_entity_id", "candidate_entity_ids")),
                                ("matching_results.tsv", pred, ("source1_entity_id", "matched_entity_ids"))):
        lists = io_utils.pairs_to_lists(frame["s1_id"].tolist(), frame["cand_id"].tolist())
        io_utils.write_id_list_tsv(OUT / name, ((s, lists.get(s, [])) for s in test_ids), header=header)


def main() -> None:
    """Run NB09 end to end."""
    import subprocess
    import sys

    import lightgbm as lgb

    t0 = time.time()
    io_utils.set_seeds(SEED)
    for d in (WORK, OUT, REP):
        d.mkdir(parents=True, exist_ok=True)
    kaggle_env.ensure_rapidfuzz()
    recs_dir = kaggle_env.find_input("records_train_S1.parquet").parent
    s1_dir = kaggle_env.find_input("p1_test.parquet").parent
    test_dir = kaggle_env.find_input_dir("test/test_source1.tsv") / "test"
    metrics_out: dict = {"combiner_config": CC, "decoder_config": DC, "p_floor": P_FLOOR}
    rng = np.random.default_rng(SEED)

    # ---------------- B: features, p1, collective ----------------
    fb = io_utils.read_parquet_compact(s1_dir / "feats_B.parquet")
    p1b = io_utils.read_parquet_compact(s1_dir / "p1_B.parquet", ["s1_id", "cand_id", "p1"])
    assert (fb["s1_id"].to_numpy() == p1b["s1_id"].to_numpy()).all() and (fb["cand_id"].to_numpy() == p1b["cand_id"].to_numpy()).all()
    fb["p1"] = p1b["p1"].to_numpy()
    del p1b
    fb = fb.sort_values("s1_id", kind="stable").reset_index(drop=True)
    feat_names, meta_names = features.names_from_columns(fb.columns)
    ids = sorted(fb["s1_id"].unique().tolist())
    gt = pd.read_parquet(recs_dir / "gt.parquet")
    truth = gt[gt["s1_id"].isin(set(ids))][["s1_id", "match_id"]]
    s1t = io_utils.read_parquet_compact(recs_dir / "records_train_S1.parquet", ["entity_id", "country"])
    cty = pd.Series(s1t["country"].to_numpy(), index=s1t["entity_id"].to_numpy())
    in_c = truth.merge(fb[["s1_id", "cand_id"]], left_on=["s1_id", "match_id"], right_on=["s1_id", "cand_id"], how="left")
    q_est = round(in_c[in_c["cand_id"].isna()]["s1_id"].nunique() / len(ids), 4)
    qs = [0.0, q_est]
    mask = fb["p1"].to_numpy() >= P_FLOOR
    metrics_out["B"] = {"n_s1": len(ids), "n_pairs": int(len(fb)), "pair_recall": round(float(in_c["cand_id"].notna().mean()), 5),
                        "q_est": q_est, "n_pairs_combiner": int(mask.sum()),
                        "true_pairs_below_floor": int(fb.loc[~mask, "label"].sum())}
    log(f"B: {metrics_out['B']}")
    recs_b = pool_records(recs_dir, "train", set(fb["cand_id"].tolist()))
    coll = collective_for(fb, recs_b)
    judge_b = load_judge("B", fb)
    metrics_out["judge_used"] = judge_b is not None
    X, names = combiner.matrix(fb[feat_names], fb["p1"].to_numpy(), coll, judge_b, base_names=feat_names)
    del coll
    gc.collect()
    Xm = X[mask]
    del X
    gc.collect()
    y, groups = fb["label"].to_numpy(), fb["s1_id"].to_numpy(dtype=object)
    log(f"combiner matrix {Xm.shape} ({len(names)} features, judge {judge_b is not None})")
    r, full, n_rounds = train_combiner(Xm, y[mask], groups[mask], CC.get("folds", 5))
    full.save_model(str(WORK / "combiner_model.txt"))
    fb["pl"] = combine_p(fb["p1"].to_numpy(), mask, r["oof"])
    fold_of_row = np.full(len(fb), -1)
    fold_of_row[mask] = stage1.group_folds(groups[mask], CC.get("folds", 5), SEED)
    gain = sorted(zip(names, full.feature_importance("gain")), key=lambda t: -t[1])
    metrics_out["combiner"] = {"best_iters": r["best_iters"], "fold_auc": r["fold_auc"], "n_rounds_full": n_rounds,
                               "auc_B_oof": stage1.eval_by_group(y, fb["pl"].to_numpy(), fb["s1_id"].map(cty).to_numpy()),
                               "auc_B_stage1": stage1.eval_by_group(y, fb["p1"].to_numpy(), fb["s1_id"].map(cty).to_numpy()),
                               "top_gain": [(k, round(float(v), 1)) for k, v in gain[:25]]}
    kaggle_env.write_json(metrics_out, WORK / "metrics.json")

    # ---------------- ablation on B ----------------
    abl = []
    floor = decoder.macro_f05(fb.iloc[:0], truth, ids)
    abl.append(("All-empty floor", floor))
    ident = type("Ident", (), {"predict": staticmethod(lambda p: p)})()
    _, thr_rows = combiner.grid_search(fb, ident, truth, ids, [0.0], [1.0], ["none"], THRESHOLDS, "p1", MAX_N, MIN_P)
    b_thr = max((r_ for r_ in thr_rows if r_["method"] == "threshold"), key=lambda r_: r_["f05"])
    abl.append(("Stage-1 + global threshold", b_thr["f05"]))
    b_dec, _, _ = evaluate(fb, "p1", truth, ids, qs, modes=["none"], thresholds=[])
    abl.append(("+ calibration + expected-F decoder", b_dec["f05"]))
    b_s1, _, pred_s1 = evaluate(fb, "p1", truth, ids, qs)
    abl.append(("+ one-owner rule (stage-1 pipeline = sub-01)", b_s1["f05"]))
    best, rows, pred_b = evaluate(fb, "pl", truth, ids, qs)
    abl.append(("+ combiner with collective features" + (" + judge" if judge_b is not None else ""), best["f05"]))
    metrics_out["ablation"] = [{"config": a, "B_f05": round(v, 6)} for a, v in abl]
    metrics_out["best"] = best
    metrics_out["grid"] = rows
    metrics_out["B"]["per_country"] = per_country(pred_b, truth, ids, cty)
    metrics_out["B"]["per_country_stage1_pipeline"] = per_country(pred_s1, truth, ids, cty)
    _, pe_new = decoder.macro_f05(pred_b, truth, ids, return_per_entity=True)
    _, pe_old = decoder.macro_f05(pred_s1, truth, ids, return_per_entity=True)
    metrics_out["bootstrap_combiner_vs_stage1"] = metrics.bootstrap_diff(pe_old.to_dict(), pe_new.to_dict())
    log(f"ablation {metrics_out['ablation']}; bootstrap {metrics_out['bootstrap_combiner_vs_stage1']}")
    fb[["s1_id", "cand_id", "label", "p1", "pl"]].to_parquet(WORK / "b_oof.parquet", index=False)
    kaggle_env.write_json(metrics_out, WORK / "metrics.json")

    # ---------------- LOCO ----------------
    metrics_out["loco"] = loco(fb, Xm, mask, truth, cty, qs, names)
    metrics_out["loco_stage1_pipeline"] = loco_stage1(fb, truth, cty, qs)
    log(f"LOCO combiner {metrics_out['loco']['mean']} vs stage-1 pipeline {metrics_out['loco_stage1_pipeline']['mean']}")
    kaggle_env.write_json(metrics_out, WORK / "metrics.json")

    # ---------------- errors ----------------
    metrics_out["error_buckets"] = error_report(fb, pred_b, truth, ids, recs_dir, REP / "errors.md")
    log(f"error buckets {metrics_out['error_buckets']}")

    # ---------------- scrambled ----------------
    cal_final = decoder.Isotonic().fit(fb["pl"].to_numpy(), fb["label"].to_numpy())
    jcols = []
    if judge_b is not None:
        fb["judge_p"] = judge_b["judge_p"].to_numpy()
        jcols = ["judge_p"]
    try:
        metrics_out["scrambled"] = scrambled_run(recs_dir, s1_dir, fb, r["models"], fold_of_row, mask, cal_final, best,
                                                 truth, rng, feat_names, meta_names, jcols)
    except Exception as e:  # diagnostic only; the submission must not depend on it
        metrics_out["scrambled"] = {"error": repr(e)}
    log(f"scrambled {metrics_out['scrambled']}")
    kaggle_env.write_json(metrics_out, WORK / "metrics.json")
    abl_md = ["| Configuration | B F0.5 | LOCO mean | Scrambled drop |", "|---|---|---|---|"]
    for a, v in abl:
        lm = metrics_out["loco"]["mean"] if a.startswith("+ combiner") else (
            metrics_out["loco_stage1_pipeline"]["mean"] if a.startswith("+ one-owner") else "")
        sd = metrics_out["scrambled"].get("drop", "") if a.startswith("+ combiner") else ""
        abl_md.append(f"| {a} | {v:.4f} | {lm} | {sd} |")
    (REP / "ablation.md").write_text("\n".join(abl_md) + "\n", encoding="utf-8")
    del Xm, fb, recs_b, pred_b, pred_s1
    gc.collect()

    # ---------------- test ----------------
    recs_t = None
    outs = []
    parts = sorted((s1_dir / "feats_test").glob("part-*.parquet"))
    p1t_all = io_utils.read_parquet_compact(s1_dir / "p1_test.parquet", ["s1_id", "cand_id", "p1"])
    start = 0
    judge_t_all = None
    for k, part in enumerate(parts):
        ft = io_utils.read_parquet_compact(part)
        p1t = p1t_all.iloc[start : start + len(ft)].reset_index(drop=True)
        start += len(ft)
        assert (ft["s1_id"].to_numpy() == p1t["s1_id"].to_numpy()).all()
        ft["p1"] = p1t["p1"].to_numpy()
        ft = ft.sort_values("s1_id", kind="stable").reset_index(drop=True)
        rt = pool_records(recs_dir, "test", set(ft["cand_id"].tolist()))
        coll = collective_for(ft, rt)
        jt = load_judge("test", ft) if metrics_out["judge_used"] else None
        Xt, _ = combiner.matrix(ft[feat_names], ft["p1"].to_numpy(), coll, jt, base_names=feat_names)
        mk = ft["p1"].to_numpy() >= P_FLOOR
        pc = stage1.predict(full, Xt[mk], N_JOBS)
        outs.append(ft[["s1_id", "cand_id"]].assign(p1=ft["p1"].to_numpy(), pl=combine_p(ft["p1"].to_numpy(), mk, pc)))
        log(f"  test part {k + 1}/{len(parts)}: {len(ft):,} pairs")
        del ft, Xt, coll, rt
        gc.collect()
    assert start == len(p1t_all), "feats_test parts and p1_test are not aligned"
    pt = pd.concat(outs, ignore_index=True)
    del outs, p1t_all
    test_s1 = io_utils.load_tsv(test_dir / "test_source1.tsv")[["entity_id", "country"]]
    test_ids = test_s1["entity_id"].tolist()
    assert set(pt["s1_id"].unique().tolist()) <= set(test_ids)
    pred_t = combiner.run_config(pt, cal_final, best, "pl", MAX_N, MIN_P)
    n_pred = pred_t.groupby("s1_id").size()
    tc = pd.Series(test_s1["country"].to_numpy(), index=test_ids)
    ncand = pt.groupby("s1_id").size()
    metrics_out["test"] = {c: {"n_s1": int(len(g)),
                               "pred_empty_rate": round(float(n_pred.reindex(g["entity_id"].tolist(), fill_value=0).eq(0).mean()), 5),
                               "mean_pred_size": round(float(n_pred.reindex(g["entity_id"].tolist(), fill_value=0).mean()), 3),
                               "mean_cands": round(float(ncand.reindex(g["entity_id"].tolist(), fill_value=0).mean()), 2),
                               "pl_mean": round(float(pt.loc[pt["s1_id"].map(tc) == c, "pl"].mean()), 4)}
                           for c, g in test_s1.groupby("country")}
    log(f"test {metrics_out['test']}")
    write_outputs(test_ids, pt, pred_t)
    pt.to_parquet(WORK / "test_scores.parquet", index=False)
    problems = check_outputs.check_outputs(OUT, test_dir)
    val = kaggle_env.find_input("utils/validate_submission.py")
    vr = subprocess.run([sys.executable, str(val), "--matching", str(OUT / "matching_results.tsv"), "--candidate",
                         str(OUT / "candidate_pairs.tsv"), "--test-dir", str(test_dir), "--check-ids"],
                        capture_output=True, text=True)
    metrics_out["validation"] = {"check_outputs": "PASS" if not problems else problems[:20],
                                 "official_validator_returncode": vr.returncode,
                                 "official_validator_tail": (vr.stdout + vr.stderr).strip().splitlines()[-8:]}
    metrics_out["runtime_s"] = round(time.time() - t0, 1)
    kaggle_env.write_json(metrics_out, WORK / "metrics.json")
    log(f"validation {metrics_out['validation']}; done in {metrics_out['runtime_s']}s")


if __name__ == "__main__":
    main()
