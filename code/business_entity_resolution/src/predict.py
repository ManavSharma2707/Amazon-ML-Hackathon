"""End-to-end inference entry point (master plan SS22, architecture.md SS6.13).

Three ways to run, the first two producing `matching_results.tsv` +
`candidate_pairs.tsv`:

1. **`run_from_features`** (exact reproduction of the shipped submission).
   Given a saved pair-feature table (one row per scored (s1_id, cand_id) pair
   — exactly NB06's `feats_test/*.parquet` output) plus the artifacts NB09a
   saved (`stage1_model.txt`, `calibrator.npz`, `final_config.json`): apply
   stage-1 -> calibrate -> sharpen -> one-owner -> decode, and write the two
   TSVs. This reproduces `sub-01` exactly (same seed, same artifacts, no
   randomness left) and is what NB11 checks against the actual submission.
   Fast (seconds to a couple of minutes) and fits the 8 GB laptop.

2. **`run_inference`** (full pipeline from raw TSVs, given trained artifacts).
   normalise -> blocking -> pair features -> stage-1 -> calibrate -> decode ->
   write, given a trained blocking pruner, stage-1 model, calibrator and
   decoder config (either the real NB09a ones via `load_artifacts`, or a
   small demo model trained by (3) below). Exercises the identical logic the
   Kaggle notebooks run (src/blocking.py, src/features.py, src/stage1.py,
   src/decoder.py, src/exclusivity.py), end to end, in one process with no
   chunking. That is fine for `sample/` or any similarly modest slice; the
   full ~2M-entity test set needs the per-country/per-chunk scale guard that
   only the Kaggle notebooks implement (CLAUDE.md SS5/SS6 — millions of rows
   do not fit an 8 GB laptop). This is the readable reference implementation
   of the same steps; the real, full-scale run is the numbered notebooks in
   `src/notebooks/` (README.md lists the exact commands and runtimes).

3. **`train_demo`** trains a blocking pruner + stage-1 model + calibrator from
   a data split's own ground truth (e.g. `sample/train/`) — a fully
   self-contained way to get artifacts for (2) without any Kaggle output,
   used by `run_train.sh`. It only ever trains on a split that HAS ground
   truth; applying the result to a different, unlabelled split (e.g. the
   real test set) is exactly what (2) is for — training and predicting are
   two different splits on purpose, never the same one (that would be
   training on the data being predicted).

Neither (1) nor (2) uses the combiner (`combiner.py` / NB09): it was
evaluated and did **not** improve on LOCO-mean over the stage-1 pipeline
below (gated out, see memory.md and the ablation table), so it is not part
of the shipped inference path. It stays available, tested and documented as
an evaluated alternative.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import blocking, corpus_stats, decoder, exclusivity, features, io_utils, normalize, stage1
from . import split as split_mod  # `split` also names run_inference's train/test-split parameter below

REC_COLS = features.REC_COLS
_DEFAULT_DECODER_CFG = {"method": "decoder", "mode": "hard", "lam": 1.0, "q": 0.0, "t": None, "max_n": 20, "min_p": 0.0}


def load_split_records(data_dir: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalised S1 and pool (S2 then S3) frames of one split ("train"/"test").

    Inputs: data_dir - folder with `train/` and `test/` subfolders (organiser layout);
            split. Outputs: (s1, pool) normalised record frames.
    """
    raw = {s: io_utils.load_tsv(data_dir / split / f"{split}_source{s}.tsv") for s in (1, 2, 3)}
    for s, df in raw.items():
        io_utils.assert_source_columns(df, f"{split}_source{s}.tsv")
        io_utils.assert_unique_ids(df, f"S{s}-", f"{split}_source{s}.tsv")
    s1 = normalize.normalize_df(raw[1], "S1")
    pool = pd.concat([normalize.normalize_df(raw[s], f"S{s}") for s in (2, 3)], ignore_index=True)
    return s1, pool


def _fill_missing_channels(cands: pd.DataFrame) -> pd.DataFrame:
    """Add any `features.SPARSE_CHANNELS` score column `blocking.build_candidates` didn't produce, as NaN.

    `blocking.py`'s core pipeline only runs the 7 sparse channels (no dense/
    reverse, see its module docstring); `features.py`'s meta-feature block
    expects every channel in `SPARSE_CHANNELS` to have a `<channel>_score`
    column (LightGBM handles the missing ones as NaN, same as NB06's driver).
    """
    for c in features.SPARSE_CHANNELS:
        col = f"{c}_score"
        if col not in cands:
            cands = cands.assign(**{col: np.float32(np.nan)})
    return cands


def build_features_and_stage1(cands: pd.DataFrame, s1: pd.DataFrame, pool: pd.DataFrame, stats: pd.DataFrame,
                              model, n_jobs: int = 1, log=lambda m: None) -> pd.DataFrame:
    """Full pair features on `cands` -> stage-1 probabilities.

    Inputs: cands - candidate table (blocking.build_candidates output, with
            SPARSE_CHANNELS columns filled in); s1, pool - normalised record
            frames with REC_COLS; stats - NB02-style token statistics
            (corpus_stats.count_tokens output); model - trained stage-1
            LightGBM Booster; n_jobs; log.
    Outputs: DataFrame s1_id, cand_id, p1.
    """
    lookups = features.token_lookups(stats)
    X = features.build_features(cands, s1, pool, lookups, n_jobs=n_jobs, log=log)
    p1 = stage1.predict(model, X.to_numpy(np.float32), n_jobs)
    return pd.DataFrame({"s1_id": cands["s1_id"].to_numpy(), "cand_id": cands["cand_id"].to_numpy(), "p1": p1})


def postprocess(p1_df: pd.DataFrame, calibrator, decoder_cfg: dict) -> pd.DataFrame:
    """calibrate -> sharpen -> one-owner rule -> expected-F0.5 decoder (plan SS18.2/SS19/SS20).

    Inputs: p1_df - s1_id, cand_id, p1; calibrator - object with `.predict(p1) -> p`
            (a `decoder.Isotonic`/`decoder.load_isotonic` result); decoder_cfg -
            {"method", "mode", "lam", "q", "t", "max_n", "min_p"} (NB09a's saved
            `final_config.json`, or `_DEFAULT_DECODER_CFG`).
    Outputs: the chosen (s1_id, cand_id) pairs.
    """
    p = decoder.sharpen(calibrator.predict(p1_df["p1"].to_numpy(np.float64)), decoder_cfg.get("lam", 1.0))
    d = p1_df[["s1_id", "cand_id"]].assign(p=p)
    d["p"] = exclusivity.apply(d, decoder_cfg.get("mode", "hard"), "p")
    if decoder_cfg.get("method", "decoder") == "threshold":
        return decoder.threshold_decode(d, "p", decoder_cfg.get("t") or 0.5)
    return decoder.decode(d, "p", q=decoder_cfg.get("q", 0.0), max_n=decoder_cfg.get("max_n", 20),
                          min_p=decoder_cfg.get("min_p", 0.0))


def write_outputs(test_s1_ids: list[str], cands: pd.DataFrame, pred: pd.DataFrame, out_dir: Path,
                  cand_col: str = "cand_id") -> None:
    """matching_results.tsv (decoder picks) + candidate_pairs.tsv (every scored pair), one row per test S1.

    `cands` IS `candidate_pairs.tsv` by construction (plan SS12.2/SS22): it is
    exactly the set scored above, never a separate, looser list.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame, col, header in (("candidate_pairs.tsv", cands, cand_col, ("source1_entity_id", "candidate_entity_ids")),
                                     ("matching_results.tsv", pred, "cand_id", ("source1_entity_id", "matched_entity_ids"))):
        lists = io_utils.pairs_to_lists(frame["s1_id"].tolist(), frame[col].tolist())
        io_utils.write_id_list_tsv(out_dir / name, ((s, lists.get(s, [])) for s in test_s1_ids), header=header)


# ---------------------------------------------------------------------------
# Artifacts: the trained/fitted pieces inference needs (stage-1 model,
# calibrator, decoder config, and optionally a blocking pruner)
# ---------------------------------------------------------------------------


def load_artifacts(artifacts_dir: Path):
    """(stage1 Booster, calibrator, decoder config dict) from an NB09a `artifacts/` folder.

    NB09a does not save a blocking pruner (the shipped run used NB05's, a
    separate Kaggle-only output); use `load_demo_artifacts` for a folder that
    has one (saved by `train_demo`/`save_demo_artifacts`).
    """
    import lightgbm as lgb

    model = lgb.Booster(model_file=str(artifacts_dir / "stage1_model.txt"))
    cal = decoder.load_isotonic(artifacts_dir / "calibrator.npz")
    cfg = json.loads((artifacts_dir / "final_config.json").read_text(encoding="utf-8"))
    dcfg = dict(cfg["decoder"], max_n=cfg["max_n"], min_p=cfg["min_p"])
    return model, cal, dcfg


def save_demo_artifacts(artifacts_dir: Path, pruner, stage1_model, calibrator, decoder_cfg: dict) -> None:
    """Save a `train_demo` result to disk, in the same layout `load_demo_artifacts` reads back."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    pruner.save_model(str(artifacts_dir / "pruner.txt"))
    stage1_model.save_model(str(artifacts_dir / "stage1_model.txt"))
    calibrator.save(artifacts_dir / "calibrator.npz")
    (artifacts_dir / "final_config.json").write_text(
        json.dumps({"decoder": {k: decoder_cfg[k] for k in ("method", "mode", "lam", "q", "t")},
                    "max_n": decoder_cfg["max_n"], "min_p": decoder_cfg["min_p"]}), encoding="utf-8")


def load_demo_artifacts(artifacts_dir: Path):
    """(pruner, stage1 Booster, calibrator, decoder config dict) saved by `save_demo_artifacts`."""
    import lightgbm as lgb

    model, cal, dcfg = load_artifacts(artifacts_dir)
    pruner = lgb.Booster(model_file=str(artifacts_dir / "pruner.txt"))
    return pruner, model, cal, dcfg


# ---------------------------------------------------------------------------
# 1. Exact reproduction from saved features + artifacts
# ---------------------------------------------------------------------------


def run_from_features(feats_path: Path, test_s1_path: Path, artifacts_dir: Path, out_dir: Path,
                      n_jobs: int = 1) -> dict:
    """Reproduce the shipped submission from a saved feature table + NB09a's artifacts.

    Inputs: feats_path - a feats_test parquet (file, or a folder of `part-*.parquet`,
            as NB06 writes it) with s1_id, cand_id and the feature columns;
            test_s1_path - test_source1.tsv (for the row order / full S1 list);
            artifacts_dir - NB09a's saved `artifacts/`; out_dir; n_jobs.
    Outputs: a small report dict (row/pair counts); writes the two TSVs.
    """
    t0 = time.time()
    model, cal, dcfg = load_artifacts(artifacts_dir)
    feats_path = Path(feats_path)
    parts = sorted(feats_path.glob("part-*.parquet")) if feats_path.is_dir() else [feats_path]
    outs = []
    for part in parts:
        f = io_utils.read_parquet_compact(part)
        names, _ = features.names_from_columns(f.columns)
        p1 = stage1.predict(model, f[names].to_numpy(np.float32), n_jobs)
        outs.append(pd.DataFrame({"s1_id": f["s1_id"].to_numpy(), "cand_id": f["cand_id"].to_numpy(), "p1": p1}))
    p1_df = pd.concat(outs, ignore_index=True)
    pred = postprocess(p1_df, cal, dcfg)
    test_ids = io_utils.load_tsv(test_s1_path)["entity_id"].tolist()
    write_outputs(test_ids, p1_df, pred, out_dir)
    return {"n_s1": len(test_ids), "n_scored_pairs": int(len(p1_df)), "n_pred_pairs": int(len(pred)),
           "runtime_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# 2. Full pipeline from raw data, given trained artifacts
# ---------------------------------------------------------------------------


def run_inference(config: dict, data_dir: Path, out_dir: Path, pruner, stage1_model, calibrator, decoder_cfg: dict,
                  split: str = "test", n_jobs: int = 1, log=print) -> dict:
    """Raw TSVs -> normalise -> blocking -> features -> stage-1 -> calibrate -> decode -> write TSVs.

    No chunking/scale guard (module docstring): intended for `sample/` or a
    similarly modest slice, not the full multi-million-row test set. Never
    trains anything itself (see `train_demo` for that) — `split` here is
    whichever split is being PREDICTED FOR, which must not be the split the
    artifacts were trained on.

    Inputs: config - the loaded `default.yaml`; data_dir - folder with
            `train/`/`test/`; out_dir; pruner, stage1_model, calibrator,
            decoder_cfg - trained artifacts (`load_artifacts`/`load_demo_artifacts`,
            or `train_demo`'s return value); split; n_jobs; log.
    Outputs: a small report dict; writes the two TSVs.
    """
    t0 = time.time()
    log(f"loading + normalising {split}/ records")
    s1, pool = load_split_records(data_dir, split)
    stats = corpus_stats.count_tokens([s1, pool])
    cands = _fill_missing_channels(blocking.build_candidates(s1, pool, config["blocking"], pruner=pruner,
                                                             n_jobs=n_jobs, log=log))
    log(f"blocking: {len(cands):,} candidate pairs for {cands['s1_id'].nunique():,} S1 ({time.time() - t0:.0f}s)")
    p1_df = build_features_and_stage1(cands, s1, pool, stats, stage1_model, n_jobs=n_jobs, log=log)
    log(f"stage-1 done ({time.time() - t0:.0f}s)")
    pred = postprocess(p1_df, calibrator, decoder_cfg)
    s1_ids = s1["entity_id"].tolist()
    write_outputs(s1_ids, cands, pred, out_dir)
    return {"n_s1": len(s1_ids), "n_cand_pairs": int(len(cands)), "n_pred_pairs": int(len(pred)),
           "runtime_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# 3. Train a small, fully self-contained demo model from a split's own labels
# ---------------------------------------------------------------------------


def _label_union(u: pd.DataFrame, s1: pd.DataFrame, pool: pd.DataFrame, gt_pairs: pd.DataFrame) -> np.ndarray:
    """0/1 label per union row from a (s1_id, match_id) ground-truth table."""
    s1_pos = pd.Series(np.arange(len(s1)), index=s1["entity_id"].to_numpy())
    p_pos = pd.Series(np.arange(len(pool)), index=pool["entity_id"].to_numpy())
    gt_pairs = gt_pairs[gt_pairs["match_id"].isin(p_pos.index)]
    keys_true = (s1_pos.loc[gt_pairs["s1_id"]].to_numpy().astype(np.int64) * len(pool)
                + p_pos.loc[gt_pairs["match_id"]].to_numpy())
    keys_u = u["q_row"].to_numpy().astype(np.int64) * len(pool) + u["p_row"].to_numpy()
    return np.isin(keys_u, keys_true).astype(np.int8)


def train_demo(config: dict, data_dir: Path, ground_truth_path: Path, n_jobs: int = 1, log=print,
              train_split: str = "train"):
    """Train a blocking pruner + stage-1 model + calibrator from one split's own ground truth.

    A fully self-contained way to get artifacts for `run_inference` without
    any Kaggle output (used by `run_train.sh`). Only ever trains on a split
    that has ground truth; apply the result to a DIFFERENT, unlabelled split
    with `run_inference` (predicting for the same rows used to fit the model
    would defeat the point of a train/test split).

    Inputs: config; data_dir; ground_truth_path - that split's ground-truth
            TSV; n_jobs; log; train_split - which split has the labels.
    Outputs: (pruner, stage1_model, calibrator, decoder_cfg).
    """
    t0 = time.time()
    s1, pool = load_split_records(data_dir, train_split)
    bc = config["blocking"]
    gt_pairs = split_mod.gt_long(io_utils.load_ground_truth(ground_truth_path))

    log("training a blocking pruner")
    u = blocking.build_union(s1, pool, bc, n_jobs=n_jobs, log=log)
    y_block = _label_union(u, s1, pool, gt_pairs)
    pruner = blocking.train_pruner(u, y_block, seed=config["seed"])
    cands = _fill_missing_channels(blocking.pruned_to_table(
        blocking.prune(u, pruner.predict(u[blocking.PRUNE_FEATURES].to_numpy(np.float32)), bc["prune_top"]), s1, pool))
    log(f"blocking: {len(cands):,} candidate pairs ({time.time() - t0:.0f}s)")

    log("training a stage-1 model")
    key_true = set((gt_pairs["s1_id"] + "|" + gt_pairs["match_id"]).tolist())
    y = np.isin((cands["s1_id"] + "|" + cands["cand_id"]).to_numpy(), list(key_true)).astype(np.int8)
    stats = corpus_stats.count_tokens([s1, pool])
    lookups = features.token_lookups(stats)
    X = features.build_features(cands, s1, pool, lookups, n_jobs=n_jobs, log=log)
    params = stage1.lgb_params(config["stage1"], config["seed"])
    stage1_model = stage1.train_full(X.to_numpy(np.float32), y, params, num_rounds=200, feature_names=features.FEATURES)
    p1 = stage1.predict(stage1_model, X.to_numpy(np.float32), n_jobs)
    calibrator = decoder.Isotonic().fit(p1, y)
    decoder_cfg = dict(_DEFAULT_DECODER_CFG, mode=config.get("exclusivity", "hard"))
    log(f"stage-1 done ({time.time() - t0:.0f}s)")
    return pruner, stage1_model, calibrator, decoder_cfg


def main() -> None:
    """CLI: train a demo model, reproduce the submission from saved features, or apply a trained model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/configs/default.yaml")
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--data-dir", help="folder with train/ and test/")
    parser.add_argument("--n-jobs", type=int, default=1)
    sub = parser.add_mutually_exclusive_group(required=True)
    sub.add_argument("--train", action="store_true", help="train a demo model on --data-dir's train/, needs --ground-truth")
    sub.add_argument("--from-features", help="feats_test parquet (file or folder) for exact reproduction")
    sub.add_argument("--load-artifacts-dir", help="apply a trained model (train_demo/NB09a output) to --split")
    parser.add_argument("--ground-truth", help="train_ground_truth.tsv (only with --train)")
    parser.add_argument("--save-artifacts-dir", help="where --train saves its model (default: --out-dir/artifacts)")
    parser.add_argument("--artifacts-dir", default="artifacts", help="NB09a's saved artifacts/ (for --from-features)")
    parser.add_argument("--test-s1", help="test_source1.tsv (row order for --from-features)")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="split to predict for (--load-artifacts-dir)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.train:
        assert args.ground_truth, "--train needs --ground-truth"
        config = io_utils.load_config(args.config)
        io_utils.set_seeds(config["seed"])
        pruner, model, cal, dcfg = train_demo(config, Path(args.data_dir), Path(args.ground_truth), n_jobs=args.n_jobs)
        save_demo_artifacts(Path(args.save_artifacts_dir or out_dir / "artifacts"), pruner, model, cal, dcfg)
        report = {"saved_to": str(Path(args.save_artifacts_dir or out_dir / "artifacts"))}
    elif args.from_features:
        report = run_from_features(Path(args.from_features), Path(args.test_s1), Path(args.artifacts_dir), out_dir,
                                   n_jobs=args.n_jobs)
    else:
        config = io_utils.load_config(args.config)
        io_utils.set_seeds(config["seed"])
        pruner, model, cal, dcfg = load_demo_artifacts(Path(args.load_artifacts_dir))
        report = run_inference(config, Path(args.data_dir), out_dir, pruner, model, cal, dcfg, split=args.split,
                               n_jobs=args.n_jobs)
    print(report)


if __name__ == "__main__":
    main()
