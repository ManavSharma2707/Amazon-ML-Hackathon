"""NB02 driver — normalise + corpus statistics + A/B split (architecture.md SS6.4).

Hand-written entry point, bundled with the src modules listed in
bundle_spec.json into the self-contained `nb02_normalize.py` (CLAUDE.md SS6.3).
References bundled modules by bare name (`normalize.normalize_df`, ...).
`CONFIG` is the inlined src/configs/default.yaml.

Inputs: er-data (found by globbing /kaggle/input). Outputs in /kaggle/working:
- records_{train,test}_S{1,2,3}.parquet  normalised records (NB02 schema)
- stats_{train,test}.parquet             token statistics (corpus_stats)
- gt.parquet                             true pairs (s1_id, match_id)
- split_s1.parquet                       entity_id, country, n_matches, bucket, split (A/B)
- split_pool.parquet                     train S2/S3 entity_id, split (A/B/shared)
- metrics.json, manifest.json            small reports fetched to the laptop

Local smoke run: ER_INPUT_ROOT=<repo>/sample ER_WORK_DIR=<tmp> python nb02_normalize.py
"""

import gc
import os
import time

import pandas as pd

WORK = kaggle_env.WORK_DIR
N_JOBS = os.cpu_count() or 1


def _source_metrics(norm: pd.DataFrame) -> dict:
    """Per-country quality rates of one normalised source frame."""
    g = norm.assign(
        has_house=norm["house_number"].ne(""),
        has_postcode=norm["postcodes"].ne(""),
        has_landmark=norm["landmark"].ne(""),
        name_len=norm["norm_name"].str.count(" ") + norm["norm_name"].ne("").astype(int),
    ).groupby("country")
    rates = g[["name_empty", "addr_empty", "name_romanized", "has_house", "has_postcode", "has_landmark"]].mean()
    rates["n"] = g.size()
    rates["mean_name_tokens"] = g["name_len"].mean()
    return {c: {k: round(float(v), 5) for k, v in row.items()} for c, row in rates.iterrows()}


def process_split(data_dir, split: str, metrics: dict) -> tuple[pd.DataFrame | None, pd.Series | None]:
    """Normalise S1/S2/S3 of one split, write parquet + stats, collect metrics.

    Inputs: data_dir - folder holding train/ and test/; split - "train"/"test";
            metrics - dict updated in place.
    Outputs: (S1 entity_id/country frame, S2+S3 entity_id Series) for the
    train split (needed by the A/B split), else (None, None).
    """
    counter = corpus_stats.TokenCounter()
    s1_frame, pool_ids = None, []
    metrics[split] = {}
    for s in (1, 2, 3):
        path = data_dir / split / f"{split}_source{s}.tsv"
        kaggle_env.log(f"{split} S{s}: loading {path.name}")
        raw = io_utils.load_tsv(path)
        io_utils.assert_source_columns(raw, path)
        io_utils.assert_unique_ids(raw, f"S{s}-", path)
        kaggle_env.log(f"{split} S{s}: {len(raw):,} rows loaded; normalising with {N_JOBS} workers")
        norm = normalize.normalize_df(raw, f"S{s}", n_jobs=N_JOBS)
        assert len(norm) == len(raw)
        out = WORK / f"records_{split}_S{s}.parquet"
        norm.to_parquet(out, index=False, compression="zstd")
        kaggle_env.log(f"{split} S{s}: wrote {out.name}; counting tokens")
        metrics[split][f"S{s}"] = _source_metrics(norm)
        counter.add(norm)
        if s == 1:
            s1_frame = norm[["entity_id", "country"]].copy()
        else:
            pool_ids.append(norm["entity_id"])
        del raw, norm
        gc.collect()
    kaggle_env.log(f"{split}: finalising token stats")
    stats = counter.finalize()
    stats.to_parquet(WORK / f"stats_{split}.parquet", index=False, compression="zstd")
    metrics[split]["stats"] = {
        "n_rows": int(len(stats)),
        "top_suffix_like": [t for t, _ in corpus_stats.top_tokens(stats, "norm_name", 25, by="suffix")],
        "top_street_type_like": [t for t, _ in corpus_stats.top_tokens(stats, "norm_addr", 25, by="street")],
        "top_name_df": [t for t, _ in corpus_stats.top_tokens(stats, "norm_name", 25)],
    }
    del stats, counter
    gc.collect()
    if split == "train":
        return s1_frame, pd.concat(pool_ids, ignore_index=True)
    return None, None


def main() -> None:
    """Run NB02 end to end and write metrics.json."""
    t0 = time.time()
    io_utils.set_seeds(CONFIG["seed"])
    WORK.mkdir(parents=True, exist_ok=True)
    data_dir = kaggle_env.find_input_dir("train/train_source1.tsv")
    kaggle_env.log(f"data dir: {data_dir}; workers: {N_JOBS}")
    metrics: dict = {"config": {"seed": CONFIG["seed"], "split": CONFIG["split"], "stats_mode": CONFIG["stats_mode"]}}

    s1_frame, pool_ids = process_split(data_dir, "train", metrics)

    kaggle_env.log("ground truth + A/B split")
    gt = io_utils.load_ground_truth(data_dir / "train" / "train_ground_truth.tsv")
    assert set(gt) == set(s1_frame["entity_id"]), "GT S1 set != train S1 set"
    pairs = split.gt_long(gt)
    pairs.to_parquet(WORK / "gt.parquet", index=False, compression="zstd")
    s1_split = split.make_ab_split(s1_frame, gt, a_frac=CONFIG["split"]["a_frac"], seed=CONFIG["seed"])
    s1_split.to_parquet(WORK / "split_s1.parquet", index=False, compression="zstd")
    pool = split.pool_split(s1_split, pairs, pool_ids)
    pool.to_parquet(WORK / "split_pool.parquet", index=False, compression="zstd")
    metrics["split"] = split.split_report(s1_split)
    metrics["split"]["pool"] = {k: int(v) for k, v in pool["split"].value_counts().items()}
    metrics["n_true_pairs"] = int(len(pairs))
    del gt, pairs, s1_split, pool, s1_frame, pool_ids
    gc.collect()

    process_split(data_dir, "test", metrics)

    metrics["runtime_s"] = round(time.time() - t0, 1)
    kaggle_env.write_json(metrics, WORK / "metrics.json")
    kaggle_env.write_json(
        {"producing_notebook": "nb02_normalize", "config_seed": CONFIG["seed"], "created_utc": time.time()},
        WORK / "manifest.json",
    )
    kaggle_env.log(f"done in {metrics['runtime_s']}s")


if __name__ == "__main__":
    main()
