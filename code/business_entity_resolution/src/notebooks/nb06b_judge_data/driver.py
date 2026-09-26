"""NB06b driver — judge inputs (plan SS16.4-16.5) from stage-1 predictions.

Bundled with kaggle_env/io_utils/normalize/corpus_stats/blocking/explain_diff/
features/scramble/judge_data into `nb06b_judge_data.py`. CPU kernel, internet OFF.

Inputs: NB02 output (records_*, stats_*), NB06 output (p1_A_oof, p1_B, p1_test),
NB00 output (wheels/, rapidfuzz).
Outputs in /kaggle/working (relayed to runner R2 as the private dataset
er-judge-inputs; must stay < 300 MB in total):
- judge_inputs_A.parquet     training prompts (A OOF band + 10% confident, 1:2, augmented)
- judge_inputs_B.parquet     B band pairs [0.10, 0.90], most uncertain first, capped
- judge_inputs_test.parquet  test band pairs, same rule
- metrics.json
"""

import os
import time

import numpy as np
import pandas as pd

WORK = kaggle_env.WORK_DIR
JC = CONFIG["judge"]
SEED = CONFIG["seed"]
MAX_TRAIN = int(os.environ.get("ER_JUDGE_TRAIN_N", JC.get("train_n", 20_000)))
CAP_B = int(os.environ.get("ER_JUDGE_CAP_B", JC.get("infer_cap_B", 60_000)))
CAP_TEST = int(os.environ.get("ER_JUDGE_CAP_TEST", JC.get("infer_cap_test", 200_000)))


def records_for(recs_dir, split: str, s1_ids: set, pool_ids: set):
    """Only the S1 / pool records referenced by the selected pairs (raw + normalised fields)."""
    cols = ["entity_id"] + judge_data.REC_FIELDS
    s1 = io_utils.read_parquet_compact(recs_dir / f"records_{split}_S1.parquet", cols)
    s1 = s1[s1["entity_id"].isin(s1_ids)]
    pool = pd.concat([io_utils.read_parquet_compact(recs_dir / f"records_{split}_S{s}.parquet", cols) for s in (2, 3)],
                     ignore_index=True)
    pool = pool[pool["entity_id"].isin(pool_ids)]
    conv = lambda d: d.astype({c: object for c in d.columns if c != "name_romanized"})  # plain str for .at lookups
    return conv(s1).reset_index(drop=True), conv(pool).reset_index(drop=True)


def build(recs_dir, split, pairs, stats_name, augment, name, metrics):
    """Prompts for one split -> judge_inputs_<name>.parquet; size and label stats into metrics."""
    t0 = time.time()
    s1, pool = records_for(recs_dir, split, set(pairs["s1_id"]), set(pairs["cand_id"]))
    lk = features.token_lookups(pd.read_parquet(recs_dir / stats_name))
    ds = judge_data.build_dataset(pairs, s1, pool, lk, augment=augment, seed=SEED, log=kaggle_env.log)
    path = WORK / f"judge_inputs_{name}.parquet"
    ds.to_parquet(path, index=False, compression="zstd")
    m = {"n": int(len(ds)), "mb": round(path.stat().st_size / 2**20, 1),
         "mean_user_chars": round(float(ds["user"].str.len().mean()), 1), "sec": round(time.time() - t0, 1)}
    if "label" in ds:
        m["pos_rate"] = round(float(ds["label"].mean()), 4)
    if "p1" in ds:
        m["p1_range"] = [round(float(ds["p1"].min()), 4), round(float(ds["p1"].max()), 4)]
    metrics[name] = m
    kaggle_env.log(f"  {name}: {m}")
    if name == "A":
        metrics["example_prompt"] = ds["system"].iloc[0] + "\n\n" + ds["user"].iloc[0]


def main() -> None:
    """Run NB06b end to end."""
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    kaggle_env.ensure_rapidfuzz()
    recs_dir = kaggle_env.find_input("records_train_S1.parquet").parent
    s1_dir = kaggle_env.find_input("p1_test.parquet").parent
    metrics: dict = {"train_band": JC["train_band"], "infer_band": JC["infer_band"], "max_train": MAX_TRAIN,
                     "cap_B": CAP_B, "cap_test": CAP_TEST}

    oof = io_utils.read_parquet_compact(s1_dir / "p1_A_oof.parquet")
    lo, hi = JC["train_band"]
    metrics["A_band_size"] = int(((oof["p1"] >= lo) & (oof["p1"] <= hi)).sum())
    tr = judge_data.select_train_band(oof, lo, hi, max_n=MAX_TRAIN, seed=SEED)
    kaggle_env.log(f"A: {metrics['A_band_size']:,} OOF pairs in band; {len(tr):,} selected")
    build(recs_dir, "train", tr, "stats_train.parquet", True, "A", metrics)

    lo, hi = JC["infer_band"]
    for name, split, fname, cap, stats in (("B", "train", "p1_B.parquet", CAP_B, "stats_train.parquet"),
                                           ("test", "test", "p1_test.parquet", CAP_TEST, "stats_test.parquet")):
        p1 = io_utils.read_parquet_compact(s1_dir / fname, ["s1_id", "cand_id", "p1"] + (["label"] if name == "B" else []))
        band = judge_data.select_infer_band(p1, lo, hi)
        metrics[f"{name}_band_size"] = int(len(band))
        kaggle_env.log(f"{name}: {len(band):,} pairs in band; keeping {min(cap, len(band)):,}")
        build(recs_dir, split, band.iloc[:cap], stats, False, name, metrics)
    metrics["total_mb"] = round(sum(metrics[k]["mb"] for k in ("A", "B", "test")), 1)
    assert metrics["total_mb"] < 300, "judge inputs must stay < 300 MB for the cross-runner relay"
    metrics["runtime_s"] = round(time.time() - t0, 1)
    kaggle_env.write_json(metrics, WORK / "metrics.json")
    kaggle_env.log(f"done in {metrics['runtime_s']}s")


if __name__ == "__main__":
    main()
