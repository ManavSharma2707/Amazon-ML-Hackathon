"""NB03 driver — frozen Qwen3-Embedding vectors + GPU dense kNN (architecture.md SS6.5).

Bundled with kaggle_env/io_utils/embed into `nb03_embed.py` (CLAUDE.md SS6.3).
Inputs (globbed under /kaggle/input): NB02 output (records_*.parquet,
gt.parquet, split_s1.parquet) and NB00 output (models/qwen3-emb-0.6b,
wheels/). Internet OFF.

Outputs in /kaggle/working (per split in {train, test}):
- emb_{split}_S1.npy, emb_{split}_pool.npy   float16 [n, dim], L2-normalised;
  pool = S2 rows then S3 rows, in records parquet order
- knn_{split}_idx.npy / knn_{split}_score.npy  S1 -> pool top-k (global pool rows)
- rev_{split}_idx.npy / rev_{split}_score.npy  pool -> S1 top-rk (global S1 rows)
- metrics.json  throughput, sanity cosines, dense recall@k on Half B (per country)

DRY_RUN (set by the dry-run kernel's bundle_spec "defines"): tiny slices, plus
an instruction vs no-instruction recall comparison on a mini retrieval task.
"""

import gc
import subprocess
import sys
import time

import numpy as np
import pandas as pd

DRY_RUN = globals().get("DRY_RUN", False)
WORK = kaggle_env.WORK_DIR
ECFG = CONFIG["embed"]
DIM = CONFIG["scale_guard"]["mrl_dim"] if CONFIG["scale_guard"]["enabled"] else 1024


def ensure_transformers() -> None:
    """Install transformers>=4.51 from NB00's offline wheels if the image's is older (Qwen3 support)."""
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    kaggle_env.log(f"transformers {transformers.__version__}")
    if (major, minor) >= (4, 51):
        return
    wheels = kaggle_env.find_input("wheels")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", str(wheels), "transformers>=4.51.0"],
        check=True,
    )
    raise SystemExit("transformers upgraded from wheels; re-run needed in a fresh process")


def load_split(recs_dir, split: str):
    """Load S1 and pool (S2 then S3) raw fields + country for one split."""
    cols = ["entity_id", "country", "raw_name", "raw_addr"]
    s1 = pd.read_parquet(recs_dir / f"records_{split}_S1.parquet", columns=cols)
    pool = pd.concat(
        [pd.read_parquet(recs_dir / f"records_{split}_S{s}.parquet", columns=cols) for s in (2, 3)], ignore_index=True
    )
    return s1, pool


def sanity_cosines(s1_emb, pool_emb, s1_ids, pool_ids, s1_cty, pool_cty, pairs, n=20000, seed=42) -> dict:
    """Mean/quantile cosine of true pairs vs random same-country pairs (bug detector)."""
    rng = np.random.default_rng(seed)
    s1_pos = pd.Series(np.arange(len(s1_ids)), index=s1_ids)
    p_pos = pd.Series(np.arange(len(pool_ids)), index=pool_ids)
    pr = pairs[pairs["s1_id"].isin(s1_pos.index) & pairs["match_id"].isin(p_pos.index)]
    pr = pr.sample(min(n, len(pr)), random_state=seed)
    a = s1_pos.loc[pr["s1_id"]].to_numpy()
    b = p_pos.loc[pr["match_id"]].to_numpy()
    pos = (s1_emb[a].astype(np.float32) * pool_emb[b].astype(np.float32)).sum(1)
    # Random pool row of the same country for each sampled S1.
    rb = np.empty_like(b)
    for c in np.unique(s1_cty[a]):
        m = s1_cty[a] == c
        cand = np.flatnonzero(pool_cty == c)
        rb[m] = rng.choice(cand, m.sum())
    neg = (s1_emb[a].astype(np.float32) * pool_emb[rb].astype(np.float32)).sum(1)
    q = lambda x: {k: round(float(np.quantile(x, v)), 4) for k, v in (("p05", 0.05), ("p50", 0.5), ("p95", 0.95))}
    return {
        "n": int(len(pos)),
        "pos_mean": round(float(pos.mean()), 4), "pos_q": q(pos),
        "rand_mean": round(float(neg.mean()), 4), "rand_q": q(neg),
        "share_pos_gt_rand": round(float((pos > neg).mean()), 4),
    }


def mini_retrieval(embedders, s1, pool, pairs, split_s1, instruction: str, n_q=3000, n_distract=60000) -> dict:
    """Recall@k of Half-B S1s against their matches + random distractors (dry run only)."""
    rng = np.random.default_rng(0)
    b_ids = split_s1.loc[split_s1["split"] == "B", "entity_id"]
    qids = rng.choice(b_ids.to_numpy(), min(n_q, len(b_ids)), replace=False)
    q = s1[s1["entity_id"].isin(set(qids))]
    m_ids = set(pairs.loc[pairs["s1_id"].isin(set(qids)), "match_id"])
    rest = pool[~pool["entity_id"].isin(m_ids)]
    p = pd.concat([pool[pool["entity_id"].isin(m_ids)], rest.sample(min(n_distract, len(rest)), random_state=0)])
    for e in embedders:
        e.instruction = instruction
    t0 = time.time()
    qe = embed.encode(embed.record_texts(q["raw_name"], q["raw_addr"]), embedders, ECFG["batch"], kaggle_env.log)
    pe = embed.encode(embed.record_texts(p["raw_name"], p["raw_addr"]), embedders, ECFG["batch"], kaggle_env.log)
    rate = (len(q) + len(p)) / (time.time() - t0)
    fi, _, _, _ = embed.knn_within_groups(
        qe, q["country"].to_numpy(), pe, p["country"].to_numpy(), k=30, rk=5,
        devices=[e.device for e in embedders], log_fn=kaggle_env.log,
    )
    rec = embed.pair_recall_at_k(fi, q["entity_id"].to_numpy(), p["entity_id"].to_numpy(), pairs)
    rec["encode_rate_per_s"] = round(rate, 1)
    return rec


def run_split(split: str, recs_dir, embedders, devices, metrics: dict, pairs=None, split_s1=None) -> None:
    """Encode S1 + pool of one split, run within-country kNN, save arrays, record metrics."""
    s1, pool = load_split(recs_dir, split)
    if DRY_RUN:
        s1, pool = s1.iloc[:5000], pool.iloc[:40000]
    m = metrics.setdefault(split, {"n_s1": int(len(s1)), "n_pool": int(len(pool))})
    for name, df in (("S1", s1), ("pool", pool)):
        kaggle_env.log(f"{split} {name}: encoding {len(df):,} records")
        t0 = time.time()
        e = embed.encode(embed.record_texts(df["raw_name"], df["raw_addr"]), embedders, ECFG["batch"], kaggle_env.log)
        dt = time.time() - t0
        m[f"encode_{name}_s"] = round(dt, 1)
        m[f"encode_{name}_rate"] = round(len(df) / max(dt, 1e-6), 1)
        assert np.isfinite(e.astype(np.float32)).all(), f"non-finite embeddings in {split} {name}"
        np.save(WORK / f"emb_{split}_{name}.npy", e)
        kaggle_env.log(f"{split} {name}: saved ({m[f'encode_{name}_rate']:.0f} rec/s)")
        if name == "S1":
            s1_emb = e
        else:
            pool_emb = e
    bcfg = CONFIG["blocking"]
    kaggle_env.log(f"{split}: dense kNN k={bcfg['dense_k']} reverse={bcfg['reverse_k']}")
    t0 = time.time()
    fi, fs, ri, rs = embed.knn_within_groups(
        s1_emb, s1["country"].to_numpy(), pool_emb, pool["country"].to_numpy(),
        k=bcfg["dense_k"], rk=bcfg["reverse_k"], devices=devices, log_fn=kaggle_env.log,
    )
    m["knn_s"] = round(time.time() - t0, 1)
    for nm, arr in (("knn_%s_idx", fi), ("knn_%s_score", fs), ("rev_%s_idx", ri), ("rev_%s_score", rs)):
        np.save(WORK / f"{nm % split}.npy", arr)
    if pairs is not None:
        s1_ids, p_ids = s1["entity_id"].to_numpy(), pool["entity_id"].to_numpy()
        m["sanity_cosine"] = sanity_cosines(
            s1_emb, pool_emb, s1_ids, p_ids, s1["country"].to_numpy(), pool["country"].to_numpy(), pairs
        )
        m["dense_recall_B"] = {}
        b = split_s1[split_s1["split"] == "B"]
        for c, g in [("ALL", b)] + list(b.groupby("country")):
            sel = s1["entity_id"].isin(set(g["entity_id"])).to_numpy()
            m["dense_recall_B"][c] = embed.pair_recall_at_k(fi[sel], s1_ids[sel], p_ids, pairs)
        kaggle_env.log(f"{split}: sanity {m['sanity_cosine']}; recall B {m['dense_recall_B']['ALL']}")
    del s1_emb, pool_emb, fi, fs, ri, rs
    gc.collect()


def main() -> None:
    """Run NB03 (or its dry run) and write metrics.json."""
    t0 = time.time()
    io_utils.set_seeds(CONFIG["seed"])
    ensure_transformers()
    import torch

    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    kaggle_env.log(f"devices: {devices}; dry_run={DRY_RUN}; dim={DIM}")
    model_dir = kaggle_env.find_input(f"{ECFG['model']}/config.json").parent
    recs_dir = kaggle_env.find_input("records_train_S1.parquet").parent
    kaggle_env.log(f"model: {model_dir}; records: {recs_dir}")
    embedders = [embed.Embedder(str(model_dir), d, dim=DIM, max_len=ECFG["max_len"], instruction=ECFG["instruction"]) for d in devices]
    pairs = pd.read_parquet(recs_dir / "gt.parquet")
    split_s1 = pd.read_parquet(recs_dir / "split_s1.parquet")
    metrics: dict = {"dry_run": DRY_RUN, "devices": devices, "dim": DIM, "max_len": ECFG["max_len"],
                     "instruction": ECFG["instruction"], "batch": ECFG["batch"]}

    metrics["param_dtype"] = embedders[0].param_dtype
    kaggle_env.log(f"model param dtype: {embedders[0].param_dtype}")
    if DRY_RUN:
        s1, pool = load_split(recs_dir, "train")
        # Throughput benchmark: batch size x instruction, with tokenise/forward split.
        texts = embed.record_texts(pool["raw_name"].iloc[:20000], pool["raw_addr"].iloc[:20000])
        metrics["bench"] = {}
        for bs in (128, 512):
            for label, instr in (("instr", CONFIG["embed"]["instruction"]), ("noinstr", "")):
                for e in embedders:
                    e.instruction = instr
                    e.timing = {"tok_s": 0.0, "fwd_s": 0.0, "tokens": 0}
                t0 = time.time()
                embed.encode(texts, embedders, bs)
                dt = time.time() - t0
                metrics["bench"][f"bs{bs}_{label}"] = {"rate": round(len(texts) / dt, 1),
                                                     "timing_gpu0": {k: round(v, 2) for k, v in embedders[0].timing.items()}}
                kaggle_env.log(f"bench bs={bs} {label}: {metrics['bench'][f'bs{bs}_{label}']}")
        metrics["instruction_ab"] = {}
        for label, instr in (("with_instruction", CONFIG["embed"]["instruction"]), ("no_instruction", "")):
            kaggle_env.log(f"mini retrieval: {label}")
            metrics["instruction_ab"][label] = mini_retrieval(embedders, s1, pool, pairs, split_s1, instr)
            kaggle_env.log(f"  {metrics['instruction_ab'][label]}")
        for e in embedders:
            e.instruction = ECFG["instruction"]
        del s1, pool
        gc.collect()

    run_split("train", recs_dir, embedders, devices, metrics, pairs, split_s1)
    kaggle_env.write_json(metrics, WORK / "metrics.json")  # partial checkpoint of the report
    run_split("test", recs_dir, embedders, devices, metrics)
    metrics["runtime_s"] = round(time.time() - t0, 1)
    kaggle_env.write_json(metrics, WORK / "metrics.json")
    kaggle_env.log(f"done in {metrics['runtime_s']}s")


if __name__ == "__main__":
    main()
