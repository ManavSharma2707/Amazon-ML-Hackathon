"""NB04 driver — fine-tune the embedder on Half A + acceptance test (Gate G2), architecture.md SS6.6.

Bundled with kaggle_env/io_utils/embed/scramble/finetune_embedder into
`nb04_finetune.py` (CLAUDE.md SS6.3). GPU (T4), internet OFF.

Inputs (globbed): NB02 output (records, gt, split_s1), NB03 output (frozen
emb_train_*.npy, knn_train_idx.npy), NB00 output (models/qwen3-emb-0.6b).

Modes (bundle_spec "defines"):
- default: build Half A triplets, LoRA-train, save merged model to
  ft_model/, then the acceptance test on a reduced but identical retrieval
  set for frozen and fine-tuned vectors (20k Half-B queries vs their matches
  + 500k distractors; plus a scrambled-letter variant) -> acceptance.json
  with the G2 verdict.
- DRY_RUN: tiny data, ~20 steps, tiny acceptance set (catches env errors).
- FULL_ENCODE (kernel er-nb04b-ft-encode, only after G2 passes): load
  ft_model/ from the NB04 output and write ft_emb_* / ft_knn_* / ft_rev_*
  for train (Half-B S1 rows only; other rows zero / -1) and test, for NB05 v2.
"""

import gc
import time

import numpy as np
import pandas as pd

DRY_RUN = globals().get("DRY_RUN", False)
FULL_ENCODE = globals().get("FULL_ENCODE", False)
WORK = kaggle_env.WORK_DIR
FT = CONFIG["finetune"]
ECFG = CONFIG["embed"]
DIM = CONFIG["scale_guard"]["mrl_dim"] if CONFIG["scale_guard"]["enabled"] else 1024
COLS = ["entity_id", "country", "raw_name", "raw_addr"]


def load(recs_dir, split):
    """S1 and pool frames with a `text` column (embed.record_texts)."""
    s1 = pd.read_parquet(recs_dir / f"records_{split}_S1.parquet", columns=COLS)
    pool = pd.concat([pd.read_parquet(recs_dir / f"records_{split}_S{s}.parquet", columns=COLS) for s in (2, 3)], ignore_index=True)
    for d in (s1, pool):
        d["text"] = embed.record_texts(d["raw_name"], d["raw_addr"])
        d.drop(columns=["raw_name", "raw_addr"], inplace=True)
    return s1, pool


def devices():
    """Visible CUDA devices."""
    import torch

    return [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]


def embedders_for(model_dir):
    """One Embedder per GPU for a model folder (no instruction: matches NB03's setting)."""
    return [embed.Embedder(str(model_dir), d, dim=DIM, max_len=ECFG["max_len"], instruction=ECFG["instruction"]) for d in devices()]


def retrieval(q, p, q_vec, p_vec, pairs, k=50) -> dict:
    """Within-country top-k recall/MRR, overall and per country."""
    fi, _, _, _ = embed.knn_within_groups(q_vec, q["country"].to_numpy(), p_vec, p["country"].to_numpy(),
                                          k=k, rk=1, devices=devices(), log_fn=kaggle_env.log)
    out = {"ALL": finetune_embedder.recall_mrr(fi, q["entity_id"].to_numpy(), p["entity_id"].to_numpy(), pairs)}
    for c in sorted(q["country"].unique()):
        m = (q["country"] == c).to_numpy()
        out[c] = finetune_embedder.recall_mrr(fi[m], q["entity_id"].to_numpy()[m], p["entity_id"].to_numpy(), pairs)
    return out


def acceptance(s1, pool, pairs, split_s1, frozen_dir, ft_dir) -> dict:
    """Frozen vs fine-tuned recall@k on the same reduced Half-B retrieval set (+ scrambled)."""
    rng = np.random.default_rng(CONFIG["seed"])
    n_q, n_d = (300, 5000) if DRY_RUN else (20000, 500000)
    b_rows = np.flatnonzero(s1["entity_id"].isin(set(split_s1.loc[split_s1["split"] == "B", "entity_id"])).to_numpy())
    qi = np.sort(rng.choice(b_rows, min(n_q, len(b_rows)), replace=False))
    q = s1.iloc[qi]
    m_ids = set(pairs.loc[pairs["s1_id"].isin(set(q["entity_id"])), "match_id"])
    in_m = pool["entity_id"].isin(m_ids).to_numpy()
    others = np.flatnonzero(~in_m & pool["country"].isin(set(q["country"])).to_numpy())
    pi = np.sort(np.concatenate([np.flatnonzero(in_m), rng.choice(others, min(n_d, len(others)), replace=False)]))
    p = pool.iloc[pi]
    fro_q = np.load(frozen_dir / "emb_train_S1.npy", mmap_mode="r")[qi]
    fro_p = np.load(frozen_dir / "emb_train_pool.npy", mmap_mode="r")[pi]
    res = {"n_queries": int(len(q)), "n_pool": int(len(p))}
    kaggle_env.log("acceptance: frozen")
    res["frozen"] = retrieval(q, p, np.asarray(fro_q), np.asarray(fro_p), pairs)
    ft_emb = embedders_for(ft_dir)
    kaggle_env.log("acceptance: fine-tuned")
    res["finetuned"] = retrieval(q, p, embed.encode(q["text"].tolist(), ft_emb, ECFG["batch"], kaggle_env.log),
                                 embed.encode(p["text"].tolist(), ft_emb, ECFG["batch"], kaggle_env.log), pairs)
    # Scrambled variant on a smaller slice: same permutation for queries and pool.
    table = scramble.make_scrambler(CONFIG["seed"])
    qs = q.iloc[: len(q) // 4]
    ms = set(pairs.loc[pairs["s1_id"].isin(set(qs["entity_id"])), "match_id"])
    ps = pd.concat([p[p["entity_id"].isin(ms)], p[~p["entity_id"].isin(ms)].iloc[: len(p) // 5]])
    qt, pt = scramble.scramble_texts(qs["text"], table), scramble.scramble_texts(ps["text"], table)
    res["finetuned_scrambled"] = retrieval(qs, ps, embed.encode(qt, ft_emb, ECFG["batch"]), embed.encode(pt, ft_emb, ECFG["batch"]), pairs)
    res["finetuned_clean_small"] = retrieval(qs, ps, embed.encode(qs["text"].tolist(), ft_emb, ECFG["batch"]),
                                             embed.encode(ps["text"].tolist(), ft_emb, ECFG["batch"]), pairs)
    del ft_emb
    gc.collect()
    fro_emb = embedders_for(kaggle_env.find_input(f"{ECFG['model']}/config.json").parent)
    kaggle_env.log("acceptance: frozen scrambled")
    res["frozen_scrambled"] = retrieval(qs, ps, embed.encode(qt, fro_emb, ECFG["batch"]), embed.encode(pt, fro_emb, ECFG["batch"]), pairs)
    res["frozen_clean_small"] = retrieval(qs, ps, embed.encode(qs["text"].tolist(), fro_emb, ECFG["batch"]),
                                          embed.encode(ps["text"].tolist(), fro_emb, ECFG["batch"]), pairs)
    del fro_emb
    gc.collect()
    # Gate G2 (plan SS11.5): recall@20 up in every country, scrambled drop not bigger than frozen's.
    countries = [c for c in res["frozen"] if c != "ALL"]
    better = {c: res["finetuned"][c]["recall@20"] > res["frozen"][c]["recall@20"] for c in countries}
    drop_ft = res["finetuned_clean_small"]["ALL"]["recall@20"] - res["finetuned_scrambled"]["ALL"]["recall@20"]
    drop_fr = res["frozen_clean_small"]["ALL"]["recall@20"] - res["frozen_scrambled"]["ALL"]["recall@20"]
    res["G2"] = {"recall20_better_per_country": better, "scrambled_drop_ft": round(drop_ft, 5),
                 "scrambled_drop_frozen": round(drop_fr, 5), "pass": bool(all(better.values()) and drop_ft <= drop_fr + 0.002)}
    return res


def train_mode(recs_dir, frozen_dir, model_dir) -> dict:
    """Build triplets on Half A, train, save the merged model, run the acceptance test."""
    s1, pool = load(recs_dir, "train")
    pairs = pd.read_parquet(recs_dir / "gt.parquet")
    split_s1 = pd.read_parquet(recs_dir / "split_s1.parquet")
    owner = pairs.set_index("match_id")["s1_id"]
    pool["owner"] = pool["entity_id"].map(owner)
    rng = np.random.default_rng(CONFIG["seed"])
    a_ids = split_s1.loc[(split_s1["split"] == "A") & (split_s1["n_matches"] > 0), "entity_id"]
    n_anchor = 300 if DRY_RUN else FT.get("max_anchors", 80000)
    a_ids = set(rng.choice(a_ids.to_numpy(), min(n_anchor, len(a_ids)), replace=False).tolist())
    a_rows = np.flatnonzero(s1["entity_id"].isin(a_ids).to_numpy())
    knn = np.asarray(np.load(frozen_dir / "knn_train_idx.npy", mmap_mode="r")[a_rows])
    trip = finetune_embedder.build_triplets(s1.iloc[a_rows].reset_index(drop=True), pairs, pool, knn,
                                            hard_negs=FT["hard_negs"], seed=CONFIG["seed"])
    kaggle_env.log(f"triplets: {len(trip):,} from {len(a_rows):,} Half-A anchors")
    budget = 120 if DRY_RUN else FT.get("time_budget_min", 90) * 60
    cfg = dict(FT, batch=8 if DRY_RUN else FT["batch"])
    merged, tok, log = finetune_embedder.train_lora(str(model_dir), trip, WORK, cfg, DIM, ECFG["max_len"], devices()[0],
                                                    budget, log_fn=kaggle_env.log, seed=CONFIG["seed"])
    ft_dir = WORK / "ft_model"
    merged.save_pretrained(str(ft_dir))
    tok.save_pretrained(str(ft_dir))
    del merged
    import torch

    torch.cuda.empty_cache()
    log["n_triplets"] = int(len(trip))
    kaggle_env.write_json(log, WORK / "train_log.json")
    acc = acceptance(s1, pool, pairs, split_s1, frozen_dir, ft_dir)
    kaggle_env.write_json(acc, WORK / "acceptance.json")
    return {"train": {k: v for k, v in log.items() if k != "loss_curve"}, "acceptance_G2": acc["G2"],
            "recall20": {m: {c: acc[m][c]["recall@20"] for c in acc[m]} for m in ("frozen", "finetuned")}}


def full_encode_mode(recs_dir) -> dict:
    """Encode Half-B train S1 rows + train pool + test with ft_model/, kNN, save ft_* arrays."""
    ft_dir = kaggle_env.find_input("ft_model/config.json").parent
    emb = embedders_for(ft_dir)
    bc = CONFIG["blocking"]
    split_s1 = pd.read_parquet(recs_dir / "split_s1.parquet")
    out = {}
    for split in ("train", "test"):
        s1, pool = load(recs_dir, split)
        if DRY_RUN:
            s1, pool = s1.iloc[:3000], pool.iloc[:20000]
        rows = np.arange(len(s1))
        if split == "train":
            rows = np.flatnonzero(s1["entity_id"].isin(set(split_s1.loc[split_s1["split"] == "B", "entity_id"])).to_numpy())
        q = np.zeros((len(s1), DIM), dtype=np.float16)
        q[rows] = embed.encode(s1["text"].to_numpy()[rows].tolist(), emb, ECFG["batch"], kaggle_env.log)
        p = embed.encode(pool["text"].tolist(), emb, ECFG["batch"], kaggle_env.log)
        np.save(WORK / f"ft_emb_{split}_S1.npy", q)
        np.save(WORK / f"ft_emb_{split}_pool.npy", p)
        fi_r, fs_r, ri, rs = embed.knn_within_groups(q[rows], s1["country"].to_numpy()[rows], p, pool["country"].to_numpy(),
                                                     k=bc["dense_k"], rk=bc["reverse_k"], devices=devices(), log_fn=kaggle_env.log)
        fi = np.full((len(s1), bc["dense_k"]), -1, dtype=np.int32)
        fs = np.zeros((len(s1), bc["dense_k"]), dtype=np.float16)
        fi[rows], fs[rows] = fi_r, fs_r
        ri = np.where(ri >= 0, rows[np.clip(ri, 0, None)], -1).astype(np.int32)  # local query rows -> global S1 rows
        for nm, arr in (("ft_knn_%s_idx", fi), ("ft_knn_%s_score", fs), ("ft_rev_%s_idx", ri), ("ft_rev_%s_score", rs)):
            np.save(WORK / f"{nm % split}.npy", arr)
        out[split] = {"n_s1_encoded": int(len(rows)), "n_pool": int(len(pool))}
        del s1, pool, q, p
        gc.collect()
    return out


def main() -> None:
    """Run NB04 (train + acceptance) or NB04b (full encode)."""
    t0 = time.time()
    io_utils.set_seeds(CONFIG["seed"])
    WORK.mkdir(parents=True, exist_ok=True)
    recs_dir = kaggle_env.find_input("records_train_S1.parquet").parent
    kaggle_env.log(f"devices {devices()}; dry_run={DRY_RUN}; full_encode={FULL_ENCODE}")
    if FULL_ENCODE:
        metrics = {"mode": "full_encode", **full_encode_mode(recs_dir)}
    else:
        frozen_dir = kaggle_env.find_input("knn_train_idx.npy").parent
        model_dir = kaggle_env.find_input(f"{ECFG['model']}/config.json").parent
        metrics = {"mode": "train", **train_mode(recs_dir, frozen_dir, model_dir)}
    metrics["dry_run"] = DRY_RUN
    metrics["runtime_s"] = round(time.time() - t0, 1)
    kaggle_env.write_json(metrics, WORK / "metrics.json")
    kaggle_env.log(f"done in {metrics['runtime_s']}s")


if __name__ == "__main__":
    main()
