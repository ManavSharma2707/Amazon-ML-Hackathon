"""NB08 driver — judge inference on the B and test uncertain bands (runner R2, GPU T4 x2; plan SS16.5).

Bundled with kaggle_env/io_utils/judge_train/judge_infer into `nb08_judge_infer.py`.
Inputs: er-judge-inputs (judge_inputs_B/test.parquet), NB07 output (adapter_final/),
runner R2's NB00 output (models/qwen3-4b/, wheels/). Internet OFF.
Outputs in /kaggle/working: judge_B.parquet, judge_test.parquet (s1_id, cand_id, p1, judge_p;
scored pairs only), metrics.json (pairs/s, depth reached, B band AUC judge vs stage-1).
"""

import os
import time

import numpy as np
import pandas as pd

WORK = kaggle_env.WORK_DIR
JC = CONFIG["judge"]
BUDGET_MIN = float(os.environ.get("ER_JUDGE_INFER_MIN", JC.get("infer_budget_min", 110)))
BATCH = int(os.environ.get("ER_JUDGE_BATCH", 16))


def main() -> None:
    """Run NB08 end to end."""
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    kaggle_env.ensure_packages(["bitsandbytes", "peft", "accelerate"])
    import torch
    from sklearn.metrics import roc_auc_score

    n_gpu = torch.cuda.device_count()
    model_dir = kaggle_env.find_input("qwen3-4b/config.json").parent
    adapter = kaggle_env.find_input("adapter_final/adapter_config.json").parent
    frames = {n: io_utils.read_parquet_compact(kaggle_env.find_input(f"judge_inputs_{n}.parquet")) for n in ("B", "test")}
    kaggle_env.log(f"gpus {n_gpu}; model {model_dir}; adapter {adapter}; B {len(frames['B']):,}, test {len(frames['test']):,}")
    q = judge_infer.merged_queue(frames)
    dcap = judge_infer.cap_depth(frames)
    q = q[q["depth"] <= dcap].reset_index(drop=True)
    models = []
    for d in range(max(1, n_gpu)):
        tok, m = judge_infer.load(model_dir, adapter, d)
        models.append((tok, m, f"cuda:{d}"))
    tok = models[0][0]
    texts = {n: judge_train.encode(tok, f["system"].tolist(), f["user"].tolist(), JC["max_len"]) for n, f in frames.items()}
    ids = [texts[s][r] for s, r in zip(q["split"], q["row"])]
    kaggle_env.log(f"queue {len(q):,} pairs (depth cap {dcap:.3f}); load+tokenise {time.time() - t0:.0f}s; budget {BUDGET_MIN} min")
    t1 = time.time()
    p = judge_infer.score_budgeted(models, ids, BUDGET_MIN * 60, batch=BATCH, log=kaggle_env.log)
    secs = time.time() - t1
    miss = np.flatnonzero(np.isnan(p))
    d_stop = float(q["depth"].iloc[miss[0]]) if len(miss) else float("inf")
    p[q["depth"].to_numpy() >= d_stop] = np.nan  # same depth on B and test
    q["judge_p"] = p
    metrics = {"n_gpu": n_gpu, "budget_min": BUDGET_MIN, "depth_cap": dcap, "depth_stop": d_stop,
               "pairs_per_s": round(float(np.isfinite(p).sum() / max(secs, 1e-9)), 2), "score_s": round(secs, 1)}
    for n, f in frames.items():
        sc = q[(q["split"] == n) & q["judge_p"].notna()]
        out = f.iloc[sc["row"].to_numpy()][["s1_id", "cand_id", "p1"]].reset_index(drop=True)
        out["judge_p"] = sc["judge_p"].to_numpy()
        out.to_parquet(WORK / f"judge_{n}.parquet", index=False)
        metrics[n] = {"n_band_file": int(len(f)), "n_scored": int(len(out))}
        if n == "B" and len(out):
            y = f.iloc[sc["row"].to_numpy()]["label"].to_numpy()
            if len(np.unique(y)) == 2:
                metrics["B"]["auc_judge"] = round(float(roc_auc_score(y, out["judge_p"])), 5)
                metrics["B"]["auc_stage1"] = round(float(roc_auc_score(y, out["p1"])), 5)
                metrics["B"]["pos_rate"] = round(float(y.mean()), 4)
    metrics["runtime_s"] = round(time.time() - t0, 1)
    kaggle_env.write_json(metrics, WORK / "metrics.json")
    kaggle_env.log(f"{metrics}")


if __name__ == "__main__":
    main()
