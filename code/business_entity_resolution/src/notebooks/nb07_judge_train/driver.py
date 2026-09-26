"""NB07 driver — QLoRA training of the Qwen3-4B judge on Half-A prompts (runner R2, GPU T4).

Bundled with kaggle_env/io_utils/judge_train into `nb07_judge_train.py`; the
dry-run kernel (`er-nb07-judge-train-dry`) is the same driver with DRY = True
(64 examples, a few steps) and always runs first (CLAUDE.md SS5 GPU rules).

Inputs (globbed under /kaggle/input):
- er-judge-inputs (private dataset relayed from runner R1's NB06b): judge_inputs_A.parquet
- runner R2's own NB00 output: models/qwen3-4b/, wheels/ (bitsandbytes, peft, accelerate)
Outputs in /kaggle/working: adapter_final/, ckpt-*/ (every 200 steps, last two kept),
val_preds.parquet, metrics.json (throughput, loss curve, held-out AUC / logloss
vs stage-1 on the same pairs). Internet OFF.
"""

import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

WORK = kaggle_env.WORK_DIR
JC = CONFIG["judge"]
SEED = CONFIG["seed"]
DRY = globals().get("DRY", False)
BUDGET_MIN = float(os.environ.get("ER_JUDGE_BUDGET_MIN", 4 if DRY else JC.get("train_budget_min", 150)))
HOLDOUT = 0.05


def install_wheels() -> None:
    """Offline install of the QLoRA stack from NB00's wheels (only what the image lacks)."""
    import importlib.util

    need = [m for m in ("bitsandbytes", "peft", "accelerate") if importlib.util.find_spec(m) is None]
    if need:
        wheels = kaggle_env.find_input("wheels")
        kaggle_env.log(f"installing {need} from {wheels}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", str(wheels), *need],
                       check=True)


def auc_ll(y, p) -> dict:
    """AUC and log-loss (None when only one class is present)."""
    from sklearn.metrics import log_loss, roc_auc_score

    if len(np.unique(y)) < 2:
        return {"auc": None, "logloss": None}
    return {"auc": round(float(roc_auc_score(y, p)), 5),
            "logloss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])), 5)}


def main() -> None:
    """Run NB07 (or its dry run) end to end."""
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    install_wheels()
    import torch

    kaggle_env.log(f"torch {torch.__version__}, cuda {torch.cuda.is_available()}, gpus {torch.cuda.device_count()}, dry {DRY}")
    data = io_utils.read_parquet_compact(kaggle_env.find_input("judge_inputs_A.parquet"))
    model_dir = kaggle_env.find_input("qwen3-4b/config.json").parent
    # held-out slice by S1 group (no S1 in both parts)
    g = pd.util.hash_array(data["s1_id"].to_numpy(dtype=object)) % 1000
    val = data[g < HOLDOUT * 1000].reset_index(drop=True)
    tr = data[g >= HOLDOUT * 1000].reset_index(drop=True)
    if DRY:
        tr, val = tr.iloc[:64], val.iloc[:32]
    metrics: dict = {"dry": DRY, "n_train": int(len(tr)), "n_val": int(len(val)), "budget_min": BUDGET_MIN,
                     "config": {k: JC[k] for k in ("lora_r", "lora_alpha", "max_len", "lr")}}
    kaggle_env.log(f"train {len(tr):,} (pos {tr['label'].mean():.3f}), val {len(val):,}; model {model_dir}")

    tok, model = judge_train.load_model(model_dir, JC["lora_r"], JC["lora_alpha"])
    y_id, n_id = judge_train.answer_ids(tok)
    metrics["answer_tokens"] = {"Yes": tok.convert_ids_to_tokens(y_id), "No": tok.convert_ids_to_tokens(n_id),
                                "single_token": len(tok.encode("Yes", add_special_tokens=False)) == 1
                                and len(tok.encode("No", add_special_tokens=False)) == 1}
    ids_tr = judge_train.encode(tok, tr["system"].tolist(), tr["user"].tolist(), JC["max_len"])
    ids_val = judge_train.encode(tok, val["system"].tolist(), val["user"].tolist(), JC["max_len"])
    lens = np.array([len(x) for x in ids_tr])
    metrics["prompt_tokens"] = {"mean": round(float(lens.mean()), 1), "p95": float(np.quantile(lens, 0.95)),
                                "max": int(lens.max()), "share_truncated": round(float((lens >= JC["max_len"]).mean()), 4)}
    metrics["example_prompt"] = judge_train.chat_prompt(tok, tr["system"].iloc[0], tr["user"].iloc[0])
    kaggle_env.log(f"prompt tokens {metrics['prompt_tokens']}; load {time.time() - t0:.0f}s")
    kaggle_env.write_json(metrics, WORK / "metrics.json")

    y_val = val["label"].to_numpy()
    n0 = min(len(ids_val), 256)
    metrics["val_before_training"] = auc_ll(y_val[:n0], judge_train.predict(model, tok, ids_val[:n0]))
    kaggle_env.log(f"before training: {metrics['val_before_training']}")
    log = judge_train.train(model, tok, ids_tr, tr["label"].to_numpy(), WORK, lr=JC["lr"],
                            micro=2 if DRY else 4, accum=2 if DRY else 8, budget_s=BUDGET_MIN * 60,
                            measure_steps=5 if DRY else 50, save_every=10 if DRY else 200, seed=SEED, log=kaggle_env.log)
    metrics["train"] = log
    kaggle_env.write_json(metrics, WORK / "metrics.json")

    t1 = time.time()
    p = judge_train.predict(model, tok, ids_val)
    metrics["infer_pairs_per_s_1gpu"] = round(len(ids_val) / max(time.time() - t1, 1e-9), 2)
    metrics["val"] = {"judge": auc_ll(y_val, p), "stage1": auc_ll(y_val, val["p1"].to_numpy())}
    if "band" in val:  # G4 preview: judge vs stage-1 on the uncertain band only
        b = (val["band"] == 1).to_numpy()
        metrics["val_band"] = {"n": int(b.sum()), "judge": auc_ll(y_val[b], p[b]), "stage1": auc_ll(y_val[b], val["p1"].to_numpy()[b])}
    pd.DataFrame({"s1_id": val["s1_id"], "cand_id": val["cand_id"], "label": y_val, "p1": val["p1"], "judge_p": p}) \
        .to_parquet(WORK / "val_preds.parquet", index=False)
    metrics["runtime_s"] = round(time.time() - t0, 1)
    kaggle_env.write_json(metrics, WORK / "metrics.json")
    kaggle_env.log(f"val {metrics['val']} band {metrics.get('val_band')}; done in {metrics['runtime_s']}s")


if __name__ == "__main__":
    main()
