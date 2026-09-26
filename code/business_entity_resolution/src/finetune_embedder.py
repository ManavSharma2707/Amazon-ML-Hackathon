"""Step 5b — fine-tune the embedder on Half A only (master plan SS11.4-11.5).

- `build_triplets`: (anchor S1, positive match, hard negative) from Half A.
  Hard negatives are the frozen-kNN neighbours of the anchor that are not
  its matches (ranks 1-30); up to `hard_negs` distinct ones per anchor,
  spread over its positives. S2<->S3 sibling pairs of the same S1 are added
  as extra positives.
- `one_s1_per_batch`: batches in which every S1 entity appears at most once
  (two positives of one S1 in a batch would be false in-batch negatives).
- `mnrl_loss`: in-batch multiple-negatives ranking loss over [positives;
  hard negatives], with a mask removing any candidate owned by the anchor's
  own S1 other than its label (one-owner makes ownership well defined).
- `train_lora`: LoRA on all linear layers, fp16 autocast + GradScaler (T4:
  no bf16), grad clipping, warmup, checkpoints every N steps, a wall-clock
  budget. Scrambled-letter augmentation on `scramble_frac` of batches and
  field dropout on `field_dropout` of positives (country-agnostic
  regularisation, plan SS11.4). The loss uses the MRL-truncated vectors
  that blocking actually uses.
"""

from __future__ import annotations

import math
import random
import time

import numpy as np
import pandas as pd

from . import scramble


def build_triplets(
    anchors: pd.DataFrame,
    pairs: pd.DataFrame,
    pool: pd.DataFrame,
    knn_idx: np.ndarray,
    hard_negs: int = 3,
    sibling_frac: float = 0.25,
    seed: int = 42,
) -> pd.DataFrame:
    """Build training triplets for Half A anchors.

    Inputs: anchors - Half A S1 rows (entity_id, text) aligned with knn_idx;
            pairs - true pairs (s1_id, match_id); pool - pool rows (entity_id,
            text, owner) in the order knn_idx points into; knn_idx - frozen
            top-k pool rows per anchor; hard_negs - distinct hard negatives per
            anchor; sibling_frac - S2<->S3 sibling triplets as a share of the
            anchor triplets; seed.
    Outputs: DataFrame (s1_id, a_text, p_text, n_text, n_owner).
    """
    rng = np.random.default_rng(seed)
    p_pos = pd.Series(np.arange(len(pool)), index=pool["entity_id"].to_numpy())
    p_text = pool["text"].to_numpy()
    p_owner = pool["owner"].to_numpy()
    a_pos = pd.Series(np.arange(len(anchors)), index=anchors["entity_id"].to_numpy())
    pr = pairs[pairs["s1_id"].isin(a_pos.index)]
    matches = pr.groupby("s1_id")["match_id"].apply(list)
    rows = []
    for s1_id, ms in matches.items():
        ai = a_pos[s1_id]
        mrows = set(p_pos.reindex(ms).dropna().astype(int).tolist())
        negs = [int(j) for j in knn_idx[ai] if j >= 0 and int(j) not in mrows][:max(hard_negs * 3, 10)]
        if not negs:
            continue
        negs = list(rng.permutation(negs)[:hard_negs])
        a_text = anchors["text"].iat[ai]
        for t, m in enumerate(ms):
            if m not in p_pos.index:
                continue
            n = negs[t % len(negs)]
            rows.append((s1_id, a_text, p_text[p_pos[m]], p_text[n], p_owner[n]))
        # Sibling positives: two matches of the same S1 (duplicate-within-source structure).
        if len(ms) >= 2 and rng.random() < sibling_frac * 2:
            i, j = rng.choice(len(ms), 2, replace=False)
            if ms[i] in p_pos.index and ms[j] in p_pos.index:
                rows.append((s1_id, p_text[p_pos[ms[i]]], p_text[p_pos[ms[j]]], p_text[negs[0]], p_owner[negs[0]]))
    return pd.DataFrame(rows, columns=["s1_id", "a_text", "p_text", "n_text", "n_owner"])


def one_s1_per_batch(s1_ids: np.ndarray, batch: int, seed: int = 42) -> list[np.ndarray]:
    """Batches of row indices in which no S1 entity appears twice.

    Triplet t of each S1 goes to "layer" t; within a layer every S1 occurs
    once, so chunking a shuffled layer gives valid batches. Batch order is
    shuffled at the end.

    Inputs: s1_ids - S1 id per triplet row; batch - batch size; seed.
    Outputs: list of index arrays.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"s1": s1_ids, "i": np.arange(len(s1_ids))})
    df = df.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
    df["layer"] = df.groupby("s1").cumcount()
    batches = []
    for _, g in df.groupby("layer"):
        idx = g["i"].to_numpy()
        batches += [idx[s : s + batch] for s in range(0, len(idx), batch) if len(idx[s : s + batch]) > 1]
    order = rng.permutation(len(batches))
    return [batches[i] for i in order]


def field_dropout(text: str, rng: random.Random) -> str:
    """Drop the name or the address part of a "name | address" text (one side of a pair)."""
    if " | " not in text:
        return text
    name, addr = text.split(" | ", 1)
    return name if rng.random() < 0.5 else addr


def mnrl_loss(a, c, labels_owner, cand_owner, scale: float = 20.0):
    """Multiple-negatives ranking loss with a same-owner mask.

    Inputs: a [B, d] anchor vectors; c [2B, d] candidates (positives then
            hard negatives), both L2-normalised; labels_owner [B] S1 codes of
            the anchors; cand_owner [2B] owner codes of candidates (-1 = none);
            scale - cosine temperature.
    Outputs: scalar loss tensor. Candidate j is masked for anchor i if it is
    owned by anchor i's S1 and j != i (a false negative).
    """
    import torch

    logits = scale * (a @ c.T)
    b = a.shape[0]
    same = labels_owner[:, None] == cand_owner[None, :]
    same[torch.arange(b), torch.arange(b)] = False
    logits = logits.masked_fill(same, float("-inf"))
    return torch.nn.functional.cross_entropy(logits.float(), torch.arange(b, device=a.device))


def train_lora(
    model_dir: str,
    triplets: pd.DataFrame,
    out_dir,
    cfg: dict,
    dim: int,
    max_len: int = 64,
    device: str = "cuda:0",
    time_budget_s: float = 3 * 3600,
    ckpt_every: int = 500,
    log_fn=print,
    seed: int = 42,
):
    """LoRA fine-tune of Qwen3-Embedding with MNRL on triplets; returns the merged model.

    Inputs: model_dir; triplets (build_triplets output); out_dir - checkpoint
            folder; cfg - finetune config (lr, batch, scramble_frac,
            field_dropout, lora_r); dim - MRL dim used in the loss; max_len;
            device; time_budget_s - stop early when exceeded; ckpt_every;
            log_fn; seed.
    Outputs: (merged model in eval mode, tokenizer, train log dict).
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float32, attn_implementation="sdpa").to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lcfg = LoraConfig(
        r=cfg.get("lora_r", 32), lora_alpha=2 * cfg.get("lora_r", 32), lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=0.01)
    batches = one_s1_per_batch(triplets["s1_id"].to_numpy(), cfg["batch"], seed)
    total = len(batches)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    )
    scaler = torch.cuda.amp.GradScaler()
    codes = {s: i for i, s in enumerate(pd.unique(triplets["s1_id"]))}
    a_owner_all = triplets["s1_id"].map(codes).to_numpy()
    n_owner_all = triplets["n_owner"].map(codes).fillna(-1).astype(np.int64).to_numpy()

    def enc(texts):
        """Forward a list of texts -> MRL-truncated normalised vectors (with grad)."""
        t = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            h = model(**t).last_hidden_state[:, -1, :dim]
        return torch.nn.functional.normalize(h.float(), dim=-1)

    model.train()
    t0, losses, log = time.time(), [], {"steps": 0, "total_batches": total, "loss_curve": []}
    for step, b in enumerate(batches):
        rows = triplets.iloc[b]
        a_t, p_t, n_t = rows["a_text"].tolist(), rows["p_text"].tolist(), rows["n_text"].tolist()
        p_t = [field_dropout(t, rng) if rng.random() < cfg["field_dropout"] else t for t in p_t]
        if rng.random() < cfg["scramble_frac"]:
            table = scramble.make_scrambler(rng.randrange(1 << 30))
            a_t, p_t, n_t = (scramble.scramble_texts(x, table) for x in (a_t, p_t, n_t))
        a = enc(a_t)
        c = torch.cat([enc(p_t), enc(n_t)])
        a_own = torch.as_tensor(a_owner_all[b], device=device)
        c_own = torch.cat([a_own, torch.as_tensor(n_owner_all[b], device=device)])
        loss = mnrl_loss(a, c, a_own, c_own)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        losses.append(float(loss))
        log["steps"] = step + 1
        if (step + 1) % 50 == 0:
            m = float(np.mean(losses[-50:]))
            log["loss_curve"].append((step + 1, round(m, 4)))
            log_fn(f"    step {step + 1}/{total} loss {m:.4f} ({time.time() - t0:.0f}s)")
        if (step + 1) % ckpt_every == 0:
            model.save_pretrained(str(out_dir / "lora_ckpt"))
        if time.time() - t0 > time_budget_s:
            log_fn(f"    time budget reached at step {step + 1}/{total}")
            break
    log["train_s"] = round(time.time() - t0, 1)
    model.save_pretrained(str(out_dir / "lora_final"))
    merged = model.merge_and_unload().half().eval()
    return merged, tok, log


def recall_mrr(fwd_idx: np.ndarray, q_ids, p_ids, pairs: pd.DataFrame, ks=(5, 10, 20, 50)) -> dict:
    """Recall@k and MRR of true pairs in top-k lists (acceptance test, plan SS11.5)."""
    q_pos = pd.Series(np.arange(len(q_ids)), index=q_ids)
    p_pos = pd.Series(np.arange(len(p_ids)), index=p_ids)
    pr = pairs[pairs["s1_id"].isin(q_pos.index) & pairs["match_id"].isin(p_pos.index)]
    lists = fwd_idx[q_pos.loc[pr["s1_id"]].to_numpy()]
    target = p_pos.loc[pr["match_id"]].to_numpy()
    k = lists.shape[1]
    rank = np.where(lists == target[:, None], np.arange(k)[None, :], k).min(1)
    out = {f"recall@{kk}": round(float((rank < kk).mean()), 5) for kk in ks if kk <= k}
    out["mrr"] = round(float(np.where(rank < k, 1.0 / (rank + 1), 0.0).mean()), 5)
    out["n_pairs"] = int(len(pr))
    return out


def main() -> None:
    """Smoke test of the sampler."""
    print([list(b) for b in one_s1_per_batch(np.array(["a", "a", "b", "c", "c", "c"]), 2)])


if __name__ == "__main__":
    main()
