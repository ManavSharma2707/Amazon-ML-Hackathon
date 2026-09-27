"""Step 5 — record embeddings (Qwen3-Embedding-0.6B) and GPU dense kNN (master plan SS11, SS12).

- `record_texts`: "name | address" built from the raw fields, lightly cleaned
  (NFKC + whitespace; accents kept, plan SS11.2). Country is left out: all
  blocking is within-country (EDA E5), so it would be the same on both sides.
- `Embedder`: transformers AutoModel with last-token pooling and left
  padding (how Qwen3-Embedding is meant to be used), fp16, sdpa attention
  (T4: no bf16, no flash-attn 2). MRL truncation to `dim` then L2-normalise.
- `encode`: length-sorted batches, split across all visible GPUs with one
  thread per GPU (tokenizer and CUDA kernels release the GIL).
- `knn_within_groups`: exact inner-product top-k on the GPU, chunked over
  both queries and pool, restricted to equal group labels (country
  equality only, no country values). Also returns the reverse top-k (pool
  record -> its best queries) from the same score blocks.

Scale guard (memory.md SS5): only the full-record text is embedded (MRL
256-d); name-only / address-only embeddings would triple GPU time and
exceed Kaggle's output-size limit at ~24M records.
"""

from __future__ import annotations

import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_WS = re.compile(r"\s+")


def light_clean(text: str) -> str:
    """NFKC + whitespace collapse; case and accents kept (plan SS11.2)."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def record_texts(names, addrs) -> list[str]:
    """Build "name | address" texts from raw name/address sequences.

    Inputs: names, addrs - equal-length sequences of raw strings.
    Outputs: list of texts; an empty address gives just the name.
    """
    out = []
    for n, a in zip(names, addrs):
        n, a = light_clean(n), light_clean(a)
        out.append(f"{n} | {a}" if a else n)
    return out


class Embedder:
    """One Qwen3-Embedding replica on one device (last-token pooling, MRL truncation)."""

    def __init__(self, model_dir: str, device: str, dim: int = 256, max_len: int = 64, instruction: str = ""):
        """Load tokenizer + model in fp16 with sdpa attention.

        Inputs: model_dir - local HF snapshot folder; device - "cuda:0"...;
                dim - MRL output dimension; max_len - token cap; instruction -
                prefix added to every text (same on both sides; "" = none).
        """
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.dim = dim
        self.max_len = max_len
        self.instruction = instruction
        self.timing = {"tok_s": 0.0, "fwd_s": 0.0, "tokens": 0}
        self.tok = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        # transformers 5 renamed `torch_dtype` to `dtype` (the old name may be
        # ignored), so pass the new name and also cast explicitly: an fp32
        # model on a T4 is several times slower (NB03 dry run v1: ~400 rec/s).
        try:
            model = AutoModel.from_pretrained(model_dir, dtype=dtype, attn_implementation="sdpa")
        except TypeError:
            model = AutoModel.from_pretrained(model_dir, torch_dtype=dtype, attn_implementation="sdpa")
        self.model = model.to(device=device, dtype=dtype)
        self.model.eval()
        self.param_dtype = str(next(self.model.parameters()).dtype)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode one batch -> float16 array [len(texts), dim], L2-normalised.

        Last-token pooling is valid because padding is on the left, so the
        final position is every sequence's last real token.
        """
        torch = self.torch
        if self.instruction:
            texts = [self.instruction + t for t in texts]
        t0 = time.perf_counter()
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        t1 = time.perf_counter()
        with torch.inference_mode():
            hid = self.model(**enc).last_hidden_state[:, -1, : self.dim].float()
            hid = torch.nn.functional.normalize(hid, dim=-1)
        out = hid.to(torch.float16).cpu().numpy()
        # Cumulative stage timings (tokenise vs forward) for throughput diagnosis.
        self.timing["tok_s"] += t1 - t0
        self.timing["fwd_s"] += time.perf_counter() - t1
        self.timing["tokens"] += int(enc["input_ids"].numel())
        return out


def encode(
    texts: list[str],
    embedders: list[Embedder],
    batch: int = 512,
    log_fn=print,
    log_every_s: float = 120.0,
) -> np.ndarray:
    """Encode all texts with length-sorted batches spread over several embedders (GPUs).

    Inputs: texts; embedders - one per device; batch - texts per batch;
            log_fn - progress logger; log_every_s - progress interval.
    Outputs: float16 array [len(texts), dim] in the original text order.
    """
    n = len(texts)
    dim = embedders[0].dim
    out = np.zeros((n, dim), dtype=np.float16)
    if n == 0:
        return out
    order = np.argsort(np.fromiter((len(t) for t in texts), dtype=np.int32, count=n), kind="stable")
    batches = [order[i : i + batch] for i in range(0, n, batch)]
    lock = threading.Lock()
    state = {"done": 0, "last": time.time(), "t0": time.time()}

    def work(worker: int) -> None:
        """Thread target: encode this worker's share of `batches` on its own GPU/embedder."""
        emb = embedders[worker]
        for b in batches[worker :: len(embedders)]:
            out[b] = emb.encode_batch([texts[i] for i in b])
            with lock:
                state["done"] += len(b)
                if time.time() - state["last"] > log_every_s:
                    state["last"] = time.time()
                    rate = state["done"] / (time.time() - state["t0"])
                    log_fn(f"    encoded {state['done']:,}/{n:,} ({rate:,.0f}/s, eta {(n - state['done']) / rate / 60:.1f} min)")

    with ThreadPoolExecutor(len(embedders)) as ex:
        list(ex.map(work, range(len(embedders))))
    return out


def _knn_one_group(q, p, k: int, rk: int, device: str, q_chunk: int, p_chunk: int):
    """Exact top-k (queries -> pool) and reverse top-rk (pool -> queries) for one group on one GPU.

    Inputs: q, p - float16 arrays (L2-normalised) of queries / pool rows;
            k, rk - forward / reverse neighbours; device; chunk sizes.
    Outputs: (fwd_idx int32 [nq,k], fwd_score f16, rev_idx int32 [np,rk],
    rev_score f16); indices are local to q / p; missing slots are -1.
    """
    import torch

    nq, npool = len(q), len(p)
    kk, rkk = min(k, npool), min(rk, nq)
    fwd_idx = np.full((nq, k), -1, dtype=np.int32)
    fwd_sc = np.zeros((nq, k), dtype=np.float16)
    rev_idx = np.full((npool, rk), -1, dtype=np.int32)
    rev_sc = np.zeros((npool, rk), dtype=np.float16)
    if nq == 0 or npool == 0:
        return fwd_idx, fwd_sc, rev_idx, rev_sc
    P = torch.from_numpy(p).to(device)
    rev_best_s = torch.full((npool, rkk), -2.0, dtype=torch.float16, device=device)
    rev_best_i = torch.full((npool, rkk), -1, dtype=torch.int64, device=device)
    for qs in range(0, nq, q_chunk):
        Q = torch.from_numpy(q[qs : qs + q_chunk]).to(device)
        best_s, best_i = None, None
        for ps in range(0, npool, p_chunk):
            S = Q @ P[ps : ps + p_chunk].T  # [cq, cp] fp16
            s, i = S.topk(min(kk, S.shape[1]), dim=1)
            i = i + ps
            if best_s is None:
                best_s, best_i = s, i
            else:
                cs, ci = torch.cat([best_s, s], 1), torch.cat([best_i, i], 1)
                best_s, sel = cs.topk(kk, dim=1)
                best_i = ci.gather(1, sel)
            # Reverse: best queries for each pool column of this block.
            rs, ri = S.topk(min(rkk, S.shape[0]), dim=0)  # [r, cp]
            rs, ri = rs.T, ri.T + qs
            cs = torch.cat([rev_best_s[ps : ps + p_chunk], rs], 1)
            ci = torch.cat([rev_best_i[ps : ps + p_chunk], ri], 1)
            ts, sel = cs.topk(rkk, dim=1)
            rev_best_s[ps : ps + p_chunk] = ts
            rev_best_i[ps : ps + p_chunk] = ci.gather(1, sel)
            del S
        fwd_idx[qs : qs + len(Q), :kk] = best_i.cpu().numpy().astype(np.int32)
        fwd_sc[qs : qs + len(Q), :kk] = best_s.cpu().numpy()
    rev_idx[:, :rkk] = rev_best_i.cpu().numpy().astype(np.int32)
    rev_sc[:, :rkk] = rev_best_s.cpu().numpy()
    del P
    torch.cuda.empty_cache()
    return fwd_idx, fwd_sc, rev_idx, rev_sc


def knn_within_groups(
    q_emb: np.ndarray,
    q_groups: np.ndarray,
    p_emb: np.ndarray,
    p_groups: np.ndarray,
    k: int = 30,
    rk: int = 5,
    devices: list[str] | None = None,
    q_chunk: int = 4096,
    p_chunk: int = 400_000,
    log_fn=print,
):
    """Exact dense kNN restricted to equal group labels (e.g. country equality).

    Each group label present in the queries is processed separately (pool
    rows with the same label only); groups are spread over the devices with
    one thread per device.

    Inputs: q_emb/p_emb - float16 L2-normalised matrices; q_groups/p_groups -
            label arrays (any values; compared only for equality); k - forward
            neighbours; rk - reverse neighbours; devices - e.g. ["cuda:0",
            "cuda:1"]; chunk sizes bound GPU memory.
    Outputs: (fwd_idx [nq,k] int32 global pool rows, fwd_score f16,
    rev_idx [np,rk] int32 global query rows, rev_score f16); -1 = no neighbour.
    """
    devices = devices or ["cuda:0"]
    fwd_idx = np.full((len(q_emb), k), -1, dtype=np.int32)
    fwd_sc = np.zeros((len(q_emb), k), dtype=np.float16)
    rev_idx = np.full((len(p_emb), rk), -1, dtype=np.int32)
    rev_sc = np.zeros((len(p_emb), rk), dtype=np.float16)
    labels = sorted(set(q_groups.tolist()))
    # Largest groups first so the two devices finish at similar times.
    labels.sort(key=lambda g: -int((q_groups == g).sum()))

    def run(dev_i: int) -> None:
        """Thread target: this device's share of country/split groups, largest first."""
        for g in labels[dev_i :: len(devices)]:
            qi = np.flatnonzero(q_groups == g)
            pi = np.flatnonzero(p_groups == g)
            if len(pi) == 0:  # label with no pool records: no neighbours (stays -1)
                continue
            t0 = time.time()
            fi, fs, ri, rs = _knn_one_group(q_emb[qi], p_emb[pi], k, rk, devices[dev_i], q_chunk, p_chunk)
            ok = fi >= 0
            fwd_idx[qi] = np.where(ok, pi[np.clip(fi, 0, None)], -1)
            fwd_sc[qi] = fs
            ok = ri >= 0
            rev_idx[pi] = np.where(ok, qi[np.clip(ri, 0, None)], -1)
            rev_sc[pi] = rs
            log_fn(f"    knn group {g!r}: {len(qi):,} queries x {len(pi):,} pool on {devices[dev_i]} ({time.time() - t0:.0f}s)")

    with ThreadPoolExecutor(len(devices)) as ex:
        list(ex.map(run, range(len(devices))))
    return fwd_idx, fwd_sc, rev_idx, rev_sc


def pair_recall_at_k(fwd_idx: np.ndarray, q_ids: np.ndarray, p_ids: np.ndarray, pairs, ks=(5, 10, 20, 30)) -> dict:
    """Share of true (s1, match) pairs whose match is in the S1's top-k dense list.

    Inputs: fwd_idx - [nq, K] global pool rows; q_ids/p_ids - ID arrays for
            query/pool rows; pairs - DataFrame (s1_id, match_id); ks.
    Outputs: {"recall@k": value, ..., "n_pairs": n} over pairs whose S1 is in q_ids.
    """
    import pandas as pd

    q_pos = pd.Series(np.arange(len(q_ids)), index=q_ids)
    p_pos = pd.Series(np.arange(len(p_ids)), index=p_ids)
    pr = pairs[pairs["s1_id"].isin(q_pos.index) & pairs["match_id"].isin(p_pos.index)]
    qi = q_pos.loc[pr["s1_id"]].to_numpy()
    pi = p_pos.loc[pr["match_id"]].to_numpy()
    lists = fwd_idx[qi]
    hit_rank = np.where(lists == pi[:, None], np.arange(lists.shape[1])[None, :], lists.shape[1]).min(1)
    res = {f"recall@{k}": round(float((hit_rank < k).mean()), 5) for k in ks if k <= lists.shape[1]}
    res["n_pairs"] = int(len(pr))
    return res


def main() -> None:
    """Smoke test: build texts for two records (no model load)."""
    print(record_texts(["Café  Rouge", "X"], ["1 Rue A,  Paris", ""]))


if __name__ == "__main__":
    main()
