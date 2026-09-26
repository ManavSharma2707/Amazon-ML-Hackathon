"""Judge inference on the uncertain band (master plan SS16.5).

One 4-bit Qwen3-4B + LoRA adapter per GPU; p = softmax(logit_Yes, logit_No)[Yes]
at the answer position (same quantity as in training, judge_train.py). Pairs
are scored most-uncertain-first under a wall-clock budget; B and test are
scored in one merged queue and cut at the same uncertainty depth |p1 - 0.5|,
so "judge_scored" means the same thing in the combiner's training data (B)
and on test. Unscored pairs get judge_p = NaN.
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np
import pandas as pd

from . import judge_train


def load(model_dir, adapter_dir, device: int):
    """4-bit base + LoRA adapter on cuda:<device>, eval mode. Outputs (tokenizer, model)."""
    from peft import PeftModel

    tok, base = judge_train.load_model(model_dir, train=False, device=device)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    return tok, model


def merged_queue(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One table of all splits' band pairs, most uncertain first (|p1 - 0.5| ascending)."""
    parts = [f.assign(split=name, row=np.arange(len(f))) for name, f in frames.items()]
    q = pd.concat(parts, ignore_index=True)
    q["depth"] = np.abs(q["p1"].to_numpy() - 0.5)
    return q.sort_values(["depth", "split", "row"], kind="stable").reset_index(drop=True)


def cap_depth(frames: dict[str, pd.DataFrame], band_half: float = 0.4) -> float:
    """Deepest |p1 - 0.5| every split covers completely (a capped split stops early)."""
    d = band_half
    for f in frames.values():
        if len(f):
            last = float(np.abs(f["p1"].to_numpy() - 0.5).max())
            if last < band_half - 1e-3:  # file was cut by the NB06b cap before the band edge
                d = min(d, last)
    return d


def score_budgeted(models: list, ids: list[list[int]], budget_s: float, batch: int = 16, log=print) -> np.ndarray:
    """Score token-id lists in order with one worker thread per (tok, model, device), until the budget ends.

    Outputs: p(Yes) per item (NaN where not scored before the deadline).
    """
    out = np.full(len(ids), np.nan, dtype=np.float32)
    jobs: queue.Queue = queue.Queue()
    for s in range(0, len(ids), batch):
        jobs.put((s, min(s + batch, len(ids))))
    deadline = time.time() + budget_s
    done = [0]
    lock = threading.Lock()

    def work(tok, model, device):
        while time.time() < deadline:
            try:
                a, b = jobs.get_nowait()
            except queue.Empty:
                return
            out[a:b] = judge_train.predict(model, tok, ids[a:b], batch=batch, device=device)
            with lock:
                done[0] += b - a
                if done[0] // (batch * 200) != (done[0] - (b - a)) // (batch * 200):
                    log(f"    judge scored {done[0]:,}/{len(ids):,}")

    threads = [threading.Thread(target=work, args=m) for m in models]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def main() -> None:
    """Show the merged-queue order on a toy example (smoke test)."""
    f = {"B": pd.DataFrame({"p1": [0.2, 0.5]}), "test": pd.DataFrame({"p1": [0.45, 0.85]})}
    print(merged_queue(f)[["split", "row", "depth"]])


if __name__ == "__main__":
    main()
