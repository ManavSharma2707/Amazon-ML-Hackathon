"""Tests for src/judge_data.py (band selection and prompt building)."""

import numpy as np
import pandas as pd

from src import judge_data, normalize


def test_select_train_band_balance_and_cap():
    """In-band pairs are kept, ~1:2 positives:negatives, capped, with some confident pairs flagged band=0."""
    rng = np.random.default_rng(0)
    n = 20000
    oof = pd.DataFrame({"s1_id": [f"S1-{i // 10}" for i in range(n)], "cand_id": [f"S2-{i}" for i in range(n)],
                        "p1": rng.random(n), "label": (rng.random(n) < 0.2).astype(int)})
    sel = judge_data.select_train_band(oof, max_n=3000, seed=1)
    assert len(sel) <= 3000 and not sel.duplicated(["s1_id", "cand_id"]).any()
    assert 0.25 < sel["label"].mean() < 0.4
    assert set(sel["band"]) == {0, 1}


def test_select_infer_band_order():
    """Only band pairs, most uncertain first."""
    p1 = pd.DataFrame({"s1_id": list("abcde"), "cand_id": list("vwxyz"), "p1": [0.05, 0.5, 0.8, 0.45, 0.95]})
    b = judge_data.select_infer_band(p1, max_n=2)
    assert b["cand_id"].tolist() == ["w", "x"] or b["cand_id"].tolist() == ["w", "y"]
    assert judge_data.select_infer_band(p1)["p1"].between(0.1, 0.9).all()


def test_build_dataset_prompt_contents():
    """Prompts carry both records and the evidence block, never the country."""
    def rec(eid, name, addr):
        r = normalize.normalize_record(name, addr)
        r.update(entity_id=eid, raw_name=name, raw_addr=addr, country="US")
        return r
    s1 = pd.DataFrame([rec("S1-1", "Acme Tools Pvt Ltd", "12 Main Road, Pune")])
    pool = pd.DataFrame([rec("S2-1", "ACME TOOLS PRIVATE LIMITED", "12 Main Rd, Pune")])
    lk = {"generic_name": set(), "generic_addr": set()}
    ds = judge_data.build_dataset(pd.DataFrame({"s1_id": ["S1-1"], "cand_id": ["S2-1"], "label": [1]}), s1, pool, lk)
    u = ds["user"].iloc[0]
    assert "Record A" in u and "Record B" in u and "(abbreviation)" in u and "Same business?" in u
    assert "US" not in u.replace("ACME", "")
