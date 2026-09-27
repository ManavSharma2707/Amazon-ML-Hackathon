"""Tests for src/predict.py: the demo train/predict flow and the --from-features
exact-reproduction path (against this repo's own bundled artifacts/)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import features, io_utils, predict

SAMPLE = Path(__file__).resolve().parents[4] / "sample"
ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


@pytest.mark.skipif(not (SAMPLE / "train" / "train_ground_truth.tsv").exists(), reason="sample/ not built")
def test_train_demo_and_apply_to_test_split(tmp_path):
    """train_demo on train/ + run_inference on test/ produces valid, non-trivial predictions."""
    config = io_utils.load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    io_utils.set_seeds(config["seed"])
    pruner, model, cal, dcfg = predict.train_demo(config, SAMPLE, SAMPLE / "train" / "train_ground_truth.tsv",
                                                  n_jobs=1, log=lambda m: None)
    report = predict.run_inference(config, SAMPLE, tmp_path, pruner, model, cal, dcfg, split="test", n_jobs=1,
                                   log=lambda m: None)
    assert report["n_s1"] == len(io_utils.load_tsv(SAMPLE / "test" / "test_source1.tsv"))
    assert 0 < report["n_pred_pairs"] < report["n_cand_pairs"]
    matched = (tmp_path / "matching_results.tsv").read_text(encoding="utf-8")
    assert matched.startswith("source1_entity_id\tmatched_entity_ids\n")
    assert (tmp_path / "candidate_pairs.tsv").exists()

    # save/reload round-trip: applying the saved artifacts reproduces the same predictions
    art_dir = tmp_path / "artifacts"
    predict.save_demo_artifacts(art_dir, pruner, model, cal, dcfg)
    pruner2, model2, cal2, dcfg2 = predict.load_demo_artifacts(art_dir)
    out2 = tmp_path / "reapplied"
    predict.run_inference(config, SAMPLE, out2, pruner2, model2, cal2, dcfg2, split="test", n_jobs=1, log=lambda m: None)
    assert (out2 / "matching_results.tsv").read_text(encoding="utf-8") == matched


@pytest.mark.skipif(not ARTIFACTS.exists(), reason="artifacts/ not present (run NB09a and copy it in)")
def test_from_features_loads_real_shipped_artifacts(tmp_path):
    """--from-features loads the actual bundled artifacts/ (stage1_model.txt, calibrator.npz,
    final_config.json) and applies them to a small synthetic feature table without error."""
    model, cal, dcfg = predict.load_artifacts(ARTIFACTS)
    assert dcfg["mode"] == "hard" and dcfg["method"] == "decoder"
    n = 40
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.random((n, len(features.FEATURES))).astype(np.float32), columns=features.FEATURES)
    df.insert(0, "cand_id", [f"S2-{i}" for i in range(n)])
    df.insert(0, "s1_id", [f"S1-{i // 8}" for i in range(n)])
    feats_path = tmp_path / "feats_test.parquet"
    df.to_parquet(feats_path, index=False)
    test_s1 = tmp_path / "test_source1.tsv"
    test_s1.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n" +
                       "".join(f"S1-{i}\ta\tb\tUS\n" for i in range(n // 8 + 1)), encoding="utf-8")
    out_dir = tmp_path / "out"
    report = predict.run_from_features(feats_path, test_s1, ARTIFACTS, out_dir, n_jobs=1)
    assert report["n_scored_pairs"] == n and (out_dir / "matching_results.tsv").exists()
