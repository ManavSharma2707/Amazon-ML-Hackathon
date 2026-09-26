"""Unit tests for src/eda.py on a tiny synthetic dataset (fast, runs locally).

The real dataset (millions of rows) is only ever run as a Kaggle CPU kernel
(memory.md SS5) — these tests just prove the logic is correct on data small
enough to fit anywhere.
"""

from pathlib import Path

import pandas as pd
import pytest

from src import eda


@pytest.fixture
def tiny_dataset(tmp_path):
    """Write a tiny train+test dataset (2 countries, a few edge cases) and return its root."""
    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir(parents=True)

    def write(path, rows, header):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\t".join(header) + "\n")
            for row in rows:
                f.write("\t".join(row) + "\n")

    write(
        root / "train" / "train_source1.tsv",
        [
            ("S1-1", "Acme Corp", "1 Main St", "US"),
            ("S1-2", "Bright Bakery", "2 Oak Ave", "US"),
            ("S1-3", "Curry House", "3 MG Road", "India"),
        ],
        ["entity_id", "business_name", "business_address", "country"],
    )
    write(
        root / "train" / "train_source2.tsv",
        [
            ("S2-1", "Acme Corp", "1 Main St", "US"),
            ("S2-2", "Unmatched Co", "9 Nowhere Rd", "US"),
        ],
        ["entity_id", "business_name", "business_address", "country"],
    )
    write(
        root / "train" / "train_source3.tsv",
        [
            ("S3-1", "Acme Corporation", "1 Main Street", "US"),
            ("S3-2", "Curry Hous", "3 MG Rd", "India"),
        ],
        ["entity_id", "business_name", "business_address", "country"],
    )
    write(
        root / "train" / "train_ground_truth.tsv",
        [
            ("S1-1", "S2-1,S3-1"),
            ("S1-2", ""),
            ("S1-3", "S3-2"),
        ],
        ["source1_entity_id", "matched_entity_ids"],
    )
    write(
        root / "test" / "test_source1.tsv",
        [
            ("S1-10", "Test Biz", "10 Test St", "France"),
        ],
        ["entity_id", "business_name", "business_address", "country"],
    )
    write(root / "test" / "test_source2.tsv", [], ["entity_id", "business_name", "business_address", "country"])
    write(root / "test" / "test_source3.tsv", [], ["entity_id", "business_name", "business_address", "country"])
    return root


def test_e1_sizes(tiny_dataset):
    train = eda.load_train(tiny_dataset)
    test = eda.load_test(tiny_dataset)
    sizes = eda.e1_sizes(train, test)
    assert sizes["train"]["s1"]["total"] == 3
    assert sizes["test"]["s1"]["total"] == 1


def test_e2_singleton_rate(tiny_dataset):
    train = eda.load_train(tiny_dataset)
    result = eda.e2_singleton_rate(train["gt_raw"], train["s1"])
    assert abs(result["overall"] - 1 / 3) < 1e-9


def test_e4_no_multi_owner_in_tiny_set(tiny_dataset):
    train = eda.load_train(tiny_dataset)
    exploded = eda.explode_ground_truth(train["gt_raw"])
    result = eda.e4_multi_owner_check(exploded)
    assert result["n_multi_owner"] == 0


def test_e4_detects_multi_owner():
    gt_raw = pd.DataFrame(
        {"source1_entity_id": ["S1-1", "S1-2"], "matched_entity_ids": ["S2-1", "S2-1"]}
    )
    exploded = eda.explode_ground_truth(gt_raw)
    result = eda.e4_multi_owner_check(exploded)
    assert result["n_multi_owner"] == 1


def test_e5_cross_country_matches(tiny_dataset):
    train = eda.load_train(tiny_dataset)
    exploded = eda.explode_ground_truth(train["gt_raw"])
    result = eda.e5_cross_country_matches(exploded, train["s1"], train["s2"], train["s3"])
    assert result["n_cross_country"] == 0


def test_e6_unmatched_share(tiny_dataset):
    train = eda.load_train(tiny_dataset)
    exploded = eda.explode_ground_truth(train["gt_raw"])
    result = eda.e6_unmatched_share(exploded, train["s2"], train["s3"])
    assert result["s2_unmatched_frac"] == 0.5  # S2-2 is unmatched, S2-1 is matched


def test_all_empty_floor_matches_e2(tiny_dataset):
    findings = eda.run_eda(tiny_dataset)
    assert findings["all_empty_floor"] == findings["e2_singleton_rate"]["overall"]


def test_shortcut_check_runs(tiny_dataset):
    train = eda.load_train(tiny_dataset)
    result = eda.shortcut_check(train["gt_raw"])
    assert "corr_id_number_vs_match_count" in result


def test_classify_name_relation_exact():
    assert eda._classify_name_relation("acme corp", "acme corp") == "exact"


def test_classify_name_relation_reorder():
    assert eda._classify_name_relation("john smith bakery", "bakery smith john") == "reorder"


def test_full_run_eda_end_to_end(tiny_dataset):
    """Smoke test: the whole pipeline runs without error on tiny data and returns all keys."""
    findings = eda.run_eda(tiny_dataset)
    for key in (
        "e1_sizes", "e2_singleton_rate", "e3_match_count_distribution", "e4_multi_owner_check",
        "e5_cross_country_matches", "e6_unmatched_share", "e7_exact_rate", "e8_noise_census",
        "e9_hard_negatives", "e10_field_quality", "e11_postcode_formats", "e12_test_composition",
        "shortcut_check", "all_empty_floor",
    ):
        assert key in findings
