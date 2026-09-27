"""Unit tests for src/io_utils.py: TSV safety and the write/round-trip contract."""

import pandas as pd
import pytest

from src import io_utils


def test_load_tsv_preserves_leading_zeros_and_na_like_strings(tmp_path):
    """dtype=str + keep_default_na=False must not mangle IDs or an 'NA' business name."""
    path = tmp_path / "s1.tsv"
    path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-00047\tNA\t123 Main St\tNA\n",
        encoding="utf-8",
    )
    df = io_utils.load_tsv(path)
    assert df.loc[0, "entity_id"] == "S1-00047"
    assert df.loc[0, "business_name"] == "NA"
    assert df.loc[0, "country"] == "NA"


def test_load_tsv_row_count_assert_on_quote_swallowing(tmp_path):
    """A stray unescaped quote must not silently drop a row; QUOTE_NONE prevents swallowing."""
    path = tmp_path / "s1.tsv"
    path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        'S1-1\tJoe\'s "Best" Diner\t1 Main St\tUS\n'
        "S1-2\tOther Co\t2 Main St\tUS\n",
        encoding="utf-8",
    )
    df = io_utils.load_tsv(path)
    assert len(df) == 2


def test_assert_unique_ids_catches_duplicates():
    """Duplicate entity_id values must raise, never pass silently."""
    df = pd.DataFrame({"entity_id": ["S1-1", "S1-1"]})
    with pytest.raises(AssertionError):
        io_utils.assert_unique_ids(df, "S1-")


def test_assert_unique_ids_catches_wrong_prefix():
    """An entity_id without the expected prefix must raise."""
    df = pd.DataFrame({"entity_id": ["S2-1", "S1-2"]})
    with pytest.raises(AssertionError):
        io_utils.assert_unique_ids(df, "S1-")


def test_load_ground_truth_parses_empty_and_nonempty(tmp_path):
    """An empty matched_entity_ids field must become an empty set, not {''}."""
    path = tmp_path / "gt.tsv"
    path.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1,S3-2\n"
        "S1-2\t\n",
        encoding="utf-8",
    )
    gt = io_utils.load_ground_truth(path)
    assert gt["S1-1"] == {"S2-1", "S3-2"}
    assert gt["S1-2"] == set()


def test_load_ground_truth_rejects_unknown_ids(tmp_path):
    """A match ID outside the provided valid_ids set must raise."""
    path = tmp_path / "gt.tsv"
    path.write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-999\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError):
        io_utils.load_ground_truth(path, valid_ids={"S2-1"})


def test_write_id_list_tsv_empty_field_and_format(tmp_path):
    """Empty match lists write as an empty field; no quoting, no trailing spaces."""
    path = tmp_path / "out.tsv"
    io_utils.write_id_list_tsv(path, [("S1-1", ["S2-1", "S3-2"]), ("S1-2", [])])
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert lines[1] == "S1-1\tS2-1,S3-2"
    assert lines[2] == "S1-2\t"
    assert "\r" not in text


def test_pairs_to_lists_arrow_and_dupes():
    """Sorted unique IDs per S1, also for Arrow-backed string columns (NB09a v1 crash)."""
    import pandas as pd

    from src import io_utils

    df = pd.DataFrame({"s1_id": ["S1-2", "S1-1", "S1-2", "S1-2"], "cand_id": ["S3-9", "S2-5", "S2-1", "S3-9"]})
    df = df.astype("string[pyarrow]")
    out = io_utils.pairs_to_lists(df["s1_id"].tolist(), df["cand_id"].tolist())
    assert out == {"S1-2": ["S2-1", "S3-9"], "S1-1": ["S2-5"]}
    assert io_utils.pairs_to_lists([], []) == {}
