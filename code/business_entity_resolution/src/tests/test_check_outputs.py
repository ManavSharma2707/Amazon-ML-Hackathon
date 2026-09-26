"""Tests for the streaming src/check_outputs.py."""

from src import check_outputs as co


def _write(path, text):
    path.write_text(text, encoding="utf-8", newline="\n")


def _setup(tmp_path):
    test = tmp_path / "test"
    test.mkdir()
    hdr = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    _write(test / "test_source1.tsv", hdr + "S1-1\ta\tx\tUS\nS1-2\tb\ty\tFrance\nS1-3\tc\tz\tIndia\n")
    _write(test / "test_source2.tsv", hdr + "S2-10\ta\tx\tUS\nS2-11\tb\ty\tFrance\n")
    _write(test / "test_source3.tsv", hdr + "S3-20\ta\tx\tUS\n")
    out = tmp_path / "out"
    out.mkdir()
    return test, out


def test_pass_and_summary(tmp_path):
    """Well-formed files pass; the summary counts empties per country."""
    test, out = _setup(tmp_path)
    _write(out / "matching_results.tsv", "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-10,S3-20\nS1-2\t\nS1-3\t\n")
    _write(out / "candidate_pairs.tsv", "source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-10,S3-20\nS1-2\tS2-11\nS1-3\t\n")
    assert co.check_outputs(out, test) == []
    s = co.output_summary(out, test)
    assert s["US"]["mean_pred_size"] == 2.0 and s["France"]["pred_empty_rate"] == 1.0


def test_failures_detected(tmp_path):
    """Missing S1 (France), unknown ID, match outside candidates and a duplicate in a list are reported."""
    test, out = _setup(tmp_path)
    _write(out / "matching_results.tsv", "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-10,S2-99\nS1-3\tS3-20,S3-20\n")
    _write(out / "candidate_pairs.tsv", "source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-10\nS1-3\tS3-20\n")
    probs = " | ".join(co.check_outputs(out, test))
    assert "1 required S1 entities missing" in probs
    assert "'France'" in probs
    assert "do not exist" in probs
    assert "not in its candidate list" in probs
    assert "duplicate IDs in the list" in probs


def test_row_order_mismatch_resolved(tmp_path):
    """Candidate rows in a different order are matched by S1 ID, not by line."""
    test, out = _setup(tmp_path)
    _write(out / "matching_results.tsv", "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-10\nS1-2\t\nS1-3\t\n")
    _write(out / "candidate_pairs.tsv", "source1_entity_id\tcandidate_entity_ids\nS1-3\t\nS1-2\t\nS1-1\tS2-10\n")
    assert co.check_outputs(out, test) == []
