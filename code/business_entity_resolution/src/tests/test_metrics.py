"""Unit tests for src/metrics.py, including the official worked example."""

from src import metrics


def test_f05_entity_worked_example():
    """The problem statement's worked example: pred {47,193,812} vs true {47,812} -> 0.714."""
    score = metrics.f05_entity({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert abs(score - 0.714) < 1e-3


def test_f05_entity_both_empty():
    """Truly empty + predicted empty scores 1.0."""
    assert metrics.f05_entity(set(), set()) == 1.0


def test_f05_entity_true_empty_pred_nonempty():
    """Truly empty + predicted anything scores 0.0."""
    assert metrics.f05_entity({"S2-1"}, set()) == 0.0


def test_f05_entity_pred_empty_true_nonempty():
    """Has matches + predicted empty scores 0.0."""
    assert metrics.f05_entity(set(), {"S2-1"}) == 0.0


def test_f05_entity_no_overlap():
    """No true positives at all scores 0.0 even if both sets are non-empty."""
    assert metrics.f05_entity({"S2-1"}, {"S2-2"}) == 0.0


def test_f05_macro_averages_over_true_map():
    """f05_macro averages per-entity scores over every ID in true_map."""
    true_map = {"S1-1": {"S2-1"}, "S1-2": set()}
    pred_map = {"S1-1": {"S2-1"}, "S1-2": set()}
    assert metrics.f05_macro(pred_map, true_map) == 1.0


def test_f05_macro_missing_prediction_scores_zero():
    """An S1 entity absent from pred_map is treated as an empty prediction."""
    true_map = {"S1-1": {"S2-1"}}
    pred_map = {}
    assert metrics.f05_macro(pred_map, true_map) == 0.0


def test_bootstrap_diff_identical_scores_includes_zero():
    """Identical score sets should never show a significant difference."""
    scores = {"S1-1": 1.0, "S1-2": 0.5, "S1-3": 0.0}
    result = metrics.bootstrap_diff(scores, scores, n_resamples=200, seed=1)
    assert result["mean_diff"] == 0.0
    assert not result["excludes_zero"]


def test_bootstrap_diff_clear_improvement_excludes_zero():
    """A uniform +1.0 improvement across many entities should be detected as significant."""
    scores_a = {f"S1-{i}": 0.0 for i in range(50)}
    scores_b = {f"S1-{i}": 1.0 for i in range(50)}
    result = metrics.bootstrap_diff(scores_a, scores_b, n_resamples=200, seed=1)
    assert result["mean_diff"] == 1.0
    assert result["excludes_zero"]
