"""Unit tests for src/explain_diff.py: token relations, alignment, field and number features."""

import math

from src import explain_diff as ed


def test_abbreviation_relations():
    """Corp~Corporation (prefix), Pvt~Private (skeleton), Bd~Boulevard (2-letter: weak short_abbrev)."""
    assert ed.token_relation("corp", "corporation") == "prefix_abbrev"
    assert ed.token_relation("corporation", "corp") == "prefix_abbrev"  # symmetric
    assert ed.token_relation("pvt", "private") == "skeleton_abbrev"
    assert ed.token_relation("ltd", "limited") == "skeleton_abbrev"
    assert ed.token_relation("blvd", "boulevard") == "skeleton_abbrev"
    assert ed.token_relation("bd", "boulevard") == "short_abbrev"


def test_st_vs_south_is_weak_only():
    """'st' vs 'south' is at most the weak 2-letter abbreviation, never a strong relation."""
    r = ed.token_relation("st", "south")
    assert r == "short_abbrev"
    assert ed.STRENGTH[r] < ed.STRENGTH["skeleton_abbrev"]
    feats, _ = ed.explain_features(["st"], ["south"], [2.0], [2.0], [False], [False])
    row = dict(zip(ed.FIELD_FEATURES, feats))
    assert row["n_short_abbrev"] == 1 and row["n_abbrev"] == 0


def test_numbers_are_never_typos():
    """12 vs 12bis share a digit base; 12 vs 13 is no relation (a conflict, not a typo)."""
    assert ed.token_relation("12", "12bis") == "numeric_equal"
    assert ed.token_relation("12", "12b") == "numeric_equal"
    assert ed.token_relation("03", "3") == "numeric_equal"
    assert ed.token_relation("12", "13") == "none"
    assert ed.token_relation("12", "21") == "none"


def test_typo_fold_and_leet():
    """Edit-distance typos, phonetic folds and look-alike digits are explained."""
    assert ed.token_relation("hendersonville", "hendresonville") == "typo"
    assert ed.token_relation("trust", "trsut") == "typo"
    assert ed.token_relation("kolkata", "k01kata") == "fold_exact"
    assert ed.token_relation("ab", "ac") == "none"  # too short for a typo


def test_initialism_sbi():
    """SBI ~ State Bank of India (the short connector 'of' may be skipped)."""
    al = ed.align(["sbi"], "state bank of india".split())
    assert al["rel1"] == ["initialism"]
    assert all(r == "initialism" for r in al["rel2"])
    al = ed.align("state bank of india branch".split(), ["sbi", "branch"])
    assert al["rel2"] == ["initialism", "exact"]


def test_split_join_walmart():
    """Walmart ~ Wal Mart in both directions."""
    al = ed.align(["walmart", "store"], ["wal", "mart", "store"])
    assert al["rel1"] == ["split_join", "exact"]
    assert al["rel2"] == ["split_join", "split_join", "exact"]
    al = ed.align(["wal", "mart"], ["walmart"])
    assert al["rel2"] == ["split_join"]


def test_explain_features_unexplained_rare_word():
    """A rare extra word on one side is unexplained content; a generic one is only dropped."""
    t1, t2 = ["acme", "tools", "pvt", "ltd"], ["acme", "tolls", "private", "limited", "zorbex"]
    idf1, idf2 = [5.0, 3.0, 1.5, 1.2], [5.0, 3.5, 1.4, 1.1, 9.0]
    feats, _ = ed.explain_features(t1, t2, idf1, idf2, [False] * 4, [False] * 5)
    row = dict(zip(ed.FIELD_FEATURES, feats))
    assert row["n_exact"] == 1 and row["n_typo"] == 1 and row["n_abbrev"] == 2
    assert row["unexpl_max_idf_2"] == 9.0 and row["n_unexpl_2"] == 1 and row["core_equal"] == 0.0
    feats, _ = ed.explain_features(t1, t2, idf1, idf2, [False] * 4, [False] * 4 + [True])
    row = dict(zip(ed.FIELD_FEATURES, feats))
    assert row["n_dropped_generic"] == 1 and row["core_equal"] == 1.0 and row["unexpl_max_idf_2"] == 0.0
    assert row["order_tau"] == 1.0


def test_explain_features_empty_side():
    """An empty field gives NaN fractions and no crash."""
    feats, _ = ed.explain_features([], ["main", "st"], [], [2.0, 1.0], [], [False, False])
    row = dict(zip(ed.FIELD_FEATURES, feats))
    assert math.isnan(row["expl_frac_idf_min"]) and row["n_unexpl_2"] == 2


def test_number_features():
    """House 12 vs 12bis is base-equal; 12 vs 14 is a conflict; missing sides are coded."""
    base = {"house_number": "12", "postcodes": "411001", "addr_numbers": "12 5 411001", "name_numbers": ""}
    other = dict(base, house_number="12bis", addr_numbers="12bis 5 411001")
    row = dict(zip(ed.NUMBER_FEATURES, ed.number_features(base, other)))
    assert row["house_no_rel"] == ed.REL_BASE_EQUAL and row["any_number_conflict"] == 0.0
    assert row["postcode_rel"] == ed.REL_EQUAL and row["unit_rel"] == ed.REL_EQUAL
    other = dict(base, house_number="14", addr_numbers="14 5 411002", postcodes="411002")
    row = dict(zip(ed.NUMBER_FEATURES, ed.number_features(base, other)))
    assert row["house_no_rel"] == ed.REL_CONFLICT and row["any_number_conflict"] == 1.0
    assert row["postcode_rel"] == ed.REL_PREFIX_EQUAL
    empty = {"house_number": "", "postcodes": "", "addr_numbers": "", "name_numbers": ""}
    row = dict(zip(ed.NUMBER_FEATURES, ed.number_features(base, empty)))
    assert row["house_no_rel"] == ed.REL_ONE_MISSING and row["any_number_conflict"] == 0.0
