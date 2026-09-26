"""Tests for src/collective.py."""

import numpy as np

from src import collective


def test_entity_features():
    """Rank, gap to top, count above 0.5, top margin and share within each S1."""
    s1 = np.array(["a", "a", "a", "b"])
    p = np.array([0.8, 0.9, 0.1, 0.4])
    f = collective.entity_features(s1, p)
    col = {k: f[:, i] for i, k in enumerate(collective.ENTITY_FEATURES)}
    assert col["rank_in_entity"].tolist() == [1, 0, 2, 0]
    assert np.allclose(col["gap_to_top"], [-0.1, 0, -0.8, 0])
    assert col["n_above_05"].tolist() == [2, 2, 2, 0]
    assert np.allclose(col["top_margin"], [0.1, 0.1, 0.1, 0.4])
    assert np.allclose(col["p_share"][:3].sum(), 1.0)


def test_sibling_features():
    """A candidate similar to a confident sibling gets high sibling similarity; self is excluded."""
    s1 = np.array(["a", "a", "a", "b", "b"])
    p = np.array([0.95, 0.3, 0.2, 0.9, 0.1])
    name = np.array(["acme tools", "acme tool", "zeta foods", "beta", "gamma"], dtype=object)
    addr = np.array(["12 main st", "12 main st", "9 oak rd", "1 x", ""], dtype=object)
    house = np.array(["12", "12", "9", "1", ""], dtype=object)
    post = np.array(["", "", "", "", ""], dtype=object)
    f = collective.sibling_features(s1, p, name, addr, house, post)
    col = {k: f[:, i] for i, k in enumerate(collective.SIBLING_FEATURES)}
    assert col["sib_count"].tolist() == [0, 1, 1, 0, 1]
    assert np.isnan(col["sib_max_name_sim"][0])  # the only confident row has no other confident sibling
    assert col["sib_max_name_sim"][1] > 90 and col["sib_max_name_sim"][2] < 50
    assert col["sib_number_agree"][1] == 1 and col["sib_number_agree"][2] == 0
