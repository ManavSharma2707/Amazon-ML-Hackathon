"""Unit tests for src/translit.py: script detection, dictionary learning, application."""

import pandas as pd

from src import translit

HI_FUTURE = "फ्यूचर"  # Devanagari, phonetic "phyuchar"
HI_TECH = "टेक्नोलॉजी"  # Devanagari, phonetic "teknolaji"
HI_PVT = "प्रा"  # Devanagari, "pra" (used for "Pvt"/"Private" in Indian filings)


def test_script_of_token():
    assert translit.script_of_token("future") == "LATIN"
    assert translit.script_of_token("123") == "LATIN"
    assert translit.script_of_token(HI_FUTURE) == "DEVANAGARI"
    assert translit.script_of_token("future" + HI_FUTURE) == "DEVANAGARI"  # mixed: non-Latin dominates


def test_has_non_latin():
    assert not translit.has_non_latin("future technology pvt ltd")
    assert translit.has_non_latin(f"{HI_FUTURE} {HI_TECH}")


def test_clean_tokens_strips_domain_and_null():
    assert translit.clean_tokens("www.Acme.com") == ["acme"]
    assert translit.clean_tokens("Acme NA Corp") == ["acme", "na", "corp"]  # NA is a real business word, not dropped
    assert translit.clean_tokens("Acme null Corp") == ["acme", "corp"]  # "null" literal marker is dropped


def test_build_dictionary_learns_distinct_mappings_from_varied_contexts():
    """Two Devanagari tokens each pair with a distinct Latin token across different
    company names; the shared co-occurrence should not force a wrong merge."""
    pairs = [
        ([HI_FUTURE, HI_PVT], ["future", "holdings", "pvt", "ltd"]),
        ([HI_FUTURE, HI_PVT], ["future", "traders", "pvt", "ltd"]),
        ([HI_FUTURE, HI_PVT], ["future", "logistics", "pvt", "ltd"]),
        ([HI_TECH, HI_PVT], ["technology", "hub", "pvt", "ltd"]),
        ([HI_TECH, HI_PVT], ["technology", "corp", "pvt", "ltd"]),
        ([HI_TECH, HI_PVT], ["technology", "systems", "pvt", "ltd"]),
    ]
    d = translit.build_dictionary(pairs, min_count=3, min_dice=0.5)
    assert d[HI_FUTURE] == "future"
    assert d[HI_TECH] == "technology"
    # HI_PVT co-occurs equally with "pvt" and "ltd" (both constant across all 6
    # pairs) -- either is a valid, perfect-Dice pick; Python's per-run string
    # hash randomisation decides the tie, so don't assert a specific winner.
    assert d[HI_PVT] in ("pvt", "ltd")


def test_build_dictionary_respects_min_count():
    pairs = [([HI_FUTURE], ["future"])] * 2  # below min_count=3
    d = translit.build_dictionary(pairs, min_count=3, min_dice=0.1)
    assert HI_FUTURE not in d


def test_transliterate_token_dictionary_hit_and_fallback():
    d = {HI_FUTURE: "future"}
    assert translit.transliterate_token(HI_FUTURE, d) == "future"
    assert translit.transliterate_token("future", d) == "future"  # Latin passthrough
    # No dictionary entry -> anyascii fallback (some non-empty ASCII string, not a crash).
    out = translit.transliterate_token(HI_TECH, d)
    assert out.isascii() and out


def test_translit_field_latin_text_is_stable_and_folded():
    from src import normalize

    d: dict = {}
    out = translit.translit_field("Future Technology Pvt Ltd", d)
    assert out == normalize.fold(" ".join(normalize.basic_clean("Future Technology Pvt Ltd")[0].split()))


def test_add_translit_columns_shapes():
    df = pd.DataFrame({"raw_name": ["Acme Corp", HI_FUTURE], "raw_addr": ["1 Main St", "2 Oak Rd"]})
    d = {HI_FUTURE: "future"}
    out = translit.add_translit_columns(df.copy(), d, n_jobs=1)
    assert list(out.columns) >= ["translit_name", "translit_addr"] or {"translit_name", "translit_addr"} <= set(out.columns)
    assert out.loc[1, "translit_name"] == "future"


def test_build_dictionary_from_pairs_end_to_end():
    s1 = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2", "S1-3"],
        "norm_name": ["future holdings", "future traders", "future logistics"],
        "norm_addr": ["1 main st", "2 oak rd", "3 elm ave"],
    })
    pool = pd.DataFrame({
        "entity_id": ["S2-1", "S2-2", "S2-3"],
        "raw_name": [HI_FUTURE + " holdings", HI_FUTURE + " traders", HI_FUTURE + " logistics"],
        "raw_addr": ["1 main st", "2 oak rd", "3 elm ave"],
    })
    pairs = pd.DataFrame({"s1_id": ["S1-1", "S1-2", "S1-3"], "match_id": ["S2-1", "S2-2", "S2-3"]})
    d = translit.build_dictionary_from_pairs(s1, pool, pairs, {"S1-1", "S1-2", "S1-3"}, min_count=3, min_dice=0.5)
    assert d.get(HI_FUTURE) == "future"


def test_save_load_dictionary_roundtrip(tmp_path):
    d = {HI_FUTURE: "future", HI_TECH: "technology"}
    p = tmp_path / "dict.tsv"
    translit.save_dictionary(d, str(p))
    assert translit.load_dictionary(str(p)) == d
