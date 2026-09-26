"""Unit tests for src/normalize.py (master plan SS8): accents/ligatures, initials,
numbers (incl. bis/ter and postcodes), landmarks, fold, romanisation, and a
smoke run over sample/ when it exists."""

from pathlib import Path

import pytest

from src import io_utils, normalize

SAMPLE = Path(__file__).resolve().parents[4] / "sample"


def test_accents_and_ligatures():
    """Latin accents are stripped, ligatures expanded, casefolded."""
    n, _ = normalize.basic_clean("Œuvre CÔTÉ Straße Æther")
    assert n == "oeuvre cote strasse aether"


def test_non_latin_marks_are_kept_or_romanised_not_deleted():
    """Devanagari vowel signs must not be stripped as 'accents'; the word is romanised."""
    n, rom = normalize.basic_clean("राम मार्केटिंग प्राइवेट लिमिटेड")
    assert rom is True
    assert n == "ram marketing praivet limited"


def test_romanise_conjunct_keeps_final_a_and_other_scripts():
    """Schwa deletion except after a conjunct; works for non-Devanagari Brahmic scripts."""
    assert normalize.romanize_word("शर्मा") == "sharma"
    assert normalize.romanize_word("मित्र") == "mitra"
    assert normalize.romanize_word("ಪ್ರೈವೇಟ್") == "praivet"  # Kannada
    assert normalize.romanize_word("abc") == "abc"


def test_dotted_initials_joined():
    """'S.B.I.' and 'O.D.' collapse to one token; a single 'N.' stays a letter."""
    assert normalize.basic_clean("S.B.I. Bank")[0] == "sbi bank"
    assert normalize.basic_clean("Nerissa N. Robertson, O.D.")[0] == "nerissa n robertson od"


def test_punctuation_connectors_null_and_domains():
    """Punctuation -> spaces, connectors -> '&', literal null dropped, domains reduced."""
    assert normalize.basic_clean("Smith and Sons (Pvt) Ltd.")[0] == "smith & sons pvt ltd"
    assert normalize.basic_clean("A&B-Co")[0] == "a & b co"
    assert normalize.basic_clean("2655 DE ARMOND DR, null")[0] == "2655 de armond dr"
    assert normalize.basic_clean("horneprecisionmitsubishi.com")[0] == "horneprecisionmitsubishi"
    assert normalize.basic_clean("www.abc-shop.co.in")[0] == "abc shop"
    assert normalize.basic_clean("@allenproducts")[0] == "allenproducts"


def test_numbers_bis_ter_house_number_and_postcode():
    """bis/ter glue to the number; house number = first number; postcode by shape+position."""
    rec = normalize.normalize_record("X", "12 bis Rue de la Paix, 75008 Paris")
    assert rec["house_number"] == "12bis"
    assert rec["addr_numbers"] == "12bis 75008"
    assert rec["postcodes"] == "75008"
    rec = normalize.normalize_record("X", "14-ter Main St, Town")
    assert rec["house_number"] == "14ter"
    # House number and ordinals are never postcodes.
    rec = normalize.normalize_record("X", "592 3rd Cross, 4567 Layout")
    assert rec["house_number"] == "592"
    assert rec["postcodes"] == "4567"
    rec = normalize.normalize_record("X", "#35840 CHESTER RD, AVON, OH")
    assert rec["house_number"] == "35840"
    assert rec["postcodes"] == ""


def test_name_numbers():
    """Numbers inside names go to name_numbers, separate from addresses."""
    rec = normalize.normalize_record("7-Eleven 24x7 Store", "1 Main St")
    assert rec["name_numbers"] == "7 24x7"


def test_landmark_extraction():
    """Trigger + up to 4 tokens of the segment move to `landmark`, except right after a number."""
    rec = normalize.normalize_record("X", "Fl D-703 Caserta, Abitante Nr. Crystal Hon, Pune")
    assert rec["landmark"] == "nr crystal hon"
    assert rec["norm_addr"] == "fl d 703 caserta abitante pune"
    rec = normalize.normalize_record("X", "Shop 4, opposite the big city mall road side, Delhi")
    assert rec["landmark"] == "opposite the big city mall"
    assert "side" in rec["norm_addr"]
    rec = normalize.normalize_record("X", "12 Near Road, Town")  # street named "Near Road"
    assert rec["landmark"] == ""
    assert rec["norm_addr"] == "12 near road town"


def test_fold():
    """Fold applies digraphs, vowel runs and doubled consonants; numbers untouched."""
    assert normalize.fold("philip shah") == "filip sah"
    assert normalize.fold("maarketting") == "marketing"
    assert normalize.fold("wazeer 100") == "vasir 100"


def test_empty_fields():
    """Empty or punctuation-only input gives empty text and no crash."""
    rec = normalize.normalize_record("--", "")
    assert rec["norm_name"] == "" and rec["norm_addr"] == "" and rec["house_number"] == ""


@pytest.mark.skipif(not (SAMPLE / "train" / "train_source2.tsv").exists(), reason="sample/ not built")
def test_normalize_df_on_sample():
    """normalize_df keeps row count and IDs on real sample data, and romanises some India names."""
    raw = io_utils.load_tsv(SAMPLE / "train" / "train_source2.tsv")
    out = normalize.normalize_df(raw, "S2", n_jobs=1, chunk=2000)
    assert len(out) == len(raw)
    assert (out["entity_id"].to_numpy() == raw["entity_id"].to_numpy()).all()
    assert out["name_romanized"].any()
    # Every romanised name ends up in Latin letters/digits only.
    rom = out.loc[out["name_romanized"], "norm_name"]
    assert rom.str.match(r"^[a-z0-9 &]*$").mean() > 0.95
