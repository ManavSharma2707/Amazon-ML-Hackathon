"""Scrambled-letter transform (master plan SS21): a random a-z permutation applied
identically to every record, digits and spaces unchanged.

Abbreviation, typo and reorder relations survive the transform while the
vocabulary is destroyed, so a model that still matches well has learned
structure rather than US/India words (proxy for the unseen country).
"""

from __future__ import annotations

import random
import string

from . import normalize


def make_scrambler(seed: int) -> dict[int, int]:
    """Build a str.translate table for one random a-z permutation (same for upper case).

    Inputs: seed. Outputs: translation table usable with `str.translate`.
    """
    rng = random.Random(seed)
    perm = list(string.ascii_lowercase)
    rng.shuffle(perm)
    table = {ord(a): ord(b) for a, b in zip(string.ascii_lowercase, perm)}
    table.update({ord(a.upper()): ord(b.upper()) for a, b in zip(string.ascii_lowercase, perm)})
    return table


def scramble_texts(texts, table: dict[int, int]) -> list[str]:
    """Apply a scrambler table to every text. Inputs: iterable of str; table. Outputs: list."""
    return [t.translate(table) for t in texts]


def scramble_df(df, table: dict[int, int], cols=("business_name", "business_address")):
    """Return a copy of a record frame with the text columns scrambled."""
    out = df.copy()
    for c in cols:
        if c in out:
            out[c] = scramble_texts(out[c].tolist(), table)
    return out


NORM_FIELDS = ["norm_name", "norm_addr", "landmark", "postcodes", "house_number", "addr_numbers", "name_numbers"]


def scramble_normalized(df, table: dict[int, int]):
    """Scramble already-normalised record fields and recompute the phonetic folds on the result.

    Scrambling after normalisation keeps romanised native-script names consistent
    with their Latin counterparts (both sides get the same permutation).
    Inputs: record frame with NORM_FIELDS; table. Outputs: scrambled copy.
    """
    out = df.copy()
    for c in NORM_FIELDS:
        if c in out:
            out[c] = [str(x).translate(table) for x in df[c].tolist()]
    out["fold_name"] = [normalize.fold(x) for x in out["norm_name"].tolist()]
    out["fold_addr"] = [normalize.fold(x) for x in out["norm_addr"].tolist()]
    return out


def main() -> None:
    """Smoke test."""
    t = make_scrambler(42)
    print("Main St 12".translate(t))


if __name__ == "__main__":
    main()
