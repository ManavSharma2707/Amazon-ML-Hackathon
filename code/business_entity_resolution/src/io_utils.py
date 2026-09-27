"""Safe I/O for the Business Entity Resolution pipeline.

Centralises TSV loading/writing (per CLAUDE.md SS5) and the sanity assertions
from the master plan SS6, so every other module reads/writes data the same
way. Also holds `load_config` and `set_seeds`, used by every module's
`main()`.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

EXPECTED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def load_tsv(path: str | Path) -> pd.DataFrame:
    """Load a source TSV (S1/S2/S3) the safe way.

    Inputs: path - path to a tab-separated file with a header row.
    Outputs: DataFrame with all columns read as `str`, no NA coercion, and no
    quote-based line-swallowing (business names/addresses can contain `"`).
    Raises: AssertionError if the row count doesn't match the file's line
    count, or if a UTF-8 BOM was not stripped correctly.
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],
        quoting=csv.QUOTE_NONE,
        encoding="utf-8-sig",  # strips a leading BOM if present, per master plan SS6.2
    )
    with open(path, encoding="utf-8-sig") as f:
        n_lines = sum(1 for _ in f)
    assert len(df) == n_lines - 1, (
        f"{path}: row count {len(df)} != file line count - 1 ({n_lines - 1}); "
        "a stray quote or line ending likely corrupted the parse."
    )
    return df


def assert_source_columns(df: pd.DataFrame, path: str | Path = "") -> None:
    """Fail loudly if a source DataFrame doesn't have exactly the expected columns.

    Inputs: df - a source1/2/3 DataFrame; path - for the error message only.
    Outputs: None. Raises AssertionError on mismatch.
    """
    assert list(df.columns) == EXPECTED_SOURCE_COLUMNS, (
        f"{path}: unexpected columns {list(df.columns)}, expected {EXPECTED_SOURCE_COLUMNS}"
    )


def assert_unique_ids(df: pd.DataFrame, prefix: str, path: str | Path = "") -> None:
    """Fail loudly if `entity_id` is not unique or doesn't carry the expected prefix.

    Inputs: df - a source DataFrame with an `entity_id` column; prefix - e.g. "S1-";
            path - for the error message only.
    Outputs: None. Raises AssertionError on duplicate IDs or a wrong/missing prefix.
    """
    ids = df["entity_id"]
    dupes = ids[ids.duplicated()].unique()
    assert len(dupes) == 0, f"{path}: duplicate entity_id(s): {list(dupes)[:5]}"
    bad_prefix = ids[~ids.str.startswith(prefix)]
    assert bad_prefix.empty, (
        f"{path}: {len(bad_prefix)} entity_id(s) missing prefix {prefix!r}, "
        f"e.g. {bad_prefix.iloc[:5].tolist()}"
    )


def load_ground_truth(path: str | Path, valid_ids: set[str] | None = None) -> dict[str, set[str]]:
    """Parse `train_ground_truth.tsv` into `{s1_entity_id: set(matched_ids)}`.

    Inputs: path - path to the ground-truth TSV; valid_ids - optional set of
            known S2/S3 IDs to validate against.
    Outputs: dict mapping every S1 ID in the file to its (possibly empty) match set.
    Raises: AssertionError if a listed match ID is not in `valid_ids` (when given).
    """
    df = load_tsv(path)
    assert list(df.columns) == GT_COLUMNS, f"{path}: unexpected columns {list(df.columns)}"
    result: dict[str, set[str]] = {}
    for s1, matches in zip(df["source1_entity_id"], df["matched_entity_ids"]):
        ids = set(matches.split(",")) if matches else set()
        if valid_ids is not None and ids:
            unknown = ids - valid_ids
            assert not unknown, f"{path}: {s1} references unknown match IDs: {sorted(unknown)[:5]}"
        result[s1] = ids
    return result


def write_id_list_tsv(
    path: str | Path,
    rows: Iterable[tuple[str, Iterable[str]]],
    header: tuple[str, str] = ("source1_entity_id", "matched_entity_ids"),
) -> None:
    """Write a matching/candidate-style TSV exactly the way the validator expects.

    Inputs: path - output path; rows - iterable of (s1_entity_id, ids) pairs, one
            row per S1 entity, in the order to write; header - the two column names.
    Outputs: None. Writes with `newline="\\n"`, no quoting, and an empty field
    (not "None" or "[]") for entities with no matches.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for s1, ids in rows:
            id_list = sorted(ids) if not isinstance(ids, (list, tuple)) else list(ids)
            f.write(f"{s1}\t{','.join(id_list)}\n")


def _arrow_strings(t):
    """pyarrow types_mapper: string columns stay Arrow-backed; other types convert as usual."""
    import pyarrow as pa

    return pd.ArrowDtype(t) if pa.types.is_string(t) or pa.types.is_large_string(t) else None


def read_parquet_compact(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read Parquet with Arrow-backed string columns (no per-value Python objects).

    10M-row record tables then take a few GB instead of ~10 GB, and forked
    workers do not copy-on-write them. Numeric columns stay NumPy (NaN intact).
    """
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=columns).to_pandas(types_mapper=_arrow_strings)


def iter_parquet_compact(path: str | Path, batch_rows: int, columns: list[str] | None = None):
    """Yield DataFrames of up to `batch_rows` rows (Arrow-backed strings) from a Parquet file."""
    import pyarrow.parquet as pq

    for rb in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=columns):
        yield rb.to_pandas(types_mapper=_arrow_strings)


def pairs_to_lists(s1_ids, cand_ids) -> dict[str, list[str]]:
    """{s1_id: sorted unique candidate IDs} from two aligned ID sequences.

    Vectorised (factorize + lexsort), no pandas groupby-to-list: aggregating to
    Python lists fails on Arrow-backed string columns in pandas 2.2 (NB09a v1 crash).
    """
    import numpy as np

    s_codes, s_uni = pd.factorize(np.asarray(s1_ids, dtype=object))
    c_codes, c_uni = pd.factorize(np.asarray(cand_ids, dtype=object), sort=True)
    if not len(s_codes):
        return {}
    order = np.lexsort((c_codes, s_codes))
    sc, cc = s_codes[order], c_codes[order]
    keep = np.r_[True, (sc[1:] != sc[:-1]) | (cc[1:] != cc[:-1])]  # drop duplicate pairs
    sc, cc = sc[keep], cc[keep]
    starts = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1]])
    ends = np.r_[starts[1:], len(sc)]
    names = c_uni.astype(object)
    return {str(s_uni[sc[a_]]): [str(x) for x in names[cc[a_:b_]]] for a_, b_ in zip(starts, ends)}


def load_config(path: str | Path) -> dict:
    """Load a YAML config file.

    Inputs: path - path to a YAML file (e.g. src/configs/default.yaml).
    Outputs: parsed config as a dict.
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seeds(seed: int) -> None:
    """Seed every RNG this pipeline touches (Python, NumPy, and torch/LightGBM if importable).

    Inputs: seed - the config's `seed` value.
    Outputs: None (side effect: global RNG state).
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main() -> None:
    """Smoke-test: load the default config and print the resolved seed."""
    import sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "src/configs/default.yaml"
    cfg = load_config(cfg_path)
    set_seeds(cfg["seed"])
    print(f"Loaded config from {cfg_path}; seed={cfg['seed']}")


if __name__ == "__main__":
    main()
