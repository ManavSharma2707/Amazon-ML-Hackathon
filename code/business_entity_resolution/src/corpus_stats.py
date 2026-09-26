"""Step 3 — corpus statistics: rarity (IDF), suffix-likeness, street-type-likeness.

Master plan SS9. Everything is computed from token counts of the records of
the current run (no labels): train stats from train files, test stats from
test files (`stats_mode: per_run`). If the organisers rule out test-time
statistics (memory.md Q3), `stats_mode: train_only` makes test reuse the
train table (plan SS9.4).

Counts are kept per country *label as found in the data* (open set, never a
hard-coded value, CLAUDE.md SS3 rule 5); global counts are the sum.

Table layout (`count_tokens` output, long format):
    field     - which text field ("norm_name", "fold_name", "norm_addr", "fold_addr")
    country   - country label, or "" for the global row
    token     - the token
    df        - number of records containing the token
    pos_count - name fields: occurrences in the last 2 positions (suffix
                position); address fields: occurrences 1-3 tokens after the
                house number (street-type position)
    n_docs    - records in that (field, country) group (repeated per row)
Tokens with global df < 2 are dropped to save space; a missing token is
treated as df = 1 by the lookup helpers.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

NAME_FIELDS = ("norm_name", "fold_name")
ADDR_FIELDS = ("norm_addr", "fold_addr")
_TYPE_WINDOW = 3  # street-type position = 1..3 tokens after the house number


def _explode_tokens(texts: pd.Series) -> pd.DataFrame:
    """Split texts into one row per token with its position and the text length.

    Inputs: texts - Series of space-separated token strings (any index).
    Outputs: DataFrame with columns row (int position of the record), token,
    pos (0-based token position), n_tok (tokens in that record).
    """
    lists = texts.reset_index(drop=True).str.split()
    lens = lists.str.len().fillna(0).astype(np.int32)
    ex = lists.explode()
    ex = ex[ex.notna()]
    rows = ex.index.to_numpy()
    out = pd.DataFrame({"row": rows, "token": ex.to_numpy()})
    out["pos"] = out.groupby("row").cumcount().astype(np.int32)
    out["n_tok"] = lens.to_numpy()[rows]
    return out


def count_field(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """Count df and positional occurrences of every token of one field, per country.

    Inputs: df - normalised records (needs `country`, `field`, and for address
            fields also `house_number`); field - column name.
    Outputs: DataFrame (country, token, df, pos_count) for this frame.
    """
    ex = _explode_tokens(df[field])
    ex["country"] = df["country"].to_numpy()[ex["row"].to_numpy()]
    if field in NAME_FIELDS:
        pos_hit = ex["pos"] >= ex["n_tok"] - 2
    else:
        # First digit-bearing token's position per record = house number position.
        is_num = ex["token"].str.contains(r"\d", regex=True)
        first_num = ex[is_num].groupby("row")["pos"].min()
        hpos = ex["row"].map(first_num)
        delta = ex["pos"] - hpos
        pos_hit = (delta >= 1) & (delta <= _TYPE_WINDOW) & ~is_num
    ex["pos_hit"] = pos_hit.fillna(False).astype(np.int32)
    # df counts a token once per record; pos_count counts every positional hit.
    dedup = ex.drop_duplicates(["row", "token"])
    dfc = dedup.groupby(["country", "token"], sort=False).size().rename("df")
    posc = ex.groupby(["country", "token"], sort=False)["pos_hit"].sum().rename("pos_count")
    return pd.concat([dfc, posc], axis=1).reset_index()


class TokenCounter:
    """Incremental version of the token-statistics count (one frame at a time).

    Lets NB02 count one source (millions of rows) and free it before loading
    the next, instead of holding all sources in memory. Call `add(frame)` per
    record frame, then `finalize()` for the long table (module docstring).
    """

    def __init__(self, fields=NAME_FIELDS + ADDR_FIELDS, min_df: int = 2):
        """Inputs: fields - text fields to count; min_df - global df cut-off."""
        self.fields = tuple(fields)
        self.min_df = min_df
        self.acc: dict[str, pd.DataFrame | None] = {f: None for f in self.fields}
        self.n_docs: dict[str, int] = {}

    def add(self, frame: pd.DataFrame) -> None:
        """Add the counts of one normalised record frame (needs country + fields)."""
        for field in self.fields:
            c = count_field(frame, field)
            prev = self.acc[field]
            self.acc[field] = (
                c if prev is None
                else pd.concat([prev, c]).groupby(["country", "token"], sort=False).sum().reset_index()
            )
        for k, v in frame["country"].value_counts().items():
            self.n_docs[k] = self.n_docs.get(k, 0) + int(v)

    def finalize(self) -> pd.DataFrame:
        """Return the long table (field, country, token, df, pos_count, n_docs)."""
        n_docs = dict(self.n_docs)
        n_docs[""] = sum(self.n_docs.values())
        parts = []
        for field in self.fields:
            acc = self.acc[field]
            glob = acc.groupby("token", sort=False)[["df", "pos_count"]].sum().reset_index()
            glob = glob[glob["df"] >= self.min_df]
            glob["country"] = ""
            local = acc[acc["token"].isin(set(glob["token"]))]
            tab = pd.concat([glob, local], ignore_index=True)
            tab["n_docs"] = tab["country"].map(n_docs).astype(np.int64)
            tab["field"] = field
            parts.append(tab[["field", "country", "token", "df", "pos_count", "n_docs"]])
            print(f"    stats {field}: {len(glob):,} tokens with df>={self.min_df}", flush=True)
        return pd.concat(parts, ignore_index=True)


def count_tokens(frames: list[pd.DataFrame], fields=NAME_FIELDS + ADDR_FIELDS, min_df: int = 2) -> pd.DataFrame:
    """Build the long-format token statistics table over several record frames.

    Inputs: frames - normalised record frames of one run (e.g. S1, S2, S3 of
            train); fields - text fields to count; min_df - drop tokens whose
            global df is below this.
    Outputs: long table (field, country, token, df, pos_count, n_docs); see
    module docstring. Global rows have country "".
    """
    counter = TokenCounter(fields, min_df)
    for f in frames:
        counter.add(f)
    return counter.finalize()


def idf(df: np.ndarray | float, n_docs: int) -> np.ndarray | float:
    """Smoothed IDF: log((N + 1) / (df + 1)) + 1 (master plan SS9.1)."""
    return np.log((n_docs + 1) / (np.asarray(df) + 1)) + 1


def idf_lookup(stats: pd.DataFrame, field: str, country: str = "") -> tuple[dict[str, float], float]:
    """Token -> IDF dict for one field and country group ("" = global).

    Inputs: stats - `count_tokens` table; field; country.
    Outputs: (dict token -> idf, default idf for an unseen token, i.e. df = 1).
    """
    sub = stats[(stats["field"] == field) & (stats["country"] == country)]
    if sub.empty:
        return {}, 1.0
    n = int(sub["n_docs"].iloc[0])
    return dict(zip(sub["token"], idf(sub["df"].to_numpy(), n))), float(idf(1, n))


def _likeness(sub: pd.DataFrame) -> pd.Series:
    """pos_ratio x sigmoid(z-scored log frequency), shared by suffix/street-type scores.

    Inputs: sub - global rows of one field (token, df, pos_count, n_docs).
    Outputs: Series of scores in [0, 1] indexed like `sub`.
    """
    ratio = (sub["pos_count"] / sub["df"].clip(lower=1)).clip(upper=1.0)
    logf = np.log(sub["df"] / sub["n_docs"])
    mu, sd = float(logf.mean()), float(logf.std() or 1.0)
    return ratio * (1.0 / (1.0 + np.exp(-(logf - mu) / sd)))


def suffix_likeness(stats: pd.DataFrame, field: str = "norm_name") -> dict[str, float]:
    """Suffix-likeness per name token: end-position ratio x frequency sigmoid (plan SS9.2).

    High for tokens that are frequent AND almost always at the end of a name
    (legal forms, generic end words). Inputs: stats table; name field.
    Outputs: dict token -> score in [0, 1].
    """
    sub = stats[(stats["field"] == field) & (stats["country"] == "")]
    return dict(zip(sub["token"], _likeness(sub)))


def street_type_likeness(stats: pd.DataFrame, field: str = "norm_addr") -> dict[str, float]:
    """Street-type-likeness per address token (plan SS9.3), same formula as suffixes.

    Inputs: stats table; address field. Outputs: dict token -> score in [0, 1].
    """
    sub = stats[(stats["field"] == field) & (stats["country"] == "")]
    return dict(zip(sub["token"], _likeness(sub)))


def top_tokens(stats: pd.DataFrame, field: str, n: int = 50, by: str = "df") -> list[tuple[str, float]]:
    """Top-n global tokens of a field by df (or by a likeness score), for sanity printouts.

    Inputs: stats; field; n; by - "df", "suffix" or "street".
    Outputs: list of (token, value).
    """
    if by == "df":
        sub = stats[(stats["field"] == field) & (stats["country"] == "")].nlargest(n, "df")
        return list(zip(sub["token"], sub["df"].astype(float)))
    scores = suffix_likeness(stats, field) if by == "suffix" else street_type_likeness(stats, field)
    return sorted(scores.items(), key=lambda kv: -kv[1])[:n]


def main() -> None:
    """Smoke test: count tokens of a few hand-made records and print top suffixes."""
    from . import normalize

    raw = pd.DataFrame(
        {
            "entity_id": [f"S1-{i}" for i in range(4)],
            "business_name": ["Alpha Pvt Ltd", "Beta Pvt Ltd", "Gamma Ltd", "Delta Inc"],
            "business_address": ["1 Main Road, X", "2 Hill Road, Y", "3 Main Street, Z", "4 Oak Street, W"],
            "country": ["C1", "C1", "C2", "C2"],
        }
    )
    stats = count_tokens([normalize.normalize_df(raw, "S1")], min_df=1)
    print(top_tokens(stats, "norm_name", 5, by="suffix"))
    print(top_tokens(stats, "norm_addr", 5, by="street"))
    print(math.isfinite(idf(1, 4)))


if __name__ == "__main__":
    main()
