"""Pre-flight checks for output/matching_results.tsv and output/candidate_pairs.tsv.

Runs the assertions from master plan SS22.2 *before* the official validator,
so a bug is caught with a clear message instead of a generic FAIL. This does
not replace `utils/validate_submission.py` — always run both (CLAUDE.md SS9).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import io_utils


def _read_result_tsv(path: Path) -> dict[str, set[str]]:
    """Read a matching/candidate-style TSV into {s1_id: set(ids)}.

    Inputs: path - the TSV to read.
    Outputs: dict mapping every row's S1 ID to its (possibly empty) ID set.
    """
    mapping: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            s1, _, rest = line.partition("\t")
            mapping[s1] = set(rest.split(",")) if rest else set()
    return mapping


def check_outputs(out_dir: Path, test_dir: Path) -> list[str]:
    """Run every pre-flight assertion; return a list of problem descriptions (empty = PASS).

    Inputs: out_dir - folder with matching_results.tsv (+ candidate_pairs.tsv);
            test_dir - folder with test_source1/2/3.tsv.
    Outputs: list of human-readable problems. Empty list means all checks passed.
    """
    problems: list[str] = []

    matching_path = out_dir / "matching_results.tsv"
    candidate_path = out_dir / "candidate_pairs.tsv"
    s1_path = test_dir / "test_source1.tsv"

    if not matching_path.exists():
        return [f"missing {matching_path}"]
    if not s1_path.exists():
        return [f"missing {s1_path}"]

    s1_df = io_utils.load_tsv(s1_path)
    required_ids = set(s1_df["entity_id"])

    valid_match_ids: set[str] = set()
    for name in ("test_source2.tsv", "test_source3.tsv"):
        p = test_dir / name
        if p.exists():
            valid_match_ids |= set(io_utils.load_tsv(p)["entity_id"])

    per_country = dict(zip(s1_df["entity_id"], s1_df["country"]))

    matched = _read_result_tsv(matching_path)
    seen_ids = list(matched.keys())

    if len(seen_ids) != len(set(seen_ids)):
        dupes = {i for i in seen_ids if seen_ids.count(i) > 1}
        problems.append(f"duplicate S1 rows in matching_results.tsv: {sorted(dupes)[:5]}")

    missing = required_ids - set(matched)
    if missing:
        problems.append(f"{len(missing)} required S1 entities missing, e.g. {sorted(missing)[:5]}")

    extra = set(matched) - required_ids
    if extra:
        problems.append(f"{len(extra)} rows use an S1 ID not in the test set, e.g. {sorted(extra)[:5]}")

    for s1, ids in matched.items():
        for mid in ids:
            if not (mid.startswith("S2-") or mid.startswith("S3-")):
                problems.append(f"{s1}: id {mid!r} has no S2-/S3- prefix")
            elif valid_match_ids and mid not in valid_match_ids:
                problems.append(f"{s1}: id {mid!r} does not exist in the test set")

    candidate = None
    if candidate_path.exists():
        candidate = _read_result_tsv(candidate_path)
        for s1, ids in matched.items():
            cand_ids = candidate.get(s1, set())
            if ids - cand_ids:
                problems.append(f"{s1}: matched ids not in its candidate list: {sorted(ids - cand_ids)[:5]}")

    # France / per-country presence: output must cover every country present in test_source1.
    output_countries = {per_country[s1] for s1 in matched if s1 in per_country}
    missing_countries = set(per_country.values()) - output_countries
    if missing_countries:
        problems.append(f"countries missing from output entirely: {sorted(missing_countries)}")

    return problems


def main() -> None:
    """CLI: `python -m src.check_outputs --out-dir output --test-dir student_resource/dataset/test`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--test-dir", default="../../student_resource/dataset/test")
    args = parser.parse_args()

    problems = check_outputs(Path(args.out_dir), Path(args.test_dir))
    if problems:
        print(f"FAIL — {len(problems)} issue(s):")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
        sys.exit(1)
    print("PASS — check_outputs found no issues.")


if __name__ == "__main__":
    main()
