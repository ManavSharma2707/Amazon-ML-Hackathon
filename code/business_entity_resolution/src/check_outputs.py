"""Pre-flight checks for output/matching_results.tsv and output/candidate_pairs.tsv.

Runs the assertions from master plan SS22.2 *before* the official validator,
so a bug is caught with a clear message instead of a generic FAIL. This does
not replace `utils/validate_submission.py` — always run both (CLAUDE.md SS9).

Memory-lean by design (the laptop has < 1 GB free): every file is streamed
line by line, IDs are encoded as int64 in NumPy arrays (prefix digit x 1e12 +
number) instead of Python string sets, and the candidate file is checked
line-aligned with the matching file (dict fallback only for rows whose order
differs).
"""

from __future__ import annotations

import argparse
import re
import sys
from array import array
from pathlib import Path

import numpy as np

_ID_RE = re.compile(r"^S([123])-(\d{1,12})$")
_BASE = 10**12
MAX_PROBLEMS = 50


def encode_id(eid: str) -> int:
    """'S2-888394598' -> 2 * 1e12 + 888394598; -1 if the ID does not have the S<k>-<digits> shape."""
    m = _ID_RE.match(eid)
    return int(m.group(1)) * _BASE + int(m.group(2)) if m else -1


def _first_column_codes(path: Path) -> np.ndarray:
    """Encoded IDs of the first column of a source TSV (header skipped), streamed."""
    out = array("q")
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            eid = line.split("\t", 1)[0].rstrip("\n")
            if eid:
                out.append(encode_id(eid))
    return np.frombuffer(out, dtype=np.int64) if len(out) else np.empty(0, np.int64)


def _s1_countries(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(encoded S1 IDs, country code per row, country names) from test_source1.tsv, streamed."""
    codes, cty = array("q"), array("i")
    names: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        header = next(f).rstrip("\n").split("\t")
        ci = header.index("country")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts[0]:
                continue
            codes.append(encode_id(parts[0]))
            cty.append(names.setdefault(parts[ci] if ci < len(parts) else "", len(names)))
    inv = sorted(names, key=names.get)
    return np.frombuffer(codes, dtype=np.int64), np.frombuffer(cty, dtype=np.int32), inv


def _parse_row(line: str, lineno: int, name: str, problems: list[str]) -> tuple[str, list[str]]:
    """Split one output row into (s1_id, ids) and record format problems (spaces, quotes, dupes, prefixes)."""
    line = line.rstrip("\n")
    if "\r" in line:
        problems.append(f"{name}:{lineno}: carriage return (Windows line ending)")
        line = line.rstrip("\r")
    if line.count("\t") != 1:
        problems.append(f"{name}:{lineno}: expected exactly one tab, got {line.count(chr(9))}")
    s1, _, rest = line.partition("\t")
    if " " in line or '"' in line or "'" in line:
        problems.append(f"{name}:{lineno}: spaces or quotes in the row")
    ids = rest.split(",") if rest else []
    if any(not i for i in ids):
        problems.append(f"{name}:{lineno}: empty ID in the list (stray comma)")
    if len(set(ids)) != len(ids):
        problems.append(f"{name}:{lineno}: duplicate IDs in the list")
    for i in ids:
        if not (i.startswith("S2-") or i.startswith("S3-")):
            problems.append(f"{name}:{lineno}: id {i!r} has no S2-/S3- prefix")
            break
    return s1, ids


def check_outputs(out_dir: Path, test_dir: Path, check_ids: bool = True) -> list[str]:
    """Run every pre-flight assertion; return a list of problem descriptions (empty = PASS).

    Inputs: out_dir - folder with matching_results.tsv (+ candidate_pairs.tsv);
            test_dir - folder with test_source1/2/3.tsv; check_ids - also
            verify that every matched/candidate ID exists in test S2/S3.
    Outputs: list of human-readable problems (capped at MAX_PROBLEMS).
    """
    problems: list[str] = []
    matching_path, candidate_path = out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv"
    s1_path = test_dir / "test_source1.tsv"
    for p in (matching_path, s1_path):
        if not p.exists():
            return [f"missing {p}"]
    req, req_cty, cty_names = _s1_countries(s1_path)
    if (req < 0).any():
        problems.append("test_source1.tsv has IDs that are not S1-<digits> (encoder assumption broken)")

    has_cand = candidate_path.exists()
    if not has_cand:
        problems.append(f"missing {candidate_path} (required in the final zip)")
    rows_s1 = array("q")
    n_pred = array("i")
    matched_codes, cand_codes = array("q"), array("q")
    pending_m: dict[str, list[str]] = {}
    pending_c: dict[str, list[str]] = {}
    with open(matching_path, encoding="utf-8", newline="") as fm:
        fc = open(candidate_path, encoding="utf-8", newline="") if has_cand else None
        try:
            hm = next(fm).rstrip("\r\n")
            if hm != "source1_entity_id\tmatched_entity_ids":
                problems.append(f"matching header is {hm!r}")
            if fc:
                hc = next(fc).rstrip("\r\n")
                if hc != "source1_entity_id\tcandidate_entity_ids":
                    problems.append(f"candidate header is {hc!r}")
            for lineno, line in enumerate(fm, start=2):
                s1, ids = _parse_row(line, lineno, "matching", problems)
                rows_s1.append(encode_id(s1))
                n_pred.append(len(ids))
                for i in ids:
                    matched_codes.append(encode_id(i))
                if fc is not None:
                    cline = fc.readline()
                    if not cline:
                        pending_m[s1] = ids
                        continue
                    cs1, cids = _parse_row(cline, lineno, "candidate", problems)
                    for i in cids:
                        cand_codes.append(encode_id(i))
                    if cs1 == s1:
                        extra = set(ids) - set(cids)
                        if extra:
                            problems.append(f"{s1}: matched ids not in its candidate list: {sorted(extra)[:5]}")
                    else:  # different row order: resolve after the pass
                        pending_m[s1] = ids
                        pending_c[cs1] = cids
                if len(problems) > MAX_PROBLEMS:
                    break
            if fc is not None:
                for lineno, cline in enumerate(fc, start=10**9):
                    cs1, cids = _parse_row(cline, lineno, "candidate", problems)
                    pending_c[cs1] = cids
                    for i in cids:
                        cand_codes.append(encode_id(i))
        finally:
            if fc:
                fc.close()
    for s1, ids in pending_m.items():
        cids = pending_c.get(s1)
        if cids is None:
            problems.append(f"{s1}: row missing from candidate_pairs.tsv")
        elif set(ids) - set(cids):
            problems.append(f"{s1}: matched ids not in its candidate list: {sorted(set(ids) - set(cids))[:5]}")

    rows = np.frombuffer(rows_s1, dtype=np.int64)
    uniq, counts = np.unique(rows, return_counts=True)
    if (counts > 1).any():
        problems.append(f"{int((counts > 1).sum())} duplicate S1 rows in matching_results.tsv")
    missing = np.setdiff1d(req, uniq)
    if len(missing):
        problems.append(f"{len(missing)} required S1 entities missing from matching_results.tsv")
    extra = np.setdiff1d(uniq, req)
    if len(extra):
        problems.append(f"{len(extra)} rows use an S1 ID not in the test set")
    if check_ids:
        valid = np.unique(np.concatenate([_first_column_codes(test_dir / f"test_source{s}.tsv") for s in (2, 3)
                                          if (test_dir / f"test_source{s}.tsv").exists()]))
        for name, codes in (("matched", matched_codes), ("candidate", cand_codes)):
            arr = np.frombuffer(codes, dtype=np.int64) if len(codes) else np.empty(0, np.int64)
            bad = int((~np.isin(arr, valid)).sum()) if len(valid) else 0
            if bad:
                problems.append(f"{bad} {name} IDs do not exist in test_source2/3")
    # France / per-country presence: every country of test_source1 must appear in the output.
    present = np.isin(req, uniq)
    for k, name in enumerate(cty_names):
        m = req_cty == k
        if m.any() and not present[m].all():
            problems.append(f"country {name!r}: {int((~present[m]).sum())} S1 rows missing")
    return problems[:MAX_PROBLEMS]


def output_summary(out_dir: Path, test_dir: Path) -> dict:
    """Per-country predicted-empty rate and mean predicted set size (distribution sanity, plan SS22.2)."""
    req, req_cty, names = _s1_countries(test_dir / "test_source1.tsv")
    size = {}
    with open(out_dir / "matching_results.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            size[encode_id(s1)] = len(rest.split(",")) if rest else 0
    k = np.array([size.get(int(c), 0) for c in req])
    return {n: {"n_s1": int((req_cty == i).sum()), "pred_empty_rate": round(float((k[req_cty == i] == 0).mean()), 5),
                "mean_pred_size": round(float(k[req_cty == i].mean()), 3)} for i, n in enumerate(names)}


def main() -> None:
    """CLI: `python -m src.check_outputs --out-dir output --test-dir student_resource/dataset/test`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--test-dir", default="../../student_resource/dataset/test")
    parser.add_argument("--no-check-ids", action="store_true", help="skip the S2/S3 existence check")
    args = parser.parse_args()

    problems = check_outputs(Path(args.out_dir), Path(args.test_dir), check_ids=not args.no_check_ids)
    print(output_summary(Path(args.out_dir), Path(args.test_dir)))
    if problems:
        print(f"FAIL — {len(problems)} issue(s):")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
        sys.exit(1)
    print("PASS — check_outputs found no issues.")


if __name__ == "__main__":
    main()
