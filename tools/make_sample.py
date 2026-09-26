"""Build a small, git-ignored `sample/` copy of the dataset for local unit tests.

Streams the full TSVs line by line (never loads them into memory; the laptop
has < 1 GB free RAM, memory.md SS5) and keeps:
- train: ~0.2% of ground-truth S1 entities (chosen by a CRC32 hash of the ID,
  so the choice is deterministic and unrelated to row order), their S1
  records, all their matched S2/S3 records, plus ~0.05% of all other S2/S3
  records as distractors;
- test: ~0.1% of S1 records and ~0.05% of S2/S3 records (bug detection only).

The output folder mirrors `student_resource/dataset/` (same file names under
`sample/train/` and `sample/test/`), so every pipeline driver can run end to
end on it locally. Lives in tools/ (not zipped). The hash is used only to
sample, never as a feature (CLAUDE.md SS3 rule 7).

Usage: python tools/make_sample.py [--src student_resource/dataset] [--dst sample]
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path


def _keep(entity_id: str, per_mille: float) -> bool:
    """Deterministically keep an ID with probability ~per_mille/1000 (CRC32-based).

    Inputs: entity_id - ID string; per_mille - keep rate in thousandths.
    Outputs: True if the ID falls in the sampled hash range.
    """
    return (zlib.crc32(entity_id.encode("utf-8")) % 100000) < per_mille * 100


def _filter_file(src: Path, dst: Path, keep_fn) -> int:
    """Copy the header plus every line whose first field passes `keep_fn`.

    Inputs: src/dst - file paths; keep_fn - callable(entity_id) -> bool.
    Outputs: number of data lines written.
    """
    n = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, encoding="utf-8", newline="") as fin, open(dst, "w", encoding="utf-8", newline="") as fout:
        fout.write(fin.readline())
        for line in fin:
            if keep_fn(line.split("\t", 1)[0]):
                fout.write(line)
                n += 1
    return n


def main() -> None:
    """Build sample/train and sample/test from the full dataset."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="student_resource/dataset")
    p.add_argument("--dst", default="sample")
    args = p.parse_args()
    src, dst = Path(args.src), Path(args.dst)

    # Train: pick S1 entities from the ground truth, collect their matched IDs.
    s1_keep: set[str] = set()
    matched_keep: set[str] = set()
    gt_dst = dst / "train" / "train_ground_truth.tsv"
    gt_dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src / "train" / "train_ground_truth.tsv", encoding="utf-8", newline="") as fin, open(
        gt_dst, "w", encoding="utf-8", newline=""
    ) as fout:
        fout.write(fin.readline())
        for line in fin:
            s1, _, matches = line.rstrip("\n").partition("\t")
            if _keep(s1, 2.0):
                s1_keep.add(s1)
                if matches:
                    matched_keep.update(matches.split(","))
                fout.write(line)
    print(f"train GT: {len(s1_keep)} S1, {len(matched_keep)} matched IDs")

    n = _filter_file(src / "train" / "train_source1.tsv", dst / "train" / "train_source1.tsv", s1_keep.__contains__)
    print(f"train S1 rows: {n}")
    for s in ("2", "3"):
        n = _filter_file(
            src / "train" / f"train_source{s}.tsv",
            dst / "train" / f"train_source{s}.tsv",
            lambda i: i in matched_keep or _keep(i, 0.5),
        )
        print(f"train S{s} rows: {n}")

    n = _filter_file(src / "test" / "test_source1.tsv", dst / "test" / "test_source1.tsv", lambda i: _keep(i, 1.0))
    print(f"test S1 rows: {n}")
    for s in ("2", "3"):
        n = _filter_file(
            src / "test" / f"test_source{s}.tsv", dst / "test" / f"test_source{s}.tsv", lambda i: _keep(i, 0.5)
        )
        print(f"test S{s} rows: {n}")


if __name__ == "__main__":
    main()
