"""Generate the markdown tables used in Documentation_template.md / approach_summary.md
from this run's fetched Kaggle metrics.json / blocking_report.json files.

Every number in the two documentation files comes from here, not from prose
written by hand — regenerating a table (e.g. after a different final config is
chosen) means re-running this script and pasting its output, not rewriting
numbers by hand. Reads only `reports/raw/<notebook>/...` (fetched Kaggle
outputs, git-ignored, not part of the submission zip: CLAUDE.md SS8.1) — not
required at inference time.

Usage (from code/business_entity_resolution/):
    python src/scripts/make_doc_tables.py [--reports-dir ../../reports/raw] [--table NAME]
Table names: ablation, blocking, loco, judge, error_buckets, per_country, scrambled.
With no --table, prints every table with a heading.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(reports_dir: Path, name: str, file: str = "metrics.json") -> dict:
    """Parse one fetched report file, or raise a clear error naming the missing run."""
    p = reports_dir / name / file
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — fetch it with `kaggle kernels output` first (see README.md)")
    return json.loads(p.read_text(encoding="utf-8"))


def ablation_table(reports_dir: Path) -> str:
    """Configuration -> B F0.5 -> LOCO-mean -> scrambled drop (plan SS21.6)."""
    combiner = _load(reports_dir, "nb09_v2")
    stage1_pipeline = _load(reports_dir, "nb09a_v2")
    rows = [("All-empty floor", combiner["ablation"][0]["B_f05"], "", ""),
            ("Stage-1 + global threshold", combiner["ablation"][1]["B_f05"], "", ""),
            ("+ calibration + expected-F decoder", combiner["ablation"][2]["B_f05"], "", ""),
            ("+ one-owner rule (**shipped: sub-01**)", stage1_pipeline["best"]["f05"],
             stage1_pipeline["loco_lite_mean"], stage1_pipeline["scrambled"]["drop"]),
            ("+ combiner with collective features (evaluated, dropped: no LOCO gain)", combiner["best"]["f05"],
             combiner["loco"]["mean"], combiner["scrambled"]["drop"])]
    out = ["| Configuration | B F0.5 | LOCO-mean | Scrambled drop |", "|---|---|---|---|"]
    for name, f05, loco, drop in rows:
        out.append(f"| {name} | {f05:.4f} | {loco if loco == '' else f'{loco:.4f}'} | {drop if drop == '' else f'{drop:.4f}'} |")
    return "\n".join(out)


def blocking_table(reports_dir: Path) -> str:
    """Pair recall / entity-complete recall / reduction ratio / candidates-per-S1, by blocking version."""
    versions = [("v4 (**shipped**: fed the final stage-1/decoder run)", "nb05_sparse_v4"),
                ("v5 (+ cross name x address pairs, look-alike digit folding)", "nb05_sparse_v5"),
                ("v6 (+ reverse pool->S1 name lookup)", "nb05_sparse_v6")]
    out = ["| Blocking version | B pair recall | B entity-complete recall | Reduction ratio | Candidates/S1 (mean) |",
          "|---|---|---|---|---|"]
    for label, name in versions:
        b = _load(reports_dir, name, "blocking_report.json")["B"]["ALL"]
        out.append(f"| {label} | {b['pair_recall']:.4f} | {b['entity_complete_recall']:.4f} | "
                   f"{b['reduction_ratio']:.6f} | {b['cands_per_s1_mean']:.1f} |")
    v4 = _load(reports_dir, "nb05_sparse_v4", "blocking_report.json")["B"]["ALL"]["channels"]
    out += ["", "Per-channel contribution, shipped v4 (Half B, `found` = pairs a channel's top-k list included; "
           "`unique` = true pairs found by no other channel):", "",
           "| Channel | Found | Unique |", "|---|---|---|"]
    for ch, v in v4.items():
        if v["found"] or v["unique"]:
            out.append(f"| {ch} | {v['found']:,} | {v['unique']:,} |")
    return "\n".join(out)


def loco_table(reports_dir: Path) -> str:
    """LOCO (train on one country, score the other) for the shipped pipeline and the (dropped) combiner."""
    s1p = _load(reports_dir, "nb09a_v2")["loco_lite"]
    comb = _load(reports_dir, "nb09_v2")["loco"]
    out = ["| Direction | Stage-1 pipeline (shipped) | Combiner (evaluated, dropped) |", "|---|---|---|"]
    dirs = sorted((set(s1p) | set(comb)) - {"mean"})
    for k in dirs:
        out.append(f"| {k} | {s1p.get(k, {}).get('f05', float('nan')):.4f} | {comb.get(k, {}).get('f05', float('nan')):.4f} |")
    out.append(f"| **mean** | **{_load(reports_dir, 'nb09a_v2')['loco_lite_mean']:.4f}** | "
               f"**{_load(reports_dir, 'nb09_v2')['loco']['mean']:.4f}** |")
    return "\n".join(out)


def judge_table(reports_dir: Path) -> str:
    """Judge (Qwen3-4B QLoRA) vs stage-1 AUC on the held-out training slice and the uncertain band (Gate G4)."""
    j = _load(reports_dir, "nb07_v1")
    out = ["| | AUC (held-out) | Log-loss (held-out) |", "|---|---|---|",
          f"| Judge | {j['val']['judge']['auc']:.4f} | {j['val']['judge']['logloss']:.4f} |",
          f"| Stage-1 | {j['val']['stage1']['auc']:.4f} | {j['val']['stage1']['logloss']:.4f} |", "",
          f"On the uncertain band only ({j['val_band']['n']} pairs, plan SS16.1's Gate G4 check):", "",
          "| | AUC |", "|---|---|",
          f"| Judge | {j['val_band']['judge']['auc']:.4f} |",
          f"| Stage-1 | {j['val_band']['stage1']['auc']:.4f} |", "",
          f"**Gate G4 (judge band AUC > stage-1 band AUC): FAILED** "
          f"({j['val_band']['judge']['auc']:.3f} < {j['val_band']['stage1']['auc']:.3f}) -> judge dropped. "
          f"Training was cut to a {j['budget_min']:.0f}-minute wall-clock budget "
          f"({j['train']['examples_seen']:,} of {j['n_train']:,} available examples seen) to fit the day's schedule."]
    return "\n".join(out)


def error_buckets_table(reports_dir: Path) -> str:
    """Wrong-entity counts by failure cause on Half B (CLAUDE.md SS4.4), shipped-pipeline candidates."""
    eb = _load(reports_dir, "nb09_v2")["error_buckets"]
    total = sum(eb.values())
    out = ["| Bucket | Count | Share of wrong entities |", "|---|---|---|"]
    for b, n in sorted(eb.items(), key=lambda kv: -kv[1]):
        out.append(f"| {b} | {n:,} | {n / total:.1%} |")
    out.append(f"| **total wrong** | **{total:,}** | of 250,000 Half-B S1 entities |")
    return "\n".join(out)


def per_country_table(reports_dir: Path) -> str:
    """Shipped pipeline: B F0.5, singleton accuracy and non-singleton F0.5 by country, vs the floor."""
    pc = _load(reports_dir, "nb09_v2")["B"]["per_country_stage1_pipeline"]
    out = ["| Country | n S1 | F0.5 | All-empty floor | Singleton accuracy | Non-singleton F0.5 |",
          "|---|---|---|---|---|---|"]
    for c in ("ALL", "India", "US"):
        v = pc[c]
        out.append(f"| {c} | {v['n_s1']:,} | {v['f05']:.4f} | {v['floor']:.4f} | "
                   f"{v['singleton_acc']:.4f} | {v['non_singleton_f05']:.4f} |")
    return "\n".join(out)


def scrambled_table(reports_dir: Path) -> str:
    """Scrambled-letter test (plan SS21.4): F0.5 with vs without a random a-z permutation on Half B."""
    s1p = _load(reports_dir, "nb09a_v2")["scrambled"]
    comb = _load(reports_dir, "nb09_v2")["scrambled"]
    out = ["| Pipeline | n S1 | F0.5 (original) | F0.5 (scrambled) | Drop |", "|---|---|---|---|---|"]
    for label, d in (("Stage-1 pipeline (shipped)", s1p), ("Combiner (evaluated, dropped)", comb)):
        out.append(f"| {label} | {d['n_s1']:,} | {d['f05_orig']:.4f} | {d['f05_scrambled']:.4f} | {d['drop']:.4f} |")
    return "\n".join(out)


TABLES = {"ablation": ablation_table, "blocking": blocking_table, "loco": loco_table, "judge": judge_table,
         "error_buckets": error_buckets_table, "per_country": per_country_table, "scrambled": scrambled_table}


def main() -> None:
    """CLI: print one table, or all of them with headings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", default="../../reports/raw")
    parser.add_argument("--table", choices=sorted(TABLES))
    args = parser.parse_args()
    reports_dir = Path(args.reports_dir)
    names = [args.table] if args.table else sorted(TABLES)
    for name in names:
        if not args.table:
            print(f"\n### {name}\n")
        print(TABLES[name](reports_dir))


if __name__ == "__main__":
    main()
