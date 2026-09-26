"""NB01 — exploratory data study (architecture.md SS6.3 / master plan SS7).

Runs on a Kaggle CPU kernel with `er-code` and `er-data` attached as dataset
inputs (this project's local RAM is too tight for the full multi-million-row
dataset — memory.md SS5). Writes the full findings to
/kaggle/working/eda.json and prints a short summary to paste into
memory.md SS6.
"""

import sys
from pathlib import Path

sys.path.insert(0, "/kaggle/input/er-code")

from src import eda  # noqa: E402  (path must be set up first)

DATA_DIR = Path("/kaggle/input/er-data/student_resource/dataset")
OUT_PATH = Path("/kaggle/working/eda.json")


def main() -> None:
    """Run the full EDA against the Kaggle-mounted dataset and write the report."""
    findings = eda.run_eda(DATA_DIR)
    import json

    OUT_PATH.write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")
    eda.print_summary(findings)
    print(f"\nFull findings written to {OUT_PATH}")


if __name__ == "__main__":
    main()
