"""NB01 driver — exploratory data study (architecture.md SS6.3 / master plan SS7).

Hand-written entry point. This file is NOT pushed to Kaggle directly: run
`python tools/bundle_kernel.py --spec <this folder>/bundle_spec.json` to embed
it together with the `io_utils`/`eda` src modules into the self-contained
`nb01_eda.py`, which is what actually gets pushed (kernel-metadata.json's
`code_file`). This keeps `src/` as the single source of truth (testable
locally with pytest) while the pushed kernel needs no dataset-mounted code at
all — only `er-data` is attached as an input.

References `io_utils`/`eda` as bare names: the bundler wires them in as real
module objects before this code runs, so it must never `sys.path.insert(...)`
or `import src`/`from src import ...`.
"""

import glob
import json
from pathlib import Path

OUT_PATH = Path("/kaggle/working/eda.json")

# Kaggle has mounted dataset inputs at both `/kaggle/input/<slug>/` (classic)
# and `/kaggle/input/datasets/<owner>/<slug>/` (observed 2026-09-26 on this
# environment) at different times. Resolve dynamically instead of hard-coding
# either convention or any owner username.
_DATA_DIR_CANDIDATES = (
    glob.glob("/kaggle/input/er-data/student_resource/dataset")
    + glob.glob("/kaggle/input/datasets/*/er-data/student_resource/dataset")
)


def find_data_dir() -> Path:
    """Locate the mounted `er-data` dataset's `dataset/` folder, whichever mount convention applies.

    Inputs: none. Outputs: a Path that exists.
    Raises: FileNotFoundError with the searched patterns, if neither convention matched.
    """
    if not _DATA_DIR_CANDIDATES:
        raise FileNotFoundError(
            "er-data not found under /kaggle/input/er-data/ or "
            "/kaggle/input/datasets/*/er-data/ -- check the kernel's dataset_sources."
        )
    return Path(_DATA_DIR_CANDIDATES[0])


def main() -> None:
    """Run the full EDA against the Kaggle-mounted dataset and write the report."""
    data_dir = find_data_dir()
    findings = eda.run_eda(data_dir)
    OUT_PATH.write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")
    eda.print_summary(findings)
    print(f"\nFull findings written to {OUT_PATH}")


if __name__ == "__main__":
    main()
