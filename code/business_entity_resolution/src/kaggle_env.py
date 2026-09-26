"""Small helpers shared by every Kaggle notebook driver.

- `find_input`: locate a file/folder under `/kaggle/input/` without assuming
  a mount convention. Datasets have been seen at `/kaggle/input/<slug>/` and
  at `/kaggle/input/datasets/<owner>/<slug>/`, and notebook outputs attached
  via `kernel_sources` mount elsewhere again, so paths are always globbed,
  never hard-coded (CLAUDE.md SS6, memory.md SS9).
- `log`: timestamped, flushed progress lines (CLAUDE.md SS5 logging rule).
- `write_json`: small JSON reports (metrics.json) that the laptop fetches.
"""

from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

_T0 = time.time()
INPUT_ROOT = os.environ.get("ER_INPUT_ROOT", "/kaggle/input")
# Local smoke runs point these at sample/ and a scratch folder.
WORK_DIR = Path(os.environ.get("ER_WORK_DIR", "/kaggle/working"))


def log(msg: str) -> None:
    """Print a line prefixed with elapsed seconds and flush immediately.

    Inputs: msg. Outputs: None (stdout).
    """
    print(f"[{time.time() - _T0:8.1f}s] {msg}", flush=True)


def find_input(relpath: str, root: str | None = None, max_depth: int = 6) -> Path:
    """Find `relpath` (e.g. "train/train_source1.tsv") anywhere under the input root.

    Searches depth 0..max_depth of wildcard folders, shallowest first, so the
    result is deterministic when several inputs contain the same file.

    Inputs: relpath - path suffix to look for; root - search root (default
            /kaggle/input, or $ER_INPUT_ROOT for local runs); max_depth.
    Outputs: Path of the first match.
    Raises: FileNotFoundError listing what was searched.
    """
    root = root or INPUT_ROOT
    for depth in range(max_depth + 1):
        pattern = os.path.join(root, *(["*"] * depth), relpath)
        hits = sorted(glob.glob(pattern))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(f"{relpath!r} not found under {root} (depth <= {max_depth})")


def find_input_dir(marker: str, root: str | None = None) -> Path:
    """Return the folder that contains `marker` (a relative path) under the input root.

    Example: find_input_dir("train/train_source1.tsv") -> .../dataset
    """
    hit = find_input(marker, root)
    parent = hit
    for _ in Path(marker).parts:
        parent = parent.parent
    return parent


def write_json(obj: dict, path: str | Path) -> None:
    """Write a small JSON report (indent 2, non-serialisable values as str)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def ensure_rapidfuzz() -> None:
    """Install rapidfuzz (>= 3.6, for `process.cpdist`) from NB00's offline wheels if missing.

    Kernels run with internet OFF, so the wheel folder of the attached NB00
    output is the only source. Fails loudly if it cannot be installed.
    """
    import importlib
    import subprocess
    import sys

    try:
        importlib.import_module("rapidfuzz.process").cpdist  # noqa: B018
        return
    except (ImportError, AttributeError):
        pass
    wheels = find_input("wheels")
    log(f"installing rapidfuzz from {wheels}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "--no-index", "--find-links", str(wheels),
                    "rapidfuzz"], check=True)
    importlib.invalidate_caches()
    importlib.import_module("rapidfuzz.process").cpdist  # noqa: B018


def main() -> None:
    """Print where the dataset would be found (smoke test)."""
    try:
        print(find_input_dir("train/train_source1.tsv"))
    except FileNotFoundError as e:
        print(e)


if __name__ == "__main__":
    main()
