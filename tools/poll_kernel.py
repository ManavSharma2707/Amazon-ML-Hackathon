"""Poll one kernel's status until it leaves QUEUED/RUNNING; print the final status line.

Used as a background job so the session is notified on completion.
Usage: python tools/poll_kernel.py <kernel-slug-without-owner> <R1|R2> [interval_s]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_ops  # noqa: E402


def main() -> None:
    """Poll until the kernel is no longer queued/running."""
    slug, runner = sys.argv[1], sys.argv[2]
    interval = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    _, user = kaggle_ops.runner_config(runner)
    while True:
        out = kaggle_ops._run_kaggle(["kernels", "status", f"{user}/{slug}"], runner, check=False).stdout.strip()
        if "RUNNING" not in out and "QUEUED" not in out:
            print(out.replace(user, "<user>"))
            return
        time.sleep(interval)


if __name__ == "__main__":
    main()
