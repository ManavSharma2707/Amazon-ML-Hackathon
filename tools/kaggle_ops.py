"""Kaggle automation helpers for the two GPU runners (R1, R2).

Lives at the repo root (NOT under code/business_entity_resolution/), so it is
never zipped into the competition submission.

Runner selection is done only through `.env.local` (git-ignored), which maps
"R1"/"R2" to a KAGGLE_CONFIG_DIR containing a single file, `access_token`
(Kaggle's newer `KGAT_...` token format — NOT the classic username+key
kaggle.json, which 401s on write calls with these tokens; see memory.md
pitfalls, 2026-09-26). Never hard-code Kaggle usernames or token values in
this file. Every dataset/kernel created here is private by default
(CLAUDE.md SS6, SS6.1).

Windows CLI quirk (kaggle==2.2.4): `datasets create`/`version`/`kernels push`
build an internal upload-cache filename by naively concatenating the target
folder's path with the uploaded filename. If that path contains a `/` or `\\`
(any multi-segment relative/absolute path), the resulting "filename" is
treated as a nested path and the upload fails with
`[Errno 2] No such file or directory`. Workaround: always run these commands
with `cwd` set to the target folder itself and pass `-p .` (a single-segment
path with no separator) — this module does that everywhere it uploads.

Usage (from repo root):
    python tools/kaggle_ops.py sync-code --runner R1
    python tools/kaggle_ops.py sync-code --runner R2
    python tools/kaggle_ops.py upsert-dataset --path student_resource --slug er-data --runner R1
    python tools/kaggle_ops.py push --dir code/business_entity_resolution/src/notebooks/nb00_download_models --runner R1
    python tools/kaggle_ops.py wait --slug <username>/nb00-download-models --runner R1
    python tools/kaggle_ops.py fetch-output --slug <username>/nb00-download-models --runner R1 --dest reports/raw/nb00
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bundle_kernel  # noqa: E402  (needs sys.path set up first)

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_LOCAL = REPO_ROOT / ".env.local"


def load_env_local() -> dict[str, str]:
    """Parse `.env.local` (KEY=VALUE per line, '#' comments) into a dict.

    Inputs: none (reads REPO_ROOT/.env.local).
    Outputs: dict of environment overrides. Empty dict if the file is missing.
    """
    env: dict[str, str] = {}
    if not ENV_LOCAL.exists():
        return env
    for line in ENV_LOCAL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def runner_config(runner: str) -> tuple[str, str]:
    """Resolve a runner name ("R1"/"R2") to (KAGGLE_CONFIG_DIR, kaggle_username).

    Inputs: runner - "R1" or "R2".
    Outputs: (config_dir, username) strings, read from .env.local.
    Raises: ValueError if the runner is unknown or config is missing.
    """
    runner = runner.upper()
    if runner not in ("R1", "R2"):
        raise ValueError(f"Unknown runner {runner!r}; expected R1 or R2")
    env = load_env_local()
    config_dir = env.get(f"{runner}_KAGGLE_CONFIG_DIR")
    username = env.get(f"{runner}_KAGGLE_USERNAME")
    if not config_dir or not username:
        raise ValueError(
            f"Missing {runner}_KAGGLE_CONFIG_DIR / {runner}_KAGGLE_USERNAME in .env.local"
        )
    return config_dir, username


def _python_exe() -> str:
    """Return the Python interpreter to invoke `kaggle` as a module.

    Falls back to `sys.executable` if PYTHON_EXE is not set in .env.local
    (handles Windows environments where a bare `python`/`python3` resolves
    to the Microsoft Store alias instead of a real interpreter).
    """
    env = load_env_local()
    return env.get("PYTHON_EXE", sys.executable)


def _read_token(runner: str) -> str:
    """Read the `KGAT_...` API token for a runner from its config dir.

    Inputs: runner - "R1"/"R2".
    Outputs: the token string, read from `<config_dir>/access_token`.
    Never printed or logged by any caller of this function.
    """
    config_dir, _ = runner_config(runner)
    token_path = Path(config_dir) / "access_token"
    if not token_path.exists():
        raise FileNotFoundError(
            f"{token_path} not found. Generate a token at kaggle.com/settings/api "
            "and save it there (one line, no trailing content)."
        )
    return token_path.read_text(encoding="utf-8").strip()


def _run_kaggle(
    args: list[str], runner: str, check: bool = True, cwd: Path | None = None
) -> subprocess.CompletedProcess:
    """Run a `kaggle` CLI subcommand authenticated as one runner.

    Inputs: args - CLI args after "kaggle" (e.g. ["kernels", "status", slug]);
            runner - "R1" or "R2"; check - raise on non-zero exit if True;
            cwd - working directory for the subprocess (needed for upload
            commands, see module docstring's Windows CLI quirk).
    Outputs: the completed subprocess (stdout/stderr captured as text).
    Never prints or logs the resolved token.
    """
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = _read_token(runner)
    # Without this, the kaggle CLI crashes writing non-ASCII output on Windows
    # (0-byte file, "'charmap' codec can't encode characters") instead of
    # failing loudly — force UTF-8 regardless of the console's codepage.
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [_python_exe(), "-m", "kaggle"] + args
    result = subprocess.run(
        cmd, cwd=str(cwd) if cwd else REPO_ROOT, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",  # CLI prints UTF-8; the Windows default codepage can't decode it
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"kaggle {' '.join(args)} failed (runner={runner}):\n{result.stdout}\n{result.stderr}"
        )
    return result


def dataset_exists(slug: str, runner: str) -> bool:
    """Check whether a private dataset `<username>/<slug>` already exists for this runner.

    Inputs: slug - dataset slug without owner prefix; runner - "R1"/"R2".
    Outputs: True if `kaggle datasets status` succeeds, False otherwise.
    """
    _, username = runner_config(runner)
    result = _run_kaggle(["datasets", "status", f"{username}/{slug}"], runner, check=False)
    return result.returncode == 0


def upsert_private_dataset(local_dir: Path | str, slug: str, runner: str, message: str = "update") -> str:
    """Create or version a private Kaggle dataset from a local folder.

    Writes a temporary dataset-metadata.json (owner/slug set to this runner's
    username) directly into `local_dir`, uploads with `cwd=local_dir` and
    `-p .` (Windows path-bug workaround, see module docstring), then removes
    the metadata file so it never lingers in a folder that also gets zipped
    for submission.

    Inputs: local_dir - folder to upload; slug - dataset slug (e.g. "er-code");
            runner - "R1"/"R2"; message - version message (only used on update).
    Outputs: the dataset id ("<username>/<slug>") that was created/updated.
    """
    local_dir = Path(local_dir).resolve()
    _, username = runner_config(runner)
    meta_path = local_dir / "dataset-metadata.json"
    meta_backup = meta_path.read_bytes() if meta_path.exists() else None
    meta = {
        "title": slug,
        "id": f"{username}/{slug}",
        "licenses": [{"name": "unknown"}],
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    try:
        if dataset_exists(slug, runner):
            _run_kaggle(
                ["datasets", "version", "-p", ".", "-m", message, "-r", "zip", "-d"],
                runner,
                cwd=local_dir,
            )
        else:
            _run_kaggle(["datasets", "create", "-p", ".", "-r", "zip"], runner, cwd=local_dir)
    finally:
        if meta_backup is not None:
            meta_path.write_bytes(meta_backup)
        else:
            meta_path.unlink(missing_ok=True)
    return f"{username}/{slug}"


def sync_code(runner: str, message: str = "sync er-code") -> str:
    """One-command refresh of the `er-code` dataset from code/business_entity_resolution/.

    Inputs: runner - "R1"/"R2"; message - version message.
    Outputs: the dataset id that was updated.
    """
    code_dir = REPO_ROOT / "code" / "business_entity_resolution"
    return upsert_private_dataset(code_dir, "er-code", runner, message=message)


def push_notebook(kernel_dir: Path | str, runner: str, accelerator: str | None = None, timeout: int | None = None) -> str:
    """Push a Kaggle notebook (kernel) from a local folder.

    Runs with `cwd=kernel_dir` and `-p .` (Windows path-bug workaround, see
    module docstring).

    If `kernel_dir` contains a `bundle_spec.json` (see bundle_kernel.py), the
    self-contained kernel script is (re)generated from the current `src/`
    modules + driver.py before every push — `src/` stays the single source of
    truth and the pushed script never depends on a mounted code dataset.

    `kernel-metadata.json` in `kernel_dir` must have its `id` field, and any
    `dataset_sources` entries, prefixed with `PLACEHOLDER/` (e.g.
    `"PLACEHOLDER/nb00-download-models"`, `"PLACEHOLDER/er-code"`); this
    function rewrites every `PLACEHOLDER/` to `<runner's username>/` before
    pushing and restores the placeholders afterwards, so the same kernel
    folder can be pushed to either runner without hard-coding a username in a
    file that gets zipped/committed.

    Inputs: kernel_dir - folder containing kernel-metadata.json + notebook source;
            runner - "R1"/"R2"; accelerator - value accepted by `kaggle kernels
            push --accelerator` (verify with `kaggle kernels push --help` and the
            Kaggle docs before relying on a specific string; not guessed here);
            timeout - optional run-time limit in seconds.
    Outputs: stdout from the push command (contains the kernel URL/slug).
    """
    kernel_dir = Path(kernel_dir).resolve()
    spec_path = kernel_dir / "bundle_spec.json"
    if spec_path.exists():
        bundle_kernel.bundle_from_spec(spec_path)
    _, username = runner_config(runner)
    meta_path = kernel_dir / "kernel-metadata.json"
    original_text = meta_path.read_text(encoding="utf-8")
    meta_path.write_text(original_text.replace("PLACEHOLDER/", f"{username}/"), encoding="utf-8")
    try:
        args = ["kernels", "push", "-p", "."]
        if accelerator:
            args += ["--accelerator", accelerator]
        if timeout:
            args += ["-t", str(timeout)]
        result = _run_kaggle(args, runner, cwd=kernel_dir)
        return result.stdout
    finally:
        meta_path.write_text(original_text, encoding="utf-8")


def wait(slug: str, runner: str, poll: int = 300, max_polls: int = 200) -> str:
    """Poll `kaggle kernels status` until the kernel finishes, erroring or completing.

    Sleeps `poll` seconds between checks (never a tight loop, per CLAUDE.md SS6).

    Inputs: slug - "<username>/<kernel-slug>"; runner - "R1"/"R2";
            poll - seconds between status checks; max_polls - safety cap.
    Outputs: the final status string (e.g. "complete", "error").
    """
    for _ in range(max_polls):
        result = _run_kaggle(["kernels", "status", slug], runner, check=False)
        status_line = result.stdout.strip()
        if "has status \"complete\"" in status_line or "complete" in status_line.lower():
            return "complete"
        if "error" in status_line.lower() or "failed" in status_line.lower():
            return status_line
        time.sleep(poll)
    return "timeout"


def fetch_output(slug: str, runner: str, dest: Path | str) -> Path:
    """Download a kernel's output files to a local folder.

    Inputs: slug - "<username>/<kernel-slug>"; runner - "R1"/"R2"; dest - local folder.
    Outputs: the resolved destination Path.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    _run_kaggle(["kernels", "output", slug, "-p", str(dest)], runner)
    return dest


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this module's subcommands."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("sync-code", help="Upload code/business_entity_resolution/ as er-code")
    sp.add_argument("--runner", required=True)
    sp.add_argument("--message", default="sync er-code")

    sp = sub.add_parser("upsert-dataset", help="Create or version a private dataset from a folder")
    sp.add_argument("--path", required=True)
    sp.add_argument("--slug", required=True)
    sp.add_argument("--runner", required=True)
    sp.add_argument("--message", default="update")

    sp = sub.add_parser("push", help="Push a notebook/kernel")
    sp.add_argument("--dir", required=True)
    sp.add_argument("--runner", required=True)
    sp.add_argument("--accelerator", default=None)
    sp.add_argument("--timeout", type=int, default=None)

    sp = sub.add_parser("wait", help="Poll a kernel until it finishes")
    sp.add_argument("--slug", required=True)
    sp.add_argument("--runner", required=True)
    sp.add_argument("--poll", type=int, default=300)

    sp = sub.add_parser("fetch-output", help="Download a kernel's output")
    sp.add_argument("--slug", required=True)
    sp.add_argument("--runner", required=True)
    sp.add_argument("--dest", required=True)

    sp = sub.add_parser("list-mine", help="List datasets owned by a runner (auth check)")
    sp.add_argument("--runner", required=True)

    return p


def main() -> None:
    """CLI entry point dispatching to the functions above."""
    args = _build_parser().parse_args()
    if args.command == "sync-code":
        print(sync_code(args.runner, args.message))
    elif args.command == "upsert-dataset":
        print(upsert_private_dataset(args.path, args.slug, args.runner, args.message))
    elif args.command == "push":
        print(push_notebook(args.dir, args.runner, args.accelerator, args.timeout))
    elif args.command == "wait":
        print(wait(args.slug, args.runner, args.poll))
    elif args.command == "fetch-output":
        print(fetch_output(args.slug, args.runner, args.dest))
    elif args.command == "list-mine":
        result = _run_kaggle(["datasets", "list", "--mine"], args.runner)
        print(result.stdout)


if __name__ == "__main__":
    main()
