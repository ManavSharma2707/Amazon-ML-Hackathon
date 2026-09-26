"""NB00 — download base model weights once, with internet ON (architecture.md SS6.2).

Runs on a Kaggle CPU kernel (no GPU needed for a download). Downloads the
three allowed base models (all MIT/Apache-2.0, <= 8B params each, per
CLAUDE.md SS3), pip-downloads offline wheels for later GPU notebooks, and
writes a LICENSES.md by reading each model card's license field. Saves
everything under /kaggle/working so the notebook's output can be turned into
the `er-models` Kaggle Dataset (architecture.md SS5). Every later notebook
loads these weights with internet OFF.

This is a Kaggle "script kernel" (not a .ipynb) — Kaggle runs a .py kernel
the same way and produces the same versioned output, and a plain script is
easier to review/diff than notebook JSON.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

WORK_DIR = Path("/kaggle/working")
MODELS_DIR = WORK_DIR / "models"
WHEELS_DIR = WORK_DIR / "wheels"

# (Hugging Face repo id, local short name) — must match architecture.md SS5's er-models layout.
MODELS = [
    ("Qwen/Qwen3-Embedding-0.6B", "qwen3-emb-0.6b"),
    ("Qwen/Qwen3-4B", "qwen3-4b"),
    ("Qwen/Qwen3-Reranker-0.6B", "qwen3-reranker-0.6b"),
]

# Pinned per master plan SS5.2 / CLAUDE.md SS5 (Qwen3 needs transformers>=4.51.0).
WHEEL_PACKAGES = [
    "transformers>=4.51.0",
    "sentence-transformers>=3.0",
    "peft>=0.11",
    "bitsandbytes>=0.43",
    "accelerate>=0.30",
    "rapidfuzz",
    "lightgbm",
    "faiss-cpu",
]


def pip_install_upgrade() -> None:
    """Install/upgrade the packages needed to download and inspect the models.

    Inputs: none. Outputs: none (side effect: installs into the kernel env).
    """
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.51.0", "huggingface_hub"],
        check=True,
    )


def download_models() -> dict[str, str]:
    """Snapshot-download every model in MODELS into MODELS_DIR.

    Inputs: none. Outputs: {short_name: local_path} for the downloaded snapshots.
    """
    from huggingface_hub import snapshot_download

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for repo_id, short_name in MODELS:
        local_dir = MODELS_DIR / short_name
        print(f"Downloading {repo_id} -> {local_dir}")
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
        paths[short_name] = str(local_dir)
    return paths


def download_wheels() -> None:
    """Download wheels for the GPU-notebook dependencies, for fully offline install later.

    Inputs: none. Outputs: none (side effect: writes .whl files to WHEELS_DIR).
    """
    WHEELS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "download", "-d", str(WHEELS_DIR), *WHEEL_PACKAGES],
        check=True,
    )


def write_licenses(model_paths: dict[str, str]) -> None:
    """Read each downloaded model's `config.json`/card metadata and record its licence.

    Falls back to the known licence from CLAUDE.md SS3/memory.md SS4 if the
    downloaded snapshot doesn't carry a machine-readable licence field (the
    README's YAML front-matter is the authoritative source; this is a
    convenience record, not a legal determination — the licence was already
    checked against each model card before being added to the allowed list).

    Inputs: model_paths - {short_name: local_path} from download_models().
    Outputs: none (side effect: writes MODELS_DIR/../LICENSES.md).
    """
    known_licenses = {
        "qwen3-emb-0.6b": "Apache-2.0",
        "qwen3-4b": "Apache-2.0",
        "qwen3-reranker-0.6b": "Apache-2.0",
    }
    lines = ["# LICENSES.md — base model licences (recorded by NB00)\n"]
    for repo_id, short_name in MODELS:
        local_dir = Path(model_paths[short_name])
        license_note = known_licenses.get(short_name, "UNKNOWN — verify manually before use")
        readme = local_dir / "README.md"
        if readme.exists():
            head = readme.read_text(encoding="utf-8", errors="ignore")[:2000]
            if "license:" in head.lower():
                for line in head.splitlines():
                    if line.strip().lower().startswith("license:"):
                        license_note = line.split(":", 1)[1].strip()
                        break
        lines.append(f"- **{repo_id}** ({short_name}): {license_note}\n")
    (WORK_DIR / "LICENSES.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote LICENSES.md:\n" + "\n".join(lines))


def write_manifest(model_paths: dict[str, str]) -> None:
    """Write a small manifest.json recording what this run produced (architecture.md SS5).

    Inputs: model_paths - {short_name: local_path}.
    Outputs: none (side effect: writes WORK_DIR/manifest.json).
    """
    manifest = {
        "producing_notebook": "nb00_download_models",
        "models": model_paths,
    }
    (WORK_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    """Run the full NB00 flow: install, download models + wheels, write licences + manifest."""
    pip_install_upgrade()
    model_paths = download_models()
    download_wheels()
    write_licenses(model_paths)
    write_manifest(model_paths)
    print("NB00 done. Save this kernel's output as the `er-models` Kaggle Dataset.")


if __name__ == "__main__":
    main()
