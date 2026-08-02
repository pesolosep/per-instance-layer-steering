"""Download Hugging Face model weights into a local snapshot dir.

Used by the Kaggle prep notebook (CPU + Internet ON) to stage weights into
`/kaggle/working/`. The committed prep notebook is then attached as a Kaggle
Notebook input to GPU consumer notebooks (Internet OFF).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download


SUPPORTED_MODELS: dict[str, dict[str, str]] = {
    "llama": {
        "hf_repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "local_dir_name": "Meta-Llama-3-8B-Instruct",
    },
    "aya": {
        "hf_repo_id": "CohereForAI/aya-expanse-8b",
        "local_dir_name": "aya-expanse-8b",
    },
    "qwen": {
        "hf_repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "local_dir_name": "Qwen2.5-7B-Instruct",
    },
}

# Skip GGUF/PyTorch-bin/sharded-original blobs — we want safetensors only.
DEFAULT_IGNORE_PATTERNS: tuple[str, ...] = (
    "*.gguf",
    "*.bin",
    "original/*",
    "consolidated.*",
)


def resolve_model_spec(model_target: str) -> dict[str, str]:
    if model_target not in SUPPORTED_MODELS:
        raise ValueError(
            f"unknown model_target {model_target!r}; "
            f"supported: {sorted(SUPPORTED_MODELS)}"
        )
    return SUPPORTED_MODELS[model_target]


def download_model(
    model_target: str,
    working_dir: Path | str,
    hf_token: str,
    ignore_patterns: Iterable[str] | None = None,
) -> Path:
    """Snapshot-download the configured model into ``working_dir/<local_dir_name>``.

    Returns the local model directory path.
    """
    spec = resolve_model_spec(model_target)
    model_dir = Path(working_dir) / spec["local_dir_name"]
    model_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=spec["hf_repo_id"],
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        token=hf_token,
        ignore_patterns=list(ignore_patterns or DEFAULT_IGNORE_PATTERNS),
    )
    return model_dir


def verify_model_snapshot(model_dir: Path | str) -> tuple[bool, list[str]]:
    """Check the snapshot has the files an offline consumer needs.

    Returns ``(ok, missing_patterns)``.
    """
    model_dir = Path(model_dir)
    required_globs = ["config.json", "tokenizer.json", "*.safetensors"]
    missing = [pat for pat in required_globs if not list(model_dir.glob(pat))]
    return (not missing, missing)


def report_snapshot_size(model_dir: Path | str) -> float:
    """Print per-file sizes; return total in GB."""
    model_dir = Path(model_dir)
    total_bytes = 0
    for p in sorted(model_dir.iterdir()):
        if p.is_file():
            sz = p.stat().st_size
            total_bytes += sz
            print(f"  {p.name:40s} {sz / 1e6:8.1f} MB")
    total_gb = total_bytes / 1e9
    print(f"\nTotal: {total_gb:.2f} GB")
    return total_gb
