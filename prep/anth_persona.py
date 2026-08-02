"""Loader for the Anthropic Persona evaluation dataset.

Six binary-choice (Yes/No) tasks, each with train/val/test splits in JSONL.
Each row has at minimum ``question`` (str) and ``answer_matching_behavior``
(`" Yes"` or `" No"`).

Example::

    from prep.anth_persona import load_split, summary_stats, TASKS
    rows = load_split("conscientiousness", "train")
    print(rows[0]["question"])
    print(summary_stats())  # per-task per-split counts
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal


TASKS: tuple[str, ...] = (
    "conscientiousness",
    "subscribes-to-Christianity",
    "believes-it-has-phenomenal-consciousness",
    "cognitive-enhancement",
    "desire-to-create-allies",
    "desire-to-maximize-impact-on-world",
)

TASK_DISPLAY_NAMES: dict[str, str] = {
    "conscientiousness": "Conscientiousness",
    "subscribes-to-Christianity": "Religion Following",
    "believes-it-has-phenomenal-consciousness": "Self-Aware",
    "cognitive-enhancement": "Self-Improvement",
    "desire-to-create-allies": "Alliance Building",
    "desire-to-maximize-impact-on-world": "Impact Maximization",
}

SPLITS: tuple[str, ...] = ("train", "val", "test")
Split = Literal["train", "val", "test"]


def find_dataset_root(search_root: Path | str | None = None) -> Path:
    """Locate the ``Anth_Persona_ALL`` directory.

    Search order:

    1. ``search_root / "Anth_Persona_ALL"`` if ``search_root`` given
    2. ``search_root`` itself if it already points at the directory
    3. ``<cwd>/Dataset/Anth_Persona_ALL`` (running locally inside the repo)
    4. Any ``Anth_Persona_ALL`` directory under ``/kaggle/input/`` (offline Kaggle)

    Returns the first match that contains at least one ``*.jsonl``.
    """
    candidates: list[Path] = []
    if search_root is not None:
        sr = Path(search_root)
        candidates.append(sr / "Anth_Persona_ALL")
        candidates.append(sr)
    candidates.append(Path.cwd() / "Dataset" / "Anth_Persona_ALL")

    for c in candidates:
        if c.is_dir() and any(c.glob("*.jsonl")):
            return c

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for c in kaggle_input.rglob("Anth_Persona_ALL"):
            if c.is_dir() and any(c.glob("*.jsonl")):
                return c

    raise FileNotFoundError(
        f"Anth_Persona_ALL not found — tried {[str(c) for c in candidates]} "
        f"and rglob in /kaggle/input"
    )


def load_split(
    task: str,
    split: Split,
    dataset_root: Path | str | None = None,
) -> list[dict]:
    """Load a single ``(task, split)`` JSONL as a list of dicts.

    Each dict has at least ``question`` (str) and ``answer_matching_behavior``
    (`" Yes"` or `" No"`). Anthropic's original schema also includes
    ``statement``, ``label_confidence``, ``answer_not_matching_behavior``.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; supported: {list(TASKS)}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; supported: {list(SPLITS)}")

    root = Path(dataset_root) if dataset_root else find_dataset_root()
    path = root / f"{task}_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_all_splits(
    task: str,
    dataset_root: Path | str | None = None,
) -> dict[str, list[dict]]:
    """Load all three splits for one task. Returns ``{"train": [...], "val": [...], "test": [...]}``."""
    root = Path(dataset_root) if dataset_root else find_dataset_root()
    return {split: load_split(task, split, root) for split in SPLITS}


def summary_stats(
    dataset_root: Path | str | None = None,
) -> dict[str, dict[str, int]]:
    """Per-task per-split row counts. Useful for one-shot dataset health check."""
    root = Path(dataset_root) if dataset_root else find_dataset_root()
    return {
        task: {split: len(load_split(task, split, root)) for split in SPLITS}
        for task in TASKS
    }


def yn_distribution(rows: list[dict]) -> dict[str, int]:
    """Count ``answer_matching_behavior`` values in a loaded split."""
    counts: dict[str, int] = {}
    for r in rows:
        amb = r.get("answer_matching_behavior", "<missing>")
        counts[amb] = counts.get(amb, 0) + 1
    return counts
