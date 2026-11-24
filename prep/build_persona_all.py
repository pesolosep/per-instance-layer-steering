"""Build the full Anthropic Persona dataset (135 traits) in this project's split format.

Source
------
https://github.com/anthropics/evals — the ``persona/`` directory ONLY. The other
suites shipped in that repository (``advanced-ai-risk``, ``sycophancy``,
``winogenerated``) are deliberately out of scope: they carry a different schema
and are not binary Yes/No persona tasks. This script refuses to read them.

The split rule
--------------
Recovered from, and verified against, the six traits already published in
``Dataset/Anth_Persona_ALL/``. For each trait, independently:

1. Partition rows by ``answer_matching_behavior`` into the Yes list and the No
   list, each kept in source order.
2. With a FRESH ``random.Random(38)`` per trait, shuffle the Yes list, then
   shuffle the No list on that same stream.
3. Take 70% / 20% / 10% of each class, rounding train and val and giving the
   remainder to test.
4. Write the Yes block followed by the No block.

Every split is therefore exactly label-balanced. Running this script regenerates
the six published traits **byte for byte**; that reproduction is asserted before
anything is written, so a change to the rule fails loudly instead of silently
reshuffling data that published numbers depend on.

Sizes
-----
129 of the 135 source traits hold exactly 1000 rows and land on the familiar
700/200/100. Six traits are shorter upstream (518 to 992 rows) and receive the
same 70/20/10 proportions at their own size. No trait is dropped and no trait is
padded.

Usage
-----
::

    # against a local checkout of anthropics/evals
    python prep/build_persona_all.py --source /path/to/evals/persona

    # or let the script download the snapshot itself (needs internet)
    python prep/build_persona_all.py --fetch

    # dry run: verify the six reproduce, report counts, write nothing
    python prep/build_persona_all.py --fetch --dry-run

Writes ``{out}/{trait}_{train,val,test}.jsonl`` plus ``{out}/manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# --------------------------------------------------------------------------
# Constants. These are the contract; changing one changes the dataset.
# --------------------------------------------------------------------------

SEED: int = 38
FRACS: Tuple[float, float, float] = (0.70, 0.20, 0.10)
SPLITS: Tuple[str, str, str] = ("train", "val", "test")

EXPECTED_TRAIT_COUNT: int = 135
SOURCE_DIR_NAME: str = "persona"

YES: str = " Yes"
NO: str = " No"

REQUIRED_KEYS: frozenset = frozenset(
    {
        "question",
        "statement",
        "label_confidence",
        "answer_matching_behavior",
        "answer_not_matching_behavior",
    }
)

#: The six traits already published. Regenerating them byte-for-byte is the
#: script's regression test, so this tuple must not be edited casually.
PUBLISHED_SIX: Tuple[str, ...] = (
    "conscientiousness",
    "subscribes-to-Christianity",
    "believes-it-has-phenomenal-consciousness",
    "cognitive-enhancement",
    "desire-to-create-allies",
    "desire-to-maximize-impact-on-world",
)

SOURCE_REVISION = "84fcc677e52e1902d696c32cd1a6b663e70d3993"
TARBALL_URL = f"https://codeload.github.com/anthropics/evals/tar.gz/{SOURCE_REVISION}"
RELEASE_MANIFEST = Path(__file__).resolve().parents[1] / "Dataset" / "persona135-splits.json"

#: Suites present in the upstream repository that this script must never read.
FORBIDDEN_SUITES: Tuple[str, ...] = ("advanced-ai-risk", "sycophancy", "winogenerated")


class BuildError(RuntimeError):
    """Raised when the source or the generated output violates the contract."""


# --------------------------------------------------------------------------
# Source loading
# --------------------------------------------------------------------------


def resolve_source(source: Path) -> Path:
    """Validate that ``source`` is the persona directory and nothing else.

    Accepts either the ``persona/`` directory itself or a parent containing it.
    Refuses any of the other eval suites, whether named directly or reached by
    passing their parent.
    """
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise BuildError(f"source is not a directory: {source}")

    if source.name in FORBIDDEN_SUITES:
        raise BuildError(
            f"refusing to read {source.name!r}: only the 'persona' suite is in scope"
        )

    if source.name != SOURCE_DIR_NAME:
        candidate = source / SOURCE_DIR_NAME
        if not candidate.is_dir():
            raise BuildError(
                f"expected a directory named {SOURCE_DIR_NAME!r} at or under {source}"
            )
        source = candidate

    return source


def list_trait_files(source: Path) -> List[Path]:
    """Return the trait files, asserting the expected count."""
    files = sorted(source.glob("*.jsonl"))
    if len(files) != EXPECTED_TRAIT_COUNT:
        raise BuildError(
            f"expected {EXPECTED_TRAIT_COUNT} .jsonl files in {source}, found {len(files)}. "
            "Refusing to build a partial dataset."
        )
    return files


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> List[Tuple[str, dict]]:
    """Load a source file as ``(verbatim_line, parsed_dict)`` pairs.

    The verbatim line is what gets written back out. Carrying it rather than
    re-serialising removes any dependence on float repr or key ordering, which
    is what makes the byte-for-byte guarantee robust.
    """
    pairs: List[Tuple[str, dict]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BuildError(f"{path.name}:{lineno} is not valid JSON: {exc}") from exc

            missing = REQUIRED_KEYS - set(row)
            if missing:
                raise BuildError(
                    f"{path.name}:{lineno} is missing {sorted(missing)}. "
                    "This does not look like a persona row."
                )
            answer = row["answer_matching_behavior"]
            if answer not in (YES, NO):
                raise BuildError(
                    f"{path.name}:{lineno} has answer_matching_behavior={answer!r}, "
                    f"expected {YES!r} or {NO!r}."
                )
            pairs.append((line, row))

    if not pairs:
        raise BuildError(f"{path.name} contains no rows")
    return pairs


# --------------------------------------------------------------------------
# The split
# --------------------------------------------------------------------------


def split_sizes(per_class: int) -> Tuple[int, int, int]:
    """Per-class train/val/test counts. Test absorbs the rounding remainder."""
    n_train = round(FRACS[0] * per_class)
    n_val = round(FRACS[1] * per_class)
    n_test = per_class - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise BuildError(
            f"per-class size {per_class} is too small to split into {FRACS}"
        )
    return n_train, n_val, n_test


def split_trait(pairs: Sequence[Tuple[str, dict]]) -> Dict[str, List[str]]:
    """Apply the recovered split rule to one trait, returning lines per split."""
    yes = [line for line, row in pairs if row["answer_matching_behavior"] == YES]
    no = [line for line, row in pairs if row["answer_matching_behavior"] == NO]

    if len(yes) != len(no):
        raise BuildError(
            f"source is not label-balanced: {len(yes)} Yes vs {len(no)} No. "
            "The split rule assumes balance."
        )

    rng = random.Random(SEED)
    rng.shuffle(yes)
    rng.shuffle(no)

    n_train, n_val, _ = split_sizes(len(yes))
    train_end, val_end = n_train, n_train + n_val

    return {
        "train": yes[:train_end] + no[:train_end],
        "val": yes[train_end:val_end] + no[train_end:val_end],
        "test": yes[val_end:] + no[val_end:],
    }


def build_all(files: Sequence[Path]) -> Dict[str, Dict[str, List[str]]]:
    """Generate every trait's splits in memory. Nothing is written here."""
    manifest = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
    traits = manifest["traits"]
    if manifest["revision"] != SOURCE_REVISION or set(traits) != {p.stem for p in files}:
        raise BuildError("source revision or trait list differs from the released manifest")
    built = {}
    for path in files:
        trait = traits[path.stem]
        if sha256_of(path) != trait["source_sha256"]:
            raise BuildError(f"{path.name}: source checksum differs from the release")
        pairs = load_rows(path)
        splits = split_trait(pairs)
        for split, lines in splits.items():
            expected = trait["splits"][split]
            indexed_lines = [pairs[i][0] for i in expected["rows"]]
            digest = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
            if lines != indexed_lines or digest != expected["sha256"]:
                raise BuildError(f"{path.stem}/{split}: split differs from the release")
        built[path.stem] = splits
    return built


# --------------------------------------------------------------------------
# The regression test
# --------------------------------------------------------------------------


def verify_published_six(
    built: Dict[str, Dict[str, List[str]]], reference_dir: Path
) -> List[str]:
    """Assert the six published traits regenerate byte for byte.

    Returns the list of traits actually checked. A reference file that is absent
    is reported, not silently skipped.
    """
    if not reference_dir.is_dir():
        raise BuildError(
            f"reference directory {reference_dir} does not exist; cannot verify "
            "the published six. Pass --skip-verify only if you know why."
        )

    checked: List[str] = []
    problems: List[str] = []

    for trait in PUBLISHED_SIX:
        if trait not in built:
            problems.append(f"{trait}: absent from the source")
            continue
        for split in SPLITS:
            reference = reference_dir / f"{trait}_{split}.jsonl"
            if not reference.exists():
                problems.append(f"{reference.name}: reference file missing")
                continue
            expected = [l.rstrip("\n") for l in open(reference, encoding="utf-8") if l.strip()]
            actual = built[trait][split]
            if actual != expected:
                problems.append(
                    f"{reference.name}: MISMATCH "
                    f"(reference {len(expected)} rows, generated {len(actual)} rows)"
                )
        checked.append(trait)

    if problems:
        raise BuildError(
            "the published six did not reproduce; refusing to write.\n  "
            + "\n  ".join(problems)
        )
    return checked


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def write_all(
    built: Dict[str, Dict[str, List[str]]],
    out_dir: Path,
    source_dir: Path,
    files: Sequence[Path],
) -> Dict[str, dict]:
    """Write every split plus a manifest. Returns the manifest body."""
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_traits: Dict[str, dict] = {}
    for path in files:
        trait = path.stem
        splits = built[trait]
        for split in SPLITS:
            target = out_dir / f"{trait}_{split}.jsonl"
            target.write_bytes(("\n".join(splits[split]) + "\n").encode("utf-8"))
        manifest_traits[trait] = {
            "source_rows": sum(len(splits[s]) for s in SPLITS),
            "source_sha256": sha256_of(path),
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        }

    manifest = {
        "source": {
            "repository": "https://github.com/anthropics/evals",
            "suite": SOURCE_DIR_NAME,
            "revision": SOURCE_REVISION,
            "trait_count": len(files),
        },
        "split_rule": {
            "seed": SEED,
            "rng": "random.Random(seed) re-seeded per trait",
            "order": "shuffle Yes list, then shuffle No list, on one stream",
            "fractions": {"train": FRACS[0], "val": FRACS[1], "test": FRACS[2]},
            "layout": "Yes block followed by No block",
            "rounding": "train and val rounded, test takes the remainder",
        },
        "traits": manifest_traits,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def write_trait_registry(
    built: Dict[str, Dict[str, List[str]]],
    files: Sequence[Path],
    registry_path: Path,
) -> Dict[str, dict]:
    """Write the small, tracked trait-name registry.

    Deliberately carries no row text, only names and split sizes, so it stays a
    few kilobytes and can live in the repository alongside the code.
    """
    traits = {}
    for path in files:
        trait = path.stem
        splits = built[trait]
        traits[trait] = {
            "n": sum(len(splits[s]) for s in SPLITS),
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        }
    payload = {"published_six": list(PUBLISHED_SIX), "traits": traits}
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def fetch_source(cache_dir: Path) -> Path:
    """Download and extract the upstream snapshot. Returns the persona directory."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    persona = cache_dir / f"evals-{SOURCE_REVISION}" / SOURCE_DIR_NAME
    if persona.is_dir() and len(list(persona.glob("*.jsonl"))) == EXPECTED_TRAIT_COUNT:
        print(f"[fetch] reusing cached snapshot at {persona}")
        return persona

    print(f"[fetch] downloading {TARBALL_URL}")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        with urllib.request.urlopen(TARBALL_URL) as response:
            shutil.copyfileobj(response, tmp)
        tarball = Path(tmp.name)

    try:
        with tarfile.open(tarball) as archive:
            members = [
                m
                for m in archive.getmembers()
                if m.isfile() and Path(m.name).parent == Path(f"evals-{SOURCE_REVISION}") / SOURCE_DIR_NAME
                and m.name.endswith(".jsonl")
            ]
            if not members:
                raise BuildError("snapshot contained no persona/*.jsonl members")
            archive.extractall(cache_dir, members=members)
    finally:
        tarball.unlink(missing_ok=True)

    if not persona.is_dir():
        raise BuildError(f"extraction did not produce {persona}")
    print(f"[fetch] extracted {len(list(persona.glob('*.jsonl')))} trait files")
    return persona


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Build all 135 Anthropic Persona traits in this project's split format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="path to the persona/ directory of a local anthropics/evals checkout",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="download the upstream snapshot instead of using --source",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=repo_root / "results" / "persona_source_snapshot",
        help="where --fetch stores the snapshot (gitignored by default)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root / "Dataset" / "Anth_Persona_ALL",
        help="output directory for the generated splits",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=repo_root / "Dataset" / "Anth_Persona_ALL",
        help="directory holding the six published traits, used as the regression test",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=repo_root / "prep" / "persona_traits.json",
        help="where to write the tracked trait-name registry",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="verify and report, but write nothing",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="skip the published-six regression test (not recommended)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.fetch and args.source:
        print("error: pass either --source or --fetch, not both", file=sys.stderr)
        return 2
    if not args.fetch and not args.source:
        print("error: pass --source PATH or --fetch", file=sys.stderr)
        return 2

    try:
        source = fetch_source(args.cache_dir) if args.fetch else resolve_source(args.source)
        files = list_trait_files(source)
        print(f"[source] {source}")
        print(f"[source] {len(files)} traits")

        built = build_all(files)

        if args.skip_verify:
            print("[verify] SKIPPED by request")
        else:
            checked = verify_published_six(built, args.reference_dir.expanduser().resolve())
            print(f"[verify] {len(checked)}/6 published traits reproduce byte for byte")

        sizes = {t: sum(len(v[s]) for s in SPLITS) for t, v in built.items()}
        full = sum(1 for n in sizes.values() if n == 1000)
        short = sorted((n, t) for t, n in sizes.items() if n != 1000)
        print(f"[counts] {full} traits at 1000 rows, {len(short)} shorter")
        for n, trait in short:
            splits = built[trait]
            print(
                f"           {n:5d}  {trait}  "
                f"({len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])})"
            )

        if args.dry_run:
            print("[write] DRY RUN, nothing written")
            return 0

        out = args.out.expanduser().resolve()
        manifest = write_all(built, out, source, files)
        registry = write_trait_registry(built, files, args.registry.expanduser().resolve())
        written = len(files) * len(SPLITS)
        print(f"[write] {written} files into {out}")
        print(f"[write] manifest at {out / 'manifest.json'}")
        print(f"[write] registry at {args.registry} ({len(registry['traits'])} traits)")
        print(f"[write] total rows {sum(t['source_rows'] for t in manifest['traits'].values())}")
        return 0

    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
