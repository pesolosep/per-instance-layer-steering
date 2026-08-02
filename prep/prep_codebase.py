"""Build prep_codebase.ipynb — clone main, save as Kaggle Dataset.

Companion to prep_model.ipynb (which downloads model weights). This notebook
clones this repository's `main` branch into `/kaggle/working/`
so the resulting Kaggle Dataset contains JUST the codebase (~few MB).

Consumer notebooks (oracle_runner, ln_global_runner, prep_vectors, bf16_smoke_test)
attach this Dataset alongside the prep-model Dataset.

Output of this notebook: ``prep_codebase.ipynb`` next to this script.
"""
from pathlib import Path

import nbformat as nbf


def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(src.strip())


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src.strip())


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    nb.cells = []

    nb.cells.append(md("""
# Offline Prep (Codebase Only)

**Scope**: clone the `main` branch of this repository into
`/kaggle/working/` and commit as a Kaggle Dataset. **No model download.**

**Why a separate notebook from `prep_model.ipynb`**:
- Codebase is small (~few MB) and changes often during development.
- Model weights are large (~16 GB) and change rarely.
- Splitting them means a codebase update doesn't require re-downloading
  the model. Run this notebook on every codebase change; run `prep_model`
  only when switching model targets.

**Output Dataset**: contains only `layerselection-repo/` (the cloned repo). Attach
this Dataset to consumer notebooks (`oracle_runner`, `ln_global_runner`,
`prep_vectors`, `bf16_smoke_test`) alongside the prep-model Dataset.

**Pre-requisites** (Kaggle Secrets):
- `GITHUB_TOKEN` — to clone the repo
"""))

    nb.cells.append(md("## 1. Configuration"))
    nb.cells.append(code('''
REPO_HTTPS  = "github.com/ANONYMOUS/ANONYMIZED-REPO.git"
REPO_BRANCH = "main"        # branch to clone — pin to a tag for reproducibility
TARGET_DIR  = "layerselection-repo"            # subdir name under /kaggle/working/

print(f"branch: {REPO_BRANCH}")
print(f"target: /kaggle/working/{TARGET_DIR}/")
'''))

    nb.cells.append(md("## 2. Auth — Kaggle Secrets"))
    nb.cells.append(code('''
import os, sys, shutil, subprocess
from pathlib import Path

from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
GITHUB_TOKEN = secrets.get_secret("GITHUB_TOKEN")
print("GITHUB_TOKEN loaded.")
'''))

    nb.cells.append(md("## 3. Clone repo into `/kaggle/working/`"))
    nb.cells.append(code('''
WORKING  = Path("/kaggle/working")
REPO_DIR = WORKING / TARGET_DIR

if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)

subprocess.run(
    ["git", "clone", "--depth", "1", "--branch", REPO_BRANCH,
     f"https://{GITHUB_TOKEN}@{REPO_HTTPS}", str(REPO_DIR)],
    check=True,
)

# Sanity check the clone landed
sentinels = [
    REPO_DIR / "oracle" / "config.py",
    REPO_DIR / "prep"   / "download_model.py",
    REPO_DIR / "model_wrapper.py",
]
for s in sentinels:
    assert s.exists(), f"missing sentinel: {s}"

print(f"cloned {REPO_DIR}")
print(f"sentinels present: {[s.name for s in sentinels]}")
'''))

    nb.cells.append(md("## 4. Strip `.git/` (optional, halves the Dataset size)"))
    nb.cells.append(code('''
# .git/ is ~50% of the cloned directory's size. Strip it for the Dataset
# (the consumer notebooks don't need git history — they just import .py files).
# Comment this cell out if you want to preserve git history in the snapshot.

git_dir = REPO_DIR / ".git"
if git_dir.exists():
    shutil.rmtree(git_dir)
    print(f"stripped {git_dir}")
else:
    print(f"{git_dir} not present (already stripped or shallow clone with no .git)")
'''))

    nb.cells.append(md("## 5. Report size"))
    nb.cells.append(code('''
def _du(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6  # MB

total_mb = _du(REPO_DIR)
n_files  = sum(1 for _ in REPO_DIR.rglob("*") if _.is_file())
n_pyfiles = sum(1 for _ in REPO_DIR.rglob("*.py"))

print(f"Repo snapshot: {n_files} files ({n_pyfiles} .py) | {total_mb:.2f} MB")
print()
print("Top-level entries:")
for entry in sorted(REPO_DIR.iterdir()):
    if entry.is_dir():
        print(f"  {entry.name}/  ({_du(entry):.2f} MB)")
    else:
        print(f"  {entry.name}")
print()
print("Next: Save Version → Save & Run All (Commit).")
print("Attach the resulting Dataset to consumer notebooks alongside the prep-model Dataset.")
'''))

    return nb


if __name__ == "__main__":
    nb = build()
    out_path = Path(__file__).parent / "prep_codebase.ipynb"
    with open(out_path, "w") as f:
        nbf.write(nb, f)
    print(f"wrote {out_path} ({len(nb.cells)} cells)")
