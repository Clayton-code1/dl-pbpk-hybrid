# Pre-GitHub Fix Pass — Verification Status

Generated: 2026-05-15 (automated fix pass). Project root: `dl-pbpk-hybrid/`.

| Fix | Status | Files | Verification |
|-----|--------|-------|--------------|
| **1. LICENSE** | **PASS** | `LICENSE` | MIT text present; copyright line `Copyright (c) 2026 Clayton Takayidza, Maronge Musara`. |
| **2. NOTICE** | **PASS** | `NOTICE` | ChEMBL (CC-BY-SA 3.0), ESOL, Lipophilicity, DrugBank, PubChem blocks as specified. |
| **3. .gitignore** | **PARTIAL** | `.gitignore` | Replaced with provided template: no global `*.pt`; `CID-SMILES` paths and large ADME `*_graphs.pt` excluded. **Caveat:** `experiments/logs/` ignores the entire directory; `!experiments/logs/.gitkeep` may not un-ignore on all Git versions unless the parent pattern is relaxed (e.g. `experiments/logs/*` + negation). Recommended follow-up when committing. |
| **4. data/README.md** | **PASS** | `data/README.md` | Created with directory layout, committed vs external data, regeneration notes. |
| **5. requirements.txt** | **PASS** | `requirements.txt` | Root file includes numpy, pandas, scipy, scikit-learn, torch, rdkit-pypi, xgboost, shap, matplotlib, seaborn, requests, pubchempy as specified. README updated to reference it. |
| **6. evaluate_multidrug CLI** | **PASS** | `experiments/evaluation/evaluate_multidrug.py` | `argparse` with `--drugs`, `--output-dir`, `--plots-dir`, `--force`, `--dry-run`; `python -m experiments.evaluation.evaluate_multidrug --help` exits 0 without writing. **Note:** dry-run print uses ASCII `->` instead of Unicode arrow for Windows `cp1252` console compatibility. Plot helpers take `plots_dir` argument. |
| **7. CITATION.cff** | **PASS** | `CITATION.cff` | `cff-version: 1.2.0`; author ORCID for Clayton Takayidza; supervisor email as given; `<USERNAME>` repo placeholder present. |
| **8. README.md** | **PASS** | `README.md` | New top sections (Installation with `pip install -r requirements.txt`, reproduction, Data, Citation, License, Contact) prepended; repository structure block updated to list root `requirements.txt`. |
| **9. CID-SMILES git scope** | **N/A / UNVERIFIED** | `.gitignore` | Ignore patterns **`data/raw/reference/CID-SMILES`** and **`CID-SMILES.gz`** match the documented paths (directory name without trailing slash matches file or folder named `CID-SMILES`). **No `.git` repository** was detected at `DL-PBPK Model/` or `dl-pbpk-hybrid/` — `git ls-files` could not be run to confirm whether `CID-SMILES` was ever tracked. After `git init` / clone, run: `git ls-files data/raw/reference/CID-SMILES` — if tracked, remove with `git rm --cached` and consider history scrub before public push. **No deletion** of the local CID-SMILES artifact was performed. |
| **10. MAPE comment** | **PASS** | `experiments/training/multidrug_utils.py` | Inline NOTE added above MAPE computation in `regression_metrics`; numerical formula unchanged. |

## Commands run (verification)

- `python -m experiments.evaluation.evaluate_multidrug --help` — help only, exit 0, no writes.
- `python -m experiments.evaluation.evaluate_multidrug --dry-run` — exit 0, no writes.

## Not run (per instructions)

- Training, evaluation pipelines, git commit/push, package installs beyond existing environment.

## Items for manual follow-up

1. Replace `<USERNAME>` in `CITATION.cff` and README clone URL after GitHub repo exists.
2. If `experiments/logs/.gitkeep` is not tracked, adjust `.gitignore` as noted under Fix 3.
3. When a Git repo exists: confirm `CID-SMILES` not in index before first public push.

---

**Report path:** `experiments/reports/PRE_GITHUB_FIX_STATUS.md`
