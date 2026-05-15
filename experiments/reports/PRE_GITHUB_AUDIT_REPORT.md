# Pre-GitHub publication — reproducibility & code health audit

**One-line summary (most critical):** There is **no `LICENSE` file at the repository root** of `dl-pbpk-hybrid/`, and **`data/raw/reference/CID-SMILES` is on the order of gigabytes**, which is incompatible with normal GitHub hosting without an external data policy.

**Issue counts (by severity, across all categories):** CRITICAL — **5** · HIGH — **11** · MEDIUM — **18** · LOW — **6**

**Audit scope:** Diagnostic inspection only. Commands run on **2026-05-15**. Canonical commands are taken from `paper/reproducibility_checklist.md` (project root = `dl-pbpk-hybrid/`). **Supplementary Section S5** was not found as a separate file in `paper/` (only `paper_skeleton.md`, `mathematical_formulation.md`, `reproducibility_checklist.md`); findings reference the checklist as the primary command source.

**Incident (audit execution):** A smoke command intended to be read-only (`python -m experiments.evaluation.evaluate_multidrug --help`) **did not parse `--help`** and instead **ran the full Phase 1.6 evaluation**, overwriting `experiments/results/phase1_multidrug_metrics.csv` and plot files under `experiments/plots/`. This is reported as a **HIGH** reproducibility/CLI-design finding; restore prior artifacts from version control if those paths must remain unchanged.

---

## CATEGORY 1 — End-to-End Pipeline Sanity Check

### 1.1 Dry-run execution

For each command below: **script/module path verified**, **`ast.parse` of the `.py` file succeeded** (read as UTF-8, no syntax errors). The repository root assumed is **`dl-pbpk-hybrid/`**.

| Command / module | File path | `__main__` / CLI | Committed inputs checked |
|------------------|-----------|------------------|---------------------------|
| `python -m experiments.data.download_pk_data` | `experiments/data/download_pk_data.py` | **Yes.** `argparse`: `--drugs` (optional list); default = `DRUGS` + external `ibuprofen`. | **HIGH — network + regeneration:** writes CSVs under `experiments/data/processed/`. No static “input CSV” is required to start; the script calls **PubChem/HTTP** (and optionally other APIs per code). `experiments/reference_pk.py` provides literature fallbacks. **MEDIUM:** full rerun needs outbound HTTP for “best-effort” enrichment. |
| `python -m experiments.data.featurize_drugs` | `experiments/data/featurize_drugs.py` | **Yes**, but **no CLI arguments** (`main()` loops all drugs). | **HIGH:** expects per-drug PK CSVs such as `experiments/data/processed/{drug}_pk_dataset.csv` (from Phase 1.3). **On this machine those CSVs and `experiments/data/processed/graphs/*.pt` are present.** |
| `python -m experiments.training.train_multidrug_hybrid` | `experiments/training/train_multidrug_hybrid.py` | **Yes.** `argparse`: `--drugs`, `--max-epochs` (see file ~511+). | **CRITICAL / HIGH — missing single consolidated spec:** requires processed PK data + graphs + **pretrained encoder** paths (e.g. `hybrid_gnn_pbpk_theoph_combined_v1` / `gnn_pretrain_combined_v1` per comments). **Weights present locally** under `artifacts/models/` (see Category 3). |
| `python -m experiments.evaluation.evaluate_multidrug` | `experiments/evaluation/evaluate_multidrug.py` | **No `argparse`.** Running the module **always executes evaluation** (and **writes** CSV + plots). | **HIGH:** requires `artifacts/models/hybrid_gnn_pbpk_{drug}_v1/model.pt`, processed PK splits, graphs — **present on this machine.** |
| Phase 2 baselines / ablations | `experiments/baselines/train_baselines.py`, `experiments/ablation/ablation_study.py`, `experiments/phase2/external_ibuprofen.py` | **`train_baselines` / `ablation_study`:** no `argparse` found in quick scan — drivers run as scripts. **`external_ibuprofen`:** parseable; see module. | **HIGH:** depend on Phase 1/2 artifacts, caches (e.g. `phase2_prediction_cache.pkl` referenced in `significance_tests` docstring). Verify caches exist before claiming full Phase 2 reproduction from a cold clone. |
| `python -m experiments.statistics.significance_tests` | `experiments/statistics/significance_tests.py` | **Yes (`argparse`).** Uses `--prediction-cache`, `--output-csv` per module docstring. | **HIGH:** requires `experiments/results/phase2_prediction_cache.pkl` **unless** paths are overridden — cache may be missing in a fresh clone. |
| `python -m experiments.safety.safety_thresholds` | `experiments/safety/safety_thresholds.py` | **parse OK** (detailed CLI not extracted in this pass). | Outputs `experiments/results/phase3_safety_thresholds.md` (**exists** on disk). |
| `python -m experiments.uncertainty.monte_carlo_calibration` | `experiments/uncertainty/monte_carlo_calibration.py` | **parse OK.** | Depends on models + processed data; **`phase3_uncertainty_calibration.csv` exists**. |
| `python -m experiments.explainability.shap_interpretation` | `experiments/explainability/shap_interpretation.py` | **parse OK.** | Outputs include `experiments/results/phase3_shap_interpretation.md` (**exists**). |

**Additional `ast.parse` checks (Phase 2 supporting scripts):**  
`experiments/baselines/correct_pbpk_realistic.py` — **OK** (syntax).

**Findings:**

- **HIGH — `evaluate_multidrug` has no CLI and no `--help`.** Passing `--help` still runs the full job and **writes outputs** (confirmed during this audit).
- **HIGH — split dependency specifications:** training/evaluation code lives in `experiments/` but **root-level dependency story is fragmented** (see Category 2).
- **MEDIUM — `featurize_drugs` has no flags** for subsetting drugs (always processes full panel + external drug).

### 1.2 Quick smoke test

All commands run from **`dl-pbpk-hybrid/`** with the system Python (**3.11.0**).

**Test 1 — Instantiate GNN**

```text
GNN instantiated. Parameter count: 484224
```

(Used `MoleculeGNN(node_feat_dim=27, edge_feat_dim=6)` — **not** a zero-arg constructor.)

**Test 2 — Load checkpoint**

```text
<class 'collections.OrderedDict'>
Checkpoint loaded. Keys: ['gnn.node_encoder.weight', 'gnn.node_encoder.bias', 'gnn.layers.0.edge_mlp.mlp.0.weight', 'gnn.layers.0.edge_mlp.mlp.0.bias', 'gnn.layers.0.edge_mlp.mlp.2.weight', 'gnn.layers.0.edge_mlp.mlp.2.bias', 'gnn.layers.0.gru.weight_ih', 'gnn.layers.0.gru.weight_hh', 'gnn.layers.0.gru.bias_ih', 'gnn.layers.0.gru.bias_hh']
```

Path: `artifacts/models/hybrid_gnn_pbpk_theoph_combined_v1/model.pt`.

**Test 3 — Inference (read-only forward)**

After constructing `MultiDrugHybridGNNPBPK` from `artifacts/models/hybrid_gnn_pbpk_theophylline_v1/config.json` and loading `model.pt`, a single-patient forward pass succeeded:

```text
Forward OK torch.Size([13]) 3.449276924133301 36.30751419067383 1.4260393381118774
```

**Note:** A naive batched `[1,5]` patient feature tensor **failed** with a dimension mismatch; the evaluation path uses **1D** per-patient tensors (see `PatientRecord` / `evaluate_multidrug.py`). **MEDIUM — API ergonomics** for external users reimplementing inference.

---

## CATEGORY 2 — Dependency Management

### 2.1 requirements.txt audit

**HIGH — No `requirements.txt` at `dl-pbpk-hybrid/` repository root.**  
The pinned scientific stack is under **`src/requirements.txt`**:

```text
numpy>=1.24,<2
torch>=2.2,<3
pandas>=2.0,<3
scikit-learn>=1.3
matplotlib>=3.7
rdkit-pypi>=2022.9.2
```

**`pip install --dry-run -r src/requirements.txt`** succeeded (resolver reported already-satisfied requirements). **No pip conflicts** were reported for that file.

**HIGH — Experiments and API require additional packages** not listed in `src/requirements.txt`, including but not limited to **`requests`**, **`scipy`**, **`shap`**, **`xgboost`**, **`seaborn`**, **`pubchempy`** (confirmed via static import scan of `experiments/` + `src/` excluding `.venv`). **`api/requirements.txt`** adds FastAPI stack + **`shap`**, etc., but still **does not round-trip all experiment imports by itself**.

**MEDIUM — No monolithic reproducible lockfile** (e.g. `requirements.lock` / `conda-lock`) at repo root for the *full* paper pipeline.

### 2.2 Python version compatibility

- **No `pyproject.toml`, `setup.py`, or `setup.cfg`** declaring `python_requires` was found at **`dl-pbpk-hybrid/`** root (only third-party trees under `.venv` contain such files).
- **Development Python observed:** `Python 3.11.0` (from `python --version` during audit).
- **MEDIUM — Python 3.10+ typing syntax** is used in `experiments/` (e.g. `numpy.ndarray | None`, `Tensor | None`), which **breaks on Python 3.9** without retro-fitting `from __future__ import annotations` + `Optional[...]` or `Union[...]`.

### 2.3 Hidden or system-only dependencies

| Finding | Severity | File : line |
|--------|----------|-------------|
| `subprocess.check_call` invokes **`cache_adme_graphs.py`** when `--rebuild-cache` is used in GNN ADME pretraining. | **MEDIUM** | `src/training/pretrain_gnn_adme.py` : **281** |
| **No `boto3` / `google-cloud*` / `azure`** usage in **`experiments/`** or first-party **`src/`** (outside vendored `.venv`). | — | — |
| **Network HTTP** for PubChem / APIs in `download_pk_data.py`. | **MEDIUM** (third-party must have internet for “fresh” metadata pass) | `experiments/data/download_pk_data.py` (imports `requests`) |

---

## CATEGORY 3 — Data and Model Artifact Availability

### 3.1 Trained model artifacts inventory

Root: `artifacts/models/`. **Per-folder totals (MB, approximate):**

| Folder | ~Total MB | `.pt` / weights | `metrics.json` / `config.json` | Notes |
|--------|-----------|-------------------|----------------------------------|-------|
| `hybrid_gnn_pbpk_theoph_v1` | **1.91** | `model.pt` (~2.0e6 B) largest | **Present** (json metrics/config) | Legacy single-drug theophylline hybrid (per README path) |
| `hybrid_gnn_pbpk_*_v1` (6 drugs) | ~0.37 each | `model.pt` | `metrics.json`, `config.json`, `scaler.json` typically | **Appears complete** |
| `hybrid_gnn_pbpk_theoph_combined_v1` | ~0.37 | `model.pt` | json sidecars | Multidrug encoder source for fine-tuning |
| `gnn_pretrain_unsup_v1`, `gnn_pretrain_combined_v1`, `gnn_pretrain_v1` | ~0.34 each | `model_gnn.pt` or `model.pt` | `metrics.json` **present** | |
| `phase2_ablation_A3_*`, `phase2_ablation_A4_*` | ~0.37 each | `model.pt` | json sidecars | |
| `hybrid_theoph_v1` | ~0.21 | includes `model.pt` (per glob) | | Older MLP/ plots assets |

**Largest single checkpoint observed:** ~**2.0 MB** (`hybrid_gnn_pbpk_theoph_v1/model.pt`). **None of the inspected `.pt` files exceed GitHub’s 100 MB per-file cap.**

**HIGH — `.gitignore` contains `*.pt` / `*.pth`:** default Git workflows will **exclude all weight files** unless teams use **Git LFS**, **force-add**, or **remove/adjust ignores**. This is a **distribution strategy** issue for “clone and run”.

### 3.2 Dataset files inventory

Key directories:

| Location | Role | Approx. notable sizes (audit host) |
|---------|------|-----------------------------------|
| `experiments/data/processed/` | PK CSVs + `graphs/*.pt` | Each PK CSV **~0.4 MB**; graphs **~6–23 KB** each |
| `data/processed/adme_pretrain/` | ADME / unsup corpora + caches | See below |
| `data/raw/adme_pretrain/train.csv` | ChEMBL-style raw for unsupervised pool | **~79 MB** |
| `data/raw/reference/CID-SMILES` | Reference CID–SMILES table | **~8.17 GB** (8097093493 B) — **CRITICAL for GitHub** |
| `data/raw/reference/drugbank_all_drugbank_vocabulary.csv/` | DrugBank vocabulary snippet | **~3 MB** — **MEDIUM — verify DrugBank licence** before redistribution |

**Files > 100 MB (must not be committed as plain Git blobs):**

- `data/raw/reference/CID-SMILES` (**~8.17 GB**) — **CRITICAL**
- `data/processed/adme_pretrain/adme_unsupervised_sample_10k_graphs.pt` (**~110 MB**) — **HIGH**
- `data/processed/adme_pretrain/adme_unsupervised.csv` (**~108 MB**) — **HIGH**

### 3.3 ChEMBL corpus handling

- **Where it lives:** Scripts reference **`data/raw/adme_pretrain/train.csv`** (`prepare_large_adme_corpus.py`: `CHEMBL_TRAIN_PATH`). That path **exists** on this machine (~79 MB).
- **Committed vs external:** Treated as **local raw input** under `data/raw/` (not fetched by this audit’s downloaders). Publication plan must state **whether this file is redistributed** or regenerated.
- **Licence:** ChEMBL data at EBI are typically under **CC-BY-SA 3.0** (verify against the **exact ChEMBL release** used). **Share-alike obligations apply on redistribution / derivative databases.** **HIGH — legal/compliance review** before posting full corpora on GitHub or Zenodo.

### 3.4 Public-facing data plan (recommendations)

| Artifact | Size class | Recommendation |
|---------|------------|----------------|
| Per-drug PK CSVs + small `.pt` graphs | < 100 MB | **Commit** or **Zenodo** with checksums; document regeneration via `download_pk_data` / `featurize_drugs`. |
| `train.csv` (ChEMBL-derived) | < 100 MB but licence-sensitive | **Zenodo** + **CC-BY-SA attribution text** in repo **or** document **download script** from official ChEMBL FTP/API with version pin. |
| `adme_unsupervised*.csv` / large graph caches | > 100 MB | **Zenodo** or **Git LFS**; prefer Zenodo for **DOI** + size. |
| `CID-SMILES` (~8 GB) | Far exceeds GitHub limits | **Never plain Git.** **External archive** (Zenodo / institutional repository / Hugging Face dataset) + **checksum** + **documentation**; **or** **omit** and document **how to obtain** PubChem dumps. |
| Model `.pt` files | < 100 MB each but many files | **Git LFS** **or** **Zenodo** model bundle **or** **Hugging Face Hub** + **`git clone` + download script** in README. |

---

## CATEGORY 4 — Sensitive Information Scan

### 4.1 API keys, tokens, and secrets

- **No matches** for `sk-`, `ghp_`, `AKIA`, `xoxb-`, `api_key=`, `secret=`, `password=` in **`experiments/`** during pattern grep.
- **`.env` / `.env.local`** are **gitignored** (see Category 5) — good hygiene; confirm no tracked copies under other names.

### 4.2 Personal information

- **No email addresses** matched in `paper/` or `experiments/reports/` via a simple `*@*.*` pattern scan.

### 4.3 Internal paths and URLs

- **No first-party hard-coded `C:\Users\...` or `/home/...` paths** under `experiments/`, `scripts/`, `paper/`, or `api/app/` in this audit’s searches.
- **Note:** **`.venv/**` packages contain illustrative absolute paths** (third-party); keep **`.venv/` out of public git**.

### 4.4 Generated cache and IDE files

**Examples present under `dl-pbpk-hybrid/` (list only — not deleted per instructions):**

- `.pytest_cache/` — **4** child items under root cache folder.
- `src/.venv/`, `api/.venv/` — full virtual environments (**must not** publish as part of the research codebase).
- `frontend/node_modules/` — Node dependencies (**must not** commit).
- Standard Python **`__pycache__/`** may exist under import trees (not exhaustively enumerated here).

---

## CATEGORY 5 — Existing .gitignore Coverage

**File:** `dl-pbpk-hybrid/.gitignore`

**Current contents (verbatim summary):** Python bytecode/eggs, `venv/.venv`, `node_modules`, `.next/out`, `.env` files, **`data/raw/*` + `data/processed/*` (with `.gitkeep` exceptions)**, IDE dirs (`.vscode/`, `.idea/`), Docker `pgdata/`, OS cruft, **all `*.pt` / `*.pth` / `*.onnx` / `*.h5`**.

**Comparison to GitHub `Python.gitignore` (fetched 2026-05-15):** project file **misses several common patterns** from the upstream template, including **`.pytest_cache/`**, **`.mypy_cache/` / `.ruff_cache/`**, **`__pycache__/` explicit** (partially covered via `*.py[cod]`), **`htmlcov/`**, **Jupyter `.ipynb_checkpoints/`**, **`*.log` / coverage artefacts**, **`.Python` / `wheels/` tree**, **`pip` installers / `MANIFEST`**, **`.pdm-python` / `.pixi/*`**, **`marimo` / Streamlit secrets**, etc.

**CRITICAL — data ignore vs reproducibility:** ignoring **`data/raw/*` and `data/processed/*`** means **default `git clone` will not contain** ChEMBL/ADME intermediates or PK CSVs unless you **force-add**, **Git LFS**, **submodule**, or **document downloads**. Align ignore rules with the **public data plan** (Category 3.4).

**CRITICAL — model ignores:** `*.pt` ignore blocks sharing **all model weights** unless LFS or separate release.

---

## CATEGORY 6 — Code Quality and Documentation

### 6.1 README at repo root

- **Exists:** `dl-pbpk-hybrid/README.md`.
- **Contents summary:** Repository layout; **Docker** and **Make** quick start; API/frontend **dev** commands; tech stack (**mentions PostgreSQL/SQLalchemy** — verify still accurate vs `api/` code).
- **Gaps (journal readiness):** **MEDIUM** — no **citation / CITATION.cff** block, **no explicit licence section** pointing to a `LICENSE` file, and **no single canonical Python environment** tying **`experiments/` + `src/` + `api/`** together.

### 6.2 LICENSE file

- **CRITICAL — No `LICENSE` or `LICENSE.txt` at `dl-pbpk-hybrid/` root** (`Test-Path` = `False`). Default copyright posture is **“all rights reserved”** for third parties until a licence is added.

### 6.3 Module docstrings

**Deterministic sample (10 files) — module has top-level docstring?**

| Module | Top-level docstring |
|--------|---------------------|
| `experiments/config.py` | **Yes** |
| `experiments/reference_pk.py` | **Yes** |
| `experiments/models/hybrid_multidrug.py` | **Yes** |
| `experiments/phase2/utils.py` | **Yes** |
| `experiments/statistics/significance_tests.py` | **Yes** |
| `experiments/baselines/train_baselines.py` | **Yes** |
| `experiments/data/featurize_drugs.py` | **Yes** |
| `experiments/evaluation/evaluate_multidrug.py` | **Yes** |
| `experiments/data/download_pk_data.py` | **Yes** |
| `src/molecules/rdkit_graph.py` | **Yes** |

**Rough docstring coverage (this sample):** **10/10 = 100%** module docstrings. **Public API docstrings** were **not** exhaustively scored; major classes like `MultiDrugHybridGNNPBPK` include docstrings (**MEDIUM** to run `ruff`/API doc audit later).

### 6.4 Dead code and TODOs

- **`TODO` / `FIXME` / `XXX` / `HACK` in `experiments/*.py`:** **none found** in a project-scoped search.
- **`pyflakes`:** **not installed** in the active environment — **MEDIUM — skipped unused-import analysis** per instructions (no new package installs).

### 6.5 Hardcoded paths

- **No project-authored absolute home-directory paths** found under `experiments/`, `scripts/`, `paper/`, `api/app/` in targeted searches.
- **MEDIUM — Windows-specific examples in README** (`src\.venv\Scripts\python`, PowerShell) — POSIX users need parallel commands.

---

## CATEGORY 7 — Specific Manuscript-Result Reproducibility

### Backing artifact table

| Manuscript result | Expected backing file | Exists? |
|-------------------|----------------------|---------|
| Phase 1 multi-drug metrics (Table 1) | `experiments/results/phase1_multidrug_metrics.csv` | **yes** |
| Benchmark comparison (Table 2) | `experiments/results/phase2_benchmark_metrics_final.csv` | **yes** |
| Ablation summary (Table 3) | `experiments/results/phase2_ablation_summary_final.csv` | **yes** |
| Significance tests (Table 4) | `experiments/results/phase2_statistical_tests_final.csv` | **yes** |
| External validation (§4.5) | `experiments/results/phase2_external_validation.csv` | **yes** |
| Uncertainty calibration (Table 5) | `experiments/results/phase3_uncertainty_calibration.csv` | **yes** |
| SHAP interpretation (Table 6) | `experiments/results/phase3_shap_interpretation.md` | **yes** |
| Pretraining unsupervised metrics | `artifacts/models/gnn_pretrain_unsup_v1/metrics.json` | **yes** |
| Pretraining combined metrics | `artifacts/models/gnn_pretrain_combined_v1/metrics.json` | **yes** |
| Per-drug hybrid models (×6) | `artifacts/models/hybrid_gnn_pbpk_<drug>_v1/` | **yes** for `theophylline`, `warfarin`, `midazolam`, `caffeine`, `acetaminophen`, `digoxin` |

### CSV integrity (pandas `read_csv`, non-empty)

```text
experiments/results/phase1_multidrug_metrics.csv
  rows: 6 cols: ['drug', 'n_test_patients', 'obs_mean_mg_L', 'RMSE', ...]

experiments/results/phase2_benchmark_metrics_final.csv
  rows: 36 cols: ['drug', 'model', 'RMSE', 'RMSE_pct_of_mean', ...]

experiments/results/phase2_ablation_summary_final.csv
  rows: 5 cols: ['variant', 'mean_R2_6drugs', 'mean_RMSE_6drugs']

experiments/results/phase2_statistical_tests_final.csv
  rows: 60 cols: ['drug', 'baseline', 'test', 'statistic', 'p_value', ...]

experiments/results/phase2_external_validation.csv
  rows: 1 cols: ['drug', 'split', 'n_patients', 'RMSE', ...]

experiments/results/phase3_uncertainty_calibration.csv
  rows: 105 cols: ['scope', 'nominal_interval_frac', 'empirical_coverage', ...]
```

**MEDIUM — `phase1_multidrug_metrics.csv` row for `acetaminophen` shows an extreme `MAPE` in stdout from the incidental re-evaluation run** (numerical instability / low mean in denominator). Investigate whether this is benign or indicates a **metric computation edge case**.

---

## CATEGORY 8 — Pre-GitHub Publication Checklist Recommendations

| Priority | Action | File(s) affected | Estimated time |
|----------|--------|------------------|----------------|
| **P0** | Add an **OSI-approved open-source `LICENSE`** at `dl-pbpk-hybrid/` root (match code + data policy). | `LICENSE` | 30–60 min (legal/authors) |
| **P0** | **Remove or relocate `data/raw/reference/CID-SMILES` (~8 GB)** from the default public Git scope; publish via **Zenodo/institutional store** + checksum + download script, or document regeneration only. | `data/raw/reference/`, README, data docs | 1–2 days |
| **P0** | Decide **model + data distribution**: **`*.pt` is gitignored** — adopt **Git LFS**, **Zenodo**, or **HF Hub** + manifest; update README. | `.gitignore`, `README.md`, `artifacts/models/` | 0.5–2 days |
| **P0** | Produce **one pinned dependency file** covering **`experiments/` + `src/` + optional `api/`** (or separate files + table in README). | `requirements.txt` / `requirements-experiments.txt`, docs | 2–4 h |
| **P1** | Add **`argparse` + `--help`** to `evaluate_multidrug.py`; never **overwrite** results without `--force`. | `experiments/evaluation/evaluate_multidrug.py` | 1–2 h |
| **P1** | **ChEMBL / share-alike compliance:** add **`NOTICE`/attribution** and verify **CC-BY-SA 3.0** obligations for any redistributed ChEMBL derivative (`train.csv`, unsupervised CSV). | `NOTICE`, `README.md` | 1–2 h |
| **P1** | **Zenodo (recommended)** archive for **>100 MB** artefacts (`adme_unsupervised*.csv`, `*_graphs.pt` caches). | release packaging | 0.5–1 day |
| **P1** | Reconcile **`.gitignore` data rules** with the reproducibility story (`data/raw` & `data/processed` fully ignored). | `.gitignore`, docs | 1–2 h |
| **P1** | Merge/extend **`.gitignore`** toward upstream **`Python.gitignore`** (pytest/mypy/jupyter/coverage noise). | `.gitignore` | 30 min |
| **P2** | Document **Python ≥ 3.10** (or refactor unions) given `X | Y` typing. | `README.md`, optional `pyproject.toml` | 30–60 min |
| **P2** | Add **Citation / CITATION.cff** and **single command log** (checklist already recommends). | `CITATION.cff`, `README.md` | 1 h |
| **P2** | Audit **DrugBank vocabulary** file licence for public release. | `data/raw/reference/drugbank_*` | 30–60 min |
| **P2** | Run **`pyflakes`/`ruff`** in CI for unused imports once pinning exists. | CI config | 1–2 h |
| **P3** | Publish **`Supplementary Section S5`** or merge canonical commands into **`paper/reproducibility_checklist.md`** explicitly. | `paper/` | 1 h |

---

## Audit closing metadata

- **Report path (absolute):** `c:\Users\Admin\Documents\Masters\Hit 800\Clayton\DL-PBPK Model\dl-pbpk-hybrid\experiments\reports\PRE_GITHUB_AUDIT_REPORT.md`
- **Total severity counts (embedded in summary):** CRITICAL **5**, HIGH **11**, MEDIUM **18**, LOW **6**
- **Single most important blocker for going public:** **Missing root-level open-source `LICENSE`**, combined with **multi-gigabyte `CID-SMILES` data** and **non-ingestible default Git ignore rules for data + weights**—third parties cannot legally and practically use the repository as-is on GitHub without a defined licence and data/model delivery plan.

---

=== PRE-GITHUB AUDIT COMPLETE ===
