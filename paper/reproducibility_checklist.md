# Reproducibility checklist — DL–PBPK hybrid project

Root for commands: repository folder **`dl-pbpk-hybrid/`**. Use the project Python environment (e.g. `src/.venv` or documented venv) with pinned versions from project requirements if present.

---

## Environment

- [ ] **OS / Python:** note version; install dependencies (PyTorch, RDKit, scikit-learn, pandas, matplotlib, SHAP, etc.).
- [ ] **Hardware:** GPU optional but note device for training runs.
- [ ] **Determinism:** `experiments/config.py` — `SEED`, `seed_everything`; CUDA deterministic flags if required.

## Data pipeline

- [ ] `python -m experiments.data.download_pk_data` — regenerates per-drug PK CSVs under `experiments/data/processed/` (document any overrides).
- [ ] `python -m experiments.data.featurize_drugs` — molecular descriptors + `experiments/data/processed/graphs/*.pt`.

## Phase 1 — multi-drug hybrid

- [ ] `python -m experiments.training.train_multidrug_hybrid` — models under `artifacts/models/hybrid_gnn_pbpk_{drug}_v1/`.
- [ ] `python -m experiments.evaluation.evaluate_multidrug` — `experiments/results/phase1_multidrug_metrics.csv`, plots in `experiments/plots/`.

## Phase 2 — benchmarks & ablations

- [ ] Run benchmark / ablation training scripts as documented in `experiments/reports/phase_2_summary.md` (paths: `phase2_ablation_*`, baselines).
- [ ] Statistical tests / heatmaps: `experiments/statistics/significance_tests.py` with documented `--prediction-cache` / `--output-csv`.
- [ ] **Authoritative tables:** `experiments/results/phase2_benchmark_metrics_final.csv`, `phase2_ablation_summary_final.csv`, `phase2_statistical_tests_final.csv` (retain `*_corrected` / originals for audit).

## Phase 3 — safety, uncertainty, SHAP

- [ ] Literature thresholds: `python -m experiments.safety.safety_thresholds` → `experiments/results/phase3_safety_thresholds.md`.
- [ ] MC calibration: `python -m experiments.uncertainty.monte_carlo_calibration` → `phase3_uncertainty_calibration.csv`, `experiments/plots/uncertainty_calibration.*`.
- [ ] SHAP: `python -m experiments.explainability.shap_interpretation` → `phase3_shap_interpretation.md`, `shap_summary_multidrug.*`.
- [ ] **Post-approval diagnostic (read-only):** `experiments/reports/phase_3_calibration_shap_diagnostic.md` — per-drug nominal 0.90 interpolation; SHAP scope note.

## API (optional inference)

- [ ] Document environment variables / model paths for `api/` services; smoke-test `assess_risk(..., drug=...)` with literature block.

## Phase 4 — writing assets

- [ ] `paper/mathematical_formulation.md` — notation aligned with code.
- [ ] `paper/paper_skeleton.md` — section map and figure table.
- [ ] This checklist updated if script entry points change.

## Archival bundle (recommended)

- [ ] Commit **hashes** for code + **hashes or manifests** for large artifacts (`model.pt`, CSVs).
- [ ] One-page **commands log** (copy-paste order) attached as supplement or `README` section.
- [ ] Archive `experiments/results/*.csv`, `experiments/reports/phase_*_summary.md`, and key plots referenced in the manuscript.
