# Multi-drug API routing (paper-aligned panel)

The hybrid service selects a **compound-specific checkpoint** when the request resolves to one of the Phase-1 training drugs. Artifacts live at:

`artifacts/models/hybrid_gnn_pbpk_{slug}_v1/` (plus cached graph `experiments/data/processed/graphs/{slug}.pt`).

## Panel slugs

`theophylline`, `warfarin`, `midazolam`, `caffeine`, `acetaminophen`, `digoxin`

## How routing works

1. **Explicit:** set `drug.panel_drug` to a slug (in `DrugInfo`).
2. **Implicit:** `patient.compound_name` is matched via aliases in `multidrug_bundle.normalize_panel_drug` (e.g. “paracetamol” → `acetaminophen`).

If a slug resolves **and** artifacts load, `simulate_curve` uses `predict_multidrug_pk`: patient covariates match `experiments/training/multidrug_utils.patient_feature_columns` (including the extra acetaminophen channels), z-scored with each drug’s `scaler.json`.

**Oral doses** are multiplied by literature **F** from `experiments.reference_pk` before PBPK-lite simulation, consistent with training.

If the panel path is not available, the API falls back to **legacy theophylline GNN** (SMILES) or **MLP** (`hybrid_theoph_v1`).

## Endpoints using `DrugInfo` / `panel_drug`

- `POST /predict/v2` — SMILES is **not** required when `panel_drug` or compound name resolves to the panel.
- `POST /recommend` — pass `drug` for panel-aware dose search.
- `POST /predict/population` — optional `drug`; when panel resolves, base CL/V/ka come from the panel hybrid (not the MLP).
- `POST /explain/v2` — SHAP and sensitivity use the **same** backend as the prediction when the panel resolves (`attribution_backend`: `panel_multidrug` vs `mlp`). Optional `shap_seed` for reproducibility.
- `POST /report/v2` — same as above when `drug` is provided.

## Health

`GET /health` returns `inference_ready` (any backend), `panel_drugs_available` per slug, and an informal `model_used_hint`.

## Scripted demo

From repo root (API running on port 8000):

```bash
python scripts/demo_multidrug_api.py --base-url http://127.0.0.1:8000 --drug warfarin
```

Optional PK uncertainty spread (parametric bootstrap style on normalized features):

```bash
python scripts/multidrug_pk_bootstrap_demo.py warfarin --B 40 --seed 1
```

(Run the second script with `PYTHONPATH` including `api` if importing `app` directly—see script header.)
