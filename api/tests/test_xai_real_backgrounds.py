"""Panel-path SHAP should sample real training CSV rows (not synthetic jitter only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

API_ROOT = Path(__file__).resolve().parents[1]
PROJ_ROOT = API_ROOT.parent
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(API_ROOT))


def test_panel_shap_background_rows_exist_in_training_csv() -> None:
    from experiments.training.multidrug_utils import (
        patient_feature_columns,
        split_patient_ids,
        split_rng_seed_for_drug,
    )

    drug = "theophylline"
    feat_names = patient_feature_columns(drug)
    from app.services import xai_service

    xai_service._PANEL_SHAP_BACKGROUND_CACHE.clear()
    bg = xai_service.load_panel_shap_background_training(drug, feat_names, n=12, seed=42)
    assert bg is not None
    assert bg.shape[1] == len(feat_names)

    csv_path = PROJ_ROOT / "experiments" / "data" / "processed" / f"{drug}_pk_dataset.csv"
    df = pd.read_csv(csv_path)
    if "dose_mgkg" not in df.columns:
        df["dose_mgkg"] = df["dose_mg"] / df["weight_kg"]
    df["dose_mg_per_kg"] = df["dose_mg"] / df["weight_kg"]
    df["log_dose_mg_per_kg"] = np.log(df["dose_mg_per_kg"] + 1e-8)
    n_pat = int(df["patient_id"].max()) + 1
    train_ids, _, _ = split_patient_ids(n_pat, seed=split_rng_seed_for_drug(drug))
    train_df = df[df["patient_id"].isin(train_ids)]
    uniq = train_df.groupby("patient_id", sort=True).head(1)

    train_matrix = uniq[feat_names].to_numpy(dtype=np.float64)
    for i in range(bg.shape[0]):
        row = bg[i]
        dist = np.linalg.norm(train_matrix - row[None, :], axis=1)
        assert dist.min() < 1e-5, f"background row {i} not found in training CSV"


def test_panel_shap_background_not_synthetic_only() -> None:
    """Synthetic fallback varies ref ±25%; exact CSV rows should appear with high precision."""
    from app.services import xai_service
    from app.services import multidrug_bundle as mdb

    drug = "warfarin"
    bundle = mdb.load_multidrug_bundle(drug)
    if bundle is None:
        return
    xai_service._PANEL_SHAP_BACKGROUND_CACHE.clear()
    bg = xai_service.load_panel_shap_background_training(
        drug, bundle.feature_names, n=20, seed=42,
    )
    assert bg is not None
    # At least one row should be exactly a training covariate row (no jitter).
    from experiments.training.multidrug_utils import (
        split_patient_ids,
        split_rng_seed_for_drug,
    )

    df = pd.read_csv(PROJ_ROOT / "experiments" / "data" / "processed" / f"{drug}_pk_dataset.csv")
    if "dose_mgkg" not in df.columns:
        df["dose_mgkg"] = df["dose_mg"] / df["weight_kg"]
    df["dose_mg_per_kg"] = df["dose_mg"] / df["weight_kg"]
    df["log_dose_mg_per_kg"] = np.log(df["dose_mg_per_kg"] + 1e-8)
    n_pat = int(df["patient_id"].max()) + 1
    train_ids, _, _ = split_patient_ids(n_pat, seed=split_rng_seed_for_drug(drug))
    first = df[df["patient_id"].isin(train_ids)].groupby("patient_id", sort=True).head(1)
    train_rows = first[bundle.feature_names].to_numpy(dtype=np.float64)
    matches = 0
    for i in range(bg.shape[0]):
        if np.any(np.all(np.abs(train_rows - bg[i]) < 1e-6, axis=1)):
            matches += 1
    assert matches >= 1
