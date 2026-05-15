"""Shared utilities for multi-drug DL-PBPK training and evaluation.

This module owns three responsibilities so individual scripts stay short:

1. Loading the per-drug PK CSVs produced by Phase 1.3 into per-patient
   tensor records.
2. Patient-level 80/10/10 splits with a fixed seed (deterministic across
   training, evaluation and ablation).
3. Loading the cached molecular graph for a drug and the pretrained GNN
   encoder from ``hybrid_gnn_pbpk_theoph_combined_v1`` (preferred) or
   ``gnn_pretrain_combined_v1``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from experiments.config import (
    HYBRID_THEOPH_COMBINED_WEIGHTS,
    PRETRAINED_GNN_CONFIG,
    PRETRAINED_GNN_WEIGHTS,
    PROCESSED_DATA_DIR,
    SEED,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PATIENT_FEATURE_COLS = [
    "weight_kg", "dose_mg", "dose_mgkg", "age_years", "sex",
]
PATIENT_FEAT_DIM = len(PATIENT_FEATURE_COLS)


def patient_feature_columns(drug: str) -> list[str]:
    """Columns used as patient covariates (z-scored) for hybrid training/eval.

    Acetaminophen adds ``dose_mg_per_kg`` (== dose_mg / weight_kg) as an
    explicit channel alongside ``dose_mgkg`` (identical numerically) so the
    fusion MLP sees a dedicated dose-normalisation input for exposure scaling.
    """
    if drug == "acetaminophen":
        return [
            "weight_kg",
            "dose_mg",
            "dose_mg_per_kg",       # dose_mg / weight_kg (explicit mg/kg)
            "log_dose_mg_per_kg",  # log exposure scale for nonlinear head
            "age_years",
            "sex",
        ]
    return list(PATIENT_FEATURE_COLS)


def patient_feat_dim(drug: str) -> int:
    return len(patient_feature_columns(drug))


# ---------------------------------------------------------------------------
# Patient records
# ---------------------------------------------------------------------------

@dataclass
class PatientRecord:
    patient_id: int
    features: Tensor             # [PATIENT_FEAT_DIM] z-scored
    times_hr: Tensor             # [T]
    concentration: Tensor        # [T]
    dose_mg: Tensor              # scalar
    weight_kg: Tensor            # scalar
    f_bio: Tensor                # scalar, oral bioavailability (matches data gen.)
    auc_true: float              # mg*h/L
    cmax_true: float             # mg/L
    cl_true_L_h: Tensor | None = None    # generators' CL_true (L/h); optional
    vd_true_L: Tensor | None = None      # generators' Vd_true (central V, L); optional


class StandardScaler:
    """Z-score scaler serialisable to JSON (no sklearn dependency)."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.feature_names: list[str] | None = None

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        # Avoid divide-by-zero on degenerate columns (e.g. constant dose).
        self.std_[self.std_ < 1e-8] = 1.0
        self.feature_names = feature_names
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None
        return (X - self.mean_) / self.std_

    def to_dict(self) -> dict:
        assert self.mean_ is not None
        return {
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
            "feature_names": list(self.feature_names or PATIENT_FEATURE_COLS),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StandardScaler":
        s = cls()
        s.mean_ = np.array(d["mean"])
        s.std_ = np.array(d["std"])
        s.feature_names = list(d.get("feature_names", PATIENT_FEATURE_COLS))
        return s


def _per_patient_records(
    df: pd.DataFrame,
    scaler: StandardScaler,
    feat_cols: list[str],
) -> list[PatientRecord]:
    """Group long-format dataframe into per-patient tensor records."""
    out: list[PatientRecord] = []
    for pid, grp in df.sort_values("time_h").groupby("patient_id", sort=True):
        first = grp.iloc[0]
        feat_raw = first[feat_cols].to_numpy(dtype=float)
        feat_norm = scaler.transform(feat_raw[None, :])[0]

        times = grp["time_h"].to_numpy(dtype=float)
        conc = grp["concentration_mg_L"].to_numpy(dtype=float)
        auc = float(np.trapz(conc, times))
        cmax = float(conc.max())

        f_val = float(first["F"]) if "F" in first.index else 1.0
        cl_tgt: Tensor | None = None
        vd_tgt: Tensor | None = None
        if {"CL_true", "Vd_true"}.issubset(first.index):
            cl_tgt = torch.tensor(float(first["CL_true"]), dtype=torch.float32)
            vd_tgt = torch.tensor(float(first["Vd_true"]), dtype=torch.float32)

        out.append(
            PatientRecord(
                patient_id=int(pid),
                features=torch.tensor(feat_norm, dtype=torch.float32),
                times_hr=torch.tensor(times, dtype=torch.float32),
                concentration=torch.tensor(conc, dtype=torch.float32),
                dose_mg=torch.tensor(float(first["dose_mg"]), dtype=torch.float32),
                weight_kg=torch.tensor(float(first["weight_kg"]), dtype=torch.float32),
                f_bio=torch.tensor(f_val, dtype=torch.float32),
                auc_true=auc,
                cmax_true=cmax,
                cl_true_L_h=cl_tgt,
                vd_true_L=vd_tgt,
            )
        )
    return out


def split_patient_ids(
    n_patients: int,
    seed: int = SEED,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic 80/10/10 split by patient index."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_patients)
    n_train = int(round(train_frac * n_patients))
    n_val = int(round(val_frac * n_patients))
    train = np.sort(perm[:n_train])
    val = np.sort(perm[n_train : n_train + n_val])
    test = np.sort(perm[n_train + n_val :])
    return train, val, test


def split_rng_seed_for_drug(drug: str, base_seed: int = SEED) -> int:
    """Stable per-drug split seed (process-independent, unlike builtins.hash)."""
    digest = hashlib.sha256(drug.encode("utf-8")).digest()
    suffix = int.from_bytes(digest[:4], "big") % (2**31)
    return int(base_seed + suffix)


def load_drug_dataset(
    drug: str,
    seed: int | None = None,
) -> tuple[
    list[PatientRecord], list[PatientRecord], list[PatientRecord], StandardScaler
]:
    """Load and split a drug's PK CSV into (train, val, test, scaler).

    Uses a **per-drug** default split seed so small held-out sets (20 patients)
    are not all sharing the same fold pattern across the entire panel.
    """
    if seed is None:
        seed = split_rng_seed_for_drug(drug)
    csv_path = PROCESSED_DATA_DIR / f"{drug}_pk_dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing dataset for {drug}: {csv_path}.  "
            "Run experiments.data.download_pk_data first."
        )
    df = pd.read_csv(csv_path)
    if "dose_mgkg" not in df.columns:
        df["dose_mgkg"] = df["dose_mg"] / df["weight_kg"]
    df["dose_mg_per_kg"] = df["dose_mg"] / df["weight_kg"]
    df["log_dose_mg_per_kg"] = np.log(df["dose_mg_per_kg"] + 1e-8)

    feat_cols = patient_feature_columns(drug)

    n_patients = int(df["patient_id"].max()) + 1
    train_ids, val_ids, test_ids = split_patient_ids(n_patients, seed=seed)

    scaler = StandardScaler().fit(
        df.loc[df["patient_id"].isin(train_ids), feat_cols]
        .drop_duplicates()
        .to_numpy(dtype=float),
        feature_names=feat_cols,
    )

    train = _per_patient_records(df[df["patient_id"].isin(train_ids)], scaler, feat_cols)
    val = _per_patient_records(df[df["patient_id"].isin(val_ids)], scaler, feat_cols)
    test = _per_patient_records(df[df["patient_id"].isin(test_ids)], scaler, feat_cols)
    return train, val, test, scaler


# ---------------------------------------------------------------------------
# Graph + encoder helpers
# ---------------------------------------------------------------------------

def load_drug_graph(drug: str) -> dict[str, Tensor]:
    """Load the cached SMILES graph produced by Phase 1.4."""
    path = PROCESSED_DATA_DIR / "graphs" / f"{drug}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing graph for {drug}: {path}.  "
            "Run experiments.data.featurize_drugs first."
        )
    blob = torch.load(path, map_location="cpu")
    return {"x": blob["x"], "edge_index": blob["edge_index"], "edge_attr": blob["edge_attr"]}


def load_pretrained_gnn_state() -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Return ``(state_dict, gnn_config)`` for the molecular encoder.

    Prefer ``hybrid_gnn_pbpk_theoph_combined_v1/model.pt`` (Phase 1 transfer
    source); fall back to ``gnn_pretrain_combined_v1/model_gnn.pt``.
    """
    if not PRETRAINED_GNN_CONFIG.exists():
        raise FileNotFoundError(
            f"Pretrained GNN config not found: {PRETRAINED_GNN_CONFIG}."
        )
    cfg = json.loads(PRETRAINED_GNN_CONFIG.read_text())["gnn_config"]

    if HYBRID_THEOPH_COMBINED_WEIGHTS.exists():
        full = torch.load(HYBRID_THEOPH_COMBINED_WEIGHTS, map_location="cpu")
        state = {
            k.replace("gnn.", "", 1): v
            for k, v in full.items()
            if k.startswith("gnn.")
        }
        if not state:
            raise RuntimeError(
                f"No gnn.* keys in {HYBRID_THEOPH_COMBINED_WEIGHTS}"
            )
        return state, cfg

    if not PRETRAINED_GNN_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Pretrained GNN weights not found: {PRETRAINED_GNN_WEIGHTS} or "
            f"{HYBRID_THEOPH_COMBINED_WEIGHTS}."
        )
    state = torch.load(PRETRAINED_GNN_WEIGHTS, map_location="cpu")
    return state, cfg


def load_pretrained_gnn_into(model: torch.nn.Module) -> None:
    """Initialise ``model.gnn`` from combined pretrained encoder weights.

    Raises if shapes don't match or any parameter is missing.
    """
    state, _ = load_pretrained_gnn_state()
    own = model.gnn.state_dict()  # type: ignore[attr-defined]
    to_load: dict[str, Tensor] = {}
    mismatches: list[str] = []
    for k, v in state.items():
        if k not in own:
            continue
        if own[k].shape != v.shape:
            mismatches.append(f"{k}: own={tuple(own[k].shape)} ckpt={tuple(v.shape)}")
        else:
            to_load[k] = v

    missing = [k for k in own.keys() if k not in to_load]
    if mismatches or missing:
        raise RuntimeError(
            "Pretrained GNN weight load failed.  "
            f"Mismatches={mismatches}  Missing={missing}"
        )

    own.update(to_load)
    model.gnn.load_state_dict(own)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def regression_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """Standard regression metrics for a flat (N,) prediction vector."""
    eps = 1e-8
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    # NOTE: MAPE is numerically unstable for drugs with very low or
    # near-zero observed concentrations (e.g., midazolam, acetaminophen
    # in trough regions). For these drugs, RMSE-as-percentage-of-mean
    # is reported as the primary scale-normalised metric in the
    # manuscript (see §4.1). See manuscript Table 1 footnote and
    # Supplementary Section S4 for full discussion.
    mape = float(np.mean(np.abs(err) / (np.abs(y_true) + eps)) * 100.0)

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + eps)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def cmax_auc_errors(
    pred_curves: np.ndarray,
    true_curves: np.ndarray,
    times_hr: np.ndarray,
) -> dict[str, float]:
    """Per-patient Cmax / AUC % error, averaged across the batch.

    ``pred_curves`` and ``true_curves`` are [N_patients, T].
    """
    eps = 1e-8
    cmax_pred = pred_curves.max(axis=1)
    cmax_true = true_curves.max(axis=1)
    auc_pred = np.trapz(pred_curves, times_hr, axis=1)
    auc_true = np.trapz(true_curves, times_hr, axis=1)

    cmax_err = float(
        np.mean(np.abs(cmax_pred - cmax_true) / (np.abs(cmax_true) + eps)) * 100.0
    )
    auc_err = float(
        np.mean(np.abs(auc_pred - auc_true) / (np.abs(auc_true) + eps)) * 100.0
    )
    return {"Cmax_pct_err": cmax_err, "AUC_pct_err": auc_err}
