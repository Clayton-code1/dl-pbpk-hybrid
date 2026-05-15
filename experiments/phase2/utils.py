"""Shared Phase 2 helpers: reference PBPK curves, descriptors, hybrid loading."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
import torch

from experiments.config import MODELS_DIR, PROCESSED_DATA_DIR
from experiments.models.hybrid_multidrug import MultiDrugHybridConfig, MultiDrugHybridGNNPBPK
from experiments.reference_pk import REFERENCE_PK_DATA
from experiments.training.multidrug_utils import (
    PatientRecord,
    cmax_auc_errors,
    regression_metrics,
)

MOL_DESCRIPTOR_COLS = [
    "MW", "logP", "TPSA", "n_HBD", "n_HBA", "n_rotatable", "n_rings", "fsp3",
]

DEFAULT_KA_H = 1.5


def load_molecular_vector(drug: str) -> np.ndarray:
    path = PROCESSED_DATA_DIR / "drug_molecular_features.csv"
    df = pd.read_csv(path)
    row = df.loc[df["drug"] == drug].iloc[0]
    return row[MOL_DESCRIPTOR_COLS].to_numpy(dtype=np.float64)


def oral_1cpt_batch(
    times_hr: np.ndarray,
    dose_mg: float,
    F: float,
    ka: float,
    cl_per_kg: float,
    vd_per_kg: float,
    weight_kg: float,
) -> np.ndarray:
    """Population mean 1-cpt oral (matches Phase 1 simulator)."""
    CL = cl_per_kg * weight_kg
    V = vd_per_kg * weight_kg
    ke = CL / V
    if abs(ka - ke) < 1e-4:
        ka = ka + 1e-3
    pre = (F * dose_mg * ka) / (V * (ka - ke))
    conc = pre * (np.exp(-ke * times_hr) - np.exp(-ka * times_hr))
    return np.clip(conc, 0.0, None)


def pbpk_reference_curves(
    records: list[PatientRecord],
    drug: str,
    ka: float = DEFAULT_KA_H,
) -> np.ndarray:
    """PBPK-only: literature CL/Vd; use each patient's dose, weight, F."""
    ref = REFERENCE_PK_DATA[drug]
    cl_pk = float(ref["CL_L_h"])
    vd_pk = float(ref["Vd_L_kg"])
    times = records[0].times_hr.cpu().numpy()
    out: list[np.ndarray] = []
    for rec in records:
        w = float(rec.weight_kg.item())
        dm = float(rec.dose_mg.item())
        fb = float(rec.f_bio.item())
        out.append(oral_1cpt_batch(times, dm, fb, ka, cl_pk, vd_pk, w))
    return np.stack(out, axis=0)


# Log-scale SD for shared multiplicative error on CL and Vd (population PK uncertainty).
CLINICAL_PBPK_POPULATION_LOG_SD = 0.4


def pbpk_population_estimate_curves(
    records: list[PatientRecord],
    drug: str,
    rng: np.random.Generator,
    ka: float = DEFAULT_KA_H,
    population_log_sd: float = CLINICAL_PBPK_POPULATION_LOG_SD,
) -> np.ndarray:
    """Literature mean CL/Vd with per-patient population uncertainty (no oracle).

    For each patient, one draw ``z ~ N(0, 1)`` scales **both** CL and Vd by
    ``exp(population_log_sd * z)``, so a single imprecise population estimate
    drives the whole predicted curve.
    """
    ref = REFERENCE_PK_DATA[drug]
    cl_mean = float(ref["CL_L_h"])
    vd_mean = float(ref["Vd_L_kg"])
    times = records[0].times_hr.cpu().numpy()
    out: list[np.ndarray] = []
    for rec in records:
        z = float(rng.standard_normal())
        m = math.exp(population_log_sd * z)
        cl_pk = cl_mean * m
        vd_pk = vd_mean * m
        w = float(rec.weight_kg.item())
        dm = float(rec.dose_mg.item())
        fb = float(rec.f_bio.item())
        out.append(oral_1cpt_batch(times, dm, fb, ka, cl_pk, vd_pk, w))
    return np.stack(out, axis=0)


def curve_metrics(pred: np.ndarray, true: np.ndarray, times: np.ndarray) -> dict[str, float]:
    flat = regression_metrics(pred.ravel(), true.ravel())
    ca = cmax_auc_errors(pred, true, times)
    obs_mean = float(true.mean())
    flat["RMSE_pct_of_mean"] = (
        flat["RMSE"] / (obs_mean + 1e-12) * 100.0 if obs_mean > 0 else float("nan")
    )
    return {**flat, **ca}


def load_phase1_hybrid(drug: str) -> tuple[MultiDrugHybridGNNPBPK, dict[str, Any]]:
    out_dir = MODELS_DIR / f"hybrid_gnn_pbpk_{drug}_v1"
    cfg_dict = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    cfg = MultiDrugHybridConfig(
        node_feat_dim=cfg_dict["node_feat_dim"],
        edge_feat_dim=cfg_dict["edge_feat_dim"],
        gnn_hidden=cfg_dict["gnn_hidden"],
        gnn_layers=cfg_dict["gnn_layers"],
        gnn_embed_dim=cfg_dict["gnn_embed_dim"],
        patient_feat_dim=cfg_dict["patient_feat_dim"],
        head_hidden=cfg_dict["head_hidden"],
        head_dropout=cfg_dict["head_dropout"],
        n_euler_steps=cfg_dict["n_euler_steps"],
    )
    model = MultiDrugHybridGNNPBPK(cfg)
    try:
        state = torch.load(out_dir / "model.pt", map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover
        state = torch.load(out_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, cfg_dict


@torch.no_grad()
def predict_hybrid_curves(
    model: MultiDrugHybridGNNPBPK,
    graph: dict[str, torch.Tensor],
    records: list[PatientRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_list: list[np.ndarray] = []
    true_list: list[np.ndarray] = []
    times: np.ndarray | None = None
    for rec in records:
        c_pred, *_ = model(
            graph["x"],
            graph["edge_index"],
            graph["edge_attr"],
            rec.features,
            rec.times_hr,
            rec.dose_mg,
            rec.weight_kg,
            rec.f_bio,
        )
        pred_list.append(c_pred.cpu().numpy())
        true_list.append(rec.concentration.cpu().numpy())
        if times is None:
            times = rec.times_hr.cpu().numpy()
    return np.stack(pred_list), np.stack(true_list), times  # type: ignore[return-value]
