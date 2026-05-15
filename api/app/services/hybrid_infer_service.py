"""Hybrid DL+ODE inference service.

Loads the trained PyTorch model once (lazy singleton) and provides:
- PK parameter prediction from subject features
- Multi-event ODE simulation (oral + IV)
- PK metric computation

Supports three inference backends (priority order in ``simulate_curve``):
- **Multi-drug GNN** — ``artifacts/models/hybrid_gnn_pbpk_{drug}_v1`` + per-drug scaler
  (aligned with ``experiments.training.multidrug_utils``) when *panel_drug* is set.
- **Legacy theophylline GNN** — ``hybrid_gnn_pbpk_theoph_combined_v1`` / ``hybrid_gnn_pbpk_theoph_v1``
  when SMILES is supplied and multi-drug path is not used.
- **MLP** — ``hybrid_theoph_v1`` tabular fallback.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.services import multidrug_bundle as _mdb

logger = logging.getLogger("uvicorn.error")

_BASE = Path(__file__).resolve().parents[3] / "artifacts" / "models"
_ARTIFACT_DIR = _BASE / "hybrid_theoph_v1"
_GNN_ARTIFACT_DIR_COMBINED = _BASE / "hybrid_gnn_pbpk_theoph_combined_v1"
_GNN_ARTIFACT_DIR_FALLBACK = _BASE / "hybrid_gnn_pbpk_theoph_v1"

_REQUIRED_GNN_FILES = ("model.pt", "config.json", "metrics.json", "scaler.json")

THEOPHYLLINE_SMILES = "Cn1c2c(c(=O)n(c1=O)C)[nH]cn2"

# ---------------------------------------------------------------------------
# Lazy singleton state - MLP
# ---------------------------------------------------------------------------
_model = None
_scaler_mean: np.ndarray | None = None
_scaler_std: np.ndarray | None = None
_config: dict[str, Any] | None = None
_loaded = False

# ---------------------------------------------------------------------------
# Lazy singleton state - GNN
# ---------------------------------------------------------------------------
_gnn_model = None
_gnn_scaler_mean: np.ndarray | None = None
_gnn_scaler_std: np.ndarray | None = None
_gnn_config: dict[str, Any] | None = None
_gnn_loaded = False
_gnn_version: str | None = None  # "hybrid_gnn_pbpk_theoph_combined_v1" or "hybrid_gnn_pbpk_theoph_v1"


# ---------------------------------------------------------------------------
# Multi-drug panel (paper-aligned)
# ---------------------------------------------------------------------------


def resolve_panel_drug_slug(compound_name: str, panel_drug_explicit: str | None) -> str | None:
    """Map free-text compound name or explicit slug to a panel drug directory, if any."""
    return _mdb.normalize_panel_drug(compound_name, panel_drug_explicit)


def _scale_oral_doses_for_f(events: list[dict], f_bio: float) -> list[dict]:
    """Apply oral bioavailability F to oral event doses (matches training ODE)."""
    out: list[dict] = []
    for e in events:
        d = dict(e)
        if d.get("route", "oral") == "oral":
            d["dose_mg"] = float(d["dose_mg"]) * float(f_bio)
        out.append(d)
    return out


def predict_multidrug_pk(
    drug: str,
    dose_mg: float,
    weight_kg: float,
    age_years: float,
    sex: float,
) -> tuple[float, float, float] | None:
    """Return (CL, V, ka) from the panel hybrid for *drug*, or None if artifacts missing."""
    bundle = _mdb.load_multidrug_bundle(drug)
    if bundle is None:
        return None
    full = _mdb.build_raw_feature_row(drug, float(dose_mg), float(weight_kg), float(age_years), float(sex))
    canon_names = _mdb.patient_feature_column_names(drug)
    idx_map = {n: i for i, n in enumerate(canon_names)}
    raw = np.array([full[idx_map[n]] for n in bundle.feature_names], dtype=np.float32)
    norm = (raw - bundle.mean) / bundle.std
    pt = torch.tensor(norm, dtype=torch.float32)
    wt = torch.tensor([float(weight_kg)], dtype=torch.float32)
    with torch.no_grad():
        emb = bundle.model.get_drug_embedding(
            bundle.graph_x, bundle.graph_edge_index, bundle.graph_edge_attr,
        )
        CL, V, ka, _, _ = bundle.model.predict_pk_params(emb, pt, wt)
    return float(CL.item()), float(V.item()), float(ka.item())


def predict_multidrug_pk_from_raw(
    drug: str,
    raw_features: np.ndarray,
    weight_kg: float,
) -> tuple[float, float, float] | None:
    """Forward panel hybrid from a raw feature row matching ``bundle.feature_names`` (training CSV scale)."""
    bundle = _mdb.load_multidrug_bundle(drug)
    if bundle is None:
        return None
    raw = np.asarray(raw_features, dtype=np.float32).reshape(-1)
    if raw.shape[0] != len(bundle.feature_names):
        raise ValueError(
            f"raw_features length {raw.shape[0]} != scaler width {len(bundle.feature_names)} for {drug}",
        )
    norm = (raw - bundle.mean) / bundle.std
    pt = torch.tensor(norm, dtype=torch.float32)
    wt = torch.tensor([float(weight_kg)], dtype=torch.float32)
    with torch.no_grad():
        emb = bundle.model.get_drug_embedding(
            bundle.graph_x, bundle.graph_edge_index, bundle.graph_edge_attr,
        )
        CL, V, ka, _, _ = bundle.model.predict_pk_params(emb, pt, wt)
    return float(CL.item()), float(V.item()), float(ka.item())


def is_inference_ready() -> bool:
    """True if at least one backend can run: multi-drug panel, legacy GNN, or MLP."""
    if any(_mdb.load_multidrug_bundle(d) is not None for d in _mdb.PANEL_DRUG_SLUGS):
        return True
    if _ensure_gnn_loaded():
        return True
    return _ensure_loaded()


def _ensure_loaded() -> bool:
    """Load MLP model artifacts on first call. Returns True if ready."""
    global _model, _scaler_mean, _scaler_std, _config, _loaded

    if _loaded:
        return _model is not None

    _loaded = True  # only attempt once

    model_pt = _ARTIFACT_DIR / "model.pt"
    scaler_json = _ARTIFACT_DIR / "scaler.json"
    config_json = _ARTIFACT_DIR / "config.json"

    if not model_pt.exists():
        logger.warning("Model artifacts not found at %s", _ARTIFACT_DIR)
        return False

    try:
        import torch

        with open(config_json, "r") as f:
            _config = json.load(f)

        with open(scaler_json, "r") as f:
            sc = json.load(f)
            _scaler_mean = np.array(sc["mean"], dtype=np.float32)
            _scaler_std = np.array(sc["std"], dtype=np.float32)

        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, n_in: int, hid: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_in, hid), nn.Tanh(),
                    nn.Linear(hid, hid), nn.Tanh(),
                )
                self.head = nn.Linear(hid, 3)

            def forward(self, x):
                return self.head(self.net(x))

        n_in = int(_config.get("n_input", 3))
        hid = int(_config.get("hidden_dim", 32))
        net = _Net(n_in, hid)
        state = torch.load(model_pt, map_location="cpu", weights_only=True)
        mapped: dict[str, Any] = {}
        for k, v in state.items():
            nk = k
            if k.startswith("net.") or k.startswith("head."):
                nk = k
            mapped[nk] = v
        net.load_state_dict(mapped, strict=True)
        net.eval()
        _model = net
        logger.info("MLP model loaded from %s (%d params)", _ARTIFACT_DIR.name, sum(p.numel() for p in net.parameters()))
        return True

    except Exception:
        logger.exception("Failed to load MLP model")
        return False


def _gnn_dir_ready(artifact_dir: Path) -> tuple[bool, list[str]]:
    """Return (True, []) if all required files exist; else (False, list of missing names)."""
    missing = [f for f in _REQUIRED_GNN_FILES if not (artifact_dir / f).exists()]
    return (len(missing) == 0, missing)


def _load_gnn_from_dir(artifact_dir: Path, version_label: str) -> bool:
    """Load GNN model from *artifact_dir*. Returns True on success. Sets global state."""
    global _gnn_model, _gnn_scaler_mean, _gnn_scaler_std, _gnn_config, _gnn_version

    model_pt = artifact_dir / "model.pt"
    scaler_json = artifact_dir / "scaler.json"
    config_json = artifact_dir / "config.json"

    try:
        import torch

        with open(config_json, "r", encoding="utf-8") as f:
            _gnn_config = json.load(f)

        with open(scaler_json, "r", encoding="utf-8") as f:
            sc = json.load(f)
            _gnn_scaler_mean = np.array(sc["mean"], dtype=np.float32)
            _gnn_scaler_std = np.array(sc["std"], dtype=np.float32)

        from app.services._gnn_inline import build_gnn_model
        gnn_model = build_gnn_model(_gnn_config)
        state = torch.load(model_pt, map_location="cpu", weights_only=True)
        gnn_model.load_state_dict(state, strict=True)
        gnn_model.eval()
        _gnn_model = gnn_model
        _gnn_version = version_label
        logger.info(
            "GNN model loaded from %s (%d params) [%s]",
            artifact_dir.name,
            sum(p.numel() for p in gnn_model.parameters()),
            version_label,
        )
        return True
    except Exception:
        logger.exception("Failed to load GNN model from %s", artifact_dir)
        return False


def _ensure_gnn_loaded() -> bool:
    """Load GNN model: try combined_v1 first, then fallback to hybrid_gnn_pbpk_theoph_v1."""
    global _gnn_loaded

    if _gnn_loaded:
        return _gnn_model is not None

    _gnn_loaded = True

    # 1) Prefer combined hybrid model
    ok, missing = _gnn_dir_ready(_GNN_ARTIFACT_DIR_COMBINED)
    if ok:
        if _load_gnn_from_dir(_GNN_ARTIFACT_DIR_COMBINED, "hybrid_gnn_pbpk_theoph_combined_v1"):
            logger.info("Loaded combined hybrid GNN-PBPK model")
            return True
        # Load failed; fall through to fallback
    else:
        logger.info(
            "Combined GNN dir missing required files: %s - trying fallback",
            missing,
        )

    # 2) Fallback to previous hybrid GNN
    ok, missing = _gnn_dir_ready(_GNN_ARTIFACT_DIR_FALLBACK)
    if not ok:
        logger.info(
            "GNN fallback dir missing required files: %s - GNN path unavailable",
            missing,
        )
        return False

    if _load_gnn_from_dir(_GNN_ARTIFACT_DIR_FALLBACK, "hybrid_gnn_pbpk_theoph_v1"):
        logger.info("Loaded fallback hybrid GNN-PBPK model (hybrid_gnn_pbpk_theoph_v1)")
        return True
    return False


def is_model_loaded() -> bool:
    return _ensure_loaded()


def is_gnn_loaded() -> bool:
    return _ensure_gnn_loaded()


def get_gnn_version() -> str:
    """Return the version string of the loaded GNN model, or empty if GNN not loaded."""
    _ensure_gnn_loaded()
    return _gnn_version or ""


def active_model_type() -> str:
    """Return which model backend is active: 'gnn', 'mlp', or 'none'."""
    if _ensure_gnn_loaded():
        return "gnn"
    if _ensure_loaded():
        return "mlp"
    return "none"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(dose_mg: float, weight_kg: float) -> np.ndarray:
    """Return normalised feature vector [dose_mg, weight_kg, dose_mgkg] for MLP."""
    dose_mgkg = dose_mg / max(weight_kg, 1.0)
    raw = np.array([dose_mg, weight_kg, dose_mgkg], dtype=np.float32)
    assert _scaler_mean is not None
    return (raw - _scaler_mean) / _scaler_std


def _build_gnn_features(dose_mg: float, weight_kg: float) -> np.ndarray:
    """Return normalised feature vector for GNN head."""
    dose_mgkg = dose_mg / max(weight_kg, 1.0)
    raw = np.array([dose_mg, weight_kg, dose_mgkg], dtype=np.float32)
    assert _gnn_scaler_mean is not None
    return (raw - _gnn_scaler_mean) / _gnn_scaler_std


# ---------------------------------------------------------------------------
# PK parameter prediction
# ---------------------------------------------------------------------------

def predict_params(features: np.ndarray) -> tuple[float, float, float]:
    """Run the MLP and return (CL, V, ka) in real units."""
    import torch
    assert _model is not None and _config is not None
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        raw = _model(x)
        params = torch.exp(raw).numpy()

    CL = float(max(params[0], _config.get("cl_floor", 0.1)))
    V  = float(max(params[1], _config.get("v_floor", 1.0)))
    ka = float(max(params[2], _config.get("ka_floor", 0.05)))
    return CL, V, ka


def predict_params_gnn(smiles: str, dose_mg: float, weight_kg: float) -> tuple[float, float, float]:
    """Run GNN model and return (CL, V, ka)."""
    import torch
    assert _gnn_model is not None and _gnn_config is not None

    from app.services.rdkit_graph import smiles_to_graph
    graph = smiles_to_graph(smiles)

    features = _build_gnn_features(dose_mg, weight_kg)
    V_typical = float(_gnn_config.get("V_typical", 30.0))

    with torch.no_grad():
        drug_emb = _gnn_model.get_drug_embedding(
            graph["x"], graph["edge_index"], graph["edge_attr"],
        )
        patient_feat = torch.tensor(features, dtype=torch.float32)
        CL, ka = _gnn_model.predict_pk_params(drug_emb, patient_feat)

    cl_floor = _gnn_config.get("cl_floor", 0.1)
    ka_floor = _gnn_config.get("ka_floor", 0.05)
    return (
        float(max(CL.item(), cl_floor)),
        V_typical,
        float(max(ka.item(), ka_floor)),
    )


# ---------------------------------------------------------------------------
# Multi-event ODE simulation - dispatches to PBPK-lite or legacy 1-cpt
# ---------------------------------------------------------------------------

def simulate_curve(
    events: list[dict],
    weight_kg: float,
    horizon_hr: float = 48.0,
    dt_min: float = 5.0,
    *,
    pbpk_mode: str = "pbpk_lite",
    return_tissues: bool = False,
    smiles: str | None = None,
    panel_drug: str | None = None,
    age_years: float = 40.0,
    sex: float = 0.0,
) -> tuple[list[float], list[float], dict[str, float], dict | None, str]:
    """Simulate concentration-time profile for one or more dosing events.

    Returns (times_hr, conc_ng_ml, pk_params, pbpk_block_or_None, model_used).
    """
    total_dose = sum(e["dose_mg"] for e in events)
    model_used = "mlp"

    if panel_drug is not None:
        pk = predict_multidrug_pk(
            panel_drug, total_dose, weight_kg, age_years, sex,
        )
        if pk is not None:
            CL, V, ka = pk
            model_used = "multidrug_gnn"
            f_bio = _mdb.oral_bioavailability(panel_drug)
            events_eff = _scale_oral_doses_for_f(events, f_bio)
            if pbpk_mode == "pbpk_lite":
                from app.services import pbpk_service
                res = pbpk_service.simulate_pbpk(
                    events_eff, weight_kg, CL, ka,
                    horizon_hr=horizon_hr, dt_min=dt_min,
                    return_tissues=return_tissues,
                )
                pk_params = {
                    "CL_l_h": res["pk_params"]["CL_l_h"],
                    "V_l": round(V, 4),
                    "ka_1_h": res["pk_params"]["ka_1_h"],
                }
                pbpk_block: dict | None = {
                    "enabled": True,
                    "tissue_units": "mg/L",
                    "params": res["pk_params"],
                    "physiology": res["pbpk_physiology"],
                }
                if return_tissues and "tissues" in res:
                    pbpk_block["tissues"] = res["tissues"]
                else:
                    pbpk_block["tissues"] = None
                return res["times_hr"], res["conc_central_ng_ml"], pk_params, pbpk_block, model_used
            return (*_simulate_1cpt(events_eff, CL, V, ka, horizon_hr, dt_min), model_used)

    if smiles is not None and _ensure_gnn_loaded():
        CL, V, ka = predict_params_gnn(smiles, total_dose, weight_kg)
        model_used = "gnn"
    else:
        _ensure_loaded()
        assert _model is not None
        features = build_features(total_dose, weight_kg)
        CL, V, ka = predict_params(features)

    if pbpk_mode == "pbpk_lite":
        from app.services import pbpk_service
        res = pbpk_service.simulate_pbpk(
            events, weight_kg, CL, ka,
            horizon_hr=horizon_hr, dt_min=dt_min,
            return_tissues=return_tissues,
        )
        pk_params = {
            "CL_l_h": res["pk_params"]["CL_l_h"],
            "V_l": round(V, 4),
            "ka_1_h": res["pk_params"]["ka_1_h"],
        }
        pbpk_block: dict | None = {
            "enabled": True,
            "tissue_units": "mg/L",
            "params": res["pk_params"],
            "physiology": res["pbpk_physiology"],
        }
        if return_tissues and "tissues" in res:
            pbpk_block["tissues"] = res["tissues"]
        else:
            pbpk_block["tissues"] = None
        return res["times_hr"], res["conc_central_ng_ml"], pk_params, pbpk_block, model_used

    return (*_simulate_1cpt(events, CL, V, ka, horizon_hr, dt_min), model_used)


def _simulate_1cpt(
    events: list[dict],
    CL: float, V: float, ka: float,
    horizon_hr: float, dt_min: float,
) -> tuple[list[float], list[float], dict[str, float], None]:
    """Legacy 1-compartment Euler solver."""
    ke = CL / V
    n_steps = int(_config.get("n_euler_steps", 300)) if _config else 300
    dt = horizon_hr / n_steps
    t_grid = np.linspace(0.0, horizon_hr, n_steps + 1)

    sorted_events = sorted(events, key=lambda e: e["time_hr"])
    A_gut = 0.0
    A_cent = 0.0
    conc_fine = np.zeros(n_steps + 1)
    event_idx = 0

    for i in range(n_steps + 1):
        t = t_grid[i]
        while event_idx < len(sorted_events) and sorted_events[event_idx]["time_hr"] <= t:
            ev = sorted_events[event_idx]
            if ev.get("route", "oral") == "iv":
                A_cent += ev["dose_mg"]
            else:
                A_gut += ev["dose_mg"]
            event_idx += 1
        conc_fine[i] = max(A_cent / V, 0.0)
        if i < n_steps:
            dA_gut = -ka * A_gut
            dA_cent = ka * A_gut - ke * A_cent
            A_gut += dA_gut * dt
            A_cent += dA_cent * dt

    dt_hr = dt_min / 60.0
    out_times = np.arange(0.0, horizon_hr + dt_hr * 0.5, dt_hr)
    out_conc = np.interp(out_times, t_grid, conc_fine)
    out_conc = np.maximum(out_conc, 0.0)
    conc_ng_ml = out_conc * 1000.0

    times_list = [round(float(t), 4) for t in out_times]
    conc_list = [round(float(c), 2) for c in conc_ng_ml]
    pk_params = {"CL_l_h": round(CL, 4), "V_l": round(V, 4), "ka_1_h": round(ka, 4)}
    return times_list, conc_list, pk_params, None


# ---------------------------------------------------------------------------
# PK metric computation
# ---------------------------------------------------------------------------

def compute_pk_metrics(
    times_hr: list[float], conc_ng_ml: list[float], CL: float, V: float
) -> dict[str, float]:
    """Derive standard PK metrics from a concentration-time profile."""
    t = np.array(times_hr)
    c = np.array(conc_ng_ml)

    cmax = float(np.max(c))
    tmax = float(t[int(np.argmax(c))])
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(_trapz(c, t))

    n = len(c)
    tail_start = int(n * 0.6)
    c_tail = c[tail_start:]
    t_tail = t[tail_start:]
    pos = c_tail > 0
    if pos.sum() >= 2:
        log_c = np.log(c_tail[pos])
        t_pos = t_tail[pos]
        slope, _ = np.polyfit(t_pos, log_c, 1)
        ke_est = -slope
        half_life = 0.693 / ke_est if ke_est > 0 else 0.0
    else:
        half_life = 0.0

    return {
        "cmax_ng_ml": round(cmax, 2),
        "tmax_h": round(tmax, 2),
        "auc_0_inf": round(auc, 2),
        "half_life_h": round(half_life, 2),
        "clearance_l_h": round(CL, 4),
        "vd_l": round(V, 4),
    }
