"""Zero-shot PK inference for unknown compounds with valid SMILES.

Uses the pretrained GNN encoder (frozen) from the theophylline-combined checkpoint
with generic population default PK parameters. Not validated on any unknown drug —
results are clearly flagged in the response.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from pathlib import Path

import torch

from app.services._multidrug_gnn_inline import InlineMultiDrugHybridGNNPBPK
from app.services.rdkit_graph import smiles_to_graph

logger = logging.getLogger("uvicorn.error")

_PRETRAINED_WEIGHTS = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "models"
    / "hybrid_gnn_pbpk_theoph_combined_v1"
    / "model.pt"
)

# Generic population defaults for unknown drugs
_ZS_CL_PER_KG = 0.10   # L/h/kg
_ZS_VD_PER_KG = 0.60   # L/kg
_ZS_KA = 1.50           # h^-1

# Fixed reference population for patient z-scoring
_W_MEAN, _W_STD = 70.0, 15.0
_AGE_MEAN, _AGE_STD = 40.0, 15.0
_SEX_MEAN, _SEX_STD = 0.5, 0.5


@lru_cache(maxsize=1)
def _load_model() -> InlineMultiDrugHybridGNNPBPK:
    """Lazy-load: GNN encoder from pretrained checkpoint (frozen); head seeded with generic defaults."""
    model = InlineMultiDrugHybridGNNPBPK()
    state = torch.load(_PRETRAINED_WEIGHTS, map_location="cpu", weights_only=True)
    gnn_state = {k: v for k, v in state.items() if k.startswith("gnn.")}
    model.load_state_dict(gnn_state, strict=False)
    linears = [m for m in model.head.modules() if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        linears[-1].bias.copy_(
            torch.tensor([math.log(_ZS_CL_PER_KG), math.log(_ZS_VD_PER_KG), math.log(_ZS_KA)])
        )
    model.eval()
    for p in model.gnn.parameters():
        p.requires_grad_(False)
    logger.info("Zero-shot model loaded (pretrained GNN encoder frozen, generic head defaults).")
    return model


def _patient_tensor(weight_kg: float, age_years: float, sex: float) -> torch.Tensor:
    """5-feature patient vector matching panel feature order; dose features pinned to z-score 0."""
    w_z = (weight_kg - _W_MEAN) / _W_STD
    age_z = (age_years - _AGE_MEAN) / _AGE_STD
    sex_z = (sex - _SEX_MEAN) / _SEX_STD
    # features: [weight_kg, dose_mg, dose_mgkg, age_years, sex] — dose columns set to 0
    return torch.tensor([w_z, 0.0, 0.0, age_z, sex_z], dtype=torch.float32)


def predict_zeroshot(
    smiles: str,
    weight_kg: float,
    dose_mg: float,
    age_years: float,
    sex: float,
    events: list[dict],
    horizon_hr: float,
    dt_min: float,
    pbpk_mode: str = "pbpk_lite",
    return_tissues: bool = False,
) -> tuple[list[float], list[float], dict[str, float], dict | None, str]:
    """Zero-shot inference for an unknown compound.

    Returns (times_hr, conc_ng_ml, pk_params, pbpk_block_or_None, model_used="zeroshot_gnn").
    """
    model = _load_model()
    graph = smiles_to_graph(smiles)

    pt = _patient_tensor(weight_kg, age_years, sex)
    wt = torch.tensor([float(weight_kg)], dtype=torch.float32)

    with torch.no_grad():
        emb = model.get_drug_embedding(graph["x"], graph["edge_index"], graph["edge_attr"])
        CL_t, V_t, ka_t, _, _ = model.predict_pk_params(emb, pt, wt)

    CL = float(CL_t.item())
    V = float(V_t.item())
    ka = float(ka_t.item())

    from app.services import pbpk_service
    from app.services.hybrid_infer_service import _simulate_1cpt

    if pbpk_mode == "pbpk_lite":
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
        pbpk_block["tissues"] = res.get("tissues") if return_tissues else None
        return res["times_hr"], res["conc_central_ng_ml"], pk_params, pbpk_block, "zeroshot_gnn"

    times, conc, pk_params, _ = _simulate_1cpt(events, CL, V, ka, horizon_hr, dt_min)
    return times, conc, pk_params, None, "zeroshot_gnn"
