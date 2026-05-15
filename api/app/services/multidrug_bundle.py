"""Load per-drug hybrid checkpoints aligned with ``experiments.training.multidrug_utils``."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("uvicorn.error")

# Canonical slugs (directory names) — keep in sync with experiments.config.DRUGS
PANEL_DRUG_SLUGS: frozenset[str] = frozenset({
    "theophylline",
    "warfarin",
    "midazolam",
    "caffeine",
    "acetaminophen",
    "digoxin",
})

_ALIASES: tuple[tuple[str, str], ...] = (
    ("theophylline", "theophylline"),
    ("theo", "theophylline"),
    ("warfarin", "warfarin"),
    ("coumadin", "warfarin"),
    ("midazolam", "midazolam"),
    ("versed", "midazolam"),
    ("caffeine", "caffeine"),
    ("acetaminophen", "acetaminophen"),
    ("paracetamol", "acetaminophen"),
    ("apap", "acetaminophen"),
    ("tylenol", "acetaminophen"),
    ("digoxin", "digoxin"),
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def patient_feature_column_names(drug: str) -> list[str]:
    """Match ``multidrug_utils.patient_feature_columns``."""
    if drug == "acetaminophen":
        return [
            "weight_kg",
            "dose_mg",
            "dose_mg_per_kg",
            "log_dose_mg_per_kg",
            "age_years",
            "sex",
        ]
    return ["weight_kg", "dose_mg", "dose_mgkg", "age_years", "sex"]


def normalize_panel_drug(compound_name: str, explicit_slug: str | None) -> str | None:
    """Return canonical drug slug or None if not a panel drug."""
    if explicit_slug:
        s = explicit_slug.strip().lower().replace(" ", "_")
        if s in PANEL_DRUG_SLUGS:
            return s
        return None
    key = compound_name.strip().lower()
    for pat, slug in _ALIASES:
        if pat == key or pat in key or key in pat:
            return slug
    if key in PANEL_DRUG_SLUGS:
        return key
    return None


def oral_bioavailability(drug: str) -> float:
    try:
        root = project_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from experiments.reference_pk import REFERENCE_PK_DATA

        if drug not in REFERENCE_PK_DATA:
            return 1.0
        return float(REFERENCE_PK_DATA[drug]["F"])
    except Exception:
        logger.debug("F fallback for %s", drug)
        return 1.0


def build_raw_feature_row(
    drug: str,
    dose_mg: float,
    weight_kg: float,
    age_years: float,
    sex: float,
) -> np.ndarray:
    w = max(float(weight_kg), 1e-6)
    dose_mgkg = float(dose_mg) / w
    cols = patient_feature_column_names(drug)
    vals: dict[str, float] = {
        "weight_kg": float(weight_kg),
        "dose_mg": float(dose_mg),
        "dose_mgkg": dose_mgkg,
        "dose_mg_per_kg": dose_mgkg,
        "log_dose_mg_per_kg": float(np.log(dose_mgkg + 1e-8)),
        "age_years": float(age_years),
        "sex": float(sex),
    }
    return np.array([vals[c] for c in cols], dtype=np.float32)


@dataclass(frozen=True)
class MultiDrugBundle:
    drug: str
    model: Any
    config: dict[str, Any]
    feature_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    graph_x: torch.Tensor
    graph_edge_index: torch.Tensor
    graph_edge_attr: torch.Tensor


def _artifact_dir(drug: str) -> Path:
    return project_root() / "artifacts" / "models" / f"hybrid_gnn_pbpk_{drug}_v1"


def _graph_path(drug: str) -> Path:
    return project_root() / "experiments" / "data" / "processed" / "graphs" / f"{drug}.pt"


@lru_cache(maxsize=8)
def load_multidrug_bundle(drug: str) -> MultiDrugBundle | None:
    """Load model + scaler + cached graph for one panel drug. Cached in-process."""
    if drug not in PANEL_DRUG_SLUGS:
        return None
    adir = _artifact_dir(drug)
    gpath = _graph_path(drug)
    if not (adir / "model.pt").exists():
        logger.info("Multi-drug artifacts missing for %s (%s)", drug, adir)
        return None
    if not gpath.exists():
        logger.info("Cached graph missing for %s (%s)", drug, gpath)
        return None
    try:
        from app.services._multidrug_gnn_inline import build_multidrug_model

        with open(adir / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        with open(adir / "scaler.json", "r", encoding="utf-8") as f:
            sc = json.load(f)
        feat_names = list(sc.get("feature_names", patient_feature_column_names(drug)))
        mean = np.array(sc["mean"], dtype=np.float32)
        std = np.array(sc["std"], dtype=np.float32)

        model = build_multidrug_model(cfg)
        state = torch.load(adir / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()

        blob = torch.load(gpath, map_location="cpu", weights_only=True)
        bundle = MultiDrugBundle(
            drug=drug,
            model=model,
            config=cfg,
            feature_names=feat_names,
            mean=mean,
            std=std,
            graph_x=blob["x"].float(),
            graph_edge_index=blob["edge_index"].long(),
            graph_edge_attr=blob["edge_attr"].float(),
        )
        logger.info("Loaded multi-drug bundle: hybrid_gnn_pbpk_%s_v1", drug)
        return bundle
    except Exception:
        logger.exception("Failed loading multi-drug bundle for %s", drug)
        return None


def panel_drug_availability() -> dict[str, bool]:
    return {d: load_multidrug_bundle(d) is not None for d in sorted(PANEL_DRUG_SLUGS)}
