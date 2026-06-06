"""DL-PBPK Hybrid API — main application module.

Endpoints
---------
GET  /health              Health check + inference readiness + panel artifact map
POST /predict             Legacy endpoint (frontend-compatible)
POST /predict/v2          Full hybrid inference (MLP, legacy theophylline GNN, or panel multi-drug)
POST /predict/population  Population simulation (MLP or panel base PK)
POST /recommend           Dosing strategy recommendations
POST /explain/v2          SHAP + sensitivity (aligned to active predictor)
POST /report/v2           PDF report generation
GET  /population/config   Population variability parameters
POST /population/config   Update population variability parameters
GET  /pbpk/config         PBPK physiology parameters
POST /pbpk/config         Update PBPK physiology parameters
GET  /model/state         Current calibration parameters
POST /model/update        Nudge calibration from clinician feedback
"""

from __future__ import annotations

import logging
import math
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app.schemas import (
    HealthResponse,
    ModelStateResponse,
    ModelUpdateRequest,
    ModelUpdateResponse,
    PredictRequest,
    PredictResponse,
    PredictV2Request,
    PredictV2Response,
    PKMetrics,
    PBPKBlock,
    DrugInfo,
    PatientInfo,
    RecommendRequest,
    RecommendResponse,
    RegimenEvent,
    SafetyBlock,
    SafeRecommendation,
    ModelMeta,
    StrategyResult,
    ClinicalReasoningItem,
    ExplainV2Request,
    ExplainV2Response,
    SHAPResult,
    SensitivityResult,
    NarrativeResult,
    DrugStructureEffect,
    ReportV2Request,
    PredictPopulationRequest,
    PredictPopulationResponse,
    PopulationConfigResponse,
    PopulationConfigUpdateRequest,
    PopulationResult,
    PBPKConfigResponse,
    PBPKConfigUpdateRequest,
)
from app.services import hybrid_infer_service as infer
from app.services import multidrug_bundle
from app.services import risk_service
from app.services import xai_service
from app.services import report_service
from app.services import population_adapter
from app.services import pbpk_service
from app.services import clinical_rules_service

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ok = infer.is_inference_ready()
    if ok:
        logger.info("Inference backends ready (MLP and/or GNN and/or multi-drug panel).")
    else:
        logger.warning(
            "No inference checkpoints found. Install artifacts under artifacts/models/ "
            "and experiments/data/processed/graphs/."
        )
    yield


app = FastAPI(
    title="DL-PBPK Hybrid API",
    version="0.2.0",
    description="Deep-learning augmented PBPK model prediction service.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_model() -> None:
    if not infer.is_inference_ready():
        raise HTTPException(
            status_code=503,
            detail=(
                "No model artifacts available. Provide hybrid_theoph_v1 (MLP), "
                "theophylline GNN dirs, and/or hybrid_gnn_pbpk_{drug}_v1 + graphs."
            ),
        )


def _resolve_smiles(drug: DrugInfo | None, patient_compound: str) -> str | None:
    """Determine the SMILES string to use, or None to fall back to MLP."""
    if drug is not None and drug.smiles:
        return drug.smiles
    name = (drug.name if drug else patient_compound).strip().lower()
    if "theophylline" in name:
        return infer.THEOPHYLLINE_SMILES
    return None


def _run_v2(
    patient_weight: float,
    patient_compound: str,
    regimen: list[RegimenEvent],
    horizon_hr: float,
    dt_min: float = 5.0,
    *,
    pbpk_mode: str = "pbpk_lite",
    include_tissues: bool = False,
    drug: DrugInfo | None = None,
    age_years: float = 40.0,
    sex: float = 0.0,
) -> dict[str, Any]:
    """Shared logic for /predict/v2 and /recommend baseline."""
    events = [{"time_hr": e.time_hr, "dose_mg": e.dose_mg, "route": e.route} for e in regimen]

    panel = infer.resolve_panel_drug_slug(
        patient_compound,
        drug.panel_drug if drug and drug.panel_drug else None,
    )
    smiles = _resolve_smiles(drug, patient_compound) if panel is None else None

    times, conc, pk_params, pbpk_raw, model_used = infer.simulate_curve(
        events, patient_weight, horizon_hr, dt_min,
        pbpk_mode=pbpk_mode, return_tissues=include_tissues,
        smiles=smiles,
        panel_drug=panel,
        age_years=age_years,
        sex=sex,
    )

    CL = pk_params["CL_l_h"]
    V = pk_params["V_l"]
    metrics_dict = infer.compute_pk_metrics(times, conc, CL, V)
    metrics = PKMetrics(**metrics_dict)

    safety_kw: dict[str, Any] = {}
    if panel is not None:
        safety_kw["drug"] = panel
    elif (
        drug is not None
        and drug.therapeutic_min_mg_L is not None
        and drug.therapeutic_max_mg_L is not None
    ):
        safety_kw["therapeutic_min_mg_L"] = drug.therapeutic_min_mg_L
        safety_kw["therapeutic_max_mg_L"] = drug.therapeutic_max_mg_L
    safety_dict = risk_service.assess_risk(
        metrics.cmax_ng_ml, metrics.auc_0_inf, **safety_kw,
    )
    safety = SafetyBlock(**safety_dict)

    state = risk_service.get_model_state()

    if model_used == "multidrug_gnn" and panel:
        version = f"hybrid_gnn_pbpk_{panel}_v1"
    elif model_used == "gnn":
        version = infer.get_gnn_version() or "hybrid_gnn_pbpk_theoph_v1"
    else:
        version = "hybrid_theoph_v1"

    result: dict[str, Any] = {
        "time_h": times,
        "concentration_ng_ml": conc,
        "pk_metrics": metrics,
        "safety": safety,
        "model": ModelMeta(
            version=version,
            updated_at=state.get("updated_at"),
            update_flag=state.get("update_flag", False),
            model_used=model_used,
        ),
        "pk_params": pk_params,
    }

    if pbpk_raw is not None:
        result["pbpk"] = PBPKBlock(**pbpk_raw)

    return result


# ---------------------------------------------------------------------------
# Legacy synthetic simulation (kept for backward-compat when model missing)
# ---------------------------------------------------------------------------

def _generate_pk_curve_legacy(
    dose_mg: float, weight_kg: float, route: str,
) -> tuple[list[float], list[float], dict[str, float]]:
    """Synthetic one-compartment PK curve (fallback)."""
    ka = 1.2 if route == "oral" else 50.0
    ke = 0.15
    vd = 0.5 * weight_kg
    f_bio = 0.8 if route == "oral" else 1.0
    dose_ng = dose_mg * 1e6
    t = np.linspace(0, 48, 200).tolist()

    conc: list[float] = []
    for ti in t:
        if route == "oral":
            c = (f_bio * dose_ng * ka / (vd * (ka - ke))) * (
                math.exp(-ke * ti) - math.exp(-ka * ti)
            )
        else:
            c = (dose_ng / vd) * math.exp(-ke * ti)
        conc.append(max(c, 0.0))

    cmax = max(conc)
    tmax = t[conc.index(cmax)]
    half_life = 0.693 / ke
    clearance = ke * vd
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(_trapz(conc, t))

    metrics = {
        "cmax_ng_ml": round(cmax, 2),
        "tmax_h": round(tmax, 2),
        "auc_0_inf": round(auc, 2),
        "half_life_h": round(half_life, 2),
        "clearance_l_h": round(clearance, 2),
        "vd_l": round(vd, 2),
    }
    return [round(v, 3) for v in t], [round(v, 2) for v in conc], metrics


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    avail = multidrug_bundle.panel_drug_availability()
    hint = "none"
    if any(avail.values()):
        hint = "multidrug_gnn"
    elif infer.is_gnn_loaded():
        hint = "gnn"
    elif infer.is_model_loaded():
        hint = "mlp"
    return {
        "status": "ok",
        "version": app.version,
        "model_loaded": infer.is_model_loaded(),
        "inference_ready": infer.is_inference_ready(),
        "panel_drugs_available": avail,
        "model_used_hint": hint,
    }


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
async def predict_legacy(req: PredictRequest):
    """Legacy endpoint — uses hybrid model if available, else falls back to synthetic."""
    if infer.is_inference_ready():
        regimen = [RegimenEvent(time_hr=0.0, dose_mg=req.dose_mg, route=req.route)]
        result = _run_v2(
            req.weight_kg,
            req.compound_name,
            regimen,
            48.0,
            age_years=40.0,
            sex=0.0,
        )
        return {
            "compound_name": req.compound_name,
            "dose_mg": req.dose_mg,
            "time_h": result["time_h"],
            "concentration_ng_ml": result["concentration_ng_ml"],
            "pk_metrics": result["pk_metrics"],
        }
    else:
        time_h, conc, metrics = _generate_pk_curve_legacy(req.dose_mg, req.weight_kg, req.route)
        return {
            "compound_name": req.compound_name,
            "dose_mg": req.dose_mg,
            "time_h": time_h,
            "concentration_ng_ml": conc,
            "pk_metrics": metrics,
        }


@app.post("/predict/v2", response_model=PredictV2Response, tags=["prediction"])
async def predict_v2(req: PredictV2Request, include_population: bool = False):
    _require_model()

    drug = req.drug
    compound_name = req.patient.compound_name
    panel = infer.resolve_panel_drug_slug(
        compound_name,
        drug.panel_drug if drug and drug.panel_drug else None,
    )

    if panel is None:
        if drug is not None and not drug.smiles:
            if "theophylline" not in drug.name.strip().lower():
                raise HTTPException(
                    status_code=400,
                    detail=f"SMILES required for non-theophylline drug '{drug.name}' unless drug.panel_drug is set.",
                )

    result = _run_v2(
        req.patient.weight_kg,
        compound_name,
        req.regimen,
        req.horizon_hr,
        req.dt_min,
        pbpk_mode=req.pbpk_mode,
        include_tissues=req.include_tissues,
        drug=drug,
        age_years=req.patient.age_years,
        sex=req.patient.sex,
    )
    if include_population:
        events = [{"time_hr": e.time_hr, "dose_mg": e.dose_mg, "route": e.route} for e in req.regimen]
        pop = population_adapter.simulate_population(
            events, req.patient.weight_kg, req.horizon_hr, req.dt_min,
            panel_drug=panel,
            age_years=req.patient.age_years,
            sex=req.patient.sex,
        )
        result["population"] = pop
    return result


def _find_safe_scaling_factor(
    regimen: list[RegimenEvent],
    patient: PatientInfo,
    horizon_hr: float,
    *,
    drug: DrugInfo | None = None,
    max_iter: int = 12,
) -> tuple[float, dict[str, Any]]:
    """Binary-search for the largest dose scale in [0.1, 1.0] that yields a safe prediction.

    Returns (scale_factor, simulation_result_at_that_scale).
    If no safe scale is found, returns (0.1, result_at_0.1).
    """
    lo, hi = 0.1, 1.0
    best_scale = lo
    best_result: dict[str, Any] | None = None

    for _ in range(max_iter):
        mid = round((lo + hi) / 2, 6)
        scaled = _scale_regimen(regimen, mid)
        result = _run_v2(
            patient.weight_kg,
            patient.compound_name,
            scaled,
            horizon_hr,
            drug=drug,
            age_years=patient.age_years,
            sex=patient.sex,
        )
        if result["safety"].is_safe:
            best_scale = mid
            best_result = result
            lo = mid  # try a higher (less aggressive) scale
        else:
            hi = mid  # need a lower scale

    # If we never found a safe point, evaluate at the floor
    if best_result is None:
        scaled = _scale_regimen(regimen, best_scale)
        best_result = _run_v2(
            patient.weight_kg,
            patient.compound_name,
            scaled,
            horizon_hr,
            drug=drug,
            age_years=patient.age_years,
            sex=patient.sex,
        )

    return best_scale, best_result


def _scale_regimen(regimen: list[RegimenEvent], scale: float) -> list[RegimenEvent]:
    """Return a copy of the regimen with every dose multiplied by *scale*."""
    return [
        RegimenEvent(time_hr=e.time_hr, dose_mg=round(e.dose_mg * scale, 2), route=e.route)
        for e in regimen
    ]


def _build_clinical_reasoning_items(
    panel_slug: str | None,
    baseline: PredictV2Response,
    safe_rec: SafeRecommendation | None,
) -> list[ClinicalReasoningItem]:
    if not panel_slug:
        return []
    items: list[ClinicalReasoningItem] = []
    b = clinical_rules_service.explain_for_recommendation(
        panel_slug,
        baseline.pk_metrics.cmax_ng_ml / 1000.0,
        baseline.pk_metrics.auc_0_inf / 1000.0,
        "baseline",
    )
    items.append(
        ClinicalReasoningItem(
            context=b.context,
            adjustment_pct=b.adjustment_pct,
            reasoning_text=b.reasoning_text,
            evidence_tier=b.evidence_tier,
        ),
    )
    if safe_rec is not None:
        r = clinical_rules_service.explain_for_recommendation(
            panel_slug,
            safe_rec.pk_metrics.cmax_ng_ml / 1000.0,
            safe_rec.pk_metrics.auc_0_inf / 1000.0,
            "recommended_regimen",
        )
        items.append(
            ClinicalReasoningItem(
                context=r.context,
                adjustment_pct=r.adjustment_pct,
                reasoning_text=r.reasoning_text,
                evidence_tier=r.evidence_tier,
            ),
        )
    return items


@app.post("/recommend", response_model=RecommendResponse, tags=["recommendation"])
async def recommend(req: RecommendRequest):
    _require_model()

    baseline_data = _run_v2(
        req.patient.weight_kg,
        req.patient.compound_name,
        req.regimen,
        req.horizon_hr,
        drug=req.drug,
        age_years=req.patient.age_years,
        sex=req.patient.sex,
    )
    baseline = PredictV2Response(**baseline_data)
    base_cmax = baseline.pk_metrics.cmax_ng_ml
    base_auc = baseline.pk_metrics.auc_0_inf

    total_dose = sum(e.dose_mg for e in req.regimen)
    is_safe = baseline.safety.is_safe
    route = req.regimen[0].route

    # Find a safe scaling factor via binary search (used by all strategies when unsafe)
    if is_safe:
        safe_scale = 0.9  # cosmetic 10% margin for optimisation suggestions
    else:
        safe_scale, _ = _find_safe_scaling_factor(
            req.regimen, req.patient, req.horizon_hr, drug=req.drug,
        )

    strategies: list[StrategyResult] = []

    # --- Strategy 1: Dose reduction (single bolus at the safe scale) ---
    reduced_dose = round(total_dose * safe_scale, 2)
    s1_regimen = [RegimenEvent(time_hr=0.0, dose_mg=reduced_dose, route=route)]
    s1_result = _run_v2(
        req.patient.weight_kg, req.patient.compound_name, s1_regimen, req.horizon_hr,
        drug=req.drug, age_years=req.patient.age_years, sex=req.patient.sex,
    )
    pct_cut = round((1 - safe_scale) * 100, 1)
    if is_safe:
        s1_desc = (
            f"Reduce total dose by {pct_cut}% ({total_dose:.0f} -> {reduced_dose:.0f} mg) "
            f"for a wider safety margin."
        )
    else:
        s1_desc = (
            f"Reduce total dose by {pct_cut}% ({total_dose:.0f} -> {reduced_dose:.0f} mg). "
            f"Optimal safe scaling factor found via binary search: {safe_scale:.4f}."
        )
    strategies.append(_build_strategy(
        "reduce", "Dose Reduction", s1_desc,
        s1_regimen, s1_result, base_cmax, base_auc, safe_scale,
    ))

    # --- Strategy 2: Split dose (BID) at the safe scale ---
    split_total = round(total_dose * safe_scale, 2) if not is_safe else total_dose
    split_half = round(split_total / 2, 2)
    half_horizon = round(req.horizon_hr / 2, 1)
    s2_regimen = [
        RegimenEvent(time_hr=0.0, dose_mg=split_half, route=route),
        RegimenEvent(time_hr=half_horizon, dose_mg=split_half, route=route),
    ]
    s2_result = _run_v2(
        req.patient.weight_kg, req.patient.compound_name, s2_regimen, req.horizon_hr,
        drug=req.drug, age_years=req.patient.age_years, sex=req.patient.sex,
    )
    # If split is still unsafe at this scale, tighten further
    s2_scale = safe_scale if not is_safe else 1.0
    if not s2_result["safety"].is_safe:
        s2_scale, s2_result = _find_safe_scaling_factor(
            s2_regimen, req.patient, req.horizon_hr, drug=req.drug,
        )
        s2_regimen = _scale_regimen(s2_regimen, s2_scale)
        split_half = s2_regimen[0].dose_mg
        s2_scale = round(s2_scale * safe_scale, 4) if not is_safe else s2_scale
    s2_desc = (
        f"Split into two {split_half:.0f} mg doses at t=0 h and t={half_horizon:.0f} h "
        f"to reduce peak concentration (scale {s2_scale:.4f})."
    )
    strategies.append(_build_strategy(
        "split", "Split Dose (BID)", s2_desc,
        s2_regimen, s2_result, base_cmax, base_auc, s2_scale,
    ))

    # --- Strategy 3: Extended interval with safe scale ---
    if is_safe:
        s3_regimen = [RegimenEvent(time_hr=0.0, dose_mg=total_dose, route=route)]
        s3_scale = 1.0
        s3_desc = "Extend dosing interval to every 36 h to reduce cumulative exposure."
    else:
        s3_dose = round(total_dose * safe_scale * 0.67, 2)  # additional 33% cut on top of safe scale
        s3_regimen = [RegimenEvent(time_hr=0.0, dose_mg=s3_dose, route=route)]
        s3_result_tmp = _run_v2(
            req.patient.weight_kg, req.patient.compound_name, s3_regimen, req.horizon_hr,
            drug=req.drug, age_years=req.patient.age_years, sex=req.patient.sex,
        )
        s3_scale = round(safe_scale * 0.67, 4)
        if not s3_result_tmp["safety"].is_safe:
            s3_scale, s3_result_tmp = _find_safe_scaling_factor(
                [RegimenEvent(time_hr=0.0, dose_mg=total_dose, route=route)],
                req.patient, req.horizon_hr, drug=req.drug,
            )
            s3_regimen = [RegimenEvent(time_hr=0.0, dose_mg=round(total_dose * s3_scale, 2), route=route)]
        s3_desc = (
            f"Reduce dose to {s3_regimen[0].dose_mg:.0f} mg with extended dosing interval "
            f"(scale {s3_scale:.4f})."
        )

    s3_result = _run_v2(
        req.patient.weight_kg, req.patient.compound_name, s3_regimen, req.horizon_hr,
        drug=req.drug, age_years=req.patient.age_years, sex=req.patient.sex,
    )
    strategies.append(_build_strategy(
        "interval", "Extended Interval", s3_desc,
        s3_regimen, s3_result, base_cmax, base_auc, s3_scale,
    ))

    # --- Build safe recommendation ---
    safe_rec: SafeRecommendation | None = None
    search_summary: str = ""

    if is_safe:
        search_summary = "Baseline regimen is within safe ranges. No dose search required."
    else:
        # Collect candidates: the binary-search result + all strategies
        candidates: list[tuple[str, float, list[RegimenEvent], dict[str, Any], float]] = []

        # Primary binary search result (dose reduction at safe_scale)
        candidates.append(("reduce", safe_scale, s1_regimen, s1_result, safe_scale))
        candidates.append(("split", s2_scale, s2_regimen, s2_result, s2_scale))
        candidates.append(("interval", s3_scale, s3_regimen, s3_result, s3_scale))

        best_safe: tuple[str, float, list[RegimenEvent], dict[str, Any], float] | None = None
        best_unsafe: tuple[str, float, list[RegimenEvent], dict[str, Any], float] | None = None

        for label, scale, reg, res, sf in candidates:
            if res["safety"].is_safe:
                if best_safe is None or scale > best_safe[1]:
                    best_safe = (label, scale, reg, res, sf)
            else:
                if best_unsafe is None or res["safety"].risk_score < best_unsafe[3]["safety"].risk_score:
                    best_unsafe = (label, scale, reg, res, sf)

        winner = best_safe if best_safe is not None else best_unsafe
        if winner is not None:
            w_label, w_scale, w_regimen, w_result, w_sf = winner
            w_dose = sum(e.dose_mg for e in w_regimen)
            w_metrics = w_result["pk_metrics"]
            w_safety = w_result["safety"]

            freq_map = {"reduce": "Single dose", "split": "Twice daily (BID)", "interval": "Extended interval"}
            frequency = freq_map.get(w_label, "Single dose")

            if w_safety.is_safe:
                rationale = (
                    f"Automatic search found a safe regimen by scaling the original dose "
                    f"to {w_dose:.1f} mg (factor {w_sf:.4f}). "
                    f"Cmax {w_metrics.cmax_ng_ml:.0f} ng/mL and AUC {w_metrics.auc_0_inf:.0f} ng*h/mL "
                    f"are within acceptable thresholds."
                )
                search_summary = (
                    f"Automatically searched lower doses until a safe exposure profile was identified. "
                    f"Safe dose found: {w_dose:.1f} mg ({frequency})."
                )
            else:
                rationale = (
                    f"Automatic search did not find a fully safe regimen within the tested range. "
                    f"Best attempt: {w_dose:.1f} mg with risk score {w_safety.risk_score:.4f}."
                )
                search_summary = (
                    "Automatic search did not find a safe regimen within the tested dose range. "
                    "The best available option is shown."
                )

            delta_cmax = ((w_metrics.cmax_ng_ml - base_cmax) / max(base_cmax, 1e-6)) * 100
            delta_auc = ((w_metrics.auc_0_inf - base_auc) / max(base_auc, 1e-6)) * 100

            safe_rec = SafeRecommendation(
                dose_mg=w_dose,
                route=w_regimen[0].route,
                frequency=frequency,
                regimen=w_regimen,
                time_h=w_result["time_h"],
                concentration_ng_ml=w_result["concentration_ng_ml"],
                pk_metrics=w_metrics,
                safety=w_safety,
                is_safe=w_safety.is_safe,
                rationale=rationale,
                scale_factor=round(w_sf, 4),
                delta_cmax_pct=round(delta_cmax, 2),
                delta_auc_pct=round(delta_auc, 2),
            )

    panel_slug = infer.resolve_panel_drug_slug(
        req.patient.compound_name,
        req.drug.panel_drug if req.drug else None,
    )
    clinical = _build_clinical_reasoning_items(panel_slug, baseline, safe_rec)

    return RecommendResponse(
        baseline=baseline,
        strategies=strategies,
        safe_recommendation=safe_rec,
        search_summary=search_summary,
        clinical_reasoning=clinical,
    )


def _build_strategy(
    sid: str,
    title: str,
    description: str,
    regimen: list[RegimenEvent],
    result: dict[str, Any],
    base_cmax: float,
    base_auc: float,
    scale_factor: float = 1.0,
) -> StrategyResult:
    m = result["pk_metrics"]
    delta_cmax = ((m.cmax_ng_ml - base_cmax) / max(base_cmax, 1e-6)) * 100
    delta_auc = ((m.auc_0_inf - base_auc) / max(base_auc, 1e-6)) * 100
    return StrategyResult(
        id=sid,
        title=title,
        description=description,
        regimen=regimen,
        time_h=result["time_h"],
        concentration_ng_ml=result["concentration_ng_ml"],
        pk_metrics=m,
        safety=result["safety"],
        delta_cmax_pct=round(delta_cmax, 2),
        delta_auc_pct=round(delta_auc, 2),
        scale_factor=round(scale_factor, 4),
    )


# ---------------------------------------------------------------------------
# Population simulation
# ---------------------------------------------------------------------------

@app.post("/predict/population", response_model=PredictPopulationResponse, tags=["population"])
async def predict_population(req: PredictPopulationRequest):
    _require_model()
    panel = infer.resolve_panel_drug_slug(
        req.patient.compound_name,
        req.drug.panel_drug if req.drug and req.drug.panel_drug else None,
    )
    det = _run_v2(
        req.patient.weight_kg,
        req.patient.compound_name,
        req.regimen,
        req.horizon_hr,
        req.dt_min,
        drug=req.drug,
        age_years=req.patient.age_years,
        sex=req.patient.sex,
    )
    events = [{"time_hr": e.time_hr, "dose_mg": e.dose_mg, "route": e.route} for e in req.regimen]
    pop = population_adapter.simulate_population(
        events,
        req.patient.weight_kg,
        req.horizon_hr,
        req.dt_min,
        n_samples=req.population.n_samples,
        omega_cl=req.population.omega_cl,
        omega_v=req.population.omega_v,
        omega_ka=req.population.omega_ka,
        seed=req.population.seed,
        panel_drug=panel,
        age_years=req.patient.age_years,
        sex=req.patient.sex,
    )
    return PredictPopulationResponse(
        deterministic=PredictV2Response(**det),
        population=PopulationResult(**pop),
    )


@app.get("/population/config", response_model=PopulationConfigResponse, tags=["population"])
async def get_pop_config():
    cfg = population_adapter.get_population_config()
    return PopulationConfigResponse(**cfg)


@app.post("/population/config", response_model=PopulationConfigResponse, tags=["population"])
async def update_pop_config(req: PopulationConfigUpdateRequest):
    updates = req.model_dump(exclude_none=True)
    cfg = population_adapter.update_population_config(updates)
    return PopulationConfigResponse(**cfg)


# ---------------------------------------------------------------------------
# PBPK configuration
# ---------------------------------------------------------------------------

@app.get("/pbpk/config", response_model=PBPKConfigResponse, tags=["pbpk"])
async def get_pbpk_config():
    cfg = pbpk_service.get_pbpk_config()
    return PBPKConfigResponse(**cfg)


@app.post("/pbpk/config", response_model=PBPKConfigResponse, tags=["pbpk"])
async def update_pbpk_config_endpoint(req: PBPKConfigUpdateRequest):
    updates = req.model_dump(exclude_none=True)
    cfg = pbpk_service.update_pbpk_config(updates)
    return PBPKConfigResponse(**cfg)


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

@app.post("/explain/v2", response_model=ExplainV2Response, tags=["explainability"])
async def explain_v2(req: ExplainV2Request):
    _require_model()

    total_dose = sum(e.dose_mg for e in req.regimen)
    weight = req.patient.weight_kg
    route = req.regimen[0].route
    panel = infer.resolve_panel_drug_slug(
        req.patient.compound_name,
        req.drug.panel_drug if req.drug and req.drug.panel_drug else None,
    )

    result = _run_v2(
        weight,
        req.patient.compound_name,
        req.regimen,
        req.horizon_hr,
        req.dt_min,
        drug=req.drug,
        age_years=req.patient.age_years,
        sex=req.patient.sex,
    )

    shap_raw = xai_service.explain_shap(
        total_dose,
        weight,
        panel_drug=panel,
        age_years=req.patient.age_years,
        sex=req.patient.sex,
        shap_seed=req.shap_seed,
    )
    sens_raw = xai_service.sensitivity_analysis(
        total_dose,
        weight,
        route=route,
        horizon_hr=req.horizon_hr,
        panel_drug=panel,
        age_years=req.patient.age_years,
        sex=req.patient.sex,
    )

    risk_score = result["safety"].risk_score
    is_safe = result["safety"].is_safe
    narrative_raw = xai_service.generate_narrative(
        shap_raw, sens_raw, total_dose, weight, risk_score, is_safe,
    )

    smiles = _resolve_smiles(req.drug, req.patient.compound_name) if panel is None else None
    drug_effect_raw = xai_service.drug_structure_effect(
        smiles,
        total_dose,
        weight,
        horizon_hr=req.horizon_hr,
        panel_drug=panel,
    )

    return ExplainV2Response(
        shap=SHAPResult(**shap_raw),
        sensitivity=SensitivityResult(**sens_raw),
        narrative=NarrativeResult(**narrative_raw),
        pk_metrics=result["pk_metrics"],
        safety=result["safety"],
        drug_structure=DrugStructureEffect(**drug_effect_raw),
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

@app.post("/report/v2", tags=["reports"])
async def report_v2(req: ReportV2Request):
    _require_model()

    panel = infer.resolve_panel_drug_slug(
        req.patient.compound_name,
        req.drug.panel_drug if req.drug and req.drug.panel_drug else None,
    )

    result = _run_v2(
        req.patient.weight_kg,
        req.patient.compound_name,
        req.regimen,
        req.horizon_hr,
        req.dt_min,
        drug=req.drug,
        age_years=req.patient.age_years,
        sex=req.patient.sex,
    )

    total_dose = sum(e.dose_mg for e in req.regimen)
    route = req.regimen[0].route
    shap_data = xai_service.explain_shap(
        total_dose,
        req.patient.weight_kg,
        panel_drug=panel,
        age_years=req.patient.age_years,
        sex=req.patient.sex,
        shap_seed=42,
    )
    sens_data = xai_service.sensitivity_analysis(
        total_dose,
        req.patient.weight_kg,
        route=route,
        horizon_hr=req.horizon_hr,
        panel_drug=panel,
        age_years=req.patient.age_years,
        sex=req.patient.sex,
    )
    narrative = xai_service.generate_narrative(
        shap_data, sens_data, total_dose, req.patient.weight_kg,
        result["safety"].risk_score, result["safety"].is_safe,
    )

    strategies_raw = None
    if req.include_recommendations:
        rec_req = RecommendRequest(
            patient=req.patient,
            regimen=req.regimen,
            horizon_hr=req.horizon_hr,
            drug=req.drug,
        )
        rec_resp = await recommend(rec_req)
        strategies_raw = [s.model_dump() for s in rec_resp.strategies]

    pop_data = None
    if req.include_population:
        events = [{"time_hr": e.time_hr, "dose_mg": e.dose_mg, "route": e.route} for e in req.regimen]
        pop_data = population_adapter.simulate_population(
            events,
            req.patient.weight_kg,
            req.horizon_hr,
            req.dt_min,
            panel_drug=panel,
            age_years=req.patient.age_years,
            sex=req.patient.sex,
        )

    pdf_bytes = report_service.generate_pdf(
        patient=req.patient.model_dump(),
        regimen=[e.model_dump() for e in req.regimen],
        pk_metrics=result["pk_metrics"].model_dump(),
        safety=result["safety"].model_dump(),
        pk_params=result["pk_params"],
        times=result["time_h"],
        conc=result["concentration_ng_ml"],
        shap_data=shap_data,
        sensitivity_data=sens_data,
        narrative=narrative,
        strategies=strategies_raw,
        population_data=pop_data,
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=pk_report.pdf"},
    )


@app.get("/model/state", response_model=ModelStateResponse, tags=["model"])
async def model_state():
    state = risk_service.get_model_state()
    return ModelStateResponse(
        calibration=state.get("calibration", {}),
        update_flag=state.get("update_flag", False),
        updated_at=state.get("updated_at"),
    )


@app.post("/model/update", response_model=ModelUpdateResponse, tags=["model"])
async def model_update(req: ModelUpdateRequest):
    if req.feedback_label.lower() not in ("safe", "unsafe"):
        raise HTTPException(400, "feedback_label must be 'safe' or 'unsafe'")
    state = risk_service.update_calibration(req.feedback_label)
    return ModelUpdateResponse(
        message=f"Calibration updated based on '{req.feedback_label}' feedback.",
        calibration=state["calibration"],
        update_flag=state["update_flag"],
    )
