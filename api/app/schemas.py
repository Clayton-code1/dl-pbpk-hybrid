"""Pydantic models shared across endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------

class PKMetrics(BaseModel):
    cmax_ng_ml: float
    tmax_h: float
    auc_0_inf: float
    half_life_h: float
    clearance_l_h: float
    vd_l: float


class SafetyBlock(BaseModel):
    is_safe: bool
    risk_score: float
    reason: str


class ModelMeta(BaseModel):
    version: str
    updated_at: str | None = None
    update_flag: bool = False
    model_used: str | None = Field(None, description="'gnn' or 'mlp' — which encoder was used")


# ---------------------------------------------------------------------------
# Legacy /predict
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    compound_name: str = Field(..., examples=["Compound-A"])
    dose_mg: float = Field(..., gt=0, examples=[100.0])
    weight_kg: float = Field(70.0, gt=0, examples=[70.0])
    route: str = Field("oral", examples=["oral", "iv"])


class PredictResponse(BaseModel):
    compound_name: str
    dose_mg: float
    time_h: list[float]
    concentration_ng_ml: list[float]
    pk_metrics: PKMetrics


# ---------------------------------------------------------------------------
# /predict/v2
# ---------------------------------------------------------------------------

class DrugInfo(BaseModel):
    name: str = Field("Theophylline", examples=["Theophylline"])
    smiles: str | None = Field(None, description="SMILES string for legacy theophylline GNN path")
    panel_drug: str | None = Field(
        None,
        description=(
            "Optional canonical slug: theophylline|warfarin|midazolam|caffeine|acetaminophen|digoxin. "
            "When set, loads artifacts/models/hybrid_gnn_pbpk_{slug}_v1 (paper-aligned)."
        ),
    )


class PatientInfo(BaseModel):
    weight_kg: float = Field(70.0, gt=0)
    compound_name: str = Field("Theophylline", examples=["Theophylline"])
    age_years: float = Field(40.0, ge=0, le=120, description="Covariate z-scored like training CSV")
    sex: float = Field(
        0.0,
        ge=0,
        le=1,
        description="Binary coding as in Phase-1 PK CSVs (0/1)",
    )


class RegimenEvent(BaseModel):
    time_hr: float = Field(0.0, ge=0, description="Administration time (hours)")
    dose_mg: float = Field(..., gt=0, description="Dose in mg")
    route: str = Field("oral", examples=["oral", "iv"])


class PredictV2Request(BaseModel):
    patient: PatientInfo = Field(default_factory=PatientInfo)
    regimen: list[RegimenEvent] = Field(..., min_length=1)
    horizon_hr: float = Field(48.0, gt=0)
    dt_min: float = Field(5.0, gt=0, description="Output time resolution in minutes")
    include_tissues: bool = Field(False, description="Return per-tissue concentration curves (PBPK)")
    pbpk_mode: str = Field("pbpk_lite", description="Simulation engine: 'pbpk_lite' or 'pk1cpt'")
    drug: DrugInfo | None = Field(None, description="Optional drug info with SMILES for GNN")


class CurvePoint(BaseModel):
    time_hr: float
    conc_mg_l: float


class PBPKBlock(BaseModel):
    enabled: bool = False
    tissue_units: str = "mg/L"
    params: dict[str, float] = Field(default_factory=dict)
    physiology: dict | None = None
    tissues: dict[str, list[float]] | None = Field(None, description="Per-tissue concentration curves")


class PredictV2Response(BaseModel):
    time_h: list[float]
    concentration_ng_ml: list[float]
    pk_metrics: PKMetrics
    safety: SafetyBlock
    model: ModelMeta
    pk_params: dict[str, float]
    pbpk: PBPKBlock | None = Field(None, description="PBPK-lite tissue data (when enabled)")
    population: dict | None = Field(None, description="Population uncertainty bands (when requested)")


# ---------------------------------------------------------------------------
# /recommend
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    patient: PatientInfo = Field(default_factory=PatientInfo)
    regimen: list[RegimenEvent] = Field(..., min_length=1)
    horizon_hr: float = Field(48.0, gt=0)
    drug: DrugInfo | None = Field(None, description="Optional panel_drug / SMILES (same as /predict/v2)")


class StrategyResult(BaseModel):
    id: str
    title: str
    description: str
    regimen: list[RegimenEvent]
    time_h: list[float]
    concentration_ng_ml: list[float]
    pk_metrics: PKMetrics
    safety: SafetyBlock
    delta_cmax_pct: float
    delta_auc_pct: float
    scale_factor: float = Field(1.0, description="Dose scaling factor applied (1.0 = unchanged)")


class SafeRecommendation(BaseModel):
    dose_mg: float
    route: str
    frequency: str
    regimen: list[RegimenEvent]
    time_h: list[float]
    concentration_ng_ml: list[float]
    pk_metrics: PKMetrics
    safety: SafetyBlock
    is_safe: bool
    rationale: str
    scale_factor: float = Field(1.0, description="Dose scaling factor vs original dose")
    delta_cmax_pct: float = 0.0
    delta_auc_pct: float = 0.0


class ClinicalReasoningItem(BaseModel):
    """Literature-band interpretation for examiner-facing /recommend payloads."""

    context: str = Field(..., description="baseline | recommended_regimen")
    adjustment_pct: float | None = Field(None)
    reasoning_text: str
    evidence_tier: str = Field(
        ...,
        description="literature_band | extrapolation | out_of_scope",
    )


class RecommendResponse(BaseModel):
    baseline: PredictV2Response
    strategies: list[StrategyResult]
    safe_recommendation: SafeRecommendation | None = Field(
        None, description="Primary recommended safe regimen found by automatic search"
    )
    search_summary: str = Field(
        "", description="Human-readable explanation of the automatic safe-dose search"
    )
    clinical_reasoning: list[ClinicalReasoningItem] = Field(
        default_factory=list,
        description="Therapeutic-window narrative aligned with experiments.reference_pk",
    )


# ---------------------------------------------------------------------------
# /model/*
# ---------------------------------------------------------------------------

class ModelStateResponse(BaseModel):
    calibration: dict[str, float]
    update_flag: bool
    updated_at: str | None


class ModelUpdateRequest(BaseModel):
    feedback_label: str = Field(..., description="'safe' or 'unsafe'")


class ModelUpdateResponse(BaseModel):
    message: str
    calibration: dict[str, float]
    update_flag: bool


# ---------------------------------------------------------------------------
# /predict/population
# ---------------------------------------------------------------------------

class PopulationOverrides(BaseModel):
    n_samples: int | None = Field(None, ge=10, le=500)
    omega_cl: float | None = Field(None, ge=0.01, le=2.0)
    omega_v: float | None = Field(None, ge=0.01, le=2.0)
    omega_ka: float | None = Field(None, ge=0.01, le=2.0)
    seed: int | None = None


class PredictPopulationRequest(BaseModel):
    patient: PatientInfo = Field(default_factory=PatientInfo)
    regimen: list[RegimenEvent] = Field(..., min_length=1)
    horizon_hr: float = Field(48.0, gt=0)
    dt_min: float = Field(5.0, gt=0)
    population: PopulationOverrides = Field(default_factory=PopulationOverrides)
    drug: DrugInfo | None = Field(
        None,
        description="Optional panel_drug; when resolved, base PK uses multi-drug hybrid (not MLP).",
    )


class PercentileTriplet(BaseModel):
    p05: float
    p50: float
    p95: float


class CurveBands(BaseModel):
    p05: list[float]
    p50: list[float]
    p95: list[float]


class MetricsDist(BaseModel):
    cmax: PercentileTriplet
    auc: PercentileTriplet
    tmax: PercentileTriplet


class PopulationRisk(BaseModel):
    p_unsafe: float
    p_safe: float


class PopulationResult(BaseModel):
    n_samples: int
    base_params: dict[str, float]
    omega: dict[str, float]
    times_hr: list[float]
    bands: CurveBands
    metrics_dist: MetricsDist
    population_risk: PopulationRisk


class PredictPopulationResponse(BaseModel):
    deterministic: PredictV2Response
    population: PopulationResult


class PopulationConfigResponse(BaseModel):
    omega_cl: float
    omega_v: float
    omega_ka: float
    n_samples: int


class PopulationConfigUpdateRequest(BaseModel):
    omega_cl: float | None = Field(None, ge=0.01, le=2.0)
    omega_v: float | None = Field(None, ge=0.01, le=2.0)
    omega_ka: float | None = Field(None, ge=0.01, le=2.0)
    n_samples: int | None = Field(None, ge=10, le=500)


# ---------------------------------------------------------------------------
# /explain/v2
# ---------------------------------------------------------------------------

class ExplainV2Request(BaseModel):
    patient: PatientInfo = Field(default_factory=PatientInfo)
    regimen: list[RegimenEvent] = Field(..., min_length=1)
    horizon_hr: float = Field(48.0, gt=0)
    dt_min: float = Field(5.0, gt=0)
    drug: DrugInfo | None = Field(None, description="Optional panel_drug for multi-drug SHAP path")
    shap_seed: int | None = Field(
        None,
        description="Optional RNG seed for KernelSHAP / background sampling (reproducibility).",
    )


class SHAPResult(BaseModel):
    features: list[str]
    values: list[float]
    base: float
    target: str
    attribution_backend: str | None = Field(
        None,
        description="'mlp' | 'panel_multidrug' — which predictor SHAP was aligned to",
    )


class SensitivityResult(BaseModel):
    parameters: list[str]
    baseline_values: list[float] = []
    delta_auc_pct: list[float]
    delta_cmax_pct: list[float]
    rank_auc: list[int]
    rank_cmax: list[int]


class NarrativeResult(BaseModel):
    summary: str
    key_drivers: list[dict] = []


class DrugStructureEffect(BaseModel):
    drug_structure_delta_risk: float = 0.0
    explanation: str = ""


class ExplainV2Response(BaseModel):
    shap: SHAPResult
    sensitivity: SensitivityResult
    narrative: NarrativeResult
    pk_metrics: PKMetrics | None = None
    safety: SafetyBlock | None = None
    drug_structure: DrugStructureEffect | None = None


# ---------------------------------------------------------------------------
# /report/v2
# ---------------------------------------------------------------------------

class ReportV2Request(BaseModel):
    patient: PatientInfo = Field(default_factory=PatientInfo)
    regimen: list[RegimenEvent] = Field(..., min_length=1)
    horizon_hr: float = Field(48.0, gt=0)
    dt_min: float = Field(5.0, gt=0)
    include_recommendations: bool = Field(False, description="Include recommendation strategies in report")
    include_population: bool = Field(False, description="Include population uncertainty bands in report")
    drug: DrugInfo | None = Field(None, description="Optional panel_drug / SMILES")


# ---------------------------------------------------------------------------
# /pbpk/config
# ---------------------------------------------------------------------------

class PBPKConfigResponse(BaseModel):
    enabled: bool
    Q_co_ref_L_per_h: float
    flow_fracs: dict[str, float]
    vol_fracs: dict[str, float]
    Kp: dict[str, float]
    f_hep: float
    fu: float


class PBPKConfigUpdateRequest(BaseModel):
    enabled: bool | None = None
    Q_co_ref_L_per_h: float | None = Field(None, gt=0)
    flow_fracs: dict[str, float] | None = None
    vol_fracs: dict[str, float] | None = None
    Kp: dict[str, float] | None = None
    f_hep: float | None = Field(None, ge=0, le=1)
    fu: float | None = Field(None, ge=0, le=1)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    inference_ready: bool = False
    panel_drugs_available: dict[str, bool] = Field(default_factory=dict)
    model_used_hint: str = Field("", description="Primary backend if detectable (informational)")
