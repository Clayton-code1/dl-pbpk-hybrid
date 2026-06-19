const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface PKMetrics {
  cmax_ng_ml: number;
  tmax_h: number;
  auc_0_inf: number;
  half_life_h: number;
  clearance_l_h: number;
  vd_l: number;
}

export interface PredictResponse {
  compound_name: string;
  dose_mg: number;
  time_h: number[];
  concentration_ng_ml: number[];
  pk_metrics: PKMetrics;
}

export interface PredictRequest {
  compound_name: string;
  dose_mg: number;
  weight_kg: number;
  route: string;
  smiles?: string;
}

/* ---- Explain V2 ---- */

export interface SHAPResult {
  features: string[];
  values: number[];
  base: number;
  target: string;
}

export interface SensitivityResult {
  parameters: string[];
  baseline_values: number[];
  delta_auc_pct: number[];
  delta_cmax_pct: number[];
  rank_auc: number[];
  rank_cmax: number[];
}

export interface NarrativeResult {
  summary: string;
  key_drivers: { feature: string; shap_value: number; direction: string }[];
}

export interface SafetyBlock {
  is_safe: boolean;
  risk_score: number;
  reason: string;
}

export interface ExplainV2Response {
  shap: SHAPResult;
  sensitivity: SensitivityResult;
  narrative: NarrativeResult;
  pk_metrics: PKMetrics | null;
  safety: SafetyBlock | null;
}

export interface ExplainV2Request {
  patient: { weight_kg: number; compound_name: string };
  regimen: { time_hr: number; dose_mg: number; route: string }[];
  horizon_hr: number;
  dt_min?: number;
}

/* ---- Population ---- */

export interface PercentileTriplet {
  p05: number;
  p50: number;
  p95: number;
}

export interface CurveBands {
  p05: number[];
  p50: number[];
  p95: number[];
}

export interface MetricsDist {
  cmax: PercentileTriplet;
  auc: PercentileTriplet;
  tmax: PercentileTriplet;
}

export interface PopulationRisk {
  p_unsafe: number;
  p_safe: number;
}

export interface PopulationResult {
  n_samples: number;
  base_params: Record<string, number>;
  omega: Record<string, number>;
  times_hr: number[];
  bands: CurveBands;
  metrics_dist: MetricsDist;
  population_risk: PopulationRisk;
}

export interface PredictPopulationResponse {
  deterministic: PredictV2Response;
  population: PopulationResult;
}

export interface PBPKBlock {
  enabled: boolean;
  tissue_units: string;
  params: Record<string, number>;
  physiology?: Record<string, unknown> | null;
  tissues?: Record<string, number[]> | null;
}

export interface PBPKConfig {
  enabled: boolean;
  Q_co_ref_L_per_h: number;
  flow_fracs: Record<string, number>;
  vol_fracs: Record<string, number>;
  Kp: Record<string, number>;
  f_hep: number;
  fu: number;
}

export interface ZeroshotMeta {
  smiles: string;
  cl_per_kg: number;
  vd_per_kg: number;
  ka: number;
  warning: string;
}

export interface DrugHint {
  name?: string;
  smiles?: string;
  panel_drug?: string;
  therapeutic_min_mg_L?: number;
  therapeutic_max_mg_L?: number;
}

export interface PredictV2Response {
  time_h: number[];
  concentration_ng_ml: number[];
  pk_metrics: PKMetrics;
  safety: SafetyBlock;
  model: { version: string; updated_at: string | null; update_flag: boolean; model_used?: string | null };
  pk_params: Record<string, number>;
  pbpk?: PBPKBlock | null;
  population?: PopulationResult | null;
  zeroshot?: ZeroshotMeta | null;
}

export interface PopulationConfig {
  omega_cl: number;
  omega_v: number;
  omega_ka: number;
  n_samples: number;
}

/* ---- Recommend ---- */

export interface RegimenEvent {
  time_hr: number;
  dose_mg: number;
  route: string;
}

export interface StrategyResult {
  id: string;
  title: string;
  description: string;
  regimen: RegimenEvent[];
  time_h: number[];
  concentration_ng_ml: number[];
  pk_metrics: PKMetrics;
  safety: SafetyBlock;
  delta_cmax_pct: number;
  delta_auc_pct: number;
  scale_factor: number;
}

export interface SafeRecommendation {
  dose_mg: number;
  route: string;
  frequency: string;
  regimen: RegimenEvent[];
  time_h: number[];
  concentration_ng_ml: number[];
  pk_metrics: PKMetrics;
  safety: SafetyBlock;
  is_safe: boolean;
  rationale: string;
  scale_factor: number;
  delta_cmax_pct: number;
  delta_auc_pct: number;
}

export interface RecommendResponse {
  baseline: PredictV2Response;
  strategies: StrategyResult[];
  safe_recommendation: SafeRecommendation | null;
  search_summary: string;
}

export interface RecommendRequest {
  patient: { weight_kg: number; compound_name: string };
  regimen: RegimenEvent[];
  horizon_hr: number;
  drug?: DrugHint;
}

/* ---- Report V2 ---- */

export interface ReportV2Request {
  patient: { weight_kg: number; compound_name: string };
  regimen: { time_hr: number; dose_mg: number; route: string }[];
  horizon_hr: number;
  dt_min?: number;
  include_recommendations?: boolean;
  include_population?: boolean;
}

/* ---- API calls ---- */

export async function fetchHealth(): Promise<{ status: string; version: string }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error("API unreachable");
  return res.json();
}

export async function fetchPrediction(req: PredictRequest): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    let detail = `Prediction failed: ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch { /* use default message */ }
    throw new Error(detail);
  }
  return res.json();
}

export async function fetchExplainV2(req: ExplainV2Request): Promise<ExplainV2Response> {
  const res = await fetch(`${API_BASE}/explain/v2`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`Explanation failed: ${res.statusText}`);
  return res.json();
}

export async function fetchReportV2(req: ReportV2Request): Promise<Blob> {
  const res = await fetch(`${API_BASE}/report/v2`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`Report generation failed: ${res.statusText}`);
  return res.blob();
}

export async function fetchPredictPopulation(
  req: ExplainV2Request,
): Promise<PredictPopulationResponse> {
  const res = await fetch(`${API_BASE}/predict/population`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      patient: req.patient,
      regimen: req.regimen,
      horizon_hr: req.horizon_hr,
      dt_min: req.dt_min ?? 5.0,
      population: { n_samples: 100 },
    }),
  });
  if (!res.ok) throw new Error(`Population simulation failed: ${res.statusText}`);
  return res.json();
}

export async function fetchPopulationConfig(): Promise<PopulationConfig> {
  const res = await fetch(`${API_BASE}/population/config`);
  if (!res.ok) throw new Error("Failed to fetch population config");
  return res.json();
}

export async function fetchPredictV2(
  req: ExplainV2Request & { include_tissues?: boolean; pbpk_mode?: string; drug?: DrugHint },
): Promise<PredictV2Response> {
  const res = await fetch(`${API_BASE}/predict/v2`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      patient: req.patient,
      regimen: req.regimen,
      horizon_hr: req.horizon_hr,
      dt_min: req.dt_min ?? 5.0,
      include_tissues: req.include_tissues ?? false,
      pbpk_mode: req.pbpk_mode ?? "pbpk_lite",
      ...(req.drug ? { drug: req.drug } : {}),
    }),
  });
  if (!res.ok) {
    let detail = `Prediction V2 failed: ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch { /* use default message */ }
    throw new Error(detail);
  }
  return res.json();
}

export async function fetchPBPKConfig(): Promise<PBPKConfig> {
  const res = await fetch(`${API_BASE}/pbpk/config`);
  if (!res.ok) throw new Error("Failed to fetch PBPK config");
  return res.json();
}

export async function fetchRecommendations(req: RecommendRequest): Promise<RecommendResponse> {
  const res = await fetch(`${API_BASE}/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    let detail = `Recommendation failed: ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch { /* use default message */ }
    throw new Error(detail);
  }
  return res.json();
}

/* ---- Molecular attribution (pre-computed, read-only) ---- */

export interface GraphAtomAttribution {
  atom_index: number;
  atom_symbol: string;
  importance_score: number;
  rank: number;
}

export interface GraphAttributionResponse {
  drug: string;
  atom_count: number;
  attributions: GraphAtomAttribution[];
}

export async function fetchGraphAttribution(drug: string): Promise<GraphAttributionResponse> {
  const res = await fetch(`${API_BASE}/explainability/graph/${encodeURIComponent(drug)}`);
  if (res.status === 404) throw new Error("not_found");
  if (!res.ok) throw new Error(`Molecular attribution failed: ${res.statusText}`);
  return res.json();
}
