"use client";

import { useState, useCallback } from "react";
import { usePrediction } from "@/context/PredictionContext";
import { useTheme } from "@/context/ThemeContext";
import {
  fetchPredictPopulation,
  fetchPredictV2,
  fetchReportV2,
  type PopulationResult,
  type PBPKBlock,
} from "@/lib/api";
import Card from "./Card";
import MetricCard from "./MetricCard";
import SafetyBadge from "./SafetyBadge";
import Button from "./Buttons";
import CurveChart, { type ChartSeries, type BandConfig } from "./CurveChart";
import {
  Loader2,
  ArrowRight,
  Users,
  ToggleLeft,
  ToggleRight,
  FlaskConical,
  TrendingUp,
  Clock,
  BarChart3,
  Timer,
  Droplets,
  Box,
  Download,
  ChevronDown,
} from "lucide-react";

const TISSUE_COLORS: Record<string, string> = {
  liver: "#f97316",
  kidney: "#a855f7",
  muscle: "#22c55e",
  adipose: "#eab308",
  lung: "#06b6d4",
};

const TISSUE_OPTIONS = ["liver", "kidney", "muscle", "adipose", "lung"];

const KNOWN_COMPOUNDS = [
  "theophylline", "theo",
  "warfarin", "coumadin",
  "midazolam", "versed",
  "caffeine",
  "acetaminophen", "paracetamol", "apap", "tylenol",
  "digoxin",
];

function isKnownCompound(name: string): boolean {
  return KNOWN_COMPOUNDS.includes(name.trim().toLowerCase());
}

function generateSessionLabel(): string {
  const d = new Date();
  const y = d.getFullYear();
  const M = String(d.getMonth() + 1).padStart(2, "0");
  const D = String(d.getDate()).padStart(2, "0");
  const h = String(d.getHours()).padStart(2, "0");
  const m = String(d.getMinutes()).padStart(2, "0");
  return `RUN-${y}${M}${D}-${h}${m}`;
}

interface FieldErrors {
  compound?: string;
  smiles?: string;
  dose?: string;
  weight?: string;
  route?: string;
  window?: string;
}

interface PredictionPanelProps {
  onGoToRecommendations?: () => void;
}

export default function PredictionPanel({ onGoToRecommendations }: PredictionPanelProps) {
  const {
    request,
    response,
    loading,
    error,
    isSafe,
    setRequest,
    setResponse,
    setDrug,
    setLoading,
    setError,
    setSessionMeta,
    sessionLabel,
    predictionGeneratedAt,
  } = usePrediction();
  const { dark } = useTheme();

  const [compound, setCompound] = useState(request?.compound_name ?? "");
  const [smiles, setSmiles] = useState("");
  const [dose, setDose] = useState(request?.dose_mg?.toString() ?? "");
  const [weight, setWeight] = useState(request?.weight_kg?.toString() ?? "");
  const [route, setRoute] = useState(request?.route ?? "");
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({});

  const [therapMin, setTherapMin] = useState("");
  const [therapMax, setTherapMax] = useState("");
  const [therapOpen, setTherapOpen] = useState(false);

  const [showPopulation, setShowPopulation] = useState(false);
  const [popData, setPopData] = useState<PopulationResult | null>(null);
  const [popLoading, setPopLoading] = useState(false);

  const [showTissues, setShowTissues] = useState(false);
  const [tissueData, setTissueData] = useState<PBPKBlock | null>(null);
  const [tissueLoading, setTissueLoading] = useState(false);
  const [selectedTissue, setSelectedTissue] = useState<string>("liver");
  const [reportDownloading, setReportDownloading] = useState(false);

  const validate = (): FieldErrors => {
    const errors: FieldErrors = {};
    const trimmedCompound = compound.trim();
    const trimmedSmiles = smiles.trim();

    if (!trimmedCompound && !trimmedSmiles) {
      errors.compound = "Please enter a compound name or provide a SMILES structure.";
    } else if (trimmedCompound && !isKnownCompound(trimmedCompound) && !trimmedSmiles) {
      errors.compound = "Unknown compound. Please provide a SMILES structure or use a supported panel drug.";
    }

    const isZeroShot = !!trimmedSmiles && !isKnownCompound(trimmedCompound);
    if (isZeroShot) {
      const minNum = parseFloat(therapMin);
      const maxNum = parseFloat(therapMax);
      const minOk = therapMin.trim() !== "" && !isNaN(minNum) && minNum > 0;
      const maxOk = therapMax.trim() !== "" && !isNaN(maxNum) && maxNum > 0;
      if (!minOk || !maxOk) {
        errors.window = "This compound requires a therapeutic window (min and max mg/L) before a safety verdict can be produced.";
      }
    }

    const doseNum = parseFloat(dose);
    if (!dose.trim() || isNaN(doseNum) || doseNum <= 0) {
      errors.dose = "Please enter a valid dose.";
    }

    const weightNum = parseFloat(weight);
    if (!weight.trim() || isNaN(weightNum) || weightNum <= 0) {
      errors.weight = "Please enter a valid body weight.";
    }

    if (!route) {
      errors.route = "Please select an administration route.";
    }

    return errors;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const errors = validate();
    setFieldErrors(errors);
    if (Object.keys(errors).length > 0) {
      if (errors.window) setTherapOpen(true);
      return;
    }

    setLoading(true);
    setError(null);
    setPopData(null);
    setTissueData(null);

    const doseNum = parseFloat(dose);
    const weightNum = parseFloat(weight);
    const trimmedSmiles = smiles.trim();
    // When SMILES is provided without a name, send a non-panel placeholder so routing works
    const compoundLabel = compound.trim() || (trimmedSmiles ? "compound" : "");

    const req = {
      compound_name: compoundLabel,
      dose_mg: doseNum,
      weight_kg: weightNum,
      route,
      ...(trimmedSmiles ? { smiles: trimmedSmiles } : {}),
    };
    setRequest(req);
    try {
      const therapMinNum = parseFloat(therapMin);
      const therapMaxNum = parseFloat(therapMax);
      const data = await fetchPredictV2({
        patient: { weight_kg: weightNum, compound_name: compoundLabel },
        regimen: [{ time_hr: 0, dose_mg: doseNum, route }],
        horizon_hr: 48,
        drug: {
          name: compoundLabel,
          ...(trimmedSmiles ? { smiles: trimmedSmiles } : {}),
          ...(therapMinNum > 0 && !isNaN(therapMinNum)
            ? { therapeutic_min_mg_L: therapMinNum }
            : {}),
          ...(therapMaxNum > 0 && !isNaN(therapMaxNum)
            ? { therapeutic_max_mg_L: therapMaxNum }
            : {}),
        },
      });
      setResponse(data);
      setDrug({
        name: compoundLabel,
        ...(trimmedSmiles ? { smiles: trimmedSmiles } : {}),
        ...(therapMinNum > 0 && !isNaN(therapMinNum)
          ? { therapeutic_min_mg_L: therapMinNum }
          : {}),
        ...(therapMaxNum > 0 && !isNaN(therapMaxNum)
          ? { therapeutic_max_mg_L: therapMaxNum }
          : {}),
      });
      setSessionMeta(generateSessionLabel(), new Date().toLocaleString());
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Prediction failed";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handlePopulationToggle = useCallback(async () => {
    if (showPopulation) {
      setShowPopulation(false);
      return;
    }
    if (!request) return;
    setShowPopulation(true);
    if (popData) return;

    setPopLoading(true);
    try {
      const result = await fetchPredictPopulation({
        patient: { weight_kg: request.weight_kg, compound_name: request.compound_name },
        regimen: [{ time_hr: 0, dose_mg: request.dose_mg, route: request.route }],
        horizon_hr: 48,
      });
      setPopData(result.population);
    } catch {
      setPopData(null);
    } finally {
      setPopLoading(false);
    }
  }, [showPopulation, request, popData]);

  const handleTissueToggle = useCallback(async () => {
    if (showTissues) {
      setShowTissues(false);
      return;
    }
    if (!request) return;
    setShowTissues(true);
    if (tissueData) return;

    setTissueLoading(true);
    try {
      const result = await fetchPredictV2({
        patient: { weight_kg: request.weight_kg, compound_name: request.compound_name },
        regimen: [{ time_hr: 0, dose_mg: request.dose_mg, route: request.route }],
        horizon_hr: 48,
        include_tissues: true,
        pbpk_mode: "pbpk_lite",
      });
      setTissueData(result.pbpk ?? null);
    } catch {
      setTissueData(null);
    } finally {
      setTissueLoading(false);
    }
  }, [showTissues, request, tissueData]);

  const handleDownloadReport = useCallback(async () => {
    if (!request) return;
    setReportDownloading(true);
    try {
      const blob = await fetchReportV2({
        patient: { weight_kg: request.weight_kg, compound_name: request.compound_name },
        regimen: [{ time_hr: 0, dose_mg: request.dose_mg, route: request.route }],
        horizon_hr: 48,
        include_recommendations: true,
        include_population: false,
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `pk_report_${request.compound_name}_${request.dose_mg}mg.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } finally {
      setReportDownloading(false);
    }
  }, [request]);

  const chartData = response
    ? response.time_h.map((t, i) => {
        const point: Record<string, number> = {
          time: t,
          concentration: response.concentration_ng_ml[i],
        };
        if (showPopulation && popData) {
          const popIdx = popData.times_hr.findIndex((pt) => Math.abs(pt - t) < 0.01);
          if (popIdx >= 0) {
            point.p05 = popData.bands.p05[popIdx];
            point.p50 = popData.bands.p50[popIdx];
            point.p95 = popData.bands.p95[popIdx];
          }
        }
        if (showTissues && tissueData?.tissues && tissueData.tissues[selectedTissue]) {
          const tissueArr = tissueData.tissues[selectedTissue];
          if (i < tissueArr.length) {
            point.tissue = tissueArr[i] * 1000;
          }
        }
        return point;
      })
    : [];

  const plasmaColor = "#14b8a6";
  const series: ChartSeries[] = [
    { key: "concentration", label: "Plasma (central)", color: plasmaColor },
  ];
  if (showTissues && tissueData?.tissues && tissueData.tissues[selectedTissue]) {
    series.push({
      key: "tissue",
      label: `${selectedTissue.charAt(0).toUpperCase() + selectedTissue.slice(1)} (PBPK)`,
      color: TISSUE_COLORS[selectedTissue] ?? "#8b5cf6",
      strokeDash: "6 3",
    });
  }

  const bandConfig: BandConfig | undefined =
    showPopulation && popData
      ? { lowerKey: "p05", upperKey: "p95", medianKey: "p50", color: "#6366f1", label: "90% CI" }
      : undefined;

  const METRIC_MAP: { key: string; label: string; unit: string; description: string; icon: typeof TrendingUp }[] = [
    { key: "cmax_ng_ml", label: "Cmax", unit: "ng/mL", description: "Peak plasma concentration", icon: TrendingUp },
    { key: "tmax_h", label: "Tmax", unit: "h", description: "Time to peak concentration", icon: Clock },
    { key: "auc_0_inf", label: "AUC₀₋∞", unit: "ng·h/mL", description: "Total drug exposure", icon: BarChart3 },
    { key: "half_life_h", label: "t½", unit: "h", description: "Elimination half-life", icon: Timer },
    { key: "clearance_l_h", label: "CL", unit: "L/h", description: "Systemic clearance", icon: Droplets },
    { key: "vd_l", label: "Vd", unit: "L", description: "Volume of distribution", icon: Box },
  ];

  return (
    <div className="space-y-6 fade-in">
      {/* Input Form — clinical sections */}
      <Card>
        <form onSubmit={handleSubmit} className="space-y-5">
          {/* A. Drug Information */}
          <div className="space-y-3">
            <p className="text-xs font-semibold uppercase tracking-wider text-teal-600 dark:text-teal-400">
              Drug Information
            </p>
            <div className="grid gap-x-4 gap-y-1 sm:grid-cols-2 lg:grid-cols-3">
              <label className="space-y-1.5">
                <span className="text-sm font-medium text-slate-700 dark:text-slate-300">Compound</span>
                <input
                  type="text"
                  value={compound}
                  onChange={(e) => { setCompound(e.target.value); setFieldErrors((p) => ({ ...p, compound: undefined, window: undefined })); }}
                  placeholder="Enter compound name"
                  className={`w-full rounded-md border bg-white px-3.5 py-2.5 text-sm dark:bg-slate-700 dark:text-white ${fieldErrors.compound ? "border-red-400 dark:border-red-500" : "border-slate-300 dark:border-slate-600"}`}
                />
                {fieldErrors.compound && (
                  <p className="text-xs text-red-600 dark:text-red-400">{fieldErrors.compound}</p>
                )}
              </label>
              <label className="space-y-1.5 sm:col-span-2 lg:col-span-2">
                <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                  SMILES
                  <span className="ml-1.5 font-normal text-slate-400 dark:text-slate-500">(optional)</span>
                </span>
                <input
                  type="text"
                  value={smiles}
                  onChange={(e) => { setSmiles(e.target.value); setFieldErrors((p) => ({ ...p, compound: undefined, window: undefined })); }}
                  placeholder="Enter SMILES structure (optional)"
                  className="w-full rounded-md border border-slate-300 bg-white px-3.5 py-2.5 text-sm font-mono dark:border-slate-600 dark:bg-slate-700 dark:text-white"
                />
                <p className="text-xs text-slate-400 dark:text-slate-500">Provide SMILES for new or unsupported compounds.</p>
              </label>
            </div>

            {/* Therapeutic window disclosure — inputs left unchanged per visual-only scope */}
            <div className="mt-2">
              <button
                type="button"
                onClick={() => setTherapOpen((v) => !v)}
                className={`flex items-center gap-1.5 text-xs font-medium transition-colors ${fieldErrors.window ? "text-red-500 hover:text-red-600 dark:text-red-400 dark:hover:text-red-300" : "text-slate-500 hover:text-teal-600 dark:text-slate-400 dark:hover:text-teal-400"}`}
              >
                <ChevronDown
                  className={`h-3.5 w-3.5 transition-transform duration-200 ${therapOpen ? "rotate-180" : ""}`}
                />
                {fieldErrors.window ? "Therapeutic window required for this compound" : "Set therapeutic window (optional — for known drugs)"}
              </button>
              {fieldErrors.window && (
                <p className="mt-1 text-xs text-red-600 dark:text-red-400">{fieldErrors.window}</p>
              )}
              {therapOpen && (
                <div className="mt-3 space-y-2">
                  <div className="grid max-w-sm grid-cols-2 gap-4">
                    <label className="space-y-1.5">
                      <span className="text-sm font-medium text-gray-700 dark:text-slate-300">
                        Min (mg/L)
                      </span>
                      <input
                        type="number"
                        min={0}
                        step="any"
                        value={therapMin}
                        onChange={(e) => { setTherapMin(e.target.value); setFieldErrors((p) => ({ ...p, window: undefined })); }}
                        placeholder="e.g. 5"
                        className={`w-full rounded-xl border bg-white px-3.5 py-2.5 text-sm dark:bg-slate-700 dark:text-white ${fieldErrors.window ? "border-red-400 dark:border-red-500" : "border-gray-300 dark:border-slate-600"}`}
                      />
                    </label>
                    <label className="space-y-1.5">
                      <span className="text-sm font-medium text-gray-700 dark:text-slate-300">
                        Max (mg/L)
                      </span>
                      <input
                        type="number"
                        min={0}
                        step="any"
                        value={therapMax}
                        onChange={(e) => { setTherapMax(e.target.value); setFieldErrors((p) => ({ ...p, window: undefined })); }}
                        placeholder="e.g. 20"
                        className={`w-full rounded-xl border bg-white px-3.5 py-2.5 text-sm dark:bg-slate-700 dark:text-white ${fieldErrors.window ? "border-red-400 dark:border-red-500" : "border-gray-300 dark:border-slate-600"}`}
                      />
                    </label>
                  </div>
                  <p className="text-xs text-gray-400 dark:text-slate-500">
                    When set, safety classification uses this window instead of the generic sigmoid model.
                  </p>
                </div>
              )}
            </div>
          </div>

          <div className="border-t border-slate-100 dark:border-slate-700" />

          {/* B. Patient Context */}
          <div className="space-y-3">
            <p className="text-xs font-semibold uppercase tracking-wider text-teal-600 dark:text-teal-400">
              Patient Context
            </p>
            <div className="grid gap-x-4 gap-y-1 sm:grid-cols-2 lg:grid-cols-3">
              <label className="space-y-1.5 max-w-xs">
                <span className="text-sm font-medium text-slate-700 dark:text-slate-300">Weight (kg)</span>
                <input
                  type="number"
                  min={1}
                  step="any"
                  value={weight}
                  onChange={(e) => { setWeight(e.target.value); setFieldErrors((p) => ({ ...p, weight: undefined })); }}
                  placeholder="Enter body weight in kg"
                  className={`w-full rounded-md border bg-white px-3.5 py-2.5 text-sm dark:bg-slate-700 dark:text-white ${fieldErrors.weight ? "border-red-400 dark:border-red-500" : "border-slate-300 dark:border-slate-600"}`}
                />
                {fieldErrors.weight && (
                  <p className="text-xs text-red-600 dark:text-red-400">{fieldErrors.weight}</p>
                )}
              </label>
            </div>
          </div>

          <div className="border-t border-slate-100 dark:border-slate-700" />

          {/* C. Administration Details */}
          <div className="space-y-3">
            <p className="text-xs font-semibold uppercase tracking-wider text-teal-600 dark:text-teal-400">
              Administration Details
            </p>
            <div className="grid gap-x-4 gap-y-1 sm:grid-cols-2 lg:grid-cols-3">
              <label className="space-y-1.5">
                <span className="text-sm font-medium text-slate-700 dark:text-slate-300">Dose (mg)</span>
                <input
                  type="number"
                  min={1}
                  step="any"
                  value={dose}
                  onChange={(e) => { setDose(e.target.value); setFieldErrors((p) => ({ ...p, dose: undefined })); }}
                  placeholder="Enter dose in mg"
                  className={`w-full rounded-md border bg-white px-3.5 py-2.5 text-sm dark:bg-slate-700 dark:text-white ${fieldErrors.dose ? "border-red-400 dark:border-red-500" : "border-slate-300 dark:border-slate-600"}`}
                />
                {fieldErrors.dose && (
                  <p className="text-xs text-red-600 dark:text-red-400">{fieldErrors.dose}</p>
                )}
              </label>
              <label className="space-y-1.5">
                <span className="text-sm font-medium text-slate-700 dark:text-slate-300">Route</span>
                <select
                  value={route}
                  onChange={(e) => { setRoute(e.target.value); setFieldErrors((p) => ({ ...p, route: undefined })); }}
                  className={`w-full rounded-md border bg-white px-3.5 py-2.5 text-sm dark:bg-slate-700 dark:text-white ${fieldErrors.route ? "border-red-400 dark:border-red-500" : "border-slate-300 dark:border-slate-600"} ${!route ? "text-slate-400 dark:text-slate-500" : ""}`}
                >
                  <option value="" disabled>Select route</option>
                  <option value="oral">Oral</option>
                  <option value="iv">Intravenous</option>
                </select>
                {fieldErrors.route && (
                  <p className="text-xs text-red-600 dark:text-red-400">{fieldErrors.route}</p>
                )}
              </label>
            </div>
          </div>

          <Button type="submit" disabled={loading} size="lg">
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Running Prediction...
              </>
            ) : (
              "Run Prediction"
            )}
          </Button>
        </form>
      </Card>

      {error && (
        <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700 fade-in dark:border-red-800 dark:bg-red-900/30 dark:text-red-300">
          {error}
        </div>
      )}

      {response && request && (
        <div className="space-y-6 slide-up">
          {/* Session metadata */}
          <div className="flex flex-wrap items-center gap-x-6 gap-y-1 text-xs text-slate-500 dark:text-slate-400">
            {predictionGeneratedAt && (
              <span>Prediction generated: {predictionGeneratedAt}</span>
            )}
            {sessionLabel && (
              <span>Session ID: <span className="font-mono font-medium text-slate-600 dark:text-slate-300">{sessionLabel}</span></span>
            )}
          </div>

          {/* Clinical Decision Summary */}
          <Card className="border-teal-100 bg-teal-50/30 dark:border-teal-800 dark:bg-teal-900/20">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-4">
                <SafetyBadge
                  safe={isSafe}
                  size="lg"
                  reason={response.safety.reason}
                />
              </div>
              <div className="flex-1 min-w-0 max-w-xl">
                <p className="text-sm text-slate-700 dark:text-slate-300">
                  <span className="font-medium">{request.dose_mg} mg</span>
                  <span className="mx-1.5">·</span>
                  <span className="capitalize">{request.route}</span>
                  <span className="mx-1.5">·</span>
                  <span className={isSafe ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400"}>
                    {response.safety.reason}
                  </span>
                </p>
              </div>
              <div className="shrink-0 text-right">
                <span className={`inline-block rounded-lg px-3 py-1.5 text-sm font-semibold ${isSafe ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/50 dark:text-emerald-200" : "bg-amber-100 text-amber-800 dark:bg-amber-900/50 dark:text-amber-200"}`}>
                  {isSafe ? "Proceed" : "Adjust dose"}
                </span>
              </div>
            </div>
          </Card>

          {/* ——— Risk ——— */}
          {(!isSafe || (showPopulation && popData)) && (
            <div className="space-y-3">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Risk
              </h2>
              <Card className="flex flex-wrap items-center gap-4">
                {showPopulation && popData && (
                  <div className="flex items-center gap-4 text-sm">
                    <span className="text-slate-500 dark:text-slate-400">Probability unsafe:</span>
                    <span className="font-semibold text-red-600 dark:text-red-400">
                      {(popData.population_risk.p_unsafe * 100).toFixed(1)}%
                    </span>
                  </div>
                )}
                {!isSafe && (
                  <p className="text-sm text-amber-700 dark:text-amber-300">
                    {response.safety.reason} Consider dose reduction or split dosing.
                  </p>
                )}
              </Card>
            </div>
          )}

          {/* ——— Exposure ——— */}
          <div className="space-y-3">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              Exposure
            </h2>
            {/* PK curve */}
            <Card>
              <h3 className="mb-4 text-base font-semibold text-slate-900 dark:text-white">
                Concentration–Time Profile
                {showPopulation && popData && (
                  <span className="ml-2 text-sm font-normal text-indigo-500 dark:text-indigo-400">
                    with 90% population band
                  </span>
                )}
                {showTissues && tissueData?.tissues && (
                  <span className="ml-2 text-sm font-normal" style={{ color: TISSUE_COLORS[selectedTissue] }}>
                    + {selectedTissue} tissue (PBPK)
                  </span>
                )}
              </h3>
              <CurveChart data={chartData} series={series} band={bandConfig} />
            </Card>


          {/* PK Metrics grid */}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {METRIC_MAP.map(({ key, label, unit, description, icon }) => (
              <MetricCard
                key={key}
                label={label}
                value={response.pk_metrics[key as keyof typeof response.pk_metrics]}
                unit={unit}
                description={description}
                icon={icon}
              />
            ))}
          </div>

          {/* Toggle cards row */}
          <div className="grid gap-4 lg:grid-cols-2">
            {/* Population toggle */}
            <Card className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-50 dark:bg-indigo-900/30">
                  <Users className="h-5 w-5 text-indigo-500 dark:text-indigo-400" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-900 dark:text-white">Population Uncertainty</p>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    90% confidence bands from IIV
                  </p>
                </div>
              </div>
              <button
                onClick={handlePopulationToggle}
                disabled={popLoading}
                className="flex items-center gap-2 text-sm font-medium text-slate-700 transition-colors hover:text-indigo-600 dark:text-slate-300 dark:hover:text-indigo-400"
              >
                {popLoading ? (
                  <Loader2 className="h-5 w-5 animate-spin text-indigo-500" />
                ) : showPopulation ? (
                  <ToggleRight className="h-7 w-7 text-indigo-600 dark:text-indigo-400" />
                ) : (
                  <ToggleLeft className="h-7 w-7 text-slate-400 dark:text-slate-600" />
                )}
                {popLoading ? "Simulating..." : showPopulation ? "On" : "Off"}
              </button>
            </Card>

            {/* Tissue toggle */}
            <Card className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-purple-50 dark:bg-purple-900/30">
                  <FlaskConical className="h-5 w-5 text-purple-500 dark:text-purple-400" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-900 dark:text-white">
                    Tissue Concentrations
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    PBPK organ distribution
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                {showTissues && (
                  <select
                    value={selectedTissue}
                    onChange={(e) => setSelectedTissue(e.target.value)}
                    className="rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-700 dark:text-white"
                  >
                    {TISSUE_OPTIONS.map((t) => (
                      <option key={t} value={t}>
                        {t.charAt(0).toUpperCase() + t.slice(1)}
                      </option>
                    ))}
                  </select>
                )}
                <button
                  onClick={handleTissueToggle}
                  disabled={tissueLoading}
                  className="flex items-center gap-2 text-sm font-medium text-slate-700 transition-colors hover:text-purple-600 dark:text-slate-300 dark:hover:text-purple-400"
                >
                  {tissueLoading ? (
                    <Loader2 className="h-5 w-5 animate-spin text-purple-500" />
                  ) : showTissues ? (
                    <ToggleRight className="h-7 w-7 text-purple-600 dark:text-purple-400" />
                  ) : (
                    <ToggleLeft className="h-7 w-7 text-slate-400 dark:text-slate-600" />
                  )}
                  {tissueLoading ? "Loading..." : showTissues ? "On" : "Off"}
                </button>
              </div>
            </Card>
          </div>

          {/* Population statistics cards */}
          {showPopulation && popData && (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <Card className="border-indigo-100 bg-indigo-50/30 dark:border-indigo-800 dark:bg-indigo-900/20">
                <p className="text-xs font-semibold uppercase tracking-wider text-indigo-400 dark:text-indigo-500">
                  Probability Safe
                </p>
                <p className="mt-1 text-2xl font-bold text-indigo-700 dark:text-indigo-300">
                  {(popData.population_risk.p_safe * 100).toFixed(1)}%
                </p>
              </Card>
              <Card className="border-red-100 bg-red-50/30 dark:border-red-800 dark:bg-red-900/20">
                <p className="text-xs font-semibold uppercase tracking-wider text-red-400 dark:text-red-500">
                  Probability Unsafe
                </p>
                <p className="mt-1 text-2xl font-bold text-red-600 dark:text-red-300">
                  {(popData.population_risk.p_unsafe * 100).toFixed(1)}%
                </p>
              </Card>
              <Card>
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500">
                  Cmax Range (5th-95th)
                </p>
                <p className="mt-1 text-lg font-bold text-slate-800 dark:text-white">
                  {popData.metrics_dist.cmax.p05.toFixed(0)} - {popData.metrics_dist.cmax.p95.toFixed(0)}
                  <span className="ml-1 text-xs font-normal text-slate-400 dark:text-slate-500">ng/mL</span>
                </p>
              </Card>
              <Card>
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500">
                  AUC Range (5th-95th)
                </p>
                <p className="mt-1 text-lg font-bold text-slate-800 dark:text-white">
                  {popData.metrics_dist.auc.p05.toFixed(0)} - {popData.metrics_dist.auc.p95.toFixed(0)}
                  <span className="ml-1 text-xs font-normal text-slate-400 dark:text-slate-500">ng*h/mL</span>
                </p>
              </Card>
            </div>
          )}

          {/* PBPK params card */}
          {showTissues && tissueData && (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <Card className="border-purple-100 bg-purple-50/30 dark:border-purple-800 dark:bg-purple-900/20">
                <p className="text-xs font-semibold uppercase tracking-wider text-purple-400 dark:text-purple-500">
                  CL Hepatic
                </p>
                <p className="mt-1 text-lg font-bold text-purple-700 dark:text-purple-300">
                  {tissueData.params.CL_hep_l_h?.toFixed(3)}
                  <span className="ml-1 text-xs font-normal text-purple-400 dark:text-purple-500">L/h</span>
                </p>
              </Card>
              <Card className="border-purple-100 bg-purple-50/30 dark:border-purple-800 dark:bg-purple-900/20">
                <p className="text-xs font-semibold uppercase tracking-wider text-purple-400 dark:text-purple-500">
                  CL Renal
                </p>
                <p className="mt-1 text-lg font-bold text-purple-700 dark:text-purple-300">
                  {tissueData.params.CL_ren_l_h?.toFixed(3)}
                  <span className="ml-1 text-xs font-normal text-purple-400 dark:text-purple-500">L/h</span>
                </p>
              </Card>
              <Card>
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500">
                  Cardiac Output
                </p>
                <p className="mt-1 text-lg font-bold text-slate-800 dark:text-white">
                  {tissueData.physiology?.Q_co_l_h
                    ? (tissueData.physiology.Q_co_l_h as number).toFixed(1)
                    : "--"}
                  <span className="ml-1 text-xs font-normal text-slate-400 dark:text-slate-500">L/h</span>
                </p>
              </Card>
              <Card>
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500">
                  V Central
                </p>
                <p className="mt-1 text-lg font-bold text-slate-800 dark:text-white">
                  {tissueData.params.V_central_l?.toFixed(2)}
                  <span className="ml-1 text-xs font-normal text-slate-400 dark:text-slate-500">L</span>
                </p>
              </Card>
            </div>
          )}

          </div>
          {/* End Exposure */}

          {/* ——— Action ——— */}
          <div className="space-y-3">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              Action
            </h2>
            <Card className="flex flex-wrap items-center justify-between gap-4">
              <div className="min-w-0">
                <p className="text-sm text-slate-600 dark:text-slate-300">
                  {isSafe
                    ? "Prediction is within acceptable range. You may proceed or review alternative regimens."
                    : "Dose adjustment recommended. View dosing strategies and download a full report."}
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-3 shrink-0">
                {onGoToRecommendations && (
                  <Button onClick={onGoToRecommendations}>
                    Get Recommendations
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                )}
                <Button
                  variant="secondary"
                  onClick={handleDownloadReport}
                  disabled={reportDownloading}
                >
                  {reportDownloading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Download className="h-4 w-4" />
                  )}
                  {reportDownloading ? "Generating..." : "Download Report"}
                </Button>
              </div>
            </Card>
          </div>
        </div>
      )}

      {!response && !loading && !error && (
        <Card className="py-16 text-center">
          <p className="text-slate-400 dark:text-slate-500">
            Enter parameters above and run a prediction to see results.
          </p>
        </Card>
      )}
    </div>
  );
}
