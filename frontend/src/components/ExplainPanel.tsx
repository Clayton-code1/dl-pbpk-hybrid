"use client";

import { useState, useCallback, useEffect } from "react";
import { usePrediction } from "@/context/PredictionContext";
import { useTheme } from "@/context/ThemeContext";
import {
  fetchExplainV2,
  fetchGraphAttribution,
  type ExplainV2Response,
  type GraphAttributionResponse,
} from "@/lib/api";
import Card from "./Card";
import Button from "./Buttons";
import Toast from "./Toast";
import {
  Brain,
  Layers,
  BarChart3,
  RefreshCw,
  ToggleLeft,
  ToggleRight,
  Loader2,
  FileText,
  AlertTriangle,
  Users,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

const EXPLANATIONS = [
  {
    icon: Layers,
    title: "Model Architecture",
    body: "The hybrid model combines a physics-informed 1-compartment ODE backbone with a Graph Neural Network (GNN) that encodes molecular structure to predict patient-specific PK parameters (CL, V, ka).",
  },
  {
    icon: Brain,
    title: "SHAP Feature Attribution",
    body: "SHAP (SHapley Additive exPlanations) quantifies how each input feature pushes the risk score above or below its baseline. Red bars increase risk; green bars decrease it.",
  },
  {
    icon: BarChart3,
    title: "Mechanistic Sensitivity",
    body: "Local sensitivity analysis perturbs each PK parameter (CL, V, ka) by +/-5% and measures the resulting change in Cmax and AUC, revealing which mechanistic lever has the greatest pharmacological impact.",
  },
];

export default function ExplainPanel() {
  const { request, response, continualLearning, modelUpdated, toggleContinualLearning, markModelUpdated } =
    usePrediction();
  const { dark } = useTheme();
  const [toastVisible, setToastVisible] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [explainData, setExplainData] = useState<ExplainV2Response | null>(null);
  const [sensMode, setSensMode] = useState<"cmax" | "auc">("cmax");
  const [graphAttrData, setGraphAttrData] = useState<GraphAttributionResponse | null>(null);
  const [graphAttrLoading, setGraphAttrLoading] = useState(false);
  const [graphAttrNotPanel, setGraphAttrNotPanel] = useState(false);
  const [graphAttrError, setGraphAttrError] = useState<string | null>(null);

  useEffect(() => {
    if (!request || !response) {
      setExplainData(null);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchExplainV2({
      patient: {
        weight_kg: request.weight_kg,
        compound_name: request.compound_name,
      },
      regimen: [{ time_hr: 0, dose_mg: request.dose_mg, route: request.route }],
      horizon_hr: 48,
    })
      .then((data) => {
        if (!cancelled) setExplainData(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message ?? "Failed to load explanations");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, [request, response]);

  useEffect(() => {
    if (!request || !response) {
      setGraphAttrData(null);
      setGraphAttrNotPanel(false);
      setGraphAttrError(null);
      return;
    }

    const drug = request.compound_name.trim().toLowerCase();
    let cancelled = false;
    setGraphAttrLoading(true);
    setGraphAttrData(null);
    setGraphAttrNotPanel(false);
    setGraphAttrError(null);

    fetchGraphAttribution(drug)
      .then((data) => {
        if (!cancelled) setGraphAttrData(data);
      })
      .catch((err: Error) => {
        if (!cancelled) {
          if (err.message === "not_found") {
            setGraphAttrNotPanel(true);
          } else {
            setGraphAttrError("Molecular attribution could not be loaded.");
          }
        }
      })
      .finally(() => {
        if (!cancelled) setGraphAttrLoading(false);
      });

    return () => { cancelled = true; };
  }, [request, response]);

  const handleUpdateModel = useCallback(() => {
    markModelUpdated();
    localStorage.setItem("dl-pbpk-model-updated", "true");
    setToastVisible(true);
  }, [markModelUpdated]);

  const handleCloseToast = useCallback(() => setToastVisible(false), []);

  const shapChartData = explainData?.shap?.features.map((f, i) => ({
    feature: f,
    value: explainData.shap.values[i],
    fill: explainData.shap.values[i] > 0 ? "#ef4444" : "#10b981",
  })) ?? [];

  const sensChartData = explainData?.sensitivity?.parameters.map((p, i) => ({
    parameter: p,
    delta_cmax: explainData.sensitivity.delta_cmax_pct[i],
    delta_auc: explainData.sensitivity.delta_auc_pct[i],
  })) ?? [];

  const graphAttrChartData = graphAttrData?.attributions.slice(0, 10).map((a) => ({
    label: `${a.atom_symbol}${a.atom_index}`,
    importance: a.importance_score,
  })) ?? [];

  const gridColor = dark ? "#334155" : "#e5e7eb";
  const tickColor = dark ? "#94a3b8" : "#6b7280";
  const tooltipBg = dark ? "#1e293b" : "#ffffff";
  const tooltipBorder = dark ? "#475569" : "#e5e7eb";
  const tooltipTextColor = dark ? "#e2e8f0" : "#1f2937";

  return (
    <div className="space-y-6 fade-in">
      <Toast
        message="Model updated (demo). New predictions will use updated parameters."
        type="success"
        visible={toastVisible}
        onClose={handleCloseToast}
      />

      {/* Info cards in a 3-column grid */}
      <div className="grid gap-4 lg:grid-cols-3">
        {EXPLANATIONS.map(({ icon: Icon, title, body }) => (
          <Card key={title} className="flex gap-4">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-teal-50 dark:bg-teal-900/30">
              <Icon className="h-5 w-5 text-teal-600 dark:text-teal-400" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-gray-900 dark:text-white">{title}</h3>
              <p className="mt-1 text-xs leading-relaxed text-gray-500 dark:text-slate-400">{body}</p>
            </div>
          </Card>
        ))}
      </div>

      {loading && (
        <Card className="flex items-center justify-center gap-3 py-12">
          <Loader2 className="h-5 w-5 animate-spin text-teal-600 dark:text-teal-400" />
          <span className="text-sm text-gray-500 dark:text-slate-400">Computing SHAP values and sensitivity...</span>
        </Card>
      )}

      {error && (
        <Card className="flex items-center gap-3 border-amber-200 bg-amber-50 py-6 dark:border-amber-800 dark:bg-amber-900/20">
          <AlertTriangle className="h-5 w-5 text-amber-600 dark:text-amber-400" />
          <span className="text-sm text-amber-800 dark:text-amber-200">{error}</span>
        </Card>
      )}

      {!response && !loading && (
        <Card className="py-12 text-center">
          <p className="text-gray-400 dark:text-slate-500">Run a prediction first to see detailed explanations.</p>
        </Card>
      )}

      {/* SHAP and Sensitivity side-by-side on large screens */}
      {explainData && (shapChartData.length > 0 || sensChartData.length > 0) && (
        <div className="grid gap-6 xl:grid-cols-2">
          {/* SHAP bar chart */}
          {shapChartData.length > 0 && (
            <Card className="slide-up">
              <div className="mb-1 flex items-center justify-between">
                <h3 className="text-base font-semibold text-gray-900 dark:text-white">
                  Feature Attribution (SHAP)
                </h3>
                <span className="rounded-full bg-teal-50 px-3 py-1 text-xs font-medium text-teal-700 dark:bg-teal-900/30 dark:text-teal-300">
                  Target: {explainData.shap.target}
                </span>
              </div>
              <p className="mb-4 text-xs text-gray-400 dark:text-slate-500">
                Base value: {explainData.shap.base.toFixed(4)} | Bars show how each feature shifts risk
              </p>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={shapChartData} layout="vertical" margin={{ left: 10, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                  <XAxis type="number" tick={{ fontSize: 12, fill: tickColor }} stroke={gridColor} />
                  <YAxis dataKey="feature" type="category" tick={{ fontSize: 12, fill: tickColor }} width={120} stroke={gridColor} />
                  <Tooltip
                    contentStyle={{ borderRadius: "12px", fontSize: "13px", border: `1px solid ${tooltipBorder}`, backgroundColor: tooltipBg, color: tooltipTextColor }}
                    formatter={(v: number) => v.toFixed(4)}
                  />
                  <Bar dataKey="value" name="SHAP value" radius={[0, 6, 6, 0]} barSize={22}>
                    {shapChartData.map((entry) => (
                      <Cell key={entry.feature} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}

          {/* Sensitivity tornado */}
          {sensChartData.length > 0 && (
            <Card className="slide-up">
              <div className="mb-1 flex items-center justify-between">
                <h3 className="text-base font-semibold text-gray-900 dark:text-white">
                  PK Sensitivity Analysis
                </h3>
                <div className="flex gap-1">
                  <button
                    onClick={() => setSensMode("cmax")}
                    className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                      sensMode === "cmax"
                        ? "bg-teal-600 text-white dark:bg-teal-500"
                        : "bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-slate-700 dark:text-slate-300 dark:hover:bg-slate-600"
                    }`}
                  >
                    Cmax
                  </button>
                  <button
                    onClick={() => setSensMode("auc")}
                    className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                      sensMode === "auc"
                        ? "bg-amber-500 text-white dark:bg-amber-600"
                        : "bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-slate-700 dark:text-slate-300 dark:hover:bg-slate-600"
                    }`}
                  >
                    AUC
                  </button>
                </div>
              </div>
              <p className="mb-4 text-xs text-gray-400 dark:text-slate-500">
                % change in {sensMode === "cmax" ? "Cmax" : "AUC"} per 1% perturbation of each PK parameter
              </p>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={sensChartData} layout="vertical" margin={{ left: 10, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                  <XAxis type="number" tick={{ fontSize: 12, fill: tickColor }} stroke={gridColor} />
                  <YAxis dataKey="parameter" type="category" tick={{ fontSize: 12, fill: tickColor }} width={50} stroke={gridColor} />
                  <Tooltip
                    contentStyle={{ borderRadius: "12px", fontSize: "13px", border: `1px solid ${tooltipBorder}`, backgroundColor: tooltipBg, color: tooltipTextColor }}
                    formatter={(v: number) => `${v.toFixed(4)}%/%`}
                  />
                  {sensMode === "cmax" ? (
                    <Bar dataKey="delta_cmax" name="Cmax sensitivity" fill="#14b8a6" radius={[0, 6, 6, 0]} barSize={20} />
                  ) : (
                    <Bar dataKey="delta_auc" name="AUC sensitivity" fill="#f59e0b" radius={[0, 6, 6, 0]} barSize={20} />
                  )}
                </BarChart>
              </ResponsiveContainer>
              {explainData.sensitivity.baseline_values.length > 0 && (
                <div className="mt-3 flex gap-4 border-t border-gray-100 pt-3 dark:border-slate-700">
                  {explainData.sensitivity.parameters.map((p, i) => (
                    <div key={p} className="text-center">
                      <p className="text-xs font-medium text-gray-400 dark:text-slate-500">{p}</p>
                      <p className="text-sm font-semibold text-gray-800 dark:text-white">
                        {explainData.sensitivity.baseline_values[i].toFixed(3)}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </Card>
          )}
        </div>
      )}

      {/* Molecular attribution */}
      {request && response && (
        <Card className="slide-up">
          <div className="mb-1">
            <h3 className="text-base font-semibold text-gray-900 dark:text-white">
              Molecular attribution (per-atom importance)
            </h3>
            <p className="mt-1 text-xs text-gray-400 dark:text-slate-500">
              Offline analysis: atom-level contribution to predicted exposure, pre-computed for the drug panel.
            </p>
          </div>
          {graphAttrLoading && (
            <p className="mt-4 text-xs text-gray-400 dark:text-slate-500">Loading molecular attribution…</p>
          )}
          {!graphAttrLoading && graphAttrNotPanel && (
            <p className="mt-4 text-xs text-gray-400 dark:text-slate-500">
              Molecular attribution is available for the six panel drugs (acetaminophen, caffeine, digoxin, midazolam, theophylline, warfarin).
            </p>
          )}
          {!graphAttrLoading && graphAttrError && (
            <p className="mt-4 text-xs text-gray-400 dark:text-slate-500">{graphAttrError}</p>
          )}
          {!graphAttrLoading && graphAttrChartData.length > 0 && (
            <div className="mt-3">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={graphAttrChartData} layout="vertical" margin={{ left: 10, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                  <XAxis type="number" tick={{ fontSize: 12, fill: tickColor }} stroke={gridColor} />
                  <YAxis dataKey="label" type="category" tick={{ fontSize: 12, fill: tickColor }} width={40} stroke={gridColor} />
                  <Tooltip
                    contentStyle={{ borderRadius: "12px", fontSize: "13px", border: `1px solid ${tooltipBorder}`, backgroundColor: tooltipBg, color: tooltipTextColor }}
                    formatter={(v: number) => v.toFixed(6)}
                  />
                  <Bar dataKey="importance" name="Atom importance" fill="#14b8a6" radius={[0, 6, 6, 0]} barSize={22} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </Card>
      )}

      {/* Narrative */}
      {explainData?.narrative && (
        <Card className="slide-up">
          <div className="mb-3 flex items-center gap-2">
            <FileText className="h-5 w-5 text-teal-600 dark:text-teal-400" />
            <h3 className="text-base font-semibold text-gray-900 dark:text-white">Narrative Explanation</h3>
          </div>
          <p className="text-sm leading-relaxed text-gray-700 dark:text-slate-300">
            {explainData.narrative.summary}
          </p>
          {explainData.narrative.key_drivers.length > 0 && (
            <div className="mt-4">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-gray-400 dark:text-slate-500">
                Key Drivers
              </p>
              <div className="flex flex-wrap gap-2">
                {explainData.narrative.key_drivers.map((d) => (
                  <span
                    key={d.feature}
                    className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium ${
                      d.direction === "risk+"
                        ? "bg-red-50 text-red-700 ring-1 ring-red-200 dark:bg-red-900/30 dark:text-red-300 dark:ring-red-700"
                        : "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200 dark:bg-emerald-900/30 dark:text-emerald-300 dark:ring-emerald-700"
                    }`}
                  >
                    {d.feature}
                    <span className="font-bold">{d.shap_value > 0 ? "+" : ""}{d.shap_value.toFixed(4)}</span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </Card>
      )}

      {/* Population Uncertainty Drivers */}
      {explainData && explainData.sensitivity?.baseline_values?.length > 0 && (
        <Card className="slide-up">
          <div className="mb-3 flex items-center gap-2">
            <Users className="h-5 w-5 text-indigo-500 dark:text-indigo-400" />
            <h3 className="text-base font-semibold text-gray-900 dark:text-white">Population Uncertainty Drivers</h3>
          </div>
          <p className="mb-3 text-sm text-gray-500 dark:text-slate-400">
            Inter-individual variability (IIV) is modelled as log-normal random effects on PK parameters.
            The omega values below represent the standard deviation of the log-normal distribution.
          </p>
          <div className="grid grid-cols-3 gap-3">
            {[
              { param: "CL", omega: 0.25, desc: "Clearance variability drives AUC spread" },
              { param: "V", omega: 0.20, desc: "Volume variability drives Cmax spread" },
              { param: "ka", omega: 0.30, desc: "Absorption variability drives Tmax spread" },
            ].map((item) => (
              <div key={item.param} className="rounded-xl border border-indigo-100 bg-indigo-50/30 p-3 dark:border-indigo-800 dark:bg-indigo-900/20">
                <p className="text-xs font-semibold text-indigo-600 dark:text-indigo-400">
                  omega_{item.param} = {item.omega}
                </p>
                <p className="mt-1 text-xs text-gray-500 dark:text-slate-400">{item.desc}</p>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Continual Learning Panel */}
      <Card>
        <h3 className="mb-3 text-base font-semibold text-gray-900 dark:text-white">Model Update</h3>
        <p className="mb-4 text-sm text-gray-500 dark:text-slate-400">
          Simulate continual-learning behaviour. When enabled, feedback signals can
          be used to fine-tune model parameters incrementally.
        </p>
        <div className="flex flex-wrap items-center gap-4">
          <button
            onClick={toggleContinualLearning}
            className="flex items-center gap-2 text-sm font-medium text-gray-700 transition-colors hover:text-teal-600 dark:text-slate-300 dark:hover:text-teal-400"
          >
            {continualLearning ? (
              <ToggleRight className="h-7 w-7 text-teal-600 dark:text-teal-400" />
            ) : (
              <ToggleLeft className="h-7 w-7 text-gray-400 dark:text-slate-600" />
            )}
            Enable continual learning (demo)
          </button>
          <Button
            variant="secondary"
            size="sm"
            onClick={handleUpdateModel}
            disabled={!continualLearning}
          >
            <RefreshCw className="h-4 w-4" />
            Update Model with Feedback
          </Button>
          {modelUpdated && (
            <span className="text-xs font-medium text-emerald-600 dark:text-emerald-400">
              Model parameters updated
            </span>
          )}
        </div>
      </Card>
    </div>
  );
}
