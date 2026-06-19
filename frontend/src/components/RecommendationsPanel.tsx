"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { usePrediction } from "@/context/PredictionContext";
import {
  fetchRecommendations,
  type RecommendResponse,
  type StrategyResult,
  type SafeRecommendation,
} from "@/lib/api";
import Card from "./Card";
import Button from "./Buttons";
import SafetyBadge from "./SafetyBadge";
import MetricCard from "./MetricCard";
import CurveChart, { type ChartSeries } from "./CurveChart";
import {
  FileDown,
  Lightbulb,
  CheckCircle,
  Search,
  AlertTriangle,
  Loader2,
  ShieldCheck,
} from "lucide-react";

export default function RecommendationsPanel() {
  const router = useRouter();
  const { response, request, isSafe, drug } = usePrediction();
  const [recData, setRecData] = useState<RecommendResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prevRequestKey, setPrevRequestKey] = useState<string | null>(null);

  const requestKey = request
    ? `${request.compound_name}-${request.dose_mg}-${request.weight_kg}-${request.route}-${drug?.therapeutic_min_mg_L ?? ""}-${drug?.therapeutic_max_mg_L ?? ""}`
    : null;

  useEffect(() => {
    if (!request || !response) return;
    if (requestKey === prevRequestKey && recData) return;

    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchRecommendations({
      patient: { weight_kg: request.weight_kg, compound_name: request.compound_name },
      regimen: [{ time_hr: 0, dose_mg: request.dose_mg, route: request.route }],
      horizon_hr: 48,
      ...(drug ? { drug } : {}),
    })
      .then((data) => {
        if (!cancelled) {
          setRecData(data);
          setPrevRequestKey(requestKey);
        }
      })
      .catch((err) => {
        if (!cancelled) setError(err.message ?? "Recommendation request failed");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, [request, response, requestKey, prevRequestKey, recData]);

  const handleExportReport = useCallback(() => {
    router.push("/reports");
  }, [router]);

  if (!response) {
    return (
      <Card className="py-16 text-center fade-in">
        <Lightbulb className="mx-auto mb-3 h-10 w-10 text-gray-300 dark:text-slate-600" />
        <p className="text-gray-400 dark:text-slate-500">Run a prediction to see recommendations.</p>
      </Card>
    );
  }

  if (loading) {
    return (
      <Card className="py-16 text-center fade-in">
        <Loader2 className="mx-auto mb-3 h-10 w-10 animate-spin text-teal-500" />
        <p className="text-gray-500 dark:text-slate-400">
          Searching for safe dosing regimens...
        </p>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="py-10 text-center fade-in">
        <AlertTriangle className="mx-auto mb-3 h-8 w-8 text-amber-500" />
        <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        <p className="mt-2 text-xs text-gray-400 dark:text-slate-500">
          Falling back to baseline prediction. Try again later.
        </p>
      </Card>
    );
  }

  const baselineChartData = response
    ? response.time_h.map((t, i) => ({ time: t, concentration: response.concentration_ng_ml[i] }))
    : [];

  const safeRec = recData?.safe_recommendation ?? null;
  const strategies = recData?.strategies ?? [];
  const searchSummary = recData?.search_summary ?? "";
  const baselineSafe = recData?.baseline?.safety?.is_safe ?? isSafe;

  return (
    <div className="space-y-6 fade-in">
      {/* Header */}
      <Card className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          {baselineSafe ? (
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-emerald-50 dark:bg-emerald-900/30">
              <CheckCircle className="h-5 w-5 text-emerald-500 dark:text-emerald-400" />
            </div>
          ) : (
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-amber-50 dark:bg-amber-900/30">
              <Lightbulb className="h-5 w-5 text-amber-500 dark:text-amber-400" />
            </div>
          )}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              {baselineSafe ? "No correction needed" : "Dose Adjustment Recommended"}
            </h3>
            <p className="mt-0.5 text-sm text-gray-500 dark:text-slate-400">
              {baselineSafe
                ? "The predicted dose is within safe ranges. Below are optional optimizations."
                : "The predicted dose exceeds the safety threshold. A safe alternative has been searched automatically."}
            </p>
          </div>
        </div>
        <Button variant="secondary" onClick={handleExportReport}>
          <FileDown className="h-4 w-4" />
          Export Report
        </Button>
      </Card>

      {/* Safe Recommendation Card */}
      {safeRec && !baselineSafe && (
        <SafeRecommendationCard
          rec={safeRec}
          baselineChartData={baselineChartData}
        />
      )}

      {/* Search Explanation */}
      {searchSummary && !baselineSafe && (
        <div className="flex items-start gap-2.5 rounded-xl border border-teal-200 bg-teal-50 px-4 py-3 dark:border-teal-800 dark:bg-teal-900/20">
          <Search className="mt-0.5 h-4 w-4 shrink-0 text-teal-600 dark:text-teal-400" />
          <p className="text-sm text-teal-800 dark:text-teal-300">{searchSummary}</p>
        </div>
      )}

      {/* Simulated Alternatives Section */}
      {strategies.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-base font-semibold text-gray-700 dark:text-slate-300">
            {baselineSafe ? "Optional Optimizations" : "Simulated Alternatives"}
          </h3>
          <div className="grid gap-6 xl:grid-cols-3">
            {strategies.map((strategy) => (
              <BackendStrategyCard
                key={strategy.id}
                strategy={strategy}
                baselineChartData={baselineChartData}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}


function SafeRecommendationCard({
  rec,
  baselineChartData,
}: {
  rec: SafeRecommendation;
  baselineChartData: Record<string, number>[];
}) {
  const chartData = rec.time_h.map((t, i) => ({
    time: t,
    Baseline: baselineChartData[i]?.concentration ?? 0,
    "Safe Regimen": rec.concentration_ng_ml[i],
  }));

  const series: ChartSeries[] = [
    { key: "Baseline", label: "Baseline", color: "#ef4444", strokeDash: "6 3" },
    { key: "Safe Regimen", label: "Safe Regimen", color: "#10b981" },
  ];

  return (
    <Card
      className={
        rec.is_safe
          ? "ring-2 ring-emerald-300 dark:ring-emerald-700"
          : "ring-2 ring-amber-300 dark:ring-amber-700"
      }
    >
      <div className="space-y-4">
        {/* Title row */}
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="flex items-start gap-3">
            <div
              className={
                rec.is_safe
                  ? "flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-emerald-50 dark:bg-emerald-900/30"
                  : "flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-amber-50 dark:bg-amber-900/30"
              }
            >
              {rec.is_safe ? (
                <ShieldCheck className="h-5 w-5 text-emerald-500 dark:text-emerald-400" />
              ) : (
                <AlertTriangle className="h-5 w-5 text-amber-500 dark:text-amber-400" />
              )}
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                {rec.is_safe ? "Recommended Safe Regimen" : "Best Available Regimen"}
              </h3>
              <p className="mt-0.5 text-sm text-gray-500 dark:text-slate-400">
                {rec.rationale}
              </p>
            </div>
          </div>
          <SafetyBadge safe={rec.is_safe} size="lg" reason={rec.safety.reason} />
        </div>

        {/* Dose info pills */}
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span className="rounded-lg bg-gray-100 px-3 py-1.5 font-medium text-gray-700 dark:bg-slate-700 dark:text-slate-300">
            {rec.dose_mg.toFixed(1)} mg
          </span>
          <span className="rounded-lg bg-gray-100 px-3 py-1.5 font-medium text-gray-700 dark:bg-slate-700 dark:text-slate-300">
            {rec.route}
          </span>
          <span className="rounded-lg bg-gray-100 px-3 py-1.5 font-medium text-gray-700 dark:bg-slate-700 dark:text-slate-300">
            {rec.frequency}
          </span>
          <span className="rounded-lg bg-teal-100 px-3 py-1.5 font-medium text-teal-800 dark:bg-teal-900/40 dark:text-teal-300">
            Scale: {(rec.scale_factor * 100).toFixed(1)}%
          </span>
        </div>

        {/* Metrics */}
        <div className="grid gap-3 sm:grid-cols-3">
          <MetricCard
            label="Cmax"
            value={rec.pk_metrics.cmax_ng_ml}
            unit="ng/mL"
            delta={rec.delta_cmax_pct}
            className="!p-3"
          />
          <MetricCard
            label="AUC"
            value={rec.pk_metrics.auc_0_inf}
            unit="ng*h/mL"
            delta={rec.delta_auc_pct}
            className="!p-3"
          />
          <MetricCard
            label="t½"
            value={rec.pk_metrics.half_life_h}
            unit="h"
            className="!p-3"
          />
        </div>

        {/* Chart */}
        <CurveChart data={chartData} series={series} height={240} />
      </div>
    </Card>
  );
}


function BackendStrategyCard({
  strategy,
  baselineChartData,
}: {
  strategy: StrategyResult;
  baselineChartData: Record<string, number>[];
}) {
  const chartData = strategy.time_h.map((t, i) => ({
    time: t,
    Baseline: baselineChartData[i]?.concentration ?? 0,
    [strategy.title]: strategy.concentration_ng_ml[i],
  }));

  const series: ChartSeries[] = [
    { key: "Baseline", label: "Baseline", color: "#ef4444", strokeDash: "6 3" },
    { key: strategy.title, label: strategy.title, color: strategy.safety.is_safe ? "#10b981" : "#f59e0b" },
  ];

  const totalDose = strategy.regimen.reduce((s, e) => s + e.dose_mg, 0);

  return (
    <Card className="flex flex-col space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{strategy.title}</h3>
          <p className="mt-0.5 text-sm text-gray-500 dark:text-slate-400">{strategy.description}</p>
        </div>
        <SafetyBadge safe={strategy.safety.is_safe} size="lg" />
      </div>

      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="rounded-lg bg-gray-100 px-3 py-1.5 font-medium text-gray-700 dark:bg-slate-700 dark:text-slate-300">
          {totalDose.toFixed(1)} mg
        </span>
        <span className="rounded-lg bg-gray-100 px-3 py-1.5 font-medium text-gray-700 dark:bg-slate-700 dark:text-slate-300">
          Scale: {(strategy.scale_factor * 100).toFixed(1)}%
        </span>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <MetricCard
          label="Cmax"
          value={strategy.pk_metrics.cmax_ng_ml}
          unit="ng/mL"
          delta={strategy.delta_cmax_pct}
          className="!p-3"
        />
        <MetricCard
          label="AUC"
          value={strategy.pk_metrics.auc_0_inf}
          unit="ng*h/mL"
          delta={strategy.delta_auc_pct}
          className="!p-3"
        />
        <MetricCard
          label="t½"
          value={strategy.pk_metrics.half_life_h}
          unit="h"
          className="!p-3"
        />
      </div>

      <CurveChart data={chartData} series={series} height={220} />
    </Card>
  );
}
