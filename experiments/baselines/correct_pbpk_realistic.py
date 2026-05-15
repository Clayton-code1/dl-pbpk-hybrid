"""Recompute PBPK-only (A1) baseline with clinical population PK uncertainty.

Does not retrain hybrids. Writes corrected benchmark, prediction cache,
ablation summary, and runs significance tests on the corrected cache.

    python -m experiments.baselines.correct_pbpk_realistic
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.config import (  # noqa: E402
    DRUGS,
    PLOTS_DIR,
    RESULTS_DIR,
    SEED,
    ensure_dirs,
    get_logger,
    seed_everything,
)
from experiments.phase2.utils import (  # noqa: E402
    CLINICAL_PBPK_POPULATION_LOG_SD,
    pbpk_population_estimate_curves,
)
from experiments.statistics.significance_tests import run as run_significance_tests  # noqa: E402
from experiments.training.multidrug_utils import (  # noqa: E402
    cmax_auc_errors,
    load_drug_dataset,
    regression_metrics,
    split_rng_seed_for_drug,
)

LOGGER = get_logger("phase2.correct_pbpk", "phase2_correct_pbpk.log")

BENCH_ORIG = RESULTS_DIR / "phase2_benchmark_metrics.csv"
BENCH_OUT = RESULTS_DIR / "phase2_benchmark_metrics_corrected.csv"
CACHE_ORIG = RESULTS_DIR / "phase2_prediction_cache.pkl"
CACHE_OUT = RESULTS_DIR / "phase2_prediction_cache_corrected.pkl"
ABLATION_BY = RESULTS_DIR / "phase2_ablation_by_drug.csv"
ABLATION_SUM_OUT = RESULTS_DIR / "phase2_ablation_summary_corrected.csv"
STATS_OUT = RESULTS_DIR / "phase2_statistical_tests_corrected.csv"


def _drug_pbpk_rng(drug: str) -> np.random.Generator:
    """Reproducible RNG for population PK draws (orthogonal to split seed)."""
    s = split_rng_seed_for_drug(drug, base_seed=SEED + 91_017)
    return np.random.default_rng(s)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--population-log-sd",
        type=float,
        default=CLINICAL_PBPK_POPULATION_LOG_SD,
        help="Log-scale SD for shared exp(sd*z) multiplier on CL and Vd",
    )
    args = parser.parse_args()

    ensure_dirs()
    seed_everything(SEED)

    if not BENCH_ORIG.exists():
        raise FileNotFoundError(BENCH_ORIG)
    if not CACHE_ORIG.exists():
        raise FileNotFoundError(CACHE_ORIG)
    if not ABLATION_BY.exists():
        raise FileNotFoundError(ABLATION_BY)

    bench = pd.read_csv(BENCH_ORIG)
    with open(CACHE_ORIG, "rb") as f:
        cache: dict[str, Any] = pickle.load(f)

    new_pbpk_rows: list[dict[str, Any]] = []

    for drug in DRUGS:
        _tr, _va, test_r, _ = load_drug_dataset(drug)
        times = test_r[0].times_hr.cpu().numpy()
        y_true = np.stack([r.concentration.numpy() for r in test_r])
        rng = _drug_pbpk_rng(drug)
        pred = pbpk_population_estimate_curves(
            test_r, drug, rng, population_log_sd=args.population_log_sd,
        )

        flat = regression_metrics(pred.ravel(), y_true.ravel())
        summ = cmax_auc_errors(pred, y_true, times)
        obs_mean = float(y_true.mean())
        rmse_pct = flat["RMSE"] / (obs_mean + 1e-12) * 100.0 if obs_mean > 0 else float("nan")

        row = {
            "drug": drug,
            "model": "PBPK-only",
            "RMSE": flat["RMSE"],
            "RMSE_pct_of_mean": rmse_pct,
            "MAE": flat["MAE"],
            "MAPE": flat["MAPE"],
            "R2": flat["R2"],
            "Cmax_pct_err": summ["Cmax_pct_err"],
            "AUC_pct_err": summ["AUC_pct_err"],
            "n_test_patients": len(test_r),
        }
        new_pbpk_rows.append(row)

        model_order = list(cache[drug]["per_patient_rmse"].keys())
        if "PBPK-only" not in model_order:
            raise KeyError(f"cache[{drug}] missing PBPK-only")

        pp = np.sqrt(np.mean((pred - y_true) ** 2, axis=1))
        cache[drug]["per_patient_rmse"]["PBPK-only"] = pp
        cache[drug]["abs_err_flat"]["PBPK-only"] = np.abs((pred - y_true).ravel())
        cache[drug]["sq_err_flat"]["PBPK-only"] = ((pred - y_true) ** 2).ravel()

    bench_corr = bench.copy()
    for row in new_pbpk_rows:
        m = (bench_corr["drug"] == row["drug"]) & (bench_corr["model"] == "PBPK-only")
        for k, v in row.items():
            bench_corr.loc[m, k] = v

    bench_corr.to_csv(BENCH_OUT, index=False)
    LOGGER.info("Wrote %s", BENCH_OUT)

    with open(CACHE_OUT, "wb") as f:
        pickle.dump(cache, f)
    LOGGER.info("Wrote %s", CACHE_OUT)

    # ---- ablation summary (A1 from corrected bench; A2,A5 from uncorrected bench; A3,A4 from file)
    ab_by = pd.read_csv(ABLATION_BY)
    a34 = ab_by[ab_by["variant"].str.startswith("A3_") | ab_by["variant"].str.startswith("A4_")].copy()

    a1_rows = []
    a2_rows = []
    a5_rows = []
    for drug in DRUGS:
        r1 = bench_corr[(bench_corr["drug"] == drug) & (bench_corr["model"] == "PBPK-only")].iloc[0]
        a1_rows.append({"variant": "A1_PBPK_only", "drug": drug, "R2": r1["R2"], "RMSE": r1["RMSE"]})
        r2o = bench[(bench["drug"] == drug) & (bench["model"] == "VanillaGNN")].iloc[0]
        a2_rows.append({"variant": "A2_GNN_only", "drug": drug, "R2": r2o["R2"], "RMSE": r2o["RMSE"]})
        r5o = bench[(bench["drug"] == drug) & (bench["model"] == "DL-PBPK")].iloc[0]
        a5_rows.append({"variant": "A5_Full_DLPBPK", "drug": drug, "R2": r5o["R2"], "RMSE": r5o["RMSE"]})

    ab_corrected = pd.concat(
        [pd.DataFrame(a1_rows), pd.DataFrame(a2_rows), pd.DataFrame(a5_rows), a34],
        ignore_index=True,
    )
    mean_by = ab_corrected.groupby("variant", sort=False)[["R2", "RMSE"]].mean().reset_index()
    mean_by = mean_by.rename(columns={"R2": "mean_R2_6drugs", "RMSE": "mean_RMSE_6drugs"})
    vorder = [
        "A1_PBPK_only",
        "A2_GNN_only",
        "A3_hybrid_no_transfer",
        "A4_hybrid_encoder_frozen",
        "A5_Full_DLPBPK",
    ]
    mean_by = mean_by.set_index("variant").reindex(vorder).reset_index()
    mean_by.to_csv(ABLATION_SUM_OUT, index=False)
    LOGGER.info("Wrote %s", ABLATION_SUM_OUT)

    heatmap_path = PLOTS_DIR / "significance_heatmap_corrected.png"
    run_significance_tests(CACHE_OUT, STATS_OUT, heatmap_path)
    LOGGER.info("Wrote %s", STATS_OUT)

    return 0


if __name__ == "__main__":
    sys.exit(main())
