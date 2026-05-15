"""Phase 2.3 — paired significance tests vs DL-PBPK on held-out test predictions.

Requires ``experiments/results/phase2_prediction_cache.pkl`` from
``experiments.baselines.train_baselines``.

    python -m experiments.statistics.significance_tests
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy import stats

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.config import (  # noqa: E402
    DRUGS,
    PLOTS_DIR,
    RESULTS_DIR,
    ensure_dirs,
    get_logger,
)

LOGGER = get_logger("phase2.significance", "phase2_significance.log")

PRED_CACHE_PATH = RESULTS_DIR / "phase2_prediction_cache.pkl"
BASELINES = ["PBPK-only", "MLP", "RandomForest", "XGBoost", "VanillaGNN"]


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def run(
    cache_path: Path,
    out_csv: Path,
    heatmap_path: Path,
) -> None:
    ensure_dirs()
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    with open(cache_path, "rb") as f:
        cache: dict[str, Any] = pickle.load(f)

    rows: list[dict[str, Any]] = []
    p_matrix: np.ndarray = np.full((len(DRUGS), len(BASELINES)), np.nan)

    for di, drug in enumerate(DRUGS):
        block = cache[drug]
        pp = block["per_patient_rmse"]
        sq_base = block["sq_err_flat"]
        ref_rmse = pp["DL-PBPK"]
        ref_sq = sq_base["DL-PBPK"]
        for bi, base in enumerate(BASELINES):
            b_rmse = pp[base]
            t_rmse, p_rmse = stats.ttest_rel(b_rmse, ref_rmse)
            n = len(b_rmse)
            mean_diff = float((b_rmse - ref_rmse).mean())
            std_diff = float((b_rmse - ref_rmse).std(ddof=1))
            sem = std_diff / math.sqrt(n) if n > 0 else float("nan")
            tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else float("nan")
            ci_lo = mean_diff - tcrit * sem if n > 1 else float("nan")
            ci_hi = mean_diff + tcrit * sem if n > 1 else float("nan")

            b_sq = sq_base[base]
            t_sq, p_sq = stats.ttest_rel(b_sq, ref_sq)

            rows.append(
                {
                    "drug": drug,
                    "baseline": base,
                    "test": "per_patient_RMSE",
                    "statistic": float(t_rmse),
                    "p_value": float(p_rmse),
                    "mean_RMSE_diff_baseline_minus_DLPBPK": mean_diff,
                    "CI95_low_RMSE_diff": ci_lo,
                    "CI95_high_RMSE_diff": ci_hi,
                    "significance": _stars(float(p_rmse)),
                }
            )
            rows.append(
                {
                    "drug": drug,
                    "baseline": base,
                    "test": "per_timepoint_squared_error",
                    "statistic": float(t_sq),
                    "p_value": float(p_sq),
                    "mean_RMSE_diff_baseline_minus_DLPBPK": float("nan"),
                    "CI95_low_RMSE_diff": float("nan"),
                    "CI95_high_RMSE_diff": float("nan"),
                    "significance": _stars(float(p_sq)),
                }
            )
            p_matrix[di, bi] = float(p_rmse)

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    LOGGER.info("Wrote %s", out_csv)

    import matplotlib.pyplot as plt
    import seaborn as sns

    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        -np.log10(np.clip(p_matrix, 1e-300, None)),
        xticklabels=BASELINES,
        yticklabels=DRUGS,
        cmap="viridis_r",
        ax=ax,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": r"$-\log_{10}(p)$ paired RMSE test"},
    )
    ax.set_title("Phase 2.3 — paired t-test: baseline vs DL-PBPK (per-patient RMSE)")
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=150)
    fig.savefig(heatmap_path.with_suffix(".pdf"))
    plt.close(fig)
    LOGGER.info("Saved %s", heatmap_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-cache",
        type=Path,
        default=PRED_CACHE_PATH,
        help="Pickle dict from train_baselines (or corrected cache)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=RESULTS_DIR / "phase2_statistical_tests.csv",
    )
    parser.add_argument(
        "--heatmap",
        type=Path,
        default=PLOTS_DIR / "significance_heatmap.png",
    )
    args = parser.parse_args()
    run(args.prediction_cache, args.output_csv, args.heatmap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
