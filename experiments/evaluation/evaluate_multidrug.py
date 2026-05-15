"""Phase 1.6 - per-drug evaluation of the multi-drug hybrid models.

For every drug in ``experiments.config.DRUGS``:

1. Load the held-out test split (same seed as training -> deterministic).
2. Reconstruct :class:`MultiDrugHybridGNNPBPK` from the saved
   ``config.json`` and load the best checkpoint from
   ``artifacts/models/hybrid_gnn_pbpk_{drug}_v1/model.pt``.
3. Compute RMSE, MAE, MAPE, R^2 on per-time-point predictions and
   Cmax/AUC % error per patient on the test set.
4. Generate the diagnostic plots:
       - 2x3 grid of observed-vs-predicted scatter (one panel per drug)
       - 2x3 grid of mean +/- 95% CI PK curves (predicted vs observed)
       - bar chart of per-drug test R^2

Outputs
-------
- ``experiments/results/phase1_multidrug_metrics.csv`` (default paths;
  overridden by ``--output-dir`` / ``--plots-dir``)

Run from project root:

    python -m experiments.evaluation.evaluate_multidrug

Use ``--dry-run`` to print planned outputs without evaluating, and ``--force``
to overwrite an existing metrics CSV.

See ``--help`` for full CLI usage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.config import (  # noqa: E402
    DRUGS,
    MODELS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    ensure_dirs,
    get_logger,
)
from experiments.models.hybrid_multidrug import (  # noqa: E402
    MultiDrugHybridConfig,
    MultiDrugHybridGNNPBPK,
)
from experiments.training.multidrug_utils import (  # noqa: E402
    PatientRecord,
    cmax_auc_errors,
    load_drug_dataset,
    load_drug_graph,
    regression_metrics,
)

LOGGER = get_logger("phase1.evaluate_multidrug", "phase1_evaluate_multidrug.log")


# ---------------------------------------------------------------------------
# Per-drug helpers
# ---------------------------------------------------------------------------

def _load_model(drug: str) -> tuple[MultiDrugHybridGNNPBPK, dict[str, Any]]:
    """Reconstruct the model and return ``(model, config_dict)``."""
    out_dir = MODELS_DIR / f"hybrid_gnn_pbpk_{drug}_v1"
    cfg_dict = json.loads((out_dir / "config.json").read_text())

    cfg = MultiDrugHybridConfig(
        node_feat_dim=cfg_dict["node_feat_dim"],
        edge_feat_dim=cfg_dict["edge_feat_dim"],
        gnn_hidden=cfg_dict["gnn_hidden"],
        gnn_layers=cfg_dict["gnn_layers"],
        gnn_embed_dim=cfg_dict["gnn_embed_dim"],
        patient_feat_dim=cfg_dict["patient_feat_dim"],
        head_hidden=cfg_dict["head_hidden"],
        head_dropout=cfg_dict["head_dropout"],
        n_euler_steps=cfg_dict["n_euler_steps"],
    )
    model = MultiDrugHybridGNNPBPK(cfg)
    state = torch.load(out_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, cfg_dict


@torch.no_grad()
def _predict_curves(
    model: MultiDrugHybridGNNPBPK,
    graph: dict[str, torch.Tensor],
    records: list[PatientRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pred_curves, true_curves, times) all shaped [N_patients, T]."""
    pred_list: list[np.ndarray] = []
    true_list: list[np.ndarray] = []
    times: np.ndarray | None = None
    for rec in records:
        c_pred, *_ = model(
            graph["x"], graph["edge_index"], graph["edge_attr"],
            rec.features, rec.times_hr, rec.dose_mg, rec.weight_kg, rec.f_bio,
        )
        pred_list.append(c_pred.cpu().numpy())
        true_list.append(rec.concentration.cpu().numpy())
        if times is None:
            times = rec.times_hr.cpu().numpy()
    return np.stack(pred_list), np.stack(true_list), times  # type: ignore[return-value]


def _evaluate_drug(drug: str) -> dict[str, Any]:
    LOGGER.info("--- evaluating %s", drug)
    train_recs, val_recs, test_recs, _ = load_drug_dataset(drug)
    graph = load_drug_graph(drug)

    model, _cfg = _load_model(drug)

    pred_curves, true_curves, times = _predict_curves(model, graph, test_recs)
    flat_metrics = regression_metrics(pred_curves.ravel(), true_curves.ravel())
    summary_metrics = cmax_auc_errors(pred_curves, true_curves, times)

    obs_mean = float(true_curves.mean())
    rmse_pct_of_mean = (
        flat_metrics["RMSE"] / (obs_mean + 1e-12) * 100.0 if obs_mean > 0 else float("nan")
    )

    LOGGER.info(
        "  test n_patients=%d obs_mean=%.4g RMSE=%.4f (%.1f%% mean) R2=%.3f Cmax%%=%.1f AUC%%=%.1f",
        len(test_recs), obs_mean, flat_metrics["RMSE"], rmse_pct_of_mean,
        flat_metrics["R2"], summary_metrics["Cmax_pct_err"], summary_metrics["AUC_pct_err"],
    )

    return {
        "drug": drug,
        "n_test_patients": len(test_recs),
        "obs_mean_mg_L": obs_mean,
        "RMSE": flat_metrics["RMSE"],
        "RMSE_pct_of_mean": rmse_pct_of_mean,
        "MAE": flat_metrics["MAE"],
        "MAPE": flat_metrics["MAPE"],
        "R2": flat_metrics["R2"],
        "Cmax_pct_err": summary_metrics["Cmax_pct_err"],
        "AUC_pct_err": summary_metrics["AUC_pct_err"],
        "_pred_curves": pred_curves,
        "_true_curves": true_curves,
        "_times": times,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_observed_vs_predicted(per_drug: list[dict[str, Any]], plots_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()
    for ax, d in zip(axes, per_drug):
        y_pred = d["_pred_curves"].ravel()
        y_true = d["_true_curves"].ravel()
        lim_max = max(y_pred.max(), y_true.max()) * 1.05
        lim_min = min(y_pred.min(), y_true.min(), 0)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], color="grey", lw=0.8, ls="--")
        ax.scatter(y_true, y_pred, s=8, alpha=0.4)
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel("Observed (mg/L)")
        ax.set_ylabel("Predicted (mg/L)")
        ax.set_title(f"{d['drug']}  R\u00b2={d['R2']:.3f}  RMSE={d['RMSE']:.3g}")
        ax.grid(alpha=0.3)
    fig.suptitle("Observed vs predicted concentrations (test set)", y=1.02, fontsize=14)
    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / "observed_vs_predicted_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_pk_curves(per_drug: list[dict[str, Any]], plots_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()
    for ax, d in zip(axes, per_drug):
        times = d["_times"]
        pred = d["_pred_curves"]
        true = d["_true_curves"]

        pred_mean = pred.mean(axis=0)
        pred_low = np.percentile(pred, 2.5, axis=0)
        pred_high = np.percentile(pred, 97.5, axis=0)
        true_mean = true.mean(axis=0)
        true_low = np.percentile(true, 2.5, axis=0)
        true_high = np.percentile(true, 97.5, axis=0)

        ax.fill_between(times, true_low, true_high, alpha=0.2, color="tab:blue", label="Obs 95% CI")
        ax.plot(times, true_mean, color="tab:blue", lw=2, label="Obs mean")
        ax.fill_between(times, pred_low, pred_high, alpha=0.2, color="tab:orange", label="Pred 95% CI")
        ax.plot(times, pred_mean, color="tab:orange", lw=2, ls="--", label="Pred mean")
        ax.set_title(d["drug"])
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Concentration (mg/L)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Predicted vs observed PK curves (test set)", y=1.02, fontsize=14)
    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / "pk_curves_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_r2_bar(per_drug: list[dict[str, Any]], plots_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    drugs = [d["drug"] for d in per_drug]
    r2s = [d["R2"] for d in per_drug]
    bars = ax.bar(drugs, r2s, color="#00796B")
    ax.axhline(0.70, ls="--", color="red", lw=0.8, label="0.70 acceptance")
    ax.set_ylabel("Test R\u00b2")
    ax.set_ylim(0, 1.0)
    ax.set_title("Per-drug test R\u00b2")
    ax.grid(alpha=0.3, axis="y")
    for b, r in zip(bars, r2s):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{r:.3f}", ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / "r2_summary_bar.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def _resolve_dir(path: Path) -> Path:
    """Resolve a user-supplied directory relative to the repository root."""
    return path.resolve() if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluate_multidrug",
        description=(
            "Phase 1.6 - evaluate multi-drug hybrid models on held-out test data; "
            "write phase1_multidrug_metrics.csv and diagnostic plots."
        ),
    )
    parser.add_argument(
        "--drugs",
        nargs="*",
        default=None,
        metavar="DRUG",
        help="Subset of training drugs (default: all six in experiments.config.DRUGS).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Where to save phase1_multidrug_metrics.csv (default: experiments/results/).",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=PLOTS_DIR,
        help="Where to save evaluation plots (default: experiments/plots/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite metrics CSV when it already exists (default: refuse).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs and exit without evaluating or writing files.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    drugs: list[str]
    if args.drugs is None:
        drugs = list(DRUGS)
    elif len(args.drugs) == 0:
        LOGGER.error("'--drugs' requires at least one drug name, or omit --drugs for all.")
        return 2
    else:
        drugs = list(args.drugs)

    unknown = [d for d in drugs if d not in DRUGS]
    if unknown:
        LOGGER.error("Unknown drug(s) %s — must be subset of DRUGS: %s.", unknown, DRUGS)
        return 2

    output_dir = _resolve_dir(args.output_dir)
    plots_dir = _resolve_dir(args.plots_dir)
    csv_path = output_dir / "phase1_multidrug_metrics.csv"

    if args.dry_run:
        print("Dry-run: no files will be written.")
        print(f"  Drugs ({len(drugs)}): {', '.join(drugs)}")
        print(f"  Metrics CSV -> {csv_path}")
        print(f"  Plots dir   -> {plots_dir}")
        return 0

    if csv_path.exists() and not args.force:
        print(
            f"Refusing to overwrite existing CSV {csv_path} — pass --force to replace it.",
            file=sys.stderr,
        )
        return 1

    ensure_dirs()
    LOGGER.info("Phase 1.6 - evaluating multi-drug hybrid models")
    LOGGER.info("output_dir=%s plots_dir=%s", output_dir, plots_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    per_drug: list[dict[str, Any]] = []
    for drug in drugs:
        per_drug.append(_evaluate_drug(drug))

    # ---- save metrics CSV (without the heavy curve arrays) ----
    table_rows = []
    for d in per_drug:
        table_rows.append({k: v for k, v in d.items() if not k.startswith("_")})
    df = pd.DataFrame(table_rows)
    df.to_csv(csv_path, index=False)
    LOGGER.info("Saved %s", csv_path.relative_to(_PROJECT_ROOT))

    # ---- plots ----
    p1 = _plot_observed_vs_predicted(per_drug, plots_dir)
    p2 = _plot_pk_curves(per_drug, plots_dir)
    p3 = _plot_r2_bar(per_drug, plots_dir)
    LOGGER.info("Saved plots:")
    for p in (p1, p2, p3):
        LOGGER.info("  - %s", p.relative_to(_PROJECT_ROOT))

    # ---- print summary table to stdout for the report ----
    print("\nPhase 1.6 summary (test set):")
    cols = ["drug", "R2", "RMSE", "RMSE_pct_of_mean", "MAE", "MAPE", "Cmax_pct_err", "AUC_pct_err"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
