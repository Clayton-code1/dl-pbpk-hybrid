"""Phase 3.2 — Monte Carlo prediction intervals; calibration curve.

Propagates ±30% (log-normal, σ=0.3 on log scale) uncertainty on predicted CL and V
from the hybrid fixed point; ka fixed. Evaluates central PI coverage on held-out test
curves (same splits as Phase 1).

Outputs
-------
- ``experiments/results/phase3_uncertainty_calibration.csv``
- ``experiments/plots/uncertainty_calibration.png`` (+ PDF)

    python -m experiments.uncertainty.monte_carlo_calibration
"""

from __future__ import annotations

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
    PLOTS_DIR,
    RESULTS_DIR,
    SEED,
    ensure_dirs,
    get_logger,
    seed_everything,
)
from experiments.evaluation.evaluate_multidrug import _load_model  # noqa: E402
from experiments.phase2.utils import oral_1cpt_batch  # noqa: E402
from experiments.training.multidrug_utils import (  # noqa: E402
    load_drug_dataset,
    load_drug_graph,
    split_rng_seed_for_drug,
)

LOGGER = get_logger("phase3.mc_calibration", "phase3_monte_carlo.log")

N_MC = 1000
MC_LOG_SD = 0.3
NOMINAL_LEVELS = np.linspace(0.5, 0.99, 15)


@torch.no_grad()
def _point_pk(
    model: torch.nn.Module,
    graph: dict[str, torch.Tensor],
    rec: Any,
) -> tuple[float, float, float, float]:
    """Return CL (L/h), V (L), ka (1/h), dose_eff (mg)."""
    emb = model.get_drug_embedding(graph["x"], graph["edge_index"], graph["edge_attr"])
    w = rec.weight_kg
    CL, V, ka, _, _ = model.predict_pk_params(emb, rec.features, w)
    dose_eff = rec.dose_mg * rec.f_bio.clamp(min=1e-4, max=1.0)
    return float(CL.item()), float(V.item()), float(ka.item()), float(dose_eff.item())


def _mc_curves_numpy(
    times: np.ndarray,
    dose_eff: float,
    ka: float,
    CL: float,
    V: float,
    weight_kg: float,
    rng: np.random.Generator,
    n_mc: int,
) -> np.ndarray:
    """Shape ``(n_mc, T)`` concentrations mg/L; independent log-noise on CL and V."""
    z1 = rng.standard_normal(n_mc)
    z2 = rng.standard_normal(n_mc)
    CL_s = CL * np.exp(MC_LOG_SD * z1)
    V_s = V * np.exp(MC_LOG_SD * z2)
    cl_pk = CL_s / weight_kg
    vd_pk = V_s / weight_kg
    out = np.empty((n_mc, len(times)), dtype=np.float64)
    for i in range(n_mc):
        out[i] = oral_1cpt_batch(times, dose_eff, 1.0, ka, cl_pk[i], vd_pk[i], weight_kg)
    return out


def main() -> int:
    ensure_dirs()
    seed_everything(SEED)

    per_drug_empirical: dict[str, list[float]] = {d: [] for d in DRUGS}

    for drug in DRUGS:
        _train_recs, _v, test_recs, _ = load_drug_dataset(drug)
        graph = load_drug_graph(drug)
        model, _ = _load_model(drug)
        model.eval()
        rng = np.random.default_rng(split_rng_seed_for_drug(drug, base_seed=SEED + 77_777))

        for nominal in NOMINAL_LEVELS:
            lo_p = 50.0 * (1.0 - float(nominal))
            hi_p = 100.0 - lo_p
            n_ok, n_tot = 0, 0
            for rec in test_recs:
                CL, V, ka, dose_eff = _point_pk(model, graph, rec)
                times = rec.times_hr.cpu().numpy()
                y_true = rec.concentration.cpu().numpy()
                w = float(rec.weight_kg.item())
                mc = _mc_curves_numpy(times, dose_eff, ka, CL, V, w, rng, N_MC)
                q_lo = np.percentile(mc, lo_p, axis=0)
                q_hi = np.percentile(mc, hi_p, axis=0)
                inside = (y_true >= q_lo) & (y_true <= q_hi)
                n_ok += int(inside.sum())
                n_tot += int(len(y_true))
            per_drug_empirical[drug].append(float(n_ok / max(n_tot, 1)))

    rows: list[dict[str, Any]] = []
    for drug in DRUGS:
        for ni, nominal in enumerate(NOMINAL_LEVELS):
            rows.append(
                {
                    "scope": drug,
                    "nominal_interval_frac": float(nominal),
                    "empirical_coverage": per_drug_empirical[drug][ni],
                    "n_mc": N_MC,
                    "mc_log_sd": MC_LOG_SD,
                }
            )

    # pooled over all test (patient × time) points
    pooled_empirical: list[float] = []
    for nominal in NOMINAL_LEVELS:
        lo_p = 50.0 * (1.0 - float(nominal))
        hi_p = 100.0 - lo_p
        n_ok, n_tot = 0, 0
        for drug in DRUGS:
            _, _, test_recs, _ = load_drug_dataset(drug)
            graph = load_drug_graph(drug)
            model, _ = _load_model(drug)
            model.eval()
            rng = np.random.default_rng(split_rng_seed_for_drug(drug, base_seed=SEED + 88_888))
            for rec in test_recs:
                CL, V, ka, dose_eff = _point_pk(model, graph, rec)
                times = rec.times_hr.cpu().numpy()
                y_true = rec.concentration.cpu().numpy()
                w = float(rec.weight_kg.item())
                mc = _mc_curves_numpy(times, dose_eff, ka, CL, V, w, rng, N_MC)
                q_lo = np.percentile(mc, lo_p, axis=0)
                q_hi = np.percentile(mc, hi_p, axis=0)
                inside = (y_true >= q_lo) & (y_true <= q_hi)
                n_ok += int(inside.sum())
                n_tot += int(len(y_true))
        pooled_empirical.append(float(n_ok / max(n_tot, 1)))
        rows.append(
            {
                "scope": "ALL_DRUGS_POOLED",
                "nominal_interval_frac": float(nominal),
                "empirical_coverage": pooled_empirical[-1],
                "n_concentration_points": n_tot,
                "n_mc": N_MC,
                "mc_log_sd": MC_LOG_SD,
            }
        )

    # fix per-drug rows: add n_concentration_points
    for drug in DRUGS:
        _, _, test_recs, _ = load_drug_dataset(drug)
        ntot = sum(len(rec.concentration) for rec in test_recs)
        for r in rows:
            if r["scope"] == drug:
                r["n_concentration_points"] = ntot

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "phase3_uncertainty_calibration.csv"
    df.to_csv(out_csv, index=False)
    LOGGER.info("Wrote %s", out_csv)

    fig, ax = plt.subplots(figsize=(6, 5))
    for drug in DRUGS:
        yvals = per_drug_empirical[drug]
        ax.plot(NOMINAL_LEVELS, yvals, alpha=0.35, label=drug)
    ax.plot(
        NOMINAL_LEVELS,
        pooled_empirical,
        color="black",
        lw=2.5,
        label="All drugs (pooled)",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.set_xlim(0.48, 1.0)
    ax.set_ylim(0.48, 1.0)
    ax.set_xlabel("Nominal central interval coverage")
    ax.set_ylabel("Empirical coverage (test conc. points)")
    ax.set_title("Phase 3.2 — MC PI calibration (CL,V log-σ=0.3)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    outp = PLOTS_DIR / "uncertainty_calibration.png"
    fig.savefig(outp, dpi=150)
    fig.savefig(outp.with_suffix(".pdf"))
    plt.close(fig)
    LOGGER.info("Saved %s", outp)

    idx90 = int(np.argmin(np.abs(NOMINAL_LEVELS - 0.9)))
    LOGGER.info(
        "Pooled empirical coverage at ~90%% nominal: %.3f",
        pooled_empirical[idx90],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
