"""Phase 2.4 — external validation on ibuprofen (zero-shot).

Loads the Phase 1 pretrained GNN encoder, **freezes** it, seeds the fusion
head with literature PK for ibuprofen, and evaluates on the held-out test
split **without any optimisation on ibuprofen data**.

    python -m experiments.phase2.external_ibuprofen
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.config import (  # noqa: E402
    EXTERNAL_DRUG,
    PLOTS_DIR,
    RESULTS_DIR,
    SEED,
    ensure_dirs,
    get_logger,
    seed_everything,
)
from experiments.evaluation.evaluate_multidrug import _predict_curves  # noqa: E402
from experiments.models.hybrid_multidrug import (  # noqa: E402
    MultiDrugHybridConfig,
    MultiDrugHybridGNNPBPK,
)
from experiments.training.multidrug_utils import (  # noqa: E402
    cmax_auc_errors,
    load_drug_dataset,
    load_drug_graph,
    load_pretrained_gnn_into,
    load_pretrained_gnn_state,
    patient_feat_dim,
    regression_metrics,
)
from experiments.training.train_multidrug_hybrid import (  # noqa: E402
    N_EULER_STEPS,
    seed_head_from_reference_pk,
)

LOGGER = get_logger("phase2.external", "phase2_external_ibuprofen.log")


def main() -> int:
    ensure_dirs()
    seed_everything(SEED)
    drug = EXTERNAL_DRUG
    LOGGER.info("External validation drug=%s", drug)

    _train, _val, test_r, _ = load_drug_dataset(drug)
    graph = load_drug_graph(drug)

    _, gnn_cfg = load_pretrained_gnn_state()
    cfg = MultiDrugHybridConfig(
        gnn_hidden=int(gnn_cfg["hidden_dim"]),
        gnn_layers=int(gnn_cfg["num_layers"]),
        gnn_embed_dim=int(gnn_cfg["embed_dim"]),
        patient_feat_dim=patient_feat_dim(drug),
        n_euler_steps=N_EULER_STEPS,
    )
    model = MultiDrugHybridGNNPBPK(cfg)
    load_pretrained_gnn_into(model)
    model.freeze_gnn()
    seed_head_from_reference_pk(model, drug)
    model.eval()

    pred, true_arr, times = _predict_curves(model, graph, test_r)
    flat = regression_metrics(pred.ravel(), true_arr.ravel())
    summ = cmax_auc_errors(pred, true_arr, times)
    obs_mean = float(true_arr.mean())
    rmse_pct = flat["RMSE"] / (obs_mean + 1e-12) * 100.0 if obs_mean > 0 else float("nan")

    import pandas as pd

    row = {
        "drug": drug,
        "split": "test",
        "n_patients": len(test_r),
        "RMSE": flat["RMSE"],
        "RMSE_pct_of_mean": rmse_pct,
        "MAE": flat["MAE"],
        "R2": flat["R2"],
        "Cmax_pct_err": summ["Cmax_pct_err"],
        "AUC_pct_err": summ["AUC_pct_err"],
        "encoder": "pretrained_frozen",
        "fine_tuned_on_ibuprofen": False,
    }
    out_csv = RESULTS_DIR / "phase2_external_validation.csv"
    pd.DataFrame([row]).to_csv(out_csv, index=False)
    LOGGER.info("Wrote %s", out_csv)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_arr.ravel(), pred.ravel(), s=12, alpha=0.35, color="#004D40")
    mx = max(true_arr.max(), pred.max()) * 1.05
    ax.plot([0, mx], [0, mx], color="grey", ls="--", lw=1)
    ax.set_xlabel("Observed (mg/L)")
    ax.set_ylabel("Predicted (mg/L)")
    ax.set_title(f"Ibuprofen zero-shot (frozen encoder) R²={flat['R2']:.3f}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    outp = PLOTS_DIR / "phase2_external_ibuprofen_scatter.png"
    fig.savefig(outp, dpi=150)
    fig.savefig(outp.with_suffix(".pdf"))
    plt.close(fig)

    print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
