"""Phase 2.2 — ablation variants A1–A5 per original specification.

| Variant | Description |
|---|---|
| A1 | PBPK-only (no GNN, no transfer) |
| A2 | GNN-only (Vanilla GNN baseline) |
| A3 | Hybrid, no transfer learning (random GNN init) |
| A4 | Hybrid, pretrained GNN, encoder never fine-tuned |
| A5 | Full DL-PBPK |

Trains A3/A3 checkpoints under ``artifacts/models/phase2_ablation_A{3,4}_{drug}/``
when missing.

    python -m experiments.ablation.ablation_study [--force-retrain]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
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
    load_drug_dataset,
    load_drug_graph,
    regression_metrics,
)
from experiments.training.train_multidrug_hybrid import _train_one_drug  # noqa: E402


LOGGER = get_logger("phase2.ablation", "phase2_ablation.log")


def _load_ablation_hybrid(drug: str, tag: str) -> MultiDrugHybridGNNPBPK:
    out_dir = MODELS_DIR / f"phase2_ablation_{tag}_{drug}"
    cfg_dict = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
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
    try:
        state = torch.load(out_dir / "model.pt", map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(out_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    seed_everything(SEED)

    bench_path = RESULTS_DIR / "phase2_benchmark_metrics.csv"
    if not bench_path.exists():
        raise FileNotFoundError(f"Need {bench_path}; run experiments.baselines.train_baselines first.")
    bench = pd.read_csv(bench_path)

    # Ensure A3/A4 checkpoints exist
    for tag in ("A3", "A4"):
        for drug in DRUGS:
            ckpt_dir = MODELS_DIR / f"phase2_ablation_{tag}_{drug}"
            if args.force_retrain or not (ckpt_dir / "model.pt").exists():
                LOGGER.info("Training ablation %s for %s", tag, drug)
                _train_one_drug(drug, phase2_ablation=tag)
            else:
                LOGGER.info("Reuse checkpoint %s", ckpt_dir)

    summary_rows: list[dict[str, Any]] = []

    for drug in DRUGS:
        _, _, test_r, _ = load_drug_dataset(drug)
        graph = load_drug_graph(drug)
        y_true = np.stack([r.concentration.numpy() for r in test_r])

        def _metrics(pred: np.ndarray) -> tuple[float, float]:
            flat = regression_metrics(pred.ravel(), y_true.ravel())
            return float(flat["R2"]), float(flat["RMSE"])

        r2_a1 = float(bench.loc[(bench["drug"] == drug) & (bench["model"] == "PBPK-only"), "R2"].iloc[0])
        rm_a1 = float(bench.loc[(bench["drug"] == drug) & (bench["model"] == "PBPK-only"), "RMSE"].iloc[0])
        r2_a2 = float(bench.loc[(bench["drug"] == drug) & (bench["model"] == "VanillaGNN"), "R2"].iloc[0])
        rm_a2 = float(bench.loc[(bench["drug"] == drug) & (bench["model"] == "VanillaGNN"), "RMSE"].iloc[0])
        r2_a5 = float(bench.loc[(bench["drug"] == drug) & (bench["model"] == "DL-PBPK"), "R2"].iloc[0])
        rm_a5 = float(bench.loc[(bench["drug"] == drug) & (bench["model"] == "DL-PBPK"), "RMSE"].iloc[0])

        for label, r2v, rmv in (
            ("A1_PBPK_only", r2_a1, rm_a1),
            ("A2_GNN_only", r2_a2, rm_a2),
            ("A5_Full_DLPBPK", r2_a5, rm_a5),
        ):
            summary_rows.append({"variant": label, "drug": drug, "R2": r2v, "RMSE": rmv})

        for tag in ("A3", "A4"):
            model = _load_ablation_hybrid(drug, tag)
            pred, true_m, _ = _predict_curves(model, graph, test_r)
            assert np.allclose(true_m, y_true, rtol=1e-4, atol=1e-4)
            r2v, rmv = _metrics(pred)
            vlabel = "A3_hybrid_no_transfer" if tag == "A3" else "A4_hybrid_encoder_frozen"
            summary_rows.append({"variant": vlabel, "drug": drug, "R2": r2v, "RMSE": rmv})

    summary = pd.DataFrame(summary_rows)
    mean_by = summary.groupby("variant", sort=False)[["R2", "RMSE"]].mean().reset_index()
    mean_by = mean_by.rename(columns={"R2": "mean_R2_6drugs", "RMSE": "mean_RMSE_6drugs"})
    out_detail = RESULTS_DIR / "phase2_ablation_by_drug.csv"
    summary.to_csv(out_detail, index=False)
    mean_by.to_csv(RESULTS_DIR / "phase2_ablation_summary.csv", index=False)
    LOGGER.info("Wrote %s and phase2_ablation_summary.csv", out_detail)

    # Bar chart: mean R2 for logical order
    order = [
        "A1_PBPK_only",
        "A2_GNN_only",
        "A3_hybrid_no_transfer",
        "A4_hybrid_encoder_frozen",
        "A5_Full_DLPBPK",
    ]
    plot_df = mean_by.set_index("variant").reindex(order).reset_index()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(plot_df["variant"], plot_df["mean_R2_6drugs"], color="#00695C")
    ax.set_ylabel("Mean test R² (6 drugs)")
    ax.set_title("Phase 2.2 — ablation study")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(plot_df["mean_R2_6drugs"]):
        if not np.isnan(v):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    outp = PLOTS_DIR / "ablation_study.png"
    fig.savefig(outp, dpi=150)
    fig.savefig(outp.with_suffix(".pdf"))
    plt.close(fig)
    LOGGER.info("Saved %s", outp)

    return 0


if __name__ == "__main__":
    sys.exit(main())
