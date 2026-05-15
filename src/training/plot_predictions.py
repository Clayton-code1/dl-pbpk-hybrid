"""Plot observed vs predicted PK curves for selected subjects.

Called automatically at the end of training; can also be run standalone:
    python src/training/plot_predictions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.hybrid_dl_pk import HybridDLPKModel, HybridConfig  # noqa: E402

ARTIFACT_DIR = _PROJECT_ROOT / "artifacts" / "models" / "hybrid_theoph_v1"
DATA_PATH = _PROJECT_ROOT / "data" / "processed" / "theoph" / "theoph_subjects.json"
N_PLOT = 3


def plot_subjects(
    model: HybridDLPKModel,
    records: list,
    out_dir: Path,
    n_plot: int = N_PLOT,
) -> None:
    """Save individual PNG files for the first ``n_plot`` subjects."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    to_plot = records[:n_plot]

    # Also create a combined figure
    fig, axes = plt.subplots(1, len(to_plot), figsize=(5 * len(to_plot), 4), squeeze=False)

    for i, (sid, x, t, c_obs, dose) in enumerate(to_plot):
        with torch.no_grad():
            c_pred, CL, V, ka = model(x, t, dose)

        t_np = t.numpy()
        obs_np = c_obs.numpy()
        pred_np = c_pred.numpy()

        # Dense curve for smooth plotting
        t_dense = torch.linspace(0, t[-1].item(), 200)
        with torch.no_grad():
            c_dense, *_ = model(x, t_dense, dose)
        td_np = t_dense.numpy()
        cd_np = c_dense.numpy()

        # --- individual PNG ---
        fig_s, ax_s = plt.subplots(figsize=(6, 4))
        ax_s.scatter(t_np, obs_np, c="black", s=40, zorder=5, label="Observed")
        ax_s.plot(td_np, cd_np, c="#4f46e5", lw=2, label="Predicted")
        ax_s.set_xlabel("Time (h)")
        ax_s.set_ylabel("Concentration (mg/L)")
        ax_s.set_title(f"Subject {sid}")
        ax_s.legend(frameon=False)
        ax_s.annotate(
            f"CL={CL.item():.2f}  V={V.item():.1f}  ka={ka.item():.2f}",
            xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top",
            fontsize=8, color="gray",
        )
        fig_s.tight_layout()
        path = out_dir / f"subject_{sid}.png"
        fig_s.savefig(path, dpi=150)
        plt.close(fig_s)
        print(f"  Saved {path.name}")

        # --- combined subplot ---
        ax = axes[0][i]
        ax.scatter(t_np, obs_np, c="black", s=30, zorder=5, label="Observed")
        ax.plot(td_np, cd_np, c="#4f46e5", lw=2, label="Predicted")
        ax.set_xlabel("Time (h)")
        if i == 0:
            ax.set_ylabel("Concentration (mg/L)")
        ax.set_title(f"Subject {sid}")
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    combined_path = out_dir / "predictions_overview.png"
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {combined_path.name}")


# ---------------------------------------------------------------------------
# Standalone entry point: load saved model and re-plot all subjects
# ---------------------------------------------------------------------------

def main() -> None:
    from src.training.train_hybrid_theoph import (
        load_subjects, split_subjects, prepare_tensors, StandardScaler,
        SEED, VAL_FRACTION,
    )

    print("Loading model and data ...")

    with open(ARTIFACT_DIR / "config.json") as f:
        cfg_dict = json.load(f)
    cfg = HybridConfig(
        hidden_dim=cfg_dict["hidden_dim"],
        n_euler_steps=cfg_dict["n_euler_steps"],
    )

    model = HybridDLPKModel(cfg)
    model.load_state_dict(torch.load(ARTIFACT_DIR / "model.pt", weights_only=True))
    model.eval()

    with open(ARTIFACT_DIR / "scaler.json") as f:
        scaler = StandardScaler.from_dict(json.load(f))

    subjects = load_subjects(DATA_PATH)
    train_subj, val_subj = split_subjects(subjects, VAL_FRACTION, SEED)
    all_subj = train_subj + val_subj
    all_recs, _ = prepare_tensors(all_subj, scaler)

    plot_subjects(model, all_recs, ARTIFACT_DIR, n_plot=min(N_PLOT, len(all_recs)))
    print("Done.")


if __name__ == "__main__":
    main()
