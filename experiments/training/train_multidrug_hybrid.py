"""Phase 1.5 - per-drug fine-tuning of the multi-drug hybrid GNN+PBPK model.

For every drug in ``experiments.config.DRUGS`` we:

1. Load the drug's PK CSV (Phase 1.3) and the cached molecular graph
   (Phase 1.4).
2. Build a :class:`MultiDrugHybridGNNPBPK` whose GNN encoder matches the
   combined theophylline hybrid checkpoint and load encoder weights from
   ``hybrid_gnn_pbpk_theoph_combined_v1`` (fallback: ``gnn_pretrain_combined_v1``).
3. Train with the schedule
       - epochs 1..FREEZE_EPOCHS : GNN frozen, head only
       - epochs FREEZE_EPOCHS+1.. : full fine-tuning
   Early stopping on validation RMSE (patience 15).
4. Save the best model under
       artifacts/models/hybrid_gnn_pbpk_{drug}_v1/
   alongside ``config.json``, ``scaler.json`` and ``metrics.json``.
5. Plot train / val composite loss curves to ``experiments/plots``.

Run from project root:

    python -m experiments.training.train_multidrug_hybrid \
        [--drugs theophylline warfarin] [--max-epochs 100]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.config import (  # noqa: E402
    DRUGS,
    MODELS_DIR,
    PLOTS_DIR,
    SEED,
    ensure_dirs,
    get_logger,
    seed_everything,
)
from experiments.models.hybrid_multidrug import (  # noqa: E402
    MultiDrugHybridConfig,
    MultiDrugHybridGNNPBPK,
)
from experiments.reference_pk import REFERENCE_PK_DATA  # noqa: E402
from experiments.training.multidrug_utils import (  # noqa: E402
    PatientRecord,
    patient_feat_dim,
    cmax_auc_errors,
    load_drug_dataset,
    load_drug_graph,
    load_pretrained_gnn_into,
    load_pretrained_gnn_state,
    regression_metrics,
)

LOGGER = get_logger("phase1.train_multidrug", "phase1_train_multidrug.log")


# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------

LR_HEAD = 5e-3
LR_FULL = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 100
PATIENCE = 15
# Head-only / permanently frozen encoder: allow early stopping on val RMSE during freeze.
FROZEN_ENCODER_ES_PATIENCE = 15
FREEZE_EPOCHS = 5
MIN_EPOCHS_BEFORE_ES = 15
BATCH_SIZE = 16
N_EULER_STEPS = 384
GRAD_CLIP = 5.0

# Legacy hook: multiplier on raw concentrations inside the loss (usually 1).
# Concentration residuals are scaled per-patient in ``_per_patient_loss``.
LOSS_CONC_SCALE: dict[str, float] = {}
# Warfarin: fine-tuning the pretrained GNN at full LR degrades encoder features;
# use an extended encoder-freeze window then a very small LR if unfreezing.
GNN_HEAD_ONLY_DRUGS: frozenset[str] = frozenset()

WARFARIN_LONG_FREEZE_EPOCHS = 40
WARFARIN_LR_FULL = 2.5e-5

# Stronger log(CL)/log(V) targets for drugs where the head must track tiny clearances.
LAMBDA_PK_SUP_BY_DRUG: dict[str, float] = {
    "warfarin": 0.72,
    "caffeine": 0.28,
    "acetaminophen": 0.35,
    "digoxin": 0.35,
}

# Loss weights from the brief.
LAMBDA_AUC = 0.10
LAMBDA_CMAX = 0.05
LAMBDA_PK_SUP = 0.12  # log CL/V vs simulator ground truth (Phase 1 CSV)
# Warfarin-only training schedule extensions (head-only converges slowly).
WARFARIN_MAX_EPOCHS = 160
WARFARIN_PATIENCE = 25

REFERENCE_KA_PER_HR = 1.5  # aligns with simulated datasets (download_pk_data default)


def _curve_norm_scale(obs: torch.Tensor) -> torch.Tensor:
    """Stable scale (~typical concentrations) per patient curve."""
    mean_c = torch.clamp(obs.mean(), min=torch.finfo(obs.dtype).tiny)
    peak = torch.clamp(obs.max(), min=torch.finfo(obs.dtype).tiny)
    return torch.maximum(mean_c, peak * torch.tensor(0.05, dtype=obs.dtype, device=obs.device))


def seed_head_from_reference_pk(model: MultiDrugHybridGNNPBPK, drug: str) -> None:
    """Set final linear bias toward literature PK means (helps low-CL drugs)."""
    ref = REFERENCE_PK_DATA[drug]
    cfg = model.config
    cl_pk = float(max(ref["CL_L_h"], cfg.cl_per_kg_floor))
    vd_pk = float(max(ref["Vd_L_kg"], cfg.v_per_kg_floor))
    ka_pk = float(max(REFERENCE_KA_PER_HR, cfg.ka_floor))

    last_lin: nn.Linear | None = None
    for m in model.head.modules():
        if isinstance(m, nn.Linear):
            last_lin = m
    assert last_lin is not None

    with torch.no_grad():
        last_lin.bias.copy_(
            torch.tensor(
                [math.log(cl_pk), math.log(vd_pk), math.log(ka_pk)],
                dtype=last_lin.bias.dtype,
                device=last_lin.bias.device,
            )
        )


def _per_patient_loss(
    pred: torch.Tensor,
    obs: torch.Tensor,
    times: torch.Tensor,
    conc_scale: float = 1.0,
) -> torch.Tensor:
    """Composite relative MSE matching Phase 1 terms (conc + AUC + Cmax weights)."""
    if conc_scale != 1.0:
        pred = pred * conc_scale
        obs = obs * conc_scale

    denom = _curve_norm_scale(obs)
    mse_conc = F.mse_loss(pred / denom, obs / denom)

    dt = times[1:] - times[:-1]
    auc_pred = torch.sum(0.5 * (pred[1:] + pred[:-1]) * dt)
    auc_obs = torch.sum(0.5 * (obs[1:] + obs[:-1]) * dt)
    t_span = (times[-1] - times[0]).clamp(min=torch.tensor(1e-6, dtype=times.dtype, device=times.device))
    mse_auc = ((auc_pred - auc_obs) / (denom * t_span)) ** 2

    cmax_pred = pred.max()
    cmax_obs = obs.max()
    mse_cmax = ((cmax_pred - cmax_obs) / denom) ** 2

    return mse_conc + LAMBDA_AUC * mse_auc + LAMBDA_CMAX * mse_cmax


def _pk_supervision_loss(
    pred_cl: torch.Tensor,
    pred_v: torch.Tensor,
    rec: PatientRecord,
) -> torch.Tensor:
    """Simulator provides CL_true / Vd_true; soft targets improve low-CL regimes."""
    if rec.cl_true_L_h is None or rec.vd_true_L is None:
        return torch.zeros((), device=pred_cl.device, dtype=pred_cl.dtype)
    tcl = rec.cl_true_L_h.to(pred_cl.device)
    tv = rec.vd_true_L.to(pred_v.device)
    return (
        F.mse_loss(torch.log(pred_cl + 1e-8), torch.log(tcl + 1e-8))
        + F.mse_loss(torch.log(pred_v + 1e-8), torch.log(tv + 1e-8))
    )


def _train_loss_sample(
    model: MultiDrugHybridGNNPBPK,
    graph: dict[str, torch.Tensor],
    rec: PatientRecord,
    conc_scale: float,
    lambda_pk_sup: float,
) -> torch.Tensor:
    c_pred, CL_p, V_p, _ka = model(
        graph["x"], graph["edge_index"], graph["edge_attr"],
        rec.features, rec.times_hr, rec.dose_mg, rec.weight_kg,
        rec.f_bio,
    )
    return (
        _per_patient_loss(c_pred, rec.concentration, rec.times_hr, conc_scale)
        + lambda_pk_sup * _pk_supervision_loss(CL_p, V_p, rec)
    )


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate(
    model: MultiDrugHybridGNNPBPK,
    graph: dict[str, torch.Tensor],
    records: list[PatientRecord],
    conc_scale: float = 1.0,
    lambda_pk_sup: float = LAMBDA_PK_SUP,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    preds_flat: list[float] = []
    truths_flat: list[float] = []
    pred_curves: list[np.ndarray] = []
    true_curves: list[np.ndarray] = []

    times_ref: np.ndarray | None = None
    for rec in records:
        c_pred, CL_p, V_p, _ka = model(
            graph["x"], graph["edge_index"], graph["edge_attr"],
            rec.features, rec.times_hr, rec.dose_mg, rec.weight_kg, rec.f_bio,
        )
        li = _per_patient_loss(c_pred, rec.concentration, rec.times_hr, conc_scale)
        li = li + lambda_pk_sup * _pk_supervision_loss(CL_p, V_p, rec)
        losses.append(float(li.item()))
        preds_flat.extend(c_pred.cpu().numpy().tolist())
        truths_flat.extend(rec.concentration.cpu().numpy().tolist())
        pred_curves.append(c_pred.cpu().numpy())
        true_curves.append(rec.concentration.cpu().numpy())
        if times_ref is None:
            times_ref = rec.times_hr.cpu().numpy()

    preds = np.array(preds_flat)
    truths = np.array(truths_flat)
    base = regression_metrics(preds, truths)

    pred_arr = np.stack(pred_curves)
    true_arr = np.stack(true_curves)
    cmax_auc = cmax_auc_errors(pred_arr, true_arr, times_ref)  # type: ignore[arg-type]

    return {
        "loss": float(np.mean(losses)),
        **base,
        **cmax_auc,
    }


def _train_one_drug(
    drug: str,
    max_epochs: int = MAX_EPOCHS,
    *,
    phase2_ablation: str | None = None,
) -> dict[str, Any]:
    LOGGER.info("=" * 60)
    LOGGER.info("Drug: %s", drug)
    LOGGER.info("=" * 60)

    train_recs, val_recs, test_recs, scaler = load_drug_dataset(drug)
    LOGGER.info(
        "  splits  | train=%d  val=%d  test=%d patients",
        len(train_recs), len(val_recs), len(test_recs),
    )

    graph = load_drug_graph(drug)
    LOGGER.info(
        "  graph   | atoms=%d  edges=%d",
        graph["x"].shape[0], graph["edge_index"].shape[1],
    )

    _, gnn_cfg = load_pretrained_gnn_state()
    cfg = MultiDrugHybridConfig(
        gnn_hidden=int(gnn_cfg["hidden_dim"]),
        gnn_layers=int(gnn_cfg["num_layers"]),
        gnn_embed_dim=int(gnn_cfg["embed_dim"]),
        patient_feat_dim=patient_feat_dim(drug),
        n_euler_steps=N_EULER_STEPS,
    )
    model = MultiDrugHybridGNNPBPK(cfg)
    if phase2_ablation == "A3":
        LOGGER.info("  ablation A3 | GNN randomly initialised (no transfer)")
    else:
        load_pretrained_gnn_into(model)
    seed_head_from_reference_pk(model, drug)
    LOGGER.info("  head init | seeded final bias from REFERENCE_PK_DATA")

    conc_scale = float(LOSS_CONC_SCALE.get(drug, 1.0))
    # A4: pretrained encoder, never fine-tuned (head-only training for all drugs).
    gnn_head_only = drug in GNN_HEAD_ONLY_DRUGS or phase2_ablation == "A4"
    lambda_pk_sup = float(LAMBDA_PK_SUP_BY_DRUG.get(drug, LAMBDA_PK_SUP))
    patience_drug = WARFARIN_PATIENCE if drug == "warfarin" else PATIENCE
    eff_max_epochs = max(max_epochs, WARFARIN_MAX_EPOCHS) if drug == "warfarin" else max_epochs
    freeze_epochs_eff = WARFARIN_LONG_FREEZE_EPOCHS if drug == "warfarin" else FREEZE_EPOCHS
    lr_full_eff = WARFARIN_LR_FULL if drug == "warfarin" else LR_FULL
    if gnn_head_only:
        es_patience = FROZEN_ENCODER_ES_PATIENCE
        min_es_epoch = MIN_EPOCHS_BEFORE_ES
    else:
        es_patience = patience_drug
        min_es_epoch = max(MIN_EPOCHS_BEFORE_ES, freeze_epochs_eff + 8)
    if gnn_head_only:
        LOGGER.info("  schedule | GNN frozen for all epochs (head-only fine-tune)")
    elif drug == "warfarin":
        LOGGER.info(
            "  schedule | GNN frozen epochs 1-%d, then fine-tune at lr=%.1e",
            freeze_epochs_eff,
            lr_full_eff,
        )
    if conc_scale != 1.0:
        LOGGER.info("  loss     | concentration term scaled by %.1f (training only)", conc_scale)

    LOGGER.info(
        "  model   | params=%d  encoder=hidden:%d/layers:%d/embed:%d",
        sum(p.numel() for p in model.parameters()),
        cfg.gnn_hidden, cfg.gnn_layers, cfg.gnn_embed_dim,
    )

    # Phase A: head only
    model.freeze_gnn()
    optimiser = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY,
    )

    best_val_rmse = float("inf")
    best_state = None
    best_epoch = -1
    no_improve = 0
    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
    }

    t0 = time.time()
    for epoch in range(1, eff_max_epochs + 1):
        if (not gnn_head_only) and epoch == freeze_epochs_eff + 1:
            model.unfreeze_gnn()
            optimiser = torch.optim.Adam(
                model.parameters(), lr=lr_full_eff, weight_decay=WEIGHT_DECAY,
            )
            LOGGER.info("  >>> unfroze GNN at epoch %d, lr=%.0e", epoch, lr_full_eff)

        model.train()
        order = np.random.permutation(len(train_recs))
        running = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            batch_idx = order[start : start + BATCH_SIZE]
            optimiser.zero_grad()
            batch_loss: torch.Tensor | None = None
            for i in batch_idx:
                rec = train_recs[i]
                li = _train_loss_sample(model, graph, rec, conc_scale, lambda_pk_sup)
                batch_loss = li if batch_loss is None else batch_loss + li
            assert batch_loss is not None
            batch_loss = batch_loss / len(batch_idx)
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimiser.step()
            running += batch_loss.item() * len(batch_idx)

        train_loss = running / len(train_recs)
        val = _evaluate(model, graph, val_recs, conc_scale, lambda_pk_sup)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_rmse"].append(val["RMSE"])

        if val["RMSE"] < best_val_rmse - 1e-6:
            best_val_rmse = val["RMSE"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 5 == 0 or no_improve >= es_patience:
            LOGGER.info(
                "  ep%3d | train=%.4f val_loss=%.4f val_rmse=%.4f best@%d (rmse=%.4f) %.0fs",
                epoch, train_loss, val["loss"], val["RMSE"], best_epoch, best_val_rmse,
                time.time() - t0,
            )

        es_ready = epoch >= min_es_epoch and (gnn_head_only or epoch > freeze_epochs_eff)
        if no_improve >= es_patience and es_ready:
            LOGGER.info("  early stopping at epoch %d", epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = _evaluate(model, graph, train_recs, conc_scale, lambda_pk_sup)
    val_metrics = _evaluate(model, graph, val_recs, conc_scale, lambda_pk_sup)
    test_metrics = _evaluate(model, graph, test_recs, conc_scale, lambda_pk_sup)

    LOGGER.info(
        "  DONE | test RMSE=%.4f R2=%.3f Cmax%%=%.1f AUC%%=%.1f (%.0fs)",
        test_metrics["RMSE"], test_metrics["R2"],
        test_metrics["Cmax_pct_err"], test_metrics["AUC_pct_err"],
        time.time() - t0,
    )

    # ---------------- save artifacts -----------------
    if phase2_ablation in ("A3", "A4"):
        out_dir = MODELS_DIR / f"phase2_ablation_{phase2_ablation}_{drug}"
    else:
        out_dir = MODELS_DIR / f"hybrid_gnn_pbpk_{drug}_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "scaler.json").write_text(json.dumps(scaler.to_dict(), indent=2))

    config_dict = {
        "drug": drug,
        "node_feat_dim": cfg.node_feat_dim,
        "edge_feat_dim": cfg.edge_feat_dim,
        "gnn_hidden": cfg.gnn_hidden,
        "gnn_layers": cfg.gnn_layers,
        "gnn_embed_dim": cfg.gnn_embed_dim,
        "patient_feat_dim": cfg.patient_feat_dim,
        "head_hidden": cfg.head_hidden,
        "head_dropout": cfg.head_dropout,
        "n_euler_steps": cfg.n_euler_steps,
        "max_epochs": eff_max_epochs,
        "freeze_epochs": freeze_epochs_eff,
        "min_epochs_before_es": MIN_EPOCHS_BEFORE_ES,
        "patience": patience_drug,
        "lr_head": LR_HEAD,
        "lr_full": lr_full_eff,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "lambda_auc": LAMBDA_AUC,
        "lambda_cmax": LAMBDA_CMAX,
        "lambda_pk_sup": lambda_pk_sup,
        "conc_scale_training": conc_scale,
        "gnn_head_only": gnn_head_only,
        "phase2_ablation": phase2_ablation,
    }
    (out_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

    metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "best_epoch": best_epoch,
        "n_epochs": history["epoch"][-1] if history["epoch"] else 0,
        "n_train_patients": len(train_recs),
        "n_val_patients": len(val_recs),
        "n_test_patients": len(test_recs),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    # ---------------- training curves -----------------
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(history["epoch"], history["train_loss"], label="train", color="tab:blue")
    ax1.plot(history["epoch"], history["val_loss"], label="val", color="tab:orange")
    if not gnn_head_only:
        ax1.axvline(
            freeze_epochs_eff + 0.5, ls="--", color="grey", lw=0.8,
            label=f"unfreeze@{freeze_epochs_eff}",
        )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Composite MSE loss")
    ax1.set_title(f"Training curves - {drug}")
    ax1.grid(alpha=0.3)
    ax1.legend()
    fig.tight_layout()
    out_png = PLOTS_DIR / (
        f"phase2_training_{phase2_ablation}_{drug}.png"
        if phase2_ablation
        else f"training_curves_{drug}.png"
    )
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)
    LOGGER.info("  saved %s", out_png.relative_to(_PROJECT_ROOT))

    return {
        "drug": drug,
        "model_dir": str(out_dir.relative_to(_PROJECT_ROOT)),
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-drug hybrid training")
    parser.add_argument("--drugs", nargs="*", default=None, help="Subset of drugs to train")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()

    ensure_dirs()
    seed_everything(SEED)

    targets = list(args.drugs) if args.drugs else list(DRUGS)
    LOGGER.info("Phase 1.5 - training drugs: %s (max_epochs=%d)", targets, args.max_epochs)

    summary: list[dict[str, Any]] = []
    for drug in targets:
        try:
            summary.append(_train_one_drug(drug, max_epochs=args.max_epochs))
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("FAILED training %s: %s", drug, exc)
            summary.append({"drug": drug, "error": str(exc)})

    LOGGER.info("=" * 60)
    LOGGER.info("Multi-drug training summary")
    LOGGER.info("=" * 60)
    for s in summary:
        if "error" in s:
            LOGGER.info("  %s: FAILED (%s)", s["drug"], s["error"])
            continue
        m = s["metrics"]["test"]
        LOGGER.info(
            "  %-13s test RMSE=%.4f R2=%.3f Cmax%%=%.1f AUC%%=%.1f",
            s["drug"], m["RMSE"], m["R2"], m["Cmax_pct_err"], m["AUC_pct_err"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
