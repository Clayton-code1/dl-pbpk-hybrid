"""Phase 2.1 — baseline models matching Phase 1 splits and metrics.

Runs: PBPK-only, MLP, Random Forest, XGBoost (or sklearn GB fallback),
Vanilla GNN (encoder + head, no ODE), and evaluates Phase 1 DL-PBPK.

Caches per-drug test predictions for ``experiments/statistics/significance_tests.py``.

    python -m experiments.baselines.train_baselines
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

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
from experiments.evaluation.evaluate_multidrug import (  # noqa: E402
    _load_model as load_dlpbpk_model,
    _predict_curves as predict_dlpbpk_curves,
)
from experiments.models.hybrid_multidrug import (  # noqa: E402
    MultiDrugHybridConfig,
)
from experiments.phase2.utils import (  # noqa: E402
    load_molecular_vector,
    pbpk_reference_curves,
)
from experiments.training.multidrug_utils import (  # noqa: E402
    PatientRecord,
    cmax_auc_errors,
    load_drug_dataset,
    load_drug_graph,
    load_pretrained_gnn_state,
    regression_metrics,
)
from src.models.gnn.molecule_gnn import MoleculeGNN  # noqa: E402

LOGGER = get_logger("phase2.baselines", "phase2_train_baselines.log")

PRED_CACHE_PATH = RESULTS_DIR / "phase2_prediction_cache.pkl"

try:
    from xgboost import XGBRegressor  # noqa: WPS433
except ImportError:  # pragma: no cover
    XGBRegressor = None


def _records_to_xy(
    records: list[PatientRecord],
    mol_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack sklearn features: [mol || patient_z] and targets [T]."""
    n = len(records)
    T = int(records[0].concentration.shape[0])
    pfeat = np.stack([r.features.numpy() for r in records], axis=0)
    mol_rep = np.broadcast_to(mol_vec[None, :], (n, mol_vec.shape[0]))
    X = np.concatenate([mol_rep, pfeat], axis=1).astype(np.float64)
    y = np.stack([r.concentration.numpy() for r in records], axis=0).astype(np.float64)
    assert y.shape == (n, T)
    return X, y


class VanillaCurveGNN(nn.Module):
    """GNN embedding + MLP -> concentration vector (no ODE)."""

    def __init__(
        self,
        patient_feat_dim: int,
        n_times: int,
        hybrid_cfg: MultiDrugHybridConfig,
    ) -> None:
        super().__init__()
        cfg = hybrid_cfg
        self.gnn = MoleculeGNN(
            node_feat_dim=cfg.node_feat_dim,
            edge_feat_dim=cfg.edge_feat_dim,
            hidden_dim=cfg.gnn_hidden,
            num_layers=cfg.gnn_layers,
            embed_dim=cfg.gnn_embed_dim,
        )
        in_dim = cfg.gnn_embed_dim + patient_feat_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_times),
        )

    def forward(self, graph: dict[str, torch.Tensor], patient_feats: torch.Tensor) -> torch.Tensor:
        z = self.gnn(graph["x"], graph["edge_index"], graph["edge_attr"])
        b = patient_feats.shape[0]
        z_rep = z.unsqueeze(0).expand(b, -1)
        h = torch.cat([z_rep, patient_feats], dim=-1)
        return self.head(h)


def _train_vanilla_gnn(
    drug: str,
    train: list[PatientRecord],
    val: list[PatientRecord],
    graph: dict[str, torch.Tensor],
    device: torch.device,
    n_times: int,
    patient_dim: int,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> VanillaCurveGNN:
    _, gnn_cfg = load_pretrained_gnn_state()
    hcfg = MultiDrugHybridConfig(
        gnn_hidden=int(gnn_cfg["hidden_dim"]),
        gnn_layers=int(gnn_cfg["num_layers"]),
        gnn_embed_dim=int(gnn_cfg["embed_dim"]),
        patient_feat_dim=patient_dim,
        n_euler_steps=200,
    )
    model = VanillaCurveGNN(patient_dim, n_times, hcfg).to(device)
    graph_dev = {k: v.to(device) for k, v in graph.items()}
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    bs = 16
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience, no_improve = 12, 0
    max_ep = 120
    y_mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.tensor(y_std, dtype=torch.float32, device=device)
    for epoch in range(1, max_ep + 1):
        model.train()
        order = np.random.permutation(len(train))
        running = 0.0
        for s in range(0, len(order), bs):
            idx = order[s : s + bs]
            batch = [train[i] for i in idx]
            pf = torch.stack([r.features for r in batch]).to(device)
            tgt = torch.stack([r.concentration for r in batch]).to(device)
            tgt_s = (tgt - y_mean_t) / y_std_t
            optim.zero_grad()
            pred = model(graph_dev, pf)
            loss = torch.mean((pred - tgt_s) ** 2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            running += float(loss.item()) * len(idx)
        train_mse = running / len(train)

        model.eval()
        with torch.no_grad():
            pf_v = torch.stack([r.features for r in val]).to(device)
            tgt_v = torch.stack([r.concentration for r in val]).to(device)
            pred_v = model(graph_dev, pf_v) * y_std_t + y_mean_t
            val_rmse = float(torch.sqrt(torch.mean((pred_v - tgt_v) ** 2)).item())

        if val_rmse < best_val - 1e-6:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if epoch % 10 == 0:
            LOGGER.info("  VanillaGNN %s ep=%d train_mse=%.5f val_rmse=%.5f", drug, epoch, train_mse, val_rmse)
        if no_improve >= patience and epoch >= 20:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _predict_vanilla_gnn(
    model: VanillaCurveGNN,
    graph: dict[str, torch.Tensor],
    records: list[PatientRecord],
    device: torch.device,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> np.ndarray:
    model.eval()
    graph_dev = {k: v.to(device) for k, v in graph.items()}
    y_mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.tensor(y_std, dtype=torch.float32, device=device)
    outs: list[np.ndarray] = []
    bs = 32
    for s in range(0, len(records), bs):
        batch = records[s : s + bs]
        pf = torch.stack([r.features for r in batch]).to(device)
        pred = model(graph_dev, pf) * y_std_t + y_mean_t
        outs.append(pred.cpu().numpy())
    return np.vstack(outs)


def _make_boosting_regressor() -> Any:
    if XGBRegressor is not None:
        return MultiOutputRegressor(
            XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED,
                n_jobs=-1,
                verbosity=0,
            )
        )
    from sklearn.ensemble import HistGradientBoostingRegressor

    LOGGER.warning("xgboost not installed; using HistGradientBoostingRegressor")
    return MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=SEED,
        )
    )


def _run_drug(
    drug: str,
    device: torch.device,
    pred_cache: dict[str, Any],
) -> list[dict[str, Any]]:
    LOGGER.info("Baselines for %s", drug)
    train_r, val_r, test_r, _ = load_drug_dataset(drug)
    graph = load_drug_graph(drug)
    mol = load_molecular_vector(drug)
    times = test_r[0].times_hr.cpu().numpy()
    n_t = len(times)

    X_tr, y_tr = _records_to_xy(train_r, mol)
    X_te, y_te = _records_to_xy(test_r, mol)

    y_scaler = StandardScaler()
    y_tr_z = y_scaler.fit_transform(y_tr)
    y_te_z = y_scaler.transform(y_te)

    drug_preds: dict[str, np.ndarray] = {}
    y_true = y_te

    y_mean_np = y_tr.mean(axis=0, keepdims=True).astype(np.float32)
    y_std_np = y_tr.std(axis=0, keepdims=True).astype(np.float32)
    y_std_np = np.maximum(y_std_np, 1e-6)

    # --- PBPK-only ---
    pbpk_pred = pbpk_reference_curves(test_r, drug)
    drug_preds["PBPK-only"] = pbpk_pred.astype(np.float64)

    # --- sklearn baselines (train on train only) ---
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=min(32, len(X_tr)),
        learning_rate_init=1e-3,
        max_iter=600,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=SEED,
        verbose=False,
    )
    mlp.fit(X_tr, y_tr_z)
    drug_preds["MLP"] = y_scaler.inverse_transform(mlp.predict(X_te)).astype(np.float64)

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr_z)
    drug_preds["RandomForest"] = y_scaler.inverse_transform(rf.predict(X_te)).astype(np.float64)

    booster = _make_boosting_regressor()
    booster.fit(X_tr, y_tr_z)
    drug_preds["XGBoost"] = y_scaler.inverse_transform(booster.predict(X_te)).astype(np.float64)

    # --- Vanilla GNN ---
    pdim = int(train_r[0].features.shape[0])
    vgnn = _train_vanilla_gnn(
        drug, train_r, val_r, graph, device, n_t, pdim, y_mean_np, y_std_np,
    )
    drug_preds["VanillaGNN"] = _predict_vanilla_gnn(
        vgnn, graph, test_r, device, y_mean_np, y_std_np,
    )

    # --- DL-PBPK (Phase 1 checkpoint) ---
    model, _ = load_dlpbpk_model(drug)
    pred_hyb, true_hyb, _t = predict_dlpbpk_curves(model, graph, test_r)
    assert np.allclose(true_hyb, y_true, rtol=1e-4, atol=1e-5)
    drug_preds["DL-PBPK"] = pred_hyb.astype(np.float64)

    model_order = ["PBPK-only", "MLP", "RandomForest", "XGBoost", "VanillaGNN", "DL-PBPK"]
    rows: list[dict[str, Any]] = []
    for model_name in model_order:
        y_pred = drug_preds[model_name]
        flat = regression_metrics(y_pred.ravel(), y_true.ravel())
        summ = cmax_auc_errors(y_pred, y_true, times)
        obs_mean = float(y_true.mean())
        rmse_pct = flat["RMSE"] / (obs_mean + 1e-12) * 100.0 if obs_mean > 0 else float("nan")
        rows.append(
            {
                "drug": drug,
                "model": model_name,
                "RMSE": flat["RMSE"],
                "RMSE_pct_of_mean": rmse_pct,
                "MAE": flat["MAE"],
                "MAPE": flat["MAPE"],
                "R2": flat["R2"],
                "Cmax_pct_err": summ["Cmax_pct_err"],
                "AUC_pct_err": summ["AUC_pct_err"],
                "n_test_patients": len(test_r),
            }
        )

    per_patient_rmse: dict[str, np.ndarray] = {}
    abs_err_flat: dict[str, np.ndarray] = {}
    sq_err_flat: dict[str, np.ndarray] = {}
    for model_name in model_order:
        y_pred = drug_preds[model_name]
        per_patient_rmse[model_name] = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=1))
        abs_err_flat[model_name] = np.abs((y_pred - y_true).ravel())
        sq_err_flat[model_name] = ((y_pred - y_true) ** 2).ravel()

    pred_cache[drug] = {
        "times": times,
        "y_true": y_true,
        "per_patient_rmse": per_patient_rmse,
        "abs_err_flat": abs_err_flat,
        "sq_err_flat": sq_err_flat,
    }

    return rows


def main() -> int:
    ensure_dirs()
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("device=%s", device)

    all_rows: list[dict[str, Any]] = []
    pred_cache: dict[str, Any] = {}
    for drug in DRUGS:
        all_rows.extend(_run_drug(drug, device, pred_cache))

    import pandas as pd

    df = pd.DataFrame(all_rows)
    out_csv = RESULTS_DIR / "phase2_benchmark_metrics.csv"
    df.to_csv(out_csv, index=False)
    LOGGER.info("Wrote %s", out_csv)

    with open(PRED_CACHE_PATH, "wb") as f:
        pickle.dump(pred_cache, f)
    LOGGER.info("Wrote prediction cache %s", PRED_CACHE_PATH)

    # Quick comparison plot: mean test R² by model
    import matplotlib.pyplot as plt

    order = ["PBPK-only", "MLP", "RandomForest", "XGBoost", "VanillaGNN", "DL-PBPK"]
    avg_r2 = df.groupby("model")["R2"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(8, 4))
    avg_r2.plot(kind="bar", ax=ax, color="#546E7A")
    ax.set_ylabel("Mean test R² (6 drugs)")
    ax.set_title("Phase 2.1 — baseline mean R²")
    ax.axhline(0.7, color="red", ls="--", lw=0.8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p1 = PLOTS_DIR / "phase2_baseline_r2_bar.png"
    fig.savefig(p1, dpi=150)
    fig.savefig(p1.with_suffix(".pdf"))
    plt.close(fig)
    LOGGER.info("Saved %s", p1)

    print(df.pivot(index="drug", columns="model", values="R2").to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
