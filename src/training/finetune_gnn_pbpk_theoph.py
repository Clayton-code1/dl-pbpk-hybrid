"""Fine-tune the Hybrid GNN-PBPK model on Theophylline PK curves.

Loads pretrained GNN weights (if available) and trains the full
HybridGNNPBPK model end-to-end on observed Theophylline concentration-time
profiles using the differentiable 1-compartment ODE solver.

Usage (from repo root):
    python src/training/finetune_gnn_pbpk_theoph.py

Input:   data/processed/theoph/theoph_subjects.json
Output:  artifacts/models/hybrid_gnn_pbpk_theoph_v1/
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path


def _preflight() -> None:
    """Validate that required packages are importable before heavy work.

    Import order matters on Windows: PyTorch must be loaded before RDKit
    to avoid DLL conflicts with c10.dll.
    """
    errors: list[str] = []
    try:
        import numpy as _np
        major = int(_np.__version__.split(".")[0])
        if major >= 2:
            errors.append(f"NumPy {_np.__version__} (>=2.x) - need less than 2 for RDKit compatibility")
    except ImportError:
        errors.append("NumPy is not installed")

    try:
        import torch as _t
    except ImportError:
        errors.append("PyTorch is not installed")

    try:
        from rdkit import Chem
        if Chem.MolFromSmiles("CCO") is None:
            errors.append("RDKit imported but SMILES parse failed")
    except Exception as exc:
        errors.append(f"RDKit import failed: {exc}")

    if errors:
        print("=" * 60)
        print(" ENVIRONMENT ERRORS - cannot continue")
        print("=" * 60)
        for i, e in enumerate(errors, 1):
            print(f"  [{i}] {e}")
        print("\nTip: .\\scripts\\setup_gnn_training.ps1 -Recreate")
        sys.exit(1)


_preflight()

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.molecules.rdkit_graph import (  # noqa: E402
    smiles_to_graph, NODE_FEAT_DIM, EDGE_FEAT_DIM,
)
from src.models.hybrid_gnn_pbpk import HybridGNNPBPK, HybridGNNConfig  # noqa: E402
from src.training.train_hybrid_theoph import (  # noqa: E402
    THEOPHYLLINE_SMILES,
    StandardScaler,
)

# -- configuration --------------------------------------------------------

SEED = 42
VAL_FRACTION = 0.25
LR = 5e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 1000
PATIENCE = 200
LOG_EVERY = 50
N_EULER_STEPS = 300
CONC_EPS = 1e-3

V_TYPICAL = 30.0  # L - fixed volume, from Theophylline population mean

DATA_PATH = _PROJECT_ROOT / "data" / "processed" / "theoph" / "theoph_subjects.json"
PRETRAIN_DIR = _PROJECT_ROOT / "artifacts" / "models" / "gnn_pretrain_v1"
COMBINED_PRETRAIN_DIR = _PROJECT_ROOT / "artifacts" / "models" / "gnn_pretrain_combined_v1"
COMBINED_PRETRAIN_WEIGHTS = COMBINED_PRETRAIN_DIR / "model_gnn.pt"
COMBINED_PRETRAIN_CONFIG = COMBINED_PRETRAIN_DIR / "config.json"
ARTIFACT_DIR = _PROJECT_ROOT / "artifacts" / "models" / "hybrid_gnn_pbpk_theoph_v1"
ARTIFACT_DIR_COMBINED = (
    _PROJECT_ROOT / "artifacts" / "models" / "hybrid_gnn_pbpk_theoph_combined_v1"
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_subjects(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_subjects(subjects: list[dict], val_frac: float, seed: int):
    rng = np.random.RandomState(seed)
    n = len(subjects)
    n_val = max(1, int(round(n * val_frac)))
    idx = rng.permutation(n)
    val_idx = set(idx[:n_val].tolist())
    train = [s for i, s in enumerate(subjects) if i not in val_idx]
    val = [s for i, s in enumerate(subjects) if i in val_idx]
    return train, val


def prepare_tensors(subjects: list[dict], scaler: StandardScaler | None = None):
    features_raw = np.array([
        [s["dose_mg"], s["weight_kg"], s["dose_mgkg"]]
        for s in subjects
    ])
    if scaler is None:
        scaler = StandardScaler().fit(features_raw)
    features_norm = scaler.transform(features_raw)

    records = []
    for i, s in enumerate(subjects):
        x = torch.tensor(features_norm[i], dtype=torch.float32)
        t = torch.tensor(s["times_hr"], dtype=torch.float32)
        c = torch.tensor(s["concentration"], dtype=torch.float32)
        d = torch.tensor(s["dose_mg"], dtype=torch.float32)
        records.append((s["subject_id"], x, t, c, d))
    return records, scaler


def log_mse_loss(pred: torch.Tensor, obs: torch.Tensor, eps: float = CONC_EPS) -> torch.Tensor:
    return nn.functional.mse_loss(torch.log(pred + eps), torch.log(obs + eps))


def _load_combined_encoder_weights(model: HybridGNNPBPK, weights_path: Path) -> None:
    """Initialise model.gnn from combined unsupervised+supervised encoder weights.

    Fails loudly if the weights file is missing or shapes do not match.
    """
    if not weights_path.exists():
        print(f"\nERROR: Combined encoder weights not found: {weights_path}")
        print("  Expected weights from gnn_pretrain_combined_v1/model_gnn.pt")
        sys.exit(1)

    print(f"\nLoading combined encoder weights from {weights_path} ...")
    ckpt = torch.load(weights_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        print("  ERROR: combined weights file does not contain a state dict")
        sys.exit(1)

    gnn_state = model.gnn.state_dict()
    to_load: dict[str, torch.Tensor] = {}
    mismatched: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    for k, v in ckpt.items():
        if k not in gnn_state:
            continue
        if gnn_state[k].shape != v.shape:
            mismatched.append((k, tuple(gnn_state[k].shape), tuple(v.shape)))
        else:
            to_load[k] = v

    missing_keys = [k for k in gnn_state.keys() if k not in ckpt]

    if mismatched or missing_keys:
        print("\n  ERROR: Incompatible combined encoder weights:")
        for name, expected, found in mismatched:
            print(f"    - {name}: expected {expected}, found {found}")
        if missing_keys:
            print("  Missing keys in checkpoint (expected in model.gnn):")
            for name in missing_keys:
                print(f"    - {name}")
        print("  Aborting hybrid fine-tuning with combined encoder initialisation.\n")
        sys.exit(1)

    gnn_state.update(to_load)
    model.gnn.load_state_dict(gnn_state)
    print("  Initialized hybrid fine-tuning from combined encoder weights")


@torch.no_grad()
def evaluate(model, graph, records, V_tensor):
    model.eval()
    all_sq: list[float] = []
    total_loss = 0.0
    for _sid, x, t, c_obs, dose in records:
        c_pred, *_ = model(
            graph["x"], graph["edge_index"], graph["edge_attr"],
            x, t, dose, V_tensor,
        )
        total_loss += log_mse_loss(c_pred, c_obs).item()
        err = (c_pred - c_obs).cpu().numpy()
        all_sq.extend((err ** 2).tolist())
    n = len(records)
    return {
        "loss": total_loss / max(n, 1),
        "rmse": float(np.sqrt(np.mean(all_sq))),
    }


def main() -> None:
    set_seed(SEED)
    print("=" * 60)
    print(" Hybrid GNN-PBPK Fine-tuning - Theophylline")
    print("=" * 60)

    if not DATA_PATH.exists():
        print(f"Data not found: {DATA_PATH}")
        sys.exit(1)

    print(f"\nBuilding Theophylline graph from SMILES ...")
    graph = smiles_to_graph(THEOPHYLLINE_SMILES)
    print(f"  Atoms: {graph['x'].shape[0]}  Edges: {graph['edge_index'].shape[1]}")

    print(f"\nLoading subjects from {DATA_PATH} ...")
    subjects = load_subjects(DATA_PATH)
    train_subj, val_subj = split_subjects(subjects, VAL_FRACTION, SEED)
    print(f"  Total subjects : {len(subjects)}")
    print(f"  Train subjects : {len(train_subj)}")
    print(f"  Val   subjects : {len(val_subj)}")

    train_recs, scaler = prepare_tensors(train_subj)
    val_recs, _ = prepare_tensors(val_subj, scaler)

    if not COMBINED_PRETRAIN_CONFIG.exists():
        print(f"\nERROR: Combined encoder config not found: {COMBINED_PRETRAIN_CONFIG}")
        print("  Expected config.json alongside gnn_pretrain_combined_v1/model_gnn.pt")
        sys.exit(1)

    with open(COMBINED_PRETRAIN_CONFIG, "r", encoding="utf-8") as f:
        combined_cfg = json.load(f)
    gnn_cfg = combined_cfg.get("gnn_config", {})

    cfg = HybridGNNConfig(
        node_feat_dim=NODE_FEAT_DIM,
        edge_feat_dim=EDGE_FEAT_DIM,
        gnn_hidden=int(gnn_cfg.get("hidden_dim", HybridGNNConfig().gnn_hidden)),
        gnn_layers=int(gnn_cfg.get("num_layers", HybridGNNConfig().gnn_layers)),
        gnn_embed_dim=int(gnn_cfg.get("embed_dim", HybridGNNConfig().gnn_embed_dim)),
        n_euler_steps=N_EULER_STEPS,
    )

    print("\nGNN configuration for hybrid fine-tuning:")
    print(f"  hidden_dim : {cfg.gnn_hidden}")
    print(f"  num_layers : {cfg.gnn_layers}")
    print(f"  embed_dim  : {cfg.gnn_embed_dim}")
    model = HybridGNNPBPK(cfg)

    print("\nEncoder initialisation:")
    print(f"  Combined weights path : {COMBINED_PRETRAIN_WEIGHTS}")
    _load_combined_encoder_weights(model, COMBINED_PRETRAIN_WEIGHTS)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params} parameters")

    V_tensor = torch.tensor(V_TYPICAL, dtype=torch.float32)

    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=MAX_EPOCHS, eta_min=1e-5)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    t0 = time.time()

    print(f"\nTraining (max {MAX_EPOCHS} epochs, patience {PATIENCE}) ...\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        indices = np.random.permutation(len(train_recs))

        for idx in indices:
            _sid, x, t, c_obs, dose = train_recs[idx]
            optimiser.zero_grad()
            c_pred, *_ = model(
                graph["x"], graph["edge_index"], graph["edge_attr"],
                x, t, dose, V_tensor,
            )
            loss = log_mse_loss(c_pred, c_obs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(train_recs)
        scheduler.step()

        val_metrics = evaluate(model, graph, val_recs, V_tensor)
        avg_val = val_metrics["loss"]

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % LOG_EVERY == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:5d} | train {avg_train:.5f} | val {avg_val:.5f} "
                f"| best {best_val_loss:.5f} | {elapsed:.0f}s"
            )

        if epochs_no_improve >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_train = evaluate(model, graph, train_recs, V_tensor)
    final_val = evaluate(model, graph, val_recs, V_tensor)

    elapsed_total = time.time() - t0

    print(f"\n--- Final Metrics ---")
    print(f"  Train RMSE : {final_train['rmse']:.4f} mg/L")
    print(f"  Val   RMSE : {final_val['rmse']:.4f} mg/L")
    print(f"  Total time : {elapsed_total:.0f}s")

    # Show per-subject PK params
    print(f"\n--- Predicted PK Parameters ---")
    model.eval()
    all_recs = train_recs + val_recs
    with torch.no_grad():
        for sid, x, t, c_obs, dose in all_recs:
            _, CL, V, ka = model(
                graph["x"], graph["edge_index"], graph["edge_attr"],
                x, t, dose, V_tensor,
            )
            print(f"  Subject {sid:>2s}: CL={CL.item():.3f} L/h  V={V.item():.2f} L  ka={ka.item():.3f} 1/h")

    ARTIFACT_DIR_COMBINED.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ARTIFACT_DIR_COMBINED / "model.pt")

    with open(ARTIFACT_DIR_COMBINED / "scaler.json", "w") as f:
        json.dump(scaler.to_dict(), f, indent=2)

    config_dict = {
        "node_feat_dim": cfg.node_feat_dim,
        "edge_feat_dim": cfg.edge_feat_dim,
        "gnn_hidden": cfg.gnn_hidden,
        "gnn_layers": cfg.gnn_layers,
        "gnn_embed_dim": cfg.gnn_embed_dim,
        "patient_feat_dim": cfg.patient_feat_dim,
        "head_hidden": cfg.head_hidden,
        "cl_floor": cfg.cl_floor,
        "ka_floor": cfg.ka_floor,
        "n_euler_steps": cfg.n_euler_steps,
        "V_typical": V_TYPICAL,
        "smiles": THEOPHYLLINE_SMILES,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "seed": SEED,
    }
    with open(ARTIFACT_DIR_COMBINED / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    metrics = {
        "train": final_train,
        "val": final_val,
        "n_train": len(train_subj),
        "n_val": len(val_subj),
        "n_epochs": epoch,
        "elapsed_seconds": round(elapsed_total, 2),
    }
    with open(ARTIFACT_DIR_COMBINED / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nArtifacts saved to {ARTIFACT_DIR_COMBINED}")
    print("=" * 60)


if __name__ == "__main__":
    main()
