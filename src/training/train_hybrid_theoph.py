"""End-to-end training of the Hybrid DL+ODE PK model on Theophylline data.

Usage (from repo root):
    python src/training/train_hybrid_theoph.py
    # or via the PowerShell wrapper:
    .\\scripts\\train_hybrid_theoph.ps1

Artifacts are saved under  artifacts/models/hybrid_theoph_v1/
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---- path bootstrap (allows running from repo root) ----
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.hybrid_dl_pk import HybridDLPKModel, HybridConfig  # noqa: E402

# Theophylline canonical SMILES (used by GNN path when SMILES not provided)
THEOPHYLLINE_SMILES = "Cn1c2c(c(=O)n(c1=O)C)[nH]cn2"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
VAL_FRACTION = 0.25          # hold-out 3 of 12 subjects
LR = 1e-2
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 1500
PATIENCE = 250               # early-stopping patience
LOG_EVERY = 50
N_EULER_STEPS = 300
HIDDEN_DIM = 32
N_INPUT = 3                  # dose_mg, weight_kg, dose_mgkg
CONC_EPS = 1e-3

DATA_PATH = _PROJECT_ROOT / "data" / "processed" / "theoph" / "theoph_subjects.json"
ARTIFACT_DIR = _PROJECT_ROOT / "artifacts" / "models" / "hybrid_theoph_v1"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    val   = [s for i, s in enumerate(subjects) if i in val_idx]
    return train, val


class StandardScaler:
    """Minimal z-score scaler serialisable to JSON."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None
        return (X - self.mean_) / self.std_

    def to_dict(self) -> dict:
        assert self.mean_ is not None
        return {
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StandardScaler":
        s = cls()
        s.mean_ = np.array(d["mean"])
        s.std_  = np.array(d["std"])
        return s


def prepare_tensors(subjects: list[dict], scaler: StandardScaler | None = None):
    """Convert subject dicts to lists of tensors; optionally fit the scaler."""
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


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def log_mse_loss(pred: torch.Tensor, obs: torch.Tensor, eps: float = CONC_EPS) -> torch.Tensor:
    """MSE in log-space: MSE( log(pred+eps), log(obs+eps) )."""
    return nn.functional.mse_loss(
        torch.log(pred + eps),
        torch.log(obs + eps),
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: HybridDLPKModel, records: list) -> dict:
    model.eval()
    total_loss = 0.0
    all_sq: list[float] = []
    all_abs: list[float] = []

    for _sid, x, t, c_obs, dose in records:
        c_pred, *_ = model(x, t, dose)
        total_loss += log_mse_loss(c_pred, c_obs).item()
        err = (c_pred - c_obs).cpu().numpy()
        all_sq.extend((err ** 2).tolist())
        all_abs.extend(np.abs(err).tolist())

    n = len(records)
    return {
        "loss": total_loss / max(n, 1),
        "rmse": float(np.sqrt(np.mean(all_sq))),
        "mae": float(np.mean(all_abs)),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: HybridDLPKModel,
    train_recs: list,
    val_recs: list,
    lr: float = LR,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> dict:
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max_epochs, eta_min=1e-5,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history: dict[str, list] = {"train_loss": [], "val_loss": []}

    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        indices = np.random.permutation(len(train_recs))

        for idx in indices:
            _sid, x, t, c_obs, dose = train_recs[idx]
            optimiser.zero_grad()
            c_pred, *_ = model(x, t, dose)
            loss = log_mse_loss(c_pred, c_obs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(train_recs)
        scheduler.step()

        val_metrics = evaluate(model, val_recs)
        avg_val = val_metrics["loss"]

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % LOG_EVERY == 0 or epoch == 1:
            elapsed = time.time() - t0
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch:5d}  |  train {avg_train:.5f}  |  val {avg_val:.5f}  "
                f"|  best {best_val_loss:.5f}  |  lr {current_lr:.1e}  |  {elapsed:.0f}s"
            )

        if epochs_no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(
    model: HybridDLPKModel,
    config: HybridConfig,
    scaler: StandardScaler,
    metrics: dict,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"  Saved model.pt")

    with open(out_dir / "scaler.json", "w") as f:
        json.dump(scaler.to_dict(), f, indent=2)
    print(f"  Saved scaler.json")

    cfg_dict = {
        "n_input": config.n_input,
        "hidden_dim": config.hidden_dim,
        "n_euler_steps": config.n_euler_steps,
        "cl_floor": config.cl_floor,
        "v_floor": config.v_floor,
        "ka_floor": config.ka_floor,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "seed": SEED,
        "conc_eps": CONC_EPS,
        "val_fraction": VAL_FRACTION,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)
    print(f"  Saved config.json")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed(SEED)
    print("=" * 60)
    print(" Hybrid DL+ODE PK Model -- Theophylline Training")
    print("=" * 60)

    # ---- data ----
    print(f"\nLoading data from {DATA_PATH.name} ...")
    subjects = load_subjects(DATA_PATH)
    train_subj, val_subj = split_subjects(subjects, VAL_FRACTION, SEED)
    print(f"  Train: {len(train_subj)} subjects  |  Val: {len(val_subj)} subjects")

    train_recs, scaler = prepare_tensors(train_subj)
    val_recs, _ = prepare_tensors(val_subj, scaler)

    # ---- model ----
    cfg = HybridConfig(n_input=N_INPUT, hidden_dim=HIDDEN_DIM, n_euler_steps=N_EULER_STEPS)
    model = HybridDLPKModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params} trainable parameters")
    print(f"  Features: {N_INPUT}  |  Hidden: {cfg.hidden_dim}  |  Euler steps: {cfg.n_euler_steps}")

    # Show initial PK param estimates for sanity
    model.eval()
    with torch.no_grad():
        sid, x, t, c, d = train_recs[0]
        _, CL0, V0, ka0 = model(x, t, d)
        print(f"  Init PK (subj {sid}):  CL={CL0.item():.2f}  V={V0.item():.1f}  ka={ka0.item():.2f}")

    # ---- train ----
    print(f"\nTraining (max {MAX_EPOCHS} epochs, patience {PATIENCE}, lr {LR}) ...\n")
    history = train(model, train_recs, val_recs)

    # ---- final metrics ----
    train_metrics = evaluate(model, train_recs)
    val_metrics   = evaluate(model, val_recs)
    all_recs = train_recs + val_recs
    all_metrics = evaluate(model, all_recs)

    metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "all": all_metrics,
        "n_epochs_trained": len(history["train_loss"]),
    }

    print(f"\n--- Final Metrics ---")
    for split, m in [("Train", train_metrics), ("Val", val_metrics), ("All", all_metrics)]:
        print(f"  {split:5s}  RMSE={m['rmse']:.4f} mg/L  |  MAE={m['mae']:.4f} mg/L  |  loss={m['loss']:.5f}")

    # ---- PK parameters per subject ----
    print(f"\n--- Predicted PK Parameters ---")
    model.eval()
    with torch.no_grad():
        for sid, x, t, c_obs, dose in all_recs:
            _, CL, V, ka = model(x, t, dose)
            print(f"  Subject {sid:>2s}:  CL={CL.item():.3f} L/h  V={V.item():.2f} L  ka={ka.item():.3f} 1/h")

    # ---- save ----
    print(f"\nSaving artifacts to {ARTIFACT_DIR} ...")
    save_artifacts(model, cfg, scaler, metrics, ARTIFACT_DIR)

    # ---- plot ----
    print("\nGenerating prediction plots ...")
    try:
        from src.training.plot_predictions import plot_subjects
        plot_subjects(model, all_recs, ARTIFACT_DIR)
    except Exception as e:
        print(f"  Plot generation skipped: {e}")

    print("\n" + "=" * 60)
    print(" Training complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
