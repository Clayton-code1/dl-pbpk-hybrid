"""Pretrain MoleculeGNN on a multi-drug ADME regression task.

Trains the GNN encoder to predict a scalar ADME property from SMILES,
so the learned molecular representations transfer to the downstream
PBPK parameter prediction task.

Usage (from repo root):
    python src/training/pretrain_gnn_adme.py
    python src/training/pretrain_gnn_adme.py --max-samples 500
    python src/training/pretrain_gnn_adme.py --rebuild-cache
    python src/training/pretrain_gnn_adme.py --data-csv path/to/custom.csv

Input:  data/processed/adme_pretrain/adme_supervised.csv
            (columns: smiles, label, task)
        OR  data/processed/adme_pretrain/adme_graphs.pt  (cached graphs)
Output: artifacts/models/gnn_pretrain_v1/
            model_gnn.pt, scaler.json, metrics.json
"""

from __future__ import annotations

# -- Pre-flight environment checks (run BEFORE heavy imports) -------------

import platform
import sys


def _preflight() -> None:
    """Validate the runtime environment and print a banner.

    Import order matters on Windows: PyTorch must be loaded before RDKit
    to avoid DLL conflicts with c10.dll.
    """
    errors: list[str] = []

    # NumPy
    np_ver = "not installed"
    try:
        import numpy as _np
        np_ver = _np.__version__
        major = int(np_ver.split(".")[0])
        if major >= 2:
            errors.append(
                f"NumPy {np_ver} detected (>= 2.x). "
                f"RDKit wheels require NumPy < 2.\n"
                f"  Fix: pip install 'numpy>=1.24,<2'"
            )
    except ImportError:
        errors.append(
            "NumPy is not installed.\n"
            "  Fix: pip install 'numpy>=1.24,<2'"
        )

    # PyTorch (must load before RDKit on Windows to avoid DLL conflicts)
    torch_ver = "not installed"
    try:
        import torch as _torch
        torch_ver = _torch.__version__
    except ImportError:
        errors.append(
            "PyTorch is not installed.\n"
            "  Fix: pip install 'torch>=2.2,<3'"
        )

    # RDKit (loaded after PyTorch)
    rdkit_ok = False
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO")
        if mol is None:
            errors.append("RDKit imported but cannot parse SMILES.")
        else:
            rdkit_ok = True
    except Exception as exc:
        errors.append(
            f"RDKit import failed: {exc}\n"
            f"  Fix: pip install rdkit-pypi\n"
            f"  If you see '_ARRAY_API not found', downgrade NumPy:\n"
            f"    pip install 'numpy>=1.24,<2'"
        )

    # Banner
    print("=" * 60)
    print(" GNN ADME Pretraining - Environment")
    print("=" * 60)
    print(f"  Python  : {platform.python_version()}")
    print(f"  NumPy   : {np_ver}")
    print(f"  PyTorch : {torch_ver}")
    print(f"  RDKit   : {'OK' if rdkit_ok else 'MISSING / BROKEN'}")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print("=" * 60)

    if errors:
        print("\nENVIRONMENT ERRORS - cannot continue:\n")
        for i, e in enumerate(errors, 1):
            print(f"  [{i}] {e}\n")
        print("Re-run after fixing the above issues.")
        print("Tip: .\\scripts\\setup_gnn_training.ps1 -Recreate")
        sys.exit(1)


_preflight()

# -- Standard imports (safe after pre-flight) -----------------------------

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.molecules.rdkit_graph import smiles_to_graph, InvalidSMILESError, NODE_FEAT_DIM, EDGE_FEAT_DIM  # noqa: E402
from src.models.gnn.molecule_gnn import MoleculeGNN  # noqa: E402

# -- Paths ----------------------------------------------------------------

DATA_CSV = _PROJECT_ROOT / "data" / "processed" / "adme_pretrain" / "adme_supervised.csv"
CACHE_PT = _PROJECT_ROOT / "data" / "processed" / "adme_pretrain" / "adme_graphs.pt"
ARTIFACT_DIR_SUP = _PROJECT_ROOT / "artifacts" / "models" / "gnn_pretrain_v1"
ARTIFACT_DIR_COMBINED = _PROJECT_ROOT / "artifacts" / "models" / "gnn_pretrain_combined_v1"
UNSUP_PRETRAINED_WEIGHTS = (
    _PROJECT_ROOT
    / "artifacts"
    / "models"
    / "gnn_pretrain_unsup_v1"
    / "model_gnn.pt"
)

# -- Default hyper-parameters ---------------------------------------------

DEFAULTS = {
    "seed": 42,
    "val_fraction": 0.15,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "max_epochs": 200,
    "patience": 30,
    "log_every": 5,
    "batch_size": 64,
    "hidden_dim": 128,
    "num_layers": 3,
    "embed_dim": 128,
}

CPU_DEFAULTS = {
    "max_epochs": 60,
    "patience": 15,
    "hidden_dim": 64,
    "num_layers": 2,
    "embed_dim": 64,
    "batch_size": 32,
}


# -- Argument parsing -----------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GNN ADME pretraining")
    p.add_argument("--data-csv", type=str, default=None,
                   help="Path to CSV with smiles,label columns "
                        "(default: data/processed/adme_pretrain/adme_supervised.csv)")
    p.add_argument("--init-weights", action="store_true",
                   help=("Initialise GNN encoder from "
                         "artifacts/models/gnn_pretrain_unsup_v1/model_gnn.pt"))
    p.add_argument("--max-samples", type=int, default=None,
                   help="Subsample dataset for quick experiments")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="Rebuild the graph cache before training")
    p.add_argument("--cpu-friendly", action="store_true",
                   help="Use smaller model / fewer epochs for CPU training")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--embed-dim", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    return p.parse_args()


def resolve_config(args: argparse.Namespace) -> dict:
    """Merge defaults, cpu-friendly overrides, and explicit CLI args."""
    cfg = dict(DEFAULTS)
    if args.cpu_friendly:
        cfg.update(CPU_DEFAULTS)
    for key in ("batch_size", "max_epochs", "patience", "hidden_dim",
                "num_layers", "embed_dim", "lr"):
        val = getattr(args, key.replace("-", "_"), None)
        if val is not None:
            cfg[key] = val
    cfg["max_samples"] = args.max_samples
    return cfg


# -- Helpers --------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# -- Data loading ---------------------------------------------------------

def load_from_cache(path: Path) -> list[dict]:
    """Load pre-computed graphs from a .pt cache file."""
    raw = torch.load(path, weights_only=False)
    records = []
    for entry in raw:
        records.append({
            "smiles": entry["smiles"],
            "graph": {
                "x": entry["x"],
                "edge_index": entry["edge_index"],
                "edge_attr": entry["edge_attr"],
            },
            "label": float(entry["label"]),
        })
    return records


def load_from_csv(path: Path) -> list[dict]:
    """Load supervised CSV and parse SMILES into graphs on-the-fly."""
    import pandas as pd
    df = pd.read_csv(path)
    records: list[dict] = []
    skipped = 0
    for _, row in df.iterrows():
        try:
            g = smiles_to_graph(row["smiles"])
            label = float(row["label"])
            records.append({
                "smiles": row["smiles"],
                "graph": g,
                "label": label,
            })
        except (InvalidSMILESError, Exception):
            skipped += 1
    print(f"  Parsed {len(records)} molecules on-the-fly, skipped {skipped}")
    return records


def _print_dataset_banner(csv_path: Path) -> None:
    """Print a summary banner for the dataset being used."""
    import pandas as pd

    print(f"\n{'=' * 60}")
    print(" Dataset Summary")
    print(f"{'=' * 60}")
    print(f"  Path   : {csv_path}")

    if not csv_path.exists():
        print("  Status : FILE NOT FOUND")
        print(f"{'=' * 60}")
        return

    df = pd.read_csv(csv_path, nrows=0)
    cols = list(df.columns)
    has_task = "task" in cols

    df_full = pd.read_csv(csv_path, usecols=["smiles"] + (["task"] if has_task else []))
    n_rows = len(df_full)
    tasks = sorted(df_full["task"].unique().tolist()) if has_task else []

    print(f"  Rows   : {n_rows:,}")
    print(f"  Columns: {cols}")
    print(f"  task?  : {'yes — ' + str(tasks) if has_task else 'no'}")
    print(f"{'=' * 60}\n")


def load_data(cfg: dict, rebuild_cache: bool, data_csv: Path) -> list[dict]:
    """Load data from cache or CSV, with optional cache rebuild."""
    if rebuild_cache:
        print("  --rebuild-cache requested, running cache builder ...")
        subprocess.check_call(
            [sys.executable, str(_PROJECT_ROOT / "scripts" / "cache_adme_graphs.py"),
             "--data-csv", str(data_csv)]
        )
        print()

    if CACHE_PT.exists() and not rebuild_cache:
        print(f"  Loading cached graphs from {CACHE_PT}")
        records = load_from_cache(CACHE_PT)
        print(f"  {len(records)} molecules loaded from cache")
    elif CACHE_PT.exists() and rebuild_cache:
        print(f"  Loading freshly rebuilt cache from {CACHE_PT}")
        records = load_from_cache(CACHE_PT)
        print(f"  {len(records)} molecules loaded from cache")
    else:
        if not data_csv.exists():
            print(f"\n  ERROR: Data file not found: {data_csv}")
            print("  Run: python scripts/prepare_large_adme_corpus.py")
            sys.exit(1)
        print(f"  No cache found - parsing SMILES from {data_csv}")
        print("  (Run `python scripts/cache_adme_graphs.py` to speed up future runs)")
        records = load_from_csv(data_csv)

    return records


# -- Model / weight initialisation ----------------------------------------

def init_gnn_from_unsupervised(gnn: MoleculeGNN, weights_path: Path) -> None:
    """Initialise *gnn* from an unsupervised-pretrained state dict.

    Fails loudly if shapes mismatch so the user can see which layers are
    incompatible, rather than silently skipping parameters.
    """
    if not weights_path.exists():
        print(f"\n  ERROR: init-weights requested but file not found: {weights_path}")
        sys.exit(1)

    print(f"  Loading unsupervised encoder weights from {weights_path}")
    ckpt = torch.load(weights_path, map_location="cpu")

    if not isinstance(ckpt, dict):
        print("  ERROR: checkpoint is not a state dict")
        sys.exit(1)

    model_state = gnn.state_dict()
    to_load = {}
    mismatched = []

    for k, v in ckpt.items():
        if k not in model_state:
            continue
        if model_state[k].shape != v.shape:
            mismatched.append((k, tuple(model_state[k].shape), tuple(v.shape)))
        else:
            to_load[k] = v

    if mismatched:
        print("\n  ERROR: Shape mismatches while loading init-weights:")
        for name, expected, found in mismatched:
            print(f"    - {name}: expected {expected}, found {found}")
        print("  Aborting supervised pretraining with transfer initialisation.\n")
        sys.exit(1)

    model_state.update(to_load)
    gnn.load_state_dict(model_state)
    print("  Initialized supervised encoder from unsupervised pretrained weights")


# -- Model ----------------------------------------------------------------

class ADMEHead(nn.Module):
    """GNN + linear head for scalar ADME prediction."""
    def __init__(self, gnn: MoleculeGNN) -> None:
        super().__init__()
        self.gnn = gnn
        self.head = nn.Sequential(
            nn.Linear(gnn.embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        emb = self.gnn(x, edge_index, edge_attr)
        return self.head(emb).squeeze(-1)


# -- Training with mini-batching ------------------------------------------

def _graph_to_device(g: dict, device: str):
    return {
        "x": g["x"].to(device),
        "edge_index": g["edge_index"].to(device),
        "edge_attr": g["edge_attr"].to(device),
    }


def train_epoch(model: ADMEHead, records: list, optimiser, device: str,
                batch_size: int) -> float:
    model.train()
    indices = np.random.permutation(len(records))
    total_loss = 0.0
    n_batches = 0

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        optimiser.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)

        for idx in batch_idx:
            r = records[idx]
            gd = _graph_to_device(r["graph"], device)
            target = torch.tensor(r["label"], dtype=torch.float32, device=device)
            pred = model(gd["x"], gd["edge_index"], gd["edge_attr"])
            batch_loss = batch_loss + nn.functional.mse_loss(pred, target)

        batch_loss = batch_loss / len(batch_idx)
        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimiser.step()
        total_loss += batch_loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model: ADMEHead, records: list, device: str) -> dict:
    model.eval()
    preds, targets = [], []
    total_loss = 0.0
    for r in records:
        gd = _graph_to_device(r["graph"], device)
        target = torch.tensor(r["label"], dtype=torch.float32, device=device)
        pred = model(gd["x"], gd["edge_index"], gd["edge_attr"])
        total_loss += nn.functional.mse_loss(pred, target).item()
        preds.append(pred.item())
        targets.append(r["label"])
    preds_arr = np.array(preds)
    targets_arr = np.array(targets)
    rmse = float(np.sqrt(np.mean((preds_arr - targets_arr) ** 2)))
    return {"loss": total_loss / max(len(records), 1), "rmse": rmse}


# -- Main -----------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_csv = Path(args.data_csv) if args.data_csv else DATA_CSV
    use_transfer = bool(args.init_weights)
    artifact_dir = ARTIFACT_DIR_COMBINED if use_transfer else ARTIFACT_DIR_SUP

    print(f"\n  Device : {device}")
    if args.cpu_friendly and device == "cpu":
        print("  Mode   : CPU-friendly (smaller model, fewer epochs)")

    _print_dataset_banner(data_csv)
    if use_transfer:
        print("  Transfer init : ENABLED")
        print(f"  Init weights  : {UNSUP_PRETRAINED_WEIGHTS}")
    else:
        print("  Transfer init : disabled")

    # -- Load data --------------------------------------------------------
    records = load_data(cfg, rebuild_cache=args.rebuild_cache, data_csv=data_csv)

    if cfg["max_samples"] is not None and len(records) > cfg["max_samples"]:
        rng = np.random.RandomState(cfg["seed"])
        keep = rng.choice(len(records), cfg["max_samples"], replace=False)
        records = [records[i] for i in keep]
        print(f"  Subsampled to {len(records)} molecules (--max-samples)")

    if len(records) < 10:
        print("  Too few valid records for training.")
        sys.exit(1)

    # -- Train / val split ------------------------------------------------
    rng = np.random.RandomState(cfg["seed"])
    n_val = max(1, int(len(records) * cfg["val_fraction"]))
    perm = rng.permutation(len(records))
    val_idx = set(perm[:n_val].tolist())
    train_recs = [r for i, r in enumerate(records) if i not in val_idx]
    val_recs = [r for i, r in enumerate(records) if i in val_idx]

    print(f"\n  Total valid molecules : {len(records)}")
    print(f"  Train split          : {len(train_recs)}")
    print(f"  Validation split     : {len(val_recs)}")

    label_mean = float(np.mean([r["label"] for r in train_recs]))
    label_std = float(np.std([r["label"] for r in train_recs]))
    if label_std < 1e-8:
        label_std = 1.0

    # -- Build model ------------------------------------------------------
    gnn = MoleculeGNN(
        node_feat_dim=NODE_FEAT_DIM,
        edge_feat_dim=EDGE_FEAT_DIM,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        embed_dim=cfg["embed_dim"],
    )
    if use_transfer:
        init_gnn_from_unsupervised(gnn, UNSUP_PRETRAINED_WEIGHTS)
    model = ADMEHead(gnn).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters     : {n_params:,}")
    print(f"  Batch size           : {cfg['batch_size']}")
    print(f"  Max epochs           : {cfg['max_epochs']}")
    print(f"  Patience             : {cfg['patience']}")

    optimiser = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg["max_epochs"], eta_min=1e-5
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    epoch_metrics: list[dict] = []
    t0 = time.time()

    print(f"\n{'-' * 60}")
    print(f"  {'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'Best':>10}  {'Time':>6}")
    print(f"{'-' * 60}")

    for epoch in range(1, cfg["max_epochs"] + 1):
        train_loss = train_epoch(
            model, train_recs, optimiser, device, cfg["batch_size"]
        )
        scheduler.step()
        val_metrics = evaluate(model, val_recs, device)
        val_loss = val_metrics["loss"]

        epoch_metrics.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_rmse": round(val_metrics["rmse"], 6),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % cfg["log_every"] == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f"  {epoch:6d}  {train_loss:10.5f}  {val_loss:10.5f}  "
                f"{best_val_loss:10.5f}  {elapsed:5.0f}s"
            )

        if epochs_no_improve >= cfg["patience"]:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    elapsed_total = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    final_train = evaluate(model, train_recs, device)
    final_val = evaluate(model, val_recs, device)

    print(f"\n{'-' * 60}")
    print(f"  Final Train RMSE : {final_train['rmse']:.4f}")
    print(f"  Final Val   RMSE : {final_val['rmse']:.4f}")
    print(f"  Total time       : {elapsed_total:.0f}s")
    print(f"{'-' * 60}")

    # -- Save artifacts ---------------------------------------------------
    artifact_dir.mkdir(parents=True, exist_ok=True)

    gnn_state = {k: v for k, v in model.state_dict().items() if k.startswith("gnn.")}
    gnn_state_clean = {k.removeprefix("gnn."): v for k, v in gnn_state.items()}
    torch.save(gnn_state_clean, artifact_dir / "model_gnn.pt")

    scaler_info = {"label_mean": label_mean, "label_std": label_std}
    with open(artifact_dir / "scaler.json", "w") as f:
        json.dump(scaler_info, f, indent=2)

    metrics = {
        "train": final_train,
        "val": final_val,
        "n_train": len(train_recs),
        "n_val": len(val_recs),
        "n_epochs": epoch,
        "elapsed_seconds": round(elapsed_total, 2),
        "config": cfg,
        "gnn_config": gnn.config_dict(),
        "device": device,
        "epoch_log": epoch_metrics,
    }
    with open(artifact_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save a small config file for reproducibility / inspection
    config_payload = {
        "gnn_config": gnn.config_dict(),
        "training_config": cfg,
        "transfer_initialisation": use_transfer,
        "unsupervised_weights": str(UNSUP_PRETRAINED_WEIGHTS) if use_transfer else None,
        "data_csv": str(data_csv),
    }
    with open(artifact_dir / "config.json", "w") as f:
        json.dump(config_payload, f, indent=2)

    print(f"\n  Artifacts saved to {artifact_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
