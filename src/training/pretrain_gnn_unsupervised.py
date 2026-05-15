"""Unsupervised GNN pretraining via masked node feature reconstruction.

Trains the same MoleculeGNN encoder used in the supervised pipeline, but
without labels.  A fraction of node features are zeroed out and the model
learns to reconstruct them from molecular context — analogous to masked-
language modelling in NLP.

Usage (from repo root):
    python src/training/pretrain_gnn_unsupervised.py --cpu-friendly
    python src/training/pretrain_gnn_unsupervised.py --max-samples 500 --cpu-friendly
    python src/training/pretrain_gnn_unsupervised.py \
        --data-csv  data/processed/adme_pretrain/adme_unsupervised_sample_10k.csv \
        --cache-path data/processed/adme_pretrain/adme_unsupervised_sample_10k_graphs.pt

Input:  A CSV with a ``smiles`` column (labels ignored / not required).
        OR a .pt graph cache produced by scripts/cache_adme_graphs.py.
Output: artifacts/models/gnn_pretrain_unsup_v1/
            model_gnn.pt, config.json, metrics.json
"""

from __future__ import annotations

# -- Pre-flight (same pattern as supervised script) -----------------------

import platform
import sys


def _preflight() -> None:
    errors: list[str] = []

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
        errors.append("NumPy is not installed.\n  Fix: pip install 'numpy>=1.24,<2'")

    torch_ver = "not installed"
    try:
        import torch as _torch
        torch_ver = _torch.__version__
    except ImportError:
        errors.append("PyTorch is not installed.\n  Fix: pip install 'torch>=2.2,<3'")

    rdkit_ok = False
    try:
        from rdkit import Chem
        if Chem.MolFromSmiles("CCO") is not None:
            rdkit_ok = True
        else:
            errors.append("RDKit imported but cannot parse SMILES.")
    except Exception as exc:
        errors.append(f"RDKit import failed: {exc}")

    print("=" * 60)
    print(" GNN Unsupervised Pretraining - Environment")
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
        sys.exit(1)


_preflight()

# -- Standard imports -----------------------------------------------------

import argparse
import json
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

DATA_CSV = _PROJECT_ROOT / "data" / "processed" / "adme_pretrain" / "adme_unsupervised_sample_10k.csv"
CACHE_PT = _PROJECT_ROOT / "data" / "processed" / "adme_pretrain" / "adme_unsupervised_sample_10k_graphs.pt"
ARTIFACT_DIR = _PROJECT_ROOT / "artifacts" / "models" / "gnn_pretrain_unsup_v1"

# -- Hyper-parameters -----------------------------------------------------

DEFAULTS = {
    "seed": 42,
    "val_fraction": 0.10,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "max_epochs": 100,
    "patience": 20,
    "log_every": 5,
    "batch_size": 64,
    "hidden_dim": 128,
    "num_layers": 3,
    "embed_dim": 128,
    "mask_rate": 0.15,
}

CPU_DEFAULTS = {
    "max_epochs": 40,
    "patience": 12,
    "hidden_dim": 64,
    "num_layers": 2,
    "embed_dim": 64,
    "batch_size": 32,
}

# -- CLI ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GNN unsupervised pretraining")
    p.add_argument("--data-csv", type=str, default=None,
                   help="CSV with a smiles column")
    p.add_argument("--cache-path", type=str, default=None,
                   help=".pt graph cache file")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--cpu-friendly", action="store_true")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--embed-dim", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--mask-rate", type=float, default=None,
                   help="Fraction of node features to mask (default 0.15)")
    return p.parse_args()


def resolve_config(args: argparse.Namespace) -> dict:
    cfg = dict(DEFAULTS)
    if args.cpu_friendly:
        cfg.update(CPU_DEFAULTS)
    for key in ("batch_size", "max_epochs", "patience", "hidden_dim",
                "num_layers", "embed_dim", "lr", "mask_rate"):
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

def load_graphs_from_cache(path: Path) -> list[dict]:
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
        })
    return records


def load_graphs_from_csv(path: Path) -> list[dict]:
    import pandas as pd
    df = pd.read_csv(path)
    records: list[dict] = []
    skipped = 0
    for _, row in df.iterrows():
        try:
            g = smiles_to_graph(str(row["smiles"]))
            records.append({"smiles": row["smiles"], "graph": g})
        except (InvalidSMILESError, Exception):
            skipped += 1
    print(f"  Parsed {len(records)} molecules on-the-fly, skipped {skipped}")
    return records


def load_data(cache_path: Path, data_csv: Path) -> list[dict]:
    if cache_path.exists():
        print(f"  Loading cached graphs from {cache_path}")
        records = load_graphs_from_cache(cache_path)
        print(f"  {len(records)} molecules loaded from cache")
    else:
        if not data_csv.exists():
            print(f"\n  ERROR: Data not found: {data_csv}")
            print("  Run: python scripts/sample_unsupervised_corpus.py")
            sys.exit(1)
        print(f"  No cache found — parsing SMILES from {data_csv}")
        print("  (Run scripts/cache_adme_graphs.py with --data-csv and --output to cache)")
        records = load_graphs_from_csv(data_csv)
    return records


def _print_dataset_banner(data_csv: Path) -> None:
    import pandas as pd

    print(f"\n{'=' * 60}")
    print(" Dataset Summary")
    print(f"{'=' * 60}")
    print(f"  Path   : {data_csv}")

    if not data_csv.exists():
        print("  Status : FILE NOT FOUND")
        print(f"{'=' * 60}")
        return

    df = pd.read_csv(data_csv, nrows=0)
    cols = list(df.columns)
    has_task = "task" in cols

    df_full = pd.read_csv(data_csv, usecols=["smiles"] + (["task"] if has_task else []))
    n_rows = len(df_full)
    tasks = sorted(df_full["task"].unique().tolist()) if has_task else []

    print(f"  Rows   : {n_rows:,}")
    print(f"  Columns: {cols}")
    print(f"  task?  : {'yes — ' + str(tasks) if has_task else 'no'}")
    print(f"{'=' * 60}\n")


# -- Masked node reconstruction model ------------------------------------

class MaskedNodeReconstructor(nn.Module):
    """GNN encoder + per-node decoder for masked feature reconstruction.

    Accesses the GNN's internal node_encoder and message-passing layers to
    obtain per-node hidden states, then decodes to reconstruct original
    node features at masked positions.
    """

    def __init__(self, gnn: MoleculeGNN) -> None:
        super().__init__()
        self.gnn = gnn
        self.decoder = nn.Sequential(
            nn.Linear(gnn.hidden_dim, gnn.hidden_dim),
            nn.ReLU(),
            nn.Linear(gnn.hidden_dim, gnn.node_feat_dim),
        )

    def forward_node_hidden(self, x, edge_index, edge_attr):
        """Return per-node hidden states [N, hidden_dim]."""
        h = self.gnn.node_encoder(x)
        for layer in self.gnn.layers:
            h = layer(h, edge_index, edge_attr)
        return h

    def forward(self, x_masked, edge_index, edge_attr, mask):
        """Reconstruct original features at masked node positions.

        Parameters
        ----------
        x_masked : [N, node_feat_dim]  – node features with masked positions zeroed
        edge_index : [2, E]
        edge_attr : [E, edge_feat_dim]
        mask : [N]  – boolean mask, True = node was masked

        Returns
        -------
        reconstructed : [M, node_feat_dim]  – predicted features for masked nodes
        """
        h = self.forward_node_hidden(x_masked, edge_index, edge_attr)
        return self.decoder(h[mask])


# -- Training helpers -----------------------------------------------------

def _graph_to_device(g: dict, device: str):
    return {
        "x": g["x"].to(device),
        "edge_index": g["edge_index"].to(device),
        "edge_attr": g["edge_attr"].to(device),
    }


def _create_mask(num_nodes: int, mask_rate: float) -> torch.BoolTensor:
    """Random boolean mask — True for nodes to be masked."""
    n_mask = max(1, int(num_nodes * mask_rate))
    perm = torch.randperm(num_nodes)
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[perm[:n_mask]] = True
    return mask


def train_epoch(model: MaskedNodeReconstructor, records: list, optimiser,
                device: str, batch_size: int, mask_rate: float) -> float:
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
            x_orig = gd["x"]
            num_nodes = x_orig.size(0)
            if num_nodes < 2:
                continue

            mask = _create_mask(num_nodes, mask_rate).to(device)
            x_masked = x_orig.clone()
            x_masked[mask] = 0.0

            target = x_orig[mask]
            pred = model(x_masked, gd["edge_index"], gd["edge_attr"], mask)
            batch_loss = batch_loss + nn.functional.mse_loss(pred, target)

        batch_loss = batch_loss / max(len(batch_idx), 1)
        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimiser.step()
        total_loss += batch_loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model: MaskedNodeReconstructor, records: list, device: str,
             mask_rate: float) -> dict:
    model.eval()
    total_loss = 0.0
    total_masked = 0
    total_se = 0.0

    for r in records:
        gd = _graph_to_device(r["graph"], device)
        x_orig = gd["x"]
        num_nodes = x_orig.size(0)
        if num_nodes < 2:
            continue

        mask = _create_mask(num_nodes, mask_rate).to(device)
        x_masked = x_orig.clone()
        x_masked[mask] = 0.0

        target = x_orig[mask]
        pred = model(x_masked, gd["edge_index"], gd["edge_attr"], mask)
        loss = nn.functional.mse_loss(pred, target)
        total_loss += loss.item()

        se = ((pred - target) ** 2).sum().item()
        total_se += se
        total_masked += target.numel()

    n = max(len(records), 1)
    rmse = float(np.sqrt(total_se / max(total_masked, 1)))
    return {"loss": total_loss / n, "rmse": rmse}


# -- Main -----------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_csv = Path(args.data_csv) if args.data_csv else DATA_CSV
    cache_path = Path(args.cache_path) if args.cache_path else CACHE_PT

    print(f"\n  Device : {device}")
    if args.cpu_friendly and device == "cpu":
        print("  Mode   : CPU-friendly (smaller model, fewer epochs)")
    print(f"  Objective: masked node feature reconstruction (mask_rate={cfg['mask_rate']})")

    _print_dataset_banner(data_csv)

    # -- Load data --------------------------------------------------------
    records = load_data(cache_path, data_csv)

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

    print(f"\n  Total molecules      : {len(records)}")
    print(f"  Train split          : {len(train_recs)}")
    print(f"  Validation split     : {len(val_recs)}")

    # -- Build model ------------------------------------------------------
    gnn = MoleculeGNN(
        node_feat_dim=NODE_FEAT_DIM,
        edge_feat_dim=EDGE_FEAT_DIM,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        embed_dim=cfg["embed_dim"],
    )
    model = MaskedNodeReconstructor(gnn).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters     : {n_params:,}")
    print(f"  Batch size           : {cfg['batch_size']}")
    print(f"  Max epochs           : {cfg['max_epochs']}")
    print(f"  Patience             : {cfg['patience']}")
    print(f"  Mask rate            : {cfg['mask_rate']}")

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
            model, train_recs, optimiser, device,
            cfg["batch_size"], cfg["mask_rate"],
        )
        scheduler.step()
        val_metrics = evaluate(model, val_recs, device, cfg["mask_rate"])
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

    final_train = evaluate(model, train_recs, device, cfg["mask_rate"])
    final_val = evaluate(model, val_recs, device, cfg["mask_rate"])

    print(f"\n{'-' * 60}")
    print(f"  Final Train RMSE : {final_train['rmse']:.4f}")
    print(f"  Final Val   RMSE : {final_val['rmse']:.4f}")
    print(f"  Total time       : {elapsed_total:.0f}s")
    print(f"{'-' * 60}")

    # -- Save artifacts ---------------------------------------------------
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    gnn_state = {k: v for k, v in model.state_dict().items() if k.startswith("gnn.")}
    gnn_state_clean = {k.removeprefix("gnn."): v for k, v in gnn_state.items()}
    torch.save(gnn_state_clean, ARTIFACT_DIR / "model_gnn.pt")

    with open(ARTIFACT_DIR / "config.json", "w") as f:
        json.dump({
            "gnn_config": gnn.config_dict(),
            "training_config": cfg,
            "objective": "masked_node_feature_reconstruction",
        }, f, indent=2)

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
    with open(ARTIFACT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Artifacts saved to {ARTIFACT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
