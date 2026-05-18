"""Fine-tune a panel hybrid checkpoint using rows from ``feedback_log.csv``.

Loads ``hybrid_gnn_pbpk_{drug}_v1``, runs a short supervised loop on observed
concentrations, and writes a new artifact directory plus ``finetune_report.json``.

Example::

    python experiments/training/finetune_from_feedback.py --drug theophylline --dry-run
    python experiments/training/finetune_from_feedback.py --drug theophylline
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from experiments.models.hybrid_multidrug import MultiDrugHybridConfig, MultiDrugHybridGNNPBPK  # noqa: E402
from experiments.reference_pk import REFERENCE_PK_DATA  # noqa: E402
from experiments.training.multidrug_utils import (  # noqa: E402
    load_drug_graph,
    patient_feature_columns,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune hybrid from feedback CSV")
    p.add_argument("--drug", type=str, required=True)
    p.add_argument("--min-feedback-points", type=int, default=20)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output-suffix", type=str, default="v_finetuned")
    p.add_argument(
        "--feedback-csv",
        type=str,
        default="",
        help="Override path (default: experiments/data/feedback/feedback_log.csv)",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _feedback_path(project: Path, override: str) -> Path:
    if override:
        return Path(override)
    return project / "experiments" / "data" / "feedback" / "feedback_log.csv"


def _load_groups(csv_path: Path, drug: str) -> dict[int, list[dict[str, str]]]:
    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("drug", "").lower() != drug.lower():
                continue
            pid = int(row["patient_id"])
            groups[pid].append(row)
    for pid in groups:
        groups[pid].sort(key=lambda r: float(r["t_hours"]))
    return dict(groups)


def _row_covariates(r0: dict[str, str], feat_names: list[str]) -> np.ndarray:
    """Build raw feature vector matching ``scaler.json`` column order."""
    w = max(float(r0["weight_kg"]), 1e-6)
    d = float(r0["dose_mg"])
    extra = {
        "dose_mgkg": d / w,
        "dose_mg_per_kg": d / w,
        "log_dose_mg_per_kg": math.log(d / w + 1e-8),
    }
    vec: list[float] = []
    for c in feat_names:
        if c in r0 and r0[c] != "":
            vec.append(float(r0[c]))
        elif c in extra:
            vec.append(float(extra[c]))
        else:
            vec.append(0.0)
    return np.array(vec, dtype=np.float32)


def run_finetune(
    drug: str,
    *,
    project_root: Path,
    feedback_csv: Path,
    min_points: int,
    epochs: int,
    lr: float,
    suffix: str,
    dry_run: bool,
) -> None:
    base_dir = project_root / "artifacts" / "models" / f"hybrid_gnn_pbpk_{drug}_v1"
    if not (base_dir / "model.pt").exists():
        raise FileNotFoundError(f"Missing base checkpoint: {base_dir}")

    groups = _load_groups(feedback_csv, drug)
    n_points = sum(len(v) for v in groups.values())
    if n_points < min_points:
        print(
            f"FAILED finetune_from_feedback: only {n_points} feedback rows for {drug}; "
            f"need >= {min_points}.",
            flush=True,
        )
        return

    if dry_run:
        print(
            f"finetune dry-run OK — drug={drug} rows={n_points} patients={len(groups)} "
            f"-> would write hybrid_gnn_pbpk_{drug}_{suffix}",
            flush=True,
        )
        return

    out_dir = project_root / "artifacts" / "models" / f"hybrid_gnn_pbpk_{drug}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    scaler = json.loads((base_dir / "scaler.json").read_text(encoding="utf-8"))
    feat_names_disk: list[str] = list(scaler.get("feature_names", patient_feature_columns(drug)))
    mean = np.array(scaler["mean"], dtype=np.float32)
    std = np.array(scaler["std"], dtype=np.float32)

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
    state = torch.load(base_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.train()

    graph = load_drug_graph(drug)
    gx = graph["x"].float()
    gei = graph["edge_index"].long()
    gea = graph["edge_attr"].float()

    f_bio = float(REFERENCE_PK_DATA[drug]["F"])

    # Build patient batches: (pf, times, obs, dose_mg, weight_kg, f_bio_tensor)
    batches: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = []
    for _pid, rows in groups.items():
        r0 = rows[0]
        raw_vec = _row_covariates(r0, feat_names_disk)
        z = (raw_vec - mean) / std
        t_h = torch.tensor([float(r["t_hours"]) for r in rows], dtype=torch.float32)
        obs = torch.tensor([float(r["observed_mg_per_L"]) for r in rows], dtype=torch.float32)
        dose_mg = torch.tensor(float(r0["dose_mg"]), dtype=torch.float32)
        w_kg = torch.tensor(float(r0["weight_kg"]), dtype=torch.float32)
        pf = torch.tensor(z, dtype=torch.float32)
        f_bt = torch.tensor(f_bio, dtype=torch.float32)
        batches.append((pf, t_h, obs, dose_mg, w_kg, f_bt))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    def eval_rmse() -> float:
        model.eval()
        se: list[float] = []
        with torch.no_grad():
            for pf, t_h, obs, dose_mg, w_kg, f_bt in batches:
                pred, *_ = model(gx, gei, gea, pf, t_h, dose_mg, w_kg, f_bt)
                se.append(float(torch.mean((pred - obs) ** 2).item()))
        model.train()
        return float(np.sqrt(np.mean(se))) if se else 0.0

    rmse_before = eval_rmse()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for pf, t_h, obs, dose_mg, w_kg, f_bt in batches:
            optimizer.zero_grad(set_to_none=True)
            pred, *_ = model(gx, gei, gea, pf, t_h, dose_mg, w_kg, f_bt)
            loss = loss_fn(pred, obs)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        print(f"  epoch {epoch + 1}/{epochs}  mean_batch_loss={epoch_loss / max(len(batches), 1):.6f}", flush=True)

    rmse_after = eval_rmse()
    model.eval()
    torch.save(model.state_dict(), out_dir / "model.pt")
    shutil.copy2(base_dir / "config.json", out_dir / "config.json")
    shutil.copy2(base_dir / "scaler.json", out_dir / "scaler.json")
    if (base_dir / "metrics.json").exists():
        shutil.copy2(base_dir / "metrics.json", out_dir / "metrics.json")

    report = {
        "drug": drug,
        "base_dir": str(base_dir),
        "output_dir": str(out_dir),
        "n_feedback_rows": n_points,
        "n_patients": len(groups),
        "epochs": epochs,
        "lr": lr,
        "rmse_mg_per_L_before": rmse_before,
        "rmse_mg_per_L_after": rmse_after,
    }
    (out_dir / "finetune_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"finetune OK — wrote {out_dir}", flush=True)

    try:
        if str(project_root / "api") not in sys.path:
            sys.path.insert(0, str(project_root / "api"))
        from app.services.model_registry import register_finetuned_model  # type: ignore

        register_finetuned_model(drug, out_dir, {"feedback_rmse_before": rmse_before, "feedback_rmse_after": rmse_after})
    except Exception as exc:
        print(f"WARNING: model_registry update skipped: {exc}", flush=True)


def main() -> int:
    args = _parse_args()
    project = _HERE
    fb = _feedback_path(project, args.feedback_csv)
    try:
        run_finetune(
            args.drug.strip().lower(),
            project_root=project,
            feedback_csv=fb,
            min_points=args.min_feedback_points,
            epochs=args.epochs,
            lr=args.lr,
            suffix=args.output_suffix.strip().strip("_") or "v_finetuned",
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"FAILED finetune_from_feedback: {exc}", flush=True)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
