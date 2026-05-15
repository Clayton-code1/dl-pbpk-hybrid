"""Pre-compute molecular graphs from SMILES and cache to disk.

Converts every valid SMILES in a CSV into a PyTorch tensor dict via
``src.molecules.rdkit_graph.smiles_to_graph`` and saves the result as a
single file for fast training startup.

Usage (from repo root, inside the training venv):
    python scripts/cache_adme_graphs.py
    python scripts/cache_adme_graphs.py --data-csv path/to/custom.csv
    python scripts/cache_adme_graphs.py --data-csv data/processed/adme_pretrain/adme_unsupervised_sample_10k.csv \
        --output data/processed/adme_pretrain/adme_unsupervised_sample_10k_graphs.pt

Outputs (defaults):
    data/processed/adme_pretrain/adme_graphs.pt
    data/processed/adme_pretrain/adme_graphs_meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.molecules.rdkit_graph import smiles_to_graph, InvalidSMILESError

DATA_CSV = _PROJECT_ROOT / "data" / "processed" / "adme_pretrain" / "adme_supervised.csv"
OUT_DIR = _PROJECT_ROOT / "data" / "processed" / "adme_pretrain"
OUT_PT = OUT_DIR / "adme_graphs.pt"
OUT_META = OUT_DIR / "adme_graphs_meta.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute molecular graph cache")
    parser.add_argument("--data-csv", type=str, default=None,
                        help="Path to CSV with a smiles column "
                             "(default: data/processed/adme_pretrain/adme_supervised.csv)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .pt path (default: data/processed/adme_pretrain/adme_graphs.pt)")
    args = parser.parse_args()

    csv_path = Path(args.data_csv) if args.data_csv else DATA_CSV
    out_pt = Path(args.output) if args.output else OUT_PT
    out_meta = out_pt.with_suffix(".json").parent / (out_pt.stem + "_meta.json")

    print("=" * 60)
    print(" Graph Cache Builder")
    print("=" * 60)

    if not csv_path.exists():
        print(f"\nERROR: Data file not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    has_label = "label" in df.columns
    print(f"\nLoaded {len(df)} rows from {csv_path}")
    print(f"  label column: {'yes' if has_label else 'no (unsupervised)'}")

    records: list[dict] = []
    skipped_rows: list[int] = []
    t0 = time.time()

    for idx, row in df.iterrows():
        smiles = str(row.get("smiles", ""))

        label = None
        if has_label:
            try:
                label = float(row["label"])
            except (ValueError, TypeError):
                pass

        try:
            g = smiles_to_graph(smiles)
        except (InvalidSMILESError, Exception):
            skipped_rows.append(int(idx))
            continue

        record: dict = {
            "smiles": smiles,
            "x": g["x"],
            "edge_index": g["edge_index"],
            "edge_attr": g["edge_attr"],
        }
        if label is not None:
            record["label"] = label

        records.append(record)

        if (len(records) % 1000) == 0:
            print(f"  processed {len(records)} valid molecules ...")

    elapsed = time.time() - t0
    print(f"\nDone: {len(records)} valid, {len(skipped_rows)} skipped in {elapsed:.1f}s")

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, out_pt)
    print(f"Saved graph cache  -> {out_pt}")

    meta = {
        "total_rows": len(df),
        "valid_molecules": len(records),
        "skipped_rows": len(skipped_rows),
        "skipped_indices": skipped_rows[:200],
        "cache_file": str(out_pt),
        "has_labels": has_label,
        "elapsed_seconds": round(elapsed, 2),
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata     -> {out_meta}")

    print("=" * 60)


if __name__ == "__main__":
    main()
