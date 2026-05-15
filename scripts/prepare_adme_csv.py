"""Prepare a standardised ADME pretraining CSV.

Reads a user-supplied CSV (or a folder of CSVs) and outputs a two-column
file ``data/raw/adme_pretrain/adme.csv`` with columns: smiles,label.

Usage
-----
    python scripts/prepare_adme_csv.py --input path/to/raw.csv \
        --smiles-col SMILES --label-col Clearance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT_DIR = _PROJECT_ROOT / "data" / "raw" / "adme_pretrain"
_OUTPUT_FILE = _OUTPUT_DIR / "adme.csv"


def prepare(
    input_path: str,
    smiles_col: str = "smiles",
    label_col: str = "label",
) -> Path:
    """Read *input_path* and write normalised adme.csv."""
    path = Path(input_path)

    if path.is_dir():
        frames = []
        for f in sorted(path.glob("*.csv")):
            frames.append(pd.read_csv(f))
        if not frames:
            print(f"No CSV files found in {path}")
            sys.exit(1)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path)

    if smiles_col not in df.columns:
        raise KeyError(f"Column '{smiles_col}' not found. Available: {list(df.columns)}")
    if label_col not in df.columns:
        raise KeyError(f"Column '{label_col}' not found. Available: {list(df.columns)}")

    out = df[[smiles_col, label_col]].copy()
    out.columns = ["smiles", "label"]

    out = out.dropna(subset=["smiles", "label"])
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out = out.dropna(subset=["label"])
    out = out[out["label"] > 0]

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(_OUTPUT_FILE, index=False)
    print(f"Wrote {len(out)} rows to {_OUTPUT_FILE}")
    return _OUTPUT_FILE


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ADME pretraining CSV")
    parser.add_argument("--input", required=True, help="Path to CSV or folder of CSVs")
    parser.add_argument("--smiles-col", default="smiles", help="Name of SMILES column")
    parser.add_argument("--label-col", default="label", help="Name of label column")
    args = parser.parse_args()
    prepare(args.input, args.smiles_col, args.label_col)


if __name__ == "__main__":
    main()
