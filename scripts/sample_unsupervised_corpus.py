"""Sample a reproducible subset from the full unsupervised ChEMBL corpus.

The full corpus (~1.36M SMILES) is too large for CPU pretraining.
This script draws a random sample and writes it to a size-specific CSV.

Usage (from repo root):
    python scripts/sample_unsupervised_corpus.py
    python scripts/sample_unsupervised_corpus.py --n-samples 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = _PROJECT_ROOT / "data" / "processed" / "adme_pretrain" / "adme_unsupervised.csv"
OUTPUT_DIR = _PROJECT_ROOT / "data" / "processed" / "adme_pretrain"

DEFAULT_N = 10_000
DEFAULT_SEED = 42


def _output_filename(n: int) -> str:
    if n >= 1_000_000:
        tag = f"{n // 1_000_000}M"
    elif n >= 1_000:
        tag = f"{n // 1_000}k"
    else:
        tag = str(n)
    return f"adme_unsupervised_sample_{tag}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample unsupervised ChEMBL corpus")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N,
                        help=f"Number of molecules to sample (default: {DEFAULT_N})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--input", type=str, default=None,
                        help="Input CSV path (default: data/processed/adme_pretrain/adme_unsupervised.csv)")
    args = parser.parse_args()

    input_csv = Path(args.input) if args.input else INPUT_CSV

    print("=" * 60)
    print(" Unsupervised Corpus Sampler")
    print("=" * 60)

    if not input_csv.exists():
        print(f"\nERROR: Input not found: {input_csv}")
        print("  Run: python scripts/prepare_large_adme_corpus.py")
        raise SystemExit(1)

    df = pd.read_csv(input_csv)
    print(f"\n  Input file   : {input_csv}")
    print(f"  Input rows   : {len(df):,}")

    n = min(args.n_samples, len(df))
    sampled = df.sample(n=n, random_state=args.seed).reset_index(drop=True)

    output_name = _output_filename(n)
    output_path = OUTPUT_DIR / output_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output_path, index=False)

    print(f"  Sampled rows : {len(sampled):,}")
    print(f"  Random seed  : {args.seed}")
    print(f"  Output file  : {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
