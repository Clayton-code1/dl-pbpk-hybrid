"""Prepare a larger ADME pretraining corpus from local datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DELaney_PATH = PROJECT_ROOT / "data" / "raw" / "adme_pretrain" / "delaney-processed.csv"
LIPOPHILICITY_PATH = PROJECT_ROOT / "data" / "raw" / "adme_pretrain" / "Lipophilicity.csv"
CHEMBL_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "adme_pretrain" / "train.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "adme_pretrain"
SUPERVISED_OUT = OUTPUT_DIR / "adme_supervised.csv"
UNSUPERVISED_OUT = OUTPUT_DIR / "adme_unsupervised.csv"
SUMMARY_OUT = OUTPUT_DIR / "dataset_summary.json"


def _resolve_input(path: Path) -> Path:
    if path.exists():
        return path
    for ext in (".csv", ".tsv", ".txt"):
        candidate = Path(f"{path}{ext}")
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Input not found: {path}")


def _find_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    normalized = {c.strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    if required:
        raise KeyError(f"Missing required column. Tried {candidates}. Available: {list(df.columns)}")
    return None


def _clean_smiles(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    invalid_tokens = {"", "nan", "none", "null"}
    return cleaned.mask(cleaned.str.lower().isin(invalid_tokens))


def _filter_invalid_smiles(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    # Without a reliable RDKit in all environments, treat "invalid" as
    # trivially empty / whitespace-only SMILES. If you run this inside
    # a chemistry-ready environment, you can tighten this by adding
    # RDKit-based validation here.
    before = len(df)
    out = df[df["smiles"].astype(str).str.strip() != ""].copy()
    removed = before - len(out)
    return out, int(removed)


def _prepare_esol(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(path)
    smiles_col = _find_column(df, ["smiles", "SMILES"])
    label_col = _find_column(
        df,
        [
            "measured log solubility in mols per litre",
            "label",
            "target",
            "y",
        ],
    )
    out = pd.DataFrame(
        {
            "smiles": _clean_smiles(df[smiles_col]),
            "label": pd.to_numeric(df[label_col], errors="coerce"),
            "task": "esol",
        }
    )
    return out, {"source_file": str(path), "source_columns": list(df.columns)}


def _prepare_lipophilicity(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(path)
    smiles_col = _find_column(df, ["smiles", "SMILES"])
    label_col = _find_column(df, ["exp", "label", "target", "y"])
    out = pd.DataFrame(
        {
            "smiles": _clean_smiles(df[smiles_col]),
            "label": pd.to_numeric(df[label_col], errors="coerce"),
            "task": "lipophilicity",
        }
    )
    return out, {"source_file": str(path), "source_columns": list(df.columns)}


def _prepare_chembl_unsupervised(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(path)
    smiles_col = _find_column(df, ["smiles", "SMILES"])
    out = pd.DataFrame(
        {
            "smiles": _clean_smiles(df[smiles_col]),
            "label": pd.Series([pd.NA] * len(df), dtype="object"),
            "task": "unsupervised_chemistry",
        }
    )
    return out, {"source_file": str(path), "source_columns": list(df.columns)}


def _drop_missing_and_dedupe(df: pd.DataFrame, require_numeric_label: bool) -> tuple[pd.DataFrame, dict[str, int]]:
    before = len(df)
    out = df.dropna(subset=["smiles"]).copy()
    removed_empty_smiles = before - len(out)

    if require_numeric_label:
        before_labels = len(out)
        out = out.dropna(subset=["label"]).copy()
        removed_missing_label = before_labels - len(out)
    else:
        removed_missing_label = 0

    before_dedup = len(out)
    out = out.drop_duplicates(subset=["smiles"], keep="first").copy()
    removed_duplicates = before_dedup - len(out)

    out, removed_invalid_smiles = _filter_invalid_smiles(out)

    return out, {
        "removed_empty_smiles": int(removed_empty_smiles),
        "removed_missing_label": int(removed_missing_label),
        "removed_duplicates_by_smiles": int(removed_duplicates),
        "removed_invalid_smiles": int(removed_invalid_smiles),
        "final_rows": int(len(out)),
    }


def prepare_corpus() -> dict[str, Any]:
    esol_path = _resolve_input(DELaney_PATH)
    lipophilicity_path = _resolve_input(LIPOPHILICITY_PATH)
    chembl_path = _resolve_input(CHEMBL_TRAIN_PATH)

    esol_df, esol_meta = _prepare_esol(esol_path)
    lip_df, lip_meta = _prepare_lipophilicity(lipophilicity_path)
    chembl_df, chembl_meta = _prepare_chembl_unsupervised(chembl_path)

    supervised = pd.concat([esol_df, lip_df], ignore_index=True)
    supervised, supervised_stats = _drop_missing_and_dedupe(supervised, require_numeric_label=True)

    unsupervised, unsupervised_stats = _drop_missing_and_dedupe(chembl_df, require_numeric_label=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    supervised.to_csv(SUPERVISED_OUT, index=False)
    unsupervised.to_csv(UNSUPERVISED_OUT, index=False)

    summary: dict[str, Any] = {
        "inputs": {
            "esol": esol_meta,
            "lipophilicity": lip_meta,
            "chembl_train": chembl_meta,
        },
        "outputs": {
            "supervised_file": str(SUPERVISED_OUT),
            "unsupervised_file": str(UNSUPERVISED_OUT),
            "summary_file": str(SUMMARY_OUT),
        },
        "supervised": {
            "tasks": {
                "esol_rows_raw": int(len(esol_df)),
                "lipophilicity_rows_raw": int(len(lip_df)),
            },
            **supervised_stats,
        },
        "unsupervised": {
            "task": "unsupervised_chemistry",
            "rows_raw": int(len(chembl_df)),
            **unsupervised_stats,
        },
    }

    with SUMMARY_OUT.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare large ADME corpus from local datasets")
    _ = parser.parse_args()
    summary = prepare_corpus()

    print("Prepared ADME corpus successfully.")
    print(f"Supervised output: {SUPERVISED_OUT}")
    print(f"Unsupervised output: {UNSUPERVISED_OUT}")
    print(f"Summary output: {SUMMARY_OUT}")
    print("Counts:")
    print(json.dumps({"supervised": summary["supervised"], "unsupervised": summary["unsupervised"]}, indent=2))
    print("\nRun these exact commands:")
    print("python scripts/inspect_local_datasets.py")
    print("python scripts/prepare_large_adme_corpus.py")


if __name__ == "__main__":
    main()
