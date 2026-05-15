"""Inspect local molecular datasets safely.

Prints, for each discovered dataset file:
- filename
- number of rows
- column names
- first 3 rows
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TARGETS = [
    "data/raw/adme_pretrain/delaney-processed",
    "data/raw/adme_pretrain/Lipophilicity",
    "data/raw/adme_pretrain/train",
    "data/raw/reference/drugbank_all_drugbank_vocabulary",
    "data/raw/reference/CID-SMILES",
]

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".smi", ".smiles", ""}


def _resolve_target_path(target: str) -> Path:
    base = PROJECT_ROOT / target
    if base.exists():
        return base

    for ext in (".csv", ".tsv", ".txt"):
        candidate = Path(f"{base}{ext}")
        if candidate.exists():
            return candidate

    return base


def _discover_files(targets: list[str]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        path = _resolve_target_path(target)
        if path.is_file():
            files.append(path)
            continue
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.is_file()))
            continue
        print(f"[WARN] Target not found: {target} -> {path}")
    return files


def _looks_numeric(token: str) -> bool:
    value = token.strip()
    if value == "":
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


def _detect_dialect(path: Path) -> csv.Dialect:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        sample = handle.read(65536)
    sniffer = csv.Sniffer()
    try:
        return sniffer.sniff(sample, delimiters=",\t;|")
    except csv.Error:
        class _Fallback(csv.Dialect):
            delimiter = ","
            quotechar = '"'
            escapechar = None
            doublequote = True
            skipinitialspace = False
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL

        return _Fallback()


def _detect_header(sample: str, first_row: list[str], second_row: list[str] | None) -> bool:
    try:
        if csv.Sniffer().has_header(sample):
            # Guard against false positives where first row starts with numeric ID
            # and second column looks like a SMILES string (CID-SMILES style).
            if first_row and _looks_numeric(first_row[0]):
                return False
            return True
    except csv.Error:
        pass

    lowered = [c.strip().lower() for c in first_row]
    known_header_tokens = (
        "smiles",
        "label",
        "target",
        "compound",
        "id",
        "inchi",
        "name",
        "cas",
        "exp",
    )
    if any(any(tok in cell for tok in known_header_tokens) for cell in lowered):
        return True

    first_numeric_count = sum(_looks_numeric(v) for v in first_row)
    second_numeric_count = sum(_looks_numeric(v) for v in (second_row or []))
    return first_numeric_count == 0 and second_numeric_count > 0


def inspect_file(path: Path) -> dict[str, Any]:
    columns: list[str] = []
    first_rows: list[list[str]] = []
    row_count = 0

    dialect = _detect_dialect(path)

    with path.open("r", encoding="utf-8", errors="replace", newline="") as probe_handle:
        sample = probe_handle.read(65536)
    probe_reader = csv.reader(sample.splitlines(), dialect=dialect)
    probe_rows = list(probe_reader)
    if not probe_rows:
        return {
            "filename": str(path),
            "rows": 0,
            "columns": [],
            "first_3_rows": [],
        }

    first_row = [cell.strip() for cell in probe_rows[0]]
    second_row = [cell.strip() for cell in probe_rows[1]] if len(probe_rows) > 1 else None
    has_header = _detect_header(sample, first_row, second_row)

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, dialect=dialect)
        first_data_consumed = False
        for idx, row in enumerate(reader):
            cleaned_row = [cell.strip() for cell in row]
            if idx == 0:
                if has_header:
                    columns = cleaned_row
                    continue
                columns = [f"col_{i + 1}" for i in range(len(cleaned_row))]
            row_count += 1
            if len(first_rows) < 3:
                first_rows.append(cleaned_row)
            first_data_consumed = True

        if not first_data_consumed and not has_header and first_row:
            # Handle one-line files that contain data but no header.
            row_count = 1
            first_rows = [first_row]

    return {
        "filename": str(path),
        "rows": row_count,
        "columns": columns,
        "first_3_rows": first_rows[:3],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local datasets safely")
    parser.add_argument(
        "--targets",
        nargs="*",
        default=DEFAULT_TARGETS,
        help="Targets to inspect (files or directories).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    args = parser.parse_args()

    all_files = _discover_files(args.targets)
    all_files = [p for p in all_files if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    all_files = sorted(set(all_files))

    reports = [inspect_file(path) for path in all_files]

    if args.json:
        print(json.dumps(reports, indent=2))
        return

    for report in reports:
        print("=" * 100)
        print(f"filename: {report['filename']}")
        print(f"number_of_rows: {report['rows']}")
        print(f"column_names: {report['columns']}")
        print("first_3_rows:")
        for idx, row in enumerate(report["first_3_rows"], start=1):
            print(f"  {idx}: {row}")


if __name__ == "__main__":
    main()
