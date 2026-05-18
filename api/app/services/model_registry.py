"""Versioned registry for hybrid checkpoints (manual promotion of fine-tuned runs)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "state" / "model_registry.json"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _default_registry() -> dict[str, Any]:
    return {"models": {}, "updated_at": None}


def _load() -> dict[str, Any]:
    if not _REGISTRY_PATH.exists():
        return _default_registry()
    with open(_REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict[str, Any]) -> None:
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def current_model_path(drug: str) -> str:
    """Preferred checkpoint directory for *drug* (defaults to ``hybrid_gnn_pbpk_{drug}_v1``)."""
    data = _load()
    entry = data.get("models", {}).get(drug.lower())
    if entry and entry.get("active"):
        return str(entry["active"])
    rel = f"artifacts/models/hybrid_gnn_pbpk_{drug.lower()}_v1"
    return str((_PROJECT_ROOT / rel).resolve())


def register_finetuned_model(
    drug: str,
    new_path: str | Path,
    validation_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Append a candidate checkpoint; does **not** switch ``active``."""
    drug = drug.lower()
    data = _load()
    data.setdefault("models", {})
    entry = data["models"].setdefault(
        drug,
        {"active": None, "candidates": []},
    )
    if entry["active"] is None:
        entry["active"] = str((_PROJECT_ROOT / f"artifacts/models/hybrid_gnn_pbpk_{drug}_v1").resolve())
    rec = {
        "path": str(Path(new_path).resolve()),
        "validation_metrics": validation_metrics,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    entry.setdefault("candidates", []).append(rec)
    _save(data)
    return rec
