"""Global configuration for the multi-drug DL-PBPK research experiments.

All scripts in `experiments/` should import paths and reproducibility
constants from this module so that hard-coded directories never drift
between phases.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42

# Journal Phase 1.1 layout: string paths relative to ``PROJECT_ROOT`` / repo root
# (``dl-pbpk-hybrid``).  Code uses the ``*_DIR`` :class:`Path` objects below.
RESULTS_DIR_STR = "experiments/results/"
PLOTS_DIR_STR = "experiments/plots/"
LOGS_DIR_STR = "experiments/logs/"
REPORTS_DIR_STR = "experiments/reports/"
MODELS_DIR_STR = "artifacts/models/"

# ---------------------------------------------------------------------------
# Drug panel
# ---------------------------------------------------------------------------

# Canonical training drugs (Phase 1).  Theophylline is the original benchmark.
DRUGS: list[str] = [
    "theophylline",
    "warfarin",
    "midazolam",
    "caffeine",
    "acetaminophen",
    "digoxin",
]

# Held-out compound for zero-shot external validation in Phase 2.4.
EXTERNAL_DRUG: str = "ibuprofen"

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------

# Project root = parent of `experiments/`, i.e. the dl-pbpk-hybrid folder
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
DATA_DIR: Path = EXPERIMENTS_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
RESULTS_DIR: Path = EXPERIMENTS_DIR / "results"
PLOTS_DIR: Path = EXPERIMENTS_DIR / "plots"
LOGS_DIR: Path = EXPERIMENTS_DIR / "logs"
REPORTS_DIR: Path = EXPERIMENTS_DIR / "reports"

MODELS_DIR: Path = PROJECT_ROOT / "artifacts" / "models"
# Fine-tuned theophylline hybrid (preferred transfer source per Phase 1 spec)
HYBRID_THEOPH_COMBINED_DIR: Path = MODELS_DIR / "hybrid_gnn_pbpk_theoph_combined_v1"
HYBRID_THEOPH_COMBINED_WEIGHTS: Path = HYBRID_THEOPH_COMBINED_DIR / "model.pt"
PRETRAINED_GNN_DIR: Path = MODELS_DIR / "gnn_pretrain_combined_v1"
PRETRAINED_GNN_WEIGHTS: Path = PRETRAINED_GNN_DIR / "model_gnn.pt"
PRETRAINED_GNN_CONFIG: Path = PRETRAINED_GNN_DIR / "config.json"


def ensure_dirs() -> None:
    """Create every directory referenced by the config (idempotent)."""
    for d in [
        RAW_DATA_DIR, PROCESSED_DATA_DIR,
        PROCESSED_DATA_DIR / "graphs",
        RESULTS_DIR, PLOTS_DIR, LOGS_DIR, REPORTS_DIR,
        MODELS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Reproducibility helper
# ---------------------------------------------------------------------------

def seed_everything(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch RNGs.

    PyTorch is imported lazily so that lightweight scripts (e.g. data
    download) do not pay the import cost.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def get_logger(name: str, log_filename: str | None = None) -> logging.Logger:
    """Return a logger that writes to console and (optionally) ``logs/<file>``."""
    ensure_dirs()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_filename is not None:
        fh = logging.FileHandler(LOGS_DIR / log_filename, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
