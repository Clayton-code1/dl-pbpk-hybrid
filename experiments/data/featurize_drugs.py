"""Phase 1.4 - Molecular featurisation for the multi-drug DL-PBPK study.

For every drug in the panel (training + external) we compute:

1. Engineered RDKit descriptors used as auxiliary inputs for baseline /
   hybrid models:

      MW, logP, TPSA, n_HBD, n_HBA, n_rotatable, n_rings, fsp3

2. The molecular graph object (atom features, bond features) produced by
   ``src.molecules.rdkit_graph.smiles_to_graph``.  These tensors are
   serialised to ``.pt`` files and reused at training time so the GNN
   doesn't recompute the SMILES parse each epoch.

Outputs
-------
- ``experiments/data/processed/drug_molecular_features.csv``
- ``experiments/data/processed/graphs/{drug}.pt`` per drug

Run from project root:

    python -m experiments.data.featurize_drugs
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.config import (  # noqa: E402
    DRUGS,
    EXTERNAL_DRUG,
    PROCESSED_DATA_DIR,
    ensure_dirs,
    get_logger,
)
from experiments.reference_pk import REFERENCE_PK_DATA  # noqa: E402
from src.molecules.rdkit_graph import (  # noqa: E402
    EDGE_FEAT_DIM,
    NODE_FEAT_DIM,
    smiles_to_graph,
)

LOGGER = get_logger("phase1.featurize_drugs", "phase1_featurize_drugs.log")

GRAPHS_DIR = PROCESSED_DATA_DIR / "graphs"


# ---------------------------------------------------------------------------
# RDKit descriptor calculation
# ---------------------------------------------------------------------------

def _descriptors(smiles: str) -> dict[str, float]:
    """Compute the 8 engineered descriptors used by baseline / hybrid models."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    return {
        "MW": float(Descriptors.MolWt(mol)),
        "logP": float(Crippen.MolLogP(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "n_HBD": int(Lipinski.NumHDonors(mol)),
        "n_HBA": int(Lipinski.NumHAcceptors(mol)),
        "n_rotatable": int(Lipinski.NumRotatableBonds(mol)),
        "n_rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "fsp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _process_drug(drug: str) -> dict[str, Any]:
    ref = REFERENCE_PK_DATA[drug]
    smiles = ref["smiles"]
    LOGGER.info("--- %s | SMILES=%s", drug, smiles)

    descriptors = _descriptors(smiles)

    graph = smiles_to_graph(smiles)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "drug": drug,
            "smiles": smiles,
            "x": graph["x"],
            "edge_index": graph["edge_index"],
            "edge_attr": graph["edge_attr"],
            "node_feat_dim": NODE_FEAT_DIM,
            "edge_feat_dim": EDGE_FEAT_DIM,
        },
        GRAPHS_DIR / f"{drug}.pt",
    )
    LOGGER.info(
        "    atoms=%d  edges=%d  desc=%s",
        graph["x"].shape[0], graph["edge_index"].shape[1],
        {k: round(v, 3) if isinstance(v, float) else v for k, v in descriptors.items()},
    )

    return {
        "drug": drug,
        "smiles": smiles,
        **descriptors,
        "n_atoms": int(graph["x"].shape[0]),
        "n_edges": int(graph["edge_index"].shape[1]),
        "graph_path": str(
            (GRAPHS_DIR / f"{drug}.pt").relative_to(_PROJECT_ROOT)
        ),
    }


def main() -> int:
    ensure_dirs()
    LOGGER.info("Phase 1.4 - molecular featurisation")
    LOGGER.info("NODE_FEAT_DIM=%d  EDGE_FEAT_DIM=%d", NODE_FEAT_DIM, EDGE_FEAT_DIM)

    rows: list[dict[str, Any]] = []
    for drug in list(DRUGS) + [EXTERNAL_DRUG]:
        rows.append(_process_drug(drug))

    df = pd.DataFrame(rows)
    out_csv = PROCESSED_DATA_DIR / "drug_molecular_features.csv"
    df.to_csv(out_csv, index=False)
    LOGGER.info("Saved %s (%d rows)", out_csv.relative_to(_PROJECT_ROOT), len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
