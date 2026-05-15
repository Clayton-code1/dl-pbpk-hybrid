"""Convert a SMILES string to a molecular graph suitable for GNN consumption.

Minimal API-side copy of src/molecules/rdkit_graph.py so the API does not
depend on importing ``src``.  Kept in sync manually.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from rdkit import Chem
from rdkit.Chem import rdchem

_ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]
_ATOM_MAP = {a: i for i, a in enumerate(_ATOM_TYPES)}
_NUM_ATOM_TYPES = len(_ATOM_TYPES) + 1

_MAX_DEGREE = 5
_MAX_NUM_H = 4

_HYBRIDISATION_MAP = {
    rdchem.HybridizationType.SP: 0,
    rdchem.HybridizationType.SP2: 1,
    rdchem.HybridizationType.SP3: 2,
}
_NUM_HYBRID = len(_HYBRIDISATION_MAP) + 1

NODE_FEAT_DIM = _NUM_ATOM_TYPES + (_MAX_DEGREE + 1) + 1 + 1 + _NUM_HYBRID + (_MAX_NUM_H + 1)

_BOND_TYPE_MAP = {
    rdchem.BondType.SINGLE: 0,
    rdchem.BondType.DOUBLE: 1,
    rdchem.BondType.TRIPLE: 2,
    rdchem.BondType.AROMATIC: 3,
}
_NUM_BOND_TYPES = len(_BOND_TYPE_MAP)

EDGE_FEAT_DIM = _NUM_BOND_TYPES + 1 + 1


class InvalidSMILESError(ValueError):
    """Raised when a SMILES string cannot be parsed by RDKit."""


def _one_hot(idx: int, length: int) -> list[float]:
    vec = [0.0] * length
    if 0 <= idx < length:
        vec[idx] = 1.0
    return vec


def _atom_features(atom: rdchem.Atom) -> list[float]:
    symbol = atom.GetSymbol()
    atom_idx = _ATOM_MAP.get(symbol, len(_ATOM_TYPES))
    feats: list[float] = []
    feats.extend(_one_hot(atom_idx, _NUM_ATOM_TYPES))
    feats.extend(_one_hot(min(atom.GetTotalDegree(), _MAX_DEGREE), _MAX_DEGREE + 1))
    feats.append(float(max(-2, min(2, atom.GetFormalCharge()))))
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    hyb = _HYBRIDISATION_MAP.get(atom.GetHybridization(), _NUM_HYBRID - 1)
    feats.extend(_one_hot(hyb, _NUM_HYBRID))
    feats.extend(_one_hot(min(atom.GetTotalNumHs(), _MAX_NUM_H), _MAX_NUM_H + 1))
    return feats


def _bond_features(bond: rdchem.Bond) -> list[float]:
    bt = _BOND_TYPE_MAP.get(bond.GetBondType(), 0)
    feats: list[float] = _one_hot(bt, _NUM_BOND_TYPES)
    feats.append(1.0 if bond.GetIsConjugated() else 0.0)
    feats.append(1.0 if bond.IsInRing() else 0.0)
    return feats


def smiles_to_graph(smiles: str) -> dict[str, Tensor]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InvalidSMILESError(f"RDKit could not parse SMILES: '{smiles}'")
    mol = Chem.AddHs(mol)
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        raise InvalidSMILESError(f"SMILES produced empty molecule: '{smiles}'")

    x = [_atom_features(mol.GetAtomWithIdx(i)) for i in range(num_atoms)]
    src_list: list[int] = []
    dst_list: list[int] = []
    edge_feats: list[list[float]] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        src_list.extend([i, j])
        dst_list.extend([j, i])
        edge_feats.extend([bf, bf])

    x_tensor = torch.tensor(x, dtype=torch.float32)
    if len(src_list) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float32)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)

    return {"x": x_tensor, "edge_index": edge_index, "edge_attr": edge_attr}
