"""Unit tests for the molecular graph builder (src side)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.molecules.rdkit_graph import (
    smiles_to_graph,
    InvalidSMILESError,
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
)


THEOPHYLLINE = "Cn1c2c(c(=O)n(c1=O)C)[nH]cn2"


def test_valid_smiles_produces_tensors():
    g = smiles_to_graph(THEOPHYLLINE)
    assert isinstance(g["x"], torch.Tensor)
    assert isinstance(g["edge_index"], torch.Tensor)
    assert isinstance(g["edge_attr"], torch.Tensor)


def test_node_feature_dim():
    g = smiles_to_graph(THEOPHYLLINE)
    assert g["x"].shape[1] == NODE_FEAT_DIM


def test_edge_feature_dim():
    g = smiles_to_graph(THEOPHYLLINE)
    assert g["edge_attr"].shape[1] == EDGE_FEAT_DIM


def test_edge_index_shape():
    g = smiles_to_graph(THEOPHYLLINE)
    assert g["edge_index"].shape[0] == 2


def test_invalid_smiles_raises():
    with pytest.raises(InvalidSMILESError):
        smiles_to_graph("INVALID!!!")


def test_empty_smiles_raises():
    with pytest.raises(InvalidSMILESError):
        smiles_to_graph("")
