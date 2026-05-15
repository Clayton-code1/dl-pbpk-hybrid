"""Unit tests for SMILES-to-graph conversion."""

from __future__ import annotations

import pytest
import torch

from app.services.rdkit_graph import (
    smiles_to_graph,
    InvalidSMILESError,
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
)

THEOPHYLLINE = "Cn1c2c(c(=O)n(c1=O)C)[nH]cn2"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ETHANOL = "CCO"


class TestValidSMILES:
    def test_theophylline_shape(self):
        g = smiles_to_graph(THEOPHYLLINE)
        assert g["x"].ndim == 2
        assert g["x"].shape[1] == NODE_FEAT_DIM
        assert g["edge_index"].shape[0] == 2
        assert g["edge_attr"].shape[1] == EDGE_FEAT_DIM
        assert g["x"].shape[0] > 0

    def test_aspirin_shape(self):
        g = smiles_to_graph(ASPIRIN)
        assert g["x"].shape[0] > 0
        assert g["edge_index"].shape[1] == g["edge_attr"].shape[0]

    def test_ethanol_small_molecule(self):
        g = smiles_to_graph(ETHANOL)
        assert g["x"].shape[0] >= 3  # C, C, O (+ hydrogens after AddHs)

    def test_edge_index_bidirectional(self):
        g = smiles_to_graph(ETHANOL)
        assert g["edge_index"].shape[1] % 2 == 0

    def test_dtypes(self):
        g = smiles_to_graph(THEOPHYLLINE)
        assert g["x"].dtype == torch.float32
        assert g["edge_index"].dtype == torch.long
        assert g["edge_attr"].dtype == torch.float32

    def test_node_features_reasonable(self):
        g = smiles_to_graph(THEOPHYLLINE)
        assert g["x"].min() >= -2.0
        assert g["x"].max() <= 1.0


class TestInvalidSMILES:
    def test_garbage_string(self):
        with pytest.raises(InvalidSMILESError):
            smiles_to_graph("not_a_smiles_at_all!!!")

    def test_empty_string(self):
        with pytest.raises(InvalidSMILESError):
            smiles_to_graph("")

    def test_partial_smiles(self):
        with pytest.raises(InvalidSMILESError):
            smiles_to_graph("C(C)(")
