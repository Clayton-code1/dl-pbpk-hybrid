"""Unit tests for the pure-torch MoleculeGNN."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.molecules.rdkit_graph import smiles_to_graph, NODE_FEAT_DIM, EDGE_FEAT_DIM
from src.models.gnn.molecule_gnn import MoleculeGNN


THEOPHYLLINE = "Cn1c2c(c(=O)n(c1=O)C)[nH]cn2"


def _make_gnn(embed_dim: int = 128) -> MoleculeGNN:
    return MoleculeGNN(
        node_feat_dim=NODE_FEAT_DIM,
        edge_feat_dim=EDGE_FEAT_DIM,
        hidden_dim=64,
        num_layers=2,
        embed_dim=embed_dim,
    )


def test_forward_produces_correct_shape():
    gnn = _make_gnn(embed_dim=128)
    g = smiles_to_graph(THEOPHYLLINE)
    emb = gnn(g["x"], g["edge_index"], g["edge_attr"])
    assert emb.shape == (128,)


def test_output_is_differentiable():
    gnn = _make_gnn()
    g = smiles_to_graph(THEOPHYLLINE)
    emb = gnn(g["x"], g["edge_index"], g["edge_attr"])
    loss = emb.sum()
    loss.backward()
    for p in gnn.parameters():
        assert p.grad is not None


def test_different_molecules_give_different_embeddings():
    gnn = _make_gnn()
    g1 = smiles_to_graph("CCO")
    g2 = smiles_to_graph(THEOPHYLLINE)
    with torch.no_grad():
        e1 = gnn(g1["x"], g1["edge_index"], g1["edge_attr"])
        e2 = gnn(g2["x"], g2["edge_index"], g2["edge_attr"])
    assert not torch.allclose(e1, e2)
