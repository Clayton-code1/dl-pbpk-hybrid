"""Inline GNN model definition for API inference.

Re-defines the architecture so the API does not depend on importing ``src/``.
Must stay in sync with ``src/models/gnn/molecule_gnn.py`` and
``src/models/hybrid_gnn_pbpk.py``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class _EdgeMLP(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, msg_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, msg_dim),
            nn.ReLU(),
            nn.Linear(msg_dim, msg_dim),
        )

    def forward(self, h_src: Tensor, h_dst: Tensor, edge_attr: Tensor) -> Tensor:
        return self.mlp(torch.cat([h_src, h_dst, edge_attr], dim=-1))


class _MPLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.edge_mlp = _EdgeMLP(hidden_dim, edge_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, h: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        if edge_index.size(1) == 0:
            return h
        src, dst = edge_index[0], edge_index[1]
        msgs = self.edge_mlp(h[src], h[dst], edge_attr)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msgs)
        return self.gru(agg, h)


class _MoleculeGNN(nn.Module):
    def __init__(self, node_feat_dim: int, edge_feat_dim: int, hidden_dim: int, num_layers: int, embed_dim: int):
        super().__init__()
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.layers = nn.ModuleList([_MPLayer(hidden_dim, edge_feat_dim) for _ in range(num_layers)])
        self.readout = nn.Linear(2 * hidden_dim, embed_dim)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        h = self.node_encoder(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        pooled = torch.cat([h.mean(dim=0), h.max(dim=0).values], dim=-1)
        return self.readout(pooled)


class InlineHybridGNNPBPK(nn.Module):
    """API-side mirror of HybridGNNPBPK for inference only."""

    def __init__(
        self,
        node_feat_dim: int = 27,
        edge_feat_dim: int = 6,
        gnn_hidden: int = 128,
        gnn_layers: int = 3,
        gnn_embed_dim: int = 128,
        patient_feat_dim: int = 3,
        head_hidden: int = 64,
        cl_floor: float = 0.1,
        ka_floor: float = 0.05,
    ):
        super().__init__()
        self.cl_floor = cl_floor
        self.ka_floor = ka_floor

        self.gnn = _MoleculeGNN(node_feat_dim, edge_feat_dim, gnn_hidden, gnn_layers, gnn_embed_dim)

        self.head = nn.Sequential(
            nn.Linear(gnn_embed_dim + patient_feat_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 2),
        )

    def get_drug_embedding(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        return self.gnn(x, edge_index, edge_attr)

    def predict_pk_params(self, drug_emb: Tensor, patient_features: Tensor) -> tuple[Tensor, Tensor]:
        combined = torch.cat([drug_emb, patient_features], dim=-1)
        raw = self.head(combined)
        params = torch.exp(raw)
        CL = params[0].clamp(min=self.cl_floor)
        ka = params[1].clamp(min=self.ka_floor)
        return CL, ka


def build_gnn_model(config: dict[str, Any]) -> InlineHybridGNNPBPK:
    return InlineHybridGNNPBPK(
        node_feat_dim=int(config.get("node_feat_dim", 27)),
        edge_feat_dim=int(config.get("edge_feat_dim", 6)),
        gnn_hidden=int(config.get("gnn_hidden", 128)),
        gnn_layers=int(config.get("gnn_layers", 3)),
        gnn_embed_dim=int(config.get("gnn_embed_dim", 128)),
        patient_feat_dim=int(config.get("patient_feat_dim", 3)),
        head_hidden=int(config.get("head_hidden", 64)),
        cl_floor=float(config.get("cl_floor", 0.1)),
        ka_floor=float(config.get("ka_floor", 0.05)),
    )
