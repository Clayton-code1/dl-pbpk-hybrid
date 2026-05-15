"""API mirror of ``experiments.models.hybrid_multidrug.MultiDrugHybridGNNPBPK``.

Must stay architecture-compatible with training checkpoints under
``artifacts/models/hybrid_gnn_pbpk_{drug}_v1/``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from app.services._gnn_inline import _MoleculeGNN


class InlineMultiDrugHybridGNNPBPK(nn.Module):
    """GNN + patient features -> (CL_per_kg, Vd_per_kg, ka) -> scaled CL, V for patient weight."""

    def __init__(
        self,
        *,
        node_feat_dim: int = 27,
        edge_feat_dim: int = 6,
        gnn_hidden: int = 64,
        gnn_layers: int = 2,
        gnn_embed_dim: int = 64,
        patient_feat_dim: int = 5,
        head_hidden: int = 64,
        head_dropout: float = 0.05,
        cl_per_kg_floor: float = 1e-4,
        v_per_kg_floor: float = 1e-3,
        ka_floor: float = 0.05,
    ) -> None:
        super().__init__()
        self.cl_per_kg_floor = cl_per_kg_floor
        self.v_per_kg_floor = v_per_kg_floor
        self.ka_floor = ka_floor
        self.patient_feat_dim = patient_feat_dim

        self.gnn = _MoleculeGNN(node_feat_dim, edge_feat_dim, gnn_hidden, gnn_layers, gnn_embed_dim)

        combined_dim = gnn_embed_dim + patient_feat_dim
        self.head = nn.Sequential(
            nn.Linear(combined_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 3),
        )
        self._init_head()

    def _init_head(self) -> None:
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        last = [m for m in self.head.modules() if isinstance(m, nn.Linear)][-1]
        with torch.no_grad():
            last.weight.mul_(0.1)
            last.bias.copy_(torch.tensor([math.log(0.05), math.log(0.5), math.log(1.5)]))

    def get_drug_embedding(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        return self.gnn(x, edge_index, edge_attr)

    def predict_pk_params(
        self,
        drug_emb: Tensor,
        patient_features: Tensor,
        weight_kg: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x = torch.cat([drug_emb, patient_features], dim=-1)
        raw = self.head(x).clamp(-6.0, 6.0)
        cl_per_kg = torch.exp(raw[0]).clamp(min=self.cl_per_kg_floor, max=3.0)
        v_per_kg = torch.exp(raw[1]).clamp(min=self.v_per_kg_floor, max=15.0)
        ka = torch.exp(raw[2]).clamp(min=self.ka_floor, max=8.0)
        CL = cl_per_kg * weight_kg
        V = v_per_kg * weight_kg
        return CL, V, ka, cl_per_kg, v_per_kg


def build_multidrug_model(cfg: dict[str, Any]) -> InlineMultiDrugHybridGNNPBPK:
    return InlineMultiDrugHybridGNNPBPK(
        node_feat_dim=int(cfg.get("node_feat_dim", 27)),
        edge_feat_dim=int(cfg.get("edge_feat_dim", 6)),
        gnn_hidden=int(cfg.get("gnn_hidden", 64)),
        gnn_layers=int(cfg.get("gnn_layers", 2)),
        gnn_embed_dim=int(cfg.get("gnn_embed_dim", 64)),
        patient_feat_dim=int(cfg.get("patient_feat_dim", 5)),
        head_hidden=int(cfg.get("head_hidden", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.05)),
        cl_per_kg_floor=float(cfg.get("cl_per_kg_floor", 1e-4)),
        v_per_kg_floor=float(cfg.get("v_per_kg_floor", 1e-3)),
        ka_floor=float(cfg.get("ka_floor", 0.05)),
    )
