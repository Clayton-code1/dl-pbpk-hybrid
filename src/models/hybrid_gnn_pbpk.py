"""Hybrid GNN + patient-feature model that predicts PK parameters.

The model combines:
1. A molecular embedding from MoleculeGNN (drug structure via SMILES)
2. Patient/regimen features (weight_kg, dose_mg, dose_mgkg)

and outputs pharmacokinetic parameters (CL_total, ka) via exp/softplus
mapping to keep values strictly positive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.gnn.molecule_gnn import MoleculeGNN


@dataclass
class HybridGNNConfig:
    node_feat_dim: int = 27
    edge_feat_dim: int = 6
    gnn_hidden: int = 128
    gnn_layers: int = 3
    gnn_embed_dim: int = 128
    patient_feat_dim: int = 3   # weight_kg, dose_mg, dose_mgkg
    head_hidden: int = 64
    cl_floor: float = 0.1
    ka_floor: float = 0.05
    n_euler_steps: int = 300


_CL_INIT = 2.5   # L/h (Theophylline typical)
_KA_INIT = 1.5   # 1/h


class HybridGNNPBPK(nn.Module):
    """Drug embedding (GNN) + patient features -> PK parameters -> ODE curve."""

    def __init__(self, config: HybridGNNConfig | None = None) -> None:
        super().__init__()
        cfg = config or HybridGNNConfig()
        self.config = cfg

        self.gnn = MoleculeGNN(
            node_feat_dim=cfg.node_feat_dim,
            edge_feat_dim=cfg.edge_feat_dim,
            hidden_dim=cfg.gnn_hidden,
            num_layers=cfg.gnn_layers,
            embed_dim=cfg.gnn_embed_dim,
        )

        combined_dim = cfg.gnn_embed_dim + cfg.patient_feat_dim

        self.head = nn.Sequential(
            nn.Linear(combined_dim, cfg.head_hidden),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden, cfg.head_hidden),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden, 2),  # CL, ka in log-space
        )

        self._init_head()

    def _init_head(self) -> None:
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        last = list(self.head.children())[-1]
        assert isinstance(last, nn.Linear)
        with torch.no_grad():
            last.bias.copy_(torch.tensor([
                math.log(_CL_INIT),
                math.log(_KA_INIT),
            ]))

    def get_drug_embedding(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> Tensor:
        """Return the graph-level drug embedding [embed_dim]."""
        return self.gnn(x, edge_index, edge_attr)

    def predict_pk_params(
        self,
        drug_emb: Tensor,
        patient_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predict (CL, ka) from drug embedding + patient features.

        Parameters
        ----------
        drug_emb : [embed_dim]
        patient_features : [patient_feat_dim] (normalised)

        Returns
        -------
        CL : scalar Tensor (L/h)
        ka : scalar Tensor (1/h)
        """
        combined = torch.cat([drug_emb, patient_features], dim=-1)
        raw = self.head(combined)  # [2] in log-space
        params = torch.exp(raw)

        CL = params[0].clamp(min=self.config.cl_floor)
        ka = params[1].clamp(min=self.config.ka_floor)
        return CL, ka

    def forward(
        self,
        graph_x: Tensor,
        graph_edge_index: Tensor,
        graph_edge_attr: Tensor,
        patient_features: Tensor,
        times_hr: Tensor,
        dose_mg: Tensor,
        V: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Full forward pass: graph + patient -> PK params -> ODE -> curve.

        V (volume of distribution) is supplied externally from physiology or
        the existing MLP-based prediction since the GNN head predicts only
        CL and ka.

        Returns (conc_pred, CL, V, ka).
        """
        drug_emb = self.get_drug_embedding(graph_x, graph_edge_index, graph_edge_attr)
        CL, ka = self.predict_pk_params(drug_emb, patient_features)

        from src.models.ode.pk_1cpt_torch import simulate
        conc_pred = simulate(
            times_hr, dose_mg, CL, V, ka,
            n_euler_steps=self.config.n_euler_steps,
        )
        return conc_pred, CL, V, ka
