"""Multi-drug hybrid GNN+PBPK model.

This module extends :class:`src.models.hybrid_gnn_pbpk.HybridGNNPBPK` so the
network can be trained across drugs whose volume of distribution differs by
orders of magnitude (e.g. warfarin ~10 L vs digoxin ~500 L).

Architecture
------------
``MoleculeGNN`` -> graph embedding (E)
``patient_features`` -> [weight_kg, dose_mg, dose_mgkg, age_years, sex]
                       (z-score normalised by the per-drug scaler)

The fusion head (E + 5 patient features) outputs three log-space numbers
that are mapped to **per-kg** PK parameters:

    log(CL_per_kg)  ->  exp(.)  ->  CL_per_kg  [L/h/kg]
    log(Vd_per_kg)  ->  exp(.)  ->  Vd_per_kg  [L/kg]
    log(ka)         ->  exp(.)  ->  ka         [1/h]

The absolute parameters used by the differentiable 1-cpt ODE are then
``CL = CL_per_kg * weight_kg`` and ``V = Vd_per_kg * weight_kg``.  Allometric
scaling lets us share one network across drugs while keeping the prediction
of CL and V physiologically grounded.

The GNN encoder shape (node_feat_dim=27, edge_feat_dim=6, hidden=64,
layers=2, embed=64) matches ``gnn_pretrain_combined_v1`` so the pretrained
weights can be loaded for transfer learning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from src.models.gnn.molecule_gnn import MoleculeGNN
from src.models.ode.pk_1cpt_torch import simulate


@dataclass
class MultiDrugHybridConfig:
    node_feat_dim: int = 27
    edge_feat_dim: int = 6

    # Match ``gnn_pretrain_combined_v1/config.json`` so encoder weights load.
    gnn_hidden: int = 64
    gnn_layers: int = 2
    gnn_embed_dim: int = 64

    patient_feat_dim: int = 5      # weight, dose, dose_mgkg, age, sex
    head_hidden: int = 64
    head_dropout: float = 0.05

    cl_per_kg_floor: float = 1e-4   # L/h/kg
    v_per_kg_floor: float = 1e-3    # L/kg
    ka_floor: float = 0.05          # 1/h

    n_euler_steps: int = 200


# Population-typical seeds for the head bias (Theophylline-like) so the
# untrained model already lives in a physiologically-reasonable region.
_INIT_CL_PER_KG = 0.05    # L/h/kg
_INIT_V_PER_KG = 0.5      # L/kg
_INIT_KA = 1.5            # 1/h


class MultiDrugHybridGNNPBPK(nn.Module):
    """GNN + patient features -> (CL_per_kg, Vd_per_kg, ka) -> ODE curve."""

    def __init__(self, config: MultiDrugHybridConfig | None = None) -> None:
        super().__init__()
        cfg = config or MultiDrugHybridConfig()
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
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, cfg.head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, 3),  # log(CL/kg), log(Vd/kg), log(ka)
        )

        self._init_head()

    # ------------------------------------------------------------------ init

    def _init_head(self) -> None:
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

        last = [m for m in self.head.modules() if isinstance(m, nn.Linear)][-1]
        with torch.no_grad():
            last.weight.mul_(0.1)  # damp final layer so bias dominates initially
            last.bias.copy_(torch.tensor([
                math.log(_INIT_CL_PER_KG),
                math.log(_INIT_V_PER_KG),
                math.log(_INIT_KA),
            ]))

    # --------------------------------------------------------- encoder utils

    def freeze_gnn(self) -> None:
        for p in self.gnn.parameters():
            p.requires_grad = False

    def unfreeze_gnn(self) -> None:
        for p in self.gnn.parameters():
            p.requires_grad = True

    def get_drug_embedding(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> Tensor:
        return self.gnn(x, edge_index, edge_attr)

    # ----------------------------------------------------------- PK params

    def predict_pk_params(
        self,
        drug_emb: Tensor,
        patient_features: Tensor,
        weight_kg: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return (CL, V, ka, CL_per_kg, Vd_per_kg) for one patient."""
        x = torch.cat([drug_emb, patient_features], dim=-1)
        raw = self.head(x).clamp(-6.0, 6.0)
        cl_per_kg = torch.exp(raw[0]).clamp(
            min=self.config.cl_per_kg_floor, max=3.0,
        )
        v_per_kg = torch.exp(raw[1]).clamp(
            min=self.config.v_per_kg_floor, max=15.0,
        )
        ka = torch.exp(raw[2]).clamp(min=self.config.ka_floor, max=8.0)

        CL = cl_per_kg * weight_kg
        V = v_per_kg * weight_kg
        return CL, V, ka, cl_per_kg, v_per_kg

    # ------------------------------------------------------------- forward

    def forward(
        self,
        graph_x: Tensor,
        graph_edge_index: Tensor,
        graph_edge_attr: Tensor,
        patient_features: Tensor,
        times_hr: Tensor,
        dose_mg: Tensor,
        weight_kg: Tensor,
        f_bio: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Single-patient forward pass.

        ``f_bio`` is the oral bioavailability fraction (0–1].  The ODE gut
        compartment is initialised with ``f_bio * dose_mg`` so predictions
        match the closed-form generator in Phase 1.3.  When omitted, F=1.
        """
        drug_emb = self.get_drug_embedding(graph_x, graph_edge_index, graph_edge_attr)
        CL, V, ka, _, _ = self.predict_pk_params(drug_emb, patient_features, weight_kg)
        if f_bio is None:
            dose_eff = dose_mg
        else:
            dose_eff = dose_mg * f_bio.clamp(min=1e-4, max=1.0)
        conc_pred = simulate(
            times_hr, dose_eff, CL, V, ka,
            n_euler_steps=self.config.n_euler_steps,
        )
        return conc_pred, CL, V, ka
