"""Hybrid Deep-Learning + ODE pharmacokinetic model.

An MLP maps subject features (dose_mg, weight_kg, dose_mgkg) to the three
1-compartment PK parameters (CL, V, ka) through an exponential mapping
that keeps outputs strictly positive and physiologically plausible.

The entire forward pass is differentiable, enabling end-to-end
gradient-based training against observed PK profiles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from src.models.ode.pk_1cpt_torch import simulate


@dataclass
class HybridConfig:
    """Hyper-parameters for the hybrid model."""
    n_input: int = 3           # dose_mg, weight_kg, dose_mgkg
    hidden_dim: int = 32
    n_euler_steps: int = 300
    cl_floor: float = 0.1
    v_floor: float = 1.0
    ka_floor: float = 0.05


# Typical Theophylline population PK values used to initialise the output
# layer bias so the model starts in a reasonable region of parameter space.
_PK_INIT = {
    "CL": 2.5,    # L/h
    "V": 30.0,    # L
    "ka": 1.5,    # 1/h
}


class HybridDLPKModel(nn.Module):
    """MLP --> (CL, V, ka) --> ODE solver --> concentration curve."""

    def __init__(self, config: HybridConfig | None = None) -> None:
        super().__init__()
        cfg = config or HybridConfig()
        self.config = cfg

        self.net = nn.Sequential(
            nn.Linear(cfg.n_input, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Tanh(),
        )
        self.head = nn.Linear(cfg.hidden_dim, 3)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier init for hidden layers; bias of head set to log of typical PK values."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

        nn.init.xavier_normal_(self.head.weight, gain=0.1)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([
                math.log(_PK_INIT["CL"]),
                math.log(_PK_INIT["V"]),
                math.log(_PK_INIT["ka"]),
            ]))

    def predict_pk_params(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return (CL, V, ka) tensors for a single subject feature vector."""
        h = self.net(x)
        raw = self.head(h)          # (3,)  — log-space
        params = torch.exp(raw)     # strictly positive

        CL = params[0].clamp(min=self.config.cl_floor)
        V  = params[1].clamp(min=self.config.v_floor)
        ka = params[2].clamp(min=self.config.ka_floor)
        return CL, V, ka

    def forward(
        self,
        x: Tensor,
        times_hr: Tensor,
        dose_mg: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Full forward: features --> PK params --> ODE --> concentrations.

        Parameters
        ----------
        x : Tensor, shape (n_input,)
            Normalised subject features.
        times_hr : Tensor, shape (T,)
            Observation time grid.
        dose_mg : Tensor, scalar
            Actual (un-normalised) dose in mg for the ODE IC.

        Returns
        -------
        conc_pred : Tensor (T,)
        CL, V, ka : Tensors (scalar each)
        """
        CL, V, ka = self.predict_pk_params(x)
        conc_pred = simulate(
            times_hr, dose_mg, CL, V, ka,
            n_euler_steps=self.config.n_euler_steps,
        )
        return conc_pred, CL, V, ka
