"""Pure-PyTorch Graph Neural Network for molecular encoding.

Implements a message-passing network with:
- EdgeMLP that computes messages from (h_src, h_dst, edge_attr)
- Aggregation via scatter-add (index_add_)
- GRU-based node state update
- Mean + max pool readout -> graph-level embedding
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class EdgeMLP(nn.Module):
    """Compute edge messages from source, destination, and edge features."""

    def __init__(self, node_dim: int, edge_dim: int, msg_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, msg_dim),
            nn.ReLU(),
            nn.Linear(msg_dim, msg_dim),
        )

    def forward(self, h_src: Tensor, h_dst: Tensor, edge_attr: Tensor) -> Tensor:
        inp = torch.cat([h_src, h_dst, edge_attr], dim=-1)
        return self.mlp(inp)


class MessagePassingLayer(nn.Module):
    """One round of message passing + GRU update."""

    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.edge_mlp = EdgeMLP(hidden_dim, edge_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(
        self, h: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> Tensor:
        """
        Parameters
        ----------
        h : [N, hidden_dim]
        edge_index : [2, E]
        edge_attr : [E, edge_dim]
        """
        if edge_index.size(1) == 0:
            return h

        src, dst = edge_index[0], edge_index[1]
        msgs = self.edge_mlp(h[src], h[dst], edge_attr)  # [E, hidden_dim]

        agg = torch.zeros_like(h)  # [N, hidden_dim]
        agg.index_add_(0, dst, msgs)

        h_new = self.gru(agg, h)
        return h_new


class MoleculeGNN(nn.Module):
    """Graph neural network that maps a molecular graph to a fixed-size embedding.

    Parameters
    ----------
    node_feat_dim : int
        Dimensionality of input node feature vectors.
    edge_feat_dim : int
        Dimensionality of input edge feature vectors.
    hidden_dim : int
        Width of internal node representations.
    num_layers : int
        Number of message-passing rounds.
    embed_dim : int
        Size of the output graph-level embedding.
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [MessagePassingLayer(hidden_dim, edge_feat_dim) for _ in range(num_layers)]
        )
        self.readout = nn.Linear(2 * hidden_dim, embed_dim)

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed_dim = embed_dim

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> Tensor:
        """Produce a graph-level embedding.

        Parameters
        ----------
        x : [N, node_feat_dim]
        edge_index : [2, E]
        edge_attr : [E, edge_feat_dim]

        Returns
        -------
        Tensor [embed_dim]
        """
        h = self.node_encoder(x)  # [N, hidden]

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)

        mean_pool = h.mean(dim=0)  # [hidden]
        max_pool = h.max(dim=0).values  # [hidden]
        pooled = torch.cat([mean_pool, max_pool], dim=-1)  # [2*hidden]

        return self.readout(pooled)  # [embed_dim]

    def config_dict(self) -> dict:
        return {
            "node_feat_dim": self.node_feat_dim,
            "edge_feat_dim": self.edge_feat_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "embed_dim": self.embed_dim,
        }
