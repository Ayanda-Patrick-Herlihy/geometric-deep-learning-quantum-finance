"""Integrated Geometric-Quantum-Graph model for financial prediction.

Combines hypersphere embedding, correlation-graph message passing, and regime-aware
gating with a per-node head. Inputs follow the flat PyTorch Geometric layout:
``(total_nodes, D)`` features and a ``batch`` vector, so a full mini-batch is one
vectorised forward pass.

References:
    Bronstein et al. (2021) "Geometric Deep Learning" arXiv:2104.13478.
    Haven and Khrennikov (2013) "Quantum Social Science."
    Kipf and Welling (2016) "Semi-Supervised Classification with
        Graph Convolutional Networks." arXiv:1609.02907.
"""

from typing import Any
import logging

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from src.layers.geometric import GeometricEmbedding
from src.layers.graph import GraphDependencyLayer
from src.layers.quantum import build_regime_detector

logger = logging.getLogger(__name__)


class GDLQuantumFinanceModel(nn.Module):
    """Full model: geometry, graphs, quantum regimes, and prediction.

    Data flows through five stages:
        1. Geometric: raw features are projected onto a hypersphere.
        2. Graph: GNN aggregates information across correlated stocks.
        3. Quantum: pooled embeddings + yield curve context produce
           regime probabilities via a density matrix.
        4. Fusion: regime probabilities gate the graph output so the
           prediction is regime-aware.
        5. Prediction: a small MLP outputs a log return forecast.

    Args:
        feature_dim: Input feature dimension per stock (D).
        embedding_dim: Hypersphere ambient dimension (E).
        hidden_dim: GNN hidden channels.
        graph_out_dim: GNN output channels (G).
        num_regimes: Number of market regimes (K).
        gat_heads: Number of GAT attention heads.
        context_dim: Yield curve context dimension.
        prediction_hidden_dim: Hidden dim for the prediction MLP.
        dropout: Feature dropout rate.
        edge_drop: DropEdge probability.
        sphere: If False, use the Euclidean twin of the geometric layer (control).
        regime_head: 'quantum' (default), or a classical control 'softmax' /
            'softmax_mlp' (see layers.quantum.build_regime_detector).
    """

    def __init__(
        self,
        feature_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        graph_out_dim: int,
        num_regimes: int,
        gat_heads: int,
        context_dim: int = 3,
        prediction_hidden_dim: int = 16,
        dropout: float = 0.1,
        edge_drop: float = 0.1,
        sphere: bool = True,
        regime_head: str = "quantum",
    ) -> None:
        super().__init__()

        # Stage 1: Project features onto the hypersphere
        self.geometric = GeometricEmbedding(
            input_dim=feature_dim, output_dim=embedding_dim, normalize=sphere
        )

        # Stage 2: Learn cross-asset dependencies from correlations
        self.graph = GraphDependencyLayer(
            in_channels=embedding_dim,
            hidden_channels=hidden_dim,
            out_channels=graph_out_dim,
            heads=gat_heads,
            dropout=dropout,
            edge_drop=edge_drop,
        )

        # Stage 3: Detect market regimes from pooled embeddings
        self.quantum = build_regime_detector(
            regime_head, embedding_dim, context_dim, num_regimes
        )

        # Stage 4: Gate graph output by regime probabilities
        self.regime_gate = nn.Sequential(
            nn.Linear(num_regimes, graph_out_dim),
            nn.Sigmoid(),
        )

        # Stage 5: Predict log returns (3-layer MLP for richer capacity)
        self.prediction_head = nn.Sequential(
            nn.Linear(graph_out_dim, prediction_hidden_dim),
            nn.ELU(),
            nn.Linear(prediction_hidden_dim, prediction_hidden_dim // 2),
            nn.ELU(),
            nn.Linear(prediction_hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        context: torch.Tensor,
        batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass through all five stages.

        Accepts the flat PyG batch format where nodes from multiple
        graphs are concatenated. The ``batch`` vector identifies which
        graph each node belongs to.

        Args:
            x: Node features (total_nodes, feature_dim).
            edge_index: Block-diagonal graph connectivity (2, E).
            edge_weight: Edge weights (E,) or None.
            context: Yield curve context (B, context_dim).
            batch: Graph membership vector (total_nodes,). If None,
                all nodes are assumed to belong to graph 0.

        Returns:
            Dictionary with 'predictions', 'regime_probs',
            'density_matrix', and 'embeddings'.
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Stage 1: Map features onto the hypersphere
        h_geo = self.geometric(x)  # (total_nodes, E)

        # Stage 2: Propagate across the stock graph (single vectorised call)
        h_graph = self.graph(h_geo, edge_index, edge_weight)  # (total_nodes, G)

        # Stage 3: Pool geometric embeddings per graph and detect regimes
        h_global = global_mean_pool(h_geo, batch)  # (B, E)
        regime_probs, rho = self.quantum(h_global, context)

        # Stage 4: Gate graph output by regime probabilities
        gate_per_graph = self.regime_gate(regime_probs)  # (B, G)
        gate_per_node = gate_per_graph[batch]  # (total_nodes, G)
        h_fused = h_graph * (1.0 + gate_per_node)

        # Stage 5: Predict log returns per stock
        predictions = self.prediction_head(h_fused)  # (total_nodes, 1)

        return {
            "predictions": predictions,
            "regime_probs": regime_probs,
            "density_matrix": rho,
            "embeddings": h_geo,
        }
