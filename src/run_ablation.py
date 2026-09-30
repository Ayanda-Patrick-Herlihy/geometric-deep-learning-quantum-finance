"""Ablation experiment runner for the GDL-Quantum-Finance project.

Defines nine model configurations (A0 through A8), trains each under
walk-forward cross-validation with multiple seeds, and compares
results using paired t-tests and bootstrap confidence intervals.

Theoretical Context:
    To prove the efficacy of the "Geometric-Quantum" architecture, we must
    isolate the marginal contribution of each component.
    We use the "Ceteris Paribus" principle: changing one architectural feature
    (e.g., removing the Graph layer) while holding all other variables (data splits,
    seeds, optimisation hyperparameters) constant. This rigorously tests the
    hypothesis that Geometry + Topology + Quantum Physics > Euclidean Deep Learning.

References:
    Standard Scientific Method (Control vs Treatment).

Usage:
    cd GDL-Quantum-Finance
    uv run python src/run_ablation.py                     # full run
    uv run python src/run_ablation.py --smoke-test        # synthetic data
    uv run python src/run_ablation.py --configs A0 A9     # subset
    uv run python src/run_ablation.py --max-folds 2       # limit folds
"""

import sys
import os
import argparse
import copy
import csv
import logging
import time
from pathlib import Path
from typing import Any

try:
    import wandb
except ImportError:
    wandb = None
from tqdm import tqdm

# Resolve project root and add to sys.path to enable 'from src...' imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

# Enable CPU fallback for MPS-unsupported operations.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from src.data_loader import (
    FinancialGraphDataset,
    WalkForwardSplitter,
    dataset_view,
    build_synthetic_dataframe,
    get_device,
)
from src.evaluation import (
    bootstrap_confidence_interval,
    evaluate_model_on_dataset,
    paired_t_test,
)
from src.layers.geometric import GeometricEmbedding
from src.layers.graph import GraphDependencyLayer
from src.layers.quantum import QuantumRegimeDetector, build_regime_detector
from src.losses import GDLQuantumLoss
from src.models import GDLQuantumFinanceModel
from src.train import (
    EarlyStopping,
    _make_dataloader,
    _validate_device,
    load_config,
    train_one_epoch,
    validate_one_epoch,
)

logger = logging.getLogger(__name__)

_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


def _get_config_name(config_id: str) -> str:
    """Helper to get a safe model name for filenames."""
    # Maps config_id to a filesystem-safe name
    names = {
        "A0": "Linear_Baseline",
        "A1": "Euclidean_MLP",
        "A2": "Geometry_Only",
        # Corrected 2026-09: earlier runs wrote A3-A7 files under swapped
        # labels (A3 Geometry_Graph, A4 Geometry_Quantum, A5 Euclidean_Quantum_Graph,
        # A6 Graph_Only, A7 Quantum_Only). The config_id prefix was always right.
        "A3": "Graph_Only",
        "A4": "Quantum_Only",
        "A5": "Geometry_Graph",
        "A6": "Geometry_Quantum",
        "A7": "Graph_Quantum",
        "A8": "Full_Model",
        "A9": "LSTM_Baseline",
    }
    base_id, _ = _resolve_variant(config_id)
    return names.get(config_id, f"{names.get(base_id, base_id)}_{config_id}")


# Ablation configuration names in order.
ABLATION_IDS: list[str] = [
    "A0",
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "A7",
    "A8",
    "A9",
]

# Single-term controls: (base config, model flags, loss overrides). Each differs
# from its base in exactly one named respect so the comparison is ceteris paribus.
ABLATION_VARIANTS: dict[str, tuple[str, dict[str, Any], dict[str, float]]] = {
    # Euclidean twins: same MLP, no sphere projection, no uniformity loss.
    "A2e": ("A2", {"sphere": False}, {"uniformity": 0.0}),
    # Sphere kept, pre-normalisation norm restored as one scalar input to the head.
    "A2m": ("A2", {"magnitude": True}, {}),
    # Sphere kept, uniformity loss removed (with A2e this separates projection from loss).
    "A2u": ("A2", {}, {"uniformity": 0.0}),
    "A6e": ("A6", {"sphere": False}, {"uniformity": 0.0}),
    "A8e": ("A8", {"sphere": False}, {"uniformity": 0.0}),
    # Sphere kept, uniformity loss removed.
    "A6u": ("A6", {}, {"uniformity": 0.0}),
    "A8u": ("A8", {}, {"uniformity": 0.0}),
    # Classical regime heads replacing the density-matrix layer.
    "A6c": ("A6", {"regime_head": "softmax_mlp"}, {}),
    "A8c": ("A8", {"regime_head": "softmax_mlp"}, {}),
    "A6s": ("A6", {"regime_head": "softmax"}, {}),
    "A8s": ("A8", {"regime_head": "softmax"}, {}),
    # LSTM baseline whose input window ends at r_t instead of r_{t-1}.
    "A9t": ("A9", {}, {}),
}

# Dataset options that a variant changes (passed to FinancialGraphDataset).
VARIANT_DATASET_KWARGS: dict[str, dict[str, Any]] = {
    "A9t": {"sequence_includes_current": True},
}
# Default runs stay A0-A9; variants run only when named with --configs.
ALL_IDS: list[str] = ABLATION_IDS + list(ABLATION_VARIANTS)


def _resolve_variant(config_id: str) -> tuple[str, dict[str, Any]]:
    """Maps a variant ID to (base ID, model flags); base IDs map to themselves."""
    if config_id in ABLATION_VARIANTS:
        base_id, flags, _ = ABLATION_VARIANTS[config_id]
        return base_id, dict(flags)
    return config_id, {}


ABLATION_DESCRIPTIONS: dict[str, str] = {
    # Intent: Null Hypothesis (A0).
    # A simple Linear Regression. If our complex model can't beat this,
    # the entire GDL approach is invalid for this dataset.
    # A0: Linear Baseline (nn.Linear)
    "A0": "Linear Baseline (nn.Linear)",
    # Intent: Strong Baseline (A1).
    # A deep but "Euclidean" MLP with the same number of parameters as A8.
    # This checks if the performance gain comes from the *Architecture* (Geometry)
    # or just from having more parameters (Capacity).
    # A1: Euclidean MLP (2-layer, ELU; matched parameter budget)
    "A1": "Euclidean MLP (2-layer, ELU; matched parameter budget)",
    "A2": "Geometric Only (GeometricEmbedding + head)",
    "A3": "Graph Only (GraphDependencyLayer on raw features + head)",
    "A4": "Quantum Only (QuantumRegimeDetector on pooled features + head)",
    "A5": "Geometric + Graph (no quantum; regime gating disabled)",
    "A6": "Geometric + Quantum (no graph)",
    "A7": "Graph + Quantum (no geometric/manifold)",
    "A8": "Full Model (all components)",
    "A9": "LSTM Baseline (on raw return sequences)",
    "A2e": "A2 without sphere projection or uniformity loss",
    "A2m": "A2 with the discarded embedding norm fed to the head",
    "A2u": "A2 without uniformity loss",
    "A6e": "A6 without sphere projection or uniformity loss",
    "A8e": "A8 without sphere projection or uniformity loss",
    "A6u": "A6 without uniformity loss",
    "A8u": "A8 without uniformity loss",
    "A6c": "A6 with parameter-matched softmax-MLP regime head",
    "A8c": "A8 with parameter-matched softmax-MLP regime head",
    "A6s": "A6 with linear softmax regime head",
    "A8s": "A8 with linear softmax regime head",
    "A9t": "A9 with the input window including the current day's return",
}


def _dummy_regime_probs(
    x: torch.Tensor,
    batch: torch.Tensor | None,
    num_regimes: int = 2,
) -> torch.Tensor:
    """Returns a neutral uniform prior over regimes for ablation baselines without a quantum head.

    The full model learns ``regime_probs`` from data; Euclidean and partial ablations
    still expose the same output dict so training and loss code stay unchanged. Each
    graph receives an identical maximum-entropy distribution (no information from ``x``).

    Args:
        x: Supplies device and dtype for the returned tensor.
        batch: PyG ``batch`` vector; used only to count distinct graphs in the batch.
        num_regimes: Output width ``K`` (must match what the loss and gate expect).

    Returns:
        Tensor of shape ``(num_graphs, num_regimes)`` with rows summing to one.
    """
    if batch is None or batch.numel() == 0:
        num_graphs = 1
    else:
        num_graphs = int(batch.max().item()) + 1

    regime_probs = torch.full(
        (num_graphs, num_regimes),
        1.0 / float(num_regimes),
        dtype=x.dtype,
        device=x.device,
    )
    return regime_probs


def _dummy_embeddings(n: int, x: torch.Tensor) -> torch.Tensor:
    """Returns a zero embedding tensor for ablation models without a geometric head.

    Matches the shape expected by GDLQuantumLoss but carries no geometric
    information; the uniformity term is effectively zero for these configs.

    Args:
        n: Number of nodes (total_nodes in the flat PyG batch).
        x: Reference tensor; supplies device and dtype.

    Returns:
        Zero tensor of shape (n, 2).
    """
    return torch.zeros(n, 2, dtype=x.dtype, device=x.device)


def _dummy_density_matrix(x: torch.Tensor) -> torch.Tensor:
    """Returns a 2×2 identity density matrix for ablation models without a quantum head.

    A single identity matrix is broadcast across the batch. The density-matrix
    regularisation term is effectively zero because the identity is already
    pure-trace and positive semi-definite.

    Args:
        x: Reference tensor; supplies device and dtype.

    Returns:
        Identity density matrix of shape (1, 2, 2).
    """
    return torch.eye(2, dtype=x.dtype, device=x.device).unsqueeze(0)


class LinearBaseline(nn.Module):
    """A0: Linear baseline -- nn.Linear(D, 1) on raw features.

    Args:
        feature_dim: Input feature dimension (D).
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        context: torch.Tensor,
        batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Forward pass ignoring graph and context.

        Args:
            x: Node features (total_nodes, feature_dim).
            edge_index: Unused (graph connectivity).
            edge_weight: Unused (edge weights).
            context: Unused (yield curve context).
            batch: Unused (graph membership vector).

        Returns:
            Dictionary with ``predictions``, minimal ``embeddings`` and ``density_matrix``
            tensors for API parity with the full model, plus ``regime_probs`` from a fixed
            uniform prior (see ``_dummy_regime_probs``).
        """
        n = x.size(0)
        preds = self.linear(x)  # (total_nodes, 1)

        return {
            "predictions": preds,
            "embeddings": _dummy_embeddings(n, x),
            "regime_probs": _dummy_regime_probs(x, batch),
            "density_matrix": _dummy_density_matrix(x),
        }


class EuclideanMLP(nn.Module):
    """A1: 2-layer Euclidean MLP with ELU activation.

    Designed to match the approximate parameter budget of the full
    model (A8), so that any performance differences are not merely
    due to model capacity.

    Args:
        feature_dim: Input feature dimension (D).
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
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
        """Forward pass: MLP applied per node.

        Args:
            x: Node features (total_nodes, feature_dim).
            edge_index: Unused.
            edge_weight: Unused.
            context: Unused.
            batch: Unused (graph membership vector).

        Returns:
            Dictionary with 'predictions', 'embeddings', 'density_matrix'.
        """
        n = x.size(0)
        preds = self.mlp(x)  # (total_nodes, 1)

        return {
            "predictions": preds,
            "embeddings": _dummy_embeddings(n, x),
            "regime_probs": _dummy_regime_probs(x, batch),
            "density_matrix": _dummy_density_matrix(x),
        }


class GeometricOnlyModel(nn.Module):
    """A2: Geometric embedding + prediction head (no graph, no quantum).

    Args:
        feature_dim: Input feature dimension.
        embedding_dim: Hypersphere ambient dimension.
        prediction_hidden_dim: Prediction head hidden size.
    """

    def __init__(
        self,
        feature_dim: int,
        embedding_dim: int,
        prediction_hidden_dim: int = 16,
        sphere: bool = True,
        magnitude: bool = False,
    ) -> None:
        super().__init__()
        self.geometric = GeometricEmbedding(
            input_dim=feature_dim,
            output_dim=embedding_dim,
            normalize=sphere,
        )
        # Mechanism probe: hand the head the norm the sphere discards.
        self.magnitude = magnitude
        self.head = nn.Sequential(
            nn.Linear(embedding_dim + int(magnitude), prediction_hidden_dim),
            nn.ELU(),
            nn.Linear(prediction_hidden_dim, 1),
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
        """Forward: geometric embedding followed by prediction head.

        Args:
            x: Node features (total_nodes, feature_dim).
            edge_index: Unused.
            edge_weight: Unused.
            context: Unused.
            batch: Unused (graph membership vector).

        Returns:
            Predictions and embeddings.
        """
        h = self.geometric(x)  # (total_nodes, E)
        head_input = h
        if self.magnitude:
            pre_norm = self.geometric.projection(F.elu(self.geometric.pre_projection(x)))
            head_input = torch.cat([h, torch.log1p(pre_norm.norm(dim=-1, keepdim=True))], dim=-1)
        preds = self.head(head_input)  # (total_nodes, 1)

        return {
            "predictions": preds,
            "embeddings": h,
            "regime_probs": _dummy_regime_probs(x, batch),
            "density_matrix": _dummy_density_matrix(x),
        }


class GraphOnlyModel(nn.Module):
    """A3: Graph layer on raw features + prediction head (no geometric, no quantum).

    Args:
        feature_dim: Input feature dimension.
        hidden_channels: GCN intermediate dimension.
        out_channels: GNN output dimension.
        gat_heads: Number of GAT attention heads.
        prediction_hidden_dim: Prediction head hidden size.
        dropout: Feature dropout rate.
        edge_drop: DropEdge probability.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_channels: int = 64,
        out_channels: int = 32,
        gat_heads: int = 4,
        prediction_hidden_dim: int = 16,
        dropout: float = 0.1,
        edge_drop: float = 0.1,
    ) -> None:
        super().__init__()
        self.graph = GraphDependencyLayer(
            in_channels=feature_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=gat_heads,
            dropout=dropout,
            edge_drop=edge_drop,
        )
        self.head = nn.Sequential(
            nn.Linear(out_channels, prediction_hidden_dim),
            nn.ELU(),
            nn.Linear(prediction_hidden_dim, 1),
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
        """Forward: graph layer on raw features, then prediction head.

        Args:
            x: Node features (total_nodes, feature_dim).
            edge_index: Block-diagonal graph connectivity (2, E).
            edge_weight: Edge weights (E,) or None.
            context: Unused.
            batch: Unused (graph membership vector).

        Returns:
            Predictions and dummy embeddings/density matrix.
        """
        n = x.size(0)
        h = self.graph(x, edge_index, edge_weight)  # (total_nodes, G)
        preds = self.head(h)  # (total_nodes, 1)

        return {
            "predictions": preds,
            "embeddings": _dummy_embeddings(n, x),
            "regime_probs": _dummy_regime_probs(x, batch),
            "density_matrix": _dummy_density_matrix(x),
        }


class QuantumOnlyModel(nn.Module):
    """A4: Quantum regime detector on pooled raw features + prediction head.

    Pools raw features to a global vector, applies quantum regime
    detection, then uses regime probabilities to modulate a simple
    linear prediction.

    Args:
        feature_dim: Input feature dimension.
        num_regimes: Number of market regimes (K).
        context_dim: Yield curve context dimension.
        prediction_hidden_dim: Prediction head hidden size.
    """

    def __init__(
        self,
        feature_dim: int,
        num_regimes: int = 4,
        context_dim: int = 3,
        prediction_hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        self.quantum = QuantumRegimeDetector(
            input_dim=feature_dim,
            context_dim=context_dim,
            num_regimes=num_regimes,
        )
        # Gate from regime probs to per-node modulation
        self.regime_gate = nn.Sequential(
            nn.Linear(num_regimes, feature_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(feature_dim, prediction_hidden_dim),
            nn.ELU(),
            nn.Linear(prediction_hidden_dim, 1),
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
        """Forward: quantum regime on pooled features, gated prediction.

        Args:
            x: Node features (total_nodes, feature_dim).
            edge_index: Unused.
            edge_weight: Unused.
            context: Yield curve context (B, context_dim).
            batch: Graph membership vector (total_nodes,).

        Returns:
            Predictions, dummy embeddings, and density matrix.
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        n = x.size(0)

        # Pool to per-graph global embedding
        h_global = global_mean_pool(x, batch)  # (B, D)
        regime_probs, rho = self.quantum(h_global, context)

        # Gate: (B, K) -> (B, D), scatter back to nodes
        gate = self.regime_gate(regime_probs)  # (B, D)
        gate_per_node = gate[batch]  # (total_nodes, D)
        h_gated = x * (1.0 + gate_per_node)  # (total_nodes, D)

        preds = self.head(h_gated)  # (total_nodes, 1)

        return {
            "predictions": preds,
            "embeddings": _dummy_embeddings(n, x),
            "regime_probs": regime_probs,
            "density_matrix": rho,
        }


class GeometricGraphModel(nn.Module):
    """A5: Geometric + Graph (no quantum; regime gating disabled).

    Args:
        feature_dim: Input feature dimension.
        embedding_dim: Hypersphere ambient dimension.
        hidden_channels: GCN intermediate dimension.
        out_channels: GNN output dimension.
        gat_heads: Number of GAT attention heads.
        prediction_hidden_dim: Prediction head hidden size.
        dropout: Feature dropout rate.
        edge_drop: DropEdge probability.
    """

    def __init__(
        self,
        feature_dim: int,
        embedding_dim: int,
        hidden_channels: int = 64,
        out_channels: int = 32,
        gat_heads: int = 4,
        prediction_hidden_dim: int = 16,
        dropout: float = 0.1,
        edge_drop: float = 0.1,
    ) -> None:
        super().__init__()
        self.geometric = GeometricEmbedding(
            input_dim=feature_dim,
            output_dim=embedding_dim,
        )
        self.graph = GraphDependencyLayer(
            in_channels=embedding_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=gat_heads,
            dropout=dropout,
            edge_drop=edge_drop,
        )
        self.head = nn.Sequential(
            nn.Linear(out_channels, prediction_hidden_dim),
            nn.ELU(),
            nn.Linear(prediction_hidden_dim, 1),
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
        """Forward: geometric embedding -> graph -> prediction head.

        Args:
            x: Node features (total_nodes, feature_dim).
            edge_index: Block-diagonal graph connectivity (2, E).
            edge_weight: Edge weights (E,) or None.
            context: Unused.
            batch: Unused (graph membership vector).

        Returns:
            Predictions, embeddings, and dummy density matrix.
        """
        h_geo = self.geometric(x)  # (total_nodes, E)
        h_graph = self.graph(h_geo, edge_index, edge_weight)  # (total_nodes, G)
        preds = self.head(h_graph)  # (total_nodes, 1)

        return {
            "predictions": preds,
            "embeddings": h_geo,
            "regime_probs": _dummy_regime_probs(x, batch),
            "density_matrix": _dummy_density_matrix(x),
        }


class GeometricQuantumModel(nn.Module):
    """A6: Geometric + Quantum (no graph).

    Args:
        feature_dim: Input feature dimension.
        embedding_dim: Hypersphere ambient dimension.
        num_regimes: Number of market regimes.
        context_dim: Yield curve context dimension.
        prediction_hidden_dim: Prediction head hidden size.
    """

    def __init__(
        self,
        feature_dim: int,
        embedding_dim: int,
        num_regimes: int = 4,
        context_dim: int = 3,
        prediction_hidden_dim: int = 16,
        sphere: bool = True,
        regime_head: str = "quantum",
    ) -> None:
        super().__init__()
        self.geometric = GeometricEmbedding(
            input_dim=feature_dim,
            output_dim=embedding_dim,
            normalize=sphere,
        )
        self.quantum = build_regime_detector(
            regime_head, embedding_dim, context_dim, num_regimes
        )
        self.regime_gate = nn.Sequential(
            nn.Linear(num_regimes, embedding_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, prediction_hidden_dim),
            nn.ELU(),
            nn.Linear(prediction_hidden_dim, 1),
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
        """Forward: geometric -> quantum gating -> prediction head.

        Args:
            x: Node features (total_nodes, feature_dim).
            edge_index: Unused.
            edge_weight: Unused.
            context: Yield curve context (B, context_dim).
            batch: Graph membership vector (total_nodes,).

        Returns:
            Predictions, embeddings, and density matrix.
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h_geo = self.geometric(x)  # (total_nodes, E)

        h_global = global_mean_pool(h_geo, batch)  # (B, E)
        regime_probs, rho = self.quantum(h_global, context)

        gate = self.regime_gate(regime_probs)  # (B, E)
        gate_per_node = gate[batch]  # (total_nodes, E)
        h_gated = h_geo * (1.0 + gate_per_node)

        preds = self.head(h_gated)  # (total_nodes, 1)

        return {
            "predictions": preds,
            "embeddings": h_geo,
            "regime_probs": regime_probs,
            "density_matrix": rho,
        }


class GraphQuantumModel(nn.Module):
    """A7: Graph + Quantum (no geometric/manifold).

    Graph layer operates on raw features; quantum regime gating is
    applied to the graph output.

    Args:
        feature_dim: Input feature dimension.
        hidden_channels: GCN intermediate dimension.
        out_channels: GNN output dimension.
        gat_heads: Number of GAT attention heads.
        num_regimes: Number of market regimes.
        context_dim: Yield curve context dimension.
        prediction_hidden_dim: Prediction head hidden size.
        dropout: Feature dropout rate.
        edge_drop: DropEdge probability.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_channels: int = 64,
        out_channels: int = 32,
        gat_heads: int = 4,
        num_regimes: int = 4,
        context_dim: int = 3,
        prediction_hidden_dim: int = 16,
        dropout: float = 0.1,
        edge_drop: float = 0.1,
    ) -> None:
        super().__init__()
        self.graph = GraphDependencyLayer(
            in_channels=feature_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=gat_heads,
            dropout=dropout,
            edge_drop=edge_drop,
        )
        # Quantum operates on pooled raw features
        self.quantum = QuantumRegimeDetector(
            input_dim=feature_dim,
            context_dim=context_dim,
            num_regimes=num_regimes,
        )
        self.regime_gate = nn.Sequential(
            nn.Linear(num_regimes, out_channels),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(out_channels, prediction_hidden_dim),
            nn.ELU(),
            nn.Linear(prediction_hidden_dim, 1),
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
        """Forward: graph on raw features -> quantum gating -> head.

        Args:
            x: Node features (total_nodes, feature_dim).
            edge_index: Block-diagonal graph connectivity (2, E).
            edge_weight: Edge weights (E,) or None.
            context: Yield curve context (B, context_dim).
            batch: Graph membership vector (total_nodes,).

        Returns:
            Predictions, dummy embeddings, and density matrix.
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        n = x.size(0)
        h_graph = self.graph(x, edge_index, edge_weight)  # (total_nodes, G)

        # Quantum on pooled raw features
        h_global = global_mean_pool(x, batch)  # (B, D)
        regime_probs, rho = self.quantum(h_global, context)

        gate = self.regime_gate(regime_probs)  # (B, G)
        gate_per_node = gate[batch]  # (total_nodes, G)
        h_fused = h_graph * (1.0 + gate_per_node)

        preds = self.head(h_fused)  # (total_nodes, 1)

        return {
            "predictions": preds,
            "embeddings": _dummy_embeddings(n, x),
            "regime_probs": regime_probs,
            "density_matrix": rho,
        }


class LSTMBaseline(nn.Module):
    """A9: LSTM Baseline on raw return sequences.

    Processes historical return sequences (x_seq) to predict the next
    return. Does not use the engineered features (x), graph, or quantum
    layers.

    Args:
        input_dim: Dimension of input sequence (1 for log returns).
        hidden_dim: LSTM hidden state dimension.
        num_layers: Number of LSTM layers.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        context: torch.Tensor,
        batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Forward pass using x_seq from kwargs.

        Args:
            x: Unused (engineered features).
            edge_index: Unused.
            edge_weight: Unused.
            context: Unused.
            batch: Unused (graph membership vector).
            **kwargs: Must contain 'x_seq' (total_nodes, T, 1).

        Returns:
            Predictions and dummy embeddings/density matrix.
        """
        x_seq = kwargs.get("x_seq")
        if x_seq is None:
            raise ValueError(
                "LSTMBaseline requires 'x_seq' in input. "
                "Ensure FinancialGraphDataset is created with sequence_length > 0."
            )

        # The universe can have >20k tickers per date, so a full batch can
        # contain over 1M sequences. Processing in chunks of 10k keeps the
        # cuDNN workspace within safe VRAM limits. Using h_n[-1] (the final
        # hidden state) avoids allocating the full (chunk, T, H) output.
        _CHUNK = 10_000
        all_h_last: list[torch.Tensor] = []
        for i in range(0, x_seq.size(0), _CHUNK):
            _, (h_n, _) = self.lstm(x_seq[i : i + _CHUNK])
            all_h_last.append(h_n[-1])  # (chunk, H)

        h_final = torch.cat(all_h_last, dim=0)  # (total_nodes, H)
        preds = self.head(h_final)  # (total_nodes, 1)

        n = x.size(0)
        return {
            "predictions": preds,
            "embeddings": _dummy_embeddings(n, x),
            "regime_probs": _dummy_regime_probs(x, batch),
            "density_matrix": _dummy_density_matrix(x),
        }


def _estimate_full_model_params(config: dict) -> int:
    """Estimates the parameter count of the full model (A8).

    Args:
        config: Full configuration dictionary.

    Returns:
        Approximate number of trainable parameters.
    """
    geo = config["model"]["geometric"]
    qm = config["model"]["quantum"]
    gr = config["model"]["graph"]
    pred = config["model"]["prediction"]

    dummy = GDLQuantumFinanceModel(
        feature_dim=geo["input_dim"],
        embedding_dim=geo["embedding_dim"],
        hidden_dim=gr["hidden_channels"],
        graph_out_dim=gr["out_channels"],
        num_regimes=qm["num_regimes"],
        gat_heads=gr["gat_heads"],
        context_dim=qm["context_dim"],
        prediction_hidden_dim=pred["hidden_dim"],
        dropout=gr.get("dropout", 0.1),
        edge_drop=gr.get("edge_drop", 0.1),
    )
    return sum(p.numel() for p in dummy.parameters() if p.requires_grad)


def build_ablation_model(
    config_id: str,
    config: dict,
    device: torch.device,
) -> nn.Module:
    """Factory: builds the model for a given ablation configuration.

    Constructs one of the nine ablation model variants (A0--A8) as
    for the ablation study.

    Args:
        config_id: One of 'A0' through 'A8'.
        config: Full configuration dictionary.
        device: Target compute device.

    Returns:
        Model instance placed on the specified device.

    Raises:
        ValueError: If config_id is not a valid ablation identifier.
    """
    variant_id = config_id
    config_id, flags = _resolve_variant(config_id)

    geo = config["model"]["geometric"]
    qm = config["model"]["quantum"]
    gr = config["model"]["graph"]
    pred = config["model"]["prediction"]

    feature_dim: int = geo["input_dim"]
    embedding_dim: int = geo["embedding_dim"]
    hidden_channels: int = gr["hidden_channels"]
    out_channels: int = gr["out_channels"]
    gat_heads: int = gr["gat_heads"]
    num_regimes: int = qm["num_regimes"]
    context_dim: int = qm["context_dim"]
    pred_hidden: int = pred["hidden_dim"]
    dropout: float = gr.get("dropout", 0.1)
    edge_drop: float = gr.get("edge_drop", 0.1)

    model: nn.Module

    if config_id == "A0":
        model = LinearBaseline(feature_dim=feature_dim)

    elif config_id == "A1":
        # Match parameter budget to full model (A8)
        target_params = _estimate_full_model_params(config)
        # Solve for hidden_dim: feature_dim*H + H + H*H + H + H*1 + 1
        # Approximate: 2*feature_dim*H + H^2 + 3H + 1 ~ target_params
        # Use a reasonable approximation
        hidden_dim = max(
            16,
            int(np.sqrt(max(1, target_params - feature_dim))),
        )
        model = EuclideanMLP(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
        )

    elif config_id == "A2":
        model = GeometricOnlyModel(
            feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            prediction_hidden_dim=pred_hidden,
            **flags,
        )

    elif config_id == "A3":
        model = GraphOnlyModel(
            feature_dim=feature_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            gat_heads=gat_heads,
            prediction_hidden_dim=pred_hidden,
            dropout=dropout,
            edge_drop=edge_drop,
        )

    elif config_id == "A4":
        model = QuantumOnlyModel(
            feature_dim=feature_dim,
            num_regimes=num_regimes,
            context_dim=context_dim,
            prediction_hidden_dim=pred_hidden,
        )

    elif config_id == "A5":
        model = GeometricGraphModel(
            feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            gat_heads=gat_heads,
            prediction_hidden_dim=pred_hidden,
            dropout=dropout,
            edge_drop=edge_drop,
        )

    elif config_id == "A6":
        model = GeometricQuantumModel(
            feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            num_regimes=num_regimes,
            context_dim=context_dim,
            prediction_hidden_dim=pred_hidden,
            **flags,
        )

    elif config_id == "A7":
        model = GraphQuantumModel(
            feature_dim=feature_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            gat_heads=gat_heads,
            num_regimes=num_regimes,
            context_dim=context_dim,
            prediction_hidden_dim=pred_hidden,
            dropout=dropout,
            edge_drop=edge_drop,
        )

    elif config_id == "A8":
        model = GDLQuantumFinanceModel(
            feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_channels,
            graph_out_dim=out_channels,
            num_regimes=num_regimes,
            gat_heads=gat_heads,
            context_dim=context_dim,
            prediction_hidden_dim=pred_hidden,
            dropout=dropout,
            edge_drop=edge_drop,
            **flags,
        )

    elif config_id == "A9":
        model = LSTMBaseline(
            input_dim=1,
            hidden_dim=hidden_channels,
            num_layers=2,
        )

    else:
        raise ValueError(f"Unknown ablation configuration: {config_id}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Built %s (%s): %d trainable parameters on %s.",
        variant_id,
        ABLATION_DESCRIPTIONS.get(variant_id, ""),
        n_params,
        device,
    )
    return model


def _build_criterion_for_config(
    config_id: str,
    config: dict,
) -> GDLQuantumLoss:
    """Builds the loss function, disabling auxiliary terms for baselines.

    For models without geometric embeddings (A0, A1, A3, A4, A7),
    uniformity loss is zeroed out.  For models without a quantum layer
    (A0, A1, A2, A3, A5), regime entropy regularisation is zeroed out.

    Args:
        config_id: Ablation identifier.
        config: Full configuration dictionary.

    Returns:
        Appropriately weighted loss function.
    """
    # Reuse profile resolution from train.py to respect 'loss_profile'
    from src.train import _resolve_loss_weights

    weights, profile_name, _ = _resolve_loss_weights(config)
    weights = dict(weights)
    if config_id in ABLATION_VARIANTS:
        weights.update(ABLATION_VARIANTS[config_id][2])
    config_id, _ = _resolve_variant(config_id)

    # Read max points from config (same as main training loop)
    training_cfg = config.get("training", {})
    uniformity_max_points = training_cfg.get("uniformity_max_points", 2048)

    w_pred = weights.get("prediction", 1.0)
    w_uniform = weights.get("uniformity", 0.01)
    w_dm = weights.get("density_matrix", 0.001)
    w_dir = weights.get("directional", 0.1)

    logger.info(
        "%s using loss profile '%s': pred=%.2f, uni=%.4f, dm=%.4f, dir=%.2f",
        config_id,
        profile_name,
        w_pred,
        w_uniform,
        w_dm,
        w_dir,
    )

    # Intent: Fair Comparison.
    # Baselines (A0, A1) and partial models (A3, A7) don't output valid
    # geometric embeddings or density matrices. We must disable the
    # auxiliary loss terms (Uniformity, Regime Entropy Regularisation) for them,
    # otherwise they would be penalised for outputs they cannot produce.
    # Disable uniformity for models without real manifold embeddings
    has_geometric = config_id in ("A2", "A5", "A6", "A8")
    if not has_geometric:
        w_uniform = 0.0

    # Disable density matrix reg for models without quantum layer
    has_quantum = config_id in ("A4", "A6", "A7", "A8")
    if not has_quantum:
        w_dm = 0.0

    return GDLQuantumLoss(
        w_pred=w_pred,
        w_uniform=w_uniform,
        w_dm=w_dm,
        w_dir=w_dir,
        uniformity_max_points=uniformity_max_points,
    )


def train_ablation_fold(
    config_id: str,
    fold_idx: int,
    train_dates: list[pd.Timestamp],
    val_dates: list[pd.Timestamp],
    config: dict,
    device: torch.device,
    seed: int,
    dataframe: pd.DataFrame | None = None,
    parquet_path: Path | None = None,
    use_wandb: bool = False,
    experiment_id: str | None = None,
    control: str | None = None,
    dump_predictions: bool = False,
    inner_val_days: int | None = None,
    experiments_dir: Path | None = None,
    stop_metric: str = "total",
    crn: bool = False,
    base_dataset: FinancialGraphDataset | None = None,
    checkpoint_root: Path | None = None,
) -> dict[str, Any]:
    """Trains and evaluates a single ablation configuration on one fold.

    Args:
        config_id: Ablation identifier (e.g. 'A0', 'A8').
        fold_idx: Zero-based fold index.
        train_dates: Training date window.
        val_dates: Validation date window.
        config: Full configuration dictionary.
        device: Compute device.
        seed: Random seed for reproducibility.
        dataframe: Pre-loaded DataFrame (avoids reloading parquet each fold).
        parquet_path: Path to fact_table.parquet.
        use_wandb: Whether to log to Weights & Biases.
        experiment_id: Optional ID to group configurations in W&B.
        control: Optional negative control (data_loader.NEGATIVE_CONTROLS).
            'shuffle_labels' is applied to training data only, so evaluation
            is always against true returns; graph controls apply to both.
        dump_predictions: Save per-stock evaluation predictions to parquet.
        inner_val_days: If set, early stopping uses the last N training dates
            (after a purge gap) and the validation window is used only for the
            reported evaluation. None keeps the original protocol, in which
            the validation window serves both purposes.
        stop_metric: 'total' (original: the full weighted loss, which includes
            the uniformity/entropy terms only for some configs) or 'supervised'
            (w_pred * MSE + w_dir * directional, identical for every config).
        crn: Common random numbers: shuffle with a dedicated generator seeded by
            ``seed`` so every config sees the same batch order.
        base_dataset: Already loaded dataset over all dates; the fold's
            datasets become views of it instead of fresh copies of the panel.
        checkpoint_root: Checkpoint directory (default output/checkpoints/ablation).

    Returns:
        Dictionary with fold results including metrics.
    """
    if use_wandb:
        w_cfg = config.get("wandb", {})

        # Determine local directory for W&B logs for this config
        wandb_root = experiments_dir or _PROJECT_ROOT / config.get("evaluation", {}).get(
            "experiments_dir", "experiments"
        )
        wandb_dir = wandb_root / config_id / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)

        wandb.init(
            project=w_cfg.get("project", "gdl-quantum-finance"),
            entity=w_cfg.get("entity"),
            dir=str(wandb_dir),
            tags=w_cfg.get("tags", []) + ["ablation", config_id],
            mode=w_cfg.get("mode", "online"),
            group=config_id,
            job_type="ablation",
            name=f"{config_id}_fold_{fold_idx:02d}_seed_{seed}{tag}",
            config={**config, "ablation_id": config_id, "seed": seed},
            reinit=True,
        )

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    base_id, _ = _resolve_variant(config_id)
    if inner_val_days is not None and inner_val_days <= 0:
        raise ValueError("inner_val_days must be a positive number of dates.")
    # Controls and protocol options get their own tag so their files never
    # overwrite the checkpoints and dumps of the original runs.
    tag = (
        (f"_{control}" if control else "")
        + (f"_iv{inner_val_days}" if inner_val_days is not None else "")
        + ("_sup" if stop_metric == "supervised" else "")
        + ("_crn" if crn else "")
    )

    t_cfg = config["training"]
    max_epochs: int = t_cfg["max_epochs"]
    patience: int = t_cfg["early_stopping_patience"]
    min_delta: float = t_cfg.get("min_delta", 0.0001)
    batch_size: int = t_cfg.get("batch_size", 64)

    logger.info(
        "INFO: Fold %d | Batch Size: %d | Min Delta: %.6f",
        fold_idx,
        batch_size,
        min_delta,
    )

    logger.info(
        "%s | Fold %d | Seed %d: training %d dates, validating %d dates.",
        config_id,
        fold_idx,
        seed,
        len(train_dates),
        len(val_dates),
    )

    # Build datasets, reusing the shared dataframe to avoid redundant
    # parquet loads per fold.  Graph construction is deferred to the
    # GPU at training time (build_dynamic_graph_gpu in _forward_step).
    common_kwargs: dict[str, Any] = {"config": config}
    if dataframe is not None:
        common_kwargs["dataframe"] = dataframe
        common_kwargs["parquet_path"] = "unused"
    elif parquet_path is not None:
        common_kwargs["parquet_path"] = parquet_path
    else:
        common_kwargs["parquet_path"] = _PROJECT_ROOT / config["data"]["parquet_path"]

    if base_dataset is None:
        base_dataset = FinancialGraphDataset(**common_kwargs)
    variant_attrs = VARIANT_DATASET_KWARGS.get(config_id, {})
    train_control = control
    eval_control = None if control == "shuffle_labels" else control
    stop_dates = val_dates
    if inner_val_days is not None:
        purge = int(config["training"]["walk_forward"].get("purge_gap", 20))
        stop_dates = train_dates[-inner_val_days:]
        train_dates = train_dates[: -(inner_val_days + purge)]
        logger.info(
            "%s | Fold %d: inner validation uses %d training dates for fitting and %d for early stopping.",
            config_id,
            fold_idx,
            len(train_dates),
            len(stop_dates),
        )
    train_ds = dataset_view(base_dataset, train_dates, train_control, seed, **variant_attrs)
    val_ds = dataset_view(base_dataset, val_dates, eval_control, seed, **variant_attrs)
    # Early stopping must see the same (possibly shuffled) labels as training,
    # otherwise selection on true labels could leak signal into a null control.
    stop_ds = (
        val_ds
        if inner_val_days is None and train_control == eval_control
        else dataset_view(base_dataset, stop_dates, train_control, seed, **variant_attrs)
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        logger.warning(
            "%s | Fold %d | Seed %d: empty dataset. Skipping.",
            config_id,
            fold_idx,
            seed,
        )
        return {
            "config_id": config_id,
            "fold": fold_idx,
            "seed": seed,
            "skipped": True,
        }

    # Build model and loss
    model = build_ablation_model(config_id, config, device)
    criterion = _build_criterion_for_config(config_id, config)

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=max_epochs,
        eta_min=t_cfg["learning_rate"] * 0.01,
    )
    early_stop = EarlyStopping(patience=patience, min_delta=min_delta)

    # AMP (mixed-precision) for CUDA tensor cores
    use_amp = t_cfg.get("use_amp", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    graph_cfg = config.get("model", {}).get("graph_construction", {})
    graph_threshold = float(graph_cfg["correlation_threshold"])
    max_neighbours_cfg = graph_cfg.get("max_neighbours")
    max_neighbours = int(max_neighbours_cfg) if max_neighbours_cfg is not None else None

    # Skip GPU correlation graph build for configs that do not use the graph.
    # Speeds up A0, A1, A2, A4, A6, A9 significantly.
    use_graph = base_id in ("A3", "A5", "A7", "A8")

    shuffle_gen = torch.Generator().manual_seed(seed) if crn else None

    # Build DataLoaders once (workers persist across all epochs in this fold).
    # A9 processes ~20k ticker sequences per date, so a full batch of 64 dates
    # requires ~40 GiB of LSTM activation memory for gradient storage.
    # We override to a smaller batch size for A9 only; all other configs use
    # the value from config.yaml unchanged.
    if base_id == "A9":
        lstm_batch = int(config.get("training", {}).get("lstm_batch_size", 8))
        loader_cfg = {**config, "training": {**config.get("training", {}), "batch_size": lstm_batch}}
        logger.info("A9: using lstm_batch_size=%d (main batch_size=%d).", lstm_batch, batch_size)
        train_loader = _make_dataloader(train_ds, loader_cfg, shuffle=True, generator=shuffle_gen)
        val_loader = _make_dataloader(stop_ds, loader_cfg, shuffle=False)
    else:
        train_loader = _make_dataloader(train_ds, config, shuffle=True, generator=shuffle_gen)
        val_loader = _make_dataloader(stop_ds, config, shuffle=False)

    best_model_state: dict | None = None
    fold_start = time.time()

    for epoch in range(max_epochs):
        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimiser,
            device,
            scaler=scaler,
            use_graph=use_graph,
            graph_threshold=graph_threshold,
            max_neighbours=max_neighbours,
        )
        val_metrics = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            use_graph=use_graph,
            graph_threshold=graph_threshold,
            max_neighbours=max_neighbours,
        )
        current_lr = optimiser.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        logger.info(
            "%s | Fold %d | Epoch %3d/%d | "
            "Train loss=%.6f (pred=%.6f, uni=%.4f, dm=%.6f, dir=%.4f) | "
            "Val loss=%.6f (pred=%.6f) | "
            "LR=%.2e | %.1fs",
            config_id,
            fold_idx,
            epoch + 1,
            max_epochs,
            train_metrics["total"],
            train_metrics["prediction"],
            train_metrics["uniformity"],
            train_metrics["density_matrix"],
            train_metrics["directional"],
            val_metrics["total"],
            val_metrics["prediction"],
            current_lr,
            epoch_time,
        )

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_metrics["total"],
                    "train/prediction": train_metrics["prediction"],
                    "train/uniformity": train_metrics["uniformity"],
                    "train/density_matrix": train_metrics["density_matrix"],
                    "train/directional": train_metrics["directional"],
                    "val/loss": val_metrics["total"],
                    "val/prediction": val_metrics["prediction"],
                    "learning_rate": current_lr,
                    "config_id": config_id,
                }
            )

        stop_value = (
            val_metrics["total"]
            if stop_metric == "total"
            else criterion.w_pred * val_metrics["prediction"] + criterion.w_dir * val_metrics["directional"]
        )
        if early_stop(stop_value, epoch):
            break

        if early_stop.best_epoch == epoch:
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

        if train_metrics["optimiser_steps"] > 0:
            scheduler.step()
        else:
            logger.warning(
                "%s | Fold %d | Epoch %d: skipping scheduler.step() because no "
                "optimiser step occurred.",
                config_id,
                fold_idx,
                epoch + 1,
            )

    fold_time = time.time() - fold_start

    # Checkpoint and PnL export paths (resolved once per fold). Controls and the
    # inner-validation protocol get their own tag so they never overwrite the
    # checkpoints of the original runs.
    safe_model_name = _get_config_name(config_id)

    checkpoint_dir = (checkpoint_root or _PROJECT_ROOT / "output" / "checkpoints" / "ablation") / config_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        checkpoint_dir
        / f"{config_id}_{safe_model_name}_fold_{fold_idx:02d}_seed_{seed}{tag}.pt"
    )

    experiments_dir = experiments_dir or _PROJECT_ROOT / config.get("evaluation", {}).get(
        "experiments_dir", "experiments"
    )
    debug_dir = experiments_dir / config_id / "debug_pnl"
    debug_dir.mkdir(parents=True, exist_ok=True)
    dump_pnl_path = str(
        debug_dir
        / f"pnl_{config_id}_{safe_model_name}_fold_{fold_idx:02d}_seed_{seed}{tag}.npy"
    )

    # Restore best weights and save the checkpoint
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

        torch.save(
            {
                "config_id": config_id,
                "model_name": safe_model_name,
                "fold": fold_idx,
                "seed": seed,
                "epoch": early_stop.best_epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_loss": early_stop.best_loss,
            },
            checkpoint_path,
        )
        logger.info("Saved ablation model checkpoint to %s", checkpoint_path)

    dump_predictions_path = (
        experiments_dir / "predictions" / f"pred_{config_id}_fold_{fold_idx:02d}_seed_{seed}{tag}.parquet"
        if dump_predictions
        else None
    )

    # Evaluate
    eval_metrics = evaluate_model_on_dataset(
        model,
        val_ds,
        device,
        debug_pnl=True,
        dump_pnl_path=dump_pnl_path,
        dump_predictions_path=dump_predictions_path,
    )

    logger.info(
        "INFO: %s | Fold %d | Seed %d: IC=%.4f, DirAcc=%.2f%%, "
        "Sharpe(daily)=%.2f, Sharpe(annual)=%.2f, MSE=%.6f, MaxDD=%.4f",
        config_id,
        fold_idx + 1,
        seed,
        eval_metrics["ic"],
        eval_metrics["directional_accuracy"],
        eval_metrics["sharpe_daily"],
        eval_metrics["sharpe_annual"],
        eval_metrics["mse"],
        eval_metrics.get("max_drawdown", float("nan")),
    )

    if use_wandb:
        wandb.log(
            {
                "eval/training_time_s": fold_time,
                "eval/best_epoch": early_stop.best_epoch + 1,
                **{f"eval/{k}": v for k, v in eval_metrics.items()},
            }
        )
        wandb.finish()

    return {
        "config_id": config_id,
        "fold": fold_idx,
        "seed": seed,
        "skipped": False,
        "control": control or "",
        "inner_val_days": inner_val_days or 0,
        "stop_metric": stop_metric,
        "crn": crn,
        "best_epoch": early_stop.best_epoch + 1,
        "best_val_loss": early_stop.best_loss,
        "training_time_s": fold_time,
        "metrics": eval_metrics,
    }


def run_ablation(
    config: dict,
    config_ids: list[str] | None = None,
    smoke_test: bool = False,
    max_folds: int | None = None,
    device_override: str | None = None,
    use_wandb: bool = False,
    experiments_dir: Path | None = None,
    seeds: list[int] | None = None,
    control: str | None = None,
    dump_predictions: bool = False,
    inner_val_days: int | None = None,
    stop_metric: str = "total",
    crn: bool = False,
    checkpoint_root: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Runs the full ablation study across configurations and seeds.

    Args:
        config: Full configuration dictionary.
        config_ids: Subset of ablation IDs to run (default: all A0--A9).
        smoke_test: If True, uses synthetic data with reduced epochs.
        max_folds: Maximum number of walk-forward folds.
        device_override: Force a specific compute device.
        use_wandb: Whether to log to Weights & Biases.
        experiments_dir: Base directory to save experiment logs and outputs.

    Returns:
        Dictionary mapping config_id to list of per-(fold, seed) results.
    """
    if config_ids is None:
        config_ids = list(ABLATION_IDS)

    # Resolve device
    if device_override is not None:
        device = torch.device(device_override)
    else:
        device = _validate_device(get_device())

    # Unique experiment ID for grouping ablation runs
    experiment_id = f"ablation_{int(time.time())}" if use_wandb else None

    if seeds is None:
        seeds = config.get("evaluation", {}).get(
            "ablation_seeds",
            [42, 123, 456, 789, 1024],
        )

    # Load or synthesise data once; reuse across all folds to avoid
    # repeatedly reading the 44M-row parquet file.
    parquet_path = _PROJECT_ROOT / config["data"]["parquet_path"]
    dataframe: pd.DataFrame | None = None

    if smoke_test or not parquet_path.exists():
        if not smoke_test:
            logger.warning(
                "Parquet file not found at %s. "
                "Falling back to synthetic smoke-test data.",
                parquet_path,
            )
        logger.info("Generating synthetic data for ablation smoke test.")
        dataframe = build_synthetic_dataframe(
            n_tickers=20,
            n_dates=400,
            seed=42,
        )
    else:
        logger.info("Loading parquet once for all folds: %s", parquet_path)
        dataframe = pd.read_parquet(parquet_path)

    # Build a dataset once to obtain the valid date universe.
    # dataframe is always set at this point (either synthetic or from parquet).
    full_ds = FinancialGraphDataset(
        parquet_path="unused",
        config=config,
        dataframe=dataframe,
    )

    # Walk-forward splitter
    wf_cfg = config["training"]["walk_forward"]
    splitter = WalkForwardSplitter(
        dates=full_ds.valid_dates,
        train_window=wf_cfg["train_window"],
        validation_window=wf_cfg["validation_window"],
        step_size=wf_cfg["step_size"],
        purge_gap=int(wf_cfg.get("purge_gap", 20)),
    )

    # Handle insufficient data (smoke test fallback)
    folds: list[tuple[list[pd.Timestamp], list[pd.Timestamp]]] = []
    if len(splitter) == 0:
        n_valid = len(full_ds.valid_dates)
        if n_valid >= 4:
            split_point = max(2, n_valid * 3 // 4)
            folds.append(
                (
                    full_ds.valid_dates[:split_point],
                    full_ds.valid_dates[split_point:],
                )
            )
            logger.info(
                "Smoke-test fallback split: train=%d, val=%d dates.",
                len(folds[0][0]),
                len(folds[0][1]),
            )
        else:
            logger.error(
                "Insufficient data for ablation (%d dates).",
                n_valid,
            )
            return {}
    else:
        for train_dates, val_dates in splitter:
            folds.append((train_dates, val_dates))

    if max_folds is not None:
        folds = folds[:max_folds]
        if len(folds) == 0:
            logger.error(
                "max_folds=%d produced zero folds. Check walk_forward "
                "settings and available dates.",
                max_folds,
            )
            return {}

    logger.info(
        "Ablation study: %d configurations, %d seeds, %d folds " "(%d total runs).",
        len(config_ids),
        len(seeds),
        len(folds),
        len(config_ids) * len(seeds) * len(folds),
    )

    # Override epochs for smoke test
    run_config = copy.deepcopy(config)
    if smoke_test:
        run_config["training"]["max_epochs"] = min(
            3,
            config["training"]["max_epochs"],
        )

    # Run experiments
    all_results: dict[str, list[dict[str, Any]]] = {cid: [] for cid in config_ids}

    for cid in config_ids:
        # Set up a dedicated file logger for this configuration
        if experiments_dir is not None:
            cid_dir = experiments_dir / cid
            cid_dir.mkdir(parents=True, exist_ok=True)
            log_path = cid_dir / f"run_{cid}.log"
            file_handler = logging.FileHandler(log_path, mode="a")
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    "%Y-%m-%d %H:%M:%S",
                )
            )
            logging.getLogger().addHandler(file_handler)
        else:
            file_handler = None

        try:
            for seed in seeds:
                for fold_idx, (train_dates, val_dates) in enumerate(folds):
                    result = train_ablation_fold(
                        config_id=cid,
                        fold_idx=fold_idx,
                        train_dates=train_dates,
                        val_dates=val_dates,
                        config=run_config,
                        device=device,
                        seed=seed,
                        dataframe=dataframe,
                        parquet_path=parquet_path,
                        use_wandb=use_wandb,
                        experiment_id=experiment_id,
                        control=control,
                        dump_predictions=dump_predictions,
                        inner_val_days=inner_val_days,
                        experiments_dir=experiments_dir,
                        stop_metric=stop_metric,
                        crn=crn,
                        base_dataset=full_ds,
                        checkpoint_root=checkpoint_root,
                    )
                    all_results[cid].append(result)
        finally:
            if file_handler is not None:
                logging.getLogger().removeHandler(file_handler)
                file_handler.close()

    return all_results


def save_raw_results(
    all_results: dict[str, list[dict[str, Any]]],
    experiments_dir: Path,
) -> Path:
    """Saves raw ablation results to a CSV file.

    Args:
        all_results: Dictionary mapping config_id to result lists.
        experiments_dir: Target directory for experiment outputs.

    Returns:
        Path to the saved CSV file.
    """
    experiments_dir.mkdir(parents=True, exist_ok=True)
    csv_path = experiments_dir / "ablation_raw_results.csv"

    rows: list[dict[str, Any]] = []
    for cid, results in all_results.items():
        for r in results:
            if r.get("skipped", False):
                continue
            row = {
                "config_id": cid,
                "description": ABLATION_DESCRIPTIONS.get(cid, ""),
                "fold": r["fold"],
                "seed": r["seed"],
                "control": r.get("control", ""),
                "inner_val_days": r.get("inner_val_days", 0),
                "stop_metric": r.get("stop_metric", "total"),
                "crn": r.get("crn", False),
                "best_epoch": r.get("best_epoch", ""),
                "best_val_loss": r.get("best_val_loss", ""),
                "training_time_s": r.get("training_time_s", ""),
            }
            for metric_name, metric_val in r.get("metrics", {}).items():
                row[metric_name] = metric_val
            rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Raw results saved to %s (%d rows).", csv_path, len(rows))
    else:
        logger.warning("No results to save.")

    return csv_path


def save_summary_table(
    all_results: dict[str, list[dict[str, Any]]],
    experiments_dir: Path,
    metrics: list[str] | None = None,
) -> Path:
    """Computes and saves mean +/- std per (config, metric) over seeds.

    Args:
        all_results: Dictionary mapping config_id to result lists.
        experiments_dir: Target directory for experiment outputs.
        metrics: Metric names to include (default: all standard).

    Returns:
        Path to the saved summary CSV file.
    """
    if metrics is None:
        metrics = [
            "ic",
            "directional_accuracy",
            "mse",
            "mae",
            "sharpe_daily",
            "sharpe_annual",
            "max_drawdown",
        ]

    experiments_dir.mkdir(parents=True, exist_ok=True)
    csv_path = experiments_dir / "ablation_summary.csv"

    rows: list[dict[str, Any]] = []
    for cid in ALL_IDS:
        if cid not in all_results:
            continue
        results = [r for r in all_results[cid] if not r.get("skipped", False)]
        if not results:
            continue

        row: dict[str, Any] = {
            "config_id": cid,
            "description": ABLATION_DESCRIPTIONS.get(cid, ""),
            "n_runs": len(results),
        }

        for metric in metrics:
            values = [
                r["metrics"][metric]
                for r in results
                if metric in r.get("metrics", {}) and np.isfinite(r["metrics"][metric])
            ]
            if values:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values, ddof=1))
            else:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")

        rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Summary table saved to %s.", csv_path)

    return csv_path


def compute_significance(
    all_results: dict[str, list[dict[str, Any]]],
    experiments_dir: Path,
    config: dict,
) -> Path:
    """Computes paired t-tests and bootstrap CIs for A8 vs each ablation.

    For every metric, computes a paired t-test between A8's per-fold
    values and each ablation's per-fold values.  Also computes 95%
    bootstrap confidence intervals for the Sharpe ratio difference
    (A8 minus ablation).

    Args:
        all_results: Dictionary mapping config_id to result lists.
        experiments_dir: Target directory for outputs.
        config: Full configuration dictionary.

    Returns:
        Path to the saved significance report CSV.
    """
    eval_cfg = config.get("evaluation", {})
    n_resamples = eval_cfg.get("bootstrap_resamples", 1000)
    alpha = eval_cfg.get("significance_alpha", 0.05)
    metrics = [
        "mse",
        "mae",
        "ic",
        "directional_accuracy",
        "sharpe_daily",
        "sharpe_annual",
        "max_drawdown",
    ]

    experiments_dir.mkdir(parents=True, exist_ok=True)
    report_path = experiments_dir / "significance_report.csv"

    if "A8" not in all_results:
        logger.warning("A8 not in results; cannot compute significance.")
        return report_path

    a8_results = [r for r in all_results["A8"] if not r.get("skipped", False)]
    if not a8_results:
        logger.warning("No completed A8 runs; skipping significance.")
        return report_path

    rows: list[dict[str, Any]] = []

    for cid in ALL_IDS:
        if cid == "A8" or cid not in all_results:
            continue
        abl_results = [r for r in all_results[cid] if not r.get("skipped", False)]
        if not abl_results:
            continue

        for metric in metrics:
            # Pair on (fold, seed) so skipped folds or extra seeds cannot
            # misalign the two series.
            a8_by_key = {
                (r["fold"], r["seed"]): r["metrics"][metric]
                for r in a8_results
                if metric in r.get("metrics", {})
            }
            abl_by_key = {
                (r["fold"], r["seed"]): r["metrics"][metric]
                for r in abl_results
                if metric in r.get("metrics", {})
            }
            keys = sorted(set(a8_by_key) & set(abl_by_key))
            if len(keys) < 2:
                continue

            a8_arr = np.array([a8_by_key[k] for k in keys])
            abl_arr = np.array([abl_by_key[k] for k in keys])

            t_result = paired_t_test(a8_arr, abl_arr)
            significant = t_result["p_value"] < alpha

            row: dict[str, Any] = {
                "comparison": f"A8 vs {cid}",
                "metric": metric,
                "a8_mean": float(np.nanmean(a8_arr)),
                "ablation_mean": float(np.nanmean(abl_arr)),
                "diff_mean": float(np.nanmean(a8_arr - abl_arr)),
                "t_statistic": t_result["t_statistic"],
                "p_value": t_result["p_value"],
                "significant": significant,
            }

            # Bootstrap CI for the daily Sharpe difference.
            if metric == "sharpe_daily":
                boot = bootstrap_confidence_interval(
                    a8_arr,
                    abl_arr,
                    n_resamples=n_resamples,
                    confidence_level=1.0 - alpha,
                )
                row["boot_ci_lower"] = boot["ci_lower"]
                row["boot_ci_upper"] = boot["ci_upper"]
                row["boot_mean_diff"] = boot["mean_diff"]

            rows.append(row)

    if rows:
        # Collect all possible fieldnames across rows (some rows have
        # bootstrap fields that others do not).
        all_fields: list[str] = []
        seen: set[str] = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    all_fields.append(k)
                    seen.add(k)

        with open(report_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=all_fields,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Significance report saved to %s.", report_path)
    else:
        logger.warning("No significance comparisons computed.")

    return report_path


def _configure_logging() -> None:
    """Configures the root logger with the project-standard format."""
    log_dir = _PROJECT_ROOT / "experiments"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                log_dir / "ablation_run.log", mode="a", encoding="utf-8"
            ),
        ],
    )


def _resolve_protocol(
    args: argparse.Namespace, config: dict
) -> tuple[int | None, str, bool, list[int] | None]:
    """Protocol settings: CLI flags, else training.protocol, else the dissertation's.

    Returns:
        (inner_val_days, stop_metric, crn, seeds). seeds is None when the
        config's evaluation.ablation_seeds should be used.
    """
    if args.original_protocol:
        return None, "total", False, args.seeds or [42]
    protocol = config.get("training", {}).get("protocol", {})
    inner_val_days = args.inner_val_days if args.inner_val_days is not None else protocol.get("inner_val_days")
    stop_metric = args.stop_metric or protocol.get("stop_metric", "total")
    crn = args.crn if args.crn is not None else bool(protocol.get("common_random_numbers", False))
    logger.info(
        "Protocol: inner_val_days=%s, stop_metric=%s, common_random_numbers=%s.",
        inner_val_days,
        stop_metric,
        crn,
    )
    return inner_val_days, stop_metric, crn, args.seeds


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="GDL-Quantum-Finance ablation experiment runner.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: project root).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run with synthetic data for quick validation.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        choices=ALL_IDS,
        help="Subset of ablation configs to run (default: all).",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Maximum number of walk-forward folds.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max_epochs from config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Force a specific compute device.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override evaluation.ablation_seeds from config.",
    )
    parser.add_argument(
        "--original-protocol",
        action="store_true",
        help="Reproduce the dissertation protocol: early stopping on the validation "
        "window with the total loss, no common random numbers, seed 42.",
    )
    parser.add_argument(
        "--control",
        choices=["shuffle_labels", "permute_graph_nodes", "stale_graph", "inject_target"],
        default=None,
        help="Run a negative control (see data_loader.NEGATIVE_CONTROLS).",
    )
    parser.add_argument(
        "--dump-predictions",
        action="store_true",
        help="Save per-stock evaluation predictions for src/posthoc_eval.py.",
    )
    parser.add_argument(
        "--inner-val-days",
        type=int,
        default=None,
        help="Early-stop on the last N (purged) training dates instead of the "
        "evaluation window (default: training.protocol.inner_val_days).",
    )
    parser.add_argument(
        "--stop-metric",
        choices=["total", "supervised"],
        default=None,
        help="Early-stopping criterion (default: training.protocol.stop_metric). "
        "'supervised' uses the same prediction loss for every config; 'total' "
        "(dissertation) adds config-specific auxiliary terms.",
    )
    parser.add_argument(
        "--crn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Common random numbers: identical batch order across configs for a seed "
        "(default: training.protocol.common_random_numbers).",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=None,
        help="Output directory (default: evaluation.experiments_dir). Use one per "
        "experiment so result tables are not overwritten by later runs.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic kernels (warn-only: PyG scatter ops on CUDA "
        "remain non-deterministic, so graph configs are reproducible only up "
        "to floating-point summation order).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the ablation experiment runner."""
    _configure_logging()
    args = parse_args()

    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False

    if args.max_epochs is not None:
        config["training"]["max_epochs"] = args.max_epochs
        logger.info("Overriding max_epochs to %d.", args.max_epochs)

    inner_val_days, stop_metric, crn, seeds = _resolve_protocol(args, config)

    eval_cfg = config.get("evaluation", {})
    experiments_dir = args.experiments_dir or _PROJECT_ROOT / eval_cfg.get(
        "experiments_dir",
        "experiments",
    )

    # Determine W&B status
    # Priority: CLI flags > Config file > Default
    config_wandb = config.get("wandb", {})
    mode = config_wandb.get("mode", "online")

    if args.no_wandb:
        use_wandb = False
    elif args.wandb:
        use_wandb = True
    else:
        # Default behavior: if smoke_test, disable unless configured otherwise
        if args.smoke_test:
            use_wandb = False
            logger.info("Smoke test: W&B disabled by default (use --wandb to enable).")
        else:
            use_wandb = mode == "online"

    # If W&B is not installed, force disable
    if wandb is None:
        if use_wandb:
            logger.warning("Weights & Biases is not installed. Disabling logging.")
        use_wandb = False

    logger.info(
        "GDL-Quantum-Finance Ablation Study | configs=%s | "
        "smoke_test=%s | max_folds=%s | wandb=%s",
        args.configs or "all (A0--A9)",
        args.smoke_test,
        args.max_folds,
        use_wandb,
    )

    all_results = run_ablation(
        config=config,
        config_ids=args.configs,
        smoke_test=args.smoke_test,
        max_folds=args.max_folds,
        device_override=args.device,
        use_wandb=use_wandb,
        experiments_dir=experiments_dir,
        seeds=seeds,
        control=args.control,
        dump_predictions=args.dump_predictions,
        inner_val_days=inner_val_days,
        stop_metric=stop_metric,
        crn=crn,
        checkpoint_root=args.experiments_dir / "checkpoints" if args.experiments_dir else None,
    )

    if not all_results:
        logger.error("Ablation study produced no results.")
        return

    # Per-config summary (mean metrics over folds/seeds) for quick comparison
    metric_names = [
        "ic",
        "directional_accuracy",
        "mse",
        "mae",
        "sharpe_daily",
        "sharpe_annual",
        "max_drawdown",
    ]
    for cid in all_results:
        results = [r for r in all_results[cid] if not r.get("skipped", False)]
        if not results:
            continue
        means = {}
        for m in metric_names:
            vals = [
                r["metrics"][m]
                for r in results
                if m in r.get("metrics", {}) and np.isfinite(r["metrics"].get(m))
            ]
            means[m] = float(np.mean(vals)) if vals else float("nan")
        logger.info(
            "%s | IC=%.4f DirAcc=%.2f%% Sharpe(daily)=%.2f "
            "Sharpe(annual)=%.2f MSE=%.6f MAE=%.6f MaxDD=%.4f",
            cid,
            means.get("ic", float("nan")),
            means.get("directional_accuracy", float("nan")),
            means.get("sharpe_daily", float("nan")),
            means.get("sharpe_annual", float("nan")),
            means.get("mse", float("nan")),
            means.get("mae", float("nan")),
            means.get("max_drawdown", float("nan")),
        )

    figures_dir = (
        args.experiments_dir / "figures"
        if args.experiments_dir
        else _PROJECT_ROOT / eval_cfg.get("figures_dir", "experiments/figures")
    )

    # Save outputs
    save_raw_results(all_results, experiments_dir)
    save_summary_table(all_results, experiments_dir)
    compute_significance(all_results, experiments_dir, config)

    # Generate diagnostic visualisations
    from src.diagnostics import generate_all_diagnostics

    generate_all_diagnostics(
        model=None,  # A8 model-based diagnostics require a loaded model
        dataset=None,
        device=None,
        experiments_dir=experiments_dir,
        figures_dir=figures_dir,
    )

    logger.info(
        "Ablation study complete. Results saved to %s.",
        experiments_dir,
    )


if __name__ == "__main__":
    main()
