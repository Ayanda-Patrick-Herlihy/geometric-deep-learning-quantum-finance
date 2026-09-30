"""Geometric Layer: Hypersphere Embedding for financial features.

Maps raw financial features onto the surface of a unit hypersphere so
that directional relationships between stocks are preserved. All
manifold operations (projection, distances, exponential/logarithmic
maps, Frechet mean) are implemented in pure PyTorch.

Theoretical Context:
    Standard Euclidean space fails to capture the latent geometry of financial
    markets. Following Wang & Isola (2020), we project features to a unit
    hypersphere to ensure Uniformity (preventing collapse) and Alignment
    (preserving similarity). We use Riemannian geometry (Bronstein et al., 2021)
    to perform rigorous operations (distances, means) on this curved surface.

References:
    Bronstein et al. (2021) "Geometric Deep Learning" arXiv:2104.13478.
    Miolane et al. (2020) "Geomstats" JMLR.
    Wang and Isola (2020) "Understanding Contrastive Representation
        Learning through Alignment and Uniformity on the Hypersphere."
        Section 3.1 ("Feature Distribution on the Hypersphere").
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses import _EPS

logger = logging.getLogger(__name__)

# geomstats is an optional dependency used only for validation and
# cross-checking against a reference manifold implementation.
# All training-loop operations use raw PyTorch for performance.
try:
    from geomstats.geometry.hypersphere import Hypersphere

    _GEOMSTATS_AVAILABLE = True
except ImportError:
    _GEOMSTATS_AVAILABLE = False
    logger.info("geomstats not available; reference manifold validation disabled.")
_MIN_PROJECTION_NORM: float = 1e-6
_REPLACEMENT_SCALE: float = 1e-3


def exponential_map(
    base: torch.Tensor,
    tangent: torch.Tensor,
    eps: float = _EPS,
) -> torch.Tensor:
    """Projects a tangent vector back onto the manifold surface.

    Takes a straight-line update in tangent space and wraps it along
    the curve of the hypersphere so the result stays on the surface.
    This is the standard Riemannian exponential map on S^{n-1}.

    Ref: Miolane et al. (2020), Section 2 and Table 1 ("Exponential and logarithmic maps").

    Args:
        base: Points on the manifold, shape (..., n).
        tangent: Tangent vectors at *base*, shape (..., n).
        eps: Stability constant to avoid division by zero.

    Returns:
        Points on S^{n-1} after the map, shape (..., n).
    """
    # Intent: "Lifting" from flat tangent space to the curved manifold.
    # We first calculate how far to travel along the geodesic arc (norm),
    # then use the closed-form Exp map for the sphere to project this
    # straight vector onto the surface.
    # Length of the tangent vector determines how far to travel
    norm = torch.linalg.norm(tangent, dim=-1, keepdim=True).clamp(min=eps)

    # Wrap the straight tangent vector along the curved surface:
    # interpolate between staying at base (cos) and moving towards
    # the tangent direction (sin)
    result = torch.cos(norm) * base + torch.sin(norm) * (tangent / norm)

    # Re-normalise to correct any floating-point drift off the manifold
    result = F.normalize(result, p=2, dim=-1, eps=eps)
    return result


def logarithmic_map(
    base: torch.Tensor,
    target: torch.Tensor,
    eps: float = _EPS,
) -> torch.Tensor:
    """Finds the tangent vector that connects two points on the manifold.

    Inverse of the exponential map. Given two points on the sphere,
    returns the direction and distance you would need to travel from
    *base* to reach *target* along the shortest path.

    Ref: Miolane et al. (2020), Section 2 and Table 1 ("Exponential and logarithmic maps").

    Args:
        base: Points on the manifold, shape (..., n).
        target: Points on the manifold, shape (..., n).
        eps: Stability constant.

    Returns:
        Tangent vectors at *base*, shape (..., n).
    """
    # Intent: "Flattening" the curved manifold into a tangent space.
    # We find the straight vector that, if projected, would reach the target.
    # This allows us to perform standard linear operations (like averaging)
    # in a flat space before projecting back.
    # Compute the great-circle distance between the two points
    dot = (base * target).sum(dim=-1, keepdim=True)
    dot = dot.clamp(-1.0 + eps, 1.0 - eps)
    dist = torch.acos(dot)

    # Guard against division by zero when base and target coincide
    sin_dist = torch.sin(dist).clamp(min=eps)

    # Compute the tangent vector pointing from base towards target,
    # scaled to the geodesic distance between them
    tangent = (dist / sin_dist) * (target - torch.cos(dist) * base)
    return tangent


def frechet_mean(
    points: torch.Tensor,
    max_iter: int = 100,
    tol: float = 1e-6,
    eps: float = _EPS,
) -> torch.Tensor:
    """Computes the average point on the hypersphere.

    The Euclidean mean does not live on the sphere, so we iteratively
    find the point that minimises total squared geodesic distance to
    all input points. Uses Riemannian gradient descent.

    Ref: Bachmann et al. (2020), Section 4.1 ("Manifold Batch Normalization").

    Args:
        points: Batch of points on S^{n-1}, shape (N, n).
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance on the tangent vector norm.
        eps: Stability constant.

    Returns:
        Frechet mean on S^{n-1}, shape (n,).
    """
    # Intent: Calculating the "Center of Mass" on a curved surface.
    # A simple Euclidean average would fall inside the sphere (off-manifold).
    # We use the iterative Karcher Mean algorithm to find the point on the surface
    # that minimises the sum of squared geodesic distances to all other points.
    # Start from the L2-normalised Euclidean mean as initial guess
    mean = F.normalize(points.mean(dim=0, keepdim=True), p=2, dim=-1, eps=eps)

    for iteration in range(max_iter):
        # Map all points into the tangent space at the current mean
        tangents = logarithmic_map(
            mean.expand_as(points),
            points,
            eps=eps,
        )

        # The average tangent vector is the Riemannian gradient
        avg_tangent = tangents.mean(dim=0, keepdim=True)

        # Check if we have converged
        tangent_norm = torch.linalg.norm(avg_tangent).item()
        if tangent_norm < tol:
            logger.debug(
                "Frechet mean converged after %d iterations "
                "(tangent norm=%.2e).",
                iteration + 1,
                tangent_norm,
            )
            break

        # Step along the manifold in the gradient direction
        mean = exponential_map(mean, avg_tangent, eps=eps)

    return mean.squeeze(0)


class GeometricEmbedding(nn.Module):
    """Maps raw financial features onto the unit hypersphere.

    A linear layer projects features into an ambient space, then L2
    normalisation places them on the sphere surface. This preserves
    angular relationships between stocks rather than absolute magnitudes.

    Ref: Wang and Isola (2020), Section 3.1 ("Feature Distribution on the Hypersphere").

    Args:
        input_dim: Number of input features per stock (D).
        output_dim: Ambient dimension of the hypersphere (n).
            The manifold itself has dimension n - 1.
        eps: Stability constant for normalisation.
        normalize: If False, skip the projection onto the sphere (control).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        eps: float = _EPS,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eps = eps
        # False gives the Euclidean twin used as a control: identical MLP,
        # no projection onto the sphere.
        self.normalize = normalize

        # Two-layer MLP projection: hidden layer adds non-linear expressiveness
        # before placing on the sphere
        hidden_dim = max(input_dim, output_dim)
        self.pre_projection = nn.Linear(input_dim, hidden_dim)
        self.projection = nn.Linear(hidden_dim, output_dim)

        # geomstats reference manifold for validation only,
        # not used in the training forward pass
        if _GEOMSTATS_AVAILABLE:
            self.manifold = Hypersphere(dim=output_dim - 1)
        else:
            self.manifold = None

        logger.info(
            "GeometricEmbedding initialised: D=%d -> S^{%d} (ambient R^%d).",
            input_dim,
            output_dim - 1,
            output_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects features onto the unit hypersphere.

        Args:
            x: Input tensor, shape (batch, num_stocks, input_dim)
                or (num_stocks, input_dim).

        Returns:
            Embeddings on S^{n-1} with unit norm, matching input
            leading dimensions.
        """
        # Intent: Mapping features to the Hypersphere (Wang & Isola, 2020).
        # By fixing vector length to 1, we ensure "Uniformity" (preventing collapse)
        # and "Alignment" (preserving relative similarity independent of magnitude).
        # Project features into the ambient space (with non-linear hidden layer)
        projected = self.projection(F.elu(self.pre_projection(x)))
        if not self.normalize:
            return projected
        projection_norms = torch.linalg.vector_norm(projected, dim=-1, keepdim=True)
        near_zero_mask = projection_norms < _MIN_PROJECTION_NORM
        if near_zero_mask.any():
            # Replace degenerate rows before normalisation so autograd does not
            # amplify near-zero vectors into unstable per-sample gradients.
            replacement = F.normalize(
                torch.randn_like(projected),
                p=2,
                dim=-1,
                eps=self.eps,
            ) * _REPLACEMENT_SCALE
            projected = torch.where(near_zero_mask, replacement, projected)

        # Normalise to unit length, placing points on the sphere
        embeddings = F.normalize(projected, p=2, dim=-1, eps=self.eps)

        return embeddings

    def geodesic_distance(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Measures the shortest-path distance between points on the sphere.

        Uses the arc length formula: the angle between two unit vectors
        gives the great-circle distance. Values are clamped before
        arccos to prevent NaN from floating-point rounding.

        Ref: Bronstein et al. (2021), Section 5.2.2 ("Riemannian Manifolds").

        Args:
            x: Points on the manifold, shape (..., output_dim).
            y: Points on the manifold, shape (..., output_dim).

        Returns:
            Geodesic distances, shape (...).
        """
        # Dot product gives the cosine of the angle between the points
        dot = (x * y).sum(dim=-1)

        # Clamp to valid arccos range to handle floating-point overshoot
        dot = dot.clamp(-1.0 + self.eps, 1.0 - self.eps)

        # The angle itself is the geodesic distance on the sphere
        return torch.acos(dot)


class ManifoldBatchNorm(nn.Module):
    """Batch normalisation adapted for data living on a curved surface.

    Standard batch norm assumes flat Euclidean space. On the sphere,
    we instead compute statistics in the tangent space at the Frechet
    mean, apply learnable scale and shift there, then project back.

    Ref: Bachmann et al. (2020), Section 4.1 ("Manifold Batch Normalization").

    Args:
        dim: Ambient dimension of the hypersphere (n).
        frechet_max_iter: Max iterations for Frechet mean.
        frechet_tol: Convergence tolerance for Frechet mean.
        eps: Stability constant.
    """

    def __init__(
        self,
        dim: int,
        frechet_max_iter: int = 100,
        frechet_tol: float = 1e-6,
        eps: float = _EPS,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.frechet_max_iter = frechet_max_iter
        self.frechet_tol = frechet_tol
        self.eps = eps

        # Learnable scale and shift applied in tangent space
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalises points on the manifold using tangent-space statistics.

        Args:
            x: Points on S^{n-1}, shape (N, dim).

        Returns:
            Normalised points on S^{n-1}, shape (N, dim).
        """
        # Find the centre of mass on the sphere
        mean = frechet_mean(
            x,
            max_iter=self.frechet_max_iter,
            tol=self.frechet_tol,
            eps=self.eps,
        )

        # Intent: Normalisation that respects geometry (Bachmann et al., 2020).
        # We cannot simply subtract the mean on a sphere. Instead, we:
        # 1. Log Map: Move data to the flat tangent space of the mean.
        # 2. Scale/Shift: Apply standard normalisation in this flat space.
        # 3. Exp Map: Project back to the sphere.
        # Lift all points into the flat tangent space at the mean
        mean_expanded = mean.unsqueeze(0).expand_as(x)
        tangents = logarithmic_map(mean_expanded, x, eps=self.eps)

        # Compute spread in tangent space and normalise
        tangent_norms_sq = (tangents * tangents).sum(dim=-1)
        sigma = torch.sqrt(tangent_norms_sq.mean()).clamp(min=self.eps)

        # Apply learnable scale and shift in tangent space
        normalised = self.gamma * (tangents / sigma) + self.beta

        # Wrap everything back onto the sphere
        result = exponential_map(mean_expanded, normalised, eps=self.eps)

        return result
