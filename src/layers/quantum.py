"""Quantum Layer: Density Matrix Regime Detection.

Detects market regimes by building data-conditioned density matrices
and extracting regime probabilities through the Born rule. Uses
Cholesky factorisation for valid density matrices and the Cayley
transform for learnable unitary evolution.

Theoretical Context:
    Market regimes are modeled as quantum density matrices to capture uncertainty
    and mixed states (Haven & Khrennikov, 2013). We use Cholesky decomposition
    (Ma et al., 2021) to ensure physical validity (positive semi-definite, unit trace)
    and the Cayley transform (Kiani et al., 2022) to learn unitary state transitions
    without vanishing gradients, reading out probabilities via the Born rule.

References:
    Haven and Khrennikov (2013) "Quantum Social Science."
    Busemeyer and Bruza (2012) "Quantum Models of Cognition and
        Decision."
    Ma et al. (2021) "Neural Networks for Quantum State Tomography
        Using Cholesky Decomposition." arXiv:2111.09504.
        Section 2.3 ("Cholesky Decomposition").
    Kiani et al. (2022) "projUNN: Efficient Method for Training
        Deep Networks with Unitary Matrices." arXiv:2203.05483.
        Section 2 ("Related Works") and Appendix B.
"""

import logging

import torch
import torch.nn as nn

from src.losses import _EPS

logger = logging.getLogger(__name__)

def validate_density_matrix(
    rho: torch.Tensor,
    tol: float = 1e-5,
) -> dict[str, bool]:
    """Checks whether a matrix is a valid density matrix.

    A density matrix must be symmetric, have trace equal to 1, and
    have all non-negative eigenvalues. These correspond to the
    physical requirements of observable predictions being real,
    total probability summing to one, and individual probabilities
    being non-negative.

    Ref: Haven and Khrennikov (2013), Section 1.2 ("The Born Rule").

    Args:
        rho: Density matrix of shape (..., K, K). If batched,
            validates the last matrix for diagnostics.
        tol: Tolerance for approximate comparisons.

    Returns:
        Dictionary with boolean results for 'is_symmetric',
        'is_trace_one', and 'is_psd'.
    """
    # Squeeze to 2-D if a single matrix is wrapped in batch dims
    if rho.dim() > 2:
        rho = rho[-1]

    # Check symmetry (Hermiticity for real matrices)
    is_symmetric = bool(torch.allclose(rho, rho.T, atol=tol))

    # Check trace equals 1
    trace_val = torch.trace(rho)
    is_trace_one = bool(torch.abs(trace_val - 1.0) < tol)

    # Check all eigenvalues are non-negative (PSD)
    eigenvalues = torch.linalg.eigvalsh(rho)
    is_psd = bool((eigenvalues >= -tol).all())

    return {
        "is_symmetric": is_symmetric,
        "is_trace_one": is_trace_one,
        "is_psd": is_psd,
    }


def von_neumann_entropy(rho: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Measures how mixed or uncertain a regime state is.

    High entropy means the model is unsure which regime the market
    is in (spread across multiple regimes). Low entropy means the
    model is confident about a single regime. Bounded between 0
    (pure state, full confidence) and log(K) (maximally mixed).

    Ref: Haven and Khrennikov (2013), Section 1.3 ("Von Neumann Entropy").

    Args:
        rho: Density matrix of shape (..., K, K).
        eps: Small constant to prevent log(0).

    Returns:
        Entropy value, shape (...).
    """
    # Get eigenvalues of the density matrix
    eigenvalues = torch.linalg.eigvalsh(rho)

    # Clamp to avoid log(0) and handle tiny negative numerical noise
    eigenvalues = eigenvalues.clamp(min=eps)

    # Intent: Measuring Model Confidence (Haven & Khrennikov, 2013).
    # If the market is in a "pure state" (low entropy), the model is confident
    # in one regime. If it's in a "mixed state" (high entropy), the market
    # is transitioning or uncertain.
    # Entropy: negative sum of lambda * log(lambda)
    entropy = -(eigenvalues * torch.log(eigenvalues)).sum(dim=-1)

    return entropy


class QuantumRegimeDetector(nn.Module):
    """Detects market regimes using density matrices and the Born rule.

    Takes geometric embeddings and yield curve context, builds a
    density matrix that represents the current market state, evolves
    it through a learnable unitary, and reads out regime probabilities.

    Uses real-valued matrices (not complex) because the regimes
    represent classical probability mixtures, not quantum
    superposition requiring phase information.

    Ref: Ma et al. (2021) Section 2.3; Kiani et al. (2022) Section 2.

    Args:
        input_dim: Dimension of the geometric embedding (n).
        context_dim: Dimension of the yield curve context vector.
        num_regimes: Number of market regimes (K).
    """

    def __init__(
        self,
        input_dim: int,
        context_dim: int = 3,
        num_regimes: int = 4,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.num_regimes = num_regimes

        combined_dim = input_dim + context_dim

        # Maps the combined feature vector to Cholesky factor entries.
        # K(K+1)/2 entries fill the lower triangle of a K x K matrix.
        num_cholesky_entries = num_regimes * (num_regimes + 1) // 2
        self.cholesky_map = nn.Linear(combined_dim, num_cholesky_entries)

        # Learnable parameter for the Cayley unitary transform.
        # Small init (0.01) keeps (I + A) well-conditioned early on.
        self.H_raw = nn.Parameter(torch.randn(num_regimes, num_regimes) * 0.01)

        # Pre-compute lower-triangle indices for building L efficiently
        tril_rows, tril_cols = torch.tril_indices(num_regimes, num_regimes)
        self.register_buffer("_tril_rows", tril_rows)
        self.register_buffer("_tril_cols", tril_cols)

        logger.info(
            "QuantumRegimeDetector initialised: input_dim=%d, "
            "context_dim=%d, num_regimes=%d, cholesky_entries=%d.",
            input_dim,
            context_dim,
            num_regimes,
            num_cholesky_entries,
        )

    def _build_cholesky_factor(
        self,
        cholesky_entries: torch.Tensor,
    ) -> torch.Tensor:
        """Fills a lower-triangular matrix from a flat vector.

        Ref: Ma et al. (2021), Page 6, Section 2.3 ("The alpha-vector").

        Args:
            cholesky_entries: Flat entries, shape (batch, K*(K+1)/2).

        Returns:
            Lower-triangular L of shape (batch, K, K).
        """
        batch_size = cholesky_entries.shape[0]
        K = self.num_regimes

        # Intent: The "alpha-vector" parameterization (Ma et al., 2021).
        # We take a flat neural network output (alpha-vector) and reshape it
        # into the lower-triangular Cholesky factor L. This is the first step
        # in ensuring the density matrix is physically valid.
        # Place entries into the lower triangle of a zero matrix
        L = cholesky_entries.new_zeros(batch_size, K, K)
        L[:, self._tril_rows, self._tril_cols] = cholesky_entries

        return L

    def _build_density_matrix(self, L: torch.Tensor) -> torch.Tensor:
        """Builds a valid density matrix from a Cholesky factor.

        Computes rho = L L^T / Tr(L L^T). This guarantees symmetry,
        positive semi-definiteness, and unit trace by construction.

        Ref: Ma et al. (2021), Page 4, Equation (3).

        Args:
            L: Lower-triangular factor, shape (batch, K, K).

        Returns:
            Density matrix rho, shape (batch, K, K).
        """
        # Intent: Constructing a valid Density Matrix (Ma et al., 2021).
        # By computing L @ L.T, we guarantee the matrix is Positive Semi-Definite
        # (no negative probabilities). We then normalize by the trace to ensure
        # unit probability mass (probabilities sum to 1).
        # L L^T is guaranteed symmetric and PSD
        rho_unnorm = L @ L.transpose(-2, -1)

        # Normalise to unit trace
        trace = (
            torch.diagonal(rho_unnorm, dim1=-2, dim2=-1)
            .sum(dim=-1, keepdim=True)
            .unsqueeze(-1)
        )
        trace = trace.clamp(min=_EPS)
        rho = rho_unnorm / trace

        return rho

    def _cayley_unitary(self) -> torch.Tensor:
        """Produces an orthogonal matrix via the Cayley transform.

        Builds a skew-symmetric matrix A from the learnable parameter,
        then maps it to an orthogonal matrix U = (I - A)(I + A)^{-1}.
        This guarantees U U^T = I without expensive matrix exponential.

        Ref: Kiani et al. (2022), Page 20, Equation (B.9).

        Returns:
            Orthogonal matrix U, shape (K, K).
        """
        # Enforce skew-symmetry so the Cayley output is orthogonal
        A = (self.H_raw - self.H_raw.T) / 2.0

        eye = torch.eye(self.num_regimes, device=A.device, dtype=A.dtype)

        # Intent: Unitary Evolution (Kiani et al., 2022).
        # Standard matrix multiplication isn't unitary (it leaks probability).
        # The Cayley Transform maps a skew-symmetric matrix A to a Unitary matrix U.
        # This allows the model to learn to "rotate" the market state without
        # violating quantum probability axioms.
        # Solve (I + A) U = (I - A) instead of explicit inverse
        U = torch.linalg.solve(eye + A, eye - A)

        return U

    def forward(
        self,
        embeddings: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes regime probabilities from embeddings and context.

        Combines the geometric embedding with yield curve context,
        builds a density matrix, evolves it through a unitary, and
        reads off regime probabilities from the diagonal (Born rule).

        Ref: Haven and Khrennikov (2013), Born rule measurement.

        Args:
            embeddings: Points on S^{n-1}, shape (batch, input_dim).
            context: Yield curve context, shape (batch, context_dim).

        Returns:
            Tuple of:
                - regime_probs: shape (batch, num_regimes), sums to 1.
                - rho_evolved: shape (batch, K, K), for diagnostics.
        """
        # Combine embedding and yield curve into a single input
        combined = torch.cat([embeddings, context], dim=-1)

        # Map to Cholesky entries and build lower-triangular factor
        cholesky_entries = self.cholesky_map(combined)
        L = self._build_cholesky_factor(cholesky_entries)

        # Construct density matrix (guaranteed valid by construction)
        rho = self._build_density_matrix(L)

        # Evolve the density matrix through the learnable unitary
        U = self._cayley_unitary()
        rho_evolved = U @ rho @ U.transpose(-2, -1)

        # Intent: The Born Rule (Regime Readout).
        # We extract the classical probabilities of each regime from the diagonal
        # of the evolved density matrix. This connects the quantum state back to
        # observable market regimes.
        # Read off regime probabilities from the diagonal (Born rule):
        # each diagonal entry is the probability of being in that regime
        regime_probs = torch.diagonal(rho_evolved, dim1=-2, dim2=-1)

        return regime_probs, rho_evolved


class ClassicalRegimeDetector(nn.Module):
    """Softmax regime head used as a control for QuantumRegimeDetector.

    Same inputs and outputs as the quantum detector. ``hidden_dim=None`` gives
    a linear softmax; an integer gives a one-hidden-layer ELU MLP, which with
    hidden_dim=10 matches the quantum detector's parameter count (E=64, C=3,
    K=4: 724 vs 696). The returned matrix is diag(p), a valid density matrix
    whose von Neumann entropy equals the Shannon entropy of p. The repo's
    entropy term H(p) - S(rho) is therefore identically zero for this head,
    which matches the quantum head's minimum (a rho diagonal in the readout basis).

    Args:
        input_dim: Dimension of the pooled embedding.
        context_dim: Dimension of the yield curve context vector.
        num_regimes: Number of market regimes (K).
        hidden_dim: Hidden width, or None for a linear map.
    """

    def __init__(
        self,
        input_dim: int,
        context_dim: int = 3,
        num_regimes: int = 4,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        combined_dim = input_dim + context_dim
        if hidden_dim is None:
            self.logits = nn.Linear(combined_dim, num_regimes)
        else:
            self.logits = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, num_regimes),
            )

    def forward(
        self,
        embeddings: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        regime_probs = torch.softmax(self.logits(torch.cat([embeddings, context], dim=-1)), dim=-1)
        return regime_probs, torch.diag_embed(regime_probs)


def build_regime_detector(
    kind: str,
    input_dim: int,
    context_dim: int,
    num_regimes: int,
) -> nn.Module:
    """Returns the regime head named by ``kind``: quantum, softmax or softmax_mlp."""
    if kind == "quantum":
        return QuantumRegimeDetector(input_dim=input_dim, context_dim=context_dim, num_regimes=num_regimes)
    if kind == "softmax":
        return ClassicalRegimeDetector(input_dim, context_dim, num_regimes, hidden_dim=None)
    if kind == "softmax_mlp":
        return ClassicalRegimeDetector(input_dim, context_dim, num_regimes, hidden_dim=10)
    raise ValueError(f"Unknown regime head {kind!r}.")
