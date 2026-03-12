"""Partial correlation computations using geographically weighted methods.

Public functions:
    compute_gwpcor(x, y, z, positions, bandwidth) -> float
"""

import numpy as np

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _bisquare_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Bisquare spatial kernel weights (see correlation.py for formula)."""
    raise NotImplementedError


def _weighted_ols_residuals(v: np.ndarray, z: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Residuals from weighted OLS regression of v on z (with intercept).

    Args:
        v: (N,) response variable.
        z: (N,) covariate.
        weights: (N,) per-point weights.

    Returns:
        (N,) residuals: v - Z @ beta, where Z = [1 | z].

    Pseudocode:
        Build design matrix Z = [column of ones, z]
        Solve weighted least squares for beta using weights
        Return v - Z @ beta
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compute_gwpcor(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    positions: np.ndarray,
    bandwidth: float,
) -> float:
    """Geographically Weighted Partial Correlation (GWPCor).

    Measures the spatial correlation between x and y after removing the linear
    effect of covariate z. Uses the same bisquare kernel as Lee's L.

    The key formula for weighted regression weights:

        w_i = sum_j W_ij   (row sums of the bisquare kernel matrix)

    These per-point weights are then used in OLS to regress out z:

        W_ij = (1 - (d_ij / h)^2)^2   if d_ij <= h,  else 0

    Args:
        x: (N,) density or feature values for cell type A.
        y: (N,) density or feature values for cell type B.
        z: (N,) covariate to control for (e.g. a structural descriptor).
        positions: (N, 2) spatial coordinates.
        bandwidth: Kernel bandwidth h.

    Returns:
        GWPCor value in [-1, 1], or nan if fewer than 3 points or z has zero variance.

    Pseudocode:
        Guard: return nan if len(x) < 3 or z has no variance
        Build bisquare kernel matrix W from positions and bandwidth
        Compute per-point weights w = row sums of W
        Compute residuals of x after regressing out z (weighted by w)
        Compute residuals of y after regressing out z (weighted by w)
        Return lees_L(residuals_x, residuals_y, positions, bandwidth)
    """
    raise NotImplementedError
