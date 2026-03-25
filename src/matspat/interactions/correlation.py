"""Spatial correlation functions for cell-type co-localization analysis.

Public functions:
    lees_L(x, y, positions, bandwidth) -> float
    lees_L_sequence(x_full, y_full, grid_coords, bandwidth, quantile_masks) -> list[float]
"""

import numpy as np
from scipy.spatial import distance

# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------


def _bisquare_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Bisquare spatial kernel weights.

    Args:
        distances: (N, N) pairwise Euclidean distance matrix.
        bandwidth: Kernel bandwidth h.

    Returns:
        (N, N) weight matrix.

        W_ij = (1 - (d_ij / h)^2)^2   if d_ij <= h,  else 0
    """
    weight_matrix = np.zeros_like(distances)
    mask = distances <= bandwidth
    weight_matrix[mask] = (1 - (distances[mask] / bandwidth) ** 2) ** 2
    return weight_matrix


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def lees_L(x: np.ndarray, y: np.ndarray, positions: np.ndarray, bandwidth: float) -> float:
    """Compute Lee's L spatial correlation between two density arrays.

    Lee's L is a spatially-weighted bivariate correlation measuring co-localization:

        L = Σ_ij W_ij · (x_i - x̄)(y_j - ȳ)
            ──────────────────────────────────────────────────────────────
            sqrt[Σ_ij W_ij · (x_i - x̄)(x_j - x̄)] × sqrt[Σ_ij W_ij · (y_i - ȳ)(y_j - ȳ)]

    where W is the bisquare kernel matrix built from pairwise distances between positions.

    Args:
        x: (N,) KDE density values for cell type A at grid positions.
        y: (N,) KDE density values for cell type B at the same grid positions.
        positions: (N, 2) spatial coordinates of the grid points.
        bandwidth: Kernel bandwidth h.

    Returns:
        Lee's L value in [-1, 1], or nan if fewer than 2 points or denominator is zero.
    """
    if len(x) < 2:
        return float(np.nan)

    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    distances = distance.cdist(positions, positions, metric="euclidean")
    W = _bisquare_kernel(distances, bandwidth)
    numerator = np.sum(W * np.outer(x_centered, y_centered))
    x_denom = np.sqrt(np.sum(W * np.outer(x_centered, x_centered)))
    y_denom = np.sqrt(np.sum(W * np.outer(y_centered, y_centered)))
    if x_denom * y_denom == 0:
        return float(np.nan)
    else:
        L: float = float(numerator / (x_denom * y_denom))
    return L


def lees_L_sequence(
    x_full: np.ndarray,
    y_full: np.ndarray,
    grid_coords: np.ndarray,
    bandwidth: float,
    quantile_masks: list[np.ndarray],
) -> list[float]:
    """Compute Lee's L for each spatial quantile bin.

    Masks are flattened internally, so both 2-D masks (as returned by
    ``density.kde.get_marginal_quantile_masks``) and pre-flattened 1-D
    masks are accepted.

    Args:
        x_full: (M,) flattened KDE density for cell type A over the full grid.
        y_full: (M,) flattened KDE density for cell type B over the full grid.
        grid_coords: (M, 2) coordinates of all grid points.
        bandwidth: Kernel bandwidth h.
        quantile_masks: Sequence of boolean arrays, one per quantile bin.
            Can be 2-D (grid_size, grid_size) or 1-D (M,).  Typically
            produced by ``density.kde`` quantile partitioning functions.

    Returns:
        List of Lee's L values of length ``len(quantile_masks)``.
        Bins with fewer than 2 grid points return ``nan``.
    """
    results = []
    for mask in quantile_masks:
        flat_mask = mask.ravel()
        x_sub = x_full[flat_mask]
        y_sub = y_full[flat_mask]
        pos_sub = grid_coords[flat_mask]
        results.append(lees_L(x_sub, y_sub, pos_sub, bandwidth))
    return results
