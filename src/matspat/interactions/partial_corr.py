"""Covariate-adjusted Lee's L, and interpolation of cell-level covariates onto the grid.

``partial_lees_L`` regresses a covariate z out of both variables with one weighted
OLS fit per region, then computes Lee's L on the residuals. It was previously
called GWPCor; it is not a geographically weighted partial correlation in the
strict (local) sense. For that, see
:func:`~matspat.interactions.local_corr.gw_partial_corr`.

Public functions:
    interpolate_to_grid(cell_positions, cell_values, grid_coords, method) -> ndarray
    partial_lees_L(x, y, z, positions, bandwidth) -> float
    partial_lees_L_sequence(x_full, y_full, z_full, grid_coords, bandwidth,
                            quantile_masks) -> list[float]

``compute_gwpcor`` and ``gwpcor_sequence`` remain as aliases of the last two.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import distance

from matspat.interactions.correlation import (
    _bisquare_kernel,
    _finite_in_bin,
    _lees_L_from_kernel,
)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _weighted_ols_residuals(v: np.ndarray, z: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Residuals from weighted OLS regression of v on z (with intercept).

    Args:
        v: (N,) response variable.
        z: (N,) covariate.
        weights: (N,) per-point weights.

    Returns:
        (N,) residuals: v - Z @ beta, where Z = [1 | z].
    """
    Z = np.column_stack([np.ones(len(z)), z])
    # beta = (Z^T W Z)^{-1} Z^T W v
    # Use element-wise multiply instead of constructing a full N×N diagonal matrix
    ZtW = Z.T * weights  # (2, N) — broadcasts weights across rows
    beta = np.linalg.solve(ZtW @ Z, ZtW @ v)
    return v - Z @ beta


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def interpolate_to_grid(
    cell_positions: np.ndarray,
    cell_values: np.ndarray,
    grid_coords: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate cell-level values onto a regular evaluation grid.

    Bridges the gap between per-cell descriptors (from
    :mod:`~matspat.descriptors`) and the grid-based correlation pipeline.
    Uses :func:`scipy.interpolate.griddata` with a nearest-neighbor
    fallback for grid points outside the convex hull of *cell_positions*.

    Args:
        cell_positions: (N, 2) spatial coordinates of cells.
        cell_values: (N,) descriptor values at each cell (e.g. local_density,
            psi_6 from :mod:`~matspat.descriptors`).
        grid_coords: (M, 2) evaluation grid coordinates (e.g. from
            :func:`~matspat.density.kde.build_grid`).
        method: Interpolation method for interior points — ``"linear"``
            (default), ``"cubic"``, or ``"nearest"``.

    Returns:
        (M,) interpolated values on the grid.  Grid points outside the
        convex hull of *cell_positions* are filled via nearest-neighbor
        interpolation.  Cells with non-finite values (e.g. Voronoi boundary
        cells, which are nan) are dropped first, so the result has no nan.

    Raises:
        ValueError: If *method* is not one of ``"linear"``, ``"cubic"``,
            or ``"nearest"``, or no cell has a finite value.
    """
    valid_methods = {"linear", "cubic", "nearest"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got {method!r}")

    cell_values = np.asarray(cell_values, dtype=float)
    finite = np.isfinite(cell_values)
    if not finite.any():
        raise ValueError("cell_values has no finite entries")
    cell_positions, cell_values = np.asarray(cell_positions)[finite], cell_values[finite]

    z_grid = griddata(cell_positions, cell_values, grid_coords, method=method)

    # Fill NaNs at grid points outside convex hull with nearest-neighbor
    if method != "nearest":
        nan_mask = np.isnan(z_grid)
        if nan_mask.any():
            z_grid[nan_mask] = griddata(
                cell_positions, cell_values, grid_coords[nan_mask], method="nearest"
            )

    return z_grid


def partial_lees_L(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    positions: np.ndarray,
    bandwidth: float,
) -> float:
    """Covariate-adjusted Lee's L (formerly called GWPCor).

    Lee's L between x and y after removing the linear effect of covariate z
    from both with one weighted OLS fit. Uses the same bisquare kernel as Lee's L.

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
        Adjusted Lee's L in [-1, 1], or nan if fewer than 3 points or z has zero variance.
    """
    if len(x) < 3:
        return float(np.nan)

    if np.var(z) == 0.0:
        return float(np.nan)

    # Build bisquare kernel matrix W from positions and bandwidth
    dists = distance.cdist(positions, positions, metric="euclidean")
    W = _bisquare_kernel(dists, bandwidth)

    # Per-point weights = row sums of kernel matrix
    w = W.sum(axis=1)

    # Regress out z from x and y using weighted OLS
    residuals_x = _weighted_ols_residuals(x, z, w)
    residuals_y = _weighted_ols_residuals(y, z, w)

    # Lee's L on the residuals — reuse the already-computed kernel matrix
    return _lees_L_from_kernel(residuals_x, residuals_y, W)


def partial_lees_L_sequence(
    x_full: np.ndarray,
    y_full: np.ndarray,
    z_full: np.ndarray,
    grid_coords: np.ndarray,
    bandwidth: float,
    quantile_masks: list[np.ndarray],
    *,
    bin_kernels: list[np.ndarray | None] | None = None,
) -> list[float]:
    """Compute covariate-adjusted Lee's L for each spatial quantile bin.

    Mirrors :func:`~matspat.interactions.correlation.lees_L_sequence` but
    controls for covariate *z* (e.g. a structural descriptor interpolated
    onto the grid via :func:`interpolate_to_grid`).

    All three value arrays (x, y, z) must live on the same grid as
    *grid_coords*.  For immune densities this happens naturally via KDE
    ``score_samples``; for structural covariates use
    :func:`interpolate_to_grid` to map cell-level descriptor values onto
    the shared grid.

    Masks are flattened internally, so both 2-D masks (as returned by
    ``density.kde.get_marginal_quantile_masks``) and pre-flattened 1-D
    masks are accepted.

    Args:
        x_full: (M,) values for cell type A over the full grid.
        y_full: (M,) values for cell type B over the full grid.
        z_full: (M,) covariate values over the full grid.
        grid_coords: (M, 2) coordinates of all grid points.
        bandwidth: Kernel bandwidth h.
        quantile_masks: Sequence of boolean arrays, one per quantile bin.
            Can be 2-D (grid_size, grid_size) or 1-D (M,).
        bin_kernels: Precomputed kernel matrices from
            :func:`~matspat.interactions.correlation.precompute_bin_kernels`.
            If provided, skips distance and kernel computation per bin.

    Returns:
        List of adjusted Lee's L values of length ``len(quantile_masks)``.
        Grid points where x, y or z is non-finite are left out. Bins with
        fewer than 3 remaining grid points or constant z return ``nan``.
    """
    results = []
    for i, mask in enumerate(quantile_masks):
        flat_mask = mask.ravel()
        finite = _finite_in_bin(flat_mask, x_full, y_full, z_full)
        if bin_kernels is not None:
            W = bin_kernels[i]
            if W is None:
                results.append(float(np.nan))
                continue
            W = W[np.ix_(finite, finite)]
            x_sub = x_full[flat_mask][finite]
            y_sub = y_full[flat_mask][finite]
            z_sub = z_full[flat_mask][finite]
            if len(x_sub) < 3 or np.var(z_sub) == 0.0:
                results.append(float(np.nan))
                continue
            w = W.sum(axis=1)
            res_x = _weighted_ols_residuals(x_sub, z_sub, w)
            res_y = _weighted_ols_residuals(y_sub, z_sub, w)
            results.append(_lees_L_from_kernel(res_x, res_y, W))
        else:
            results.append(
                partial_lees_L(
                    x_full[flat_mask][finite],
                    y_full[flat_mask][finite],
                    z_full[flat_mask][finite],
                    grid_coords[flat_mask][finite],
                    bandwidth,
                )
            )
    return results


# Backward-compatible names: this statistic was previously called GWPCor
compute_gwpcor = partial_lees_L
gwpcor_sequence = partial_lees_L_sequence
