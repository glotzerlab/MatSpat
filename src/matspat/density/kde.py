"""Kernel density estimation and quantile partitioning for cell-type densities.

Functions
---------
build_grid      Build a regular 2-D evaluation grid from coordinate bounds.
compute_kde     Fit a Gaussian KDE for one cell type and evaluate on a grid.
get_marginal_quantile_masks   Non-overlapping equal-mass density bins.
get_cumulative_quantile_masks Nested cumulative density regions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import KernelDensity

DEFAULT_GRID_SIZE: int = 100


def build_grid(
    coords: NDArray[np.floating],
    grid_size: int = DEFAULT_GRID_SIZE,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Build a regular 2-D evaluation grid spanning the coordinate bounds.

    Args:
        coords: Array of shape (N, 2) with (x, y) positions.  Must contain
            at least one point.
        grid_size: Number of points along each axis.

    Returns:
        Tuple of (xs, ys, grid_points) where *xs* and *ys* are 1-D arrays
        of length *grid_size* and *grid_points* has shape (grid_size**2, 2).

    Raises:
        IndexError: If *coords* is empty.
    """
    x0, x1 = coords[:, 0].min(), coords[:, 0].max()
    y0, y1 = coords[:, 1].min(), coords[:, 1].max()
    xs = np.linspace(x0, x1, grid_size)
    ys = np.linspace(y0, y1, grid_size)
    X, Y = np.meshgrid(xs, ys)
    return xs, ys, np.column_stack([X.ravel(), Y.ravel()])


def compute_kde(
    coords: NDArray[np.floating],
    bandwidth: float,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> dict[str, NDArray[np.floating]]:
    """Fit a Gaussian KDE on *coords* and evaluate on a regular grid.

    Uses ``sklearn.neighbors.KernelDensity`` with an explicit bandwidth,
    which is required for the bandwidth-sweep analysis in the pipeline.
    The returned density values are in probability-density space (not
    log-density).

    Args:
        coords: Array of shape (N, 2) with (x, y) positions for a single
            cell type.  The caller is responsible for filtering the
            DataFrame to the desired cell type before extracting coordinates.
        bandwidth: KDE bandwidth in the same spatial units as *coords*.
        grid_size: Number of points per grid axis.

    Returns:
        Dict with keys:

        - ``"density"``: 2-D array of shape (grid_size, grid_size) with
          probability-density values.
        - ``"grid_coords"``: Flattened grid of shape (grid_size**2, 2).
        - ``"xs"``: 1-D x-axis ticks of length *grid_size*.
        - ``"ys"``: 1-D y-axis ticks of length *grid_size*.

    Raises:
        IndexError: If *coords* is empty.
    """
    xs, ys, grid = build_grid(coords, grid_size)
    kde = KernelDensity(bandwidth=bandwidth).fit(coords)
    density = np.exp(kde.score_samples(grid)).reshape(len(ys), len(xs))
    return {"density": density, "grid_coords": grid, "xs": xs, "ys": ys}


def get_marginal_quantile_masks(
    density: NDArray[np.floating],
    n_quantiles: int,
) -> dict[int, NDArray[np.bool_]]:
    """Split *density* into *n_quantiles* non-overlapping bins of equal mass.

    Args:
        density: 2-D density array (e.g. from :func:`compute_kde`).
        n_quantiles: Number of bins.

    Returns:
        Dict mapping bin index (0-based) to a boolean mask of the same shape
        as *density*.
    """
    flat = density.flatten()
    edges = np.quantile(flat, np.linspace(0, 1, n_quantiles + 1))
    masks: dict[int, NDArray[np.bool_]] = {}
    for i in range(n_quantiles):
        low, high = edges[i], edges[i + 1]
        if i < n_quantiles - 1:
            masks[i] = (density >= low) & (density < high)
        else:
            # last bin includes the upper edge
            masks[i] = (density >= low) & (density <= high)
    return masks


def get_cumulative_quantile_masks(
    density: NDArray[np.floating],
    n_quantiles: int,
) -> dict[int, NDArray[np.bool_]]:
    """Create nested cumulative density masks.

    Each mask *i* includes all grid points with density <= the i-th quantile
    threshold, so masks grow monotonically.

    Args:
        density: 2-D density array.
        n_quantiles: Number of cumulative thresholds.

    Returns:
        Dict mapping index (0-based) to boolean mask.
    """
    thresholds = np.quantile(
        density.flatten(),
        np.linspace(1, n_quantiles, n_quantiles) / n_quantiles,
    )
    masks: dict[int, NDArray[np.bool_]] = {}
    for i, thr in enumerate(thresholds):
        masks[i] = density <= thr
    return masks
