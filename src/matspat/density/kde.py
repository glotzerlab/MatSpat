"""Kernel density estimation and quantile partitioning for cell-type densities.

Functions
---------
build_grid      Build a regular 2-D evaluation grid from coordinate bounds.
compute_kde     Fit a Gaussian KDE for one cell type and evaluate on a grid.
get_marginal_quantile_masks   Non-overlapping equal-mass density bins.
get_cumulative_quantile_masks Nested cumulative density regions.
tissue_mask     Flag grid points that lie within tissue (near at least one cell).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from sklearn.neighbors import KernelDensity

from matspat.config import DEFAULT_GRID_SIZE


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
    *,
    grid: tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]] | None = None,
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
        grid: Optional pre-built grid tuple ``(xs, ys, grid_points)`` as
            returned by :func:`build_grid`.  When provided, the KDE is fit
            on *coords* but evaluated on this shared grid instead of
            building a new one from *coords*.  This is useful when multiple
            cell types must be evaluated on the same grid.

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
    if grid is not None:
        xs, ys, grid_points = grid
    else:
        xs, ys, grid_points = build_grid(coords, grid_size)
    kde = KernelDensity(bandwidth=bandwidth).fit(coords)
    density = np.exp(kde.score_samples(grid_points)).reshape(len(ys), len(xs))
    return {"density": density, "grid_coords": grid_points, "xs": xs, "ys": ys}


def tissue_mask(
    coords: NDArray[np.floating],
    grid_points: NDArray[np.floating],
    max_distance: float,
) -> NDArray[np.bool_]:
    """Flag grid points that lie within tissue.

    A grid point is considered in-tissue if at least one cell lies within
    *max_distance* of it.  Grid points spanning empty space (e.g. the corners
    of a bounding box around a round TMA core) are flagged ``False``.

    Args:
        coords: (N, 2) cell coordinates.
        grid_points: (M, 2) grid coordinates (e.g. from :func:`build_grid`).
        max_distance: Maximum distance to the nearest cell, in the same
            spatial units as *coords*.

    Returns:
        (M,) boolean array, ``True`` for in-tissue grid points.
    """
    dist, _ = cKDTree(coords).query(grid_points)
    return np.asarray(dist <= max_distance)


def _valid_mask(
    density: NDArray[np.floating], valid: NDArray[np.bool_] | None
) -> NDArray[np.bool_]:
    """Boolean validity mask shaped like *density*; all True when *valid* is None."""
    if valid is None:
        return np.ones(density.shape, dtype=bool)
    valid_arr = np.asarray(valid, dtype=bool).reshape(density.shape)
    if not valid_arr.any():
        raise ValueError("valid mask has no True entries")
    return valid_arr


def get_marginal_quantile_masks(
    density: NDArray[np.floating],
    n_quantiles: int,
    *,
    valid: NDArray[np.bool_] | None = None,
) -> dict[int, NDArray[np.bool_]]:
    """Split *density* into *n_quantiles* non-overlapping equal-count bins.

    Bin edges are quantiles of the density values over grid points, so each
    bin contains (approximately) the same number of grid points.

    Args:
        density: 2-D density array (e.g. from :func:`compute_kde`).
        n_quantiles: Number of bins.
        valid: Optional boolean array (same shape as *density*).  When given,
            quantile edges are computed over valid points only and every mask
            excludes invalid points (e.g. off-tissue grid points from
            :func:`tissue_mask`).

    Returns:
        Dict mapping bin index (0-based) to a boolean mask of the same shape
        as *density*.

    Raises:
        ValueError: If *valid* has no True entries.
    """
    valid_arr = _valid_mask(density, valid)
    flat = density[valid_arr]
    edges = np.quantile(flat, np.linspace(0, 1, n_quantiles + 1))
    masks: dict[int, NDArray[np.bool_]] = {}
    for i in range(n_quantiles):
        low, high = edges[i], edges[i + 1]
        if i < n_quantiles - 1:
            masks[i] = (density >= low) & (density < high) & valid_arr
        else:
            # last bin includes the upper edge
            masks[i] = (density >= low) & (density <= high) & valid_arr
    return masks


def get_cumulative_quantile_masks(
    density: NDArray[np.floating],
    n_quantiles: int,
    *,
    valid: NDArray[np.bool_] | None = None,
) -> dict[int, NDArray[np.bool_]]:
    """Create nested cumulative density masks.

    Each mask *i* includes all grid points with density <= the i-th quantile
    threshold, so masks grow monotonically.

    Args:
        density: 2-D density array.
        n_quantiles: Number of cumulative thresholds.
        valid: Optional boolean array (same shape as *density*).  When given,
            thresholds are computed over valid points only and every mask
            excludes invalid points.

    Returns:
        Dict mapping index (0-based) to boolean mask.

    Raises:
        ValueError: If *valid* has no True entries.
    """
    valid_arr = _valid_mask(density, valid)
    thresholds = np.quantile(
        density[valid_arr],
        np.linspace(1, n_quantiles, n_quantiles) / n_quantiles,
    )
    masks: dict[int, NDArray[np.bool_]] = {}
    for i, thr in enumerate(thresholds):
        masks[i] = (density <= thr) & valid_arr
    return masks
