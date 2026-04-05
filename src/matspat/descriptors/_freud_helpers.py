"""Freud-specific helper functions for spatial descriptor computation."""

from __future__ import annotations

import warnings

import freud
import numpy as np


def setup_freud_box(
    positions: np.ndarray,
    *,
    box_scale: float = 10.0,
    Lx: float | None = None,
    Ly: float | None = None,
) -> tuple[freud.box.Box, np.ndarray]:
    """Create a non-periodic freud box and 3D position array from 2D coordinates.

    For non-periodic spatial data (tissue imaging), the box must be
    significantly larger than the data extents to avoid boundary artifacts.
    A scale factor of 10-100x is recommended. The default is 10x.

    Parameters
    ----------
    positions : np.ndarray
        (N, 2) array of [x, y] coordinates.
    box_scale : float, default 10.0
        Scale factor applied to data extents to set box dimensions.
        Ignored if Lx and Ly are provided explicitly.
        For non-periodic data, use at least 10x. Values below 2.0 will
        raise a warning.
    Lx : float | None, optional
        Explicit box length in x. Overrides box_scale for x dimension.
    Ly : float | None, optional
        Explicit box length in y. Overrides box_scale for y dimension.

    Returns:
    -------
    tuple[freud.box.Box, np.ndarray]
        - freud 2D box (is2D=True)
        - (N, 3) positions with z=0 appended

    Raises:
    ------
    ValueError
        If positions is empty or not shape (N, 2).

    Warns:
    -----
    UserWarning
        If box_scale < 2.0 (too small for non-periodic data).
    """
    if positions.size == 0:
        raise ValueError("positions array is empty")
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions must have shape (N, 2), got {positions.shape}")

    # Warn if box_scale is too small
    if Lx is None and Ly is None and box_scale < 2.0:
        warnings.warn(
            f"box_scale={box_scale} is very small for non-periodic data. "
            f"Recommend at least 10x to avoid boundary artifacts.",
            UserWarning,
            stacklevel=3,
        )

    # Compute box dimensions
    if Lx is None:
        x_extent: float = float(np.ptp(positions[:, 0]))  # max - min
        Lx = box_scale * x_extent + 1.0  # Add buffer to avoid zero extent
    if Ly is None:
        y_extent: float = float(np.ptp(positions[:, 1]))
        Ly = box_scale * y_extent + 1.0

    # Create 2D box
    box = freud.box.Box(Lx=Lx, Ly=Ly, xy=0, is2D=True)

    # Append z=0 column for freud compatibility
    positions_3d = np.column_stack([positions, np.zeros(len(positions))])

    return box, positions_3d


def build_neighbor_list_r_max(
    box: freud.box.Box,
    positions: np.ndarray,
    r_max: float,
    exclude_ii: bool = True,
) -> freud.locality.NeighborList:
    """Build neighbor list with all neighbors within r_max.

    Parameters
    ----------
    box : freud.box.Box
        Freud box.
    positions : np.ndarray
        (N, 3) positions array.
    r_max : float
        Cutoff radius.
    exclude_ii : bool
        Exclude self-pairs (i, i).

    Returns:
    -------
    freud.locality.NeighborList
        Neighbor list with all neighbors within r_max.
    """
    aq = freud.locality.AABBQuery(box, positions)
    query_result = aq.query(positions, {"r_max": r_max, "exclude_ii": exclude_ii})
    return query_result.toNeighborList()


def build_neighbor_list_k_nearest(
    box: freud.box.Box,
    positions: np.ndarray,
    k: int,
    r_max: float,
) -> tuple[freud.locality.NeighborList, np.ndarray]:
    """Build neighbor list with exactly k nearest neighbors within r_max.

    Filters to points with at least k neighbors, then keeps only the
    k closest for each point.

    Parameters
    ----------
    box : freud.box.Box
        Freud box.
    positions : np.ndarray
        (N, 3) positions array.
    k : int
        Number of nearest neighbors to keep.
    r_max : float
        Maximum radius to search within.

    Returns:
    -------
    tuple[freud.locality.NeighborList, np.ndarray]
        - Neighbor list with exactly k neighbors per valid point
        - Boolean mask indicating which points have valid k neighbors
          (shape (N,), True if point has >= k neighbors within r_max)
    """
    # Query all neighbors within r_max
    aq = freud.locality.AABBQuery(box, positions)
    query_result = aq.query(positions, {"r_max": r_max, "exclude_ii": True})
    nlist = query_result.toNeighborList()

    # Count neighbors per point
    n_points = len(positions)
    neighbor_counts = np.bincount(nlist.query_point_indices, minlength=n_points)

    # Mask for points with >= k neighbors
    valid_mask = neighbor_counts >= k

    # If no valid points, return empty neighbor list
    if not np.any(valid_mask):
        empty_nlist = freud.locality.NeighborList.from_arrays(
            n_points,
            n_points,
            np.array([], dtype=np.uint32),
            np.array([], dtype=np.uint32),
            np.array([], dtype=np.float32).reshape(0, 3),  # Empty (0, 3) array for vectors
        )
        return empty_nlist, valid_mask

    # Vectorized filtering: sort all pairs by (query_point, distance),
    # then keep only the first k per query point.
    all_qi = nlist.query_point_indices
    all_pi = nlist.point_indices
    all_dists = nlist.distances

    # Sort by query index first, then by distance within each group
    order = np.lexsort((all_dists, all_qi))
    sorted_qi = all_qi[order]
    sorted_pi = all_pi[order]

    # Compute rank of each entry within its query-point group
    group_sizes = np.bincount(sorted_qi, minlength=n_points)
    group_starts = np.empty(n_points + 1, dtype=np.intp)
    group_starts[0] = 0
    np.cumsum(group_sizes, out=group_starts[1:])
    within_group_rank = np.arange(len(sorted_qi)) - group_starts[sorted_qi]

    # Keep entries where rank < k and the query point is valid
    keep = (within_group_rank < k) & valid_mask[sorted_qi]

    filt_qi = sorted_qi[keep]
    filt_pi = sorted_pi[keep]

    # Compute displacement vectors in batch
    filt_vectors = box.wrap(positions[filt_pi] - positions[filt_qi])

    # Build filtered neighbor list
    filtered_nlist = freud.locality.NeighborList.from_arrays(
        n_points,
        n_points,
        filt_qi.astype(np.uint32),
        filt_pi.astype(np.uint32),
        filt_vectors.astype(np.float32),
    )

    return filtered_nlist, valid_mask
