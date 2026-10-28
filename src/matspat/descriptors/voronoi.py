"""Voronoi cell geometry per cell (area, perimeter, neighbour count, shape index).

Each cell centroid gets the polygon of all points closer to it than to any other
centroid. From that polygon:

- ``voronoi_area`` A and ``voronoi_perimeter`` P
- ``voronoi_neighbors``: number of polygon edges (Voronoi neighbours)
- ``shape_index`` s = P / sqrt(A): ≈ 3.72 for a regular hexagon, 4 for a square,
  larger for elongated cells. In vertex models of epithelia s ≈ 3.81 marks the
  jamming transition (Bi et al., Nat. Phys. 2015).

The tessellation uses scipy's Qhull Voronoi rather than freud.locality.Voronoi:
freud's needs a periodic box, and tissue sections are not periodic, so the box
must be padded far beyond the tissue. That makes freud slow and memory-hungry
(several GB for a few hundred cells at 10x padding).

Cells at the tissue edge have unbounded Voronoi regions, so their geometry is
meaningless. A cell is flagged as ``voronoi_boundary`` when its region is
unbounded, any polygon vertex lies outside the convex hull of the included
cells, or its area exceeds ``max_area``; its geometry is set to nan. Cells
sharing exact coordinates with another cell are flagged too: Qhull assigns
them a single region, so their geometry is not defined per cell.

Functions: compute_voronoi_geometry()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError, Voronoi

from . import _validation as val
from .spatial_descriptors import _apply_per_image

GEOMETRY_COLS = ["voronoi_area", "voronoi_perimeter", "voronoi_neighbors", "shape_index"]


def compute_voronoi_geometry(
    df: pd.DataFrame,
    *,
    cell_type_col: str = "cell_type",
    focus_cell_type: str | None = None,
    include_cell_types: list[str] | None = None,
    max_area: float | None = None,
    image_id_col: str = "image_id",
) -> pd.DataFrame:
    """Compute Voronoi cell geometry for each cell.

    Parameters
    ----------
    df : pd.DataFrame
        Standardized DataFrame from load_spatial_data() (columns cell_id, x, y,
        cell type and image ID).
    cell_type_col : str, default "cell_type"
        Column name for cell type labels.
    focus_cell_type : str | None, optional
        If provided, only return rows for this cell type (the tessellation still
        uses all included cells).
    include_cell_types : list[str] | None, optional
        Cell types that take part in the tessellation. None = all.
    max_area : float | None, optional
        Cells with a larger Voronoi area are flagged as boundary cells. Useful
        for concave tissue edges and holes, which the convex-hull test misses.
    image_id_col : str, default "image_id"
        Column name for image ID.

    Returns:
    -------
    pd.DataFrame
        Columns: cell_id, image_id, cell_type, voronoi_area, voronoi_perimeter,
        voronoi_neighbors, shape_index, voronoi_boundary. Geometry is nan for
        boundary cells.

    Raises:
    ------
    ValueError
        If required columns are missing, cell type validation fails, or
        max_area <= 0.
    """
    if max_area is not None and max_area <= 0:
        raise ValueError(f"max_area must be positive, got {max_area}")
    val.validate_dataframe_columns(df, ["cell_id", "x", "y", cell_type_col, image_id_col])
    val.validate_cell_types(df, cell_type_col, focus_cell_type, include_cell_types)

    def process_image(image_df: pd.DataFrame) -> pd.DataFrame | None:
        return _voronoi_single_image(
            image_df,
            cell_type_col=cell_type_col,
            focus_cell_type=focus_cell_type,
            include_cell_types=include_cell_types,
            max_area=max_area,
        )

    output_cols = ["cell_id", image_id_col, cell_type_col, *GEOMETRY_COLS, "voronoi_boundary"]
    return _apply_per_image(df, image_id_col, process_image, output_cols)


def _voronoi_single_image(
    image_df: pd.DataFrame,
    *,
    cell_type_col: str,
    focus_cell_type: str | None,
    include_cell_types: list[str] | None,
    max_area: float | None,
) -> pd.DataFrame | None:
    """Voronoi geometry for one image; None if fewer than 4 cells after filtering."""
    df_filtered, positions, focus_mask = val.filter_and_extract_data(
        image_df, cell_type_col, include_cell_types, focus_cell_type
    )
    if len(df_filtered) < 4:
        return None

    n = len(positions)
    area = np.full(n, np.nan)
    perimeter = np.full(n, np.nan)
    boundary: np.ndarray = np.ones(n, dtype=bool)
    neighbors = np.zeros(n)
    try:
        vor = Voronoi(positions)
        hull = Delaunay(positions)
    except QhullError:  # degenerate (e.g. collinear) point sets: no bounded cells
        vor = None
    if vor is not None:
        np.add.at(neighbors, vor.ridge_points.ravel(), 1.0)
        for i, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]
            if not region or -1 in region:  # unbounded (edge) cell
                continue
            poly = _ordered(vor.vertices[region])
            if np.any(hull.find_simplex(poly) < 0):
                continue
            edges = np.roll(poly, -1, axis=0) - poly
            area[i] = 0.5 * abs(np.sum(poly[:, 0] * edges[:, 1] - poly[:, 1] * edges[:, 0]))
            perimeter[i] = np.sum(np.hypot(edges[:, 0], edges[:, 1]))
            boundary[i] = False
    if max_area is not None:
        boundary |= area > max_area
    _, inverse, counts = np.unique(positions, axis=0, return_inverse=True, return_counts=True)
    boundary |= counts[inverse.ravel()] > 1

    result = pd.DataFrame(
        {
            "cell_id": df_filtered["cell_id"].values,
            cell_type_col: df_filtered[cell_type_col].values,
            "voronoi_area": area,
            "voronoi_perimeter": perimeter,
            "voronoi_neighbors": neighbors,
            "shape_index": perimeter / np.sqrt(area),
            "voronoi_boundary": boundary,
        }
    )
    result.loc[boundary, GEOMETRY_COLS] = np.nan
    if focus_mask is not None:
        result = result[focus_mask].reset_index(drop=True)
    return result


def _ordered(poly: np.ndarray) -> np.ndarray:
    """Vertices of a convex polygon sorted counter-clockwise around their centroid."""
    c = poly.mean(axis=0)
    ordered: np.ndarray = poly[np.argsort(np.arctan2(poly[:, 1] - c[1], poly[:, 0] - c[0]))]
    return ordered
