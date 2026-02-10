"""Freud-based spatial descriptors: local density, k-atic order, Voronoi.

Functions: compute_local_density(), compute_katic_order(), count_neighbor_cell_types()
"""

from __future__ import annotations

import warnings

import freud
import numpy as np
import pandas as pd

from . import _freud_helpers as fh
from . import _validation as val


def compute_local_density(
    df: pd.DataFrame,
    r_max: float,
    *,
    cell_type_col: str = "cell_type",
    focus_cell_type: str | None = None,
    include_cell_types: list[str] | None = None,
    box_scale: float = 10.0,
    Lx: float | None = None,
    Ly: float | None = None,
    diameter: float = 0.0,
    norm_by: str | None = None,
    include_neighbor_counts: bool = False,
    image_id_col: str = "image_id",
) -> pd.DataFrame:
    """Compute freud LocalDensity for cells in a spatial DataFrame.

    Computes local density per image. When focus_cell_type is specified,
    only returns density values for cells of that type. When
    include_cell_types is specified, only those cell types participate
    in the point cloud (neighbor list construction).

    Parameters
    ----------
    df : pd.DataFrame
        Standardized DataFrame from load_spatial_data(). Must have
        x, y, cell_type_col, and image_id_col columns.
    r_max : float
        Cutoff radius for local density computation.
    cell_type_col : str, default "cell_type"
        Column name for cell type labels.
    focus_cell_type : str | None, optional
        If provided, only return density values for this cell type.
        All include_cell_types still participate in the neighbor list.
        If None, return values for all included cells.
    include_cell_types : list[str] | None, optional
        Cell types to include in the point cloud. None means include all.
        Must contain focus_cell_type if both are specified.
    box_scale : float, default 10.0
        Box scale factor. See setup_freud_box() for details.
        For non-periodic data, use at least 10x.
    Lx : float | None, optional
        Explicit box length in x. Overrides box_scale.
    Ly : float | None, optional
        Explicit box length in y. Overrides box_scale.
    diameter : float, default 0.0
        Particle diameter for freud LocalDensity.
    norm_by : str | None, optional
        Normalization method: "total" (all cells in image), "included"
        (included cell types count), "focus" (focus cell type count),
        or None (raw density).
    include_neighbor_counts : bool, default False
        If True, include a num_neighbors column in output.
    image_id_col : str, default "image_id"
        Column name for image ID.

    Returns:
    -------
    pd.DataFrame
        DataFrame with columns:
        - cell_id : from input (for joining back)
        - image_id : image identifier
        - cell_type : cell type label
        - local_density : computed local density value
        - num_neighbors : (only if include_neighbor_counts=True) neighbor count

    Raises:
    ------
    ValueError
        If required columns are missing.
        If focus_cell_type not in include_cell_types (when both specified).
        If include_cell_types contains types not in data.
        If norm_by not in {None, "total", "included", "focus"}.
        If r_max <= 0.

    Warns:
    -----
    UserWarning
        If an image has < 2 cells after filtering (skips that image).

    Examples:
    --------
    >>> import matspat.preprocessing as prep
    >>> import matspat.descriptors as desc
    >>> df, _ = prep.load_spatial_data(
    ...     "data.csv", x_col="X", y_col="Y",
    ...     cell_type_col="type", image_id_col="image"
    ... )
    >>> # Local density for all cells
    >>> density_df = desc.compute_local_density(df, r_max=50.0)
    >>> # Density for Epi cells only, with Treg in neighborhood
    >>> density_epi = desc.compute_local_density(
    ...     df, r_max=50.0,
    ...     focus_cell_type="Epi",
    ...     include_cell_types=["Epi", "Treg"],
    ...     norm_by="total"
    ... )
    """
    # Validate inputs
    val.validate_r_max(r_max)
    val.validate_norm_by(norm_by)
    required_cols = ["cell_id", "x", "y", cell_type_col, image_id_col]
    val.validate_dataframe_columns(df, required_cols)
    val.validate_cell_types(df, cell_type_col, focus_cell_type, include_cell_types)

    # Process per image
    results = []

    for image_id, image_df in df.groupby(image_id_col):
        result = _compute_local_density_single_image(
            image_df,
            r_max,
            cell_type_col=cell_type_col,
            focus_cell_type=focus_cell_type,
            include_cell_types=include_cell_types,
            box_scale=box_scale,
            Lx=Lx,
            Ly=Ly,
            diameter=diameter,
            norm_by=norm_by,
            include_neighbor_counts=include_neighbor_counts,
        )

        if result is not None:
            result[image_id_col] = image_id
            results.append(result)

    if not results:
        warnings.warn(
            "No valid images found (all had < 2 cells after filtering)",
            UserWarning,
            stacklevel=2,
        )
        # Return empty DataFrame with correct columns
        cols = ["cell_id", image_id_col, cell_type_col, "local_density"]
        if include_neighbor_counts:
            cols.append("num_neighbors")
        return pd.DataFrame(columns=cols)

    # Combine results
    output_df = pd.concat(results, ignore_index=True)

    # Reorder columns
    base_cols = ["cell_id", image_id_col, cell_type_col, "local_density"]
    if include_neighbor_counts:
        base_cols.append("num_neighbors")
    output_df = output_df[base_cols]

    return output_df


def _compute_local_density_single_image(
    image_df: pd.DataFrame,
    r_max: float,
    *,
    cell_type_col: str,
    focus_cell_type: str | None,
    include_cell_types: list[str] | None,
    box_scale: float,
    Lx: float | None,
    Ly: float | None,
    diameter: float,
    norm_by: str | None,
    include_neighbor_counts: bool,
) -> pd.DataFrame | None:
    """Compute local density for a single image.

    Returns None if too few cells after filtering.
    """
    # Get total cell count for normalization (before filtering)
    total_count = len(image_df)

    # Filter and extract data
    df_filtered, positions, focus_mask = val.filter_and_extract_data(
        image_df, cell_type_col, include_cell_types, focus_cell_type
    )

    # Skip if too few cells
    if len(df_filtered) < 2:
        return None

    # Setup freud box
    box, positions_3d = fh.setup_freud_box(positions, box_scale=box_scale, Lx=Lx, Ly=Ly)

    # Build neighbor list
    nlist = fh.build_neighbor_list_r_max(box, positions_3d, r_max, exclude_ii=True)

    # Compute local density
    ld = freud.density.LocalDensity(r_max, diameter)
    ld.compute((box, positions_3d), neighbors=nlist)
    density_values = ld.density

    # Apply normalization
    if norm_by == "total":
        density_values = density_values / total_count
    elif norm_by == "included":
        density_values = density_values / len(df_filtered)
    elif norm_by == "focus":
        if focus_cell_type is None:
            raise ValueError("norm_by='focus' requires focus_cell_type to be specified")
        focus_count = np.sum(focus_mask) if focus_mask is not None else len(df_filtered)
        if focus_count == 0:
            warnings.warn(
                "No focus cells found for normalization, skipping image",
                UserWarning,
                stacklevel=4,
            )
            return None
        density_values = density_values / focus_count

    # Compute neighbor counts if requested
    neighbor_counts = None
    if include_neighbor_counts:
        n_points = len(positions_3d)
        neighbor_counts = np.bincount(nlist.query_point_indices, minlength=n_points)

    # Filter to focus cells if specified
    if focus_mask is not None:
        df_filtered = df_filtered[focus_mask].copy()
        density_values = density_values[focus_mask]
        if neighbor_counts is not None:
            neighbor_counts = neighbor_counts[focus_mask]

    # Build output DataFrame
    result = pd.DataFrame(
        {
            "cell_id": df_filtered["cell_id"].values,
            cell_type_col: df_filtered[cell_type_col].values,
            "local_density": density_values,
        }
    )

    if include_neighbor_counts:
        result["num_neighbors"] = neighbor_counts

    return result


def compute_katic_order(
    df: pd.DataFrame,
    k: int | list[int],
    r_max: float,
    *,
    cell_type_col: str = "cell_type",
    focus_cell_type: str | None = None,
    include_cell_types: list[str] | None = None,
    box_scale: float = 10.0,
    Lx: float | None = None,
    Ly: float | None = None,
    neighbor_mode: str = "r_max",
    image_id_col: str = "image_id",
) -> pd.DataFrame:
    """Compute k-atic (hexatic) order parameters for cells.

    Uses freud.order.Hexatic to compute ψ_k, the k-fold orientational
    order parameter. Returns |ψ_k| (magnitude) per cell.

    Two neighbor list modes:
    - "r_max": all neighbors within r_max (variable neighbor count)
    - "k_nearest": exactly k nearest neighbors within r_max
      (points with fewer than k neighbors within r_max are assigned NaN)

    Parameters
    ----------
    df : pd.DataFrame
        Standardized DataFrame from load_spatial_data().
    k : int | list[int]
        Symmetry order(s) to compute. E.g., 6 for hexatic, [4, 6, 8] for multiple.
    r_max : float
        Cutoff radius for neighbor search.
    cell_type_col : str, default "cell_type"
        Column name for cell type labels.
    focus_cell_type : str | None, optional
        If provided, only return order values for this cell type.
    include_cell_types : list[str] | None, optional
        Cell types to include in point cloud. None = all.
    box_scale : float, default 10.0
        Box scale factor. For non-periodic data, use at least 10x.
    Lx : float | None, optional
        Explicit box length in x.
    Ly : float | None, optional
        Explicit box length in y.
    neighbor_mode : str, default "r_max"
        "r_max" (all within cutoff) or "k_nearest" (exactly k nearest within cutoff).
    image_id_col : str, default "image_id"
        Column name for image ID.

    Returns:
    -------
    pd.DataFrame
        DataFrame with columns:
        - cell_id : from input
        - image_id : image identifier
        - cell_type : cell type label
        - psi_{k} : one column per k value, containing |ψ_k| (float, 0 to 1)

    Raises:
    ------
    ValueError
        If k < 2 (or any value in list < 2).
        If r_max <= 0.
        If neighbor_mode not in {"r_max", "k_nearest"}.
        If required columns missing or cell type validation fails.

    Warns:
    -----
    UserWarning
        If an image has < 2 cells after filtering (skips that image).

    Examples:
    --------
    >>> # Hexatic order (k=6) for all cells
    >>> order_df = desc.compute_katic_order(df, k=6, r_max=50.0)
    >>> # Multiple k values with k-nearest mode
    >>> order_df = desc.compute_katic_order(
    ...     df, k=[4, 6, 8], r_max=50.0,
    ...     focus_cell_type="Epi",
    ...     include_cell_types=["Epi", "Treg"],
    ...     neighbor_mode="k_nearest"
    ... )
    """
    # Validate inputs
    k_list = val.validate_k_values(k)
    val.validate_r_max(r_max)
    val.validate_neighbor_mode(neighbor_mode)
    required_cols = ["cell_id", "x", "y", cell_type_col, image_id_col]
    val.validate_dataframe_columns(df, required_cols)
    val.validate_cell_types(df, cell_type_col, focus_cell_type, include_cell_types)

    # Process per image
    results = []

    for image_id, image_df in df.groupby(image_id_col):
        result = _compute_katic_order_single_image(
            image_df,
            k_list,
            r_max,
            cell_type_col=cell_type_col,
            focus_cell_type=focus_cell_type,
            include_cell_types=include_cell_types,
            box_scale=box_scale,
            Lx=Lx,
            Ly=Ly,
            neighbor_mode=neighbor_mode,
        )

        if result is not None:
            result[image_id_col] = image_id
            results.append(result)

    if not results:
        warnings.warn(
            "No valid images found (all had < 2 cells after filtering)",
            UserWarning,
            stacklevel=2,
        )
        # Return empty DataFrame with correct columns
        cols = ["cell_id", image_id_col, cell_type_col] + [f"psi_{ki}" for ki in k_list]
        return pd.DataFrame(columns=cols)

    # Combine results
    output_df = pd.concat(results, ignore_index=True)

    # Reorder columns
    base_cols = ["cell_id", image_id_col, cell_type_col]
    psi_cols = [f"psi_{ki}" for ki in k_list]
    output_df = output_df[base_cols + psi_cols]

    return output_df


def _compute_katic_order_single_image(
    image_df: pd.DataFrame,
    k_list: list[int],
    r_max: float,
    *,
    cell_type_col: str,
    focus_cell_type: str | None,
    include_cell_types: list[str] | None,
    box_scale: float,
    Lx: float | None,
    Ly: float | None,
    neighbor_mode: str,
) -> pd.DataFrame | None:
    """Compute k-atic order for a single image.

    Returns None if too few cells after filtering.
    """
    # Filter and extract data
    df_filtered, positions, focus_mask = val.filter_and_extract_data(
        image_df, cell_type_col, include_cell_types, focus_cell_type
    )

    # Skip if too few cells
    if len(df_filtered) < 2:
        return None

    # Setup freud box
    box, positions_3d = fh.setup_freud_box(positions, box_scale=box_scale, Lx=Lx, Ly=Ly)

    # Initialize output data
    result_data = {
        "cell_id": df_filtered["cell_id"].values,
        cell_type_col: df_filtered[cell_type_col].values,
    }

    # Compute order parameter for each k
    for ki in k_list:
        # Build neighbor list based on mode
        if neighbor_mode == "r_max":
            nlist = fh.build_neighbor_list_r_max(box, positions_3d, r_max, exclude_ii=True)
            valid_mask: np.ndarray = np.ones(len(positions_3d), dtype=bool)
        else:  # k_nearest
            nlist, valid_mask = fh.build_neighbor_list_k_nearest(box, positions_3d, ki, r_max)

        # Compute hexatic order
        hexatic = freud.order.Hexatic(k=ki)
        hexatic.compute((box, positions_3d), neighbors=nlist)
        order_values = np.abs(hexatic.particle_order)

        # Set invalid points to NaN in k_nearest mode
        if neighbor_mode == "k_nearest":
            order_values = order_values.copy()
            order_values[~valid_mask] = np.nan

        # Filter to focus cells if specified
        if focus_mask is not None:
            order_values_focus = order_values[focus_mask]
        else:
            order_values_focus = order_values

        result_data[f"psi_{ki}"] = order_values_focus

    # Build output DataFrame (only focus cells if specified)
    if focus_mask is not None:
        result_data["cell_id"] = df_filtered[focus_mask]["cell_id"].values
        result_data[cell_type_col] = df_filtered[focus_mask][cell_type_col].values

    result = pd.DataFrame(result_data)

    return result


def count_neighbor_cell_types(
    df: pd.DataFrame,
    r_max: float,
    *,
    cell_type_col: str = "cell_type",
    focus_cell_type: str | None = None,
    include_cell_types: list[str] | None = None,
    box_scale: float = 10.0,
    Lx: float | None = None,
    Ly: float | None = None,
    image_id_col: str = "image_id",
) -> pd.DataFrame:
    """Count neighbors of each cell type within r_max for each cell.

    For each cell, counts how many neighbors of each type are within
    the cutoff radius. Useful for characterizing the local cell
    type composition around a focus cell type.

    Parameters
    ----------
    df : pd.DataFrame
        Standardized DataFrame from load_spatial_data().
    r_max : float
        Cutoff radius for neighbor counting.
    cell_type_col : str, default "cell_type"
        Column name for cell type labels.
    focus_cell_type : str | None, optional
        If provided, only return counts for this cell type.
    include_cell_types : list[str] | None, optional
        Cell types to include in point cloud. None = all.
    box_scale : float, default 10.0
        Box scale factor. For non-periodic data, use at least 10x.
    Lx : float | None, optional
        Explicit box length in x.
    Ly : float | None, optional
        Explicit box length in y.
    image_id_col : str, default "image_id"
        Column name for image ID.

    Returns:
    -------
    pd.DataFrame
        DataFrame with columns:
        - cell_id : from input
        - image_id : image identifier
        - cell_type : cell type label
        - n_neighbors_{type_name} : one column per cell type in included set

    Raises:
    ------
    ValueError
        If r_max <= 0.
        If required columns missing or cell type validation fails.

    Warns:
    -----
    UserWarning
        If an image has < 2 cells after filtering (skips that image).

    Examples:
    --------
    >>> # Count neighbor types for all cells
    >>> counts_df = desc.count_neighbor_cell_types(df, r_max=50.0)
    >>> # Count neighbors around Epi cells only
    >>> counts_epi = desc.count_neighbor_cell_types(
    ...     df, r_max=50.0, focus_cell_type="Epi"
    ... )
    """
    # Validate inputs
    val.validate_r_max(r_max)
    required_cols = ["cell_id", "x", "y", cell_type_col, image_id_col]
    val.validate_dataframe_columns(df, required_cols)
    val.validate_cell_types(df, cell_type_col, focus_cell_type, include_cell_types)

    # Determine which cell types to count
    if include_cell_types is not None:
        types_to_count = sorted(include_cell_types)
    else:
        types_to_count = sorted(df[cell_type_col].unique())

    # Process per image
    results = []

    for image_id, image_df in df.groupby(image_id_col):
        result = _count_neighbor_cell_types_single_image(
            image_df,
            r_max,
            types_to_count,
            cell_type_col=cell_type_col,
            focus_cell_type=focus_cell_type,
            include_cell_types=include_cell_types,
            box_scale=box_scale,
            Lx=Lx,
            Ly=Ly,
        )

        if result is not None:
            result[image_id_col] = image_id
            results.append(result)

    if not results:
        warnings.warn(
            "No valid images found (all had < 2 cells after filtering)",
            UserWarning,
            stacklevel=2,
        )
        # Return empty DataFrame with correct columns
        cols = ["cell_id", image_id_col, cell_type_col]
        cols += [f"n_neighbors_{t}" for t in types_to_count]
        return pd.DataFrame(columns=cols)

    # Combine results
    output_df = pd.concat(results, ignore_index=True)

    # Reorder columns
    base_cols = ["cell_id", image_id_col, cell_type_col]
    count_cols = [f"n_neighbors_{t}" for t in types_to_count]
    output_df = output_df[base_cols + count_cols]

    return output_df


def _count_neighbor_cell_types_single_image(
    image_df: pd.DataFrame,
    r_max: float,
    types_to_count: list[str],
    *,
    cell_type_col: str,
    focus_cell_type: str | None,
    include_cell_types: list[str] | None,
    box_scale: float,
    Lx: float | None,
    Ly: float | None,
) -> pd.DataFrame | None:
    """Count neighbor cell types for a single image.

    Returns None if too few cells after filtering.
    """
    # Filter and extract data
    df_filtered, positions, focus_mask = val.filter_and_extract_data(
        image_df, cell_type_col, include_cell_types, focus_cell_type
    )

    # Skip if too few cells
    if len(df_filtered) < 2:
        return None

    # Setup freud box
    box, positions_3d = fh.setup_freud_box(positions, box_scale=box_scale, Lx=Lx, Ly=Ly)

    # Build neighbor list
    nlist = fh.build_neighbor_list_r_max(box, positions_3d, r_max, exclude_ii=True)

    # Get cell types for all included cells
    cell_types_array = df_filtered[cell_type_col].values

    # Count neighbors by type for each cell
    n_cells = len(df_filtered)
    count_matrix: np.ndarray = np.zeros((n_cells, len(types_to_count)), dtype=int)

    for i in range(n_cells):
        # Get neighbors of cell i
        mask = nlist.query_point_indices == i
        neighbor_indices = nlist.point_indices[mask]
        neighbor_types = cell_types_array[neighbor_indices]

        # Count by type
        for j, cell_type in enumerate(types_to_count):
            count_matrix[i, j] = np.sum(neighbor_types == cell_type)

    # Build result data
    result_data = {
        "cell_id": df_filtered["cell_id"].values,
        cell_type_col: df_filtered[cell_type_col].values,
    }

    for j, cell_type in enumerate(types_to_count):
        result_data[f"n_neighbors_{cell_type}"] = count_matrix[:, j]

    # Filter to focus cells if specified
    if focus_mask is not None:
        for key in result_data:
            result_data[key] = result_data[key][focus_mask]

    result = pd.DataFrame(result_data)

    return result
