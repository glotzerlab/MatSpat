"""Shared validation and filtering logic for spatial descriptors."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def validate_r_max(r_max: float) -> None:
    """Validate that r_max is positive.

    Parameters
    ----------
    r_max : float
        Cutoff radius for neighbor search.

    Raises:
    ------
    ValueError
        If r_max <= 0.
    """
    if r_max <= 0:
        raise ValueError(f"r_max must be positive, got {r_max}")


def validate_k_values(k: int | list[int]) -> list[int]:
    """Validate k values and convert to list.

    Parameters
    ----------
    k : int | list[int]
        Symmetry order(s) to compute.

    Returns:
    -------
    list[int]
        List of k values.

    Raises:
    ------
    ValueError
        If any k < 2.
    """
    k_list = [k] if isinstance(k, int) else k
    if any(ki < 2 for ki in k_list):
        raise ValueError(f"All k values must be >= 2, got {k_list}")
    return k_list


def validate_neighbor_mode(neighbor_mode: str) -> None:
    """Validate neighbor mode parameter.

    Parameters
    ----------
    neighbor_mode : str
        Neighbor list construction mode.

    Raises:
    ------
    ValueError
        If neighbor_mode not in {"r_max", "k_nearest"}.
    """
    valid_modes = {"r_max", "k_nearest"}
    if neighbor_mode not in valid_modes:
        raise ValueError(f"neighbor_mode must be one of {valid_modes}, got '{neighbor_mode}'")


def validate_norm_by(norm_by: str | None) -> None:
    """Validate normalization method parameter.

    Parameters
    ----------
    norm_by : str | None
        Normalization method.

    Raises:
    ------
    ValueError
        If norm_by not in {None, "total", "included", "focus"}.
    """
    valid_norms = {None, "total", "included", "focus"}
    if norm_by not in valid_norms:
        raise ValueError(f"norm_by must be one of {valid_norms}, got '{norm_by}'")


def validate_dataframe_columns(
    df: pd.DataFrame,
    required_cols: list[str],
) -> None:
    """Validate that DataFrame has required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    required_cols : list[str]
        List of required column names.

    Raises:
    ------
    ValueError
        If any required columns are missing.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def validate_cell_types(
    df: pd.DataFrame,
    cell_type_col: str,
    focus_cell_type: str | None,
    include_cell_types: list[str] | None,
) -> None:
    """Validate cell type parameters.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cell_type_col : str
        Column name for cell types.
    focus_cell_type : str | None
        Focus cell type (must be in include_cell_types if both specified).
    include_cell_types : list[str] | None
        Cell types to include in analysis.

    Raises:
    ------
    ValueError
        If focus_cell_type specified but not in include_cell_types.
        If include_cell_types contains types not in the data.
    """
    available_types = set(df[cell_type_col].unique())

    # Validate include_cell_types are in the data
    if include_cell_types is not None:
        invalid_types = set(include_cell_types) - available_types
        if invalid_types:
            raise ValueError(
                f"include_cell_types contains types not found in data: {invalid_types}. "
                f"Available types: {available_types}"
            )

    # Validate focus_cell_type is in include_cell_types
    if focus_cell_type is not None and include_cell_types is not None:
        if focus_cell_type not in include_cell_types:
            raise ValueError(
                f"focus_cell_type '{focus_cell_type}' must be in include_cell_types {include_cell_types}"
            )

    # Validate focus_cell_type is in the data
    if focus_cell_type is not None and focus_cell_type not in available_types:
        raise ValueError(
            f"focus_cell_type '{focus_cell_type}' not found in data. "
            f"Available types: {available_types}"
        )


def filter_and_extract_data(
    df: pd.DataFrame,
    cell_type_col: str,
    include_cell_types: list[str] | None,
    focus_cell_type: str | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    """Filter DataFrame by cell types and extract positions.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with x, y, cell_type columns.
    cell_type_col : str
        Column name for cell types.
    include_cell_types : list[str] | None
        Cell types to include. None means include all.
    focus_cell_type : str | None
        Focus cell type for returning results. None means return all included.

    Returns:
    -------
    tuple[pd.DataFrame, np.ndarray, np.ndarray | None]
        - Filtered DataFrame (included cells only)
        - (N, 2) positions array
        - Boolean mask for focus cells (None if no focus type)

    Warns:
    -----
    UserWarning
        If fewer than 2 cells remain after filtering.
    """
    # Filter to included cell types
    if include_cell_types is not None:
        df_filtered = df[df[cell_type_col].isin(include_cell_types)].copy()
    else:
        df_filtered = df.copy()

    # Check minimum cell count
    if len(df_filtered) < 2:
        warnings.warn(
            f"Fewer than 2 cells after filtering (got {len(df_filtered)}). "
            f"Skipping this group.",
            UserWarning,
            stacklevel=4,
        )

    # Extract positions
    positions = df_filtered[["x", "y"]].values.astype(float)

    # Create focus mask if needed
    focus_mask = None
    if focus_cell_type is not None:
        focus_mask = (df_filtered[cell_type_col] == focus_cell_type).values

    return df_filtered, positions, focus_mask


def warn_if_few_cells(n_cells: int, threshold: int = 2) -> bool:
    """Warn and return False if too few cells.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    threshold : int
        Minimum required cells.

    Returns:
    -------
    bool
        True if enough cells, False otherwise.
    """
    if n_cells < threshold:
        warnings.warn(
            f"Fewer than {threshold} cells (got {n_cells}). Skipping.",
            UserWarning,
            stacklevel=4,
        )
        return False
    return True
