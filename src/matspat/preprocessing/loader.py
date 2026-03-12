"""Load and standardize spatial CSV data with arbitrary column names.

Main function: load_spatial_data()
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd


def _validate_file_and_columns(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    cell_type_col: str,
    cell_id_col: str | None,
    pathology_col: str | None,
    image_id_col: str | None,
    patient_id_col: str | None,
) -> None:
    """Validate that required and optional columns exist in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded DataFrame
    x_col, y_col, cell_type_col : str
        Required column names
    cell_id_col, pathology_col, image_id_col, patient_id_col : str | None
        Optional column names

    Raises:
    ------
    ValueError
        If required columns or specified optional columns are missing
    """
    # Validate required columns exist
    required_cols = [x_col, y_col, cell_type_col]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(
            f"Required columns not found in CSV: {missing_required}. "
            f"Available columns: {list(df.columns)}"
        )

    # Validate optional columns exist (if specified)
    optional_cols_map = {
        "cell_id_col": cell_id_col,
        "pathology_col": pathology_col,
        "image_id_col": image_id_col,
        "patient_id_col": patient_id_col,
    }
    for param_name, col_name in optional_cols_map.items():
        if col_name is not None and col_name not in df.columns:
            raise ValueError(
                f"Specified {param_name}='{col_name}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )


def _handle_nan_in_required_columns(
    df: pd.DataFrame, x_col: str, y_col: str, cell_type_col: str
) -> pd.DataFrame:
    """Drop rows with NaN in required columns and warn user.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    x_col, y_col, cell_type_col : str
        Required column names

    Returns:
    -------
    pd.DataFrame
        DataFrame with NaN rows removed

    Raises:
    ------
    ValueError
        If DataFrame is empty after dropping NaN rows

    Warns:
    -----
    UserWarning
        If rows are dropped due to NaN values
    """
    required_data_cols = [x_col, y_col, cell_type_col]
    nan_mask = df[required_data_cols].isna().any(axis=1)

    if nan_mask.any():
        nan_indices = df.index[nan_mask].tolist()
        num_dropped = nan_mask.sum()
        warnings.warn(
            f"Dropped {num_dropped} rows with NaN in required columns (x, y, cell_type). "
            f"Original row indices: {nan_indices}",
            UserWarning,
            stacklevel=3,
        )
        df = df[~nan_mask].copy()

    if df.empty:
        raise ValueError("No data rows found in CSV (empty after dropping NaN rows)")

    return df


def _warn_nan_in_optional_columns(
    df: pd.DataFrame,
    pathology_col: str | None,
    image_id_col: str | None,
    patient_id_col: str | None,
) -> None:
    """Warn user if optional columns have mixed NaN/non-NaN values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    pathology_col, image_id_col, patient_id_col : str | None
        Optional column names

    Warns:
    -----
    UserWarning
        If optional columns have mixed NaN/non-NaN values
    """
    optional_data_cols = []
    if pathology_col:
        optional_data_cols.append(pathology_col)
    if image_id_col:
        optional_data_cols.append(image_id_col)
    if patient_id_col:
        optional_data_cols.append(patient_id_col)

    for col in optional_data_cols:
        has_nan = df[col].isna().any()
        has_non_nan = df[col].notna().any()
        if has_nan and has_non_nan:
            nan_indices = df.index[df[col].isna()].tolist()
            warnings.warn(
                f"Column '{col}' has mixed NaN/non-NaN values. "
                f"Rows with NaN: {nan_indices}. "
                f"These rows are kept in the output.",
                UserWarning,
                stacklevel=3,
            )


def _process_cell_id(df: pd.DataFrame, cell_id_col: str | None) -> pd.Series:
    """Process cell_id column: validate if provided, auto-generate if not.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    cell_id_col : str | None
        Name of cell_id column, or None to auto-generate

    Returns:
    -------
    pd.Series
        Cell ID series (1-indexed)

    Raises:
    ------
    ValueError
        If provided cell_id column contains duplicates
    """
    if cell_id_col is not None:
        # Check for duplicates
        if df[cell_id_col].duplicated().any():
            dup_values = df[cell_id_col][df[cell_id_col].duplicated()].unique().tolist()
            raise ValueError(
                f"Duplicate cell_id values found: {dup_values}. "
                f"Please provide unique cell IDs or omit cell_id_col to use auto-generated IDs."
            )
        return df[cell_id_col].astype(int)
    else:
        # Auto-generate 1-indexed cell_ids
        return pd.Series(range(1, len(df) + 1), index=df.index)


def _process_cell_type(
    df: pd.DataFrame, cell_type_col: str
) -> tuple[pd.Series, pd.Series, dict[str, int]]:
    """Process cell_type: convert to string and create alphabetical encoding.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    cell_type_col : str
        Name of cell_type column

    Returns:
    -------
    tuple[pd.Series, pd.Series, dict[str, int]]
        (cell_type series, cell_type_encoded series, mapping dict)
    """
    cell_type = df[cell_type_col].astype(str)
    unique_cell_types = sorted(cell_type.unique())
    cell_type_mapping = {ct: idx + 1 for idx, ct in enumerate(unique_cell_types)}
    cell_type_encoded = cell_type.map(cell_type_mapping)
    return cell_type, cell_type_encoded, cell_type_mapping


def _process_image_id(df: pd.DataFrame, image_id_col: str | None) -> pd.Series:
    """Process image_id: use column if provided, auto-generate with warning if not.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    image_id_col : str | None
        Name of image_id column, or None to auto-generate

    Returns:
    -------
    pd.Series
        Image ID series

    Warns:
    -----
    UserWarning
        If image_id is auto-generated
    """
    if image_id_col is not None:
        return df[image_id_col].astype(str)
    else:
        warnings.warn(
            "No image_id_col provided. Treating all cells as single image (image_id='image_1').",
            UserWarning,
            stacklevel=3,
        )
        return pd.Series("image_1", index=df.index)


def load_spatial_data(
    filepath: str | Path,
    *,
    # Required column mappings
    x_col: str,
    y_col: str,
    cell_type_col: str,
    # Optional: cell_id (auto-generated if not provided)
    cell_id_col: str | None = None,
    # Optional: pathology (omitted from output if not provided)
    pathology_col: str | None = None,
    # Optional: image_id (auto-generated as "image_1" if not provided, with warning)
    image_id_col: str | None = None,
    # Optional: patient_id
    patient_id_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load a spatial data CSV and standardize to MatSpat format.

    Returns a tuple of (DataFrame, cell_type_mapping) where:
    - DataFrame has standardized columns
    - cell_type_mapping is a dict mapping cell_type (str) -> cell_type_encoded (int)

    DataFrame columns:
        - cell_id (int): Unique cell identifier (1-indexed)
        - x (float): X coordinate
        - y (float): Y coordinate
        - cell_type (str): Cell type/category label
        - cell_type_encoded (int): Integer encoding of cell_type (1-indexed, alphabetical)
        - image_id (str): Image/sample identifier (auto-generated if not provided)
        - pathology (str, optional): Disease type / pathology identifier (only if pathology_col provided)
        - patient_id (str, optional): Patient identifier (only if patient_id_col provided)

    Parameters
    ----------
    filepath : str | Path
        Path to CSV file
    x_col : str
        Column name for X coordinates
    y_col : str
        Column name for Y coordinates
    cell_type_col : str
        Column name for cell type labels
    cell_id_col : str | None, optional
        Column name for cell IDs (auto-generated if None)
    pathology_col : str | None, optional
        Column name for pathology/disease type (omitted if None)
    image_id_col : str | None, optional
        Column name for image/sample ID (auto-generated as "image_1" if None)
    patient_id_col : str | None, optional
        Column name for patient ID (omitted if None)

    Returns:
    -------
    tuple[pd.DataFrame, dict[str, int]]
        Standardized DataFrame and cell_type -> encoded ID mapping

    Raises:
    ------
    FileNotFoundError
        If the specified file does not exist
    ValueError
        If required columns are missing or invalid
        If specified optional columns don't exist in the CSV
        If empty CSV after header (no data rows)
        If provided cell_id has duplicates (prompts for auto-generation)

    Warns:
    -----
    UserWarning
        If image_id_col not provided (treating all cells as single image)
        If rows with NaN in required columns (x, y, cell_type) are dropped
        If optional columns have mixed NaN/non-NaN values

    Examples:
    --------
    Minimal usage (required columns only):

    >>> df, mapping = load_spatial_data(
    ...     "data/my_cells.csv",
    ...     x_col="X",
    ...     y_col="Y",
    ...     cell_type_col="type"
    ... )
    >>> df.columns
    Index(['cell_id', 'x', 'y', 'cell_type', 'cell_type_encoded', 'image_id'], dtype='object')
    >>> mapping
    {'CD4': 1, 'CD8': 2, 'Tumor': 3}

    Full mapping with all optional columns:

    >>> df, mapping = load_spatial_data(
    ...     "data/crc_raw.csv",
    ...     x_col="X",
    ...     y_col="Y",
    ...     cell_type_col="type",
    ...     pathology_col="pathology",
    ...     image_id_col="img",
    ...     patient_id_col="patient"
    ... )
    >>> df.columns
    Index(['cell_id', 'x', 'y', 'cell_type', 'cell_type_encoded',
           'image_id', 'pathology', 'patient_id'], dtype='object')
    """
    # Convert filepath to Path object
    filepath = Path(filepath)

    # Check file existence
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Load CSV
    df = pd.read_csv(filepath)

    # Validate column names
    _validate_file_and_columns(
        df,
        x_col,
        y_col,
        cell_type_col,
        cell_id_col,
        pathology_col,
        image_id_col,
        patient_id_col,
    )

    # Handle NaN in required columns (drop rows with NaN)
    df = _handle_nan_in_required_columns(df, x_col, y_col, cell_type_col)

    # Warn about NaN in optional columns (keep rows, just warn)
    _warn_nan_in_optional_columns(df, pathology_col, image_id_col, patient_id_col)

    # Build output DataFrame
    output_df = pd.DataFrame()

    # Process cell_id
    output_df["cell_id"] = _process_cell_id(df, cell_id_col)

    # Process x, y (explicit conversion to float)
    output_df["x"] = df[x_col].astype(float)
    output_df["y"] = df[y_col].astype(float)

    # Process cell_type and create encoding
    cell_type, cell_type_encoded, cell_type_mapping = _process_cell_type(df, cell_type_col)
    output_df["cell_type"] = cell_type
    output_df["cell_type_encoded"] = cell_type_encoded

    # Process image_id
    output_df["image_id"] = _process_image_id(df, image_id_col)

    # Handle optional columns (pathology, patient_id)
    if pathology_col is not None:
        output_df["pathology"] = df[pathology_col].astype(str)
    if patient_id_col is not None:
        output_df["patient_id"] = df[patient_id_col].astype(str)

    # Reorder columns: cell_id, x, y, cell_type, cell_type_encoded, image_id, then optional
    base_cols = ["cell_id", "x", "y", "cell_type", "cell_type_encoded", "image_id"]
    optional_cols = []
    if pathology_col is not None:
        optional_cols.append("pathology")
    if patient_id_col is not None:
        optional_cols.append("patient_id")

    output_df = output_df[base_cols + optional_cols]

    # Reset index to be clean
    output_df = output_df.reset_index(drop=True)

    return output_df, cell_type_mapping
