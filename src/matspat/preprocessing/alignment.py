"""Coordinate transformations and spatial alignment utilities.

Functions:
    rotate_coords        Rotate 2-D coordinates by a given angle.
    process_rotation     ICA-based alignment of the dominant tumor axis to +Y.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import gaussian_kde
from sklearn.decomposition import FastICA


def rotate_coords(coords: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    """Rotate 2D coordinates by a given angle.

    Args:
        coords: Array of shape (N, 2) containing (x, y) coordinates.
        angle: Rotation angle in radians (counterclockwise).

    Returns:
        Rotated coordinates as array of shape (N, 2).

    Examples:
        >>> coords = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> rotated = rotate_coords(coords, np.pi / 2)  # 90 degrees
        >>> np.allclose(rotated, [[0.0, 1.0], [-1.0, 0.0]])
        True
    """
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return coords.dot(rotation_matrix.T)


def _density_peak(vals: NDArray[np.floating]) -> float:
    """Peak density of a 1-D KDE over *vals*."""
    kde = gaussian_kde(vals)
    grid = np.linspace(float(vals.min()), float(vals.max()), 200)
    return float(kde(grid).max())


def process_rotation(
    df: pd.DataFrame,
    tumor_label: str,
    *,
    x_col: str = "x",
    y_col: str = "y",
    cell_type_col: str = "cell_type",
    random_state: int = 0,
) -> pd.DataFrame:
    """Rotate all cells so the dominant tumor ICA axis points north (+Y).

    Fits FastICA on tumor-cell coordinates to find two independent axes,
    picks the one with higher density peak, and rotates all cells so that
    axis aligns with +Y.

    Args:
        df: DataFrame with spatial coordinates and cell-type labels.
        tumor_label: Value in *cell_type_col* that identifies tumor cells.
        x_col: Column name for x coordinates.
        y_col: Column name for y coordinates.
        cell_type_col: Column name for cell-type annotations.
        random_state: Seed for FastICA reproducibility.

    Returns:
        Copy of *df* with added ``x_rot`` and ``y_rot`` columns.
        If fewer than 2 tumor cells are found, ``x_rot``/``y_rot`` are
        copies of the original coordinates (no rotation applied).
    """
    df = df.copy()
    coords = df[[x_col, y_col]].to_numpy(dtype=np.float64)

    tumor_coords = df.loc[df[cell_type_col] == tumor_label, [x_col, y_col]].to_numpy(
        dtype=np.float64
    )

    if tumor_coords.shape[0] < 2:
        warnings.warn(
            f"Skipping ICA rotation: only {tumor_coords.shape[0]} " f"'{tumor_label}' cells found",
            stacklevel=2,
        )
    else:
        ica = FastICA(n_components=2, random_state=random_state)
        S = ica.fit_transform(tumor_coords)

        # Pick the component with higher density peak
        comp = 0 if _density_peak(S[:, 0]) >= _density_peak(S[:, 1]) else 1
        vec = ica.components_[comp]

        # Ensure the vector points into the +Y half-plane
        if vec[1] < 0:
            vec = -vec

        # Angle to rotate this vector onto +Y
        angle = np.arctan2(vec[1], vec[0])
        rot = (np.pi / 2) - angle
        coords = rotate_coords(coords, rot)

    df["x_rot"] = coords[:, 0]
    df["y_rot"] = coords[:, 1]
    return df
