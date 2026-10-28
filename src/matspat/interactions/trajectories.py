"""Lee's L trajectories across tumor density zones.

Computes Lee's L spatial correlation stratified by quantile bins of the
tumor density surface.  Supports three pairing modes:

- ``"tumor_immune"`` (default): each immune type vs. tumor density
- ``"immune_immune"``: all unordered immune–immune pairs
- ``"all"``: both tumor–immune and immune–immune pairs

The output is a long-format DataFrame that plugs directly into
:func:`~matspat.classification.sequences.build_sequences`.

Public functions:
    compute_pairwise_trajectories
"""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

from matspat.config import (  # noqa: F401
    CELL_TYPE,
    DEFAULT_GRID_SIZE,
    IMAGE_ID,
    IMMUNE_PAIR,
    LEES_L,
    QUANTILE_BIN,
    X_ROT,
    Y_ROT,
)
from matspat.density.kde import (
    build_grid,
    get_cumulative_quantile_masks,
    get_marginal_quantile_masks,
    tissue_mask,
)
from matspat.interactions.correlation import lees_L_sequence, precompute_bin_kernels

_QUANTILE_METHODS = {
    "marginal": get_marginal_quantile_masks,
    "cumulative": get_cumulative_quantile_masks,
}

_PAIR_MODES = {"tumor_immune", "immune_immune", "all"}


def compute_pairwise_trajectories(
    df: pd.DataFrame,
    tumor_label: str,
    bandwidth: float,
    n_quantiles: int,
    *,
    pair_mode: str = "tumor_immune",
    quantile_method: str = "marginal",
    x_col: str = X_ROT,
    y_col: str = Y_ROT,
    cell_type_col: str = CELL_TYPE,
    image_id_col: str = IMAGE_ID,
    grid_size: int = DEFAULT_GRID_SIZE,
    exclude_types: list[str] | None = None,
    min_tumor_cells: int = 2,
    tissue_mask_distance: float | None = None,
) -> pd.DataFrame:
    """Compute Lee's L trajectories across the tumor density gradient.

    For each image, builds a shared evaluation grid from all cell coordinates,
    computes the tumor KDE to define quantile bins along the density gradient,
    then evaluates KDEs on the same grid and measures Lee's L within each bin.

    Args:
        df: DataFrame with spatial cell data.  Must contain columns for
            coordinates, cell type, and image identity.
        tumor_label: Value in *cell_type_col* that identifies tumor cells.
        bandwidth: KDE bandwidth (same spatial units as coordinates).
        n_quantiles: Number of quantile bins along the tumor density gradient.
        pair_mode: Which pairs to compute:

            - ``"tumor_immune"`` (default): Lee's L between tumor density and
              each immune type — one trajectory per immune type.
            - ``"immune_immune"``: Lee's L between every unordered pair of
              immune types.
            - ``"all"``: both tumor–immune and immune–immune pairs.
        quantile_method: ``"marginal"`` (non-overlapping equal-mass bins) or
            ``"cumulative"`` (nested regions).
        x_col: Column with x-coordinates (typically rotated).
        y_col: Column with y-coordinates (typically rotated).
        cell_type_col: Column with cell type annotations.
        image_id_col: Column identifying each image/slide.
        grid_size: Number of grid points per axis for KDE evaluation.
        exclude_types: Cell types to exclude from pair computation
            (in addition to *tumor_label*, which is always excluded from the
            immune set).
        min_tumor_cells: Minimum number of tumor cells required per image.
            Images below this threshold are skipped.  Default is 2 (the
            KDE minimum); set higher (e.g. 100) to exclude images whose
            sparse tumor KDE produces unreliable density gradients.
        tissue_mask_distance: If given, grid points farther than this distance
            from every cell are treated as off-tissue: they are excluded from
            the quantile thresholds and from every bin.  Without a mask, the
            empty corners of the bounding-box grid fall into the lowest
            tumor-density bins.  ``None`` (default) keeps the full grid.

    Returns:
        Long-format DataFrame with columns:

        - *image_id_col*: image identifier
        - ``"immune_pair"``: pair label (e.g. ``"Tumor x CD4"`` or
          ``"CD4 x CD8"``)
        - ``"quantile_bin"``: bin index (0-based)
        - ``"lees_L"``: Lee's L correlation value

        Ready for :func:`~matspat.classification.sequences.build_sequences`.

    Raises:
        ValueError: If *quantile_method* or *pair_mode* is invalid.
        KeyError: If required columns are missing from *df*.
    """
    if quantile_method not in _QUANTILE_METHODS:
        raise ValueError(
            f"quantile_method must be one of {set(_QUANTILE_METHODS)}, " f"got {quantile_method!r}"
        )
    if pair_mode not in _PAIR_MODES:
        raise ValueError(f"pair_mode must be one of {_PAIR_MODES}, got {pair_mode!r}")

    required_cols = {x_col, y_col, cell_type_col, image_id_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame missing required columns: {missing}")

    mask_fn = _QUANTILE_METHODS[quantile_method]
    skip = {tumor_label}
    if exclude_types:
        skip.update(exclude_types)

    rows: list[dict] = []

    for image_id, img_df in df.groupby(image_id_col):
        _process_image(
            img_df,
            image_id=image_id,
            tumor_label=tumor_label,
            bandwidth=bandwidth,
            n_quantiles=n_quantiles,
            pair_mode=pair_mode,
            mask_fn=mask_fn,
            skip=skip,
            x_col=x_col,
            y_col=y_col,
            cell_type_col=cell_type_col,
            image_id_col=image_id_col,
            grid_size=grid_size,
            min_tumor_cells=min_tumor_cells,
            tissue_mask_distance=tissue_mask_distance,
            rows=rows,
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _min_immune_types(pair_mode: str) -> int:
    """Minimum number of immune types required for the given pair mode."""
    if pair_mode == "immune_immune":
        return 2
    return 1  # tumor_immune or all


def _grid_tissue_mask(
    coords: np.ndarray,
    grid_points: np.ndarray,
    grid_size: int,
    max_distance: float | None,
) -> np.ndarray | None:
    """2-D in-tissue mask for the grid, or None when masking is disabled."""
    if max_distance is None:
        return None
    mask: np.ndarray = np.asarray(tissue_mask(coords, grid_points, max_distance), dtype=bool)
    return mask.reshape(grid_size, grid_size)


def _process_image(
    img_df: pd.DataFrame,
    *,
    image_id: object,
    tumor_label: str,
    bandwidth: float,
    n_quantiles: int,
    pair_mode: str,
    mask_fn,
    skip: set[str],
    x_col: str,
    y_col: str,
    cell_type_col: str,
    image_id_col: str,
    grid_size: int,
    min_tumor_cells: int,
    tissue_mask_distance: float | None,
    rows: list[dict],
) -> None:
    """Compute pairwise trajectories for a single image."""
    tumor_coords = img_df.loc[img_df[cell_type_col] == tumor_label, [x_col, y_col]].values

    if tumor_coords.shape[0] < min_tumor_cells:
        warnings.warn(
            f"Image {image_id!r}: {tumor_coords.shape[0]} tumor cells "
            f"(< min_tumor_cells={min_tumor_cells}), skipping.",
            stacklevel=3,
        )
        return

    # Shared grid from ALL cells so every KDE is evaluated at the same points
    all_coords = img_df[[x_col, y_col]].values
    _xs, _ys, grid_points = build_grid(all_coords, grid_size)

    # Tumor KDE -> quantile masks
    tumor_kde = KernelDensity(bandwidth=bandwidth).fit(tumor_coords)
    tumor_density_flat = np.exp(tumor_kde.score_samples(grid_points))
    tumor_density_2d = tumor_density_flat.reshape(grid_size, grid_size)
    valid = _grid_tissue_mask(all_coords, grid_points, grid_size, tissue_mask_distance)
    quantile_masks = mask_fn(tumor_density_2d, n_quantiles, valid=valid)
    masks_list = [quantile_masks[i] for i in range(n_quantiles)]

    # Identify immune types with enough cells for KDE
    min_types = _min_immune_types(pair_mode)
    immune_types = sorted(
        ct
        for ct in img_df[cell_type_col].unique()
        if ct not in skip and (img_df[cell_type_col] == ct).sum() >= 2
    )
    if len(immune_types) < min_types:
        warnings.warn(
            f"Image {image_id!r}: fewer than {min_types} immune type(s) with "
            f">= 2 cells, skipping.",
            stacklevel=3,
        )
        return

    # KDE for each immune type on the shared grid
    immune_densities: dict[str, np.ndarray] = {}
    for ct in immune_types:
        coords = img_df.loc[img_df[cell_type_col] == ct, [x_col, y_col]].values
        kde = KernelDensity(bandwidth=bandwidth).fit(coords)
        immune_densities[ct] = np.exp(kde.score_samples(grid_points))

    # Precompute the bisquare kernel matrix for each quantile bin once —
    # the grid subsets are identical across all immune pairs.
    bin_kernels = precompute_bin_kernels(grid_points, bandwidth, masks_list)

    # --- Tumor–immune pairs ---
    if pair_mode in ("tumor_immune", "all"):
        for ct in immune_types:
            L_values = lees_L_sequence(
                tumor_density_flat,
                immune_densities[ct],
                grid_points,
                bandwidth,
                masks_list,
                bin_kernels=bin_kernels,
            )
            pair_label = f"{tumor_label} x {ct}"
            for q, L in enumerate(L_values):
                rows.append(
                    {
                        image_id_col: image_id,
                        IMMUNE_PAIR: pair_label,
                        QUANTILE_BIN: q,
                        LEES_L: L,
                    }
                )

    # --- Immune–immune pairs ---
    if pair_mode in ("immune_immune", "all"):
        for ct_a, ct_b in combinations(immune_types, 2):
            L_values = lees_L_sequence(
                immune_densities[ct_a],
                immune_densities[ct_b],
                grid_points,
                bandwidth,
                masks_list,
                bin_kernels=bin_kernels,
            )
            pair_label = f"{ct_a} x {ct_b}"
            for q, L in enumerate(L_values):
                rows.append(
                    {
                        image_id_col: image_id,
                        IMMUNE_PAIR: pair_label,
                        QUANTILE_BIN: q,
                        LEES_L: L,
                    }
                )
