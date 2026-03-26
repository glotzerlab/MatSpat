"""Assemble correlation trajectories into multivariate sequences for HMM input.

Public functions:
    build_sequences         DataFrame → dict of (n_bins, n_features) arrays.
    build_sequences_from_dict  Nested dict → dict of (n_bins, n_features) arrays.
    extract_labels          DataFrame → dict of str keys to int labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_sequences(
    df: pd.DataFrame,
    *,
    group_by: str = "image_id",
    bin_col: str = "quantile_bin",
    feature_col: str = "immune_pair",
    value_col: str = "lees_L",
    exclude_features: list[str] | None = None,
    reverse: bool = False,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Assemble per-image multivariate sequences from a long-format DataFrame.

    The DataFrame should have one row per (image, quantile_bin, immune_pair)
    with the Lee's L value.  This function pivots it into a matrix per image.

    Args:
        df: Long-format DataFrame with columns for group identity, quantile bin
            index, immune pair name, and Lee's L value.
        group_by: Column identifying each sample (image/slide).
        bin_col: Column with quantile bin indices (integer, determines order).
        feature_col: Column with immune pair names (multivariate dimensions).
        value_col: Column with the Lee's L correlation values.
        exclude_features: Optional list of feature names to drop.
        reverse: If True, reverse the bin ordering.

    Returns:
        Tuple of:
        - Dict mapping group key → ndarray of shape (n_bins, n_features).
          Missing values are filled with 0.
        - Sorted list of feature names (column order in the arrays).

    Raises:
        KeyError: If required columns are missing from *df*.
    """
    required = {group_by, bin_col, feature_col, value_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame missing required columns: {missing}")

    all_features = sorted(df[feature_col].unique())
    if exclude_features:
        all_features = [f for f in all_features if f not in exclude_features]

    sequences: dict[str, np.ndarray] = {}
    for key, grp in df.groupby(group_by):
        pivot = grp.pivot_table(
            index=bin_col,
            columns=feature_col,
            values=value_col,
            aggfunc="first",
        )
        pivot = pivot.sort_index(ascending=not reverse)
        pivot = pivot.reindex(columns=all_features, fill_value=0.0)
        sequences[str(key)] = pivot.to_numpy(dtype=np.float64, na_value=0.0)

    return sequences, all_features


def extract_labels(
    df: pd.DataFrame,
    *,
    group_by: str = "image_id",
    label_col: str = "label",
) -> dict[str, int]:
    """Extract integer class labels keyed by group, matching :func:`build_sequences`.

    Uses the same ``str(key)`` coercion as :func:`build_sequences` so that
    the returned dict keys are guaranteed to align with the sequence dict.

    String labels are automatically mapped to integers (alphabetically sorted,
    0-based).

    Args:
        df: DataFrame containing at least *group_by* and *label_col* columns.
        group_by: Column identifying each sample.
        label_col: Column with the class label.

    Returns:
        Dict mapping ``str(group_key)`` → integer class label.

    Raises:
        KeyError: If required columns are missing.
    """
    required = {group_by, label_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame missing required columns: {missing}")

    label_series = df.groupby(group_by)[label_col].first()

    # Convert string labels to int if needed
    if label_series.dtype == object or isinstance(label_series.iloc[0], str):
        unique_sorted = sorted(label_series.unique())
        label_map = {v: i for i, v in enumerate(unique_sorted)}
        label_series = label_series.map(label_map)

    return {str(k): int(v) for k, v in label_series.items()}


def build_sequences_from_dict(
    trajectories: dict[str, dict[str, list[float]]],
    *,
    feature_order: list[str] | None = None,
    reverse: bool = False,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Assemble sequences from a nested dict (alternative to DataFrame input).

    Args:
        trajectories: Nested dict of
            ``{image_id: {immune_pair: [L_values_per_bin]}}``.
        feature_order: Explicit ordering of immune pair columns.  If *None*,
            sorted alphabetically from all keys found.
        reverse: If True, reverse the bin ordering.

    Returns:
        Same as :func:`build_sequences`: ``(sequences_dict, feature_names)``.

    Raises:
        ValueError: If *trajectories* is empty.
    """
    if not trajectories:
        raise ValueError("trajectories dict is empty")

    if feature_order is None:
        all_features: set[str] = set()
        for pairs in trajectories.values():
            all_features.update(pairs.keys())
        features = sorted(all_features)
    else:
        features = list(feature_order)

    sequences: dict[str, np.ndarray] = {}
    for image_id, pairs in trajectories.items():
        n_bins = max((len(v) for v in pairs.values()), default=0)
        arr = np.zeros((n_bins, len(features)), dtype=np.float64)
        for j, feat in enumerate(features):
            vals = pairs.get(feat, [])
            arr[: len(vals), j] = vals
        if reverse:
            arr = arr[::-1].copy()
        sequences[image_id] = arr

    return sequences, features
