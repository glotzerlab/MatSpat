"""Bias-corrected distance correlation and partial distance correlation.

Distance correlation (Székely, Rizzo & Bakirov 2007) detects any dependence
between two variables, not only linear or monotonic association. Partial
distance correlation (Székely & Rizzo 2014) removes the dependence on a third
variable z. Both are computed here from U-centred distance matrices, which give
the bias-corrected estimator.

Unlike the correlation measures in :mod:`~matspat.interactions.correlation` and
:mod:`~matspat.interactions.local_corr`, these are not spatially weighted: within
a quantile bin, every grid point is one sample of (x, y, z).

The U-centred estimator estimates the *squared* distance correlation, hence
``distance_corr_sq``. Partial distance correlation is built from these squared
values, as defined by Székely & Rizzo (2014). Both can be slightly negative in
finite samples; values near 0 indicate (conditional) independence.

Degenerate cases follow Székely & Rizzo (2014): R* is 0 when a sample is
constant, and pdCor is 0 when z fully explains x or y (R*_xz^2 = 1 or
R*_yz^2 = 1).

Public functions:
    distance_corr_sq(x, y) -> float
    partial_distance_corr(x, y, z) -> float
    distance_corr_sq_sequence(x_full, y_full, quantile_masks) -> list[float]
    partial_distance_corr_sequence(x_full, y_full, z_full, quantile_masks) -> list[float]
"""

import numpy as np

# Default minimum points per bin. With very few points the U-centred space is
# low-dimensional and |R*| near 1 is common even for independent samples.
DEFAULT_MIN_POINTS = 10

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _u_centered(v: np.ndarray) -> np.ndarray:
    """U-centred pairwise distance matrix of a 1-D sample (requires n > 3)."""
    n = len(v)
    a = np.abs(v[:, None] - v[None, :])
    row = a.sum(axis=1)
    total = row.sum()
    u: np.ndarray = (
        a - row[:, None] / (n - 2) - row[None, :] / (n - 2) + total / ((n - 1) * (n - 2))
    )
    np.fill_diagonal(u, 0.0)
    return u


def _inner(u: np.ndarray, v: np.ndarray) -> float:
    """Unbiased inner product of two U-centred matrices."""
    n = u.shape[0]
    return float((u * v).sum() / (n * (n - 3)))


def _dcor_from_u(u: np.ndarray, v: np.ndarray) -> float:
    """R* from two U-centred matrices; 0 if either has zero norm (constant sample)."""
    uu, vv = _inner(u, u), _inner(v, v)
    if uu <= 0 or vv <= 0:
        return 0.0
    return float(_inner(u, v) / np.sqrt(uu * vv))


def _finite_bin(mask: np.ndarray, *fields: np.ndarray) -> np.ndarray:
    """Indices of a bin's points that are finite in every field."""
    flat = mask.ravel()
    ok = np.ones(len(flat), dtype=bool) & flat
    for f in fields:
        ok &= np.isfinite(f)
    return np.flatnonzero(ok)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def distance_corr_sq(x: np.ndarray, y: np.ndarray) -> float:
    """Bias-corrected squared distance correlation between two 1-D samples.

    Args:
        x: (N,) values of variable A.
        y: (N,) values of variable B.

    Returns:
        Squared distance correlation (≈ 0 under independence, 1 for exact
        linear dependence); 0 if either sample is constant; nan if N < 4.
    """
    if len(x) < 4:
        return float(np.nan)
    return _dcor_from_u(_u_centered(np.asarray(x, float)), _u_centered(np.asarray(y, float)))


def partial_distance_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Partial distance correlation of x and y given z (Székely & Rizzo 2014).

        pdCor(x, y; z) = (R_xy - R_xz R_yz) / sqrt((1 - R_xz^2)(1 - R_yz^2))

    where R are bias-corrected squared distance correlations.

    Args:
        x: (N,) values of variable A.
        y: (N,) values of variable B.
        z: (N,) control variable.

    Returns:
        Partial distance correlation in [-1, 1]; 0 if z fully determines x or
        y; equal to R*_xy if z is constant; nan if N < 4.
    """
    if len(x) < 4:
        return float(np.nan)
    ux, uy, uz = (_u_centered(np.asarray(v, float)) for v in (x, y, z))
    r_xy, r_xz, r_yz = _dcor_from_u(ux, uy), _dcor_from_u(ux, uz), _dcor_from_u(uy, uz)
    denom = (1.0 - r_xz**2) * (1.0 - r_yz**2)
    if denom <= 1e-12:
        return 0.0
    return float(np.clip((r_xy - r_xz * r_yz) / np.sqrt(denom), -1.0, 1.0))


def distance_corr_sq_sequence(
    x_full: np.ndarray,
    y_full: np.ndarray,
    quantile_masks: list[np.ndarray],
    *,
    min_points: int = DEFAULT_MIN_POINTS,
) -> list[float]:
    """Bias-corrected squared distance correlation within each spatial quantile bin.

    Args:
        x_full: (M,) values for variable A over the full grid.
        y_full: (M,) values for variable B over the full grid.
        quantile_masks: Boolean masks, one per bin (2-D or 1-D).
        min_points: Bins with fewer points give nan (at least 4).

    Returns:
        One value per bin. Grid points where x or y is non-finite are left out.
    """
    out = []
    for m in quantile_masks:
        idx = _finite_bin(m, x_full, y_full)
        if len(idx) < max(min_points, 4):
            out.append(float(np.nan))
        else:
            out.append(distance_corr_sq(x_full[idx], y_full[idx]))
    return out


def partial_distance_corr_sequence(
    x_full: np.ndarray,
    y_full: np.ndarray,
    z_full: np.ndarray,
    quantile_masks: list[np.ndarray],
    *,
    min_points: int = DEFAULT_MIN_POINTS,
) -> list[float]:
    """Partial distance correlation (x, y given z) within each spatial quantile bin.

    Args:
        x_full: (M,) values for variable A over the full grid.
        y_full: (M,) values for variable B over the full grid.
        z_full: (M,) control variable over the full grid.
        quantile_masks: Boolean masks, one per bin (2-D or 1-D).
        min_points: Bins with fewer points give nan (at least 4).

    Returns:
        One value per bin. Grid points where x, y or z is non-finite are left out.
    """
    out = []
    for m in quantile_masks:
        idx = _finite_bin(m, x_full, y_full, z_full)
        if len(idx) < max(min_points, 4):
            out.append(float(np.nan))
        else:
            out.append(partial_distance_corr(x_full[idx], y_full[idx], z_full[idx]))
    return out
