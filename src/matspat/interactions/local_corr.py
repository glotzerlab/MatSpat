"""Geographically weighted (local) correlation and partial correlation.

Unlike Lee's L, which returns one number per region and mixes co-location with
each variable's own spatial autocorrelation, these statistics are *local*: at
every point i a weighted Pearson correlation is computed over the surrounding
points, with bisquare weights centred on i. The result is a map of
coefficients, which the ``*_sequence`` functions summarise per quantile bin.

- GWCor (Fotheringham, Brunsdon & Charlton 2002): local weighted Pearson r.
- GW partial correlation (Percival & Tsutsumida 2017): local partial correlation
  of x and y given z, built from the three local correlations.

Public functions:
    gw_corr(x, y, positions, bandwidth) -> ndarray
    gw_partial_corr(x, y, z, positions, bandwidth) -> ndarray
    gw_corr_sequence(x_full, y_full, grid_coords, bandwidth, quantile_masks) -> list[float]
    gw_partial_corr_sequence(x_full, y_full, z_full, grid_coords, bandwidth,
                             quantile_masks) -> list[float]
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.spatial import distance

from matspat.interactions.correlation import _bisquare_kernel, _finite_in_bin

# Relative tolerance below which a local variance is treated as zero
_VAR_RTOL = 1e-12

_SUMMARIES: dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda r: float(np.nanmean(r)),
    "median": lambda r: float(np.nanmedian(r)),
    "sd": lambda r: float(np.nanstd(r)),
}

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _kernel(positions: np.ndarray, bandwidth: float) -> np.ndarray:
    """Bisquare kernel matrix for a set of positions."""
    return _bisquare_kernel(distance.cdist(positions, positions, metric="euclidean"), bandwidth)


def _local_corr_from_kernel(a: np.ndarray, b: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Local weighted Pearson correlation of a and b at every point.

    Row i of W holds the weights of all points relative to point i, so

        r_i = cov_w(a, b) / sqrt(var_w(a) var_w(b)),  weights w_ij = W_ij.

    Values are centred on their global mean first to limit cancellation error.

    Returns:
        (N,) local correlations; nan where either local variance is ~0.
    """
    a = a - a.mean()
    b = b - b.mean()
    sw = W.sum(axis=1)
    ma, mb = (W @ a) / sw, (W @ b) / sw
    var_a = (W @ (a * a)) / sw - ma**2
    var_b = (W @ (b * b)) / sw - mb**2
    cov = (W @ (a * b)) / sw - ma * mb
    tol_a = _VAR_RTOL * max(float(np.mean(a * a)), float(np.finfo(float).tiny))
    tol_b = _VAR_RTOL * max(float(np.mean(b * b)), float(np.finfo(float).tiny))
    ok = (var_a > tol_a) & (var_b > tol_b)
    r = np.full(len(a), np.nan)
    r[ok] = cov[ok] / np.sqrt(var_a[ok] * var_b[ok])
    clipped: np.ndarray = np.clip(r, -1.0, 1.0)
    return clipped


def _local_partial_from_kernel(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, W: np.ndarray
) -> np.ndarray:
    """Local partial correlation of x and y given z at every point.

        r_xy.z = (r_xy - r_xz r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))

    Where z is locally constant, controlling for it changes nothing, so the
    result falls back to r_xy.

    Returns:
        (N,) local partial correlations; nan where z explains x or y entirely
        or the local variance of x or y is ~0.
    """
    r_xy = _local_corr_from_kernel(x, y, W)
    r_xz = _local_corr_from_kernel(x, z, W)
    r_yz = _local_corr_from_kernel(y, z, W)
    denom = (1.0 - r_xz**2) * (1.0 - r_yz**2)
    with np.errstate(invalid="ignore"):
        ok = denom > 1e-12
    out = np.full(len(x), np.nan)
    out[ok] = (r_xy[ok] - r_xz[ok] * r_yz[ok]) / np.sqrt(denom[ok])
    # nan r_xz / r_yz with finite r_xy means z has ~0 local variance
    z_const = np.isnan(r_xz) | np.isnan(r_yz)
    out[z_const] = r_xy[z_const]
    clipped: np.ndarray = np.clip(out, -1.0, 1.0)
    return clipped


def _effective_n(W: np.ndarray) -> np.ndarray:
    """Kish effective number of points in each local window, (sum w)^2 / sum w^2."""
    n_eff: np.ndarray = W.sum(axis=1) ** 2 / (W**2).sum(axis=1)
    return n_eff


def _summarise(values: np.ndarray, summary: str) -> float:
    if np.all(np.isnan(values)):
        return float(np.nan)
    return _SUMMARIES[summary](values)


def _check_summary(summary: str) -> None:
    if summary not in _SUMMARIES:
        raise ValueError(f"summary must be one of {sorted(_SUMMARIES)}, got {summary!r}")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def gw_corr(x: np.ndarray, y: np.ndarray, positions: np.ndarray, bandwidth: float) -> np.ndarray:
    """Geographically weighted correlation (GWCor) at every point.

    Args:
        x: (N,) values for variable A (e.g. KDE density of a cell type).
        y: (N,) values for variable B at the same positions.
        positions: (N, 2) spatial coordinates.
        bandwidth: Bisquare kernel bandwidth h.

    Returns:
        (N,) local correlations in [-1, 1]; nan where a local variance is ~0.
        All-nan if fewer than 3 points.
    """
    if len(x) < 3:
        nan_out: np.ndarray = np.full(len(x), np.nan)
        return nan_out
    return _local_corr_from_kernel(
        np.asarray(x, float), np.asarray(y, float), _kernel(positions, bandwidth)
    )


def gw_partial_corr(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, positions: np.ndarray, bandwidth: float
) -> np.ndarray:
    """Geographically weighted partial correlation of x and y given z, at every point.

    Args:
        x: (N,) values for variable A.
        y: (N,) values for variable B.
        z: (N,) control variable (e.g. local density or an order parameter).
        positions: (N, 2) spatial coordinates.
        bandwidth: Bisquare kernel bandwidth h.

    Returns:
        (N,) local partial correlations in [-1, 1]; nan where undefined.
        Where z is locally constant, equals the local correlation of x and y.
        All-nan if fewer than 3 points.
    """
    if len(x) < 3:
        nan_out: np.ndarray = np.full(len(x), np.nan)
        return nan_out
    return _local_partial_from_kernel(
        np.asarray(x, float),
        np.asarray(y, float),
        np.asarray(z, float),
        _kernel(positions, bandwidth),
    )


def gw_corr_sequence(
    x_full: np.ndarray,
    y_full: np.ndarray,
    grid_coords: np.ndarray,
    bandwidth: float,
    quantile_masks: list[np.ndarray],
    *,
    summary: str = "mean",
    bin_kernels: list[np.ndarray | None] | None = None,
    min_n_eff: float | None = None,
) -> list[float]:
    """Summarise GWCor within each spatial quantile bin.

    Local windows are restricted to the points of each bin, matching
    :func:`~matspat.interactions.correlation.lees_L_sequence`.

    Args:
        x_full: (M,) values for variable A over the full grid.
        y_full: (M,) values for variable B over the full grid.
        grid_coords: (M, 2) coordinates of all grid points.
        bandwidth: Bisquare kernel bandwidth h.
        quantile_masks: Boolean masks, one per bin (2-D or 1-D).
        summary: How to summarise the local map within a bin: "mean",
            "median" or "sd" (spread, i.e. within-bin heterogeneity).
        bin_kernels: Optional precomputed kernels from
            :func:`~matspat.interactions.correlation.precompute_bin_kernels`.
        min_n_eff: If given, local correlations whose window has a Kish
            effective sample size below this are set to nan before
            summarising. Windows with ~2 effective points always give ±1,
            e.g. at the edge of a thin bin. None keeps every window.

    Returns:
        One value per bin. Grid points where x or y is non-finite are left
        out; bins with fewer than 3 remaining points give nan.
    """
    _check_summary(summary)
    out = []
    for i, mask in enumerate(quantile_masks):
        W, idx = _bin_kernel(i, mask, grid_coords, bandwidth, bin_kernels, x_full, y_full)
        if W is None:
            out.append(float(np.nan))
            continue
        r = _local_corr_from_kernel(x_full[idx].astype(float), y_full[idx].astype(float), W)
        out.append(_summarise(_drop_thin_windows(r, W, min_n_eff), summary))
    return out


def gw_partial_corr_sequence(
    x_full: np.ndarray,
    y_full: np.ndarray,
    z_full: np.ndarray,
    grid_coords: np.ndarray,
    bandwidth: float,
    quantile_masks: list[np.ndarray],
    *,
    summary: str = "mean",
    bin_kernels: list[np.ndarray | None] | None = None,
    min_n_eff: float | None = None,
) -> list[float]:
    """Summarise GW partial correlation (x, y given z) within each spatial quantile bin.

    Args:
        x_full: (M,) values for variable A over the full grid.
        y_full: (M,) values for variable B over the full grid.
        z_full: (M,) control variable over the full grid (see
            :func:`~matspat.interactions.partial_corr.interpolate_to_grid`).
        grid_coords: (M, 2) coordinates of all grid points.
        bandwidth: Bisquare kernel bandwidth h.
        quantile_masks: Boolean masks, one per bin (2-D or 1-D).
        summary: "mean", "median" or "sd".
        bin_kernels: Optional precomputed kernels (see :func:`gw_corr_sequence`).
        min_n_eff: Minimum effective window size (see :func:`gw_corr_sequence`).

    Returns:
        One value per bin. Grid points where x, y or z is non-finite are left
        out; bins with fewer than 3 remaining points give nan. Where z is
        locally constant the local value is the GWCor of x and y.
    """
    _check_summary(summary)
    out = []
    for i, mask in enumerate(quantile_masks):
        W, idx = _bin_kernel(i, mask, grid_coords, bandwidth, bin_kernels, x_full, y_full, z_full)
        if W is None:
            out.append(float(np.nan))
            continue
        r = _local_partial_from_kernel(
            x_full[idx].astype(float), y_full[idx].astype(float), z_full[idx].astype(float), W
        )
        out.append(_summarise(_drop_thin_windows(r, W, min_n_eff), summary))
    return out


def _bin_kernel(
    i: int,
    mask: np.ndarray,
    grid_coords: np.ndarray,
    bandwidth: float,
    bin_kernels: list[np.ndarray | None] | None,
    *fields: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Kernel and grid indices for bin *i*, restricted to points finite in every field.

    Returns:
        ``(W, idx)``; W is None when fewer than 3 points remain or the
        precomputed kernel for the bin is None.
    """
    flat = mask.ravel()
    finite = _finite_in_bin(flat, *fields)
    idx = np.flatnonzero(flat)[finite]
    if len(idx) < 3:
        return None, idx
    if bin_kernels is None:
        return _kernel(grid_coords[idx], bandwidth), idx
    W = bin_kernels[i]
    if W is None:
        return None, idx
    return (W if finite.all() else W[np.ix_(finite, finite)]), idx


def _drop_thin_windows(r: np.ndarray, W: np.ndarray, min_n_eff: float | None) -> np.ndarray:
    """Set local values to nan where the window's effective size is below *min_n_eff*."""
    if min_n_eff is None:
        return r
    out: np.ndarray = np.where(_effective_n(W) < min_n_eff, np.nan, r)
    return out
