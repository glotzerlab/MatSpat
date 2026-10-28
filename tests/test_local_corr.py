"""Tests for geographically weighted (local) correlation and partial correlation."""

import numpy as np
import pandas as pd
import pytest

from matspat.descriptors import compute_voronoi_geometry
from matspat.interactions.correlation import _bisquare_kernel, precompute_bin_kernels
from matspat.interactions.local_corr import (
    gw_corr,
    gw_corr_sequence,
    gw_partial_corr,
    gw_partial_corr_sequence,
)
from matspat.interactions.partial_corr import interpolate_to_grid


@pytest.fixture
def grid():
    xs, ys = np.meshgrid(np.arange(20.0), np.arange(20.0))
    return np.column_stack([xs.ravel(), ys.ravel()])


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _weighted_pearson(a, b, w):
    return np.cov(a, b, aweights=w)[0, 1] / np.sqrt(np.cov(a, aweights=w) * np.cov(b, aweights=w))


class TestGWCor:
    def test_perfect_positive_and_negative(self, grid, rng):
        x = rng.normal(size=len(grid))
        assert np.allclose(gw_corr(x, 3 * x + 2, grid, 5.0), 1.0)
        assert np.allclose(gw_corr(x, -x, grid, 5.0), -1.0)

    def test_matches_direct_weighted_pearson(self, grid, rng):
        x, y = rng.normal(size=(2, len(grid)))
        r = gw_corr(x, y, grid, 4.0)
        d = np.linalg.norm(grid - grid[37], axis=1)
        w = _bisquare_kernel(d, 4.0)
        assert r[37] == pytest.approx(_weighted_pearson(x, y, w), rel=1e-8)

    def test_bounded_and_shape(self, grid, rng):
        x, y = rng.normal(size=(2, len(grid)))
        r = gw_corr(x, y, grid, 3.0)
        assert r.shape == (len(grid),)
        assert np.all((r >= -1) & (r <= 1))

    def test_local_sign_change(self, grid, rng):
        # y follows x on the left half, opposes it on the right half
        x = rng.normal(size=len(grid))
        y = np.where(grid[:, 0] < 10, x, -x)
        r = gw_corr(x, y, grid, 2.5)
        assert np.nanmean(r[grid[:, 0] < 6]) > 0.9
        assert np.nanmean(r[grid[:, 0] > 13]) < -0.9

    def test_constant_input_gives_nan(self, grid, rng):
        x = rng.normal(size=len(grid))
        assert np.all(np.isnan(gw_corr(x, np.ones(len(grid)), grid, 5.0)))

    def test_too_few_points(self):
        assert np.all(np.isnan(gw_corr(np.ones(2), np.ones(2), np.zeros((2, 2)), 1.0)))


class TestGWPartialCorr:
    def test_removes_shared_driver(self, grid, rng):
        z = rng.normal(size=len(grid))
        x = z + 0.3 * rng.normal(size=len(grid))
        y = z + 0.3 * rng.normal(size=len(grid))
        raw = np.nanmean(gw_corr(x, y, grid, 5.0))
        partial = np.nanmean(gw_partial_corr(x, y, z, grid, 5.0))
        assert raw > 0.8
        assert abs(partial) < 0.15

    def test_irrelevant_control_keeps_correlation(self, grid, rng):
        x = rng.normal(size=len(grid))
        y = x + 0.2 * rng.normal(size=len(grid))
        z = rng.normal(size=len(grid))
        raw = gw_corr(x, y, grid, 5.0)
        partial = gw_partial_corr(x, y, z, grid, 5.0)
        assert np.nanmean(np.abs(partial - raw)) < 0.05

    def test_control_equal_to_x_is_nan(self, grid, rng):
        x, y = rng.normal(size=(2, len(grid)))
        assert np.all(np.isnan(gw_partial_corr(x, y, x, grid, 5.0)))

    def test_bounded(self, grid, rng):
        x, y, z = rng.normal(size=(3, len(grid)))
        r = gw_partial_corr(x, y, z, grid, 3.0)
        assert np.all((r[~np.isnan(r)] >= -1) & (r[~np.isnan(r)] <= 1))


class TestSequences:
    @pytest.fixture
    def masks(self, grid):
        return [grid[:, 0] < 7, (grid[:, 0] >= 7) & (grid[:, 0] < 14), grid[:, 0] >= 14]

    def test_length_and_floats(self, grid, rng, masks):
        x, y, z = rng.normal(size=(3, len(grid)))
        for seq in (
            gw_corr_sequence(x, y, grid, 4.0, masks),
            gw_partial_corr_sequence(x, y, z, grid, 4.0, masks),
        ):
            assert len(seq) == 3
            assert all(isinstance(v, float) for v in seq)

    def test_accepts_2d_masks(self, grid, rng, masks):
        x, y = rng.normal(size=(2, len(grid)))
        m2d = [m.reshape(20, 20) for m in masks]
        assert gw_corr_sequence(x, y, grid, 4.0, m2d) == pytest.approx(
            gw_corr_sequence(x, y, grid, 4.0, masks)
        )

    def test_window_restricted_to_bin(self, grid, rng, masks):
        x, y = rng.normal(size=(2, len(grid)))
        seq = gw_corr_sequence(x, y, grid, 4.0, masks)
        m = masks[1]
        expected = np.nanmean(gw_corr(x[m], y[m], grid[m], 4.0))
        assert seq[1] == pytest.approx(expected)

    def test_precomputed_kernels_match(self, grid, rng, masks):
        x, y, z = rng.normal(size=(3, len(grid)))
        k = precompute_bin_kernels(grid, 4.0, masks)
        assert gw_corr_sequence(x, y, grid, 4.0, masks, bin_kernels=k) == pytest.approx(
            gw_corr_sequence(x, y, grid, 4.0, masks)
        )
        assert gw_partial_corr_sequence(x, y, z, grid, 4.0, masks, bin_kernels=k) == pytest.approx(
            gw_partial_corr_sequence(x, y, z, grid, 4.0, masks)
        )

    def test_summaries(self, grid, rng, masks):
        x, y = rng.normal(size=(2, len(grid)))
        r = gw_corr(x[masks[0]], y[masks[0]], grid[masks[0]], 4.0)
        for name, fn in [("mean", np.nanmean), ("median", np.nanmedian), ("sd", np.nanstd)]:
            seq = gw_corr_sequence(x, y, grid, 4.0, masks, summary=name)
            assert seq[0] == pytest.approx(fn(r))

    def test_invalid_summary_raises(self, grid, rng, masks):
        x, y = rng.normal(size=(2, len(grid)))
        with pytest.raises(ValueError, match="summary"):
            gw_corr_sequence(x, y, grid, 4.0, masks, summary="max")

    def test_small_bin_gives_nan(self, grid, rng):
        x, y = rng.normal(size=(2, len(grid)))
        tiny = np.zeros(len(grid), bool)
        tiny[:2] = True
        assert np.isnan(gw_corr_sequence(x, y, grid, 4.0, [tiny])[0])

    def test_constant_z_gives_gw_corr(self, grid, rng):
        # Controlling for a constant changes nothing
        x, y = rng.normal(size=(2, len(grid)))
        full = np.ones(len(grid), bool)
        z = np.ones(len(grid))
        assert gw_partial_corr_sequence(x, y, z, grid, 4.0, [full])[0] == pytest.approx(
            gw_corr_sequence(x, y, grid, 4.0, [full])[0]
        )
        np.testing.assert_allclose(gw_partial_corr(x, y, z, grid, 4.0), gw_corr(x, y, grid, 4.0))

    def test_nan_points_dropped_with_and_without_kernels(self, grid, rng):
        x, y, z = rng.normal(size=(3, len(grid)))
        z[::7] = np.nan
        full = np.ones(len(grid), bool)
        ok = np.isfinite(z)
        expected = float(np.mean(gw_partial_corr(x[ok], y[ok], z[ok], grid[ok], 4.0)))
        kernels = precompute_bin_kernels(grid, 4.0, [full])
        no_k = gw_partial_corr_sequence(x, y, z, grid, 4.0, [full])[0]
        with_k = gw_partial_corr_sequence(x, y, z, grid, 4.0, [full], bin_kernels=kernels)[0]
        assert no_k == pytest.approx(expected)
        assert with_k == pytest.approx(expected)

    def test_none_kernel_gives_nan(self, grid, rng):
        x, y = rng.normal(size=(2, len(grid)))
        full = np.ones(len(grid), bool)
        assert np.isnan(gw_corr_sequence(x, y, grid, 4.0, [full], bin_kernels=[None])[0])

    def test_min_n_eff_drops_thin_windows(self, grid, rng):
        x, y = rng.normal(size=(2, len(grid)))
        strip = np.zeros((20, 20), bool)
        strip[:, 0] = True  # one-pixel-wide bin: windows hold ~2 effective points
        strip = strip.ravel()
        assert np.isfinite(gw_corr_sequence(x, y, grid, 1.5, [strip])[0])
        assert np.isnan(gw_corr_sequence(x, y, grid, 1.5, [strip], min_n_eff=3)[0])
        full = np.ones(len(grid), bool)
        assert np.isfinite(gw_corr_sequence(x, y, grid, 4.0, [full], min_n_eff=3)[0])


def test_voronoi_to_grid_partial_corr_is_finite():
    """Boundary cells are nan; the documented pipeline must still give finite values."""
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 100, size=(400, 2))
    df = pd.DataFrame(
        {
            "cell_id": np.arange(400),
            "image_id": "img",
            "x": xy[:, 0],
            "y": xy[:, 1],
            "cell_type": "A",
        }
    )
    geo = compute_voronoi_geometry(df)
    assert geo["shape_index"].isna().any()
    gx, gy = np.meshgrid(np.linspace(0, 100, 15), np.linspace(0, 100, 15))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    z = interpolate_to_grid(xy, geo["shape_index"].to_numpy(), grid)
    assert np.isfinite(z).all()
    x, y = rng.normal(size=(2, len(grid)))
    seq = gw_partial_corr_sequence(x, y, z, grid, 20.0, [np.ones(len(grid), bool)])
    assert np.isfinite(seq[0])
