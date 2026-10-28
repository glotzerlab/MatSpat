"""Tests for bias-corrected distance correlation and partial distance correlation."""

import numpy as np
import pytest

from matspat.interactions.distance_corr import (
    _u_centered,
    distance_corr_sq,
    distance_corr_sq_sequence,
    partial_distance_corr,
    partial_distance_corr_sequence,
)


@pytest.fixture
def rng():
    return np.random.default_rng(1)


class TestUCentering:
    def test_rows_sum_to_zero(self, rng):
        u = _u_centered(rng.normal(size=30))
        assert np.allclose(u.sum(axis=1), 0.0)
        assert np.allclose(np.diag(u), 0.0)


class TestDistanceCorr:
    def test_linear_dependence_is_one(self, rng):
        x = rng.normal(size=200)
        assert distance_corr_sq(x, 2 * x - 1) == pytest.approx(1.0)

    def test_detects_nonmonotonic_dependence(self, rng):
        x = rng.uniform(-1, 1, size=300)
        y = x**2
        assert abs(np.corrcoef(x, y)[0, 1]) < 0.15  # Pearson misses it
        # population squared dCor for y = x^2, x ~ U(-1, 1) is ~0.25
        assert distance_corr_sq(x, y) > 0.15

    def test_independent_near_zero(self, rng):
        x, y = rng.normal(size=(2, 500))
        assert abs(distance_corr_sq(x, y)) < 0.05

    def test_constant_gives_zero_and_small_gives_nan(self, rng):
        # Szekely & Rizzo (2014): R* = 0 when a sample is constant
        assert distance_corr_sq(rng.normal(size=20), np.ones(20)) == 0.0
        assert np.isnan(distance_corr_sq(np.arange(3.0), np.arange(3.0)))


class TestPartialDistanceCorr:
    def test_removes_shared_driver(self, rng):
        z = rng.normal(size=400)
        x = z + 0.3 * rng.normal(size=400)
        y = z + 0.3 * rng.normal(size=400)
        assert distance_corr_sq(x, y) > 0.6
        assert abs(partial_distance_corr(x, y, z)) < 0.15

    def test_irrelevant_control_keeps_dependence(self, rng):
        x = rng.normal(size=300)
        y = x + 0.2 * rng.normal(size=300)
        z = rng.normal(size=300)
        assert partial_distance_corr(x, y, z) == pytest.approx(distance_corr_sq(x, y), abs=0.05)

    def test_control_equal_to_x_is_zero(self, rng):
        # Szekely & Rizzo (2014): pdCor = 0 when z fully explains x
        x, y = rng.normal(size=(2, 50))
        assert partial_distance_corr(x, y, x) == 0.0

    def test_constant_control_gives_dcor(self, rng):
        x = rng.normal(size=50)
        y = x + 0.5 * rng.normal(size=50)
        assert partial_distance_corr(x, y, np.ones(50)) == pytest.approx(distance_corr_sq(x, y))

    def test_bounded(self, rng):
        x, y, z = rng.normal(size=(3, 80))
        assert -1 <= partial_distance_corr(x, y, z) <= 1


class TestSequences:
    def test_matches_per_bin_calls(self, rng):
        x, y, z = rng.normal(size=(3, 400))
        masks = [np.arange(400) < 200, np.arange(400) >= 200]
        assert distance_corr_sq_sequence(x, y, masks) == pytest.approx(
            [distance_corr_sq(x[m], y[m]) for m in masks]
        )
        assert partial_distance_corr_sequence(x, y, z, masks) == pytest.approx(
            [partial_distance_corr(x[m], y[m], z[m]) for m in masks]
        )

    def test_accepts_2d_masks_and_small_bin_nan(self, rng):
        x, y = rng.normal(size=(2, 400))
        m = np.zeros((20, 20), bool)
        m[:2, :1] = True
        full = np.ones((20, 20), bool)
        seq = distance_corr_sq_sequence(x, y, [m, full])
        assert np.isnan(seq[0]) and np.isfinite(seq[1])


class TestSequenceGuards:
    def test_small_bins_give_nan(self, rng):
        x, y, z = rng.normal(size=(3, 100))
        small = np.zeros(100, bool)
        small[:6] = True
        assert np.isnan(distance_corr_sq_sequence(x, y, [small])[0])
        assert np.isnan(partial_distance_corr_sequence(x, y, z, [small])[0])
        assert np.isfinite(distance_corr_sq_sequence(x, y, [small], min_points=4)[0])

    def test_nan_points_are_dropped(self, rng):
        x, y, z = rng.normal(size=(3, 100))
        full = np.ones(100, bool)
        z_nan = z.copy()
        z_nan[:10] = np.nan
        expected = partial_distance_corr(x[10:], y[10:], z[10:])
        assert partial_distance_corr_sequence(x, y, z_nan, [full])[0] == pytest.approx(expected)
        x_nan = x.copy()
        x_nan[:10] = np.nan
        expected = distance_corr_sq(x[10:], y[10:])
        assert distance_corr_sq_sequence(x_nan, y, [full])[0] == pytest.approx(expected)
