import numpy as np
import pytest

from matspat.interactions.correlation import lees_L, lees_L_sequence
from matspat.interactions.partial_corr import (
    _bisquare_kernel,
    _weighted_ols_residuals,
    compute_gwpcor,
    gwpcor_sequence,
    interpolate_to_grid,
    partial_lees_L,
    partial_lees_L_sequence,
)

rng = np.random.default_rng(42)

# Shared 5x5 grid
x_array = np.arange(5)
y_array = np.arange(5)
x_grid, y_grid = np.meshgrid(x_array, y_array)
positions = np.column_stack((x_grid.ravel(), y_grid.ravel()))
N = len(positions)
bandwidth = 2.0


class TestBisquareKernel:
    def test_self_distance_is_one(self):
        """Diagonal entries (distance 0) should have weight 1."""
        dists = np.zeros((3, 3))
        W = _bisquare_kernel(dists, bandwidth=1.0)
        np.testing.assert_array_equal(W, np.ones((3, 3)))

    def test_beyond_bandwidth_is_zero(self):
        """Distances beyond bandwidth should have zero weight."""
        dists = np.array([[0, 5], [5, 0]])
        W = _bisquare_kernel(dists, bandwidth=2.0)
        assert W[0, 1] == 0.0
        assert W[1, 0] == 0.0

    def test_symmetry(self):
        """Kernel matrix should be symmetric."""
        from scipy.spatial import distance

        dists = distance.cdist(positions, positions)
        W = _bisquare_kernel(dists, bandwidth)
        np.testing.assert_array_equal(W, W.T)


class TestWeightedOlsResiduals:
    def test_perfect_linear_gives_zero_residuals(self):
        """If v = a + b*z exactly, residuals should be ~0."""
        z = np.linspace(0, 1, 20)
        v = 3.0 + 2.0 * z
        w = np.ones(20)
        resid = _weighted_ols_residuals(v, z, w)
        np.testing.assert_allclose(resid, 0.0, atol=1e-12)

    def test_residuals_orthogonal_to_z(self):
        """Residuals should be uncorrelated with z (weighted)."""
        z = rng.standard_normal(50)
        v = rng.standard_normal(50)
        w = rng.uniform(0.5, 2.0, size=50)
        resid = _weighted_ols_residuals(v, z, w)
        # Weighted correlation of residuals with z should be ~0
        cov = np.sum(w * resid * z) / np.sum(w)
        assert abs(cov) < 1e-10


class TestInterpolateToGrid:
    """Tests for interpolate_to_grid — bridges cell-level descriptors to grid."""

    def test_exact_at_cell_positions(self):
        """Interpolation at the cell positions themselves should recover values."""
        cell_pos = positions.astype(float)
        cell_vals = np.sin(cell_pos[:, 0]) + np.cos(cell_pos[:, 1])
        # Evaluate at the same positions
        result = interpolate_to_grid(cell_pos, cell_vals, cell_pos)
        np.testing.assert_allclose(result, cell_vals, atol=1e-10)

    def test_output_shape(self):
        """Output should have same length as grid_coords."""
        cell_pos = rng.uniform(0, 4, size=(30, 2))
        cell_vals = rng.random(30)
        grid = positions.astype(float)
        result = interpolate_to_grid(cell_pos, cell_vals, grid)
        assert result.shape == (len(grid),)

    def test_no_nans_with_fallback(self):
        """Linear interpolation with nearest-neighbor fallback should have no NaNs."""
        # Cell positions in a smaller region than the grid
        cell_pos = rng.uniform(1, 3, size=(20, 2))
        cell_vals = rng.random(20)
        grid = positions.astype(float)  # extends 0-4, outside cell hull
        result = interpolate_to_grid(cell_pos, cell_vals, grid, method="linear")
        assert not np.any(np.isnan(result))

    def test_nearest_method(self):
        """Nearest-neighbor interpolation should work and have no NaNs."""
        cell_pos = rng.uniform(0, 4, size=(15, 2))
        cell_vals = rng.random(15)
        grid = positions.astype(float)
        result = interpolate_to_grid(cell_pos, cell_vals, grid, method="nearest")
        assert not np.any(np.isnan(result))

    def test_cubic_method(self):
        """Cubic interpolation should work."""
        cell_pos = rng.uniform(0, 4, size=(30, 2))
        cell_vals = rng.random(30)
        grid = positions.astype(float)
        result = interpolate_to_grid(cell_pos, cell_vals, grid, method="cubic")
        assert not np.any(np.isnan(result))

    def test_invalid_method_raises(self):
        """Invalid method should raise ValueError."""
        cell_pos = np.array([[0, 0], [1, 1]])
        cell_vals = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="method must be one of"):
            interpolate_to_grid(cell_pos, cell_vals, positions.astype(float), method="spline")

    def test_smooth_field_interpolates_well(self):
        """A smooth spatial field should interpolate accurately."""
        # Dense cell positions
        cell_pos = rng.uniform(0.5, 3.5, size=(100, 2))
        # Smooth function: z = x + y
        cell_vals = cell_pos[:, 0] + cell_pos[:, 1]
        # Grid points inside the cell hull
        inner_grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        expected = inner_grid[:, 0] + inner_grid[:, 1]
        result = interpolate_to_grid(cell_pos, cell_vals, inner_grid)
        np.testing.assert_allclose(result, expected, atol=0.3)


class TestComputeGwpcor:
    def test_too_few_points(self):
        """Fewer than 3 points should return nan."""
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        z = np.array([5.0, 6.0])
        pos = np.array([[0, 0], [1, 0]])
        assert np.isnan(compute_gwpcor(x, y, z, pos, bandwidth=1.0))

    def test_zero_variance_covariate(self):
        """Constant z should return nan."""
        x = rng.random(N)
        y = rng.random(N)
        z = np.ones(N)
        assert np.isnan(compute_gwpcor(x, y, z, positions, bandwidth))

    def test_returns_scalar(self):
        """Result should be a float."""
        x = rng.random(N)
        y = rng.random(N)
        z = rng.random(N)
        result = compute_gwpcor(x, y, z, positions, bandwidth)
        assert isinstance(result, float)

    def test_bounded_minus_one_to_one(self):
        """GWPCor should be in [-1, 1]."""
        x = rng.random(N)
        y = rng.random(N)
        z = rng.random(N)
        result = compute_gwpcor(x, y, z, positions, bandwidth)
        assert -1.0 <= result <= 1.0 or np.isnan(result)

    def test_uncorrelated_z_matches_lees_L(self):
        """When z is independent noise uncorrelated with x and y,
        regressing it out should barely change the Lee's L value."""
        x = np.sin(positions[:, 0])
        y = np.sin(positions[:, 0])
        z = rng.standard_normal(N) * 0.001  # tiny noise, nearly zero effect
        gwp = compute_gwpcor(x, y, z, positions, bandwidth)
        raw_L = lees_L(x, y, positions, bandwidth)
        # Should be close when z explains almost nothing
        assert abs(gwp - raw_L) < 0.15

    def test_controlling_for_confound(self):
        """When x and y are both driven by z, controlling for z
        should reduce the correlation magnitude."""
        z = rng.standard_normal(N)
        x = z + rng.standard_normal(N) * 0.1
        y = z + rng.standard_normal(N) * 0.1
        raw_L = lees_L(x, y, positions, bandwidth)
        gwp = compute_gwpcor(x, y, z, positions, bandwidth)
        # Partial correlation should be smaller in magnitude
        assert abs(gwp) < abs(raw_L)


class TestGwpcorSequence:
    """Tests for gwpcor_sequence — mirrors lees_L_sequence with a covariate."""

    def _make_masks_1d(self, n_bins):
        """Split the 25-point grid into n_bins roughly equal 1D masks."""
        indices = np.arange(N)
        splits = np.array_split(indices, n_bins)
        masks = []
        for s in splits:
            m = np.zeros(N, dtype=bool)
            m[s] = True
            masks.append(m)
        return masks

    def _make_masks_2d(self, n_bins):
        """Split a 5x5 grid into n_bins 2D masks (row-wise)."""
        masks = []
        rows_per_bin = 5 // n_bins
        for i in range(n_bins):
            m = np.zeros((5, 5), dtype=bool)
            start = i * rows_per_bin
            end = start + rows_per_bin if i < n_bins - 1 else 5
            m[start:end, :] = True
            masks.append(m)
        return masks

    def test_returns_correct_length(self):
        """Output list length should match number of masks."""
        x = rng.random(N)
        y = rng.random(N)
        z = rng.random(N)
        masks = self._make_masks_1d(3)
        result = gwpcor_sequence(x, y, z, positions, bandwidth, masks)
        assert len(result) == 3

    def test_all_floats(self):
        """Every entry should be a float."""
        x = rng.random(N)
        y = rng.random(N)
        z = rng.random(N)
        masks = self._make_masks_1d(3)
        result = gwpcor_sequence(x, y, z, positions, bandwidth, masks)
        for val in result:
            assert isinstance(val, float)

    def test_accepts_2d_masks(self):
        """Should handle 2D masks the same as 1D."""
        x = rng.random(N)
        y = rng.random(N)
        z = rng.random(N)
        masks_2d = self._make_masks_2d(5)
        result = gwpcor_sequence(x, y, z, positions, bandwidth, masks_2d)
        assert len(result) == 5

    def test_small_bin_returns_nan(self):
        """A bin with fewer than 3 points should return nan."""
        x = rng.random(N)
        y = rng.random(N)
        z = rng.random(N)
        tiny_mask = np.zeros(N, dtype=bool)
        tiny_mask[0:2] = True  # only 2 points
        result = gwpcor_sequence(x, y, z, positions, bandwidth, [tiny_mask])
        assert np.isnan(result[0])

    def test_constant_z_bin_returns_nan(self):
        """A bin where z is constant should return nan."""
        x = rng.random(N)
        y = rng.random(N)
        z = np.ones(N)  # constant everywhere
        masks = self._make_masks_1d(3)
        result = gwpcor_sequence(x, y, z, positions, bandwidth, masks)
        assert all(np.isnan(v) for v in result)

    def test_zero_covariate_matches_lees_L_sequence(self):
        """With near-zero z, gwpcor_sequence should approximate lees_L_sequence."""
        x = np.sin(positions[:, 0]) + 0.5
        y = np.cos(positions[:, 1]) + 0.5
        z = rng.standard_normal(N) * 1e-6
        masks = self._make_masks_1d(2)
        gwp_vals = gwpcor_sequence(x, y, z, positions, bandwidth, masks)
        raw_vals = lees_L_sequence(x, y, positions, bandwidth, masks)
        for gwp, raw in zip(gwp_vals, raw_vals):
            if np.isnan(gwp) or np.isnan(raw):
                continue
            assert abs(gwp - raw) < 0.15

    def test_confound_reduces_correlation_per_bin(self):
        """Controlling for a confound should reduce correlation in each bin."""
        z = rng.standard_normal(N)
        x = z + rng.standard_normal(N) * 0.1
        y = z + rng.standard_normal(N) * 0.1
        masks = self._make_masks_1d(2)
        gwp_vals = gwpcor_sequence(x, y, z, positions, bandwidth, masks)
        raw_vals = lees_L_sequence(x, y, positions, bandwidth, masks)
        for gwp, raw in zip(gwp_vals, raw_vals):
            if np.isnan(gwp) or np.isnan(raw):
                continue
            assert abs(gwp) <= abs(raw) + 1e-10


def test_legacy_gwpcor_names_are_aliases():
    assert compute_gwpcor is partial_lees_L
    assert gwpcor_sequence is partial_lees_L_sequence


class TestInterpolateNonFinite:
    def test_nan_cells_dropped(self):
        rng = np.random.default_rng(0)
        pos = rng.uniform(0, 10, size=(60, 2))
        vals = pos[:, 0].copy()
        vals[:10] = np.nan
        grid = np.column_stack([np.linspace(1, 9, 20), np.linspace(1, 9, 20)])
        z = interpolate_to_grid(pos, vals, grid)
        assert np.isfinite(z).all()
        np.testing.assert_allclose(z, interpolate_to_grid(pos[10:], vals[10:], grid))

    def test_all_nan_raises(self):
        pos = np.random.default_rng(0).uniform(size=(10, 2))
        with pytest.raises(ValueError, match="finite"):
            interpolate_to_grid(pos, np.full(10, np.nan), pos)
