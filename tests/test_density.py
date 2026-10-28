"""Tests for matspat.density.kde quantile masks and tissue mask."""

import numpy as np
import pytest

from matspat.density import (
    build_grid,
    get_cumulative_quantile_masks,
    get_marginal_quantile_masks,
    tissue_mask,
)


@pytest.fixture
def density():
    rng = np.random.default_rng(0)
    return rng.random((10, 10))


@pytest.fixture
def valid():
    v = np.ones((10, 10), dtype=bool)
    v[:3, :] = False  # top 3 rows off-tissue
    return v


class TestTissueMask:
    def test_points_near_cells_are_valid(self):
        coords = np.array([[0.0, 0.0], [10.0, 10.0]])
        grid = np.array([[0.5, 0.0], [5.0, 5.0], [10.0, 9.0]])
        assert tissue_mask(coords, grid, 1.5).tolist() == [True, False, True]

    def test_shape_matches_grid(self):
        coords = np.random.default_rng(1).random((50, 2)) * 100
        _, _, grid = build_grid(coords, 20)
        assert tissue_mask(coords, grid, 10.0).shape == (400,)


class TestQuantileMasksValid:
    def test_marginal_default_unchanged(self, density):
        a = get_marginal_quantile_masks(density, 4)
        b = get_marginal_quantile_masks(density, 4, valid=np.ones_like(density, dtype=bool))
        for i in range(4):
            np.testing.assert_array_equal(a[i], b[i])

    def test_marginal_excludes_invalid_and_partitions_valid(self, density, valid):
        masks = get_marginal_quantile_masks(density, 4, valid=valid)
        stacked = np.stack([masks[i] for i in range(4)])
        assert not stacked[:, ~valid].any()
        np.testing.assert_array_equal(stacked.sum(axis=0)[valid], 1)
        counts = stacked.sum(axis=(1, 2))
        assert counts.max() - counts.min() <= 1

    def test_cumulative_excludes_invalid_and_last_is_all_valid(self, density, valid):
        masks = get_cumulative_quantile_masks(density, 4, valid=valid)
        for i in range(4):
            assert not masks[i][~valid].any()
        np.testing.assert_array_equal(masks[3], valid)
        for i in range(3):
            assert (masks[i] <= masks[i + 1]).all()

    def test_flat_valid_mask_accepted(self, density, valid):
        a = get_marginal_quantile_masks(density, 4, valid=valid)
        b = get_marginal_quantile_masks(density, 4, valid=valid.ravel())
        for i in range(4):
            np.testing.assert_array_equal(a[i], b[i])


class TestValidMaskChecks:
    def test_integer_valid_mask_treated_as_bool(self, density, valid):
        as_int = valid.astype(int)
        for fn in (get_marginal_quantile_masks, get_cumulative_quantile_masks):
            a, b = fn(density, 4, valid=valid), fn(density, 4, valid=as_int)
            for k in a:
                np.testing.assert_array_equal(a[k], b[k])

    def test_all_false_valid_raises(self, density):
        none = np.zeros_like(density, dtype=bool)
        for fn in (get_marginal_quantile_masks, get_cumulative_quantile_masks):
            with pytest.raises(ValueError, match="no True"):
                fn(density, 4, valid=none)
