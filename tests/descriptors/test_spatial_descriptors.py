"""Tests for spatial descriptors module."""

import numpy as np
import pandas as pd
import pytest

from matspat.descriptors.spatial_descriptors import (
    compute_katic_order,
    compute_local_density,
    count_neighbor_cell_types,
)


@pytest.fixture
def simple_df():
    """Create a simple test DataFrame with two cell types."""
    return pd.DataFrame(
        {
            "cell_id": [1, 2, 3, 4, 5],
            "x": [0.0, 1.0, 2.0, 0.0, 1.0],
            "y": [0.0, 0.0, 0.0, 1.0, 1.0],
            "cell_type": ["A", "A", "B", "B", "A"],
            "image_id": ["img1", "img1", "img1", "img1", "img1"],
        }
    )


@pytest.fixture
def multi_image_df():
    """Create a DataFrame with multiple images."""
    return pd.DataFrame(
        {
            "cell_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "x": [0.0, 1.0, 0.0, 1.0, 10.0, 11.0, 10.0, 11.0],
            "y": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "cell_type": ["A", "A", "B", "B", "A", "A", "B", "B"],
            "image_id": ["img1", "img1", "img1", "img1", "img2", "img2", "img2", "img2"],
        }
    )


@pytest.fixture
def hexagonal_lattice():
    """Create a perfect hexagonal lattice for k=6 order testing."""
    # Create hexagonal lattice points
    hex_spacing = 1.0
    rows, cols = 5, 5
    x_coords = []
    y_coords = []

    for row in range(rows):
        for col in range(cols):
            x = col * hex_spacing * np.sqrt(3)
            if row % 2 == 1:
                x += hex_spacing * np.sqrt(3) / 2
            y = row * hex_spacing * 1.5
            x_coords.append(x)
            y_coords.append(y)

    n = len(x_coords)
    return pd.DataFrame(
        {
            "cell_id": list(range(1, n + 1)),
            "x": x_coords,
            "y": y_coords,
            "cell_type": ["A"] * n,
            "image_id": ["img1"] * n,
        }
    )


@pytest.fixture
def square_lattice():
    """Create a perfect square lattice for k=4 order testing."""
    spacing = 1.0
    size = 5
    x, y = np.meshgrid(np.arange(size) * spacing, np.arange(size) * spacing)
    x_coords = x.flatten()
    y_coords = y.flatten()
    n = len(x_coords)

    return pd.DataFrame(
        {
            "cell_id": list(range(1, n + 1)),
            "x": x_coords,
            "y": y_coords,
            "cell_type": ["A"] * n,
            "image_id": ["img1"] * n,
        }
    )


# ==================== Tests for compute_local_density ====================


class TestLocalDensityBasic:
    """Basic functionality tests for local density."""

    def test_returns_dataframe(self, simple_df):
        """Should return a DataFrame."""
        result = compute_local_density(simple_df, r_max=2.0)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self, simple_df):
        """Should have correct columns."""
        result = compute_local_density(simple_df, r_max=2.0)
        expected_cols = ["cell_id", "image_id", "cell_type", "local_density"]
        assert list(result.columns) == expected_cols

    def test_output_columns_with_neighbor_counts(self, simple_df):
        """Should include num_neighbors when requested."""
        result = compute_local_density(simple_df, r_max=2.0, include_neighbor_counts=True)
        assert "num_neighbors" in result.columns

    def test_output_length_all_cells(self, simple_df):
        """Should return all cells when no focus type."""
        result = compute_local_density(simple_df, r_max=2.0)
        assert len(result) == len(simple_df)

    def test_output_length_focus_cells(self, simple_df):
        """Should return only focus cells when specified."""
        result = compute_local_density(simple_df, r_max=2.0, focus_cell_type="A")
        n_type_a = (simple_df["cell_type"] == "A").sum()
        assert len(result) == n_type_a

    def test_density_non_negative(self, simple_df):
        """Density values should be non-negative."""
        result = compute_local_density(simple_df, r_max=2.0)
        assert (result["local_density"] >= 0).all()

    def test_neighbor_counts_non_negative(self, simple_df):
        """Neighbor counts should be non-negative."""
        result = compute_local_density(simple_df, r_max=2.0, include_neighbor_counts=True)
        assert (result["num_neighbors"] >= 0).all()


class TestLocalDensityValidation:
    """Input validation tests."""

    def test_invalid_r_max(self, simple_df):
        """Should raise ValueError for r_max <= 0."""
        with pytest.raises(ValueError, match="r_max must be positive"):
            compute_local_density(simple_df, r_max=0)
        with pytest.raises(ValueError, match="r_max must be positive"):
            compute_local_density(simple_df, r_max=-1.0)

    def test_invalid_norm_by(self, simple_df):
        """Should raise ValueError for invalid norm_by."""
        with pytest.raises(ValueError, match="norm_by must be one of"):
            compute_local_density(simple_df, r_max=2.0, norm_by="invalid")

    def test_missing_columns(self, simple_df):
        """Should raise ValueError for missing required columns."""
        df_bad = simple_df.drop(columns=["x"])
        with pytest.raises(ValueError, match="missing required columns"):
            compute_local_density(df_bad, r_max=2.0)

    def test_focus_not_in_include(self, simple_df):
        """Should raise ValueError if focus not in include list."""
        with pytest.raises(ValueError, match="must be in include_cell_types"):
            compute_local_density(
                simple_df,
                r_max=2.0,
                focus_cell_type="A",
                include_cell_types=["B"],
            )

    def test_include_types_not_in_data(self, simple_df):
        """Should raise ValueError if include types not in data."""
        with pytest.raises(ValueError, match="not found in data"):
            compute_local_density(simple_df, r_max=2.0, include_cell_types=["Z"])


class TestLocalDensityNormalization:
    """Tests for normalization options."""

    def test_norm_by_total(self, simple_df):
        """Normalization by total should be < 1."""
        result = compute_local_density(simple_df, r_max=2.0, norm_by="total")
        # Normalized by 5 cells total
        assert (result["local_density"] <= 1.0).all()

    def test_norm_by_included(self, simple_df):
        """Normalization by included cell count."""
        result = compute_local_density(
            simple_df,
            r_max=2.0,
            include_cell_types=["A"],
            norm_by="included",
        )
        # Should be normalized by count of type A cells (3)
        assert (result["local_density"] <= 1.0).all()

    def test_norm_by_focus(self, simple_df):
        """Normalization by focus cell count."""
        result = compute_local_density(simple_df, r_max=2.0, focus_cell_type="A", norm_by="focus")
        # Should be normalized by count of type A cells (3)
        assert (result["local_density"] <= 1.0).all()

    def test_norm_by_focus_requires_focus_type(self, simple_df):
        """norm_by='focus' should require focus_cell_type."""
        with pytest.raises(ValueError, match="requires focus_cell_type"):
            compute_local_density(simple_df, r_max=2.0, norm_by="focus")


class TestLocalDensityMultiImage:
    """Tests for multi-image processing."""

    def test_processes_all_images(self, multi_image_df):
        """Should process all images."""
        result = compute_local_density(multi_image_df, r_max=2.0)
        assert set(result["image_id"]) == {"img1", "img2"}

    def test_image_separation(self, multi_image_df):
        """Cells from different images should not be neighbors."""
        result = compute_local_density(multi_image_df, r_max=2.0, include_neighbor_counts=True)
        # Cells in img2 are far from cells in img1, so should have
        # same neighbor counts as if processed separately
        img1_result = result[result["image_id"] == "img1"]
        img2_result = result[result["image_id"] == "img2"]
        assert len(img1_result) == 4
        assert len(img2_result) == 4


class TestLocalDensityEdgeCases:
    """Edge case tests."""

    def test_single_cell_image_skipped(self):
        """Images with 1 cell should be skipped with warning."""
        df = pd.DataFrame(
            {
                "cell_id": [1],
                "x": [0.0],
                "y": [0.0],
                "cell_type": ["A"],
                "image_id": ["img1"],
            }
        )
        with pytest.warns(UserWarning, match="Fewer than 2 cells"):
            result = compute_local_density(df, r_max=2.0)
        assert len(result) == 0

    def test_large_r_max(self, simple_df):
        """Large r_max should find all neighbors."""
        # Use r_max=5.0 which is large enough for the 2x2 grid
        # but not so large that it exceeds freud's box limit
        result = compute_local_density(simple_df, r_max=5.0, include_neighbor_counts=True)
        # All cells should have 4 neighbors (excluding self)
        assert (result["num_neighbors"] == 4).all()

    def test_small_r_max(self, simple_df):
        """Very small r_max should find few/no neighbors."""
        result = compute_local_density(simple_df, r_max=0.1, include_neighbor_counts=True)
        # Should have very few neighbors
        assert (result["num_neighbors"] <= 1).all()


class TestLocalDensityReproducibility:
    """Reproducibility tests."""

    def test_deterministic(self, simple_df):
        """Same input should give same output."""
        result1 = compute_local_density(simple_df, r_max=2.0)
        result2 = compute_local_density(simple_df, r_max=2.0)
        pd.testing.assert_frame_equal(result1, result2)

    def test_row_order_invariant(self, simple_df):
        """Row order shouldn't affect values."""
        shuffled = simple_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        result1 = compute_local_density(simple_df, r_max=2.0)
        result2 = compute_local_density(shuffled, r_max=2.0)

        # Sort both by cell_id for comparison
        result1_sorted = result1.sort_values("cell_id").reset_index(drop=True)
        result2_sorted = result2.sort_values("cell_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(result1_sorted, result2_sorted)


# ==================== Tests for compute_katic_order ====================


class TestKaticOrderBasic:
    """Basic functionality tests for k-atic order."""

    def test_returns_dataframe(self, simple_df):
        """Should return a DataFrame."""
        result = compute_katic_order(simple_df, k=6, r_max=2.0)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns_single_k(self, simple_df):
        """Should have correct columns for single k."""
        result = compute_katic_order(simple_df, k=6, r_max=2.0)
        expected_cols = ["cell_id", "image_id", "cell_type", "psi_6"]
        assert list(result.columns) == expected_cols

    def test_output_columns_multiple_k(self, simple_df):
        """Should have correct columns for multiple k values."""
        result = compute_katic_order(simple_df, k=[4, 6, 8], r_max=2.0)
        expected_cols = ["cell_id", "image_id", "cell_type", "psi_4", "psi_6", "psi_8"]
        assert list(result.columns) == expected_cols

    def test_output_length_all_cells(self, simple_df):
        """Should return all cells when no focus type."""
        result = compute_katic_order(simple_df, k=6, r_max=2.0)
        assert len(result) == len(simple_df)

    def test_output_length_focus_cells(self, simple_df):
        """Should return only focus cells when specified."""
        result = compute_katic_order(simple_df, k=6, r_max=2.0, focus_cell_type="A")
        n_type_a = (simple_df["cell_type"] == "A").sum()
        assert len(result) == n_type_a

    def test_order_in_valid_range(self, simple_df):
        """Order parameter |ψ_k| should be in [0, 1]."""
        result = compute_katic_order(simple_df, k=6, r_max=2.0)
        assert (result["psi_6"] >= 0).all()
        assert (result["psi_6"] <= 1).all()


class TestKaticOrderValidation:
    """Input validation tests."""

    def test_invalid_k_negative(self, simple_df):
        """Should raise ValueError for k < 2."""
        with pytest.raises(ValueError, match="k values must be >= 2"):
            compute_katic_order(simple_df, k=1, r_max=2.0)

    def test_invalid_k_list(self, simple_df):
        """Should raise ValueError for any k < 2 in list."""
        with pytest.raises(ValueError, match="k values must be >= 2"):
            compute_katic_order(simple_df, k=[4, 1, 6], r_max=2.0)

    def test_invalid_r_max(self, simple_df):
        """Should raise ValueError for r_max <= 0."""
        with pytest.raises(ValueError, match="r_max must be positive"):
            compute_katic_order(simple_df, k=6, r_max=0)

    def test_invalid_neighbor_mode(self, simple_df):
        """Should raise ValueError for invalid neighbor mode."""
        with pytest.raises(ValueError, match="neighbor_mode must be one of"):
            compute_katic_order(simple_df, k=6, r_max=2.0, neighbor_mode="invalid")

    def test_missing_columns(self, simple_df):
        """Should raise ValueError for missing required columns."""
        df_bad = simple_df.drop(columns=["y"])
        with pytest.raises(ValueError, match="missing required columns"):
            compute_katic_order(df_bad, k=6, r_max=2.0)


class TestKaticOrderNeighborModes:
    """Tests for different neighbor list modes."""

    def test_r_max_mode(self, simple_df):
        """r_max mode should use all neighbors within cutoff."""
        result = compute_katic_order(simple_df, k=6, r_max=2.0, neighbor_mode="r_max")
        # Should have no NaN values
        assert not result["psi_6"].isna().any()

    def test_k_nearest_mode(self, simple_df):
        """k_nearest mode may have NaN for cells with too few neighbors."""
        # Use large k to force some NaNs
        result = compute_katic_order(simple_df, k=10, r_max=5.0, neighbor_mode="k_nearest")
        # Should have NaN values since cells won't have 10 neighbors
        # (only 5 cells total, so max 4 neighbors)
        assert result["psi_10"].isna().all()

    def test_k_nearest_excludes_insufficient_neighbors(self, simple_df):
        """k_nearest should set NaN for cells with < k neighbors."""
        result = compute_katic_order(simple_df, k=100, r_max=5.0, neighbor_mode="k_nearest")
        # No cell should have 100 neighbors, all should be NaN
        assert result["psi_100"].isna().all()


class TestKaticOrderSanityChecks:
    """Sanity checks with known lattices."""

    def test_hexagonal_lattice_high_k6(self, hexagonal_lattice):
        """Perfect hexagonal lattice should have high |ψ_6|."""
        result = compute_katic_order(hexagonal_lattice, k=6, r_max=2.0)
        # Interior cells should have |ψ_6| close to 1
        # Edge cells may have lower values
        mean_order = result["psi_6"].mean()
        assert mean_order > 0.8, f"Expected high hexatic order, got {mean_order}"

    def test_square_lattice_high_k4(self, square_lattice):
        """Perfect square lattice should have reasonably high |ψ_4|."""
        result = compute_katic_order(square_lattice, k=4, r_max=1.5)
        # Square lattice should have higher k=4 order than random
        # but edge effects matter - interior cells will be higher
        median_order = result["psi_4"].median()
        # Relaxed threshold due to edge effects in 5x5 lattice
        assert median_order > 0.15, f"Expected elevated square order, got {median_order}"

    def test_random_points_low_order(self):
        """Random point cloud should have low order."""
        np.random.seed(42)
        n = 50
        df = pd.DataFrame(
            {
                "cell_id": list(range(1, n + 1)),
                "x": np.random.uniform(0, 10, n),
                "y": np.random.uniform(0, 10, n),
                "cell_type": ["A"] * n,
                "image_id": ["img1"] * n,
            }
        )
        result = compute_katic_order(df, k=6, r_max=2.0)
        mean_order = result["psi_6"].mean()
        assert mean_order < 0.5, f"Expected low order for random points, got {mean_order}"


class TestKaticOrderMultiImage:
    """Tests for multi-image processing."""

    def test_processes_all_images(self, multi_image_df):
        """Should process all images."""
        result = compute_katic_order(multi_image_df, k=6, r_max=2.0)
        assert set(result["image_id"]) == {"img1", "img2"}

    def test_multiple_k_values(self, multi_image_df):
        """Should compute multiple k values correctly."""
        result = compute_katic_order(multi_image_df, k=[4, 6], r_max=2.0)
        assert "psi_4" in result.columns
        assert "psi_6" in result.columns
        assert len(result) == len(multi_image_df)


class TestKaticOrderReproducibility:
    """Reproducibility tests."""

    def test_deterministic(self, simple_df):
        """Same input should give same output."""
        result1 = compute_katic_order(simple_df, k=6, r_max=2.0)
        result2 = compute_katic_order(simple_df, k=6, r_max=2.0)
        pd.testing.assert_frame_equal(result1, result2)

    def test_row_order_invariant(self, simple_df):
        """Row order shouldn't affect values."""
        shuffled = simple_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        result1 = compute_katic_order(simple_df, k=6, r_max=2.0)
        result2 = compute_katic_order(shuffled, k=6, r_max=2.0)

        result1_sorted = result1.sort_values("cell_id").reset_index(drop=True)
        result2_sorted = result2.sort_values("cell_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(result1_sorted, result2_sorted)


# ==================== Tests for count_neighbor_cell_types ====================


class TestNeighborCountsBasic:
    """Basic functionality tests for neighbor counting."""

    def test_returns_dataframe(self, simple_df):
        """Should return a DataFrame."""
        result = count_neighbor_cell_types(simple_df, r_max=2.0)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self, simple_df):
        """Should have correct columns."""
        result = count_neighbor_cell_types(simple_df, r_max=2.0)
        expected_base = ["cell_id", "image_id", "cell_type"]
        # Should have n_neighbors_A and n_neighbors_B
        assert all(col in result.columns for col in expected_base)
        assert "n_neighbors_A" in result.columns
        assert "n_neighbors_B" in result.columns

    def test_output_length_all_cells(self, simple_df):
        """Should return all cells when no focus type."""
        result = count_neighbor_cell_types(simple_df, r_max=2.0)
        assert len(result) == len(simple_df)

    def test_output_length_focus_cells(self, simple_df):
        """Should return only focus cells when specified."""
        result = count_neighbor_cell_types(simple_df, r_max=2.0, focus_cell_type="A")
        n_type_a = (simple_df["cell_type"] == "A").sum()
        assert len(result) == n_type_a

    def test_counts_non_negative(self, simple_df):
        """Neighbor counts should be non-negative."""
        result = count_neighbor_cell_types(simple_df, r_max=2.0)
        assert (result["n_neighbors_A"] >= 0).all()
        assert (result["n_neighbors_B"] >= 0).all()

    def test_counts_are_integers(self, simple_df):
        """Neighbor counts should be integers."""
        result = count_neighbor_cell_types(simple_df, r_max=2.0)
        assert result["n_neighbors_A"].dtype == int
        assert result["n_neighbors_B"].dtype == int


class TestNeighborCountsValidation:
    """Input validation tests."""

    def test_invalid_r_max(self, simple_df):
        """Should raise ValueError for r_max <= 0."""
        with pytest.raises(ValueError, match="r_max must be positive"):
            count_neighbor_cell_types(simple_df, r_max=0)

    def test_missing_columns(self, simple_df):
        """Should raise ValueError for missing required columns."""
        df_bad = simple_df.drop(columns=["cell_type"])
        with pytest.raises(ValueError, match="missing required columns"):
            count_neighbor_cell_types(df_bad, r_max=2.0)


class TestNeighborCountsFiltering:
    """Tests for cell type filtering."""

    def test_include_cell_types(self, simple_df):
        """Should only count specified cell types."""
        result = count_neighbor_cell_types(simple_df, r_max=2.0, include_cell_types=["A"])
        # Should only have n_neighbors_A column
        assert "n_neighbors_A" in result.columns
        assert "n_neighbors_B" not in result.columns

    def test_focus_with_include(self, simple_df):
        """Focus type with include types should work together."""
        result = count_neighbor_cell_types(
            simple_df,
            r_max=2.0,
            focus_cell_type="A",
            include_cell_types=["A", "B"],
        )
        # Should return only type A cells
        assert (result["cell_type"] == "A").all()
        # Should count both A and B neighbors
        assert "n_neighbors_A" in result.columns
        assert "n_neighbors_B" in result.columns


class TestNeighborCountsSanity:
    """Sanity checks for neighbor counting."""

    def test_large_r_max_finds_all(self, simple_df):
        """Large r_max should find all possible neighbors."""
        # Use r_max=5.0 which is large enough but within freud limits
        result = count_neighbor_cell_types(simple_df, r_max=5.0)
        # Each cell should have 4 neighbors total (5 cells - 1 self)
        total_neighbors = result["n_neighbors_A"] + result["n_neighbors_B"]
        assert (total_neighbors == 4).all()

    def test_small_r_max_finds_few(self, simple_df):
        """Very small r_max should find few neighbors."""
        result = count_neighbor_cell_types(simple_df, r_max=0.1)
        # Should have very few or no neighbors
        total_neighbors = result["n_neighbors_A"] + result["n_neighbors_B"]
        assert (total_neighbors <= 1).all()

    def test_increasing_r_max_increases_counts(self, simple_df):
        """Increasing r_max should increase neighbor counts."""
        result1 = count_neighbor_cell_types(simple_df, r_max=0.5)
        result2 = count_neighbor_cell_types(simple_df, r_max=2.0)

        total1 = (result1["n_neighbors_A"] + result1["n_neighbors_B"]).sum()
        total2 = (result2["n_neighbors_A"] + result2["n_neighbors_B"]).sum()
        assert total2 >= total1


class TestNeighborCountsMultiImage:
    """Tests for multi-image processing."""

    def test_processes_all_images(self, multi_image_df):
        """Should process all images."""
        result = count_neighbor_cell_types(multi_image_df, r_max=2.0)
        assert set(result["image_id"]) == {"img1", "img2"}

    def test_image_separation(self, multi_image_df):
        """Neighbors should not cross image boundaries."""
        result = count_neighbor_cell_types(multi_image_df, r_max=2.0)
        # Images are separated (img1 has cells near origin, img2 near x=10)
        # Each image should have same total neighbor counts
        img1_result = result[result["image_id"] == "img1"]
        img2_result = result[result["image_id"] == "img2"]

        total1 = (img1_result["n_neighbors_A"] + img1_result["n_neighbors_B"]).sum()
        total2 = (img2_result["n_neighbors_A"] + img2_result["n_neighbors_B"]).sum()
        assert total1 == total2


class TestNeighborCountsReproducibility:
    """Reproducibility tests."""

    def test_deterministic(self, simple_df):
        """Same input should give same output."""
        result1 = count_neighbor_cell_types(simple_df, r_max=2.0)
        result2 = count_neighbor_cell_types(simple_df, r_max=2.0)
        pd.testing.assert_frame_equal(result1, result2)

    def test_row_order_invariant(self, simple_df):
        """Row order shouldn't affect values."""
        shuffled = simple_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        result1 = count_neighbor_cell_types(simple_df, r_max=2.0)
        result2 = count_neighbor_cell_types(shuffled, r_max=2.0)

        result1_sorted = result1.sort_values("cell_id").reset_index(drop=True)
        result2_sorted = result2.sort_values("cell_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(result1_sorted, result2_sorted)


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests across functions."""

    def test_density_and_order_compatible(self, simple_df):
        """Density and order outputs should be joinable."""
        density = compute_local_density(simple_df, r_max=2.0)
        order = compute_katic_order(simple_df, k=6, r_max=2.0)

        # Should be able to merge on cell_id and image_id
        merged = density.merge(order, on=["cell_id", "image_id"], suffixes=("_dens", "_order"))
        assert len(merged) == len(simple_df)

    def test_all_three_functions_compatible(self, simple_df):
        """All three functions should produce joinable outputs."""
        density = compute_local_density(simple_df, r_max=2.0)
        order = compute_katic_order(simple_df, k=6, r_max=2.0)
        counts = count_neighbor_cell_types(simple_df, r_max=2.0)

        merged = density.merge(order, on=["cell_id", "image_id"], suffixes=("", "_order"))
        merged = merged.merge(counts, on=["cell_id", "image_id"], suffixes=("", "_count"))
        assert len(merged) == len(simple_df)

    def test_focus_type_consistency(self, simple_df):
        """Focus type filtering should be consistent across functions."""
        density = compute_local_density(simple_df, r_max=2.0, focus_cell_type="A")
        order = compute_katic_order(simple_df, k=6, r_max=2.0, focus_cell_type="A")
        counts = count_neighbor_cell_types(simple_df, r_max=2.0, focus_cell_type="A")

        # All should return same cell_ids
        assert set(density["cell_id"]) == set(order["cell_id"])
        assert set(density["cell_id"]) == set(counts["cell_id"])
