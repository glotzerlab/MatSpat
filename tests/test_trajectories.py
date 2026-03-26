"""Tests for matspat.interactions.trajectories."""

import numpy as np
import pandas as pd
import pytest

from matspat.interactions.trajectories import compute_pairwise_trajectories

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spatial_df(n_tumor=50, n_immune_a=30, n_immune_b=30, n_images=1, rng_seed=42):
    """Create a synthetic spatial DataFrame with tumor + 2 immune types."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for img_idx in range(n_images):
        img_id = f"img_{img_idx}"
        # Tumor cells clustered near center
        for _ in range(n_tumor):
            rows.append(
                {
                    "image_id": img_id,
                    "cell_type": "Tumor",
                    "x_rot": rng.normal(50, 5),
                    "y_rot": rng.normal(50, 5),
                }
            )
        # Immune A scattered broadly
        for _ in range(n_immune_a):
            rows.append(
                {
                    "image_id": img_id,
                    "cell_type": "CD4",
                    "x_rot": rng.normal(50, 15),
                    "y_rot": rng.normal(50, 15),
                }
            )
        # Immune B in a different region
        for _ in range(n_immune_b):
            rows.append(
                {
                    "image_id": img_id,
                    "cell_type": "CD8",
                    "x_rot": rng.normal(70, 10),
                    "y_rot": rng.normal(30, 10),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tumor–immune pair mode (default)
# ---------------------------------------------------------------------------


class TestTumorImmunePairMode:
    """Tests for pair_mode='tumor_immune' (default)."""

    def test_default_pair_mode(self):
        """Default pair_mode is 'tumor_immune'."""
        df = _make_spatial_df()
        result = compute_pairwise_trajectories(
            df,
            tumor_label="Tumor",
            bandwidth=10.0,
            n_quantiles=4,
            grid_size=20,
        )
        # 2 immune types -> 2 tumor-immune pairs x 4 bins = 8 rows
        assert len(result) == 8
        pairs = sorted(result["immune_pair"].unique())
        assert pairs == ["Tumor x CD4", "Tumor x CD8"]

    def test_tumor_immune_single_immune_type(self):
        """Works with only 1 immune type (unlike immune_immune mode)."""
        df = _make_spatial_df(n_immune_b=0)
        result = compute_pairwise_trajectories(
            df,
            tumor_label="Tumor",
            bandwidth=10.0,
            n_quantiles=3,
            grid_size=20,
        )
        assert len(result) == 3  # 1 pair x 3 bins
        assert result["immune_pair"].unique().tolist() == ["Tumor x CD4"]

    def test_tumor_immune_values_bounded(self):
        """Lee's L values should be in [-1, 1] or NaN."""
        df = _make_spatial_df()
        result = compute_pairwise_trajectories(
            df,
            tumor_label="Tumor",
            bandwidth=10.0,
            n_quantiles=3,
            grid_size=20,
        )
        valid = result["lees_L"].dropna()
        assert (valid >= -1.0 - 1e-10).all()
        assert (valid <= 1.0 + 1e-10).all()

    def test_tumor_immune_multiple_images(self):
        """Each image produces its own set of rows."""
        df = _make_spatial_df(n_images=3)
        result = compute_pairwise_trajectories(
            df,
            tumor_label="Tumor",
            bandwidth=10.0,
            n_quantiles=2,
            grid_size=20,
        )
        # 3 images x 2 pairs x 2 bins = 12 rows
        assert len(result) == 12
        assert set(result["image_id"].unique()) == {"img_0", "img_1", "img_2"}

    def test_tumor_immune_compatible_with_build_sequences(self):
        """Output plugs directly into classification.sequences.build_sequences."""
        from matspat.classification.sequences import build_sequences

        df = _make_spatial_df(n_images=2)
        traj = compute_pairwise_trajectories(
            df,
            tumor_label="Tumor",
            bandwidth=10.0,
            n_quantiles=3,
            grid_size=20,
        )
        sequences, features = build_sequences(traj)

        assert len(sequences) == 2
        assert features == ["Tumor x CD4", "Tumor x CD8"]
        for seq in sequences.values():
            assert seq.shape == (3, 2)  # 3 bins, 2 features


# ---------------------------------------------------------------------------
# Immune–immune pair mode
# ---------------------------------------------------------------------------


class TestImmuneImmunePairMode:
    """Tests for pair_mode='immune_immune'."""

    def test_basic_output_shape(self):
        """Output has correct columns and one row per (image, pair, bin)."""
        df = _make_spatial_df()
        result = compute_pairwise_trajectories(
            df,
            tumor_label="Tumor",
            bandwidth=10.0,
            n_quantiles=4,
            pair_mode="immune_immune",
            grid_size=20,
        )
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"image_id", "immune_pair", "quantile_bin", "lees_L"}

        # 1 image, 1 pair (CD4 x CD8), 4 bins
        assert len(result) == 4
        assert result["immune_pair"].unique().tolist() == ["CD4 x CD8"]
        assert sorted(result["quantile_bin"].unique()) == [0, 1, 2, 3]

    def test_lees_L_values_bounded(self):
        """Lee's L values should be in [-1, 1] or NaN."""
        df = _make_spatial_df()
        result = compute_pairwise_trajectories(
            df,
            tumor_label="Tumor",
            bandwidth=10.0,
            n_quantiles=3,
            pair_mode="immune_immune",
            grid_size=20,
        )
        valid = result["lees_L"].dropna()
        assert (valid >= -1.0 - 1e-10).all()
        assert (valid <= 1.0 + 1e-10).all()

    def test_multiple_images(self):
        """Each image produces its own set of rows."""
        df = _make_spatial_df(n_images=3)
        result = compute_pairwise_trajectories(
            df,
            tumor_label="Tumor",
            bandwidth=10.0,
            n_quantiles=2,
            pair_mode="immune_immune",
            grid_size=20,
        )
        # 3 images x 1 pair x 2 bins = 6 rows
        assert len(result) == 6
        assert set(result["image_id"].unique()) == {"img_0", "img_1", "img_2"}

    def test_three_immune_types(self):
        """Three immune types produce 3 pairs."""
        df = _make_spatial_df()
        rng = np.random.default_rng(99)
        extra = pd.DataFrame(
            {
                "image_id": "img_0",
                "cell_type": "Bcell",
                "x_rot": rng.normal(30, 10, 20),
                "y_rot": rng.normal(60, 10, 20),
            }
        )
        df = pd.concat([df, extra], ignore_index=True)

        result = compute_pairwise_trajectories(
            df,
            tumor_label="Tumor",
            bandwidth=10.0,
            n_quantiles=2,
            pair_mode="immune_immune",
            grid_size=20,
        )
        pairs = sorted(result["immune_pair"].unique())
        assert pairs == ["Bcell x CD4", "Bcell x CD8", "CD4 x CD8"]
        assert len(result) == 3 * 2  # 3 pairs x 2 bins

    def test_exclude_types(self):
        """Excluded immune types are omitted from pairs."""
        df = _make_spatial_df()
        rng = np.random.default_rng(99)
        extra = pd.DataFrame(
            {
                "image_id": "img_0",
                "cell_type": "Bcell",
                "x_rot": rng.normal(30, 10, 20),
                "y_rot": rng.normal(60, 10, 20),
            }
        )
        df = pd.concat([df, extra], ignore_index=True)

        result = compute_pairwise_trajectories(
            df,
            tumor_label="Tumor",
            bandwidth=10.0,
            n_quantiles=2,
            pair_mode="immune_immune",
            exclude_types=["Bcell"],
            grid_size=20,
        )
        assert result["immune_pair"].unique().tolist() == ["CD4 x CD8"]

    def test_too_few_immune_types_warns(self):
        """Image with <2 immune types is skipped with a warning."""
        df = _make_spatial_df(n_immune_b=0)

        with pytest.warns(UserWarning, match="fewer than 2 immune type"):
            result = compute_pairwise_trajectories(
                df,
                tumor_label="Tumor",
                bandwidth=10.0,
                n_quantiles=2,
                pair_mode="immune_immune",
                grid_size=20,
            )
        assert result.empty

    def test_compatible_with_build_sequences(self):
        """Output plugs directly into classification.sequences.build_sequences."""
        from matspat.classification.sequences import build_sequences

        df = _make_spatial_df(n_images=2)
        traj = compute_pairwise_trajectories(
            df,
            tumor_label="Tumor",
            bandwidth=10.0,
            n_quantiles=3,
            pair_mode="immune_immune",
            grid_size=20,
        )
        sequences, features = build_sequences(traj)

        assert len(sequences) == 2  # 2 images
        assert features == ["CD4 x CD8"]
        for seq in sequences.values():
            assert seq.shape == (3, 1)  # 3 bins, 1 feature


# ---------------------------------------------------------------------------
# "all" pair mode
# ---------------------------------------------------------------------------


class TestAllPairMode:
    """Tests for pair_mode='all'."""

    def test_all_mode_combines_both(self):
        """'all' produces tumor-immune AND immune-immune pairs."""
        df = _make_spatial_df()
        result = compute_pairwise_trajectories(
            df,
            tumor_label="Tumor",
            bandwidth=10.0,
            n_quantiles=2,
            pair_mode="all",
            grid_size=20,
        )
        pairs = sorted(result["immune_pair"].unique())
        # 2 tumor-immune + 1 immune-immune = 3 pairs
        assert pairs == ["CD4 x CD8", "Tumor x CD4", "Tumor x CD8"]
        assert len(result) == 3 * 2  # 3 pairs x 2 bins


# ---------------------------------------------------------------------------
# Shared edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases shared across pair modes."""

    def test_cumulative_method(self):
        """Cumulative quantile method produces same structure."""
        df = _make_spatial_df()
        result = compute_pairwise_trajectories(
            df,
            tumor_label="Tumor",
            bandwidth=10.0,
            n_quantiles=3,
            quantile_method="cumulative",
            grid_size=20,
        )
        # Default tumor_immune: 2 pairs x 3 bins = 6
        assert len(result) == 6
        assert sorted(result["quantile_bin"].unique()) == [0, 1, 2]

    def test_too_few_tumor_cells_warns(self):
        """Image with <2 tumor cells is skipped with a warning."""
        df = _make_spatial_df()
        tumor_mask = df["cell_type"] == "Tumor"
        drop_idx = df[tumor_mask].index[1:]
        df = df.drop(drop_idx)

        with pytest.warns(UserWarning, match="fewer than 2 tumor cells"):
            result = compute_pairwise_trajectories(
                df,
                tumor_label="Tumor",
                bandwidth=10.0,
                n_quantiles=2,
                grid_size=20,
            )
        assert result.empty

    def test_invalid_quantile_method(self):
        """Unknown quantile method raises ValueError."""
        df = _make_spatial_df()
        with pytest.raises(ValueError, match="quantile_method"):
            compute_pairwise_trajectories(
                df,
                tumor_label="Tumor",
                bandwidth=10.0,
                n_quantiles=2,
                quantile_method="invalid",
            )

    def test_invalid_pair_mode(self):
        """Unknown pair_mode raises ValueError."""
        df = _make_spatial_df()
        with pytest.raises(ValueError, match="pair_mode"):
            compute_pairwise_trajectories(
                df,
                tumor_label="Tumor",
                bandwidth=10.0,
                n_quantiles=2,
                pair_mode="invalid",
            )

    def test_missing_columns(self):
        """Missing required columns raise KeyError."""
        df = pd.DataFrame({"x_rot": [1], "y_rot": [2]})
        with pytest.raises(KeyError, match="missing required columns"):
            compute_pairwise_trajectories(
                df,
                tumor_label="Tumor",
                bandwidth=10.0,
                n_quantiles=2,
            )

    def test_empty_dataframe(self):
        """Empty input returns empty DataFrame."""
        df = pd.DataFrame(columns=["x_rot", "y_rot", "cell_type", "image_id"])
        result = compute_pairwise_trajectories(
            df,
            tumor_label="Tumor",
            bandwidth=10.0,
            n_quantiles=2,
        )
        assert result.empty
