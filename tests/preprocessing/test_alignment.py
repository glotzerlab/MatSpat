"""Tests for coordinate alignment and rotation."""

import numpy as np
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from matspat.preprocessing.alignment import process_rotation, rotate_coords


def test_rotate_coords_90_degrees():
    """Test 90-degree counterclockwise rotation."""
    coords = np.array([[1.0, 0.0], [0.0, 1.0]])
    rotated = rotate_coords(coords, np.pi / 2)
    expected = np.array([[0.0, 1.0], [-1.0, 0.0]])
    assert np.allclose(rotated, expected, atol=1e-10)


def test_rotate_coords_180_degrees():
    """Test 180-degree rotation."""
    coords = np.array([[1.0, 0.0], [0.0, 1.0]])
    rotated = rotate_coords(coords, np.pi)
    expected = np.array([[-1.0, 0.0], [0.0, -1.0]])
    assert np.allclose(rotated, expected, atol=1e-10)


def test_rotate_coords_identity():
    """Test that 0-degree rotation is identity."""
    coords = np.array([[1.0, 2.0], [3.0, 4.0]])
    rotated = rotate_coords(coords, 0.0)
    assert np.allclose(rotated, coords, atol=1e-10)


def test_rotate_coords_preserves_shape():
    """Test that rotation preserves array shape."""
    coords = np.random.rand(100, 2)
    rotated = rotate_coords(coords, np.pi / 4)
    assert rotated.shape == coords.shape


def test_rotate_coords_preserves_distances():
    """Test that rotation preserves pairwise distances (isometry)."""
    coords = np.random.rand(10, 2)
    rotated = rotate_coords(coords, np.pi / 3)

    # Compute pairwise distances before and after
    from scipy.spatial.distance import pdist

    dist_before = pdist(coords)
    dist_after = pdist(rotated)

    assert np.allclose(dist_before, dist_after, atol=1e-10)


# ---------------------------------------------------------------------------
# process_rotation tests
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_tissue_df(n_tumor: int = 50, n_immune: int = 50) -> pd.DataFrame:
    """Synthetic tissue with tumor cells elongated along x-axis."""
    # Tumor cells: elongated along x (should get rotated to y)
    tumor_x = RNG.normal(0, 10, n_tumor)
    tumor_y = RNG.normal(0, 1, n_tumor)
    # Immune cells: scattered
    immune_x = RNG.normal(0, 5, n_immune)
    immune_y = RNG.normal(0, 5, n_immune)
    return pd.DataFrame(
        {
            "x": np.concatenate([tumor_x, immune_x]),
            "y": np.concatenate([tumor_y, immune_y]),
            "cell_type": ["tumor"] * n_tumor + ["immune"] * n_immune,
        }
    )


def test_process_rotation_adds_columns():
    df = _make_tissue_df()
    result = process_rotation(df, "tumor")
    assert "x_rot" in result.columns
    assert "y_rot" in result.columns
    assert len(result) == len(df)


def test_process_rotation_preserves_distances():
    df = _make_tissue_df()
    result = process_rotation(df, "tumor")
    from scipy.spatial.distance import pdist

    orig = pdist(df[["x", "y"]].to_numpy())
    rotated = pdist(result[["x_rot", "y_rot"]].to_numpy())
    assert np.allclose(orig, rotated, atol=1e-10)


def test_process_rotation_does_not_mutate_input():
    df = _make_tissue_df()
    x_orig = df["x"].copy()
    process_rotation(df, "tumor")
    pd.testing.assert_series_equal(df["x"], x_orig)


def test_process_rotation_few_tumor_warns():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "cell_type": ["tumor", "immune"]})
    with pytest.warns(UserWarning, match="Skipping ICA"):
        result = process_rotation(df, "tumor")
    # Should still have x_rot/y_rot (copies of original)
    np.testing.assert_array_equal(result["x_rot"], result["x"])
    np.testing.assert_array_equal(result["y_rot"], result["y"])


def test_process_rotation_reproducible():
    df = _make_tissue_df()
    r1 = process_rotation(df, "tumor", random_state=0)
    r2 = process_rotation(df, "tumor", random_state=0)
    np.testing.assert_array_equal(r1["x_rot"].to_numpy(), r2["x_rot"].to_numpy())
