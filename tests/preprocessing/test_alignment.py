"""Tests for coordinate alignment and rotation."""

import numpy as np
import pytest
from matspat.preprocessing.alignment import rotate_coords


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
