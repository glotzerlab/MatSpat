import pytest
import numpy as np
from matspat.interactions import lees_L

rng = np.random.default_rng(42)

x_array = np.arange(5)
y_array = np.arange(5)

x_grid, y_grid = np.meshgrid(x_array,y_array)

x_flat = x_grid.flatten()
y_flat = y_grid.flatten()

positions = np.column_stack((x_flat, y_flat))

x = rng.random(25)

bandwidth = 2.0

def test_lees_L_points_minimum():
    x_slice = x[:1]
    pos_slice = positions[:1]
    L = lees_L(x_slice, x_slice, pos_slice, bandwidth)
    assert np.isnan(L)

def test_lees_L_identical_inputs():
    L = lees_L(x, x, positions, bandwidth)
    assert np.isclose(L, 1.0)

def test_lees_L_inverted_inputs():
    L = lees_L(x, -x, positions, bandwidth)
    assert np.isclose(L, -1.0)


