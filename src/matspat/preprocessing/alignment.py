"""Coordinate transformations and spatial alignment utilities. transforms to
align "densest tumor axis" with y axis(this will make more sense once implemented)
Functions: rotate_coords(), align_to_reference()
"""

import numpy as np
from numpy.typing import NDArray


def rotate_coords(coords: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    """Rotate 2D coordinates by a given angle.

    Args:
        coords: Array of shape (N, 2) containing (x, y) coordinates.
        angle: Rotation angle in radians (counterclockwise).

    Returns:
        Rotated coordinates as array of shape (N, 2).

    Examples:
        >>> coords = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> rotated = rotate_coords(coords, np.pi / 2)  # 90 degrees
        >>> np.allclose(rotated, [[0.0, 1.0], [-1.0, 0.0]])
        True
    """
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return coords.dot(rotation_matrix.T)
