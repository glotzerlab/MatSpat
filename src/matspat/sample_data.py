"""Generate synthetic spatial cell data for pipeline demonstrations.

Produces a DataFrame mimicking spatially resolved tissue imaging data with
two disease classes that have distinct immune-tumor spatial patterns:

- **"inflamed"**: immune cells cluster near the tumor core
- **"excluded"**: immune cells are pushed to the tissue periphery

The data is intentionally small (∼190–200 cells per image) so the full
pipeline runs in seconds on a laptop.  Tumor cells outnumber each immune
type by roughly 4–5x, reflecting typical tissue compositions.

Public functions:
    generate_sample_dataset  Create a ready-to-save DataFrame of synthetic cells.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd


def _place_cluster(
    rng: np.random.Generator,
    center: tuple[float, float],
    std: float,
    n: int,
) -> np.ndarray:
    """Sample *n* points from a 2-D Gaussian centered at *center*."""
    return rng.normal(loc=center, scale=std, size=(n, 2))


def _generate_image(
    rng: np.random.Generator,
    image_id: str,
    disease: str,
    *,
    n_tumor: int = 120,
    n_immune_per_type: int = 25,
    field_size: float = 500.0,
) -> pd.DataFrame:
    """Generate one synthetic tissue image.

    Args:
        rng: Numpy random generator.
        image_id: Identifier for this image.
        disease: ``"inflamed"`` or ``"excluded"``.
        n_tumor: Number of tumor cells.
        n_immune_per_type: Base count per immune cell type (jittered ±20 %).
        field_size: Spatial extent of the tissue field.
    """
    center = (field_size / 2, field_size / 2)
    tumor_std = field_size * 0.12

    # Tumor cells — always clustered in the center
    tumor_xy = _place_cluster(rng, center, tumor_std, n_tumor)

    immune_types = ["CD4", "CD8", "Macrophage"]
    immune_rows: list[np.ndarray] = []
    immune_labels: list[str] = []

    for ct in immune_types:
        n = int(n_immune_per_type * rng.uniform(0.8, 1.2))

        if disease == "inflamed":
            # Immune cells close to tumor — slightly wider spread
            xy = _place_cluster(rng, center, tumor_std * 1.4, n)
        else:
            # Immune cells pushed outward — ring around tumor
            angles = rng.uniform(0, 2 * np.pi, n)
            radii = rng.normal(loc=field_size * 0.30, scale=field_size * 0.06, size=n)
            xy = np.column_stack(
                [center[0] + radii * np.cos(angles), center[1] + radii * np.sin(angles)]
            )

        immune_rows.append(xy)
        immune_labels.extend([ct] * len(xy))

    all_xy = np.vstack([tumor_xy, *immune_rows])
    all_types = ["Tumor"] * n_tumor + immune_labels

    return pd.DataFrame(
        {
            "X": all_xy[:, 0],
            "Y": all_xy[:, 1],
            "cell_type": all_types,
            "image_id": image_id,
            "pathology": disease,
        }
    )


def generate_sample_dataset(
    *,
    n_per_class: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a complete synthetic dataset for the MatSpat tutorial.

    Creates *n_per_class* images for each of two disease classes
    (``"inflamed"`` and ``"excluded"``), each containing tumor, CD4, CD8,
    and Macrophage cells with class-specific spatial arrangements.

    Args:
        n_per_class: Number of images per disease class.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns ``X``, ``Y``, ``cell_type``, ``image_id``,
        ``pathology`` — ready to save as CSV and load with
        :func:`~matspat.preprocessing.load_spatial_data`.

    Example::

        from matspat.sample_data import generate_sample_dataset

        df = generate_sample_dataset()
        df.to_csv("sample_data.csv", index=False)
    """
    rng = np.random.default_rng(seed)
    diseases = ["excluded", "inflamed"]
    frames: list[pd.DataFrame] = []

    for disease, i in product(diseases, range(1, n_per_class + 1)):
        image_id = f"{disease}_{i:02d}"
        frames.append(_generate_image(rng, image_id, disease))

    return pd.concat(frames, ignore_index=True)
