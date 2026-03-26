"""Centralized configuration for the MatSpat analysis pipeline.

Defines:
- Canonical column names produced by the preprocessing stage
- :class:`DatasetConfig` for per-dataset column mappings
- Default analysis parameters (KDE, HMM, bootstrap, train/test split)
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Canonical column names — output by load_spatial_data()
# ---------------------------------------------------------------------------

CELL_ID: str = "cell_id"
X: str = "x"
Y: str = "y"
CELL_TYPE: str = "cell_type"
CELL_TYPE_ENCODED: str = "cell_type_encoded"
IMAGE_ID: str = "image_id"
PATHOLOGY: str = "pathology"
PATIENT_ID: str = "patient_id"

# Added by process_rotation()
X_ROT: str = "x_rot"
Y_ROT: str = "y_rot"

# Produced by compute_pairwise_trajectories()
IMMUNE_PAIR: str = "immune_pair"
QUANTILE_BIN: str = "quantile_bin"
LEES_L: str = "lees_L"

# Label column (user-supplied, used by extract_labels)
LABEL: str = "label"


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Per-dataset column mapping and metadata.

    Stores the raw column names for a specific dataset so they can be
    passed directly to :func:`~matspat.preprocessing.loader.load_spatial_data`
    via :meth:`loader_kwargs`.

    Args:
        x_col: Raw column name for x coordinates.
        y_col: Raw column name for y coordinates.
        cell_type_col: Raw column name for cell type annotations.
        tumor_label: Value in *cell_type_col* that identifies tumor cells.
        cell_id_col: Raw column name for cell IDs (None to auto-generate).
        pathology_col: Raw column name for disease/pathology labels.
        image_id_col: Raw column name for image/slide identifiers.
        patient_id_col: Raw column name for patient identifiers.

    Example::

        crc = DatasetConfig(
            x_col="X",
            y_col="Y",
            cell_type_col="cell.type",
            tumor_label="Tumor",
            pathology_col="pathology",
            image_id_col="image",
            patient_id_col="patient",
        )
        df, mapping = load_spatial_data("crc.csv", **crc.loader_kwargs())
    """

    x_col: str
    y_col: str
    cell_type_col: str
    tumor_label: str
    cell_id_col: str | None = None
    pathology_col: str | None = None
    image_id_col: str | None = None
    patient_id_col: str | None = None

    def loader_kwargs(self) -> dict[str, str | None]:
        """Return a kwargs dict for :func:`load_spatial_data`."""
        kw: dict[str, str | None] = {
            "x_col": self.x_col,
            "y_col": self.y_col,
            "cell_type_col": self.cell_type_col,
        }
        for attr, key in [
            ("cell_id_col", "cell_id_col"),
            ("pathology_col", "pathology_col"),
            ("image_id_col", "image_id_col"),
            ("patient_id_col", "patient_id_col"),
        ]:
            val = getattr(self, attr)
            if val is not None:
                kw[key] = val
        return kw


# ---------------------------------------------------------------------------
# Analysis defaults
# ---------------------------------------------------------------------------

# KDE grid resolution
DEFAULT_GRID_SIZE: int = 100

# HMM hyperparameters (match train_class_hmm / classify signatures)
DEFAULT_N_STATES: int = 3
DEFAULT_COVARIANCE_TYPE: str = "full"
DEFAULT_MAX_ITER: int = 100
DEFAULT_N_REPLICATES: int = 3
DEFAULT_TOL: float = 1e-4

# Bootstrap confidence intervals
DEFAULT_N_BOOTSTRAPS: int = 1000
DEFAULT_CI_ALPHA: float = 0.05

# Train/test split
DEFAULT_TEST_SIZE: float = 0.3
DEFAULT_RANDOM_STATE: int = 42
