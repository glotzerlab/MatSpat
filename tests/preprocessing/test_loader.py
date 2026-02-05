"""Tests for matspat.preprocessing.loader module."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

from matspat.preprocessing import load_spatial_data

# ==================== Fixtures ====================


@pytest.fixture
def minimal_csv(tmp_path: Path) -> Path:
    """Create a minimal CSV with required columns only."""
    csv_path = tmp_path / "minimal.csv"
    df = pd.DataFrame(
        {
            "X": [1.0, 2.0, 3.0],
            "Y": [4.0, 5.0, 6.0],
            "type": ["CD4", "CD8", "Tumor"],
        }
    )
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def full_csv(tmp_path: Path) -> Path:
    """Create a CSV with all optional columns."""
    csv_path = tmp_path / "full.csv"
    df = pd.DataFrame(
        {
            "cell_id": [101, 102, 103, 104],
            "X": [1.0, 2.0, 3.0, 4.0],
            "Y": [5.0, 6.0, 7.0, 8.0],
            "type": ["CD4", "CD8", "Tumor", "CD4"],
            "pathology": ["PDAC", "PDAC", "CRC", "CRC"],
            "img": ["img1", "img1", "img2", "img2"],
            "patient": ["P1", "P1", "P2", "P2"],
        }
    )
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def csv_with_nans_required(tmp_path: Path) -> Path:
    """Create a CSV with NaN in required columns."""
    csv_path = tmp_path / "nan_required.csv"
    df = pd.DataFrame(
        {
            "X": [1.0, None, 3.0, 4.0],
            "Y": [5.0, 6.0, None, 8.0],
            "type": ["CD4", "CD8", "Tumor", None],
        }
    )
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def csv_with_nans_optional(tmp_path: Path) -> Path:
    """Create a CSV with NaN in optional columns (mixed)."""
    csv_path = tmp_path / "nan_optional.csv"
    df = pd.DataFrame(
        {
            "X": [1.0, 2.0, 3.0, 4.0],
            "Y": [5.0, 6.0, 7.0, 8.0],
            "type": ["CD4", "CD8", "Tumor", "CD4"],
            "pathology": ["PDAC", None, "CRC", "PDAC"],  # Mixed NaN
            "img": ["img1", "img1", "img2", "img2"],
        }
    )
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def empty_csv(tmp_path: Path) -> Path:
    """Create an empty CSV (header only)."""
    csv_path = tmp_path / "empty.csv"
    df = pd.DataFrame(columns=["X", "Y", "type"])
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def csv_with_duplicates(tmp_path: Path) -> Path:
    """Create a CSV with duplicate cell_ids."""
    csv_path = tmp_path / "duplicates.csv"
    df = pd.DataFrame(
        {
            "cell_id": [1, 2, 2, 3],  # Duplicate cell_id=2
            "X": [1.0, 2.0, 3.0, 4.0],
            "Y": [5.0, 6.0, 7.0, 8.0],
            "type": ["CD4", "CD8", "Tumor", "CD4"],
        }
    )
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def csv_numeric_cell_types(tmp_path: Path) -> Path:
    """Create a CSV with numeric cell types (should be converted to strings)."""
    csv_path = tmp_path / "numeric_types.csv"
    df = pd.DataFrame(
        {
            "X": [1.0, 2.0, 3.0],
            "Y": [4.0, 5.0, 6.0],
            "type": [0, 1, 2],  # Numeric cell types
        }
    )
    df.to_csv(csv_path, index=False)
    return csv_path


# ==================== Basic Functionality Tests ====================


def test_minimal_load(minimal_csv: Path):
    """Test loading with only required columns."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df, mapping = load_spatial_data(
            minimal_csv,
            x_col="X",
            y_col="Y",
            cell_type_col="type",
        )

        # Should warn about image_id auto-generation
        assert len(w) == 1
        assert "image_id" in str(w[0].message).lower()

    # Check columns
    assert list(df.columns) == ["cell_id", "x", "y", "cell_type", "cell_type_encoded", "image_id"]

    # Check data
    assert len(df) == 3
    assert df["cell_id"].tolist() == [1, 2, 3]
    assert df["x"].tolist() == [1.0, 2.0, 3.0]
    assert df["y"].tolist() == [4.0, 5.0, 6.0]
    assert df["cell_type"].tolist() == ["CD4", "CD8", "Tumor"]
    assert all(df["image_id"] == "image_1")

    # Check dtypes
    assert df["cell_id"].dtype == int
    assert df["x"].dtype == float
    assert df["y"].dtype == float
    assert df["cell_type"].dtype == object
    assert df["cell_type_encoded"].dtype == int
    assert df["image_id"].dtype == object

    # Check mapping (alphabetical order, 1-indexed)
    assert mapping == {"CD4": 1, "CD8": 2, "Tumor": 3}
    assert df["cell_type_encoded"].tolist() == [1, 2, 3]


def test_full_load(full_csv: Path):
    """Test loading with all optional columns."""
    df, mapping = load_spatial_data(
        full_csv,
        x_col="X",
        y_col="Y",
        cell_type_col="type",
        cell_id_col="cell_id",
        pathology_col="pathology",
        image_id_col="img",
        patient_id_col="patient",
    )

    # Check columns
    expected_cols = [
        "cell_id",
        "x",
        "y",
        "cell_type",
        "cell_type_encoded",
        "image_id",
        "pathology",
        "patient_id",
    ]
    assert list(df.columns) == expected_cols

    # Check data
    assert len(df) == 4
    assert df["cell_id"].tolist() == [101, 102, 103, 104]
    assert df["pathology"].tolist() == ["PDAC", "PDAC", "CRC", "CRC"]
    assert df["image_id"].tolist() == ["img1", "img1", "img2", "img2"]
    assert df["patient_id"].tolist() == ["P1", "P1", "P2", "P2"]

    # Check cell_type_encoded (alphabetical)
    assert mapping == {"CD4": 1, "CD8": 2, "Tumor": 3}
    assert df["cell_type_encoded"].tolist() == [1, 2, 3, 1]


# ==================== Validation Tests ====================


def test_file_not_found():
    """Test FileNotFoundError when file doesn't exist."""
    with pytest.raises(FileNotFoundError, match="File not found"):
        load_spatial_data(
            "nonexistent.csv",
            x_col="X",
            y_col="Y",
            cell_type_col="type",
        )


def test_missing_required_columns(minimal_csv: Path):
    """Test ValueError when required columns are missing."""
    with pytest.raises(ValueError, match="Required columns not found"):
        load_spatial_data(
            minimal_csv,
            x_col="WRONG_X",
            y_col="Y",
            cell_type_col="type",
        )


def test_missing_optional_column(minimal_csv: Path):
    """Test ValueError when specified optional column doesn't exist."""
    with pytest.raises(ValueError, match="pathology_col='WRONG' not found"):
        load_spatial_data(
            minimal_csv,
            x_col="X",
            y_col="Y",
            cell_type_col="type",
            pathology_col="WRONG",
        )


def test_empty_csv_raises_error(empty_csv: Path):
    """Test that empty CSV raises ValueError."""
    with pytest.raises(ValueError, match="No data rows found in CSV"):
        load_spatial_data(
            empty_csv,
            x_col="X",
            y_col="Y",
            cell_type_col="type",
        )


def test_duplicate_cell_ids_raises_error(csv_with_duplicates: Path):
    """Test that duplicate cell_ids raise ValueError."""
    with pytest.raises(ValueError, match="Duplicate cell_id values found: \\[2\\]"):
        load_spatial_data(
            csv_with_duplicates,
            x_col="X",
            y_col="Y",
            cell_type_col="type",
            cell_id_col="cell_id",
        )


# ==================== NaN Handling Tests ====================


def test_nan_in_required_columns_dropped(csv_with_nans_required: Path):
    """Test that rows with NaN in required columns are dropped with warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df, mapping = load_spatial_data(
            csv_with_nans_required,
            x_col="X",
            y_col="Y",
            cell_type_col="type",
        )

        # Should have warnings: 1 for NaN drops, 1 for image_id
        assert len(w) == 2
        nan_warning = [warning for warning in w if "Dropped" in str(warning.message)][0]
        assert "3 rows" in str(nan_warning.message)
        assert "[1, 2, 3]" in str(nan_warning.message)

    # Check only valid row remains (row 0)
    assert len(df) == 1
    assert df["x"].tolist() == [1.0]
    assert df["y"].tolist() == [5.0]
    assert df["cell_type"].tolist() == ["CD4"]

    # cell_id should be re-indexed to 1
    assert df["cell_id"].tolist() == [1]


def test_nan_in_optional_columns_kept_with_warning(csv_with_nans_optional: Path):
    """Test that rows with NaN in optional columns are kept but warned."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df, mapping = load_spatial_data(
            csv_with_nans_optional,
            x_col="X",
            y_col="Y",
            cell_type_col="type",
            pathology_col="pathology",
            image_id_col="img",
        )

        # Should warn about mixed NaN in optional column
        assert len(w) == 1
        assert "mixed nan/non-nan" in str(w[0].message).lower()
        assert "pathology" in str(w[0].message)
        assert "[1]" in str(w[0].message)  # Row index with NaN

    # All rows should be kept
    assert len(df) == 4
    assert df["pathology"].tolist() == [
        "PDAC",
        "nan",
        "CRC",
        "PDAC",
    ]  # NaN converted to string "nan"


# ==================== cell_id Tests ====================


def test_cell_id_auto_generation(minimal_csv: Path):
    """Test that cell_id is auto-generated (1-indexed) when not provided."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        df, _ = load_spatial_data(
            minimal_csv,
            x_col="X",
            y_col="Y",
            cell_type_col="type",
        )

    assert df["cell_id"].tolist() == [1, 2, 3]


def test_cell_id_from_column(full_csv: Path):
    """Test that cell_id is used from column when provided."""
    df, _ = load_spatial_data(
        full_csv,
        x_col="X",
        y_col="Y",
        cell_type_col="type",
        cell_id_col="cell_id",
    )

    assert df["cell_id"].tolist() == [101, 102, 103, 104]


# ==================== image_id Tests ====================


def test_image_id_auto_generation_with_warning(minimal_csv: Path):
    """Test that image_id is auto-generated with warning when not provided."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df, _ = load_spatial_data(
            minimal_csv,
            x_col="X",
            y_col="Y",
            cell_type_col="type",
        )

        assert len(w) == 1
        assert "image_id" in str(w[0].message).lower()
        assert "image_1" in str(w[0].message)

    assert all(df["image_id"] == "image_1")


def test_image_id_from_column(full_csv: Path):
    """Test that image_id is used from column when provided."""
    df, _ = load_spatial_data(
        full_csv,
        x_col="X",
        y_col="Y",
        cell_type_col="type",
        image_id_col="img",
    )

    assert df["image_id"].tolist() == ["img1", "img1", "img2", "img2"]


# ==================== cell_type_encoded Tests ====================


def test_cell_type_encoded_alphabetical(minimal_csv: Path):
    """Test that cell_type_encoded is alphabetical and 1-indexed."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        df, mapping = load_spatial_data(
            minimal_csv,
            x_col="X",
            y_col="Y",
            cell_type_col="type",
        )

    # Cell types: CD4, CD8, Tumor
    # Alphabetical: CD4=1, CD8=2, Tumor=3
    assert mapping == {"CD4": 1, "CD8": 2, "Tumor": 3}
    assert df["cell_type_encoded"].tolist() == [1, 2, 3]


def test_cell_type_encoded_consistency(full_csv: Path):
    """Test that cell_type_encoded is consistent for duplicate cell types."""
    df, mapping = load_spatial_data(
        full_csv,
        x_col="X",
        y_col="Y",
        cell_type_col="type",
        image_id_col="img",
    )

    # Cell types: CD4, CD8, Tumor, CD4
    # Alphabetical: CD4=1, CD8=2, Tumor=3
    assert mapping == {"CD4": 1, "CD8": 2, "Tumor": 3}
    assert df["cell_type_encoded"].tolist() == [1, 2, 3, 1]


# ==================== Type Conversion Tests ====================


def test_numeric_cell_types_converted_to_string(csv_numeric_cell_types: Path):
    """Test that numeric cell types are converted to strings."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        df, mapping = load_spatial_data(
            csv_numeric_cell_types,
            x_col="X",
            y_col="Y",
            cell_type_col="type",
        )

    # Cell types should be strings
    assert df["cell_type"].dtype == object
    assert df["cell_type"].tolist() == ["0", "1", "2"]

    # Mapping should use string keys, alphabetical order
    assert mapping == {"0": 1, "1": 2, "2": 3}


def test_explicit_type_conversions(full_csv: Path):
    """Test that x, y are float and other columns are strings."""
    df, _ = load_spatial_data(
        full_csv,
        x_col="X",
        y_col="Y",
        cell_type_col="type",
        pathology_col="pathology",
        image_id_col="img",
        patient_id_col="patient",
    )

    assert df["x"].dtype == float
    assert df["y"].dtype == float
    assert df["cell_type"].dtype == object  # string
    assert df["image_id"].dtype == object  # string
    assert df["pathology"].dtype == object  # string
    assert df["patient_id"].dtype == object  # string


# ==================== Edge Cases ====================


def test_single_row_csv(tmp_path: Path):
    """Test loading CSV with single row."""
    csv_path = tmp_path / "single_row.csv"
    df_input = pd.DataFrame(
        {
            "X": [1.0],
            "Y": [2.0],
            "type": ["CD4"],
        }
    )
    df_input.to_csv(csv_path, index=False)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        df, mapping = load_spatial_data(
            csv_path,
            x_col="X",
            y_col="Y",
            cell_type_col="type",
        )

    assert len(df) == 1
    assert df["cell_id"].tolist() == [1]
    assert mapping == {"CD4": 1}


def test_column_names_with_spaces(tmp_path: Path):
    """Test loading CSV with column names containing spaces."""
    csv_path = tmp_path / "spaces.csv"
    df_input = pd.DataFrame(
        {
            "X Coordinate": [1.0, 2.0],
            "Y Coordinate": [3.0, 4.0],
            "Cell Type": ["CD4", "CD8"],
        }
    )
    df_input.to_csv(csv_path, index=False)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        df, mapping = load_spatial_data(
            csv_path,
            x_col="X Coordinate",
            y_col="Y Coordinate",
            cell_type_col="Cell Type",
        )

    assert len(df) == 2
    assert df["x"].tolist() == [1.0, 2.0]
    assert df["y"].tolist() == [3.0, 4.0]
    assert df["cell_type"].tolist() == ["CD4", "CD8"]


def test_pathlib_path_input(minimal_csv: Path):
    """Test that function accepts pathlib.Path objects."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        df, _ = load_spatial_data(
            minimal_csv,  # Already a Path object
            x_col="X",
            y_col="Y",
            cell_type_col="type",
        )

    assert len(df) == 3


def test_string_path_input(minimal_csv: Path):
    """Test that function accepts string paths."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        df, _ = load_spatial_data(
            str(minimal_csv),  # Convert to string
            x_col="X",
            y_col="Y",
            cell_type_col="type",
        )

    assert len(df) == 3


# ==================== Integration Test ====================


def test_full_pipeline_like_crc_example(tmp_path: Path):
    """Test a realistic pipeline similar to CRC pathology processing."""
    csv_path = tmp_path / "crc_like.csv"
    df_input = pd.DataFrame(
        {
            "pathology": ["CRC", "CRC", "PDAC", "PDAC"],
            "patient": ["P1", "P1", "P2", "P2"],
            "img": ["img1", "img1", "img2", "img2"],
            "X": [10.5, 20.3, 30.7, 40.1],
            "Y": [15.2, 25.6, 35.9, 45.3],
            "type": ["Tumor", "CD8", "Tumor", "CD4"],
        }
    )
    df_input.to_csv(csv_path, index=False)

    df, mapping = load_spatial_data(
        csv_path,
        x_col="X",
        y_col="Y",
        cell_type_col="type",
        pathology_col="pathology",
        image_id_col="img",
        patient_id_col="patient",
    )

    # Check structure
    assert len(df) == 4
    assert list(df.columns) == [
        "cell_id",
        "x",
        "y",
        "cell_type",
        "cell_type_encoded",
        "image_id",
        "pathology",
        "patient_id",
    ]

    # Check cell_type_encoded (alphabetical: CD4=1, CD8=2, Tumor=3)
    assert mapping == {"CD4": 1, "CD8": 2, "Tumor": 3}
    assert df["cell_type_encoded"].tolist() == [3, 2, 3, 1]

    # Check data integrity
    assert df["pathology"].tolist() == ["CRC", "CRC", "PDAC", "PDAC"]
    assert df["patient_id"].tolist() == ["P1", "P1", "P2", "P2"]
    assert df["x"].tolist() == [10.5, 20.3, 30.7, 40.1]
